#!/usr/bin/env python3
"""把小毛豆在一个群里攒下的数据搬到另一个群。

为什么要有这个脚本：所有长期数据（记忆卡、称呼、叫法统一、群偏好、定时任务、对话摘要）
都以 group_openid 为键。QQ 换群/重建群会拿到一个全新的 openid，机器人在新群里就等于
彻底失忆——线上真出现过「查看记忆」回「我现在还没有长期记忆」，而库里躺着 92 张卡的情况
（旧群 3161FB… 有数据，新群 86FD26… 一条没有）。

脚本只搬数据，不写死任何群号、日期、人名：源群和目标群一律命令行给出。

    python scripts/migrate_group.py --db data/maodou.sqlite3 --list
    python scripts/migrate_group.py --db data/maodou.sqlite3 \
        --from <旧群openid> --to <新群openid> --dry-run
    确认无误后去掉 --dry-run 才真正写库；写库前自动做一次快照备份（--no-backup 可关）。

搬：memory_cards / person_aliases / person_addresses / group_preferences /
    scheduled_tasks（只搬还没跑完的）/ conversation_summaries（只搬不会盖住新群消息的）。
不搬：messages（message_key 全局唯一，搬过去只会撞键；新群的上下文从它自己的消息攒起）、
      care_log（主动关心的节流账本，搬过去会白占新群额度）、
      image_descriptions（按图片内容 hash 存，本来就跟群无关）。

搬完不用重启服务：机器人每轮都是现查库，没有在内存里缓存群数据。
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

GROUP_ID_RE = re.compile(r"^[0-9A-Za-z_-]{8,64}$")

# 主键是 (group_openid, xxx) 的键值表：目标群已有同名的就以源群为准覆盖。
KEY_VALUE_TABLES = ("person_aliases", "person_addresses", "group_preferences")
# 还值得搬的任务状态；done/deleted 是历史，搬过去只会污染「查看任务」。
LIVE_TASK_STATUSES = ("active", "paused")


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _insert_sql(table: str, columns: list[str], *, conflict: str) -> str:
    verb = {"ignore": "INSERT OR IGNORE", "replace": "INSERT OR REPLACE"}[conflict]
    cols = ", ".join(columns)
    marks = ", ".join("?" * len(columns))
    return f"{verb} INTO {table} ({cols}) VALUES ({marks})"


def copy_memory_cards(conn: sqlite3.Connection, src: str, dst: str) -> tuple[int, int]:
    """记忆卡：content 上有 UNIQUE(group_openid, content)，重跑天然幂等。"""
    columns = [c for c in table_columns(conn, "memory_cards") if c != "id"]
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM memory_cards WHERE group_openid = ? ORDER BY id",
        (src,),
    ).fetchall()
    copied = 0
    for row in rows:
        values = [dst if columns[i] == "group_openid" else row[i] for i in range(len(columns))]
        cursor = conn.execute(_insert_sql("memory_cards", columns, conflict="ignore"), values)
        copied += int(cursor.rowcount or 0)
    return copied, len(rows) - copied


def copy_key_value_table(conn: sqlite3.Connection, table: str, src: str, dst: str) -> tuple[int, int]:
    columns = table_columns(conn, table)
    rows = conn.execute(f"SELECT {', '.join(columns)} FROM {table} WHERE group_openid = ?", (src,)).fetchall()
    copied = 0
    for row in rows:
        values = [dst if columns[i] == "group_openid" else row[i] for i in range(len(columns))]
        cursor = conn.execute(_insert_sql(table, columns, conflict="replace"), values)
        copied += int(cursor.rowcount or 0)
    return copied, len(rows) - copied


def copy_tasks(conn: sqlite3.Connection, src: str, dst: str) -> tuple[int, int]:
    """定时任务：表上没有唯一键，靠 (title, schedule_type, next_run_at) 自己判重。

    预热任务用 linked_task_id 指向主任务，id 搬过去会变，所以先记下 src id -> 新 id
    的映射，插完再统一改回来。
    """
    columns = [c for c in table_columns(conn, "scheduled_tasks") if c != "id"]
    marks = ", ".join("?" * len(columns))
    placeholders = ", ".join(str(status) for status in ("?",) * len(LIVE_TASK_STATUSES))
    rows = conn.execute(
        f"""SELECT id, {', '.join(columns)} FROM scheduled_tasks
            WHERE group_openid = ? AND status IN ({placeholders}) ORDER BY id""",
        (src, *LIVE_TASK_STATUSES),
    ).fetchall()
    copied = 0
    skipped = 0
    id_map: dict[int, int] = {}
    for row in rows:
        src_id = int(row[0])
        values = list(row[1:])
        title = values[columns.index("title")]
        schedule_type = values[columns.index("schedule_type")]
        next_run_at = values[columns.index("next_run_at")]
        exists = conn.execute(
            """SELECT id FROM scheduled_tasks
               WHERE group_openid = ? AND title = ? AND schedule_type = ? AND next_run_at = ?""",
            (dst, title, schedule_type, next_run_at),
        ).fetchone()
        if exists:
            # 已经搬过了：记下目标群的 id，预热任务的 linked_task_id 还要靠它指过去。
            id_map[src_id] = int(exists[0])
            skipped += 1
            continue
        # linked_task_id 先原样带着源群的 id 插进去，全部插完再按映射改，
        # 因为预热任务可能排在主任务前面，那会儿新 id 还不存在。
        values[columns.index("group_openid")] = dst
        cursor = conn.execute(
            f"INSERT INTO scheduled_tasks ({', '.join(columns)}) VALUES ({marks})", values
        )
        id_map[src_id] = int(cursor.lastrowid or 0)
        copied += 1
    if "linked_task_id" in columns:
        for src_id, new_id in id_map.items():
            if not new_id:
                continue
            conn.execute(
                "UPDATE scheduled_tasks SET linked_task_id = ? WHERE group_openid = ? AND linked_task_id = ?",
                (new_id, dst, src_id),
            )
        # 主任务没跟着搬（比如已经跑完了）就别留悬空 id 指着别的群
        conn.execute(
            "UPDATE scheduled_tasks SET linked_task_id = NULL"
            " WHERE group_openid = ? AND linked_task_id IS NOT NULL"
            " AND linked_task_id NOT IN (SELECT id FROM scheduled_tasks WHERE group_openid = ?)",
            (dst, dst),
        )
    return copied, skipped


def copy_summaries(conn: sqlite3.Connection, src: str, dst: str) -> tuple[int, int]:
    """对话摘要：start/end_row_id 指的是全局 messages.row_id。

    摘要器取「本群 end_row_id 最大值之后的消息」来总结，所以只有当源群摘要的
    end_row_id 全部小于新群第一条消息的 row_id 时，搬过去才不会把新群自己的消息
    盖成「已总结」。不满足就跳过，宁可少几条背景，也不能让新群的记录再也总结不到。
    """
    columns = [c for c in table_columns(conn, "conversation_summaries") if c != "id"]
    dst_first = conn.execute(
        "SELECT COALESCE(MIN(row_id), 0) FROM messages WHERE group_openid = ?", (dst,)
    ).fetchone()[0]
    dst_first = int(dst_first or 0)
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM conversation_summaries WHERE group_openid = ? ORDER BY id",
        (src,),
    ).fetchall()
    copied = 0
    skipped = 0
    for row in rows:
        values = [dst if columns[i] == "group_openid" else row[i] for i in range(len(columns))]
        end_row_id = int(values[columns.index("end_row_id")] or 0)
        if dst_first and end_row_id >= dst_first:
            skipped += 1
            continue
        cursor = conn.execute(
            _insert_sql("conversation_summaries", columns, conflict="ignore"), values
        )
        if int(cursor.rowcount or 0):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def group_overview(conn: sqlite3.Connection) -> list[tuple[str, int, int, int, str]]:
    """列出每个群攒了多少东西。

    群的来源要取并集：搬完数据、消息还没攒起来的群，只有记忆卡里有它，
    只查 messages 的话正好把最该确认的那个群漏掉。
    """
    openids: dict[str, None] = {}
    for table in ("messages", "memory_cards", "scheduled_tasks", "person_addresses"):
        for row in conn.execute(f"SELECT DISTINCT group_openid FROM {table}"):
            openids[str(row[0])] = None
    out = []
    for openid in openids:
        row = conn.execute(
            """SELECT COUNT(*) AS n, MIN(created_at), MAX(created_at)
               FROM messages WHERE group_openid = ?""",
            (openid,),
        ).fetchone()
        count, first, last = int(row[0] or 0), row[1] or "-", row[2] or "-"
        cards = conn.execute(
            "SELECT COUNT(*) FROM memory_cards WHERE group_openid = ?", (openid,)
        ).fetchone()[0]
        tasks = conn.execute(
            "SELECT COUNT(*) FROM scheduled_tasks WHERE group_openid = ? AND status IN ('active','paused')",
            (openid,),
        ).fetchone()[0]
        out.append((openid, count, int(cards), int(tasks), f"{first} ~ {last}"))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def migrate(conn: sqlite3.Connection, src: str, dst: str) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = [("memory_cards", *copy_memory_cards(conn, src, dst))]
    for table in KEY_VALUE_TABLES:
        results.append((table, *copy_key_value_table(conn, table, src, dst)))
    results.append(("scheduled_tasks", *copy_tasks(conn, src, dst)))
    results.append(("conversation_summaries", *copy_summaries(conn, src, dst)))
    return results


def validate_group_id(value: str, label: str) -> str:
    text = (value or "").strip()
    if not GROUP_ID_RE.match(text):
        print(f"{label} 不像 group_openid：{text!r}（应为 8-64 位字母数字）", file=sys.stderr)
        raise SystemExit(2)
    return text


def backup_database(conn: sqlite3.Connection, db_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(f"{db_path.name}.bak-migrate-{stamp}")
    conn.execute("VACUUM INTO ?", (str(target),))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把小毛豆的群数据从一个群搬到另一个群")
    parser.add_argument("--db", default="data/maodou.sqlite3", help="sqlite 数据库路径")
    parser.add_argument("--from", dest="src", help="源群 group_openid")
    parser.add_argument("--to", dest="dst", help="目标群 group_openid")
    parser.add_argument("--list", action="store_true", help="只列出各个群的数据量")
    parser.add_argument("--dry-run", action="store_true", help="试跑：算清楚会搬多少，但不写库")
    parser.add_argument("--no-backup", action="store_true", help="不备份（默认备份）")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"找不到数据库：{db_path}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        if args.list:
            print(f"{'group_openid':36s} {'消息':>6s} {'记忆卡':>6s} {'未跑任务':>8s}  消息时间范围")
            overview = group_overview(conn)
            if not overview:
                print("（库里还没有任何群数据）")
                return 0
            for openid, count, cards, tasks, span in overview:
                print(f"{openid:36s} {count:6d} {cards:6d} {tasks:8d}  {span}")
            return 0

        if not args.src or not args.dst:
            parser.error("要么用 --list，要么同时给 --from 和 --to")
        src = validate_group_id(args.src, "--from")
        dst = validate_group_id(args.dst, "--to")
        if src == dst:
            parser.error("--from 和 --to 不能是同一个群")

        src_cards = conn.execute(
            "SELECT COUNT(*) FROM memory_cards WHERE group_openid = ?", (src,)
        ).fetchone()[0]
        if not src_cards:
            print(f"源群 {src} 一张记忆卡都没有，先确认 openid 对不对（--list 可以列出来）。", file=sys.stderr)
            return 2
        dst_msgs = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE group_openid = ?", (dst,)
        ).fetchone()[0]
        if not dst_msgs:
            print(f"提醒：目标群 {dst} 还没有任何消息记录，确认机器人已经被拉进那个群了。")

        if args.dry_run:
            conn.execute("BEGIN")
            try:
                results = migrate(conn, src, dst)
            finally:
                conn.execute("ROLLBACK")
            print("试跑（没有写库）：")
        else:
            if not args.no_backup:
                snapshot = backup_database(conn, db_path)
                print(f"已备份数据库：{snapshot}")
            conn.execute("BEGIN IMMEDIATE")
            try:
                results = migrate(conn, src, dst)
            except Exception:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            print("已写库：")
        for table, copied, skipped in results:
            print(f"  {table:24s} 新增/覆盖 {copied:4d} 条，跳过 {skipped:3d} 条")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())