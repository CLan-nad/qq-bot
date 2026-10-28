"""scripts/migrate_group.py 的测试：换群之后记忆不能跟着丢。

线上真实场景：机器人被拉进新群，group_openid 变了，92 张记忆卡、称呼、群偏好、
定时任务全都留在旧群，新群里「查看记忆」直接回「我现在还没有长期记忆」。
搬运脚本必须做到：试跑不落盘、重跑不重复、只搬还活着的任务、摘要不盖住新群自己的消息。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from maodou.store import ChatStore


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_group.py"
_spec = importlib.util.spec_from_file_location("migrate_group_script", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migrate_group = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_group)

SRC = "SRCGROUP01"
DST = "DSTGROUP02"


class MigrateGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "maodou.sqlite3"
        self.store = ChatStore(str(self.db))
        self.seed()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # ---------- 造数据 ----------

    def seed(self) -> None:
        # 旧群：两张长期记忆、叫法统一、称呼、群偏好、任务、摘要
        self.store.add_memory_card(group_openid=SRC, kind="fact", person="蔡乐", content="用户叫蔡乐")
        self.store.add_memory_card(
            group_openid=SRC, kind="preference", person="小虾", content="讲话要俏皮"
        )
        self.store.set_person_alias(SRC, "罗晓霞", "小虾")
        self.store.set_person_address(SRC, "蔡乐", "主人")
        self.store.set_preference(SRC, "care_daily_max", "3")
        self.main_task = self.store.create_task(
            group_openid=SRC,
            title="纪念日提醒",
            content="纪念日快乐",
            generation_prompt=None,
            schedule_type="yearly",
            next_run_at="2026-12-13T09:00:00+08:00",
            timezone="Asia/Shanghai",
            time_of_day="09:00",
            weekday=None,
            month=12,
            day=13,
            created_by_openid="m-1",
            created_by_name="蔡乐",
        )
        self.warm_task = self.store.create_task(
            group_openid=SRC,
            title="[预热] 纪念日提醒",
            content="还有三天纪念日",
            generation_prompt=None,
            schedule_type="once",
            next_run_at="2026-12-10T09:00:00+08:00",
            timezone="Asia/Shanghai",
            time_of_day="09:00",
            weekday=None,
            created_by_openid="m-1",
            created_by_name="蔡乐",
            linked_task_id=self.main_task.id,
        )
        self.done_task = self.store.create_task(
            group_openid=SRC,
            title="办社保卡",
            content="去银行",
            generation_prompt=None,
            schedule_type="once",
            next_run_at="2026-10-13T11:20:00+08:00",
            timezone="Asia/Shanghai",
            time_of_day=None,
            weekday=None,
            created_by_openid="m-1",
            created_by_name="蔡乐",
        )
        self.store.set_task_status(SRC, self.done_task.id, "done")
        self.store.save_memory_batch(
            group_openid=SRC,
            start_row_id=1,
            end_row_id=900,
            summary="两人一起抓拉特。",
            memories=[],
        )

    def run_script(self, *extra: str) -> tuple[int, str]:
        argv = ["--db", str(self.db), *extra]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = migrate_group.main(argv)
        return code, out.getvalue()

    def migrate(self, *extra: str) -> str:
        code, out = self.run_script("--from", SRC, "--to", DST, *extra)
        self.assertEqual(code, 0, out)
        return out

    def raw(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        try:
            return list(conn.execute(sql, params))
        finally:
            conn.close()

    # ---------- 用例 ----------

    def test_list_mode_shows_both_groups(self) -> None:
        self.store.save_message(
            message_id="d1", msg_seq=0, group_openid=DST, member_openid="m-1",
            username="蔡乐", role="owner", content="hi", timestamp="t", source="GROUP_AT_MESSAGE_CREATE", raw={},
        )
        code, out = self.run_script("--list")
        self.assertEqual(code, 0)
        self.assertIn(SRC, out)
        self.assertIn(DST, out)

    def test_dry_run_does_not_write(self) -> None:
        out = self.migrate("--dry-run", "--no-backup")
        self.assertIn("试跑", out)
        self.assertIn("memory_cards", out)
        self.assertEqual(self.store.memory_cards(DST), [])
        self.assertEqual(self.store.person_addresses(DST), {})
        self.assertEqual(self.store.tasks(DST), [])

    def test_migration_moves_everything_that_still_matters(self) -> None:
        out = self.migrate("--no-backup")
        self.assertIn("已写库", out)

        cards = {card.content: card for card in self.store.memory_cards(DST)}
        self.assertEqual(set(cards), {"用户叫蔡乐", "讲话要俏皮"})

        # 叫法统一和称呼都要跟着走，否则新群里模型又开始乱叫人
        self.assertEqual(self.store.person_aliases(DST), {"罗晓霞": "小虾"})
        self.assertEqual(self.store.canonical_person(DST, "罗晓霞"), "小虾")
        self.assertEqual(self.store.address_for(DST, "蔡乐"), "主人")
        self.assertEqual(self.store.get_preference(DST, "care_daily_max"), "3")

    def test_only_live_tasks_are_copied_and_links_are_remapped(self) -> None:
        self.migrate("--no-backup")
        titles = sorted(task.title for task in self.store.tasks(DST))
        self.assertEqual(titles, ["[预热] 纪念日提醒", "纪念日提醒"])

        rows = self.raw(
            "SELECT title, linked_task_id FROM scheduled_tasks WHERE group_openid = ? ORDER BY id",
            (DST,),
        )
        by_title = {row["title"]: row["linked_task_id"] for row in rows}
        main_id = [r["id"] for r in self.raw(
            "SELECT id FROM scheduled_tasks WHERE group_openid = ? AND title = '纪念日提醒'", (DST,)
        )][0]
        self.assertEqual(by_title["[预热] 纪念日提醒"], main_id)
        self.assertNotEqual(main_id, self.main_task.id)
        # done 的历史任务不该跟着搬
        self.assertNotIn("办社保卡", by_title)

    def test_summaries_come_along_when_they_cannot_shadow_new_messages(self) -> None:
        self.migrate("--no-backup")
        self.assertEqual(self.store.conversation_summaries(DST), ["两人一起抓拉特。"])
        # 摘要器按 MAX(end_row_id) 往后取消息，边界要正好停在新群第一条之前
        self.assertEqual(self.store.latest_summary_row_id(DST), 900)

    def test_summaries_are_skipped_when_destination_messages_come_first(self) -> None:
        # 新群的消息 row_id 比旧群摘要覆盖的范围还小，搬过去会让这些消息永远总结不到
        self.store.save_message(
            message_id="d1", msg_seq=0, group_openid=DST, member_openid="m-1",
            username="蔡乐", role="owner", content="hi", timestamp="t", source="GROUP_AT_MESSAGE_CREATE", raw={},
        )
        first_row = self.raw(
            "SELECT MIN(row_id) AS r FROM messages WHERE group_openid = ?", (DST,)
        )[0]["r"]
        self.assertLess(first_row, 900)
        out = self.migrate("--no-backup")
        self.assertEqual(self.store.conversation_summaries(DST), [])
        self.assertIn("conversation_summaries", out)

    def test_second_run_changes_nothing(self) -> None:
        self.migrate("--no-backup")
        before_cards = len(self.store.memory_cards(DST))
        before_tasks = len(self.store.tasks(DST))
        out = self.migrate("--no-backup")
        self.assertEqual(len(self.store.memory_cards(DST)), before_cards)
        self.assertEqual(len(self.store.tasks(DST)), before_tasks)
        self.assertIn("跳过", out)

    def test_backup_snapshot_is_written_by_default(self) -> None:
        self.migrate()
        backups = list(self.db.parent.glob(f"{self.db.name}.bak-migrate-*"))
        self.assertEqual(len(backups), 1)
        # 备份得是能独立打开的完整库
        conn = sqlite3.connect(str(backups[0]))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM memory_cards WHERE group_openid = ?", (SRC,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 2)

    def test_source_without_cards_is_refused(self) -> None:
        # 群号打错时最该拦住：不然会拿一个空群把目标群的数据「搬」成空的
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = self.run_script("--from", "EMPTYGROUP9", "--to", DST)
        self.assertEqual(code, 2)
        self.assertIn("EMPTYGROUP9", err.getvalue())

    def test_same_group_and_bad_openid_are_refused(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.run_script("--from", SRC, "--to", SRC)
            self.assertEqual(ctx.exception.code, 2)
            with self.assertRaises(SystemExit) as ctx:
                self.run_script("--from", "a b", "--to", DST)
            self.assertEqual(ctx.exception.code, 2)
        self.assertIn("group_openid", err.getvalue())

    def test_missing_database_exits_2(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = migrate_group.main(["--db", str(self.db.parent / "nope.sqlite3"), "--list"])
        self.assertEqual(code, 2)

    def test_destination_group_is_rewritten_not_left_behind(self) -> None:
        """回归：任务曾经忘了改 group_openid，结果是往源群里又插了一份重复任务。"""
        self.migrate("--no-backup")
        rows = self.raw(
            "SELECT group_openid, COUNT(*) AS n FROM scheduled_tasks GROUP BY group_openid"
        )
        counts = {row["group_openid"]: row["n"] for row in rows}
        self.assertEqual(counts.get(DST), 2)
        self.assertEqual(counts.get(SRC), 3)
        # 再跑一遍也不能把任何一边的条数拱上去
        self.migrate("--no-backup")
        rows = self.raw(
            "SELECT group_openid, COUNT(*) AS n FROM scheduled_tasks GROUP BY group_openid"
        )
        counts = {row["group_openid"]: row["n"] for row in rows}
        self.assertEqual(counts, {DST: 2, SRC: 3})


if __name__ == "__main__":
    unittest.main()