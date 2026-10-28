from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from maodou.store import ChatStore


GROUP = "test-group"

# 这是加农历/每年任务之前的老 schema：memory_cards 没有 pinned/scope/expires_at，
# scheduled_tasks 的 CHECK 里也没有 yearly。用它模拟线上那个 489 条消息的库。
OLD_SCHEMA = """
CREATE TABLE memory_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_openid TEXT NOT NULL,
    kind TEXT NOT NULL,
    person TEXT NOT NULL,
    content TEXT NOT NULL,
    source_message_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX idx_memory_cards_content ON memory_cards(group_openid, content);
CREATE TABLE scheduled_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_openid TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    generation_prompt TEXT,
    schedule_type TEXT NOT NULL CHECK (schedule_type IN ('once', 'daily', 'weekly')),
    next_run_at TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
    time_of_day TEXT,
    weekday INTEGER,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'done', 'deleted')),
    skip_next INTEGER NOT NULL DEFAULT 0,
    last_sent_at TEXT,
    failure_count INTEGER NOT NULL DEFAULT 0,
    created_by_openid TEXT NOT NULL,
    created_by_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_scheduled_tasks_due ON scheduled_tasks(status, next_run_at);
"""


class StoreMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "old.sqlite3")
        conn = sqlite3.connect(self.db_path)
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO memory_cards (group_openid, kind, person, content) VALUES (?, ?, ?, ?)",
            (GROUP, "fact", "她", "她的真名叫小虾"),
        )
        conn.execute(
            """
            INSERT INTO scheduled_tasks (
                group_openid, title, content, schedule_type, next_run_at, timezone,
                time_of_day, weekday, created_by_openid, created_by_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (GROUP, "互道晚安", "晚安", "daily", "2026-10-12T22:00:00+08:00",
             "Asia/Shanghai", "22:00", None, "openid", "CL"),
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _columns(self, table: str) -> set[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        finally:
            conn.close()

    def test_old_database_Upgrade_and_keep_data(self) -> None:
        store = ChatStore(self.db_path)
        try:
            self.assertTrue(
                {"pinned", "scope", "expires_at", "source_row_id"} <= self._columns("memory_cards")
            )
            self.assertTrue(
                {"month", "day", "leap_month", "remind_days_ahead", "offset_days", "linked_task_id"}
                <= self._columns("scheduled_tasks")
            )

            cards = store.memory_cards(GROUP)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].content, "她的真名叫小虾")
            # 业务对象只暴露长期记忆，不再带置顶/临时标记。
            for legacy in ("pinned", "scope", "expires_at"):
                self.assertFalse(hasattr(cards[0], legacy))
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT pinned, scope, expires_at FROM memory_cards WHERE group_openid = ?",
                    (GROUP,),
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(row, (0, "permanent", None))

            tasks = store.tasks(GROUP)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].id, 1)
            self.assertEqual(tasks[0].next_run_at, "2026-10-12T22:00:00+08:00")
            self.assertEqual(tasks[0].offset_days, 0)

            # CHECK 约束升级后才能真正建出每年任务
            yearly = store.create_task(
                group_openid=GROUP, title="纪念日", content="在一起 3 年",
                generation_prompt=None, schedule_type="yearly",
                next_run_at="2026-12-13T09:00:00+08:00", timezone="Asia/Shanghai",
                time_of_day="09:00", weekday=None, created_by_openid="openid",
                created_by_name="CL", month=12, day=13, remind_days_ahead=3,
            )
            self.assertEqual(yearly.month, 12)
            self.assertEqual(yearly.remind_days_ahead, 3)
            self.assertTrue(store.due_tasks("2026-10-13T00:00:00+08:00"))
        finally:
            store.close()

    def test_migration_is_idempotent(self) -> None:
        first = ChatStore(self.db_path)
        first.close()
        second = ChatStore(self.db_path)
        try:
            self.assertEqual(len(second.memory_cards(GROUP)), 1)
            self.assertEqual(len(second.tasks(GROUP)), 1)
            third = ChatStore(self.db_path)
            third.close()
            self.assertEqual(len(second.tasks(GROUP)), 1)
        finally:
            second.close()

    def test_delete_cascades_to_warmup_task(self) -> None:
        store = ChatStore(self.db_path)
        try:
            main = store.create_task(
                group_openid=GROUP, title="纪念日", content="在一起 3 年",
                generation_prompt=None, schedule_type="yearly",
                next_run_at="2026-12-13T09:00:00+08:00", timezone="Asia/Shanghai",
                time_of_day="09:00", weekday=None, created_by_openid="openid",
                created_by_name="CL", month=12, day=13, remind_days_ahead=3,
            )
            warm = store.create_task(
                group_openid=GROUP, title="[预热] 纪念日", content="还有 3 天",
                generation_prompt=None, schedule_type="yearly",
                next_run_at="2026-12-10T09:00:00+08:00", timezone="Asia/Shanghai",
                time_of_day="09:00", weekday=None, created_by_openid="openid",
                created_by_name="CL", month=12, day=13, offset_days=3,
                linked_task_id=main.id,
            )
            self.assertEqual(len(store.tasks(GROUP)), 3)
            self.assertTrue(store.delete_task(GROUP, main.id))
            remaining = {task.id: task.status for task in store.tasks(GROUP)}
            self.assertNotIn(warm.id, remaining)
            self.assertEqual(store.get_task(GROUP, warm.id).status, "deleted")
            self.assertEqual(store.get_task(GROUP, main.id).status, "deleted")
            # 老的 daily 任务不该被牵连
            self.assertEqual(len([t for t in store.tasks(GROUP) if t.schedule_type == "daily"]), 1)
        finally:
            store.close()

    def test_legacy_pin_and_expiry_are_normalized_to_long_term(self) -> None:
        first = ChatStore(self.db_path)
        first.close()

        # 模拟线上旧库：一张卡曾被置顶、另一张曾被标成 14 天后过期的临时卡。
        # 业务上它们现在都只是普通长期记忆，启动迁移只清标记，不删任何内容。
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_cards SET pinned = 1, scope = 'permanent' WHERE group_openid = ?",
                (GROUP,),
            )
            conn.execute(
                """
                INSERT INTO memory_cards (
                    group_openid, kind, person, content, scope, expires_at
                ) VALUES (?, 'fact', '两人', '临时事项', 'temporary', '2026-10-20T00:00:00+08:00')
                """,
                (GROUP,),
            )
            conn.commit()
        finally:
            conn.close()

        store = ChatStore(self.db_path)
        try:
            cards = store.all_memory_cards(GROUP)
            self.assertEqual(
                [card.content for card in cards], ["她的真名叫小虾", "临时事项"]
            )
            self.assertEqual(store.count_memory_cards(GROUP), 2)
            self.assertFalse(hasattr(store, "set_memory_pinned"))
            self.assertFalse(hasattr(store, "purge_expired_memories"))

            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    """
                    SELECT content, pinned, scope, expires_at
                    FROM memory_cards WHERE group_openid = ? ORDER BY id
                    """,
                    (GROUP,),
                ).fetchall()
            finally:
                conn.close()
            self.assertEqual(
                rows,
                [
                    ("她的真名叫小虾", 0, "permanent", None),
                    ("临时事项", 0, "permanent", None),
                ],
            )
        finally:
            store.close()


class CareTablesTests(unittest.TestCase):
    """主动关心那两张表是启动时顺手建出来的，旧库不用停机迁移。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "old2.sqlite3")
        conn = sqlite3.connect(self.db_path)
        conn.executescript(OLD_SCHEMA)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tables_are_created_and_usable(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("care_log", names)
            self.assertNotIn("group_preferences", names)
        finally:
            conn.close()

        store = ChatStore(self.db_path)
        try:
            key = "m-1:1"
            first = store.reserve_care_message(
                message_key=key,
                group_openid=GROUP,
                member_openid="openid",
                content="睡不着",
                created_at="2026-10-14T15:00:00+08:00",
            )
            self.assertTrue(first)
            self.assertFalse(
                store.reserve_care_message(
                    message_key=key,
                    group_openid=GROUP,
                    member_openid="openid",
                    content="睡不着",
                    created_at="2026-10-14T15:00:00+08:00",
                )
            )

            def sent_count() -> int:
                return store.sent_care_count(
                    group_openid=GROUP,
                    member_openid="openid",
                    since_iso="2026-10-14T00:00:00+08:00",
                )

            self.assertEqual(sent_count(), 0)
            store.mark_care_sent(key)
            self.assertEqual(sent_count(), 1)
            self.assertEqual(
                store.last_care_at(group_openid=GROUP, member_openid="openid"),
                "2026-10-14T15:00:00+08:00",
            )

            store.set_preference(GROUP, "care_daily_max", "3")
            self.assertEqual(store.get_preference(GROUP, "care_daily_max"), "3")
            store.set_preference(GROUP, "care_daily_max", "1")
            self.assertEqual(store.preferences(GROUP), {"care_daily_max": "1"})
            self.assertTrue(store.clear_preference(GROUP, "care_daily_max"))
            self.assertIsNone(store.get_preference(GROUP, "care_daily_max"))
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
