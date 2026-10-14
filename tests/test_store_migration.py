from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from xiaoxia.store import ChatStore


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
            self.assertFalse(cards[0].pinned)
            self.assertEqual(cards[0].scope, "permanent")

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

    def test_pinned_and_expiry_survive_migration(self) -> None:
        store = ChatStore(self.db_path)
        try:
            card = store.memory_cards(GROUP)[0]
            self.assertTrue(store.set_memory_pinned(GROUP, card.id, True))
            self.assertEqual(store.pinned_memory_contents(GROUP), ["她的真名叫小虾"])
            store.add_memory_card(
                group_openid=GROUP, kind="fact", person="两人", content="临时事项",
                scope="temporary", expires_at="2026-10-20T00:00:00+08:00",
            )
            live = store.live_memory_cards(GROUP, now_iso="2026-10-14T00:00:00+08:00")
            self.assertEqual(len(live), 2)
            expired = store.live_memory_cards(GROUP, now_iso="2026-10-21T00:00:00+08:00")
            self.assertEqual([c.content for c in expired], ["她的真名叫小虾"])
            self.assertEqual(store.purge_expired_memories("2027-01-20T00:00:00+08:00"), 1)
            self.assertEqual(store.count_live_memory_cards(GROUP), 1)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
