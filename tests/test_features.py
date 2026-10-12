from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from xiaoxia.llm import XiaoxiaLLM
from xiaoxia.scheduling import advance_task, in_quiet_hours, parse_time_of_day, resolve_next_run
from xiaoxia.store import ChatStore


GROUP = "test-group"


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite3"
        self.store = ChatStore(str(self.db_path))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save(self, content: str, username: str = "CL") -> None:
        self.store.save_message(
            message_id=f"id-{content}",
            msg_seq="",
            group_openid=GROUP,
            member_openid=f"openid-{username}",
            username=username,
            role="member",
            content=content,
            timestamp="2026-10-12T21:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_chinese_search_and_memory_cards(self) -> None:
        self._save("我们上次聊到那家火锅餐厅")
        self._save("今天天气不错")

        results = self.store.search_messages(GROUP, "火锅餐厅")
        self.assertEqual(len(results), 1)
        self.assertIn("火锅餐厅", results[0].content)

        card = self.store.add_memory_card(
            group_openid=GROUP,
            kind="preference",
            person="对象",
            content="她不吃香菜",
        )
        self.assertIsNotNone(card)
        self.assertEqual(len(self.store.search_memory_cards(GROUP, "香菜")), 1)
        self.assertTrue(self.store.delete_memory_card(GROUP, card.id))
        self.assertEqual(self.store.search_memory_cards(GROUP, "香菜"), [])

    def test_known_members_keeps_stable_authors(self) -> None:
        self._save("第一条消息", username="CL")
        self.store.save_message(
            message_id="partner-message",
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-对象",
            username="二十四桥明月夜",
            role="member",
            content="第二条消息",
            timestamp="2026-10-12T21:01:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

        members = self.store.known_members(GROUP)
        self.assertEqual(len(members), 2)
        self.assertEqual(
            {member.username for member in members},
            {"CL", "二十四桥明月夜"},
        )
        self.assertEqual(
            {member.member_openid for member in members},
            {"openid-CL", "openid-对象"},
        )

    def test_bot_reply_enters_recent_context(self) -> None:
        self._save("刚才让你搜东西你出故障了知道吗", username="CL")
        self.assertTrue(
            self.store.save_bot_message(
                group_openid=GROUP,
                content="搜索失败了，我这边网络连不上。",
                timestamp="2026-10-13T00:27:00+08:00",
            )
        )
        self._save("啥情况啊", username="CL")

        context = self.store.format_recent_messages(GROUP, 10, 5000)
        self.assertIn("小毛豆", context)
        self.assertIn("搜索失败了", context)
        # 机器人不能被当成第三位群成员
        self.assertEqual(len(self.store.known_members(GROUP)), 1)
        # 自己的回复也要能被历史检索到
        self.assertEqual(len(self.store.search_messages(GROUP, "搜索失败")), 1)

    def test_task_lifecycle(self) -> None:
        task = self.store.create_task(
            group_openid=GROUP,
            title="互道晚安",
            content="晚安",
            generation_prompt=None,
            schedule_type="daily",
            next_run_at="2026-10-12T22:00:00+08:00",
            timezone="Asia/Shanghai",
            time_of_day="22:00",
            weekday=None,
            created_by_openid="openid",
            created_by_name="CL",
        )
        self.assertEqual(self.store.active_task_count(GROUP), 1)
        self.assertEqual(self.store.due_tasks("2026-10-12T22:00:01+08:00"), [task])

        self.store.set_task_status(GROUP, task.id, "paused")
        self.assertEqual(self.store.due_tasks("2026-10-12T22:00:01+08:00"), [])
        self.store.set_task_status(GROUP, task.id, "active")
        self.assertEqual(len(self.store.due_tasks("2026-10-12T22:00:01+08:00")), 1)

        self.store.set_task_skip_next(GROUP, task.id, True)
        self.assertTrue(self.store.tasks(GROUP)[0].skip_next)

        self.store.skip_task_run(
            group_openid=GROUP,
            task_id=task.id,
            next_run_at="2026-10-13T22:00:00+08:00",
        )
        self.assertFalse(self.store.tasks(GROUP)[0].skip_next)

        self.store.delete_task(GROUP, task.id)
        self.assertEqual(self.store.tasks(GROUP), [])

    def test_memory_batch(self) -> None:
        for index in range(3):
            self._save(f"测试消息 {index}")

        batch = self.store.pending_memory_batch(GROUP, min_messages=3)
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch.messages), 3)

        self.store.save_memory_batch(
            group_openid=GROUP,
            start_row_id=batch.start_row_id,
            end_row_id=batch.end_row_id,
            summary="这是一段测试摘要",
            memories=[{"kind": "fact", "person": "两人", "content": "测试记忆"}],
        )
        self.assertEqual(self.store.conversation_summaries(GROUP), ["这是一段测试摘要"])
        self.assertEqual(len(self.store.search_memory_cards(GROUP, "测试记忆")), 1)
        self.assertIsNone(self.store.pending_memory_batch(GROUP, min_messages=3))

    def test_scheduling(self) -> None:
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime(2026, 10, 12, 21, 0, tzinfo=tz)
        parsed_time = parse_time_of_day("22:30")
        self.assertEqual((parsed_time.hour, parsed_time.minute), (22, 30))

        daily = resolve_next_run(
            schedule_type="daily",
            now=now,
            timezone="Asia/Shanghai",
            time_of_day="22:00",
        )
        self.assertEqual(daily.isoformat(), "2026-10-12T22:00:00+08:00")
        next_daily = advance_task(
            schedule_type="daily",
            from_run_at=daily.isoformat(),
            timezone="Asia/Shanghai",
            time_of_day="22:00",
            weekday=None,
        )
        self.assertEqual(next_daily.isoformat(), "2026-10-13T22:00:00+08:00")

        weekly = resolve_next_run(
            schedule_type="weekly",
            now=now,
            timezone="Asia/Shanghai",
            time_of_day="19:00",
            weekday=5,
        )
        self.assertEqual(weekly.weekday(), 4)
        next_weekly = advance_task(
            schedule_type="weekly",
            from_run_at=weekly.isoformat(),
            timezone="Asia/Shanghai",
            time_of_day="19:00",
            weekday=5,
        )
        self.assertEqual((next_weekly - weekly).days, 7)

        once = resolve_next_run(
            schedule_type="once",
            now=now,
            timezone="Asia/Shanghai",
            once_datetime="2027-02-06T09:00:00",
        )
        self.assertEqual(once.isoformat(), "2027-02-06T09:00:00+08:00")

        self.assertTrue(in_quiet_hours(datetime(2026, 10, 13, 0, 1, tzinfo=tz), parse_time_of_day("23:30"), parse_time_of_day("07:00")))

    def test_json_extraction(self) -> None:
        parsed = XiaoxiaLLM._json_content('```json\n{"action":"chat"}\n```')
        self.assertEqual(parsed, {"action": "chat"})

if __name__ == "__main__":
    unittest.main()
