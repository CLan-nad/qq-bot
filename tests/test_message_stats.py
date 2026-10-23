from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from maodou.bot import IdentityContext, MaodouClient
from maodou.store import ChatStore, StoredMessage


GROUP = "test-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=TZ)


class FakeStatsLLM:
    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def generate_tool_reply(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.reply


def _member(username: str, *, openid: str | None = None) -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid=openid if openid is not None else f"openid-{username}",
        role="member",
        content="",
        timestamp=NOW.isoformat(timespec="seconds"),
        source="GROUP_MESSAGE_CREATE",
    )


class MessageStatsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "stats.sqlite3"))
        self._seq = 0

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save(
        self,
        username: str,
        timestamp: str,
        *,
        openid: str | None = None,
        role: str = "member",
    ) -> None:
        self._seq += 1
        self.store.save_message(
            message_id=f"m-{self._seq}",
            msg_seq="",
            group_openid=GROUP,
            member_openid=openid if openid is not None else f"openid-{username}",
            username=username,
            role=role,
            content=f"消息 {self._seq}",
            timestamp=timestamp,
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_counts_by_member_and_excludes_bot_by_default(self) -> None:
        for _ in range(3):
            self._save("CL", "2026-10-22T10:00:00+08:00")
        for _ in range(2):
            self._save("二十四桥明月夜", "2026-10-22T11:00:00+08:00")
        self.store.save_bot_message(
            group_openid=GROUP,
            content="小毛豆发言",
            timestamp="2026-10-22T12:00:00+08:00",
        )

        stats = self.store.message_stats(
            GROUP,
            since="2026-10-20T00:00:00+08:00",
            until="2026-10-23T00:00:00+08:00",
        )

        self.assertEqual(stats.total, 5)
        self.assertEqual([(m.username, m.count) for m in stats.members], [
            ("CL", 3),
            ("二十四桥明月夜", 2),
        ])
        self.assertFalse(any(m.is_bot for m in stats.members))

    def test_include_bot_adds_a_single_bot_bucket(self) -> None:
        self._save("CL", "2026-10-22T10:00:00+08:00")
        self.store.save_bot_message(
            group_openid=GROUP,
            content="第一句",
            timestamp="2026-10-22T11:00:00+08:00",
        )
        self.store.save_bot_message(
            group_openid=GROUP,
            content="第二句",
            timestamp="2026-10-22T12:00:00+08:00",
        )

        stats = self.store.message_stats(
            GROUP,
            since="2026-10-20T00:00:00+08:00",
            until="2026-10-23T00:00:00+08:00",
            include_bot=True,
        )

        self.assertEqual(stats.total, 3)
        bot = next(member for member in stats.members if member.is_bot)
        self.assertEqual(bot.username, "小毛豆")
        self.assertEqual(bot.count, 2)

    def test_same_openid_keeps_one_person_after_rename(self) -> None:
        self._save("旧名字", "2026-10-22T10:00:00+08:00", openid="same-openid")
        self._save("旧名字", "2026-10-22T11:00:00+08:00", openid="same-openid")
        self._save("新名字", "2026-10-22T12:00:00+08:00", openid="same-openid")
        self._save("另一个人", "2026-10-22T13:00:00+08:00", openid="other-openid")

        stats = self.store.message_stats(GROUP)

        renamed = next(member for member in stats.members if member.member_key == "openid:same-openid")
        self.assertEqual(renamed.count, 3)
        self.assertEqual(renamed.username, "新名字")
        self.assertEqual(stats.total, 4)

    def test_missing_openid_falls_back_to_username(self) -> None:
        self._save("CL", "2026-10-22T10:00:00+08:00", openid="")
        self._save("CL", "2026-10-22T11:00:00+08:00", openid="")
        self._save("二十四桥明月夜", "2026-10-22T12:00:00+08:00", openid="")

        stats = self.store.message_stats(GROUP)

        self.assertEqual(
            [(m.member_key, m.username, m.count) for m in stats.members],
            [
                ("name:CL", "CL", 2),
                ("name:二十四桥明月夜", "二十四桥明月夜", 1),
            ],
        )

    def test_since_is_inclusive_and_until_is_exclusive(self) -> None:
        self._save("CL", "2026-10-20T09:00:00+08:00")
        self._save("CL", "2026-10-20T10:00:00+08:00")
        self._save("CL", "2026-10-20T11:00:00+08:00")

        stats = self.store.message_stats(
            GROUP,
            since="2026-10-20T09:00:00+08:00",
            until="2026-10-20T11:00:00+08:00",
        )

        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.members[0].count, 2)

    def test_range_outside_messages_are_not_counted(self) -> None:
        self._save("CL", "2026-10-19T10:00:00+08:00")
        self._save("CL", "2026-10-20T10:00:00+08:00")
        self._save("CL", "2026-10-21T10:00:00+08:00")

        stats = self.store.message_stats(
            GROUP,
            since="2026-10-20T00:00:00+08:00",
            until="2026-10-21T00:00:00+08:00",
        )

        self.assertEqual(stats.total, 1)

    def test_all_without_range_includes_everything(self) -> None:
        self._save("CL", "2026-10-01T10:00:00+08:00")
        self._save("CL", "2026-10-22T10:00:00+08:00")

        stats = self.store.message_stats(GROUP)

        self.assertEqual(stats.total, 2)


class MessageStatsRangeTests(unittest.TestCase):
    def _range(self, period: str, **extra):
        return MaodouClient._message_stats_range({"period": period, **extra}, NOW)

    def test_today_starts_at_midnight(self) -> None:
        since, until, label = self._range("today")
        self.assertEqual(since, datetime(2026, 10, 23, 0, 0, tzinfo=TZ))
        self.assertEqual(until, NOW)
        self.assertEqual(label, "今天")

    def test_yesterday_ends_at_midnight(self) -> None:
        since, until, label = self._range("yesterday")
        self.assertEqual(since, datetime(2026, 10, 22, 0, 0, tzinfo=TZ))
        self.assertEqual(until, datetime(2026, 10, 23, 0, 0, tzinfo=TZ))
        self.assertEqual(label, "昨天")

    def test_last_three_and_seven_days_are_rolling_windows(self) -> None:
        for period, days in (("last_3_days", 3), ("last_7_days", 7), ("this_week", 7)):
            since, until, _label = self._range(period)
            self.assertEqual(since, NOW - timedelta(days=days))
            self.assertEqual(until, NOW)

    def test_this_month_starts_on_first_day(self) -> None:
        since, until, label = self._range("this_month")
        self.assertEqual(since, datetime(2026, 10, 1, 0, 0, tzinfo=TZ))
        self.assertEqual(until, NOW)
        self.assertEqual(label, "本月")

    def test_all_has_no_start(self) -> None:
        since, until, label = self._range("all")
        self.assertIsNone(since)
        self.assertEqual(until, NOW)
        self.assertEqual(label, "全部历史记录")

    def test_custom_range_includes_end_date(self) -> None:
        since, until, label = self._range(
            "custom", start_date="2026-10-01", end_date="2026-10-03"
        )
        self.assertEqual(since, datetime(2026, 10, 1, 0, 0, tzinfo=TZ))
        self.assertEqual(until, datetime(2026, 10, 4, 0, 0, tzinfo=TZ))
        self.assertIn("2026-10-01", label)
        self.assertIn("2026-10-03", label)

    def test_custom_requires_both_dates(self) -> None:
        with self.assertRaises(ValueError):
            self._range("custom", start_date="2026-10-01")


class MessageStatsHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "handler.sqlite3"))
        self.settings = SimpleNamespace(
            timezone="Asia/Shanghai",
            max_context_chars=24000,
            sticker_dir="",
            face_tags=True,
        )
        self.identity = IdentityContext(
            requester=_member("CL", openid="openid-cl"),
            partner=_member("二十四桥明月夜", openid="openid-xia"),
            speaker_labels={"openid-cl": "主人", "openid-xia": "小虾主人"},
        )
        self._save("CL", "openid-cl", "2026-10-21T10:00:00+08:00")
        self._save("CL", "openid-cl", "2026-10-22T10:00:00+08:00")
        self._save("CL", "openid-cl", "2026-10-23T10:00:00+08:00")
        self._save("二十四桥明月夜", "openid-xia", "2026-10-22T11:00:00+08:00")
        self._save("二十四桥明月夜", "openid-xia", "2026-10-23T11:00:00+08:00")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _save(self, username: str, openid: str, timestamp: str) -> None:
        self.store.save_message(
            message_id=f"{username}-{timestamp}",
            msg_seq="",
            group_openid=GROUP,
            member_openid=openid,
            username=username,
            role="member",
            content="测试消息",
            timestamp=timestamp,
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def _reply(self, llm: FakeStatsLLM) -> str:
        client = MaodouClient(settings=self.settings, store=self.store, llm=llm)
        client._now = lambda: NOW
        return self.loop.run_until_complete(
            client._handle_message_stats(
                intent={"action": "message_stats", "period": "last_3_days"},
                message=SimpleNamespace(group_openid=GROUP),
                request="最近三天我们谁发得多",
                topic="",
                identity=self.identity,
            )
        )

    def test_handler_passes_exact_counts_to_persona_reply(self) -> None:
        llm = FakeStatsLLM("最近三天一共 5 条，主人 3 条，小虾主人 2 条，小虾主人发得多～")
        reply = self._reply(llm)

        self.assertIn("5 条", reply)
        self.assertIn("主人 3 条", reply)
        self.assertIn("小虾主人 2 条", reply)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("小虾主人：2 条", llm.calls[0]["tool_output"])

    def test_handler_falls_back_when_model_drops_counts(self) -> None:
        reply = self._reply(FakeStatsLLM("CL 发得更多～"))

        self.assertIn("共 5 条", reply)
        self.assertIn("主人：3 条", reply)
        self.assertIn("小虾主人：2 条", reply)

    def test_handler_uses_registered_addresses(self) -> None:
        llm = FakeStatsLLM("最近三天一共 5 条，主人 3 条，小虾主人 2 条。")
        reply = self._reply(llm)

        self.assertNotIn("CL：", reply)
        self.assertNotIn("二十四桥明月夜：", reply)


if __name__ == "__main__":
    unittest.main()
