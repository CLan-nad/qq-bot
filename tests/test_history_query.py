"""聊天记录查询：按时间回看、按人筛选，以及 message_stats 的结束日期边界。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from maodou.agent import AgentContext, _execute_tool
from maodou.store import ChatStore


GROUP = "history-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 3, 16, 0, tzinfo=TZ)


class HistoryQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "history.sqlite3"))
        rows = [
            ("m-1", "CL", "openid-CL", "owner", "中午吃了牛肉面", "2026-11-03T12:10:00+08:00"),
            ("m-2", "二十四桥明月夜", "openid-XIA", "admin", "我也想喝奶茶", "2026-11-03T12:30:00+08:00"),
            ("m-3", "CL", "openid-CL", "owner", "下午开会好困", "2026-11-03T15:00:00"),
            ("m-4", "CL", "openid-CL", "owner", "昨天的旧消息", "2026-11-02T12:00:00"),
        ]
        for mid, name, openid, role, content, ts in rows:
            self.store.save_message(
                message_id=mid,
                msg_seq=1,
                group_openid=GROUP,
                member_openid=openid,
                username=name,
                role=role,
                content=content,
                timestamp=ts,
                source="GROUP_MESSAGE_CREATE",
                raw={},
            )
        self.ctx = AgentContext(
            store=self.store,
            group_openid=GROUP,
            now=NOW,
            requester_name="CL",
            requester_openid="openid-CL",
            speaker_labels={"openid-XIA": "小虾主人", "openid-CL": "主人"},
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _run(self, name: str, arguments: dict) -> str:
        return asyncio.run(_execute_tool(name, arguments, self.ctx))

    def test_message_stats_same_day_until_is_inclusive(self) -> None:
        text = self._run(
            "message_stats", {"since": "2026-11-03", "until": "2026-11-03"}
        )
        self.assertIn("共 3 条", text)
        self.assertIn("主人：2 条", text)
        self.assertIn("小虾主人：1 条", text)

    def test_message_stats_until_datetime_excludes_later_records(self) -> None:
        text = self._run(
            "message_stats",
            {"since": "2026-11-03T00:00:00", "until": "2026-11-03T13:00:00"},
        )
        self.assertIn("共 2 条", text)
        self.assertIn("主人：1 条", text)

    def test_memory_search_by_time_returns_actual_messages(self) -> None:
        text = self._run(
            "memory_search",
            {"since": "2026-11-03T12:00:00", "until": "2026-11-03T13:00:00"},
        )
        self.assertIn("中午吃了牛肉面", text)
        self.assertIn("我也想喝奶茶", text)
        self.assertNotIn("下午开会好困", text)

    def test_memory_search_date_until_includes_whole_day(self) -> None:
        text = self._run(
            "memory_search", {"since": "2026-11-03", "until": "2026-11-03"}
        )
        self.assertIn("中午吃了牛肉面", text)
        self.assertIn("下午开会好困", text)
        self.assertNotIn("昨天的旧消息", text)

    def test_memory_search_speaker_filter_resolves_and_limits(self) -> None:
        text = self._run(
            "memory_search",
            {"since": "2026-11-03", "until": "2026-11-03", "speaker": "小虾主人"},
        )
        self.assertIn("我也想喝奶茶", text)
        self.assertNotIn("中午吃了牛肉面", text)

    def test_memory_search_keyword_can_be_limited_by_speaker_and_time(self) -> None:
        text = self._run(
            "memory_search",
            {
                "query": "牛肉面",
                "since": "2026-11-03T12:00:00",
                "until": "2026-11-03T13:00:00",
                "speaker": "我",
            },
        )
        self.assertIn("中午吃了牛肉面", text)
        self.assertNotIn("我也想喝奶茶", text)

    def test_memory_search_requires_query_or_time_range(self) -> None:
        text = self._run("memory_search", {})
        self.assertIn("至少填搜索词或时间范围", text)

    def test_time_fallback_is_explicitly_labeled(self) -> None:
        text = self._run(
            "memory_search",
            {
                "query": "完全不存在的关键词",
                "since": "2026-11-03T12:00:00",
                "until": "2026-11-03T13:00:00",
            },
        )
        self.assertIn("关键词没有直接命中", text)
        self.assertIn("中午吃了牛肉面", text)


if __name__ == "__main__":
    unittest.main()