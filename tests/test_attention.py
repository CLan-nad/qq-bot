"""无 @ 点名：先由模型判断 direct / ambient / ignore，再决定是否进入主流程。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from maodou.bot import MaodouClient
from maodou.store import ChatStore


GROUP = "attention-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 3, 16, 0, tzinfo=TZ)


class FakeApi:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def post_group_message(self, **kwargs) -> None:
        self.sent.append(kwargs)


class FakeLLM:
    def __init__(self) -> None:
        self.attention_verdicts: list[dict] = []
        self.active_replies: list[str] = []
        self.care_verdicts: list[dict] = []
        self.attention_calls: list[dict] = []
        self.active_judge_calls: list[dict] = []
        self.generate_active_calls: list[dict] = []
        self.care_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def judge_attention(self, **kwargs) -> dict:
        self.attention_calls.append(kwargs)
        return self.attention_verdicts.pop(0) if self.attention_verdicts else {
            "attention": "ignore",
            "reason": "test default",
        }

    async def judge_active_reply(self, **kwargs) -> dict:
        self.active_judge_calls.append(kwargs)
        return {"reply": False, "reason": "legacy default"}

    async def generate_active_reply(self, **kwargs) -> str:
        self.generate_active_calls.append(kwargs)
        return self.active_replies.pop(0) if self.active_replies else ""

    async def judge_care(self, **kwargs) -> dict:
        self.care_calls.append(kwargs)
        return self.care_verdicts.pop(0) if self.care_verdicts else {
            "care": False,
            "text": "",
        }

    async def parse_intent(self, **kwargs) -> dict:
        self.parse_calls.append(kwargs)
        return {"action": "chat"}


def _message(content: str, *, mid: str = "m-1", mention: bool = False):
    mentions = [{"id": "bot-1", "bot": True, "is_you": True}] if mention else []
    return SimpleNamespace(
        content=content,
        id=mid,
        msg_seq=1,
        group_openid=GROUP,
        username="CL",
        member_role="member",
        timestamp=NOW.isoformat(timespec="seconds"),
        author=SimpleNamespace(member_openid="openid-CL", username="CL", member_role="member"),
        raw={"mentions": mentions},
    )


class AttentionBase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "attention.sqlite3"))
        self.settings = SimpleNamespace(
            timezone="Asia/Shanghai",
            allowed_group_openids=frozenset(),
            active_window_seconds=600,
            active_followup_seconds=180,
            presence_quiet_default_minutes=10,
            presence_quiet_max_minutes=1440,
            presence_active_default_minutes=10,
            presence_active_max_minutes=120,
            presence_status_events=8,
            recent_messages=40,
            max_context_chars=4000,
            reply_chunk_chars=900,
            reply_max_chars=500,
            face_tags=False,
            care_enabled=True,
            care_daily_max=2,
            care_cooldown_minutes=0,
            active_msg_daily_max=50,
            vision_max_images_per_request=0,
            ack_after_seconds=0.0,
        )
        self.llm = FakeLLM()
        self.client = self._build()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _build(self) -> MaodouClient:
        client = MaodouClient(settings=self.settings, store=self.store, llm=self.llm)
        client._connection = SimpleNamespace(
            state=SimpleNamespace(robot=SimpleNamespace(name="小毛豆"))
        )
        client.api = FakeApi()
        client._now = lambda: NOW
        client._clock = 1000.0
        client._monotonic = lambda: client._clock
        return client

    def _run(self, awaitable):
        return self.loop.run_until_complete(awaitable)

    def _capture_process(self) -> list[dict]:
        calls: list[dict] = []

        async def process(message, source, *, trigger="mention"):
            calls.append({"message": message, "source": source, "trigger": trigger})

        self.client._process_at_message = process
        return calls


class NameAttentionTests(AttentionBase):
    def test_food_mention_is_ignored_without_intent_call(self) -> None:
        self.llm.attention_verdicts = [
            {"attention": "ignore", "reason": "只是在说毛豆炒肉"}
        ]
        self._run(
            self.client.on_group_message_create(_message("晚上吃毛豆炒肉", mid="m-food"))
        )
        self.assertEqual(len(self.llm.attention_calls), 1)
        self.assertEqual(self.llm.parse_calls, [])
        self.assertEqual(self.client.api.sent, [])

    def test_direct_name_enters_full_intent_flow(self) -> None:
        self.llm.attention_verdicts = [
            {"attention": "direct", "reason": "明确在叫小毛豆"}
        ]
        calls = self._capture_process()
        self._run(
            self.client.on_group_message_create(_message("小毛豆你看看", mid="m-direct"))
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["trigger"], "name")
        self.assertIsNotNone(self.client._active_window(GROUP))

    def test_ambient_name_speaks_naturally(self) -> None:
        self.llm.attention_verdicts = [
            {"attention": "ambient", "reason": "可以接一句"}
        ]
        self.llm.active_replies = ["毛豆在这儿，怎么啦"]
        self._run(
            self.client.on_group_message_create(_message("毛豆今天有点傻", mid="m-ambient"))
        )
        self.assertEqual(len(self.llm.generate_active_calls), 1)
        self.assertEqual(self.llm.generate_active_calls[0]["mode"], "name")
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["毛豆在这儿，怎么啦"],
        )

    def test_quiet_proactive_blocks_ambient_but_allows_direct(self) -> None:
        self.client._handle_social_control(
            {"op": "quiet", "minutes": 30, "scope": "proactive"}, _message("@小毛豆 安静")
        )
        self.llm.attention_verdicts = [
            {"attention": "ambient", "reason": "想接一句"},
            {"attention": "direct", "reason": "明确叫它"},
        ]
        calls = self._capture_process()
        self._run(
            self.client.on_group_message_create(_message("毛豆你看这个", mid="m-q1"))
        )
        self.assertEqual(self.llm.generate_active_calls, [])
        self.assertEqual(self.client.api.sent, [])
        self._run(
            self.client.on_group_message_create(_message("小毛豆你快看", mid="m-q2"))
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["trigger"], "name")

    def test_at_mention_takes_precedence_over_name_candidate(self) -> None:
        calls = self._capture_process()
        self._run(
            self.client.on_group_message_create(
                _message("毛豆在吗", mid="m-at", mention=True)
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["trigger"], "mention")
        self.assertEqual(self.llm.attention_calls, [])

    def test_care_runs_when_name_trigger_stayed_silent(self) -> None:
        self.llm.attention_verdicts = [
            {"attention": "ignore", "reason": "没在叫它"}
        ]
        self.llm.care_verdicts = [{"care": True, "text": "累了就早点休息"}]
        self._run(
            self.client.on_group_message_create(_message("今天好累啊毛豆", mid="m-care"))
        )
        self.assertEqual(len(self.llm.care_calls), 1)
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["累了就早点休息"],
        )

    def test_care_does_not_follow_an_actual_reply(self) -> None:
        self.llm.attention_verdicts = [
            {"attention": "ambient", "reason": "可以接一句"}
        ]
        self.llm.active_replies = ["累就靠过来歇会儿"]
        self.llm.care_verdicts = [{"care": True, "text": "多喝热水"}]
        self._run(
            self.client.on_group_message_create(_message("今天好累啊毛豆", mid="m-care-2"))
        )
        self.assertEqual(len(self.llm.generate_active_calls), 1)
        self.assertEqual(self.llm.care_calls, [])
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["累就靠过来歇会儿"],
        )


if __name__ == "__main__":
    unittest.main()