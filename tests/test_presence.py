"""社交在场状态：模型决定安静/回来，代码只做持久化与边界保护。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from maodou.bot import MaodouClient
from maodou.store import ChatStore


GROUP = "presence-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 3, 15, 0, tzinfo=TZ)


class FakeApi:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def post_group_message(self, **kwargs) -> None:
        self.sent.append(kwargs)


class FakeLLM:
    def __init__(self) -> None:
        self.intent = {"action": "chat"}
        self.parse_calls: list[dict] = []
        self.attention_verdicts: list[dict] = []
        self.active_verdicts: list[dict] = []
        self.active_replies: list[str] = []
        self.generate_active_calls: list[dict] = []

    async def parse_intent(self, **kwargs) -> dict:
        self.parse_calls.append(kwargs)
        return dict(self.intent)

    async def judge_attention(self, **kwargs) -> dict:
        return self.attention_verdicts.pop(0) if self.attention_verdicts else {
            "attention": "ignore",
            "reason": "test default",
        }

    async def judge_active_reply(self, **kwargs) -> dict:
        return self.active_verdicts.pop(0) if self.active_verdicts else {
            "reply": False,
            "reason": "test default",
        }

    async def generate_active_reply(self, **kwargs) -> str:
        self.generate_active_calls.append(kwargs)
        return self.active_replies.pop(0) if self.active_replies else ""

    async def judge_care(self, **kwargs) -> dict:
        return {"care": False, "text": ""}


def _message(content: str = "@小毛豆 在吗", *, mid: str = "m-1", mention: bool = True):
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


class PresenceBase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "presence.sqlite3"))
        self.now = NOW
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
        client._now = lambda: self.now
        client._clock = 1000.0
        client._monotonic = lambda: client._clock
        return client

    def _run(self, awaitable):
        return self.loop.run_until_complete(awaitable)

    def _quiet(self, **intent) -> str:
        payload = {"op": "quiet", "minutes": 15, "scope": "proactive"}
        payload.update(intent)
        return self.client._handle_social_control(payload, _message())


class QuietStateTests(PresenceBase):
    def test_quiet_persists_across_clients(self) -> None:
        self._quiet(reason="今天先安静一会儿")
        state = self.store.get_social_state(GROUP)
        self.assertIsNotNone(state)
        self.assertEqual(state.scope, "proactive")
        self.assertEqual(state.reason, "今天先安静一会儿")
        until = datetime.fromisoformat(state.until_at)
        self.assertEqual(until, NOW + timedelta(minutes=15))

        # 模拟重启：关闭旧连接，用同一个数据库文件重新建客户端。
        self.store.close()
        self.store = ChatStore(str(Path(self.tmp.name) / "presence.sqlite3"))
        self.client = self._build()
        self.assertTrue(self.client._is_quiet(GROUP))

    def test_quiet_expires_lazily_and_is_cleared(self) -> None:
        self._quiet(minutes=5)
        self.now = NOW + timedelta(minutes=6)
        self.assertFalse(self.client._is_quiet(GROUP))
        self.assertIsNone(self.store.get_social_state(GROUP))
        events = self.store.recent_social_events(GROUP, limit=5)
        self.assertIn("quiet_expired", [item.event for item in events])

    def test_model_minutes_are_clamped_to_config(self) -> None:
        self._quiet(minutes=99999)
        state = self.store.get_social_state(GROUP)
        until = datetime.fromisoformat(state.until_at)
        self.assertEqual(until, NOW + timedelta(minutes=1440))

    def test_bad_scope_falls_back_to_proactive(self) -> None:
        self._quiet(scope="everything")
        self.assertEqual(self.store.get_social_state(GROUP).scope, "proactive")

    def test_quiet_and_available_are_audited(self) -> None:
        self._quiet()
        self.client._handle_social_control({"op": "available"}, _message())
        events = [item.event for item in self.store.recent_social_events(GROUP, 10)]
        self.assertIn("quiet", events)
        self.assertIn("available", events)
        self.assertIsNone(self.store.get_social_state(GROUP))
        self.assertIsNotNone(self.client._active_window(GROUP))


class QuietSuppressionTests(PresenceBase):
    def test_proactive_quiet_blocks_active_reply_and_care(self) -> None:
        self.client._handle_social_control({"op": "quiet", "minutes": 30}, _message())
        self._run(
            self.client.on_group_message_create(
                _message("今天真的好累", mid="m-2", mention=False)
            )
        )
        self.assertEqual(self.llm.generate_active_calls, [])
        self.assertEqual(self.client.api.sent, [])

    def test_all_quiet_suppresses_plain_at_before_answer(self) -> None:
        self.client._handle_social_control(
            {"op": "quiet", "minutes": 30, "scope": "all"}, _message()
        )
        answered: list[dict] = []

        async def answer(**kwargs):
            answered.append(kwargs)
            return True

        self.client._answer_with_ack = answer
        self.llm.intent = {"action": "chat"}
        self._run(self.client._process_at_message(_message(mid="m-3"), "GROUP_AT_MESSAGE_CREATE"))
        self.assertEqual(self.llm.parse_calls and len(self.llm.parse_calls), 1)
        self.assertEqual(answered, [])
        self.assertEqual(self.client.api.sent, [])

    def test_all_quiet_still_allows_available_command(self) -> None:
        self.client._handle_social_control(
            {"op": "quiet", "minutes": 30, "scope": "all"}, _message()
        )
        answered: list[dict] = []

        async def answer(*, intent, message, **kwargs):
            answered.append(intent)
            self.client._handle_social_control(intent, message)
            return True

        self.client._answer_with_ack = answer
        self.llm.intent = {
            "action": "social_control",
            "op": "available",
            "reply": "好，我回来啦。",
        }
        self._run(self.client._process_at_message(_message(mid="m-4"), "GROUP_AT_MESSAGE_CREATE"))
        self.assertEqual(len(answered), 1)
        self.assertIsNone(self.store.get_social_state(GROUP))
        self.assertIsNotNone(self.client._active_window(GROUP))


class PresenceStatusTests(PresenceBase):
    def test_show_reports_deadline_scope_and_recent_decisions(self) -> None:
        self._quiet(minutes=30, scope="all", reason="主人让闭嘴")
        text = self.client._handle_social_control({"op": "show"}, _message())
        self.assertIn("安静模式", text)
        self.assertIn("连 @ 也暂不回应", text)
        self.assertIn("主人让闭嘴", text)
        self.assertIn("最近决策", text)

    def test_available_without_quiet_still_activates(self) -> None:
        text = self.client._handle_social_control({"op": "available"}, _message())
        self.assertTrue(text)
        window = self.client._active_window(GROUP)
        self.assertIsNotNone(window)
        self.assertEqual(window.minutes, 10)


class ActiveMinutesTests(PresenceBase):
    def test_model_can_choose_active_window(self) -> None:
        self.client._activate_active_group(GROUP, minutes=45, source="test")
        window = self.client._active_window(GROUP)
        self.assertEqual(window.minutes, 45)
        self.assertEqual(int(window.until - self.client._clock), 45 * 60)

    def test_active_minutes_from_intent_resizes_window(self) -> None:
        async def answer(**kwargs):
            return True

        self.client._answer_with_ack = answer
        self.llm.intent = {"action": "chat", "active_minutes": 25}
        self._run(self.client._process_at_message(_message(mid="m-5"), "GROUP_AT_MESSAGE_CREATE"))
        window = self.client._active_window(GROUP)
        self.assertIsNotNone(window)
        self.assertEqual(window.minutes, 25)

    def test_zero_active_minutes_ends_window(self) -> None:
        self.client._activate_active_group(GROUP)
        self.client._activate_active_group(GROUP, minutes=0, source="model")
        self.assertIsNone(self.client._active_window(GROUP))
        events = [item.event for item in self.store.recent_social_events(GROUP, 5)]
        self.assertIn("active_ended", events)


if __name__ == "__main__":
    unittest.main()