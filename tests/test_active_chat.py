"""@ 后十分钟的活跃会话：自然接话、静默补话，以及不越过 action 边界。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from maodou.bot import MaodouClient
from maodou.llm import ACTIVE_REPLY_ADDENDUM, MaodouLLM
from maodou.store import ChatStore, StoredMessage


GROUP = "active-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 29, 15, 0, tzinfo=TZ)


class FakeApi:
    def __init__(self, error: BaseException | None = None) -> None:
        self.sent: list[dict] = []
        self.error = error

    async def post_group_message(self, **kwargs) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(kwargs)


class FakeLLM:
    def __init__(
        self,
        *,
        active_verdicts: list[dict] | None = None,
        active_replies: list[str] | None = None,
        care_verdicts: list[dict] | None = None,
    ) -> None:
        self.active_verdicts = list(active_verdicts or [])
        self.active_replies = list(active_replies or [])
        self.care_verdicts = list(care_verdicts or [])
        self.judge_active_calls: list[dict] = []
        self.generate_active_calls: list[dict] = []
        self.judge_care_calls: list[dict] = []
        self.intent_calls = 0

    async def judge_active_reply(self, **kwargs) -> dict:
        self.judge_active_calls.append(kwargs)
        if self.active_verdicts:
            return self.active_verdicts.pop(0)
        return {"reply": False, "reason": "test default"}

    async def generate_active_reply(self, **kwargs) -> str:
        self.generate_active_calls.append(kwargs)
        return self.active_replies.pop(0) if self.active_replies else ""

    async def judge_care(self, **kwargs) -> dict:
        self.judge_care_calls.append(kwargs)
        return self.care_verdicts.pop(0) if self.care_verdicts else {
            "care": False,
            "text": "",
        }

    async def parse_intent(self, **kwargs):
        self.intent_calls += 1
        raise AssertionError("活跃会话的非 @ 消息绝不能进入 action 分发")


def _message(
    content: str,
    *,
    message_id: str = "message-1",
    username: str = "CL",
    member_openid: str = "openid-CL",
    mentions: list[dict] | None = None,
):
    return SimpleNamespace(
        content=content,
        id=message_id,
        msg_seq=1,
        group_openid=GROUP,
        username=username,
        member_role="member",
        timestamp=NOW.isoformat(timespec="seconds"),
        author=SimpleNamespace(
            member_openid=member_openid,
            username=username,
            member_role="member",
        ),
        raw={"mentions": list(mentions or [])},
    )


class ActiveChatBase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "active.sqlite3"))
        self.settings = SimpleNamespace(
            timezone="Asia/Shanghai",
            allowed_group_openids=frozenset(),
            active_window_seconds=600,
            active_followup_seconds=180,
            recent_messages=40,
            max_context_chars=4000,
            reply_chunk_chars=900,
            reply_max_chars=500,
            face_tags=False,
            care_enabled=True,
            care_daily_max=2,
            care_cooldown_minutes=0,
            active_msg_daily_max=50,
        )
        self.llm = FakeLLM()
        self.client = self._build()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _build(self) -> MaodouClient:
        client = MaodouClient(
            settings=self.settings,
            store=self.store,
            llm=self.llm,
        )
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

    def _save_human(self, content: str, *, message_id: str = "human-1") -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq=1,
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content=content,
            timestamp=NOW.isoformat(timespec="seconds"),
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )


class ActiveWindowLifecycleTests(ActiveChatBase):
    def test_at_starts_the_window(self) -> None:
        async def no_process(*args, **kwargs) -> None:
            return None

        self.client._process_at_message = no_process
        message = _message(
            "@小毛豆 在吗",
            mentions=[{"id": "bot-1", "bot": True, "is_you": True}],
        )
        self._run(self.client.on_group_message_create(message))
        self.assertIsNotNone(self.client._active_window(GROUP))

    def test_window_expires_after_ten_minutes(self) -> None:
        self.client._activate_active_group(GROUP)
        self.client._clock += 599
        self.assertIsNotNone(self.client._active_window(GROUP))
        self.client._clock += 1
        self.assertIsNone(self.client._active_window(GROUP))
        self.assertNotIn(GROUP, self.client._active_windows)

    def test_disabled_window_never_activates(self) -> None:
        self.settings.active_window_seconds = 0
        self.client._activate_active_group(GROUP)
        self.assertIsNone(self.client._active_window(GROUP))


class ActiveMessageTests(ActiveChatBase):
    def test_active_chat_replies_without_entering_action_dispatch(self) -> None:
        self.llm.active_verdicts = [{"reply": True, "reason": "适合接一句"}]
        self.llm.active_replies = ["你们俩又背着我偷偷笑什么"]
        self.client._activate_active_group(GROUP)

        self._run(self.client.on_group_message_create(_message("哈哈哈哈她又开始了")))

        self.assertEqual(self.llm.intent_calls, 0)
        self.assertEqual(len(self.llm.judge_active_calls), 1)
        self.assertEqual(len(self.llm.generate_active_calls), 1)
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["你们俩又背着我偷偷笑什么"],
        )
        recent = self.store.recent_messages(GROUP, 10)
        self.assertEqual(recent[-1].source, "BOT_ACTIVE")

    def test_judge_false_stays_quiet(self) -> None:
        self.llm.active_verdicts = [{"reply": False, "reason": "两人正在直接对话"}]
        self.client._activate_active_group(GROUP)

        self._run(self.client.on_group_message_create(_message("我七点到家")))

        self.assertEqual(len(self.llm.judge_active_calls), 1)
        self.assertEqual(self.llm.generate_active_calls, [])
        self.assertEqual(self.client.api.sent, [])

    def test_failed_send_is_not_reported_as_handled(self) -> None:
        self.llm.active_verdicts = [{"reply": True, "reason": "接一句"}]
        self.llm.active_replies = ["我在呢"]
        self.client.api = FakeApi(RuntimeError("network down"))
        self.client._activate_active_group(GROUP)

        handled = self._run(
            self.client._maybe_active_reply(_message("有人吗"))
        )

        self.assertFalse(handled)
        self.assertEqual(self.client.api.sent, [])

    def test_expired_window_falls_back_to_existing_care_flow(self) -> None:
        self.llm.care_verdicts = [{"care": True, "text": "累了就早点躺下"}]
        self.client._activate_active_group(GROUP)
        self.client._clock += 601

        self._run(self.client.on_group_message_create(_message("今天真的好累")))

        self.assertEqual(self.llm.judge_active_calls, [])
        self.assertEqual(len(self.llm.judge_care_calls), 1)
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["累了就早点躺下"],
        )


class ActiveFollowupTests(ActiveChatBase):
    def test_followup_is_judged_at_most_once(self) -> None:
        self._save_human("那我们晚点再聊")
        self.llm.active_verdicts = [{"reply": True, "reason": "话题还没完全落下"}]
        self.llm.active_replies = ["好呀，我在这儿等你们"]
        self.client._activate_active_group(GROUP)
        self.client._clock += 181

        self._run(self.client._maybe_active_followup())
        self._run(self.client._maybe_active_followup())

        self.assertEqual(len(self.llm.judge_active_calls), 1)
        self.assertEqual(len(self.llm.generate_active_calls), 1)
        self.assertEqual(
            [item["content"] for item in self.client.api.sent],
            ["好呀，我在这儿等你们"],
        )
        self.assertTrue(self.client._active_windows[GROUP].followup_done)

    def test_followup_stays_quiet_when_the_judge_says_no(self) -> None:
        self._save_human("晚安")
        self.llm.active_verdicts = [{"reply": False, "reason": "话题已经结束"}]
        self.client._activate_active_group(GROUP)
        self.client._clock += 181

        self._run(self.client._maybe_active_followup())
        self._run(self.client._maybe_active_followup())

        self.assertEqual(len(self.llm.judge_active_calls), 1)
        self.assertEqual(self.client.api.sent, [])

    def test_followup_waits_until_the_quiet_threshold(self) -> None:
        self._save_human("我先去洗澡")
        self.client._activate_active_group(GROUP)
        self.client._clock += 179

        self._run(self.client._maybe_active_followup())

        self.assertEqual(self.llm.judge_active_calls, [])
        self.assertFalse(self.client._active_windows[GROUP].followup_done)


class ActiveSendingTests(ActiveChatBase):
    def test_text_is_sent_before_the_sticker(self) -> None:
        events: list[tuple[str, str]] = []
        sticker = SimpleNamespace(name="抱抱")

        async def post_group_message(**kwargs) -> None:
            events.append(("text", kwargs["content"]))

        async def upload_group_image(group_openid: str, item) -> str:
            events.append(("upload", item.name))
            return "file-info"

        async def post_sticker_message(group_openid: str, file_info: str, **kwargs) -> None:
            events.append(("sticker", file_info))

        self.client.api = SimpleNamespace(post_group_message=post_group_message)
        self.client._outgoing_parts = lambda group, text: ("想抱抱你", sticker)
        self.client._upload_group_image = upload_group_image
        self.client._post_sticker_message = post_sticker_message

        sent = self._run(
            self.client._send_active_message(GROUP, "想抱抱你[抱抱]", source="BOT_ACTIVE")
        )

        self.assertTrue(sent)
        self.assertEqual(
            events,
            [
                ("text", "想抱抱你"),
                ("upload", "抱抱"),
                ("sticker", "file-info"),
            ],
        )

    def test_rate_limit_response_is_remembered(self) -> None:
        self.client.api = FakeApi(RuntimeError("retcode=40034102"))

        sent = self._run(
            self.client._send_active_message(GROUP, "在呢", source="BOT_ACTIVE")
        )

        self.assertFalse(sent)
        self.assertEqual(
            self.client._proactive_blocked_until[GROUP],
            NOW.date().isoformat(),
        )


class ActiveLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.settings = SimpleNamespace(
            llm_api_key="test-key",
            llm_base_url="https://example.invalid",
            model="main-model",
            care_model="",
            intent_model="cheap-intent",
            vision_model="vision-model",
            intent_reasoning_effort="none",
            reasoning_effort="none",
            reply_max_chars=500,
        )
        self.llm = MaodouLLM(self.settings)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_judge_uses_the_cheap_intent_model_and_json_mode(self) -> None:
        captured: dict = {}

        async def fake_chat(**kwargs) -> str:
            captured.update(kwargs)
            return '{"reply": true, "reason": "这句话在等小毛豆接"}'

        self.llm._chat = fake_chat
        verdict = self.loop.run_until_complete(
            self.llm.judge_active_reply(
                content="快评评理",
                topic_context="【小虾主人】快评评理",
            )
        )

        self.assertTrue(verdict["reply"])
        self.assertEqual(captured["model"], "cheap-intent")
        self.assertTrue(captured["json_mode"])
        self.assertIn("不执行任何任务", captured["system_prompt"])

    def test_silent_marker_becomes_an_empty_reply(self) -> None:
        captured: dict = {}

        async def fake_chat(**kwargs) -> str:
            captured.update(kwargs)
            return "[SILENT]"

        self.llm._chat = fake_chat
        reply = self.loop.run_until_complete(
            self.llm.generate_active_reply(
                requester=StoredMessage(
                    username="CL",
                    member_openid="openid-CL",
                    role="member",
                    content="晚安",
                    timestamp=NOW.isoformat(timespec="seconds"),
                    source="GROUP_MESSAGE_CREATE",
                ),
                request="晚安",
                topic_context="【CL】晚安",
                mode="followup",
                idle_seconds=180,
            )
        )

        self.assertEqual(reply, "")
        self.assertIn(ACTIVE_REPLY_ADDENDUM, captured["system_prompt"])
        self.assertIn("群里已经安静了约 180 秒", captured["user_prompt"])


if __name__ == "__main__":
    unittest.main()
