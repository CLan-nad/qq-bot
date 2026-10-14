"""主动关心：关键词预筛、便宜模型二级判定、节流、频控退避、群内偏好。

_now() 钉在 2026-10-14 15:00，否则冷却和“今天第几次”的断言会随真实时间漂。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from xiaoxia.bot import XiaoxiaClient
from xiaoxia.care import matched_signals, normalize_care_text, should_ask_model
from xiaoxia.llm import XiaoxiaLLM
from xiaoxia.store import ChatStore


GROUP = "test-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=TZ)
TODAY_START = "2026-10-14T00:00:00+08:00"


class FakeApi:
    def __init__(self, error=None):
        self.sent: list[dict] = []
        self.error = error

    async def post_group_message(self, **kwargs):
        if self.error is not None:
            raise self.error
        self.sent.append(kwargs)


class FakeLLM:
    def __init__(self, verdicts=()):
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def judge_care(self, *, content: str, recent_context: str = ""):
        self.calls.append({"content": content, "recent_context": recent_context})
        verdict = self.verdicts.pop(0) if self.verdicts else {"care": False, "text": ""}
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict


def _message(
    content="今天好累，睡不着",
    *,
    mid="m-1",
    seq=1,
    member="openid-CL",
    username="CL",
    group=GROUP,
    raw=None,
):
    return SimpleNamespace(
        content=content,
        id=mid,
        msg_seq=seq,
        group_openid=group,
        username=username,
        member_role="member",
        timestamp="2026-10-14T15:00:00+08:00",
        author=SimpleNamespace(member_openid=member, username=username, member_role="member"),
        raw={"mentions": []} if raw is None else raw,
    )


class CareBase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "care.sqlite3"))
        self.settings = SimpleNamespace(
            timezone="Asia/Shanghai",
            allowed_group_openids=frozenset(),
            care_enabled=True,
            care_model="",
            vision_model="cheap-model",
            care_daily_max=2,
            care_cooldown_minutes=180,
            active_msg_daily_max=6,
            max_context_chars=4000,
            reply_max_chars=500,
            task_max_count=20,
        )
        self.llm = FakeLLM()
        self.build()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def build(self, **overrides) -> XiaoxiaClient:
        for key, value in overrides.items():
            setattr(self.settings, key, value)
        self.client = XiaoxiaClient(
            settings=self.settings, store=self.store, llm=self.llm
        )
        # _remember_bot_reply 会读 robot.name，裸实例没有连接，这里塞个假的。
        self.client._connection = SimpleNamespace(
            state=SimpleNamespace(robot=SimpleNamespace(name="小毛豆"))
        )
        self.api = FakeApi()
        self.client.api = self.api
        self.client._now = lambda: NOW
        return self.client

    def care(self, message) -> None:
        self.loop.run_until_complete(self.client._maybe_care(message))

    def sent_texts(self) -> list[str]:
        return [call["content"] for call in self.api.sent]


class SignalGateTests(CareBase):
    def test_plain_chat_never_reaches_the_model(self) -> None:
        self.care(_message("今天天气不错"))
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.api.sent, [])

    def test_sticker_and_link_and_at_are_ignored(self) -> None:
        for content in ("[笑哭]", "看看这个 https://example.com/a", "@小毛豆 帮我查天气"):
            with self.subTest(content=content):
                self.care(_message(content, mid=content[:4]))
        self.assertEqual(self.llm.calls, [])

    def test_body_and_mood_signals_hit(self) -> None:
        self.assertTrue(should_ask_model("中午吃了冰的 肚子好疼"))
        self.assertTrue(should_ask_model("emo了 被导师骂了一顿"))
        self.assertIn("拉肚子", matched_signals("拉肚子难受"))
        self.assertIn("睡不着", matched_signals("一直睡不着"))

    def test_pain_words_split_by_a_colloquial_adverb_still_hit(self) -> None:
        # 口语会说“肚子好疼”“头有点痛”，整词匹配不到，得靠部位+痛感组合。
        self.assertIn("肚子疼", matched_signals("中午吃了冰的 肚子好疼"))
        self.assertIn("头痛", matched_signals("吹空调吹的 头有点痛"))
        self.assertTrue(should_ask_model("吹空调吹的 头有点痛"))

    def test_very_long_forward_is_ignored(self) -> None:
        self.assertFalse(should_ask_model("睡不着" + "啊" * 90))


class CareFlowTests(CareBase):
    def test_care_is_sent_and_recorded(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "拉肚子记得多喝点温水"}]
        self.care(_message())
        self.assertEqual(len(self.api.sent), 1)
        call = self.api.sent[0]
        self.assertEqual(call["group_openid"], GROUP)
        self.assertEqual(call["content"], "拉肚子记得多喝点温水")
        self.assertNotIn("msg_id", call)
        self.assertNotIn("msg_seq", call)
        self.assertEqual(
            self.store.proactive_message_count(group_openid=GROUP, since_iso=TODAY_START), 1
        )
        self.assertEqual(self.store.sent_care_count(
            group_openid=GROUP, member_openid="openid-CL", since_iso=TODAY_START
        ), 1)

    def test_long_care_text_is_cut_to_one_short_sentence(self) -> None:
        self.llm.verdicts = [
            {"care": True, "text": "\u201c拉肚子记得多喝温水，别吃冰的了，晚上早点休息，我在呢\u201d"}
        ]
        self.care(_message())
        text = self.sent_texts()[0]
        self.assertEqual(len(text), 25)
        self.assertTrue(text.startswith("拉肚子记得多喝温水"))
        self.assertNotIn("\u201c", text)

    def test_model_saying_no_sends_nothing_and_keeps_quota(self) -> None:
        self.llm.verdicts = [{"care": False, "text": ""}]
        self.care(_message(mid="m-a"))
        self.assertEqual(self.api.sent, [])
        # 没发出去就不占额度，下一条照样有资格被判断。
        self.llm.verdicts = [{"care": True, "text": "抱抱"}]
        self.care(_message("还是有点难受", mid="m-b"))
        self.assertEqual(len(self.llm.calls), 2)
        self.assertEqual(self.sent_texts(), ["抱抱"])

    def test_empty_text_from_model_is_not_sent(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "   "}]
        self.care(_message())
        self.assertEqual(self.api.sent, [])

    def test_model_failure_means_stay_quiet(self) -> None:
        self.llm.verdicts = [RuntimeError("model down")]
        self.care(_message())
        self.assertEqual(self.api.sent, [])

    def test_unmentioned_group_message_reaches_care(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "别熬夜了，明天还要上班"}]
        self.loop.run_until_complete(
            self.client.on_group_message_create(_message("昨晚又失眠了"))
        )
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(self.sent_texts(), ["别熬夜了，明天还要上班"])
        recent = self.store.recent_messages(GROUP, 10)
        self.assertEqual({row.source for row in recent}, {"GROUP_MESSAGE_CREATE", "BOT_CARE"})

    def test_context_from_recent_chat_is_passed_to_the_model(self) -> None:
        self.store.save_bot_message(
            group_openid=GROUP,
            content="已经安慰过她啦",
            timestamp="2026-10-14T14:50:00+08:00",
            source="BOT_REPLY",
        )
        self.care(_message())
        self.assertIn("已经安慰过她啦", self.llm.calls[0]["recent_context"])


class ThrottleTests(CareBase):
    def test_disabled_by_env_never_calls_the_model(self) -> None:
        self.build(care_enabled=False)
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

    def test_group_preference_beats_env(self) -> None:
        self.store.set_preference(GROUP, "care_enabled", "false")
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

        self.settings.care_enabled = False
        self.store.set_preference(GROUP, "care_enabled", "true")
        self.care(_message())
        self.assertEqual(len(self.llm.calls), 1)

    def test_same_person_has_to_cooldown(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "早点休息"}] * 2
        self.care(_message(mid="m-1"))
        self.care(_message("还是难受", mid="m-2"))
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(len(self.api.sent), 1)

    def test_other_person_is_not_blocked_by_the_cooldown(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "辛苦了"}, {"care": True, "text": "我在"}]
        self.care(_message(mid="m-1", member="openid-CL", username="CL"))
        self.care(_message("肚子疼", mid="m-2", member="openid-X", username="小虾"))
        self.assertEqual(len(self.llm.calls), 2)
        self.assertEqual(len(self.api.sent), 2)

    def test_per_person_daily_cap(self) -> None:
        self.build(care_cooldown_minutes=0, care_daily_max=2)
        self.llm.verdicts = [{"care": True, "text": "抱抱"}] * 3
        for index in range(3):
            self.care(_message("还是有点难受", mid=f"m-{index}"))
        self.assertEqual(len(self.llm.calls), 2)
        self.assertEqual(len(self.api.sent), 2)

    def test_group_budget_is_shared_with_scheduled_tasks(self) -> None:
        for index in range(6):
            self.store.save_bot_message(
                group_openid=GROUP,
                content=f"定时任务 {index}",
                timestamp="2026-10-14T09:00:00+08:00",
                source="BOT_TASK",
            )
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

    def test_zero_caps_shut_it_down(self) -> None:
        self.build(active_msg_daily_max=0)
        self.care(_message())
        self.assertEqual(self.llm.calls, [])
        self.build(active_msg_daily_max=6, care_daily_max=0)
        self.care(_message("mid"))
        self.assertEqual(self.llm.calls, [])

    def test_same_message_is_only_considered_once(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "休息一下吧"}]
        message = _message()
        self.care(message)
        self.care(message)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(len(self.api.sent), 1)


class RateLimitTests(CareBase):
    def test_rate_limited_care_mutes_the_group_for_today(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "早点睡"}]
        self.api.error = RuntimeError("post_group_message failed: retcode=40034102")
        self.care(_message())
        self.assertEqual(
            self.client._proactive_blocked_until.get(GROUP), "2026-10-14"
        )
        # 换人、换消息也不再尝试
        self.care(_message("肚子疼", mid="m-9", member="openid-X"))
        self.assertEqual(len(self.llm.calls), 1)

    def test_failed_send_does_not_consume_the_quota(self) -> None:
        self.llm.verdicts = [{"care": True, "text": "早点睡"}]
        self.api.error = RuntimeError("retcode=40034102")
        self.care(_message())
        self.assertEqual(self.store.sent_care_count(
            group_openid=GROUP, member_openid="openid-CL", since_iso=TODAY_START
        ), 0)

    def test_rate_limit_on_task_send_mutes_care_too(self) -> None:
        self.api.error = RuntimeError("qq rate limit 40034100")
        self.store.create_task(
            group_openid=GROUP,
            title="已经很晚了的任务",
            content="该睡了",
            generation_prompt=None,
            schedule_type="once",
            next_run_at="2026-10-14T14:00:00+08:00",
            timezone="Asia/Shanghai",
            time_of_day=None,
            weekday=None,
            created_by_openid="openid-CL",
            created_by_name="CL",
        )
        self.loop.run_until_complete(self.client._send_due_tasks())
        self.assertEqual(
            self.client._proactive_blocked_until.get(GROUP), "2026-10-14"
        )
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

    def test_other_errors_do_not_mute_the_group(self) -> None:
        self.api.error = RuntimeError("network timeout")
        self.care(_message())
        self.assertEqual(self.client._proactive_blocked_until, {})


class GroupPreferenceTests(CareBase):
    def test_show_lists_every_tunable_item(self) -> None:
        reply = self.client._handle_bot_pref({"action": "bot_pref", "op": "show"}, _message())
        self.assertIn("这个群现在的设置", reply)
        for label in ("主动关心", "每人每天被关心次数", "同一个人两次关心的间隔（分钟）", "群里每天主动消息上限"):
            self.assertIn(label, reply)
        self.assertIn("默认", reply)

    def test_set_number_preference(self) -> None:
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_daily_max", "value": 3},
            _message(),
        )
        self.assertIn("每人每天被关心次数改为3", reply)
        self.assertEqual(self.store.get_preference(GROUP, "care_daily_max"), "3")
        self.assertEqual(self.client._pref_int(GROUP, "care_daily_max", 2), 3)

    def test_set_boolean_from_model_value(self) -> None:
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_enabled", "value": False},
            _message(),
        )
        self.assertIn("不主动插话", reply)
        self.assertEqual(self.store.get_preference(GROUP, "care_enabled"), "false")
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

    def test_zero_is_a_real_value_not_a_missing_one(self) -> None:
        # 0 表示“不再主动发”，曾经被当成没填，务必钉住。
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_daily_max", "value": 0}, _message()
        )
        self.assertIn("改为0", reply)
        self.assertEqual(self.store.get_preference(GROUP, "care_daily_max"), "0")
        self.care(_message())
        self.assertEqual(self.llm.calls, [])

    def test_unknown_key_is_rejected(self) -> None:
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "model_name", "value": "gpt"},
            _message(),
        )
        self.assertIn("只能改这几项", reply)
        self.assertIsNone(self.store.get_preference(GROUP, "model_name"))

    def test_out_of_range_number_is_rejected(self) -> None:
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "active_msg_daily_max", "value": 999},
            _message(),
        )
        self.assertIn("0 到 50", reply)
        self.assertIsNone(self.store.get_preference(GROUP, "active_msg_daily_max"))

    def test_vague_boolean_is_rejected(self) -> None:
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_enabled", "value": "也许"},
            _message(),
        )
        self.assertIn("要说开还是关", reply)

    def test_reset_to_default(self) -> None:
        self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_daily_max", "value": 5}, _message()
        )
        reply = self.client._handle_bot_pref(
            {"action": "bot_pref", "op": "set", "key": "care_daily_max", "value": "恢复默认"},
            _message(),
        )
        self.assertIn("恢复成默认值 2", reply)
        self.assertIsNone(self.store.get_preference(GROUP, "care_daily_max"))

    def test_intent_dispatches_bot_pref(self) -> None:
        reply = self.loop.run_until_complete(
            self.client._handle_intent(
                intent={"action": "bot_pref", "op": "show"},
                message=_message(),
                requester=SimpleNamespace(username="CL", member_openid="openid-CL"),
                request="看看群设置",
                topic="",
                background="",
                full_recent="",
                identity=SimpleNamespace(prompt=lambda: ""),
            )
        )
        self.assertIn("这个群现在的设置", reply)


class JudgeCareModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def _llm(self, **overrides) -> XiaoxiaLLM:
        settings = SimpleNamespace(
            llm_api_key="k",
            llm_base_url="https://example.invalid",
            model="big-model",
            care_model="",
            vision_model="cheap-model",
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return XiaoxiaLLM(settings)

    def test_falls_back_to_the_cheap_vision_model(self) -> None:
        captured: dict = {}

        async def fake_chat(**kwargs):
            captured.update(kwargs)
            return '{"care": true, "text": "别熬了，早点睡"}'

        llm = self._llm()
        llm._chat = fake_chat
        verdict = self.loop.run_until_complete(llm.judge_care(content="睡不着"))
        self.assertTrue(verdict["care"])
        self.assertEqual(verdict["text"], "别熬了，早点睡")
        self.assertEqual(captured["model"], "cheap-model")
        self.assertTrue(captured["json_mode"])

    def test_explicit_care_model_wins(self) -> None:
        captured: dict = {}

        async def fake_chat(**kwargs):
            captured.update(kwargs)
            return '{"care": false}'

        llm = self._llm(care_model="ultra-cheap")
        llm._chat = fake_chat
        self.loop.run_until_complete(llm.judge_care(content="累了"))
        self.assertEqual(captured["model"], "ultra-cheap")

    def test_garbage_answer_is_treated_as_no_care(self) -> None:
        async def fake_chat(**kwargs):
            return "我不知道该不该说"

        llm = self._llm()
        llm._chat = fake_chat
        verdict = self.loop.run_until_complete(llm.judge_care(content="累了"))
        self.assertEqual(verdict, {"care": False, "text": ""})

    def test_api_error_is_treated_as_no_care(self) -> None:
        async def fake_chat(**kwargs):
            raise RuntimeError("500")

        llm = self._llm()
        llm._chat = fake_chat
        verdict = self.loop.run_until_complete(llm.judge_care(content="累了"))
        self.assertFalse(verdict["care"])


class NormalizeTextTests(unittest.TestCase):
    def test_strips_wrapping_quotes_and_newlines(self) -> None:
        self.assertEqual(normalize_care_text("\n“抱抱你，我在。”\n"), "抱抱你，我在。")

    def test_drops_label_prefixes(self) -> None:
        self.assertEqual(normalize_care_text("text: 记得多喝水"), "记得多喝水")

    def test_empty_stays_empty(self) -> None:
        self.assertEqual(normalize_care_text("   "), "")
        self.assertEqual(normalize_care_text(""), "")

    def test_hard_length_cap(self) -> None:
        self.assertEqual(len(normalize_care_text("啊" * 60)), 25)


if __name__ == "__main__":
    unittest.main()
