"""本地操作的确认话也要有小毛豆的人设，但事实必须锁死。

以前 memory_remember / memory_pin / address_set 这些直接回模板字符串，
SYSTEM_PROMPT 里那套人设一个 token 都没参与——这就是「说话死板」的确切来源。
现在模板当事实凭据交给模型润色，同时守住两条底线：
失败/歧义路径绝不润色（那些话必须精确可预测），事实丢了就退回模板。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from xiaoxia.bot import ActionResult, IdentityContext, XiaoxiaClient
from xiaoxia.llm import ACTION_REPLY_ADDENDUM
from xiaoxia.store import ChatStore, StoredMessage


GROUP = "test-group"


def _member(username: str, role: str = "member") -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid=f"openid-{username}",
        role=role,
        content="",
        timestamp="2026-10-15T15:00:00+08:00",
        source="GROUP_MESSAGE_CREATE",
    )


def _message(username: str = "CL") -> SimpleNamespace:
    return SimpleNamespace(group_openid=GROUP, id="msg-1", username=username)


def _settings() -> SimpleNamespace:
    # max_context_chars 必须给：_memory_context 末尾要靠它截断。少给一个字段，
    # 异常会被 _persona_or_raw 的 try 吞掉、静默回退模板，测试就会「看起来通过其实错了」。
    return SimpleNamespace(timezone="Asia/Shanghai", max_context_chars=24000)


class FakeLLM:
    """按剧本返回或抛错，同时把收到的参数记下来供断言。"""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    async def generate_tool_reply(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


class ActionResultTests(unittest.TestCase):
    def test_it_still_behaves_like_the_plain_template_string(self) -> None:
        result = ActionResult("记住了，共 2 条。", persona=True, facts=("2",))
        # 继承 str 的全部意义就在这：老的字符串断言一点感觉都没有
        self.assertEqual(result, "记住了，共 2 条。")
        self.assertIn("共 2 条", result)
        self.assertIsInstance(result, str)
        self.assertTrue(result.persona)
        self.assertEqual(result.facts, ("2",))

    def test_defaults_to_no_persona_so_plain_returns_stay_plain(self) -> None:
        result = ActionResult("找到了")
        self.assertFalse(result.persona)
        self.assertEqual(result.facts, ())

    def test_blank_facts_are_dropped(self) -> None:
        result = ActionResult("好", persona=True, facts=("", "  ", "主人"))
        self.assertEqual(result.facts, ("主人",))


class PersonaOrRawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "persona.sqlite3"))
        self.identity = IdentityContext(
            requester=_member("CL"), partner=_member("二十四桥明月夜")
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, llm) -> XiaoxiaClient:
        return XiaoxiaClient(settings=_settings(), store=self.store, llm=llm)

    def _run(self, action: str, result: str, llm) -> str:
        return self.loop.run_until_complete(
            self._client(llm)._persona_or_raw(
                action, result, "以后叫我主人", _message(), self.identity, "话题"
            )
        )

    def test_persona_result_gets_rewritten_by_the_model(self) -> None:
        llm = FakeLLM(reply="好嘞主人，这个称呼我记下啦～")
        template = "好，以后就叫CL「主人」。"
        reply = self._run(
            "address_set",
            ActionResult(template, persona=True, facts=("主人",)),
            llm,
        )
        self.assertEqual(reply, "好嘞主人，这个称呼我记下啦～")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tool_name"], "登记称呼")
        self.assertEqual(llm.calls[0]["tool_output"], template)
        self.assertIs(llm.calls[0]["addendum"], ACTION_REPLY_ADDENDUM)
        self.assertIn("身份", llm.calls[0]["identity_context"])

    def test_plain_template_never_touches_the_model(self) -> None:
        # 失败/歧义路径必须精确可预测：模型一活泼，用户就不知道该报哪个编号了
        llm = FakeLLM(reply="随便说点什么")
        template = "匹配到多条记忆，请说编号：\n1. [fact] CL: 喜欢吃辣"
        self.assertEqual(self._run("memory_pin", template, llm), template)
        self.assertEqual(llm.calls, [])

    def test_chinese_numeral_counts_as_the_same_fact(self) -> None:
        # 模型说「这两条」而不是「这 2 条」，这是地道中文，不能判成丢事实
        llm = FakeLLM(reply="收到，这两条我都记牢啦～")
        reply = self._run(
            "memory_remember",
            ActionResult("记住了，共 2 条。", persona=True, facts=("2",)),
            llm,
        )
        self.assertEqual(reply, "收到，这两条我都记牢啦～")

    def test_a_different_number_is_still_rejected(self) -> None:
        # 放宽中文写法不等于放弃校验：说成三条就是错的
        llm = FakeLLM(reply="收到，这三条我都记牢啦～")
        template = "记住了，共 2 条。"
        self.assertEqual(
            self._run("memory_remember", ActionResult(template, persona=True, facts=("2",)), llm),
            template,
        )

    def test_dropped_fact_falls_back_to_the_template(self) -> None:
        llm = FakeLLM(reply="都记下啦～")
        template = "记住了，共 3 条。"
        self.assertEqual(
            self._run("memory_remember", ActionResult(template, persona=True, facts=("3",)), llm),
            template,
        )

    def test_flipped_pin_direction_falls_back(self) -> None:
        # 方向说反比死板严重得多：用户会以为标成永久了，其实没有
        llm = FakeLLM(reply="已经置顶啦，以后不会忘～")
        template = "已取消置顶记忆 7：生日是农历冬月十一"
        self.assertEqual(
            self._run(
                "memory_unpin",
                ActionResult(template, persona=True, facts=("取消置顶", "7")),
                llm,
            ),
            template,
        )

    def test_model_error_falls_back_to_the_template(self) -> None:
        llm = FakeLLM(error=RuntimeError("boom"))
        template = "记住了，共 2 条。"
        self.assertEqual(
            self._run("memory_remember", ActionResult(template, persona=True, facts=("2",)), llm),
            template,
        )

    def test_empty_reply_falls_back_to_the_template(self) -> None:
        llm = FakeLLM(reply="   ")
        template = "记住了，共 2 条。"
        self.assertEqual(
            self._run("memory_remember", ActionResult(template, persona=True, facts=("2",)), llm),
            template,
        )

    def test_reply_budget_stays_short(self) -> None:
        llm = FakeLLM(reply="好")
        self._run("address_set", ActionResult("好", persona=True), llm)
        self.assertLessEqual(llm.calls[0]["max_chars"], 60)


class HandlerPersonaFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        # XiaoxiaClient.__init__ 里要建 asyncio.Lock，没有当前 loop 会直接 RuntimeError
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "flags.sqlite3"))
        self.identity = IdentityContext(
            requester=_member("CL"), partner=_member("二十四桥明月夜")
        )
        self.client = XiaoxiaClient(
            settings=_settings(), store=self.store, llm=SimpleNamespace()
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _remember(self, *contents: str):
        return self.client._handle_memory_remember(
            {
                "memories": [
                    {"kind": "fact", "person": "CL", "content": text} for text in contents
                ]
            },
            _message(),
            self.identity,
        )

    def test_remember_success_is_persona_wrapped_with_the_count(self) -> None:
        result = self._remember("喜欢吃辣", "不吃香菜")
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.persona)
        self.assertIn("2", result.facts)

    def test_single_memory_does_not_force_the_number(self) -> None:
        # 只有一条时逼模型报「1 条」，最自然的说法反而会被判成丢事实
        result = self._remember("喜欢吃辣")
        self.assertTrue(result.persona)
        self.assertEqual(result.facts, ())

    def test_pinned_save_carries_the_pin_fact(self) -> None:
        result = self.client._handle_memory_remember(
            {
                "pin": True,
                "memories": [{"kind": "fact", "person": "CL", "content": "生日是农历冬月十一"}],
            },
            _message(),
            self.identity,
        )
        self.assertTrue(result.persona)
        self.assertIn("置顶", result.facts)

    def test_sensitive_refusal_stays_plain(self) -> None:
        result = self._remember("我的身份证号是 110101199001011234")
        self.assertFalse(getattr(result, "persona", False))

    def test_empty_remember_stays_plain(self) -> None:
        result = self.client._handle_memory_remember({"memories": []}, _message(), self.identity)
        self.assertFalse(getattr(result, "persona", False))

    def test_pin_and_unpin_carry_the_direction(self) -> None:
        self._remember("生日是农历冬月十一")
        card_id = self.store.memory_cards(GROUP)[0].id
        pinned = self.client._handle_memory_pin({"memory_id": [card_id]}, _message(), pinned=True)
        self.assertTrue(pinned.persona)
        self.assertIn("置顶", pinned.facts)
        self.assertIn(str(card_id), pinned.facts)

        unpinned = self.client._handle_memory_pin(
            {"memory_id": [card_id]}, _message(), pinned=False
        )
        self.assertTrue(unpinned.persona)
        self.assertIn("取消置顶", unpinned.facts)

    def test_pin_failure_paths_stay_plain(self) -> None:
        result = self.client._handle_memory_pin({"memory_id": [999999]}, _message(), pinned=True)
        self.assertFalse(getattr(result, "persona", False))
        self.assertIn("没有找到这条记忆", result)

    def test_address_set_success_locks_the_address_not_the_display_name(self) -> None:
        result = self.client._handle_address_set(
            {"op": "set", "person": "CL", "address": "主人"}, _message(), self.identity
        )
        self.assertTrue(result.persona)
        self.assertEqual(result.facts, ("主人",))

    def test_address_clear_success_is_wrapped_too(self) -> None:
        self.client._handle_address_set(
            {"op": "set", "person": "CL", "address": "主人"}, _message(), self.identity
        )
        result = self.client._handle_address_set(
            {"op": "clear", "person": "CL"}, _message(), self.identity
        )
        self.assertTrue(result.persona)
        self.assertIn("CL", result.facts)

    def test_address_show_and_failures_stay_plain(self) -> None:
        for intent in (
            {"op": "show"},
            {"op": "set", "person": "CL"},
            {"op": "clear", "person": "没人"},
        ):
            result = self.client._handle_address_set(intent, _message(), self.identity)
            self.assertFalse(getattr(result, "persona", False), intent)


class IntentEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "e2e.sqlite3"))
        self.identity = IdentityContext(
            requester=_member("CL"), partner=_member("二十四桥明月夜")
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _intent(self, llm, intent: dict, request: str) -> str:
        client = XiaoxiaClient(settings=_settings(), store=self.store, llm=llm)
        return self.loop.run_until_complete(
            client._handle_intent(
                intent=intent,
                message=_message(),
                requester=_member("CL"),
                request=request,
                topic="CL: 以后叫我主人",
                background="",
                full_recent="",
                identity=self.identity,
            )
        )

    def test_remember_reaches_the_persona_layer(self) -> None:
        llm = FakeLLM(reply="收到，这两条我都记牢啦～")
        reply = self._intent(
            llm,
            {
                "action": "memory_remember",
                "memories": [
                    {"kind": "preference", "person": "CL", "content": "喜欢吃辣"},
                    {"kind": "preference", "person": "CL", "content": "不吃香菜"},
                ],
            },
            "记住我喜欢吃辣，不吃香菜",
        )
        self.assertEqual(reply, "收到，这两条我都记牢啦～")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(self.store.count_live_memory_cards(GROUP), 2)

    def test_address_set_reaches_the_persona_layer(self) -> None:
        llm = FakeLLM(reply="好嘞主人～")
        reply = self._intent(
            llm,
            {"action": "address_set", "op": "set", "person": "CL", "address": "主人"},
            "以后叫我主人",
        )
        self.assertEqual(reply, "好嘞主人～")
        self.assertEqual(self.store.address_for(GROUP, "CL"), "主人")

    def test_a_failed_operation_is_not_dressed_up(self) -> None:
        # 没存进去就不许让模型说得好听，这是规则 15 的底线
        llm = FakeLLM(reply="好嘞主人～")
        reply = self._intent(
            llm, {"action": "address_set", "op": "set", "person": "CL"}, "以后叫我主人"
        )
        self.assertIn("想让我怎么称呼", reply)
        self.assertEqual(llm.calls, [])


class AddendumTests(unittest.TestCase):
    def test_it_locks_facts_and_forbids_extra_promises(self) -> None:
        self.assertIn("事实一个字都不许改", ACTION_REPLY_ADDENDUM)
        self.assertIn("置顶 / 取消置顶", ACTION_REPLY_ADDENDUM)
        self.assertIn("不许添加", ACTION_REPLY_ADDENDUM)


if __name__ == "__main__":
    unittest.main()
