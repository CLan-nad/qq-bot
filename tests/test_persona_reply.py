"""本地操作的确认话也要有小毛豆的人设，但事实必须锁死。

以前 memory_remember / memory_pin / address_set 这些直接回模板字符串，
SYSTEM_PROMPT 里那套人设一个 token 都没参与——这就是「说话死板」的确切来源。
现在模板当事实凭据交给模型润色，同时守住两条底线：
失败/歧义路径绝不润色（那些话必须精确可预测），事实丢了就退回模板。
"""
from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from maodou.bot import (
    PERSONA_TOOL_LABELS,
    ActionResult,
    IdentityContext,
    MaodouClient,
    _contradicts_success,
)
from maodou.llm import ACTION_REPLY_ADDENDUM
from maodou.store import ChatStore, StoredMessage


GROUP = "test-group"
NOW = datetime(2026, 10, 15, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def _task_settings() -> SimpleNamespace:
    return SimpleNamespace(
        timezone="Asia/Shanghai", max_context_chars=24000, task_max_count=20
    )


def _task_message() -> SimpleNamespace:
    # 建任务要写 created_by，比记忆那几个 handler 多要一个 author
    return SimpleNamespace(
        group_openid=GROUP,
        id="msg-1",
        username="CL",
        author=SimpleNamespace(member_openid="openid-CL"),
    )


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

    def _client(self, llm) -> MaodouClient:
        return MaodouClient(settings=_settings(), store=self.store, llm=llm)

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

    def test_reply_that_contradicts_the_result_falls_back(self) -> None:
        # 真实模型跑出来的翻车样本：模板是「已忘记记忆 2：不吃香菜」，
        # 模型回「第一条记忆没删成，实际删掉的是第 2 条」。
        # 「删掉」「2」都在，子串校验照样放行，只有语义闸拦得住。
        llm = FakeLLM(reply="第一条记忆没删成，实际删掉的是第 2 条。")
        template = "已忘记记忆 2：不吃香菜"
        self.assertEqual(
            self._run(
                "memory_forget",
                ActionResult(template, persona=True, facts=("忘记", "2")),
                llm,
            ),
            template,
        )

    def test_reply_that_asks_a_followup_falls_back(self) -> None:
        # 事已经办完了，这时候反问「是不是这条」只会让人以为没办成
        llm = FakeLLM(reply="记好啦，要不要顺便把不吃香菜也删掉？")
        template = "已忘记记忆 2：不吃香菜"
        self.assertEqual(
            self._run(
                "memory_forget",
                ActionResult(template, persona=True, facts=("忘记", "2")),
                llm,
            ),
            template,
        )

    def test_positive_wording_containing_mei_is_not_killed(self) -> None:
        # 「一条都没漏」是好话，不能因为有个「没」就退回模板
        self.assertFalse(_contradicts_success("好嘞，两条都记下啦，一条都没漏～"))
        llm = FakeLLM(reply="好嘞，两条都记下啦，一条都没漏～")
        self.assertEqual(
            self._run(
                "memory_remember", ActionResult("记住了，共 2 条。", persona=True, facts=("2",)), llm
            ),
            "好嘞，两条都记下啦，一条都没漏～",
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
        # MaodouClient.__init__ 里要建 asyncio.Lock，没有当前 loop 会直接 RuntimeError
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "flags.sqlite3"))
        self.identity = IdentityContext(
            requester=_member("CL"), partner=_member("二十四桥明月夜")
        )
        self.client = MaodouClient(
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
        client = MaodouClient(settings=_settings(), store=self.store, llm=llm)
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


class TaskPersonaTests(unittest.TestCase):
    """定时任务和删记忆的确认话，以前最像后台日志：「已设置任务 3：每天 09:00，喝水。」"""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "tasks.sqlite3"))
        self.identity = IdentityContext(
            requester=_member("CL"), partner=_member("二十四桥明月夜")
        )
        self.client = MaodouClient(
            settings=_task_settings(), store=self.store, llm=SimpleNamespace()
        )
        self.client._now = lambda: NOW

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client_with(self, llm) -> MaodouClient:
        client = MaodouClient(settings=_task_settings(), store=self.store, llm=llm)
        client._now = lambda: NOW
        return client

    def _create(self, **extra):
        intent = {
            "action": "task_create",
            "schedule_type": "daily",
            "time": "09:00",
            "content": "喝水",
            "title": "喝水",
        }
        intent.update(extra)
        return self.client._handle_task_create(intent, _task_message())

    def _act(self, action: str, task_id: int):
        return self.client._handle_task_action(
            {"action": action, "task_id": task_id}, _task_message()
        )

    def test_create_success_locks_id_time_and_title(self) -> None:
        result = self._create()
        self.assertTrue(result.persona)
        task = self.store.tasks(GROUP)[0]
        self.assertIn(str(task.id), result.facts)
        self.assertIn(self.client._schedule_label(task), result.facts)
        self.assertIn("喝水", result.facts)

    def test_warmup_task_is_locked_too(self) -> None:
        # 预热任务也是一条真任务，编号和时间同样不许被模型说糊
        result = self._create(
            schedule_type="yearly", month=12, day=13, remind_days_ahead=3
        )
        tasks = {task.title: task for task in self.store.tasks(GROUP)}
        self.assertIn("[预热] 喝水", tasks)
        for task in tasks.values():
            self.assertIn(str(task.id), result.facts)
            self.assertIn(self.client._schedule_label(task), result.facts)
        # 模板本身两行，预算必须跟着放开，否则必然被截断→必然退回模板→白花一次调用
        self.assertGreaterEqual(len(str(result)), 60)

    def test_cancel_pause_resume_skip_carry_their_direction(self) -> None:
        self._create()
        task = self.store.tasks(GROUP)[0]
        for action, direction in (
            ("task_pause", "暂停"),
            ("task_resume", "恢复"),
            ("task_skip", "跳过"),
            ("task_cancel", "取消"),
        ):
            result = self._act(action, task.id)
            self.assertTrue(result.persona, action)
            self.assertIn(direction, result.facts)
            self.assertIn(str(task.id), result.facts)
        self.assertEqual(self.store.tasks(GROUP), [])

    def test_task_list_stays_a_plain_table(self) -> None:
        self._create()
        result = self._act("task_list", 0)
        self.assertFalse(getattr(result, "persona", False))

    def test_task_failures_stay_plain(self) -> None:
        bad_create = self.client._handle_task_create(
            {"action": "task_create", "schedule_type": "monthly", "content": "x"},
            _task_message(),
        )
        self.assertFalse(getattr(bad_create, "persona", False))
        empty_create = self.client._handle_task_create(
            {"action": "task_create", "schedule_type": "daily"}, _task_message()
        )
        self.assertFalse(getattr(empty_create, "persona", False))
        missing = self._act("task_cancel", 99999)
        self.assertFalse(getattr(missing, "persona", False))
        self.assertIn("没有找到这个任务", missing)

    def test_forget_carries_the_direction(self) -> None:
        self.client._handle_memory_remember(
            {"memories": [{"kind": "fact", "person": "CL", "content": "喜欢吃辣"}]},
            _task_message(),
            self.identity,
        )
        card = self.store.memory_cards(GROUP)[0]
        result = self.client._handle_memory_forget({"memory_id": [card.id]}, _task_message())
        self.assertTrue(result.persona)
        self.assertIn("忘记", result.facts)
        self.assertIn(str(card.id), result.facts)

    def test_flipped_task_direction_falls_back(self) -> None:
        # 把「取消」说成「恢复」，用户会以为提醒还在跑——比死板严重得多
        llm = FakeLLM(reply="好啦，3 号已经恢复运行了～")
        template = "已取消任务 3：喝水。"
        reply = self.loop.run_until_complete(
            self._client_with(llm)._persona_or_raw(
                "task_cancel",
                ActionResult(template, persona=True, facts=("取消", "3", "喝水")),
                "把喝水的提醒取消",
                _task_message(),
                self.identity,
                "话题",
            )
        )
        self.assertEqual(reply, template)

    def test_direction_synonym_is_accepted(self) -> None:
        # 「撤掉啦」是最自然的中文，方向也没说反，不该被判成丢事实
        llm = FakeLLM(reply="好嘞，3 号喝水的提醒给你撤掉啦～")
        reply = self.loop.run_until_complete(
            self._client_with(llm)._persona_or_raw(
                "task_cancel",
                ActionResult("已取消任务 3：喝水。", persona=True, facts=("取消", "3", "喝水")),
                "把喝水的提醒取消",
                _task_message(),
                self.identity,
                "话题",
            )
        )
        self.assertEqual(reply, "好嘞，3 号喝水的提醒给你撤掉啦～")

    def test_create_goes_through_the_persona_layer(self) -> None:
        llm = FakeLLM(reply="好嘞，1 号建好了：每天 09:00 叫你喝水～")
        reply = self.loop.run_until_complete(
            self._client_with(llm)._handle_intent(
                intent={
                    "action": "task_create",
                    "schedule_type": "daily",
                    "time": "09:00",
                    "content": "喝水",
                    "title": "喝水",
                },
                message=_task_message(),
                requester=_member("CL"),
                request="每天九点提醒我喝水",
                topic="CL：每天九点提醒我喝水",
                background="",
                full_recent="",
                identity=self.identity,
            )
        )
        self.assertEqual(reply, "好嘞，1 号建好了：每天 09:00 叫你喝水～")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tool_name"], "创建定时提醒")
        self.assertEqual(self.store.tasks(GROUP)[0].title, "喝水")


class AddendumTests(unittest.TestCase):
    def test_it_locks_facts_and_forbids_extra_promises(self) -> None:
        self.assertIn("事实一个字都不许改", ACTION_REPLY_ADDENDUM)
        self.assertIn("置顶 / 取消置顶", ACTION_REPLY_ADDENDUM)
        self.assertIn("取消 / 暂停 / 恢复 / 跳过", ACTION_REPLY_ADDENDUM)
        self.assertIn("任务的时间和标题", ACTION_REPLY_ADDENDUM)
        self.assertIn("不许添加", ACTION_REPLY_ADDENDUM)


def _pref_settings() -> SimpleNamespace:
    """bot_pref 要读这四个默认值，少给一个字段异常会被 _persona_or_raw 吞掉。"""
    return SimpleNamespace(
        timezone="Asia/Shanghai",
        max_context_chars=24000,
        care_enabled=True,
        care_daily_max=2,
        care_cooldown_minutes=180,
        active_msg_daily_max=6,
    )


class AliasAndPrefPersonaTests(unittest.TestCase):
    """memory_alias / bot_pref 是最后两个「办成了事却直接回裸模板」的本地动作。

    统一叫法、改群设置都是用户明确下的一条指令，办成之后回一句
    「记下了，这个群的主动关心改为关。」——和已经修好的那 10 个动作比，
    这两个明显还是后台日志的口气。规则不变：只润色成功路径，事实锁死。
    """

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "alias-pref.sqlite3"))
        self.identity = IdentityContext(requester=_member("CL"), partner=_member("小虾"))
        self.client = MaodouClient(
            settings=_pref_settings(), store=self.store, llm=SimpleNamespace()
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_alias_set_locks_the_canonical_name(self) -> None:
        result = self.client._handle_memory_alias(
            {"op": "set", "alias": ["罗晓霞"], "canonical": "小虾"}, _message()
        )
        self.assertTrue(result.persona)
        self.assertIn("小虾", result.facts)

    def test_alias_set_keeps_the_rewritten_count_as_a_fact(self) -> None:
        self.client._handle_memory_remember(
            {"memories": [{"kind": "fact", "person": "罗晓霞", "content": "不吃香菜"}]},
            _message(),
            self.identity,
        )
        result = self.client._handle_memory_alias(
            {"op": "set", "alias": ["罗晓霞"], "canonical": "小虾"}, _message()
        )
        self.assertTrue(result.persona)
        # 顺手改掉了几条旧记忆，是用户唯一能核对的数字，不能被润色掉
        self.assertIn("1", result.facts)
        self.assertIn("小虾", result.facts)

    def test_alias_clear_is_wrapped_too(self) -> None:
        self.client._handle_memory_alias(
            {"op": "set", "alias": ["罗晓霞"], "canonical": "小虾"}, _message()
        )
        result = self.client._handle_memory_alias({"op": "clear", "alias": ["罗晓霞"]}, _message())
        self.assertTrue(result.persona)
        self.assertIn("罗晓霞", result.facts)

    def test_alias_show_and_failures_stay_plain(self) -> None:
        for intent in (
            {"op": "show"},
            {"op": "set", "canonical": "小虾"},
            {"op": "set", "alias": ["小虾"], "canonical": "小虾"},
            {"op": "clear", "alias": ["没人"]},
        ):
            result = self.client._handle_memory_alias(intent, _message())
            self.assertFalse(getattr(result, "persona", False), intent)

    def test_pref_set_locks_label_and_value(self) -> None:
        result = self.client._handle_bot_pref(
            {"op": "set", "key": "care_daily_max", "value": 3}, _message()
        )
        self.assertTrue(result.persona)
        self.assertIn("每人每天被关心次数", result.facts)
        self.assertIn("3", result.facts)

    def test_turning_care_off_keeps_the_recovery_hint(self) -> None:
        result = self.client._handle_bot_pref(
            {"op": "set", "key": "care_enabled", "value": False}, _message()
        )
        self.assertTrue(result.persona)
        self.assertIn("要主动关心", result.facts)

    def test_restoring_the_default_locks_the_default_value(self) -> None:
        self.client._handle_bot_pref(
            {"op": "set", "key": "care_daily_max", "value": 5}, _message()
        )
        result = self.client._handle_bot_pref(
            {"op": "set", "key": "care_daily_max", "value": "默认"}, _message()
        )
        self.assertTrue(result.persona)
        self.assertIn("2", result.facts)

    def test_pref_show_and_bad_values_stay_plain(self) -> None:
        for intent in (
            {"op": "show"},
            {"op": "set", "key": "care_daily_max", "value": "很多"},
            {"op": "set", "key": "care_daily_max", "value": 999},
            {"op": "set", "key": "不存在的项", "value": 1},
            {"op": "set", "key": "care_enabled", "value": "随便"},
        ):
            result = self.client._handle_bot_pref(intent, _message())
            self.assertFalse(getattr(result, "persona", False), intent)

    def test_both_actions_are_routed_through_the_persona_layer(self) -> None:
        source = inspect.getsource(MaodouClient._handle_intent)
        for action in ("memory_alias", "bot_pref"):
            self.assertIn(f"speak(self._handle_{action}", source)
            self.assertIn(action, PERSONA_TOOL_LABELS)


class AliasPrefEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "alias-e2e.sqlite3"))
        self.identity = IdentityContext(requester=_member("CL"), partner=_member("小虾"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _intent(self, llm, intent: dict, request: str) -> str:
        client = MaodouClient(settings=_pref_settings(), store=self.store, llm=llm)
        return self.loop.run_until_complete(
            client._handle_intent(
                intent=intent,
                message=_message(),
                requester=_member("CL"),
                request=request,
                topic=f"CL：{request}",
                background="",
                full_recent="",
                identity=self.identity,
            )
        )

    def test_alias_reply_gets_its_own_tool_label(self) -> None:
        llm = FakeLLM(reply="好嘞，以后都叫她小虾～")
        reply = self._intent(
            llm,
            {"action": "memory_alias", "op": "set", "alias": ["罗晓霞"], "canonical": "小虾"},
            "以后把「罗晓霞」都叫「小虾」",
        )
        # calls 必须真有 1 条：_persona_or_raw 会吞异常静默回退，只看回复分不出来
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tool_name"], "统一某个人的叫法")
        self.assertEqual(reply, "好嘞，以后都叫她小虾～")

    def test_pref_reply_reaches_the_persona_layer(self) -> None:
        llm = FakeLLM(reply="行，那我就不自己插话了，你们 @ 我随时在～想恢复就说「要主动关心」。")
        reply = self._intent(
            llm,
            {"action": "bot_pref", "op": "set", "key": "care_enabled", "value": False},
            "以后别主动安慰我们",
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tool_name"], "改这个群的设置")
        self.assertEqual(reply, llm.reply)

    def test_dropping_a_fact_falls_back_to_the_template(self) -> None:
        llm = FakeLLM(reply="好啦，都听你的～")
        reply = self._intent(
            llm,
            {"action": "bot_pref", "op": "set", "key": "care_daily_max", "value": 3},
            "每人每天关心次数改成三次",
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(reply, "记下了，这个群的每人每天被关心次数改为3。")


class FactToleranceTests(unittest.TestCase):
    """事实闸要认得出「说短了但没说错」，同时照样拦住说错的。

    真实模型实测：模板「记下了，这个群的每人每天被关心次数改为3。」，
    模型回「好嘞，每人每天关心次数已经改成 3 次了。」——又对又像人话，
    却因为那个全称名字没被一字不差抄下来而被判丢事实，白花一次调用退回模板。
    """

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "tolerance.sqlite3"))
        self.identity = IdentityContext(requester=_member("CL"), partner=_member("小虾"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _polish(self, template: str, facts: tuple[str, ...], model_says: str) -> str:
        llm = FakeLLM(reply=model_says)
        client = MaodouClient(settings=_pref_settings(), store=self.store, llm=llm)
        return self.loop.run_until_complete(
            client._persona_or_raw(
                "bot_pref",
                ActionResult(template, persona=True, facts=facts),
                "改一下群设置",
                _message(),
                self.identity,
                "话题",
            )
        )

    def test_shortened_pref_label_still_counts(self) -> None:
        says = "好嘞，每人每天关心次数已经改成 3 次了。"
        self.assertEqual(
            self._polish(
                "记下了，这个群的每人每天被关心次数改为3。", ("每人每天被关心次数", "3"), says
            ),
            says,
        )

    def test_spaces_do_not_break_a_number_fact(self) -> None:
        # 模型爱在数字和单位之间加空格，「30 分钟」和「30分钟」必须是同一个事实
        says = "好嘞，关心间隔改成 30 分钟了～"
        self.assertEqual(
            self._polish(
                "记下了，这个群的同一个人两次关心的间隔（分钟）改为30。",
                ("同一个人两次关心的间隔（分钟）", "30"),
                says,
            ),
            says,
        )

    def test_a_wrong_number_is_still_rejected(self) -> None:
        # 放宽只针对说法，不针对数值：改成 5 就是错的，必须退回模板
        template = "记下了，这个群的每人每天被关心次数改为3。"
        self.assertEqual(
            self._polish(template, ("每人每天被关心次数", "3"), "好嘞，关心次数改成 5 次了。"),
            template,
        )

    def test_talking_about_a_different_setting_is_still_rejected(self) -> None:
        # 别名表是逐个事实挂的，不会让「说了别的设置」蒙混过关
        template = "记下了，这个群的每人每天被关心次数改为3。"
        self.assertEqual(
            self._polish(template, ("每人每天被关心次数", "3"), "好嘞，主动消息上限改成 3 条了。"),
            template,
        )

    def test_the_direction_gate_still_wins_over_tolerance(self) -> None:
        template = "已取消任务 3：喝水。"
        self.assertEqual(
            self._polish(template, ("取消", "3", "喝水"), "任务 3 喝水我给它恢复了。"),
            template,
        )


if __name__ == "__main__":
    unittest.main()
