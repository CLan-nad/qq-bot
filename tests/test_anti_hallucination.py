"""批次⑧：反幻觉硬规则。

昨天线上最难受的三段回复，都不是「答得不好」，是「说了没发生的事」：

- 18:01:23 用户连说三遍“把个人信息设为永久”，意图始终没落到 memory_pin（走的是 chat），
  模型却回“收到收到，我这就把…都标成永久★”“以后这些个人信息默认永久保存”；
  18:03:56 查看记忆，一个星号都没有。
- 18:05:45 / 18:09:16 编内部原因 + 甩锅：“我这边没匹配上”“那就是后台的锅，喊我来背也没用”；
  13:54:16 还说搜索结果“被其他信息覆盖了”。
- 18:19:55 被问“你联网了吗”，答“没联网，我这边是纯本地模式”，
  18:22:25 下一轮又改口“联网了呀”。

规则写在 SYSTEM_PROMPT 第 7 条（原来是 15/16/17 三条，压缩后合并），光靠自觉不够：模型看不到自己这轮干了什么，
所以 generate_reply 每轮都把「本轮实际执行的操作」写进 user_prompt，没执行就明写“（无）”。
"""
from __future__ import annotations

import asyncio
import inspect
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import xiaoxia.bot as bot_module
from xiaoxia.bot import XiaoxiaClient
from xiaoxia.llm import NO_ACTION_NOTE, SYSTEM_PROMPT, TOOL_REPLY_ADDENDUM, XiaoxiaLLM
from xiaoxia.store import ChatStore


GROUP = "test-group"
RULE_PATTERN = re.compile(r"^(\d+)\. ", re.M)


def _rule_numbers(text: str) -> list[int]:
    return [int(match) for match in RULE_PATTERN.findall(text)]


def _llm() -> XiaoxiaLLM:
    settings = SimpleNamespace(
        llm_api_key="k",
        llm_base_url="https://example.invalid",
        model="big-model",
        care_model="",
        vision_model="cheap-model",
        reply_max_chars=900,
    )
    return XiaoxiaLLM(settings)


class SystemPromptRuleTests(unittest.TestCase):
    def test_claiming_success_requires_evidence(self) -> None:
        self.assertIn("说“已经办好”必须有凭据", SYSTEM_PROMPT)
        for phrase in ("而且结果里明确写了成功", "就直说这次没做成", "绝不许先答应下来再说"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_no_invented_internals_and_no_blame_shifting(self) -> None:
        self.assertIn("不要猜、也不要编自己的内部实现", SYSTEM_PROMPT)
        # 这三句都是线上原话，必须被点名禁掉
        for phrase in ("后台没同步", "指令没匹配上", "结果被别的信息覆盖了", "后台的锅"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_capabilities_are_not_made_up(self) -> None:
        self.assertIn("你的能力也以这一轮实际能用的功能为准", SYSTEM_PROMPT)
        for phrase in ("纯本地模式", "没有联网权限", "不要下一轮又改口", "本轮实际执行的操作"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_rule_numbers_are_contiguous(self) -> None:
        numbers = _rule_numbers(SYSTEM_PROMPT)
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))
        # 17 条压到 7 条：4 条「别翻旧账」并成规则 4，3 条反幻觉补丁并成规则 7。
        # 这条断言是防止 prompt 再往回长——规则越多，人设越被淹没。
        self.assertEqual(numbers[-1], 7)
        # 上限从 1400 放到 1700：规则 2 里补了「【】就是说话人」「纠正的是话题不是你自己」
        # 「〔图〕是你自己看图的记录」三件事——都是线上真实翻车（说话人错位、把 codex
        # 当成自己、认了图却说没识别）换来的，不是可选项。
        self.assertLessEqual(len(SYSTEM_PROMPT), 1700)

    def test_persona_still_outranks_the_bug_patches(self) -> None:
        # 压缩的目的：让「性格」在 prompt 里的分量压过历史补丁
        head = SYSTEM_PROMPT.index("你的性格")
        self.assertLess(head, SYSTEM_PROMPT.index("原则："))
        self.assertLess(head, SYSTEM_PROMPT.index("本轮实际执行的操作"))
        for phrase in ("可爱、调皮", "不像客服，也不像助理", "只输出将直接发送到 QQ 群的正文"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_tool_addendum_picks_up_where_the_main_rules_stop(self) -> None:
        main = _rule_numbers(SYSTEM_PROMPT)
        tool = _rule_numbers(TOOL_REPLY_ADDENDUM)
        self.assertEqual(tool[0], main[-1] + 1)
        self.assertEqual(tool, list(range(tool[0], tool[0] + len(tool))))
        # 两处都出现同一个编号，模型就分不清该听谁的
        self.assertEqual(set(main) & set(tool), set())


class ExecutedActionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def _capture(self, **kwargs) -> str:
        captured: dict = {}

        async def fake_chat(**chat_kwargs):
            captured.update(chat_kwargs)
            return "好呀"

        llm = _llm()
        llm._chat = fake_chat
        self.loop.run_until_complete(
            llm.generate_reply(
                requester=SimpleNamespace(username="CL", role="member"),
                request="把我们的个人信息设为永久",
                topic_context="CL：设为永久",
                **kwargs,
            )
        )
        return captured["user_prompt"]

    def test_generate_reply_takes_executed_actions(self) -> None:
        params = inspect.signature(XiaoxiaLLM.generate_reply).parameters
        self.assertIn("executed_actions", params)
        self.assertEqual(params["executed_actions"].default, "")

    def test_no_action_note_is_spelled_out_by_default(self) -> None:
        prompt = self._capture()
        self.assertIn("本轮实际执行的操作：", prompt)
        self.assertIn(NO_ACTION_NOTE, prompt)
        for phrase in ("没有调用任何功能", "不要声称已经记住", "已经设成永久"):
            self.assertIn(phrase, prompt)

    def test_executed_actions_replace_the_note(self) -> None:
        prompt = self._capture(executed_actions="在本地聊天记录里检索过")
        self.assertIn("在本地聊天记录里检索过", prompt)
        self.assertNotIn(NO_ACTION_NOTE, prompt)

    def test_note_comes_after_the_context_blocks(self) -> None:
        # 放在最后才压得住前面那堆记忆和上下文
        prompt = self._capture(memory_context="- [fact] 蔡乐: 用户叫蔡乐")
        self.assertLess(prompt.index("长期记忆"), prompt.index("本轮实际执行的操作"))


class ChatPathSaysNoActionTests(unittest.TestCase):
    """走 chat 的那一轮必须收到“（无）”，这正是 18:01:23 那次幻觉的现场。"""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))
        self.captured: dict = {}

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self) -> XiaoxiaClient:
        async def fake_chat(**kwargs):
            self.captured.update(kwargs)
            return "收到收到，我这就标★"

        llm = _llm()
        llm._chat = fake_chat
        return XiaoxiaClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                allowed_group_openids=frozenset(),
                memory_prompt_chars=2000,
                memory_prompt_per_kind=6,
                memory_prompt_total=40,
                memory_prompt_event_cards=5,
                memory_search_messages=50,
                max_context_chars=4000,
                face_tags=False,
            ),
            store=self.store,
            llm=llm,
        )

    def _run_intent(self, client: XiaoxiaClient, action: str, **extra) -> str:
        return self.loop.run_until_complete(
            client._handle_intent(
                intent={"action": action, **extra},
                message=SimpleNamespace(group_openid=GROUP, id="m-1", username="CL"),
                requester=SimpleNamespace(username="CL", role="member"),
                request="把我们的个人信息设为永久",
                topic="CL：把我们的个人信息设为永久",
                background="",
                full_recent="",
                identity=SimpleNamespace(prompt=lambda: "触发者：CL"),
            )
        )

    def test_chat_round_is_told_nothing_was_executed(self) -> None:
        reply = self._run_intent(self._client(), "chat")
        self.assertEqual(reply, "收到收到，我这就标★")
        self.assertIn(NO_ACTION_NOTE, self.captured["user_prompt"])

    def test_memory_search_round_reports_the_real_action(self) -> None:
        client = self._client()
        self.store.save_message(
            message_id="m-0",
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content="我们把纪念日定在12月13日",
            timestamp="2026-10-14T15:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )
        self._run_intent(client, "memory_search", query="纪念日")
        prompt = self.captured["user_prompt"]
        self.assertIn("本轮实际执行的操作：", prompt)
        self.assertIn("在本地聊天记录和长期记忆里检索过", prompt)
        self.assertNotIn(NO_ACTION_NOTE, prompt)

    def test_memory_search_without_hits_does_not_reach_the_model(self) -> None:
        # 没检索到东西时是固定文案，压根不该让模型开口
        client = self._client()
        reply = self._run_intent(client, "memory_search")
        self.assertIn("没找到", reply)
        self.assertEqual(self.captured, {})

    def test_chat_branch_does_not_pass_executed_actions(self) -> None:
        source = inspect.getsource(bot_module.XiaoxiaClient._handle_intent)
        self.assertIn("executed_actions", inspect.getsource(bot_module.XiaoxiaClient._handle_memory_search))
        tail = source[source.index("return await self.llm.generate_reply"):]
        self.assertNotIn("executed_actions", tail)


if __name__ == "__main__":
    unittest.main()
