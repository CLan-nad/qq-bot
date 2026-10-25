"""批次③：工具回复必须和普通回复走同一套人设。

以前 generate_tool_reply 用一段临时 system prompt，绕过了 SYSTEM_PROMPT，也没有记忆/身份/话题，
于是 web_search/web_fetch 的回复全是分点堆来源、不像小毛豆、还会把两个人叫混。这几条测试盯着：
工具回复要带人设、带身份/记忆/话题，并按 action 限长。
"""
from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from unittest import mock

from maodou.bot import TOOL_REPLY_MAX_CHARS, MaodouClient
from maodou.llm import SYSTEM_PROMPT, MaodouLLM
from maodou.store import ChatStore


GROUP = "test-group"


def _llm(**overrides) -> MaodouLLM:
    settings = SimpleNamespace(
        llm_api_key="k",
        llm_base_url="https://example.invalid",
        model="big-model",
        care_model="",
        vision_model="cheap-model",
        reply_max_chars=900,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return MaodouLLM(settings)


class GenerateToolReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_uses_persona_system_prompt(self) -> None:
        captured: dict = {}

        async def fake_chat(**kwargs):
            captured.update(kwargs)
            return "查到了～"

        llm = _llm()
        llm._chat = fake_chat
        self.loop.run_until_complete(
            llm.generate_tool_reply(
                request="帮我查查腹泻的原因",
                tool_name="web_search",
                tool_output="1. 腹泻常见原因……（来源：mayoclinic.org）",
                identity_context="触发者 CL；对象 小虾",
                memory_context="- [preference/两人] 回复别分点",
                topic_context="CL：肚子疼拉肚子",
            )
        )
        system_prompt = captured["system_prompt"]
        # 必须走 SYSTEM_PROMPT 的人设（小毛豆 + 那 7 条原则），再叠加工具专属规则
        self.assertTrue(system_prompt.startswith(SYSTEM_PROMPT))
        self.assertIn("小毛豆", system_prompt)
        self.assertIn("来源站点名", system_prompt)
        self.assertIn("不许拿无关内容凑答案", system_prompt)
        # 身份/记忆/话题都要进 user_prompt，否则称呼会乱、「别分点」这类规则不生效
        user_prompt = captured["user_prompt"]
        self.assertIn("触发者 CL", user_prompt)
        self.assertIn("回复别分点", user_prompt)
        self.assertIn("肚子疼拉肚子", user_prompt)
        self.assertIn("腹泻常见原因", user_prompt)

    def test_max_chars_truncates(self) -> None:
        async def fake_chat(**kwargs):
            return "这是一个很长很长很长很长的回复"

        llm = _llm(reply_max_chars=900)
        llm._chat = fake_chat
        out = self.loop.run_until_complete(
            llm.generate_tool_reply(
                request="1+1", tool_name="math_eval", tool_output="2", max_chars=6
            )
        )
        self.assertEqual(len(out), 6)

    def test_default_cap_is_reply_max_chars(self) -> None:
        async def fake_chat(**kwargs):
            return "短回复"

        llm = _llm(reply_max_chars=3)
        llm._chat = fake_chat
        out = self.loop.run_until_complete(
            llm.generate_tool_reply(request="q", tool_name="web_fetch", tool_output="x")
        )
        self.assertEqual(len(out), 3)

    def test_signature_has_persona_and_cap(self) -> None:
        params = inspect.signature(MaodouLLM.generate_tool_reply).parameters
        for name in ("identity_context", "memory_context", "topic_context", "max_chars"):
            self.assertIn(name, params)


class ToolHandlerPersonaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "tool.sqlite3"))
        self.captured: dict = {}

        async def fake_tool_reply(**kwargs):
            self.captured.update(kwargs)
            return "算好了"

        self.client = MaodouClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                task_max_count=20,
                max_context_chars=4000,
                reply_max_chars=500,
                face_tags=False,
                llm_api_key="k",
                llm_base_url="https://example.invalid",
                model="test-model",
                reasoning_effort="low",
                search_max_results=5,
                search_timeout_seconds=12.0,
                web_fetch_max_chars=8000,
                document_max_chars=24000,
                shell_enabled=False,
                shell_cwd="/tmp",
                shell_mode="readonly",
                shell_timeout_seconds=10.0,
                shell_max_output_chars=50000,
                memory_search_messages=10,
            ),
            store=self.store,
            llm=SimpleNamespace(generate_tool_reply=fake_tool_reply, client=None),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_math_eval_forwards_persona_and_cap(self) -> None:
        out = self.loop.run_until_complete(
            self.client._handle_math_eval(
                "1+1 等于几",
                {"action": "math_eval", "expression": "1+1"},
                identity_context="触发者 CL",
                memory_context="- [fact/CL] CL 爱打游戏",
                topic_context="CL：1+1 等于几",
            )
        )
        self.assertEqual(out, "算好了")
        self.assertEqual(self.captured["tool_name"], "math_eval")
        self.assertEqual(self.captured["identity_context"], "触发者 CL")
        self.assertEqual(self.captured["memory_context"], "- [fact/CL] CL 爱打游戏")
        self.assertEqual(self.captured["topic_context"], "CL：1+1 等于几")
        self.assertEqual(self.captured["max_chars"], TOOL_REPLY_MAX_CHARS["math_eval"])

    def test_dispatch_builds_persona(self) -> None:
        # 走 _handle_intent 分发 agent_run 时，persona 要自动构建（身份 + 记忆 + 话题）
        # 并透传给 generate_tool_reply。
        import maodou.bot as bot_module

        async def fake_agent_run(**kwargs):
            return {"text": "2*3=6", "tool_calls_made": ["math_eval"]}

        self.store.add_memory_card(
            group_openid=GROUP, kind="preference", person="两人", content="回复别分点"
        )
        with mock.patch.object(bot_module, "agent_run_loop", fake_agent_run):
            self.loop.run_until_complete(
                self.client._handle_intent(
                    intent={"action": "agent_run", "query": "2*3"},
                    message=SimpleNamespace(group_openid=GROUP, id="m-1"),
                    requester=SimpleNamespace(username="CL", member_openid="openid-CL"),
                    request="2*3",
                    topic="CL：2*3 等于几",
                    background="",
                    full_recent="",
                    identity=SimpleNamespace(
                        prompt=lambda: "触发者 CL；对象 小虾",
                        speaker_labels=None,
                    ),
                )
            )
        self.assertEqual(self.captured["tool_name"], "math_eval")
        self.assertEqual(self.captured["identity_context"], "触发者 CL；对象 小虾")
        self.assertIn("回复别分点", self.captured["memory_context"])
        self.assertEqual(self.captured["topic_context"], "CL：2*3 等于几")


if __name__ == "__main__":
    unittest.main()
