"""agent_run 的工具预算和最终收口协议。"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import maodou.agent as agent_module
from maodou.agent import AgentContext, agent_run_loop


def _tool_call(index: int):
    function = SimpleNamespace(
        name="math_eval",
        arguments='{"expression":"1+1"}',
    )
    return SimpleNamespace(id=f"call-{index}", function=function)


class _QueuedCompletions:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_message(count: int, start: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[_tool_call(start + index) for index in range(count)],
    )


def _final_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=[])


class AgentLoopBudgetTests(unittest.TestCase):
    def _run(self, mode: str, responses: list[SimpleNamespace]):
        completions = _QueuedCompletions(responses)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        ctx = AgentContext(
            model="deepseek-flash",
            reasoning_effort="none" if mode == "fast" else "high",
            max_output_chars=3000,
        )
        executed: list[str] = []

        async def fake_execute(name, _arguments, _ctx):
            executed.append(name)
            return "2"

        with mock.patch.object(agent_module, "_execute_tool", side_effect=fake_execute):
            result = asyncio.run(
                agent_run_loop(
                    client=client,
                    ctx=ctx,
                    query="算一下",
                    mode=mode,
                )
            )
        return result, executed, completions.calls

    def test_fast_executes_at_most_three_tools_and_final_call_has_no_tools(self) -> None:
        responses = [
            _tool_message(4, start=0),
            _final_message("基于已有结果：2"),
        ]
        result, executed, calls = self._run("fast", responses)

        self.assertEqual(result["text"], "基于已有结果：2")
        self.assertEqual(executed, ["math_eval", "math_eval", "math_eval"])
        self.assertIn("tools", calls[0])
        self.assertIn("max_tokens", calls[1])
        self.assertNotIn("tools", calls[1])
        self.assertEqual(calls[1]["max_tokens"], agent_module.FAST_FINAL_MAX_TOKENS)

    def test_deep_can_use_twelve_tool_calls_before_forcing_a_final_answer(self) -> None:
        responses = [
            _tool_message(3, start=0),
            _tool_message(3, start=3),
            _tool_message(3, start=6),
            _tool_message(3, start=9),
            _final_message("完整调研结论"),
        ]
        result, executed, calls = self._run("deep", responses)

        self.assertEqual(result["text"], "完整调研结论")
        self.assertEqual(len(executed), 12)
        self.assertEqual(len(calls), 5)
        self.assertIn("tools", calls[3])
        self.assertNotIn("tools", calls[4])
        self.assertEqual(calls[4]["max_tokens"], agent_module.DEEP_FINAL_MAX_TOKENS)

    def test_only_three_parallel_tools_run_in_one_model_turn(self) -> None:
        # fast 首轮要了 4 个工具：只执行 3 个，第 4 个收到“预算已用完”的 tool 消息。
        responses = [
            _tool_message(4, start=0),
            _final_message("收口"),
        ]
        _result, executed, calls = self._run("fast", responses)

        assistant = calls[1]["messages"][2]
        tool_messages = [
            item for item in calls[1]["messages"] if item["role"] == "tool"
        ]
        self.assertEqual(len(executed), 3)
        self.assertEqual(len(assistant["tool_calls"]), 4)
        self.assertEqual(len(tool_messages), 4)
        self.assertIn("预算已用完", tool_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
