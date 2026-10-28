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
        response = self.responses.pop(0)
        if getattr(response, "choices", None):
            return response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=response, finish_reason="stop")]
        )


def _tool_message(count: int, start: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[_tool_call(start + index) for index in range(count)],
    )


def _final_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=[])


def _response(
    text: str,
    finish_reason: str = "stop",
    *,
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    usage = (
        SimpleNamespace(completion_tokens=completion_tokens)
        if completion_tokens is not None
        else None
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=_final_message(text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


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

    def test_length_finish_reason_continues_without_regenerating_original(self) -> None:
        responses = [
            _tool_message(1),
            _response("已查到的时间线是 10:24 你", "length"),
            _response("说要去图书馆，12:57 又改口。", "stop"),
        ]
        result, _executed, calls = self._run("fast", responses)

        self.assertEqual(
            result["text"],
            "已查到的时间线是 10:24 你说要去图书馆，12:57 又改口。",
        )
        self.assertEqual(len(calls), 3)
        self.assertIn("tools", calls[1])
        self.assertNotIn("tools", calls[2])
        self.assertEqual(calls[2]["reasoning_effort"], agent_module.FINAL_REASONING_EFFORT)
        self.assertEqual(
            calls[2]["max_tokens"],
            agent_module.FAST_CONTINUATION_MAX_TOKENS,
        )
        continuation = calls[2]["messages"][-1]["content"]
        self.assertIn("不要重头回答", continuation)
        self.assertIn("只输出续写正文", continuation)

    def test_dangling_arrow_triggers_continuation_even_without_length_reason(self) -> None:
        responses = [
            _response("时间线：10:24 你 →"),
            _response("12:57 去图书馆。"),
        ]
        result, _executed, calls = self._run("fast", responses)

        self.assertEqual(result["text"], "时间线：10:24 你 →12:57 去图书馆。")
        self.assertEqual(len(calls), 2)

    def test_complete_response_does_not_trigger_an_extra_model_call(self) -> None:
        responses = [_response("查完了，结论很明确。")]
        result, _executed, calls = self._run("fast", responses)

        self.assertEqual(result["text"], "查完了，结论很明确。")
        self.assertEqual(len(calls), 1)

    def test_continuation_stops_after_two_attempts(self) -> None:
        responses = [
            _response("第一段", "length"),
            _response("第二段", "length"),
            _response("第三段", "length"),
            _response("这段不应再被调用。", "stop"),
        ]
        result, _executed, calls = self._run("fast", responses)

        self.assertEqual(result["text"], "第一段第二段第三段")
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            len([call for call in calls if "tools" not in call]),
            2,
        )

    def test_continuation_overlap_is_not_duplicated(self) -> None:
        responses = [
            _response("他去图书馆", "length"),
            _response("图书馆待了两小时。", "stop"),
        ]
        result, _executed, _calls = self._run("fast", responses)

        self.assertEqual(result["text"], "他去图书馆待了两小时。")

    def test_forced_final_answer_is_also_repaired(self) -> None:
        responses = [
            _tool_message(3),
            _response("基于三项计算结果，结论", "length"),
            _response("是 6。", "stop"),
        ]
        result, executed, calls = self._run("fast", responses)

        self.assertEqual(result["text"], "基于三项计算结果，结论是 6。")
        self.assertEqual(len(executed), 3)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("tools", calls[1])
        self.assertEqual(calls[1]["max_tokens"], agent_module.FAST_FINAL_MAX_TOKENS)
        self.assertEqual(calls[1]["reasoning_effort"], agent_module.FINAL_REASONING_EFFORT)

    def test_long_complete_answer_is_not_hard_clipped_before_send(self) -> None:
        report = "调研正文" * 120 + "。"
        completions = _QueuedCompletions([_response(report)])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        ctx = AgentContext(model="deepseek-flash", max_output_chars=100)

        result = asyncio.run(
            agent_run_loop(client=client, ctx=ctx, query="写报告", mode="deep")
        )

        self.assertEqual(result["text"], report)
        self.assertGreater(len(result["text"]), ctx.max_output_chars)
        self.assertEqual(len(completions.calls), 1)


if __name__ == "__main__":
    unittest.main()
