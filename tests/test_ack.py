"""慢任务的「先应一句」：超过 ack_after_seconds 没结果就先发 ack，正文用 msg_seq=2。

QQ 同一条 msg_id 下 msg_seq 不能重复，ack 占 1、正文必须用 2，否则正文会被拒。
chat 也在 ack 名单里（带图或模型排队时同样卡十几秒），但它垫的那句是「我想想～」
而不是「稍等，我查一下～」——闲聊回一句「我查一下」读着像在敷衍。
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import ACK_TEXT, ACK_TEXT_CHAT, MaodouClient


def _client(ack_after_seconds: float = 0.05):
    settings = SimpleNamespace(
        timezone="Asia/Shanghai",
        ack_after_seconds=ack_after_seconds,
        allowed_group_openids=frozenset(),
        face_tags=True,
    )
    client = MaodouClient(
        settings=settings,
        store=SimpleNamespace(face_book=lambda group: {}),
        llm=SimpleNamespace(),
    )
    sent: list[tuple[int, str]] = []

    async def post_group_message(**kwargs):
        sent.append((kwargs["msg_seq"], kwargs["content"]))

    client.api = SimpleNamespace(post_group_message=post_group_message)
    client._remember_bot_reply = lambda group, text, source: None
    message = SimpleNamespace(group_openid="g", id="m-1", content="查一下")
    return client, message, sent


def _args(client, message, action, handler):
    async def fake_handle_intent(**kwargs):
        return await handler()

    client._handle_intent = fake_handle_intent
    return dict(
        message=message,
        intent={"action": action},
        requester=SimpleNamespace(),
        request="查一下",
        topic="",
        background="",
        full_recent="",
        identity=SimpleNamespace(prompt=lambda: ""),
        source="GROUP_AT_MESSAGE_CREATE",
    )


class AckTests(unittest.TestCase):
    def setUp(self) -> None:
        # botpy.Client.__init__ 需要一个当前事件循环，测试里手动建一个。
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_slow_search_sends_ack_then_answer(self) -> None:
        client, message, sent = _client()

        async def slow():
            await asyncio.sleep(0.2)
            return "查到啦"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "agent_run", slow)))
        self.assertTrue(acked)
        self.assertEqual(sent, [(1, ACK_TEXT), (2, "查到啦")])

    def test_fast_reply_sends_single_message(self) -> None:
        client, message, sent = _client()

        async def quick():
            return "好"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "chat", quick)))
        self.assertFalse(acked)
        self.assertEqual(sent, [(1, "好")])

    def test_non_ack_action_never_acks(self) -> None:
        client, message, sent = _client()

        async def slow():
            await asyncio.sleep(0.2)
            return "记住了"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "task_create", slow)))
        self.assertFalse(acked)
        self.assertEqual(len(sent), 1)

    def test_zero_timeout_disables_ack(self) -> None:
        client, message, sent = _client(ack_after_seconds=0)

        async def slow():
            await asyncio.sleep(0.05)
            return "结果"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "agent_run", slow)))
        self.assertFalse(acked)
        self.assertEqual(sent, [(1, "结果")])

    def test_handler_error_still_uses_free_seq(self) -> None:
        client, message, sent = _client()

        async def broken():
            await asyncio.sleep(0.2)
            raise RuntimeError("boom")

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "agent_run", broken)))
        self.assertTrue(acked)
        self.assertEqual(sent[0], (1, ACK_TEXT))
        self.assertEqual(sent[1][0], 2)
        self.assertEqual(sent[1][1], bot_module.FALLBACK_TEXT)

    def test_pre_acked_answer_uses_seq_two_and_skips_second_ack(self) -> None:
        client, message, sent = _client()

        async def slow():
            await asyncio.sleep(0.2)
            return "九张看完了"

        args = _args(client, message, "chat", slow)
        args["pre_acked"] = True
        acked = asyncio.run(client._answer_with_ack(**args))
        self.assertTrue(acked)
        self.assertEqual(sent, [(2, "九张看完了")])

    def test_failed_ack_keeps_seq_one_for_answer(self) -> None:
        client, message, _sent = _client()

        async def post_group_message(**kwargs):
            if kwargs["msg_seq"] == 1:
                raise RuntimeError("rate limited")

        client.api = SimpleNamespace(post_group_message=post_group_message)

        async def slow():
            await asyncio.sleep(0.1)
            return "答案"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "agent_run", slow)))
        self.assertFalse(acked)

    def test_ack_actions_cover_slow_tools(self) -> None:
        self.assertIn("agent_run", bot_module.ACK_ACTIONS)

    def test_voice_chat_uses_the_chat_ack_and_never_the_tool_ack(self) -> None:
        client, message, sent = _client()

        async def slow():
            await asyncio.sleep(0.2)
            return "晚安"

        acked = asyncio.run(
            client._answer_with_ack(
                **_args(client, message, "voice_chat", slow)
            )
        )
        self.assertTrue(acked)
        self.assertEqual(sent, [(1, ACK_TEXT_CHAT), (2, "晚安")])
        self.assertIn("voice_chat", bot_module.ACK_ACTIONS)
        self.assertNotEqual(ACK_TEXT_CHAT, ACK_TEXT)

    def test_slow_chat_acks_with_thinking_text(self) -> None:
        client, message, sent = _client()

        async def slow():
            await asyncio.sleep(0.2)
            return "在的～"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "chat", slow)))
        self.assertTrue(acked)
        self.assertEqual(sent, [(1, ACK_TEXT_CHAT), (2, "在的～")])

    def test_chat_ack_wording_differs_from_tool_ack(self) -> None:
        self.assertIn("chat", bot_module.ACK_ACTIONS)
        self.assertNotEqual(ACK_TEXT_CHAT, ACK_TEXT)

    def test_slow_chat_ack_failure_leaves_seq_one_for_answer(self) -> None:
        client, message, sent = _client()
        calls = {"n": 0}

        async def post_group_message(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # 只让 ack 那一次失败，正文仍然可以占 seq=1
                raise RuntimeError("rate limited")
            sent.append((kwargs["msg_seq"], kwargs["content"]))

        client.api = SimpleNamespace(post_group_message=post_group_message)

        async def slow():
            await asyncio.sleep(0.1)
            return "在的～"

        acked = asyncio.run(client._answer_with_ack(**_args(client, message, "chat", slow)))
        self.assertFalse(acked)
        self.assertEqual(sent, [(1, "在的～")])


if __name__ == "__main__":
    unittest.main()
