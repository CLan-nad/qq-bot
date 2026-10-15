"""被动回复发送失败的补救：seq 撞车换号重发、重试、主动消息兜底。

线上那次 40054005 的后果是「@ 了没反应」——正文被 QQ 当重复请求拒掉，
而代码直接 return False，既没重发也没兜底，用户啥也没收到。
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import xiaoxia.bot as bot_module
from xiaoxia.bot import XiaoxiaClient


DEDUP = RuntimeError("消息被去重，请检查请求msgseq code=40054005")
NETWORK = RuntimeError("connection reset")
THROTTLED = RuntimeError("post_group_message failed: retcode=40034102")


def _client(script):
    """script: 依次决定每次 post_group_message 的行为。

    每项是 None（成功）或一个异常（失败）。用完之后一律成功。
    返回 (client, message, sent, proactive)。
    """
    client = XiaoxiaClient(
        settings=SimpleNamespace(timezone="Asia/Shanghai"),
        store=SimpleNamespace(),
        llm=SimpleNamespace(),
    )
    sent: list[tuple] = []
    proactive: list[str] = []
    remembered: list[tuple[str, str]] = []
    calls = {"n": 0}

    async def post_group_message(**kwargs):
        index = calls["n"]
        calls["n"] += 1
        if "msg_seq" not in kwargs:
            # 没有 msg_id/msg_seq 的就是主动消息兜底
            failure = script[index] if index < len(script) else None
            if failure is not None:
                raise failure
            proactive.append(kwargs["content"])
            return
        sent.append((kwargs["msg_seq"], kwargs["content"]))
        failure = script[index] if index < len(script) else None
        if failure is not None:
            raise failure

    client.api = SimpleNamespace(post_group_message=post_group_message)
    client._remember_bot_reply = lambda group, text, source: remembered.append((text, source))
    message = SimpleNamespace(group_openid="g", id="m-1", content="在吗")
    return client, message, sent, proactive, remembered


class SendReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        # botpy.Client.__init__ 会去取当前事件循环，没有就直接抛，所以先建一个。
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._delay = bot_module.SEND_RETRY_DELAY_SECONDS
        bot_module.SEND_RETRY_DELAY_SECONDS = 0.0

    def tearDown(self) -> None:
        bot_module.SEND_RETRY_DELAY_SECONDS = self._delay
        asyncio.set_event_loop(None)
        self.loop.close()

    def _send(self, script, *, seq=1, attempts=bot_module.SEND_ATTEMPTS, fallback=True):
        client, message, sent, proactive, remembered = _client(script)
        ok = asyncio.run(
            client._send_reply(
                message,
                "答案",
                seq,
                "GROUP_MESSAGE_CREATE:REPLY",
                attempts=attempts,
                proactive_fallback=fallback,
            )
        )
        return ok, sent, proactive, remembered, client

    def test_happy_path(self) -> None:
        ok, sent, proactive, remembered, _ = self._send([None])
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "答案")])
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [("答案", "GROUP_MESSAGE_CREATE:REPLY")])

    def test_dedup_on_first_try_resends_with_next_seq(self) -> None:
        # ack 已经占掉 seq=1（本地却以为没发出去），正文必须换 seq=2 再发一次
        ok, sent, proactive, remembered, _ = self._send([DEDUP, None], seq=1)
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "答案"), (2, "答案")])
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [("答案", "GROUP_MESSAGE_CREATE:REPLY")])

    def test_dedup_on_retry_means_it_already_landed(self) -> None:
        # 第一次是网络抖动，第二次 QQ 说重复 => 其实第一次已经送到了，不能再发一遍
        ok, sent, proactive, remembered, _ = self._send([NETWORK, DEDUP], seq=1)
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "答案"), (1, "答案")])
        self.assertEqual(proactive, [])
        self.assertEqual(len(remembered), 1)

    def test_transient_errors_retry_the_same_seq(self) -> None:
        ok, sent, proactive, remembered, _ = self._send([NETWORK, NETWORK, None], seq=2)
        self.assertTrue(ok)
        self.assertEqual([item[0] for item in sent], [2, 2, 2])
        self.assertEqual(proactive, [])
        self.assertEqual(len(remembered), 1)

    def test_all_passive_attempts_fail_then_proactive_fallback(self) -> None:
        ok, sent, proactive, remembered, _ = self._send([NETWORK, NETWORK, NETWORK, None])
        self.assertTrue(ok)
        self.assertEqual(len(sent), 3)
        self.assertEqual(proactive, ["答案"])
        # 兜底发出去的也要记进本地库，否则下一轮它看不到自己说过什么
        self.assertEqual(remembered, [("答案", "GROUP_MESSAGE_CREATE:REPLY:FALLBACK")])

    def test_everything_fails_returns_false_and_remembers_nothing(self) -> None:
        ok, sent, proactive, remembered, client = self._send(
            [NETWORK, NETWORK, NETWORK, THROTTLED]
        )
        self.assertFalse(ok)
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [])
        # 主动通道撞上频控，当天就别再主动发消息了
        self.assertEqual(client._proactive_blocked_until.get("g"), client._now().date().isoformat())

    def test_ack_neither_retries_nor_falls_back(self) -> None:
        # ack 发不出去就算了：重试会白等，兜底会拿主动消息额度发一句“稍等”
        ok, sent, proactive, remembered, _ = self._send(
            [NETWORK], attempts=1, fallback=False
        )
        self.assertFalse(ok)
        self.assertEqual(len(sent), 1)
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [])

    def test_dedup_detection(self) -> None:
        self.assertTrue(XiaoxiaClient._is_dedup_error(DEDUP))
        self.assertFalse(XiaoxiaClient._is_dedup_error(NETWORK))
        self.assertFalse(XiaoxiaClient._is_dedup_error(THROTTLED))


if __name__ == "__main__":
    unittest.main()
