"""被动回复发送失败的补救：seq 撞车换号重发、重试、主动消息兜底。

线上那次 40054005 的后果是「@ 了没反应」——正文被 QQ 当重复请求拒掉，
而代码直接 return False，既没重发也没兜底，用户啥也没收到。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.replies import split_reply_text
from maodou.stickers import Sticker, StickerLibrary


DEDUP = RuntimeError("消息被去重，请检查请求msgseq code=40054005")
NETWORK = RuntimeError("connection reset")
THROTTLED = RuntimeError("post_group_message failed: retcode=40034102")


def _client(script):
    """script: 依次决定每次 post_group_message 的行为。

    每项是 None（成功）或一个异常（失败）。用完之后一律成功。
    返回 (client, message, sent, proactive)。
    """
    client = MaodouClient(
        settings=SimpleNamespace(timezone="Asia/Shanghai", face_tags=True),
        store=SimpleNamespace(face_book=lambda group: {}),
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
        self.assertTrue(MaodouClient._is_dedup_error(DEDUP))
        self.assertFalse(MaodouClient._is_dedup_error(NETWORK))
        self.assertFalse(MaodouClient._is_dedup_error(THROTTLED))

    def test_long_text_uses_one_contiguous_seq_per_chunk(self) -> None:
        text = "一二三四。五六七八。九十"
        expected = split_reply_text(text, 6)
        client, message, sent, proactive, remembered = _client([None, None, None])
        client.settings.reply_chunk_chars = 6

        ok = asyncio.run(
            client._send_reply(message, text, 1, "GROUP_MESSAGE_CREATE:REPLY")
        )

        self.assertTrue(ok)
        self.assertEqual(sent, list(enumerate(expected, start=1)))
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [(text, "GROUP_MESSAGE_CREATE:REPLY")])

    def test_ack_offset_leaves_a_contiguous_seq_range_for_all_text_chunks(self) -> None:
        text = "一二三四。五六七八。九十"
        expected = split_reply_text(text, 6)
        client, message, sent, proactive, remembered = _client([None, None, None])
        client.settings.reply_chunk_chars = 6

        ok = asyncio.run(
            client._send_reply(message, text, 2, "GROUP_MESSAGE_CREATE:REPLY")
        )

        self.assertTrue(ok)
        self.assertEqual(sent, list(enumerate(expected, start=2)))
        self.assertEqual(proactive, [])
        self.assertEqual(remembered, [(text, "GROUP_MESSAGE_CREATE:REPLY")])

    def test_partial_chunk_failure_only_falls_back_with_remaining_text(self) -> None:
        text = "第一段。第二段。第三段。"
        chunks = split_reply_text(text, 5)
        self.assertEqual(chunks, ["第一段。", "第二段。", "第三段。"])
        client, message, sent, proactive, remembered = _client(
            [None, NETWORK, NETWORK, NETWORK, None, None]
        )
        client.settings.reply_chunk_chars = 5

        ok = asyncio.run(
            client._send_reply(
                message,
                text,
                1,
                "GROUP_MESSAGE_CREATE:REPLY",
                attempts=3,
            )
        )

        self.assertTrue(ok)
        self.assertEqual(sent[0], (1, chunks[0]))
        self.assertEqual(proactive, chunks[1:])
        self.assertEqual((sent + [(None, item) for item in proactive]).count((1, chunks[0])), 1)
        self.assertEqual(remembered, [(text, "GROUP_MESSAGE_CREATE:REPLY")])


class StickerSendTests(unittest.TestCase):
    """表情图单独走图片消息：先发正文，再用下一个 seq 传图并发送。

    图失败时正文已经送达，不能重发正文，也不能把整轮判成失败。
    """

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        gif = Path(self.tmp.name) / "a_1f917.gif"
        gif.write_bytes(b"GIF89a")
        self.sticker = Sticker(name="抱抱", path=gif, emoji="🤗")
        self.library = StickerLibrary(Path(self.tmp.name), {"抱抱": self.sticker})
        self._delay = bot_module.SEND_RETRY_DELAY_SECONDS
        bot_module.SEND_RETRY_DELAY_SECONDS = 0.0

    def tearDown(self) -> None:
        bot_module.SEND_RETRY_DELAY_SECONDS = self._delay
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, script, *, upload="file-info-1", reply_chunk_chars=900):
        """script 同 _client：依次决定每次发送的成败，用完之后一律成功。"""
        client = MaodouClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                face_tags=True,
                reply_chunk_chars=reply_chunk_chars,
            ),
            store=SimpleNamespace(face_book=lambda group: {}),
            llm=SimpleNamespace(),
        )
        client.stickers = self.library
        sent: list[tuple[int, str, str]] = []
        remembered: list[tuple[str, str]] = []
        calls = {"n": 0}
        uploads = {"n": 0}

        async def post_group_message(**kwargs):
            index = calls["n"]
            calls["n"] += 1
            failure = script[index] if index < len(script) else None
            if failure is not None:
                raise failure
            if "msg_seq" not in kwargs:
                return
            kind = "sticker" if kwargs.get("msg_type") == 7 else "text"
            sent.append((kwargs["msg_seq"], kwargs["content"], kind))

        async def upload_group_image(group_openid, sticker):
            # 传图是独立的一次请求，不能吃掉 script 里给「发送」准备的那一格
            uploads["n"] += 1
            return upload

        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._upload_group_image = upload_group_image
        client._remember_bot_reply = lambda group, text, source: remembered.append((text, source))
        message = SimpleNamespace(group_openid="g", id="m-1", content="在吗")
        return client, message, sent, remembered, uploads

    def _send(
        self,
        script,
        text,
        *,
        upload="file-info-1",
        seq=1,
        attempts=3,
        fallback=True,
        remember=True,
        reply_chunk_chars=900,
    ):
        client, message, sent, remembered, uploads = self._client(
            script,
            upload=upload,
            reply_chunk_chars=reply_chunk_chars,
        )
        ok = self.loop.run_until_complete(
            client._send_reply(
                message,
                text,
                seq,
                "GROUP_MESSAGE_CREATE:REPLY",
                attempts=attempts,
                proactive_fallback=fallback,
                remember=remember,
            )
        )
        return ok, sent, remembered, uploads

    def test_remember_false_sends_without_persisting(self) -> None:
        ok, sent, remembered, _uploads = self._send([None], "我一张张看～", remember=False)
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "我一张张看～", "text")])
        self.assertEqual(remembered, [])

    def test_text_goes_first_then_sticker_uses_the_next_seq(self) -> None:
        ok, sent, remembered, uploads = self._send([None, None], "抱抱你呀[抱抱]")
        self.assertTrue(ok)
        self.assertEqual(
            sent,
            [
                (1, "抱抱你呀", "text"),
                (2, None, "sticker"),
            ],
        )
        # 本地库里留可读名字，下一轮它才认得自己刚发过什么
        self.assertEqual(remembered, [("抱抱你呀[抱抱]", "GROUP_MESSAGE_CREATE:REPLY")])
        self.assertEqual(uploads["n"], 1)

    def test_sticker_upload_runs_while_the_text_is_being_sent(self) -> None:
        """图的上传和文字发送并行：用户看到的顺序仍是文字在前，但少等一次上传。"""
        client, message, _sent, _remembered, _uploads = self._client([None, None])
        events: list[str] = []

        async def slow_text_post(**kwargs):
            if kwargs.get("msg_type") == 7:
                return
            events.append("text-start")
            await asyncio.sleep(0.05)
            events.append("text-done")

        async def upload(_group, _sticker):
            events.append("upload")
            await asyncio.sleep(0.01)
            return "file-info"

        client.api.post_group_message = slow_text_post
        client._upload_group_image = upload
        ok = self.loop.run_until_complete(
            client._send_reply(message, "抱抱你呀[抱抱]", 1, "TEST")
        )

        self.assertTrue(ok)
        self.assertEqual(events[0], "text-start")
        self.assertIn("upload", events)
        self.assertLess(events.index("upload"), events.index("text-done"))

    def test_upload_failure_keeps_the_text_and_does_not_retry(self) -> None:
        ok, sent, remembered, uploads = self._send([None], "抱抱你呀[抱抱]", upload="")
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "抱抱你呀", "text")])
        self.assertEqual(len(remembered), 1)
        self.assertEqual(uploads["n"], 1)

    def test_pure_sticker_that_cannot_be_uploaded_reports_failure(self) -> None:
        # 只剩一张图、图又发不出去：没有能被文字兜住的正文，就当没发出去
        ok, sent, remembered, uploads = self._send([None], "[抱抱]", upload="")
        self.assertFalse(ok)
        self.assertEqual(sent, [])
        self.assertEqual(remembered, [])
        self.assertEqual(uploads["n"], 1)

    def test_pure_sticker_sends_image_with_no_caption(self) -> None:
        ok, sent, _, _ = self._send([None], "[抱抱]")
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, None, "sticker")])

    def test_dedup_on_pure_sticker_resends_with_the_next_seq(self) -> None:
        ok, sent, _, _ = self._send([DEDUP, None], "[抱抱]", seq=1)
        self.assertTrue(ok)
        self.assertEqual(sent, [(2, None, "sticker")])

    def test_sticker_send_error_after_text_does_not_resend_text(self) -> None:
        ok, sent, remembered, _ = self._send([None, NETWORK], "抱抱你呀[抱抱]")
        self.assertTrue(ok)
        self.assertEqual(sent, [(1, "抱抱你呀", "text")])
        self.assertEqual(remembered, [("抱抱你呀[抱抱]", "GROUP_MESSAGE_CREATE:REPLY")])

    def test_text_failure_never_sends_the_sticker(self) -> None:
        ok, sent, remembered, uploads = self._send(
            [NETWORK], "抱抱你呀[抱抱]", attempts=1, fallback=False
        )
        self.assertFalse(ok)
        self.assertEqual(sent, [])
        self.assertEqual(remembered, [])
        self.assertEqual(uploads["n"], 0)

    def test_ack_seq_offset_is_preserved_for_text_and_sticker(self) -> None:
        ok, sent, _, _ = self._send([None, None], "抱抱你呀[抱抱]", seq=2)
        self.assertTrue(ok)
        self.assertEqual([item[0] for item in sent], [2, 3])

    def test_text_dedup_shift_leaves_room_for_sticker(self) -> None:
        # ack 实际占了 seq=1，文字退到 2，图片必须继续用 3
        ok, sent, _, _ = self._send([DEDUP, None, None], "抱抱你呀[抱抱]", seq=1)
        self.assertTrue(ok)
        self.assertEqual([item[0] for item in sent], [2, 3])

    def test_all_text_chunks_go_before_the_sticker(self) -> None:
        text = "一二三四。五六七八。[抱抱]"
        expected = split_reply_text("一二三四。五六七八。", 6)
        self.assertEqual(len(expected), 2)

        ok, sent, remembered, _ = self._send(
            [None, None, None],
            text,
            reply_chunk_chars=6,
        )

        self.assertTrue(ok)
        self.assertEqual(
            sent,
            [
                (1, expected[0], "text"),
                (2, expected[1], "text"),
                (3, None, "sticker"),
            ],
        )
        self.assertEqual(remembered, [(text, "GROUP_MESSAGE_CREATE:REPLY")])


if __name__ == "__main__":
    unittest.main()
