from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.voice import VoiceReply


class _FakeSynth:
    def __init__(self, *, result: bytes = b"audio", error: Exception | None = None) -> None:
        self.enabled = True
        self.result = result
        self.error = error
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.result


def _client() -> tuple[MaodouClient, SimpleNamespace, list[dict], list[bytes]]:
    client = MaodouClient(
        settings=SimpleNamespace(
            timezone="Asia/Shanghai",
            face_tags=False,
            reply_chunk_chars=900,
        ),
        store=SimpleNamespace(face_book=lambda group: {}),
        llm=SimpleNamespace(),
    )
    sent: list[dict] = []
    uploads: list[bytes] = []

    async def post_group_message(**kwargs):
        sent.append(kwargs)

    async def upload_voice(group_openid: str, data: bytes) -> str:
        uploads.append(data)
        return "voice-file-info"

    client.api = SimpleNamespace(post_group_message=post_group_message)
    client._upload_group_voice = upload_voice
    client._remember_bot_reply = lambda group, text, source: None
    message = SimpleNamespace(group_openid="g", id="m-1", content="发语音")
    return client, message, sent, uploads


class VoiceSendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._delay = bot_module.SEND_RETRY_DELAY_SECONDS
        bot_module.SEND_RETRY_DELAY_SECONDS = 0.0

    def tearDown(self) -> None:
        bot_module.SEND_RETRY_DELAY_SECONDS = self._delay
        asyncio.set_event_loop(None)
        self.loop.close()

    def _send(self, text, synth, *, attempts=3, fallback=False, seq=1):
        client, message, sent, uploads = _client()
        client.voice_synthesizer = synth
        ok = self.loop.run_until_complete(
            client._send_reply(
                message,
                text,
                seq,
                "TEST:REPLY",
                attempts=attempts,
                proactive_fallback=fallback,
            )
        )
        return ok, sent, uploads, synth

    def test_voice_reply_sends_only_one_voice(self) -> None:
        synth = _FakeSynth()
        ok, sent, uploads, _ = self._send(VoiceReply("晚安呀"), synth)

        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["msg_type"], 7)
        self.assertEqual(sent[0]["msg_seq"], 1)
        self.assertIsNone(sent[0]["content"])
        self.assertEqual(sent[0]["media"], {"file_info": "voice-file-info"})
        self.assertEqual(uploads, [b"audio"])
        self.assertEqual(synth.calls, ["晚安呀"])

    def test_long_voice_text_is_not_split_into_preceding_text_messages(self) -> None:
        text = "一二三四。五六七八。"
        synth = _FakeSynth()
        client, message, sent, _ = _client()
        client.settings.reply_chunk_chars = 6
        client.voice_synthesizer = synth

        ok = self.loop.run_until_complete(
            client._send_reply(message, VoiceReply(text), 1, "TEST:REPLY")
        )

        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["msg_type"], 7)
        self.assertEqual(sent[0]["msg_seq"], 1)
        self.assertEqual(synth.calls, [text])

    def test_voice_upload_failure_falls_back_to_text(self) -> None:
        client, message, sent, uploads = _client()
        synth = _FakeSynth()
        client.voice_synthesizer = synth

        async def upload_failed(group_openid: str, data: bytes) -> str:
            uploads.append(data)
            return ""

        client._upload_group_voice = upload_failed
        ok = self.loop.run_until_complete(
            client._send_reply(message, VoiceReply("晚安"), 1, "TEST:REPLY")
        )

        self.assertTrue(ok)
        self.assertEqual(uploads, [b"audio"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["content"], "晚安")

    def test_tts_failure_falls_back_to_text_and_never_uploads(self) -> None:
        synth = _FakeSynth(error=RuntimeError("TTS down"))
        ok, sent, uploads, _ = self._send(VoiceReply("晚安"), synth)
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["content"], "晚安")
        self.assertEqual(uploads, [])

    def test_voice_send_failure_falls_back_to_text(self) -> None:
        client, message, sent, uploads = _client()
        client.voice_synthesizer = _FakeSynth()

        async def post(**kwargs):
            sent.append(kwargs)
            if kwargs.get("msg_type") == 7:
                raise RuntimeError("voice rejected")

        client.api.post_group_message = post
        ok = self.loop.run_until_complete(
            client._send_reply(message, VoiceReply("晚安"), 1, "TEST:REPLY", attempts=1)
        )
        self.assertTrue(ok)
        self.assertEqual(len(uploads), 1)
        self.assertEqual([item["msg_seq"] for item in sent], [1, 1])
        self.assertEqual(sent[0]["msg_type"], 7)
        self.assertEqual(sent[1]["content"], "晚安")

    def test_no_tts_configuration_degrades_to_text_only(self) -> None:
        synth = SimpleNamespace(enabled=False)
        ok, sent, uploads, _ = self._send(VoiceReply("晚安"), synth)
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        self.assertEqual(uploads, [])

    def test_voice_reply_does_not_append_a_sticker(self) -> None:
        client, message, sent, _uploads = _client()
        client.voice_synthesizer = _FakeSynth()
        client._outgoing_parts = lambda group, text: ("晚安", object())

        async def forbidden_upload(*args, **kwargs):
            raise AssertionError("语音回复不应附表情包")

        client._upload_group_image = forbidden_upload
        ok = self.loop.run_until_complete(
            client._send_reply(message, VoiceReply("晚安 [抱抱]"), 1, "TEST:REPLY")
        )
        self.assertTrue(ok)
        self.assertEqual([item.get("msg_type") for item in sent], [7])


class VoiceIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, *, enabled: bool = True) -> MaodouClient:
        client = MaodouClient(
            settings=SimpleNamespace(face_tags=False),
            store=SimpleNamespace(face_book=lambda group: {}),
            llm=SimpleNamespace(),
        )
        client.voice_synthesizer = SimpleNamespace(enabled=enabled)
        client.reply_calls = []

        async def generate_reply(**kwargs):
            client.reply_calls.append(kwargs)
            return "晚安，做个好梦"

        client.llm = SimpleNamespace(generate_reply=generate_reply)
        client._memory_context = lambda group: ""
        client._summaries_context = lambda group: ""
        return client

    def _run(self, client: MaodouClient, action: str) -> str:
        return self.loop.run_until_complete(
            client._handle_intent(
                intent={"action": action},
                message=SimpleNamespace(group_openid="g"),
                requester=SimpleNamespace(),
                request="发语音说晚安",
                topic="",
                background="",
                full_recent="",
                identity=SimpleNamespace(prompt=lambda: ""),
            )
        )

    def test_voice_chat_marks_reply_for_separate_voice_delivery(self) -> None:
        client = self._client()
        answer = self._run(client, "voice_chat")
        self.assertIsInstance(answer, VoiceReply)
        self.assertEqual(str(answer), "晚安，做个好梦")
        self.assertTrue(client.reply_calls[-1]["voice_reply"])

    def test_voice_chat_without_tts_returns_plain_text(self) -> None:
        client = self._client(enabled=False)
        answer = self._run(client, "voice_chat")
        self.assertNotIsInstance(answer, VoiceReply)
        self.assertEqual(answer, "晚安，做个好梦")
        self.assertFalse(client.reply_calls[-1]["voice_reply"])

    def test_regular_chat_never_becomes_voice(self) -> None:
        client = self._client()
        answer = self._run(client, "chat")
        self.assertNotIsInstance(answer, VoiceReply)
        self.assertFalse(client.reply_calls[-1]["voice_reply"])


if __name__ == "__main__":
    unittest.main()
