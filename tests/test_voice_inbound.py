from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from maodou.bot import (
    ACK_TEXT_VOICE,
    ACK_TEXT_VOICE_IMAGES,
    MaodouClient,
)
from maodou.store import ChatStore
from maodou.voice import VoiceAttachment


GROUP = "test-group"
BOT_ID = "bot-1"
VOICE_RAW = {
    "attachments": [
        {
            "content_type": "voice",
            "filename": "voice.silk",
            "url": "https://example.com/voice.silk",
            "voice_wav_url": "https://example.com/voice.wav",
            "asr_refer_text": "今天下班好累",
        }
    ]
}


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        timezone="Asia/Shanghai",
        ack_after_seconds=0.0,
        allowed_group_openids=frozenset(),
        recent_messages=30,
        max_context_chars=4000,
        vision_recent_messages=20,
        vision_max_images_per_request=0,
        vision_max_image_bytes=1024 * 1024,
        image_cache_enabled=False,
        image_cache_max_rows=0,
        asr_max_per_request=5,
        face_tags=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _at_message(content: str = "[语音]", raw: dict | None = None) -> SimpleNamespace:
    payload = {
        "mentions": [{"bot": True, "is_you": True, "id": BOT_ID}],
    }
    if raw:
        payload.update(raw)
    return SimpleNamespace(
        group_openid=GROUP,
        id="at-message",
        msg_seq="0",
        content=content,
        timestamp="2026-10-28T20:00:00+08:00",
        username="CL",
        member_role="member",
        author=SimpleNamespace(member_openid="openid-CL"),
        raw=payload,
    )


class _FakeTranscriber:
    def __init__(self, values: list[str]) -> None:
        self.values = values
        self.calls: list[VoiceAttachment] = []

    async def transcribe(self, attachment: VoiceAttachment) -> str:
        self.calls.append(attachment)
        return self.values.pop(0) if self.values else ""


class _LoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "voice.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()


class VoiceInboundTests(_LoopTestCase):
    def _client(
        self,
        *,
        replies: list[str] | None = None,
        **overrides,
    ) -> tuple[MaodouClient, list[tuple[int, str]], list[str], list[str]]:
        client = MaodouClient(
            settings=_settings(**overrides),
            store=self.store,
            llm=SimpleNamespace(),
        )
        sent: list[tuple[int, str]] = []
        intent_requests: list[str] = []
        topics: list[str] = []

        async def parse_intent(**kwargs):
            intent_requests.append(kwargs["request"])
            topics.append(kwargs["topic_context"])
            return {"action": "chat"}

        async def handle_intent(**kwargs):
            return "听起来今天挺辛苦的，先歇会儿吧"

        async def post_group_message(**kwargs):
            sent.append((kwargs["msg_seq"], kwargs.get("content")))

        client.llm = SimpleNamespace(parse_intent=parse_intent)
        client._handle_intent = handle_intent
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._identity_context = lambda group, requester: SimpleNamespace(
            prompt=lambda: ""
        )
        client._memory_context = lambda group: ""
        client._summaries_context = lambda group: ""
        client._remember_bot_reply = lambda group, text, source: None
        client._queue_image_backfill = lambda *args, **kwargs: None
        client.voice_transcriber = _FakeTranscriber(
            list(replies or ["今天下班好累"])
        )
        return client, sent, intent_requests, topics

    def test_direct_voice_at_is_transcribed_acked_and_visible_in_topic(self) -> None:
        client, sent, intents, topics = self._client()
        self.loop.run_until_complete(
            client._process_at_message(
                _at_message(raw=VOICE_RAW),
                "GROUP_AT_MESSAGE_CREATE",
            )
        )

        self.assertEqual(sent[0], (1, ACK_TEXT_VOICE))
        self.assertEqual(sent[1], (2, "听起来今天挺辛苦的，先歇会儿吧"))
        self.assertIn("今天下班好累", intents[0])
        self.assertIn("〔语音〕今天下班好累", topics[0])
        stored = self.store.recent_messages_with_raw(GROUP, 5)
        self.assertIn("[语音内容：今天下班好累]", stored[-1].content)

    def test_voice_sent_before_the_at_is_synchronously_transcribed(self) -> None:
        self.store.save_message(
            message_id="voice-before-at",
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content="[语音]",
            timestamp="2026-10-28T19:59:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw=VOICE_RAW,
        )
        client, sent, intents, topics = self._client()
        self.loop.run_until_complete(
            client._process_at_message(
                _at_message("刚才那条语音说了啥"),
                "GROUP_AT_MESSAGE_CREATE",
            )
        )

        self.assertEqual(sent[0], (1, ACK_TEXT_VOICE))
        self.assertIn("〔语音〕今天下班好累", topics[0])
        self.assertIn("刚才那条语音说了啥", intents[0])
        stored = {
            item.message_key.split(":", 1)[0]: item.content
            for item in self.store.recent_messages_with_raw(GROUP, 10)
        }
        self.assertIn("今天下班好累", stored["voice-before-at"])

    def test_empty_transcript_is_marked_unrecognized_and_not_invented(self) -> None:
        client, sent, intents, topics = self._client(replies=[""])
        self.loop.run_until_complete(
            client._process_at_message(
                _at_message(raw=VOICE_RAW),
                "GROUP_AT_MESSAGE_CREATE",
            )
        )

        self.assertEqual(sent[0], (1, ACK_TEXT_VOICE))
        self.assertIn("〔语音〕（没听清）", topics[0])
        self.assertNotIn("今天下班好累", intents[0])
        stored = self.store.recent_messages_with_raw(GROUP, 5)
        self.assertIn("[语音未识别]", stored[-1].content)

    def test_image_and_voice_share_one_media_ack(self) -> None:
        raw = {
            "attachments": [
                {
                    "content_type": "image/jpeg",
                    "filename": "cat.jpg",
                    "url": "https://example.com/cat.jpg",
                },
                dict(VOICE_RAW["attachments"][0]),
            ]
        }
        client, sent, _intents, topics = self._client(
            vision_max_images_per_request=1
        )

        async def describe(attachments, hint=""):
            return ["一只橘猫"]

        client._describe_attachments = describe
        self.loop.run_until_complete(
            client._process_at_message(
                _at_message(raw=raw),
                "GROUP_AT_MESSAGE_CREATE",
            )
        )

        self.assertEqual(sent[0], (1, ACK_TEXT_VOICE_IMAGES))
        self.assertIn("〔图〕一只橘猫", topics[0])
        self.assertIn("〔语音〕今天下班好累", topics[0])
        self.assertEqual([item[0] for item in sent], [1, 2])

    def test_voice_batch_respects_the_per_request_budget(self) -> None:
        for index in range(3):
            self.store.save_message(
                message_id=f"voice-{index}",
                msg_seq="",
                group_openid=GROUP,
                member_openid="openid-CL",
                username="CL",
                role="member",
                content="[语音]",
                timestamp=f"2026-10-28T19:5{index}:00+08:00",
                source="GROUP_MESSAGE_CREATE",
                raw={
                    "attachments": [
                        {
                            "content_type": "voice",
                            "voice_wav_url": f"https://example.com/{index}.wav",
                            "asr_refer_text": f"第{index}条",
                        }
                    ]
                },
            )
        client, _sent, _intents, topics = self._client(
            # 批次从旧到新处理，预算只保留最新的 voice-1、voice-2。
            replies=["第1条", "第2条"],
            asr_max_per_request=2,
        )
        self.loop.run_until_complete(
            client._process_at_message(
                _at_message("听听前面的语音"),
                "GROUP_AT_MESSAGE_CREATE",
            )
        )

        stored = {
            item.message_key.split(":", 1)[0]: item.content
            for item in self.store.recent_messages_with_raw(GROUP, 10)
        }
        self.assertIn("第2条", stored["voice-2"])
        self.assertIn("第1条", stored["voice-1"])
        self.assertEqual(stored["voice-0"], "[语音]")
        self.assertNotIn("第0条", topics[0])


if __name__ == "__main__":
    unittest.main()
