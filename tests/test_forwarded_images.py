"""转发聊天记录（message_type=102）里的图片也要能被识别。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.images import (
    UNRECOGNIZED_IMAGE_MARK,
    format_message_content,
    forwarded_image_attachments,
    image_attachments,
    message_with_image_descriptions,
    message_with_unrecognized_images,
    needs_image_description,
)
from maodou.store import ChatStore

GROUP = "forward-group"
JPEG_BYTES = b"\xff\xd8\xff\xe0forwarded-stub"


def _attachment_line(
    index: int,
    *,
    name: str = "40B49982C61ABEE73F3CA2867031F2E3.jpg",
    kind: str = "图片",
    file_id: str = "demo",
) -> str:
    return (
        f"[附件{index}] 类型:{kind} 文件名:{name} 尺寸:1224x1613 大小:196.8KB "
        f"URL:https://multimedia.nt.qq.com.cn/download?appid=1407&fileid={file_id}&rkey=demo"
    )


FORWARDED_CONTENT = "\n".join(
    [
        "[群聊的聊天记录]",
        "=== 消息 1 ===",
        "[发送者] 玊",
        "",
        "=== 消息 2 ===",
        "[发送者] 玊",
        _attachment_line(1),
        "",
    ]
)

TWO_IMAGE_CONTENT = "\n".join(
    [
        "[群聊的聊天记录]",
        "=== 消息 1 ===",
        "[发送者] 玊",
        _attachment_line(1, name="a.jpg", file_id="a"),
        "=== 消息 2 ===",
        "[发送者] 阿",
        _attachment_line(2, name="b.png", file_id="b"),
        "",
    ]
)


class ForwardedAttachmentTests(unittest.TestCase):
    def test_extracts_forwarded_image(self) -> None:
        attachments = forwarded_image_attachments(FORWARDED_CONTENT)
        self.assertEqual(len(attachments), 1)
        attachment = attachments[0]
        self.assertEqual(attachment.filename, "40B49982C61ABEE73F3CA2867031F2E3.jpg")
        self.assertEqual(attachment.content_type, "image/jpeg")
        self.assertTrue(attachment.url.startswith("https://multimedia.nt.qq.com.cn/"))
        self.assertTrue(attachment.anchor.startswith("[附件1]"))
        self.assertEqual(attachment.index, 1)

    def test_ignores_non_image_attachment(self) -> None:
        content = "[群聊的聊天记录]\n[发送者] 玊\n" + _attachment_line(1, kind="视频")
        self.assertEqual(forwarded_image_attachments(content), [])

    def test_image_attachments_includes_forwarded(self) -> None:
        attachments = image_attachments({}, FORWARDED_CONTENT)
        self.assertEqual(len(attachments), 1)
        # 常规消息没有正文可解析时行为不变。
        self.assertEqual(image_attachments({}, "今天好累"), [])

    def test_needs_image_description_until_url_is_consumed(self) -> None:
        self.assertTrue(needs_image_description(FORWARDED_CONTENT))
        done = message_with_image_descriptions(
            FORWARDED_CONTENT, ["一张长截图的描述"], image_attachments({}, FORWARDED_CONTENT)
        )
        self.assertFalse(needs_image_description(done))
        self.assertNotIn("URL:", done)

    def test_description_is_written_under_its_own_attachment(self) -> None:
        attachments = image_attachments({}, TWO_IMAGE_CONTENT)
        merged = message_with_image_descriptions(
            TWO_IMAGE_CONTENT, ["第一张", "第二张"], attachments
        )
        self.assertEqual(merged.count("〔图〕"), 2)
        self.assertNotIn("URL:", merged)
        first_line = merged.index("[附件1]")
        first_note = merged.index("〔图〕第一张")
        second_line = merged.index("[附件2]")
        second_note = merged.index("〔图〕第二张")
        self.assertLess(first_line, first_note)
        self.assertLess(first_note, second_line)
        self.assertLess(second_line, second_note)

    def test_partial_failure_marks_only_the_failed_attachment(self) -> None:
        attachments = image_attachments({}, TWO_IMAGE_CONTENT)
        merged = message_with_image_descriptions(
            TWO_IMAGE_CONTENT, ["第一张", ""], attachments
        )
        self.assertIn("〔图〕第一张", merged)
        self.assertEqual(merged.count(UNRECOGNIZED_IMAGE_MARK), 1)
        self.assertNotIn("URL:", merged)

    def test_all_failed_marks_each_forwarded_attachment_once(self) -> None:
        attachments = image_attachments({}, TWO_IMAGE_CONTENT)
        marked = message_with_unrecognized_images(TWO_IMAGE_CONTENT, attachments)
        self.assertEqual(marked.count(UNRECOGNIZED_IMAGE_MARK), 2)
        self.assertNotIn("URL:", marked)
        self.assertFalse(needs_image_description(marked))
        # 普通消息的全局标记保持原样。
        self.assertEqual(message_with_unrecognized_images("[图片]"), UNRECOGNIZED_IMAGE_MARK)

    def test_rendered_context_keeps_sender_and_note_together(self) -> None:
        attachments = image_attachments({}, TWO_IMAGE_CONTENT)
        merged = message_with_image_descriptions(
            TWO_IMAGE_CONTENT, ["第一张", "第二张"], attachments
        )
        rendered = format_message_content(merged)
        self.assertIn("[发送者] 玊", rendered)
        self.assertLess(rendered.index("[发送者] 玊"), rendered.index("〔图〕第一张"))
        self.assertLess(rendered.index("〔图〕第一张"), rendered.index("〔图〕第二张"))


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        timezone="Asia/Shanghai",
        ack_after_seconds=5.0,
        allowed_group_openids=frozenset(),
        recent_messages=30,
        max_context_chars=4000,
        vision_recent_messages=20,
        vision_max_images_per_request=9,
        vision_prewarm=True,
        vision_prewarm_wait_seconds=5.0,
        vision_max_image_bytes=1024 * 1024,
        image_cache_enabled=False,
        image_cache_max_rows=0,
        face_tags=False,
        asr_max_per_request=0,
        active_window_seconds=0,
        active_followup_seconds=0,
        reply_chunk_chars=900,
        care_enabled=False,
        stickers_enabled=False,
        sticker_dir="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Vision:
    def __init__(self, text: str = "一张 1224x1613 的长截图，内容是课程表") -> None:
        self.text = text
        self.calls: list[str] = []

    async def __call__(self, image_ref, hint=""):
        self.calls.append(hint)
        return self.text


class _FakeLLM:
    def __init__(self, vision: _Vision) -> None:
        self.describe_image = vision
        self.intent_calls = 0
        self.topics: list[str] = []

    async def judge_active_reply(self, **kwargs):
        return {"reply": False, "reason": "test"}

    async def generate_active_reply(self, **kwargs):
        return ""

    async def parse_intent(self, **kwargs):
        self.intent_calls += 1
        self.topics.append(kwargs.get("topic_context", ""))
        return {"action": "chat"}


def _message(
    *,
    message_id: str = "forward-1",
    content: str = FORWARDED_CONTENT,
    mentions: list[dict] | None = None,
):
    return SimpleNamespace(
        group_openid=GROUP,
        id=message_id,
        msg_seq=1,
        content=content,
        timestamp="2026-11-04T18:46:02+08:00",
        username="CL",
        member_role="owner",
        author=SimpleNamespace(member_openid="openid-CL"),
        # 转发记录没有 attachments，图片只在正文里。
        raw={"mentions": list(mentions or [])},
    )


class ForwardedPrewarmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "forwarded.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _client(self, vision: _Vision, **overrides) -> MaodouClient:
        client = MaodouClient(
            settings=_settings(**overrides), store=self.store, llm=_FakeLLM(vision)
        )

        async def noop(*args, **kwargs) -> None:
            return None

        client._maybe_care = noop
        return client

    def _stub_fetch(self, calls: list[str] | None = None):
        async def fake_fetch(url, *, max_bytes):
            if calls is not None:
                calls.append(url)
            return JPEG_BYTES + url.encode(), "image/jpeg"

        original = bot_module.fetch_image_bytes
        bot_module.fetch_image_bytes = fake_fetch
        self.addCleanup(setattr, bot_module, "fetch_image_bytes", original)

    def _wire_chat(self, client: MaodouClient, sent) -> None:
        async def handle_intent(**kwargs):
            return "看到啦"

        async def post_group_message(**kwargs):
            sent.append((kwargs["msg_seq"], kwargs["content"]))

        client._handle_intent = handle_intent
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._identity_context = lambda group, requester: SimpleNamespace(
            prompt=lambda: ""
        )
        client._memory_context = lambda group: ""
        client._summaries_context = lambda group: ""
        client._remember_bot_reply = lambda *args, **kwargs: None
        client._queue_image_backfill = lambda *args, **kwargs: None

    def test_forwarded_image_is_prewarmed_without_at(self) -> None:
        async def scenario() -> None:
            fetched: list[str] = []
            self._stub_fetch(fetched)
            vision = _Vision()
            client = self._client(vision)

            await client.on_group_message_create(_message())
            await client._wait_image_prewarm(
                client._image_prewarm_events(GROUP), timeout=5
            )

            self.assertEqual(len(fetched), 1)
            self.assertEqual(len(vision.calls), 1)
            stored = self.store.get_message(GROUP, "forward-1:1")
            assert stored is not None
            self.assertIn("〔图〕一张 1224x1613 的长截图", stored.content)
            self.assertIn("[发送者] 玊", stored.content)
            self.assertNotIn("URL:", stored.content)
            self.assertFalse(needs_image_description(stored.content, stored.raw))
            self.assertEqual(client.llm.intent_calls, 0)

        asyncio.run(scenario())

    def test_later_at_sees_the_forwarded_image(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            client = self._client(vision)
            sent: list[tuple[int, str]] = []
            self._wire_chat(client, sent)

            await client.on_group_message_create(_message())
            await client._wait_image_prewarm(
                client._image_prewarm_events(GROUP), timeout=5
            )

            at_message = _message(
                message_id="at-1",
                content="@小毛豆 总结一下",
                mentions=[{"bot": True, "is_you": True, "id": "bot-1"}],
            )
            await asyncio.wait_for(
                client._process_at_message(at_message, "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )

            self.assertEqual(len(vision.calls), 1, "不能重复调用视觉模型")
            self.assertEqual(sent, [(1, "看到啦")])
            self.assertTrue(client.llm.topics)
            topic = client.llm.topics[0]
            self.assertIn("〔图〕一张 1224x1613 的长截图", topic)
            self.assertIn("[发送者] 玊", topic)

        asyncio.run(scenario())

    def test_pending_batch_picks_up_forwarded_image(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            client = self._client(_Vision())
            client._save_message(_message(), "GROUP_MESSAGE_CREATE")

            batch = client._pending_image_batch(
                GROUP, exclude_key="other:1", budget=9
            )
            self.assertEqual(len(batch), 1)
            stored, attachments = batch[0]
            self.assertEqual(stored.message_key, "forward-1:1")
            self.assertEqual(len(attachments), 1)
            self.assertEqual(
                attachments[0].filename, "40B49982C61ABEE73F3CA2867031F2E3.jpg"
            )

        asyncio.run(scenario())

    def test_described_forwarded_image_is_not_queued_again(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            client = self._client(_Vision())
            message = _message()
            client._save_message(message, "GROUP_MESSAGE_CREATE")
            stored = self.store.get_message(GROUP, "forward-1:1")
            assert stored is not None
            attachments = image_attachments(stored.raw, stored.content)
            self.store.update_message_content(
                GROUP,
                "forward-1:1",
                message_with_image_descriptions(
                    stored.content, ["已经看过了"], attachments
                ),
            )

            self.assertFalse(client._prewarm_image_message(message))
            self.assertEqual(
                client._pending_image_batch(GROUP, exclude_key="other:1", budget=9),
                [],
            )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
