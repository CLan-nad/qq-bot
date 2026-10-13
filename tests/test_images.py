from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx2

import xiaoxia.bot as bot_module
import xiaoxia.images as images_module
import xiaoxia.llm as llm_module
from xiaoxia.images import (
    fetch_image_bytes,
    image_attachments,
    message_with_image_descriptions,
    message_with_unrecognized_images,
    needs_image_description,
    normalize_face_tags,
    sniff_image_content_type,
    strip_image_placeholders,
    to_data_url,
)
from xiaoxia.llm import XiaoxiaLLM
from xiaoxia.store import ChatStore


GROUP = "test-group"
IMAGE_RAW = {
    "attachments": [
        {
            "content": "[图片]",
            "content_type": "image/jpeg",
            "filename": "cat.jpg",
            "url": "https://example.com/cat.jpg",
        },
        {"content_type": "image/jpeg", "filename": "broken.jpg", "url": "file:///tmp/a.jpg"},
        {"content_type": "video/mp4", "url": "https://example.com/a.mp4"},
        "not-a-dict",
    ]
}
SINGLE_IMAGE_RAW = {"attachments": [dict(IMAGE_RAW["attachments"][0])]}
RED_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
STICKER_TAG = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiW+Wkp+iDhl0ifQ==">'
SMALL_FACE_TAG = '<faceType=1,faceId="267",ext="eyJ0ZXh0Ijoi5aS056eDIn0=">'
EMPTY_FACE_TAG = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="一只橘猫趴在键盘上。")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_llm() -> tuple[XiaoxiaLLM, _FakeCompletions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-chat",
        vision_model="deepseek-flash",
        reply_max_chars=900,
    )
    llm = XiaoxiaLLM(settings)
    completions = _FakeCompletions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


class ImageHelperTests(unittest.TestCase):
    def test_image_attachments_filters_unsupported_entries(self) -> None:
        attachments = image_attachments(IMAGE_RAW)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].filename, "cat.jpg")
        self.assertEqual(attachments[0].content_type, "image/jpeg")

    def test_message_with_image_descriptions_replaces_placeholder(self) -> None:
        merged = message_with_image_descriptions(
            "[图片]",
            ["一只橘猫", "桌面有点乱"],
        )
        self.assertEqual(merged, "[图片内容：一只橘猫]\n[图片内容：桌面有点乱]")

    def test_message_with_image_descriptions_keeps_text(self) -> None:
        merged = message_with_image_descriptions(
            "你看这个 [图片] 好笑吗",
            ["一只柴犬在跳舞"],
        )
        self.assertEqual(merged, "你看这个 好笑吗\n[图片内容：一只柴犬在跳舞]")

    def test_message_without_description_keeps_placeholder(self) -> None:
        self.assertEqual(message_with_image_descriptions("[image/jpeg]", []), "[图片]")
        self.assertEqual(strip_image_placeholders("[image/png] [图片]"), "")

    def test_needs_image_description(self) -> None:
        self.assertTrue(needs_image_description("[图片]"))
        self.assertTrue(needs_image_description("[image/jpeg]"))
        self.assertTrue(needs_image_description(STICKER_TAG))
        self.assertTrue(needs_image_description(SMALL_FACE_TAG))
        self.assertFalse(needs_image_description("[图片内容：一只猫]"))
        self.assertFalse(needs_image_description("[图片未识别]"))
        self.assertFalse(needs_image_description("[non-text message]"))
        self.assertFalse(needs_image_description(normalize_face_tags(STICKER_TAG)))

    def test_normalize_face_tags(self) -> None:
        self.assertEqual(normalize_face_tags(STICKER_TAG), "[大胆]")
        self.assertEqual(normalize_face_tags(SMALL_FACE_TAG), "头秃")
        self.assertEqual(normalize_face_tags(EMPTY_FACE_TAG), "[表情]")
        self.assertEqual(
            normalize_face_tags('<faceType=6,faceId="0",ext="no!base64">'),
            "[表情]",
        )
        self.assertEqual(normalize_face_tags(""), "")
        self.assertEqual(normalize_face_tags("哈哈" + STICKER_TAG), "哈哈[大胆]")

    def test_sticker_message_gets_image_description(self) -> None:
        self.assertEqual(strip_image_placeholders(STICKER_TAG), "[大胆]")
        merged = message_with_image_descriptions(
            STICKER_TAG, ["一只白色垂耳兔闭着眼抱爪"]
        )
        self.assertEqual(merged, "[大胆]\n[图片内容：一只白色垂耳兔闭着眼抱爪]")

    def test_unrecognized_mark_stops_retry(self) -> None:
        marked = message_with_unrecognized_images("[image/jpeg]")
        self.assertEqual(marked, "[图片未识别]")
        self.assertFalse(needs_image_description(marked))
        self.assertEqual(
            message_with_unrecognized_images("你看 [图片]"),
            "你看\n[图片未识别]",
        )

    def test_sniff_image_content_type(self) -> None:
        png = base64.b64decode(RED_PNG_BASE64)
        self.assertEqual(sniff_image_content_type(png), "image/png")
        self.assertEqual(sniff_image_content_type(b"\xff\xd8\xff\xe0jpeg"), "image/jpeg")
        self.assertEqual(sniff_image_content_type(b"GIF89a..."), "image/gif")
        self.assertEqual(sniff_image_content_type(b"RIFF????WEBPVP8 "), "image/webp")
        self.assertIsNone(sniff_image_content_type(b"<svg></svg>"))

    def test_to_data_url(self) -> None:
        data_url = to_data_url(b"abc", "image/png")
        self.assertEqual(data_url, "data:image/png;base64,YWJj")


class ImageDownloadTests(unittest.TestCase):
    def _patch_transport(self, handler):
        original = images_module.httpx2.AsyncClient
        transport = httpx2.MockTransport(handler)
        images_module.httpx2.AsyncClient = lambda **kwargs: original(
            transport=transport, **kwargs
        )
        return original

    def test_fetch_image_bytes_uses_sniffed_type(self) -> None:
        png = base64.b64decode(RED_PNG_BASE64)

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, content=png, headers={"content-type": "application/octet-stream"})

        original = self._patch_transport(handler)
        try:
            result = asyncio.run(fetch_image_bytes("https://example.com/a.bin", max_bytes=4096))
        finally:
            images_module.httpx2.AsyncClient = original

        self.assertIsNotNone(result)
        data, content_type = result
        self.assertEqual(data, png)
        self.assertEqual(content_type, "image/png")

    def test_fetch_image_bytes_rejects_large_or_invalid_payload(self) -> None:
        png = base64.b64decode(RED_PNG_BASE64)

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/big.png":
                return httpx2.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx2.Response(200, content=b"<svg></svg>", headers={"content-type": "image/svg+xml"})

        original = self._patch_transport(handler)
        try:
            too_big = asyncio.run(fetch_image_bytes("https://example.com/big.png", max_bytes=4))
            invalid = asyncio.run(fetch_image_bytes("https://example.com/a.svg"))
            bad_scheme = asyncio.run(fetch_image_bytes("file:///tmp/a.png"))
        finally:
            images_module.httpx2.AsyncClient = original

        self.assertIsNone(too_big)
        self.assertIsNone(invalid)
        self.assertIsNone(bad_scheme)

    def test_fetch_image_bytes_handles_http_error(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(403, text="forbidden")

        original = self._patch_transport(handler)
        try:
            result = asyncio.run(fetch_image_bytes("https://example.com/a.png"))
        finally:
            images_module.httpx2.AsyncClient = original

        self.assertIsNone(result)


class VisionModelTests(unittest.TestCase):
    def test_describe_image_sends_image_content_part(self) -> None:
        llm, completions = _fake_llm()
        description = asyncio.run(
            llm.describe_image("https://example.com/cat.jpg", hint="看这张")
        )
        self.assertEqual(description, "一只橘猫趴在键盘上。")
        self.assertEqual(len(completions.calls), 1)
        call = completions.calls[0]
        self.assertEqual(call["model"], "deepseek-flash")
        content = call["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("看这张", content[0]["text"])
        self.assertEqual(
            content[1]["image_url"]["url"],
            "https://example.com/cat.jpg",
        )

    def test_chat_still_accepts_plain_text_prompt(self) -> None:
        llm, completions = _fake_llm()
        asyncio.run(
            llm._chat(
                system_prompt="sys",
                user_prompt="hello",
                temperature=0.2,
                max_tokens=10,
            )
        )
        call = completions.calls[0]
        self.assertEqual(call["model"], "deepseek-chat")
        self.assertEqual(call["messages"][1]["content"], "hello")


class ImageStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save_image_message(self) -> None:
        self.store.save_message(
            message_id="img-1",
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content="",
            timestamp="2026-10-12T21:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw=SINGLE_IMAGE_RAW,
        )

    def test_save_and_enrich_image_message(self) -> None:
        self._save_image_message()
        stored = self.store.recent_messages_with_raw(GROUP, 10)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].content, "[image/jpeg]")
        self.assertTrue(needs_image_description(stored[0].content))
        self.assertEqual(len(image_attachments(stored[0].raw)), 1)

        enriched = message_with_image_descriptions(stored[0].content, ["一只橘猫"])
        self.assertTrue(
            self.store.update_message_content(GROUP, stored[0].message_key, enriched)
        )
        self.assertEqual(
            self.store.recent_messages(GROUP, 10)[0].content,
            "[图片内容：一只橘猫]",
        )
        self.assertEqual(len(self.store.search_messages(GROUP, "橘猫")), 1)

    def test_update_message_content_ignores_unknown_key(self) -> None:
        self._save_image_message()
        self.assertFalse(self.store.update_message_content(GROUP, "missing", "[图片内容：x]"))
        self.assertFalse(self.store.update_message_content(GROUP, "", "[图片内容：x]"))


class _ScriptedCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        message = SimpleNamespace(content=self.contents[index])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


def _scripted_llm(contents: list[str]) -> tuple[XiaoxiaLLM, _ScriptedCompletions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-chat",
        vision_model="deepseek-flash",
        reply_max_chars=900,
    )
    llm = XiaoxiaLLM(settings)
    completions = _ScriptedCompletions(contents)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


class VisionRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        original = llm_module.VISION_RETRY_DELAY_SECONDS
        llm_module.VISION_RETRY_DELAY_SECONDS = 0.0
        self.addCleanup(setattr, llm_module, "VISION_RETRY_DELAY_SECONDS", original)

    def test_describe_image_uses_bigger_token_budget(self) -> None:
        self.assertGreater(llm_module.VISION_MAX_TOKENS, 700)

    def test_describe_image_retries_after_empty_content(self) -> None:
        llm, completions = _scripted_llm(["", "   ", "a cat on a keyboard"])
        description = asyncio.run(
            llm.describe_image("https://example.com/cat.jpg")
        )
        self.assertEqual(description, "a cat on a keyboard")
        self.assertEqual(len(completions.calls), 3)
        for call in completions.calls:
            self.assertEqual(call["max_tokens"], llm_module.VISION_MAX_TOKENS)
            self.assertEqual(call["model"], "deepseek-flash")

    def test_describe_image_gives_up_after_max_attempts(self) -> None:
        llm, completions = _scripted_llm([""])
        description = asyncio.run(
            llm.describe_image("https://example.com/cat.jpg")
        )
        self.assertEqual(description, "")
        self.assertEqual(
            len(completions.calls), llm_module.VISION_DESCRIBE_ATTEMPTS
        )

    def test_describe_image_retries_after_exception(self) -> None:
        class _Flaky:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom")
                message = SimpleNamespace(content="a course timetable")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        llm, _ = _scripted_llm([""])
        flaky = _Flaky()
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=flaky))
        description = asyncio.run(
            llm.describe_image("https://example.com/a.jpg")
        )
        self.assertEqual(description, "a course timetable")
        self.assertEqual(flaky.calls, 2)


class StickerDescribeTests(unittest.TestCase):
    """表情包（<faceType=6,...> + 图片附件）要能被懒加载识别出来。"""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            vision_recent_messages=20,
            vision_max_images_per_request=3,
            vision_max_image_bytes=1024 * 1024,
            allowed_group_openids=frozenset(),
        )

    def _save(self, message_id: str, content: str, raw: dict) -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content=content,
            timestamp="2026-10-13T16:56:04+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw=raw,
        )

    def test_recent_sticker_message_gets_described(self) -> None:
        raw = {
            "attachments": [
                {
                    "content_type": "image/jpeg",
                    "filename": "sticker.jpg",
                    "url": "https://example.com/sticker.jpg",
                }
            ]
        }
        self._save("sticker-1", STICKER_TAG, raw)

        async def describe_image(url, hint=""):
            return "一只白色垂耳兔闭着眼抱爪"

        client = bot_module.XiaoxiaClient(
            settings=self._settings(),
            store=self.store,
            llm=SimpleNamespace(describe_image=describe_image),
        )

        async def fake_fetch(url, *, max_bytes):
            return b"\xff\xd8\xff\xe0stub", "image/jpeg"

        original_fetch = bot_module.fetch_image_bytes
        bot_module.fetch_image_bytes = fake_fetch
        try:
            budget = asyncio.run(
                client._describe_recent_images(GROUP, exclude_key="", budget=3)
            )
        finally:
            bot_module.fetch_image_bytes = original_fetch

        self.assertEqual(budget, 2)
        self.assertEqual(
            self.store.recent_messages(GROUP, 5)[0].content,
            "[大胆]\n[图片内容：一只白色垂耳兔闭着眼抱爪]",
        )

    def test_small_face_message_without_attachment_is_left_alone(self) -> None:
        self._save("face-1", SMALL_FACE_TAG, {})
        client = bot_module.XiaoxiaClient(
            settings=self._settings(),
            store=self.store,
            llm=SimpleNamespace(),
        )
        budget = asyncio.run(
            client._describe_recent_images(GROUP, exclude_key="", budget=3)
        )
        self.assertEqual(budget, 3)
        self.assertEqual(
            self.store.recent_messages(GROUP, 5)[0].content, SMALL_FACE_TAG
        )


if __name__ == "__main__":
    unittest.main()
