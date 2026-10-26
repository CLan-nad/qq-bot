from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx2

import maodou.bot as bot_module
import maodou.images as images_module
import maodou.llm as llm_module
from maodou.images import (
    fetch_image_bytes,
    format_message_content,
    image_attachments,
    message_with_image_descriptions,
    message_with_unrecognized_images,
    needs_image_description,
    normalize_face_tags,
    sniff_image_content_type,
    split_image_notes,
    strip_image_placeholders,
    to_data_url,
)
from maodou.llm import MaodouLLM
from maodou.store import ChatStore


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


def _fake_llm() -> tuple[MaodouLLM, _FakeCompletions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        vision_model="deepseek-flash",
        reply_max_chars=900,
    )
    llm = MaodouLLM(settings)
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
        self.assertFalse(needs_image_description("[无文字内容]"))
        self.assertFalse(needs_image_description(normalize_face_tags(STICKER_TAG)))

    def test_needs_image_description_falls_back_to_raw_attachments(self) -> None:
        # 表情标签入库前就解码了，content 里只剩 [大胆]，得靠 raw 认出这条还带图。
        decoded = normalize_face_tags(STICKER_TAG)
        self.assertEqual(decoded, "[大胆]")
        self.assertTrue(needs_image_description(decoded, SINGLE_IMAGE_RAW))
        self.assertFalse(needs_image_description(decoded, {}))
        self.assertFalse(needs_image_description(decoded, None))
        self.assertFalse(needs_image_description("[大胆]\n[图片内容：一只兔子]", SINGLE_IMAGE_RAW))
        self.assertFalse(needs_image_description("[大胆]\n[图片未识别]", SINGLE_IMAGE_RAW))
        # 纯文字消息不该被误判成待识别。
        self.assertFalse(needs_image_description("今天好累[头秃]", {"attachments": []}))

    def test_normalize_face_tags(self) -> None:
        self.assertEqual(normalize_face_tags(STICKER_TAG), "[大胆]")
        # 系统小表情解出来的名字没有方括号，统一补上，别让它混进句子里当普通词。
        self.assertEqual(normalize_face_tags(SMALL_FACE_TAG), "[头秃]")
        self.assertEqual(normalize_face_tags(EMPTY_FACE_TAG), "[表情]")
        self.assertEqual(
            normalize_face_tags('<faceType=6,faceId="0",ext="no!base64">'),
            "[表情]",
        )
        # ext 整个缺失时也要能兜住，不能把标签原样留在上下文里。
        self.assertEqual(normalize_face_tags('<faceType=1,faceId="267">'), "[表情]")
        self.assertEqual(normalize_face_tags('<faceType=1,faceId="267" >'), "[表情]")
        self.assertEqual(normalize_face_tags(""), "")
        self.assertEqual(normalize_face_tags("哈哈" + STICKER_TAG), "哈哈[大胆]")
        self.assertEqual(
            normalize_face_tags(SMALL_FACE_TAG + "今天好累" + SMALL_FACE_TAG),
            "[头秃]今天好累[头秃]",
        )

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
        self.assertEqual(call["model"], "deepseek-v4-pro")
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


class FaceTagStorageTests(unittest.TestCase):
    """表情乱码既不能进库，也不能留在历史消息里。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "face.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, store: ChatStore, content: str, *, message_id: str) -> None:
        store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content=content,
            timestamp="2026-10-14T21:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_save_message_decodes_face_tags(self) -> None:
        store = ChatStore(self.db_path)
        try:
            self._save(store, SMALL_FACE_TAG, message_id="face-1")
            self._save(store, "好累啊" + STICKER_TAG, message_id="face-2")
            contents = [m.content for m in store.recent_messages(GROUP, 10)]
        finally:
            store.close()
        self.assertEqual(contents, ["[头秃]", "好累啊[大胆]"])

    def test_legacy_rows_are_decoded_on_startup(self) -> None:
        store = ChatStore(self.db_path)
        # 绕过 save_message，模拟老版本直接存进去的乱码。
        with store._conn:
            store._conn.execute(
                "INSERT INTO messages (message_key, message_id, msg_seq, group_openid,"
                " member_openid, username, role, content, timestamp, source, raw_json)"
                " VALUES ('legacy:1', 'legacy', '1', ?, 'openid-CL', 'CL', 'member', ?, '',"
                " 'GROUP_MESSAGE_CREATE', '{}')",
                (GROUP, "好累啊" + SMALL_FACE_TAG),
            )
        store.close()

        reopened = ChatStore(self.db_path)
        try:
            contents = [m.content for m in reopened.recent_messages(GROUP, 10)]
            self.assertEqual(contents, ["好累啊[头秃]"])
            # FTS 索引由触发器同步，解码后的表情名要能被搜到。
            self.assertEqual(len(reopened.search_messages(GROUP, "头秃")), 1)
            # 再跑一次迁移必须是幂等的。
            self.assertEqual(reopened._normalize_legacy_face_tags(), 0)
        finally:
            reopened.close()


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


def _scripted_llm(contents: list[str]) -> tuple[MaodouLLM, _ScriptedCompletions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        vision_model="deepseek-flash",
        reply_max_chars=900,
    )
    llm = MaodouLLM(settings)
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
        self.assertEqual(
            [call["reasoning_effort"] for call in completions.calls],
            ["low", "high", "high"],
        )

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
                self.efforts: list[str] = []

            async def create(self, **kwargs):
                self.calls += 1
                self.efforts.append(kwargs.get("reasoning_effort"))
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
        self.assertEqual(flaky.efforts, ["low", "high"])


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
            image_cache_enabled=False,
            image_cache_max_rows=0,
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

        client = bot_module.MaodouClient(
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
        client = bot_module.MaodouClient(
            settings=self._settings(),
            store=self.store,
            llm=SimpleNamespace(),
        )
        budget = asyncio.run(
            client._describe_recent_images(GROUP, exclude_key="", budget=3)
        )
        # 系统小表情没有图片附件，不该浪费一次视觉调用；但入库时乱码已经解码成可读文本。
        self.assertEqual(budget, 3)
        self.assertEqual(self.store.recent_messages(GROUP, 5)[0].content, "[头秃]")


class ImageNoteRenderingTests(unittest.TestCase):
    """入库仍用 [图片内容：…] 标记，进 prompt 时拆成独立〔图〕行。

    线上原样回放：小毛豆看图认出了白桃乌龙雪糕，被问起却说「图片我是真看不了」——
    因为描述被拼成了一行用户文字，它以为那是对方打出来的。
    """

    def test_split_image_notes_pulls_notes_out_of_the_text(self) -> None:
        text, notes = split_image_notes("八块，我去\n[图片内容：手里拿着白桃乌龙雪糕]")
        self.assertEqual(text, "八块，我去")
        self.assertEqual(notes, ["手里拿着白桃乌龙雪糕"])

    def test_split_image_notes_keeps_text_only_content(self) -> None:
        text, notes = split_image_notes("就是普通一句话")
        self.assertEqual(text, "就是普通一句话")
        self.assertEqual(notes, [])

    def test_split_image_notes_handles_several_images(self) -> None:
        text, notes = split_image_notes(
            "[图片内容：一只橘猫]\n[图片内容：桌面有点乱]"
        )
        self.assertEqual(text, "")
        self.assertEqual(notes, ["一只橘猫", "桌面有点乱"])

    def test_split_image_notes_collapses_multiline_descriptions(self) -> None:
        text, notes = split_image_notes("看图\n[图片内容：第一行\n第二行]")
        self.assertEqual(text, "看图")
        self.assertEqual(notes, ["第一行 第二行"])

    def test_format_message_content_keeps_note_on_its_own_line(self) -> None:
        rendered = format_message_content(
            "八块，我去\n[图片内容：手里拿着白桃乌龙雪糕]", indent="    "
        )
        lines = rendered.splitlines()
        self.assertEqual(lines[0], "八块，我去")
        self.assertEqual(lines[1], "    〔图〕手里拿着白桃乌龙雪糕")
        self.assertNotIn("[图片内容：", rendered)

    def test_format_message_content_pure_image_message(self) -> None:
        rendered = format_message_content("[图片内容：一只橘猫]")
        self.assertEqual(rendered, "〔图〕一只橘猫")

    def test_pending_placeholder_becomes_readable(self) -> None:
        # 还没被识别（那几条 [image/jpeg] 就是这种）也要让人看得懂
        self.assertEqual(
            format_message_content("[image/jpeg]"),
            "〔图〕（这张还没识别）",
        )
        self.assertEqual(
            format_message_content("[图片]"),
            "〔图〕（这张还没识别）",
        )

    def test_unreadable_mark_becomes_readable(self) -> None:
        self.assertEqual(
            format_message_content("看这个\n[图片未识别]"),
            "看这个\n〔图〕（没能看清）",
        )

    def test_describe_image_output_is_single_line(self) -> None:
        llm, _ = _fake_llm()

        class _Multiline:
            async def create(self, **kwargs):
                message = SimpleNamespace(content="第一行\n\n第二行  第三行")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=_Multiline()))
        description = asyncio.run(llm.describe_image("https://example.com/a.jpg"))
        self.assertEqual(description, "第一行 第二行 第三行")


if __name__ == "__main__":
    unittest.main()
