"""DeepSeek 原生 OCR：触发条件、结果清洗、缓存与看图合并。"""
from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.images import ImageAttachment, image_dimensions
from maodou.ocr import (
    OcrResult,
    clean_ocr_text,
    compose_image_note,
    description_mentions_text,
    flatten_ocr_text,
    ocr_image,
    ocr_precheck,
    parse_ocr_payload,
)
from maodou.store import ChatStore


def _png(width: int, height: int) -> bytes:
    """只造一个能被 image_dimensions 读出来的 PNG 头。"""
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00" * 8
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"payload"
    )


def _attachment(name: str = "shot.png") -> ImageAttachment:
    return ImageAttachment(
        url="https://example.com/" + name,
        filename=name,
        content_type="image/png",
    )


class OcrParsingTests(unittest.TestCase):
    def test_clean_strips_layout_coordinates(self) -> None:
        raw = "【标题】[[52, 40, 948, 96]]\n\n正文第一行\n\n\n\n正文第二行"
        self.assertEqual(clean_ocr_text(raw), "【标题】\n\n正文第一行\n\n正文第二行")

    def test_parse_joins_pages_and_truncates(self) -> None:
        payload = {
            "page_count": 2,
            "pages": [
                {"index": 0, "text": "第一页 " + "啊" * 20},
                {"index": 1, "text": "第二页 " + "呀" * 20},
            ],
        }
        text = parse_ocr_payload(payload, max_chars=12)
        self.assertTrue(text.startswith("第一页"))
        self.assertLessEqual(len(text), 13)
        self.assertTrue(text.endswith("…"))

    def test_table_is_compacted(self) -> None:
        raw = (
            "<table>\n<tr>\n<td>时间</td>\n<td>周一</td>\n</tr>\n"
            "<tr><td>08:00</td><td>高数</td></tr>\n</table>"
        )
        self.assertEqual(clean_ocr_text(raw), "时间 | 周一\n08:00 | 高数")

    def test_parse_skips_blank_pages(self) -> None:
        payload = {"pages": [{"text": "  "}, {"text": "有字"}]}
        self.assertEqual(parse_ocr_payload(payload), "有字")

    def test_parse_tolerates_garbage(self) -> None:
        self.assertEqual(parse_ocr_payload(None), "")
        self.assertEqual(parse_ocr_payload({"pages": "nope"}), "")
        self.assertEqual(parse_ocr_payload({}), "")


class OcrTriggerTests(unittest.TestCase):
    def test_user_asking_to_read_triggers(self) -> None:
        for hint in ("把上面的字念一下", "帮我提取一下表格", "这截图写了什么", "读一下原文"):
            self.assertTrue(ocr_precheck(hint=hint), hint)

    def test_long_screenshot_triggers(self) -> None:
        self.assertTrue(ocr_precheck(width=800, height=2000, content_type="image/png"))

    def test_plain_photo_does_not_trigger(self) -> None:
        self.assertFalse(
            ocr_precheck(width=1200, height=900, byte_size=90_000, content_type="image/jpeg")
        )

    def test_big_png_triggers(self) -> None:
        self.assertTrue(
            ocr_precheck(width=100, height=100, byte_size=500_000, content_type="image/png")
        )

    def test_description_signal(self) -> None:
        self.assertTrue(description_mentions_text("这是一张课表截图，上面有课程名称"))
        self.assertFalse(description_mentions_text("一只橘猫趴在键盘上"))


class OcrComposeTests(unittest.TestCase):
    def test_flatten_protects_note_brackets(self) -> None:
        value = flatten_ocr_text("第一行\n[链接]第二行")
        self.assertNotIn("[", value)
        self.assertNotIn("]", value)
        self.assertIn(" / ", value)

    def test_compose_without_ocr_is_plain_description(self) -> None:
        self.assertEqual(compose_image_note("一只猫", ""), "一只猫")

    def test_compose_keeps_both(self) -> None:
        note = compose_image_note("一张课表截图", "周一 高等数学")
        self.assertIn("一张课表截图", note)
        self.assertIn("图内文字：周一 高等数学", note)


class ImageDimensionTests(unittest.TestCase):
    def test_png(self) -> None:
        self.assertEqual(image_dimensions(_png(800, 2000)), (800, 2000))

    def test_gif(self) -> None:
        data = b"GIF89a" + (320).to_bytes(2, "little") + (240).to_bytes(2, "little")
        self.assertEqual(image_dimensions(data), (320, 240))

    def test_unknown(self) -> None:
        self.assertIsNone(image_dimensions(b"not-an-image"))


class OcrResponseTests(unittest.TestCase):
    def test_ocr_image_parses_success(self) -> None:
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "page_count": 1,
                    "pages": [{"index": 0, "text": "# 课表\n<table><tr><td>数学</td></tr></table>"}],
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                captured["url"] = url
                captured["files"] = kwargs.get("files")
                captured["data"] = kwargs.get("data")
                return FakeResponse()

        original = maodou.ocr.httpx2.AsyncClient if False else None  # noqa: F841
        import maodou.ocr as ocr_module

        saved = ocr_module.httpx2.AsyncClient
        ocr_module.httpx2.AsyncClient = FakeClient
        try:
            result = asyncio.run(
                ocr_image(
                    b"fake-bytes",
                    api_key="k",
                    base_url="https://api.deepseek.com",
                    max_chars=500,
                )
            )
        finally:
            ocr_module.httpx2.AsyncClient = saved

        self.assertTrue(result.ok)
        self.assertIn("课表", result.text)
        self.assertEqual(captured["url"], "https://api.deepseek.com/v1/tools/ocr")
        self.assertEqual(captured["data"], {"purpose": "text_recognition"})
        self.assertIn("file", captured["files"])

    def test_ocr_image_reports_error_payload(self) -> None:
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"error": {"message": "file format is not supported"}}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                return FakeResponse()

        import maodou.ocr as ocr_module

        saved = ocr_module.httpx2.AsyncClient
        ocr_module.httpx2.AsyncClient = FakeClient
        try:
            result = asyncio.run(ocr_image(b"x", api_key="k"))
        finally:
            ocr_module.httpx2.AsyncClient = saved

        self.assertFalse(result.ok)
        self.assertIn("not supported", result.error)

    def test_empty_result_is_retried_once(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            status_code = 200

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    return FakeResponse(
                        {"page_count": 1, "pages": [{"index": 0, "text": ""}]}
                    )
                return FakeResponse(
                    {"page_count": 1, "pages": [{"index": 0, "text": "第二次才有字"}]}
                )

        import maodou.ocr as ocr_module

        saved = ocr_module.httpx2.AsyncClient
        ocr_module.httpx2.AsyncClient = FakeClient
        try:
            result = asyncio.run(ocr_image(b"x", api_key="k"))
        finally:
            ocr_module.httpx2.AsyncClient = saved

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "第二次才有字")
        self.assertEqual(calls["count"], 2)

    def test_bad_format_is_not_retried(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            status_code = 400

            def json(self):
                return {"error": {"message": "file format is not supported"}}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                calls["count"] += 1
                return FakeResponse()

        import maodou.ocr as ocr_module

        saved = ocr_module.httpx2.AsyncClient
        ocr_module.httpx2.AsyncClient = FakeClient
        try:
            result = asyncio.run(ocr_image(b"x", api_key="k"))
        finally:
            ocr_module.httpx2.AsyncClient = saved

        self.assertFalse(result.ok)
        self.assertEqual(calls["count"], 1, "确定性错误不该重试")

    def test_missing_key_is_reported(self) -> None:
        result = asyncio.run(ocr_image(b"x", api_key=""))
        self.assertFalse(result.ok)
        self.assertIn("没有配置", result.error)


class OcrStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "ocr.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_roundtrip_and_hits(self) -> None:
        self.store.save_image_ocr(digest="d1", text="第一行\n第二行")
        self.assertEqual(self.store.get_image_ocr("d1"), "第一行\n第二行")
        self.assertEqual(self.store.get_image_ocr("d1"), "第一行\n第二行")
        row = self.store._conn.execute(
            "SELECT hits, chars FROM image_ocr WHERE digest = ?", ("d1",)
        ).fetchone()
        self.assertEqual(row["hits"], 2)
        self.assertEqual(row["chars"], len("第一行\n第二行"))

    def test_empty_text_not_stored(self) -> None:
        self.store.save_image_ocr(digest="d2", text="   ")
        self.assertEqual(self.store.get_image_ocr("d2"), "")


class DescribeAttachmentsOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "bot.sqlite3"))
        self.ocr_calls: list[bytes] = []
        self.describe_calls = 0

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, description: str = "一张课表截图") -> MaodouClient:
        settings = SimpleNamespace(
            vision_max_image_bytes=1024 * 1024,
            image_cache_enabled=True,
            image_cache_max_rows=100,
            allowed_group_openids=frozenset(),
            llm_api_key="test-key",
            llm_base_url="https://api.deepseek.com",
            search_api_key="",
            search_base_url="",
            ocr_enabled=True,
            ocr_max_chars=1200,
            ocr_timeout_seconds=30.0,
        )
        client = MaodouClient(
            settings=settings,
            store=self.store,
            llm=SimpleNamespace(describe_image=self._describe),
        )
        self._description = description
        return client

    async def _describe(self, image_ref, hint=""):
        self.describe_calls += 1
        return self._description

    def _patch(self, payload: bytes) -> None:
        async def fake_fetch(url, *, max_bytes):
            return payload, "image/png"

        async def fake_ocr(data, **kwargs):
            self.ocr_calls.append(data)
            return OcrResult(ok=True, text="周一 高等数学 A301", page_count=1)

        for name, value in (("fetch_image_bytes", fake_fetch), ("ocr_image", fake_ocr)):
            original = getattr(bot_module, name)
            setattr(bot_module, name, value)
            self.addCleanup(setattr, bot_module, name, original)

    def test_long_screenshot_triggers_ocr_and_merges(self) -> None:
        client = self._client()
        self._patch(_png(800, 2000))
        notes = asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(len(self.ocr_calls), 1)
        self.assertIn("一张课表截图", notes[0])
        self.assertIn("图内文字：周一 高等数学 A301", notes[0])

    def test_ocr_result_is_cached(self) -> None:
        client = self._client()
        payload = _png(800, 2000)
        self._patch(payload)
        asyncio.run(client._describe_attachments([_attachment()]))
        asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(len(self.ocr_calls), 1, "同一张图不该重复 OCR")
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(self.store.get_image_ocr(digest), "周一 高等数学 A301")

    def test_plain_photo_skips_ocr(self) -> None:
        client = self._client(description="一只橘猫趴在键盘上")
        self._patch(_png(1200, 900))
        notes = asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(self.ocr_calls, [])
        self.assertEqual(notes[0], "一只橘猫趴在键盘上")

    def test_description_signal_triggers_ocr(self) -> None:
        client = self._client(description="这是一张成绩单截图，有很多数字")
        self._patch(_png(1200, 900))
        notes = asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(len(self.ocr_calls), 1)
        self.assertIn("图内文字", notes[0])

    def test_ocr_disabled_never_calls(self) -> None:
        client = self._client()
        client.settings.ocr_enabled = False
        self._patch(_png(800, 2000))
        notes = asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(self.ocr_calls, [])
        self.assertEqual(notes[0], "一张课表截图")


if __name__ == "__main__":
    unittest.main()
