"""图片描述缓存：同一张图重复出现时不再二次调用视觉模型。

表情包、课表这类图会被反复带进上下文（懒识别、历史检索都会碰到），
按图片字节的 sha256 缓存描述既省 token，也省一次几秒的外部调用。
"""
from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.images import ImageAttachment
from maodou.store import ChatStore


GROUP = "test-group"


def _attachment(name: str = "a.jpg") -> ImageAttachment:
    return ImageAttachment(
        url="https://example.com/" + name,
        filename=name,
        content_type="image/jpeg",
    )


class ImageCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "cache.sqlite3"))
        self.calls: list[str] = []

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, *, cache_enabled: bool = True) -> MaodouClient:
        settings = SimpleNamespace(
            vision_max_image_bytes=1024 * 1024,
            image_cache_enabled=cache_enabled,
            image_cache_max_rows=100,
            allowed_group_openids=frozenset(),
        )

        async def describe_image(image_ref, hint=""):
            self.calls.append(image_ref)
            return "一只橘猫趴在键盘上"

        client = MaodouClient(
            settings=settings,
            store=self.store,
            llm=SimpleNamespace(describe_image=describe_image),
        )
        return client

    def _patch_fetch(self, payload: bytes) -> None:
        async def fake_fetch(url, *, max_bytes):
            return payload, "image/jpeg"

        self.addCleanup(setattr, bot_module, "fetch_image_bytes", bot_module.fetch_image_bytes)
        bot_module.fetch_image_bytes = fake_fetch

    def _describe(self, client, payload: bytes) -> list[str]:
        self._patch_fetch(payload)
        return asyncio.run(client._describe_attachments([_attachment()]))

    def test_second_hit_skips_vision_model(self) -> None:
        client = self._client()
        first = self._describe(client, b"\xff\xd8\xfffirst-image")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(first[0])

        second = self._describe(client, b"\xff\xd8\xfffirst-image")
        self.assertEqual(len(self.calls), 1, "同一张图不该再调视觉模型")
        self.assertEqual(second, first)
        stats = self.store.image_cache_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)

    def test_different_bytes_are_different_entries(self) -> None:
        client = self._client()
        self._describe(client, b"\xff\xd8\xffcat")
        self._describe(client, b"\xff\xd8\xffdog")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.store.image_cache_stats()["entries"], 2)

    def test_digest_is_sha256_of_bytes(self) -> None:
        client = self._client()
        payload = b"\xff\xd8\xff-hash-me"
        self._describe(client, payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(self.store.get_image_description(digest), "一只橘猫趴在键盘上")

    def test_empty_description_is_not_cached(self) -> None:
        client = self._client()

        async def empty_describe(image_ref, hint=""):
            self.calls.append(image_ref)
            return ""

        client.llm = SimpleNamespace(describe_image=empty_describe)
        self._patch_fetch(b"\xff\xd8\xffblank")
        first = asyncio.run(client._describe_attachments([_attachment()]))
        second = asyncio.run(client._describe_attachments([_attachment()]))
        self.assertEqual(first, [""])
        self.assertEqual(len(self.calls), 2, "识别失败要留下次重试的机会")
        self.assertEqual(self.store.image_cache_stats()["entries"], 0)

    def test_cache_disabled_calls_every_time(self) -> None:
        client = self._client(cache_enabled=False)
        self._describe(client, b"\xff\xd8\xffsame")
        self._describe(client, b"\xff\xd8\xffsame")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.store.image_cache_stats()["entries"], 0)

    def test_store_evicts_over_max_rows(self) -> None:
        for index in range(3):
            self.store.save_image_description(
                digest=f"digest-{index}",
                description=f"desc-{index}",
                content_type="image/jpeg",
                byte_size=10,
                max_rows=2,
            )
        self.assertEqual(self.store.image_cache_stats()["entries"], 2)

    def test_store_ignores_blank_inputs(self) -> None:
        self.store.save_image_description(digest="", description="x", max_rows=10)
        self.store.save_image_description(digest="d", description="   ", max_rows=10)
        self.assertEqual(self.store.image_cache_stats()["entries"], 0)
        self.assertEqual(self.store.get_image_description(""), "")


if __name__ == "__main__":
    unittest.main()
