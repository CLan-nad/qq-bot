"""图片链路的兜底：下载失败不白烧视觉模型、旧图补识别丢后台、@ 回复不被它拖住。

这两个问题都是从昨天的线上聊天记录里翻出来的：

1. QQ 的图片链接自带时效签名，过一会儿就失效。旧代码在取图失败时把原始 url
   透传给视觉模型，模型那边同样打不开，白花一次调用还只拿回一个空描述，
   于是同一张失效图每次 @ 都要重试一遍；
2. 「补识别最近的旧图」原来是同步 await 在 @ 回复路径上的，一次最多能串着
   识别好几张，每张都要下载 + 一次视觉调用，用户干等十几秒才看到回复。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import xiaoxia.bot as bot_module
from xiaoxia.bot import XiaoxiaClient
from xiaoxia.images import UNRECOGNIZED_IMAGE_MARK, ImageAttachment
from xiaoxia.store import ChatStore


GROUP = "test-group"
OTHER_GROUP = "other-group"
CAT_URL = "https://multimedia.nt.qq.com.cn/cat.jpg?rkey=expired"
JPEG_BYTES = b"\xff\xd8\xff\xe0stub-bytes"
CAT_RAW = {
    "attachments": [
        {"content_type": "image/jpeg", "filename": "cat.jpg", "url": CAT_URL}
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
        vision_max_images_per_request=3,
        vision_max_image_bytes=1024 * 1024,
        image_cache_enabled=False,
        image_cache_max_rows=0,
        face_tags=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _client(store=None, llm=None, **overrides) -> XiaoxiaClient:
    return XiaoxiaClient(
        settings=_settings(**overrides),
        store=store if store is not None else SimpleNamespace(face_book=lambda group: {}),
        llm=llm if llm is not None else SimpleNamespace(),
    )


class _CountingVision:
    """假视觉模型：记下每次被喂进去的引用，方便断言「压根没被叫到」。"""

    def __init__(self, text: str = "一只橘猫趴在键盘上") -> None:
        self.text = text
        self.calls: list[str] = []

    async def describe_image(self, image_ref, hint=""):
        self.calls.append(image_ref)
        return self.text


class _BackfillRecorder:
    """替代 _describe_recent_images：记录调用参数，可用 gate 卡住模拟耗时。"""

    def __init__(self, gate: asyncio.Event | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.gate = gate
        self.error = error

    async def __call__(self, group_openid, *, exclude_key, budget):
        self.calls.append((group_openid, exclude_key, budget))
        if self.error is not None:
            raise self.error
        if self.gate is not None:
            await self.gate.wait()
        return budget


def _attachment(url: str = CAT_URL) -> ImageAttachment:
    return ImageAttachment(url=url, content_type="image/jpeg", filename="cat.jpg")


class _LoopTestCase(unittest.TestCase):
    """botpy.Client.__init__ 会取当前事件循环，没有就直接抛，所以先建一个。"""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()


class ImageDownloadFailureTests(_LoopTestCase):
    """图取不下来时：跳过视觉模型、留空描述、消息标成 [图片未识别] 不再重试。"""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))
        self.vision = _CountingVision()
        self.client = _client(
            store=self.store,
            llm=SimpleNamespace(describe_image=self.vision.describe_image),
        )
        self.fetched_urls: list[str] = []

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        super().tearDown()

    def _stub_fetch(self, result):
        async def fake_fetch(url, *, max_bytes):
            self.fetched_urls.append(url)
            return result

        original = bot_module.fetch_image_bytes
        bot_module.fetch_image_bytes = fake_fetch
        self.addCleanup(setattr, bot_module, "fetch_image_bytes", original)

    def _save(self, message_id: str, content: str, raw: dict) -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content=content,
            timestamp="2026-10-14T15:00:00+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw=raw,
        )

    def _latest_content(self) -> str:
        return self.store.recent_messages(GROUP, 5)[0].content

    def test_image_reference_yields_no_ref_when_download_fails(self) -> None:
        self._stub_fetch(None)
        image_ref, digest, size, content_type = asyncio.run(
            self.client._image_reference(_attachment())
        )
        # 旧代码这里返回的是 attachment.url，视觉模型拿到也打不开
        self.assertEqual((image_ref, digest, size), ("", "", 0))
        self.assertEqual(content_type, "image/jpeg")

    def test_failed_download_skips_vision_model(self) -> None:
        self._stub_fetch(None)
        with self.assertLogs(bot_module.logger, level="WARNING") as logs:
            descriptions = asyncio.run(self.client._describe_attachments([_attachment()]))
        self.assertEqual(descriptions, [""])
        self.assertEqual(self.vision.calls, [])
        self.assertTrue(any("skipping vision call" in line for line in logs.output))

    def test_failed_download_marks_message_unrecognized(self) -> None:
        self._stub_fetch(None)
        self._save("cat-1", "[图片]", CAT_RAW)
        budget = asyncio.run(
            self.client._describe_recent_images(GROUP, exclude_key="", budget=2)
        )
        # 额度照扣（这一轮已经试过了），但一次视觉调用都没发生
        self.assertEqual(budget, 1)
        self.assertEqual(self.vision.calls, [])
        self.assertEqual(self.fetched_urls, [CAT_URL])
        self.assertIn(UNRECOGNIZED_IMAGE_MARK, self._latest_content())

    def test_unrecognized_mark_stops_the_next_round_from_retrying(self) -> None:
        self._stub_fetch(None)
        self._save("cat-2", "[图片]", CAT_RAW)
        asyncio.run(self.client._describe_recent_images(GROUP, exclude_key="", budget=2))
        budget = asyncio.run(
            self.client._describe_recent_images(GROUP, exclude_key="", budget=2)
        )
        self.assertEqual(budget, 2)
        self.assertEqual(self.fetched_urls, [CAT_URL])

    def test_successful_download_still_inlines_and_describes(self) -> None:
        self._stub_fetch((JPEG_BYTES, "image/jpeg"))
        self._save("cat-3", "[图片]", CAT_RAW)
        budget = asyncio.run(
            self.client._describe_recent_images(GROUP, exclude_key="", budget=2)
        )
        self.assertEqual(budget, 1)
        self.assertEqual(len(self.vision.calls), 1)
        self.assertTrue(self.vision.calls[0].startswith("data:image/jpeg;base64,"))
        self.assertIn("[图片内容：一只橘猫趴在键盘上]", self._latest_content())


class ImageBackfillTests(_LoopTestCase):
    """补识别旧图片：后台跑、同群只跑一份、出错不外泄也不占坑。"""

    def test_runs_in_background_and_clears_the_flag_when_done(self) -> None:
        async def scenario() -> None:
            client = _client()
            gate = asyncio.Event()
            recorder = _BackfillRecorder(gate=gate)
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="k-1", budget=2)
            tasks = list(client._image_backfill_tasks)
            self.assertEqual(len(tasks), 1)

            await asyncio.sleep(0)
            # 任务已经在后台跑起来、但仍卡在 gate 上 —— 调用方没有被它堵住
            self.assertEqual(recorder.calls, [(GROUP, "k-1", 2)])
            self.assertIn(GROUP, client._image_backfill_running)

            gate.set()
            await asyncio.gather(*tasks)
            self.assertEqual(recorder.calls, [(GROUP, "k-1", 2)])
            self.assertNotIn(GROUP, client._image_backfill_running)
            self.assertEqual(client._image_backfill_tasks, set())

        asyncio.run(scenario())

    def test_same_group_is_deduplicated_while_one_is_running(self) -> None:
        async def scenario() -> None:
            client = _client()
            gate = asyncio.Event()
            recorder = _BackfillRecorder(gate=gate)
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="k-1", budget=2)
            first = list(client._image_backfill_tasks)
            await asyncio.sleep(0)
            # 紧接着又来一条 @，不能把同一批旧图重新烧一遍额度
            client._queue_image_backfill(GROUP, exclude_key="k-2", budget=2)
            self.assertEqual(len(client._image_backfill_tasks), 1)

            gate.set()
            await asyncio.gather(*first)
            self.assertEqual(recorder.calls, [(GROUP, "k-1", 2)])

        asyncio.run(scenario())

    def test_other_group_can_run_at_the_same_time(self) -> None:
        async def scenario() -> None:
            client = _client()
            gate = asyncio.Event()
            recorder = _BackfillRecorder(gate=gate)
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="", budget=1)
            client._queue_image_backfill(OTHER_GROUP, exclude_key="", budget=1)
            tasks = list(client._image_backfill_tasks)
            self.assertEqual(len(tasks), 2)

            gate.set()
            await asyncio.gather(*tasks)
            self.assertEqual({call[0] for call in recorder.calls}, {GROUP, OTHER_GROUP})
            self.assertEqual(client._image_backfill_running, set())

        asyncio.run(scenario())

    def test_finished_backfill_lets_the_next_one_in(self) -> None:
        async def scenario() -> None:
            client = _client()
            recorder = _BackfillRecorder()
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="", budget=1)
            await asyncio.gather(*list(client._image_backfill_tasks))
            client._queue_image_backfill(GROUP, exclude_key="", budget=1)
            await asyncio.gather(*list(client._image_backfill_tasks))
            self.assertEqual(len(recorder.calls), 2)

        asyncio.run(scenario())

    def test_zero_budget_starts_nothing(self) -> None:
        async def scenario() -> None:
            client = _client()
            recorder = _BackfillRecorder()
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="", budget=0)
            await asyncio.sleep(0.01)
            self.assertEqual(recorder.calls, [])
            self.assertEqual(client._image_backfill_running, set())
            self.assertEqual(client._image_backfill_tasks, set())

        asyncio.run(scenario())

    def test_error_is_logged_and_does_not_block_later_runs(self) -> None:
        async def scenario() -> None:
            client = _client()
            recorder = _BackfillRecorder(error=RuntimeError("boom"))
            client._describe_recent_images = recorder

            client._queue_image_backfill(GROUP, exclude_key="", budget=1)
            with self.assertLogs(bot_module.logger, level="ERROR") as logs:
                await asyncio.gather(*list(client._image_backfill_tasks))
            self.assertTrue(any("image backfill failed" in line for line in logs.output))
            self.assertNotIn(GROUP, client._image_backfill_running)

            client._queue_image_backfill(GROUP, exclude_key="", budget=1)
            retry = list(client._image_backfill_tasks)
            self.assertEqual(len(retry), 1)
            with self.assertLogs(bot_module.logger, level="ERROR"):
                await asyncio.gather(*retry)
            self.assertEqual(len(recorder.calls), 2)

        asyncio.run(scenario())


def _at_message(content: str = "在吗") -> SimpleNamespace:
    return SimpleNamespace(
        group_openid=GROUP,
        id="m-at-1",
        msg_seq="0",
        content=content,
        timestamp="2026-10-14T15:00:00+08:00",
        username="CL",
        member_role="member",
        author=SimpleNamespace(member_openid="openid-CL"),
        raw={"mentions": [{"bot": True, "is_you": True, "id": "bot-1"}]},
    )


class ReplyIsNotBlockedByBackfillTests(_LoopTestCase):
    """@ 一次就回复一次：补识别旧图不许再排在回复前面。"""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        super().tearDown()

    def _wire(self, client: XiaoxiaClient, sent: list[tuple[int, str]], answer: str) -> None:
        async def parse_intent(**kwargs):
            return {"action": "chat", "reply": answer}

        async def post_group_message(**kwargs):
            sent.append((kwargs["msg_seq"], kwargs["content"]))

        async def handle_intent(**kwargs):
            return answer

        client.llm = SimpleNamespace(parse_intent=parse_intent)
        client._handle_intent = handle_intent
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._topic_context = lambda group: ("当前话题", "", "全量记录")
        client._identity_context = lambda group, requester: SimpleNamespace(prompt=lambda: "")
        client._memory_context = lambda group: ""
        client._remember_bot_reply = lambda group, text, source: None

    def test_reply_goes_out_while_backfill_is_still_running(self) -> None:
        async def scenario() -> None:
            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            self._wire(client, sent, "在的～")

            gate = asyncio.Event()
            started = asyncio.Event()

            async def slow_backfill(group_openid, *, exclude_key, budget):
                started.set()
                await gate.wait()
                return budget

            client._describe_recent_images = slow_backfill

            await asyncio.wait_for(
                client._process_at_message(_at_message(), "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )
            self.assertEqual(sent, [(1, "在的～")])
            self.assertTrue(started.is_set())
            self.assertFalse(gate.is_set())
            self.assertIn(GROUP, client._image_backfill_running)

            gate.set()
            await asyncio.gather(*list(client._image_backfill_tasks))

        asyncio.run(scenario())

    def test_backfill_is_skipped_when_the_request_ate_the_whole_budget(self) -> None:
        async def scenario() -> None:
            client = _client(store=self.store, vision_max_images_per_request=1)
            sent: list[tuple[int, str]] = []
            self._wire(client, sent, "看到啦")

            calls: list[str] = []

            async def backfill(group_openid, *, exclude_key, budget):
                calls.append(group_openid)
                return budget

            client._describe_recent_images = backfill

            async def fake_fetch(url, *, max_bytes):
                return JPEG_BYTES, "image/jpeg"

            original_fetch = bot_module.fetch_image_bytes
            bot_module.fetch_image_bytes = fake_fetch
            self.addCleanup(setattr, bot_module, "fetch_image_bytes", original_fetch)

            async def describe_image(image_ref, hint=""):
                return "一张课表"

            client.llm = SimpleNamespace(
                parse_intent=client.llm.parse_intent, describe_image=describe_image
            )

            message = _at_message("看下这张课表")
            message.raw = dict(message.raw)
            message.raw["attachments"] = [
                {"content_type": "image/jpeg", "filename": "plan.jpg", "url": CAT_URL}
            ]
            await asyncio.wait_for(
                client._process_at_message(message, "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )
            await asyncio.sleep(0.01)
            # 这一轮的额度全给了当前这张图，没有余量就不该再起后台任务
            self.assertEqual(calls, [])
            self.assertEqual(client._image_backfill_tasks, set())
            self.assertEqual(sent, [(1, "我看看图～"), (2, "看到啦")])

        asyncio.run(scenario())


class SyncImageBatchBeforeReplyTests(_LoopTestCase):
    """图片先单独发、下一条再 @ 问图：第一轮 prompt 必须已经包含最近一批图片。"""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "test.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        super().tearDown()

    def _save_image(self, message_id: str, url: str, *, timestamp: str) -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content="[图片]",
            timestamp=timestamp,
            source="GROUP_MESSAGE_CREATE",
            raw={
                "attachments": [
                    {"content_type": "image/jpeg", "filename": f"{message_id}.jpg", "url": url}
                ]
            },
        )

    def _save_text(self, message_id: str, content: str, *, timestamp: str) -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content=content,
            timestamp=timestamp,
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def _wire(
        self,
        client: XiaoxiaClient,
        sent: list[tuple[int, str]],
        topics: list[str],
        fetched: list[str],
        vision_calls: list[str],
        *,
        answer: str = "看完啦",
    ) -> None:
        async def parse_intent(**kwargs):
            topics.append(kwargs["topic_context"])
            return {"action": "chat", "reply": answer}

        async def describe_image(image_ref, hint=""):
            vision_calls.append(image_ref)
            return f"第{len(vision_calls)}张图"

        async def post_group_message(**kwargs):
            sent.append((kwargs["msg_seq"], kwargs["content"]))

        async def handle_intent(**kwargs):
            return answer

        async def fake_fetch(url, *, max_bytes):
            fetched.append(url)
            return JPEG_BYTES + url.encode(), "image/jpeg"

        client.llm = SimpleNamespace(parse_intent=parse_intent, describe_image=describe_image)
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._handle_intent = handle_intent
        client._queue_image_backfill = lambda *args, **kwargs: None
        client._identity_context = lambda group, requester: SimpleNamespace(prompt=lambda: "")
        client._memory_context = lambda group: ""
        client._remember_bot_reply = lambda group, text, source: None

        original_fetch = bot_module.fetch_image_bytes
        bot_module.fetch_image_bytes = fake_fetch
        self.addCleanup(setattr, bot_module, "fetch_image_bytes", original_fetch)

    def test_pending_image_batch_is_described_before_prompt(self) -> None:
        async def scenario() -> None:
            self._save_image(
                "image-1",
                "https://example.com/1.jpg",
                timestamp="2026-10-20T14:40:00+08:00",
            )
            self._save_image(
                "image-2",
                "https://example.com/2.jpg",
                timestamp="2026-10-20T14:40:10+08:00",
            )
            self._save_image(
                "image-3",
                "https://example.com/3.jpg",
                timestamp="2026-10-20T14:40:20+08:00",
            )

            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(
                client,
                sent,
                topics,
                fetched,
                vision_calls,
                answer="三张都看到啦",
            )

            await asyncio.wait_for(
                client._process_at_message(
                    _at_message("这三张图片是啥"), "GROUP_AT_MESSAGE_CREATE"
                ),
                timeout=5,
            )

            self.assertEqual(
                fetched,
                [
                    "https://example.com/1.jpg",
                    "https://example.com/2.jpg",
                    "https://example.com/3.jpg",
                ],
            )
            self.assertEqual(len(vision_calls), 3)
            self.assertEqual(
                sent,
                [(1, "我一张张看～"), (2, "三张都看到啦")],
            )
            for index in (1, 2, 3):
                self.assertIn(f"〔图〕第{index}张图", topics[0])
            contents = {
                m.message_key.split(":", 1)[0]: m.content
                for m in self.store.recent_messages_with_raw(GROUP, 10)
            }
            for index in (1, 2, 3):
                self.assertIn(f"第{index}张图", contents[f"image-{index}"])

        asyncio.run(scenario())

    def _save_images(self, message_id: str, urls: list[str], *, timestamp: str) -> None:
        self.store.save_message(
            message_id=message_id,
            msg_seq="",
            group_openid=GROUP,
            member_openid="openid-CL",
            username="CL",
            role="member",
            content="[图片]" * len(urls),
            timestamp=timestamp,
            source="GROUP_MESSAGE_CREATE",
            raw={
                "attachments": [
                    {
                        "content_type": "image/jpeg",
                        "filename": f"{message_id}-{index}.jpg",
                        "url": url,
                    }
                    for index, url in enumerate(urls, start=1)
                ]
            },
        )

    def test_multiple_attachments_in_one_message_are_all_described(self) -> None:
        async def scenario() -> None:
            urls = [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
                "https://example.com/c.jpg",
            ]
            self._save_images(
                "one-message",
                urls,
                timestamp="2026-10-20T14:40:00+08:00",
            )

            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(client, sent, topics, fetched, vision_calls, answer="三张都是猫")

            await asyncio.wait_for(
                client._process_at_message(
                    _at_message("这三张是什么"), "GROUP_AT_MESSAGE_CREATE"
                ),
                timeout=5,
            )

            self.assertEqual(fetched, urls)
            self.assertEqual(len(vision_calls), 3)
            self.assertEqual(sent, [(1, "我一张张看～"), (2, "三张都是猫")])
            for index in (1, 2, 3):
                self.assertIn(f"〔图〕第{index}张图", topics[0])

        asyncio.run(scenario())

    def test_text_message_breaks_the_pending_batch(self) -> None:
        async def scenario() -> None:
            self._save_image(
                "old-image",
                "https://example.com/old.jpg",
                timestamp="2026-10-20T14:40:00+08:00",
            )
            self._save_text(
                "chatter",
                "先聊点别的",
                timestamp="2026-10-20T14:41:00+08:00",
            )

            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(client, sent, topics, fetched, vision_calls, answer="在呀")

            await asyncio.wait_for(
                client._process_at_message(_at_message("在吗"), "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )

            # 文字把这一批截断了，不能继续往前把旧图拖进同步识图。
            self.assertEqual(fetched, [])
            self.assertEqual(vision_calls, [])
            self.assertEqual(sent, [(1, "在呀")])

        asyncio.run(scenario())

    def test_image_batch_respects_the_total_budget(self) -> None:
        async def scenario() -> None:
            for index in range(1, 11):
                self._save_image(
                    f"image-{index}",
                    f"https://example.com/{index}.jpg",
                    timestamp=f"2026-10-20T14:40:{index:02d}+08:00",
                )

            client = _client(store=self.store, vision_max_images_per_request=9)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(client, sent, topics, fetched, vision_calls, answer="九张看完啦")

            await asyncio.wait_for(
                client._process_at_message(
                    _at_message("这些图都看看"), "GROUP_AT_MESSAGE_CREATE"
                ),
                timeout=5,
            )

            # 从最新往前收满 9 张：最老的 image-1 留给后台补识别。
            self.assertEqual(
                fetched,
                [f"https://example.com/{index}.jpg" for index in range(2, 11)],
            )
            self.assertEqual(len(vision_calls), 9)
            self.assertEqual(
                sent,
                [(1, "我一张张看～"), (2, "九张看完啦")],
            )

        asyncio.run(scenario())

    def test_single_pending_image_also_gets_a_quick_ack(self) -> None:
        async def scenario() -> None:
            self._save_image(
                "only-image",
                "https://example.com/only.jpg",
                timestamp="2026-10-20T14:40:00+08:00",
            )

            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(client, sent, topics, fetched, vision_calls, answer="看到了")

            await asyncio.wait_for(
                client._process_at_message(
                    _at_message("这张图片是啥"), "GROUP_AT_MESSAGE_CREATE"
                ),
                timeout=5,
            )

            self.assertEqual(fetched, ["https://example.com/only.jpg"])
            self.assertEqual(len(vision_calls), 1)
            # 单张也要下载 + 调视觉模型，等好几秒，所以同样先垫一句。
            self.assertEqual(sent, [(1, "我看看图～"), (2, "看到了")])

        asyncio.run(scenario())

    def test_no_pending_image_keeps_the_old_fast_path(self) -> None:
        async def scenario() -> None:
            client = _client(store=self.store)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            fetched: list[str] = []
            vision_calls: list[str] = []
            self._wire(client, sent, topics, fetched, vision_calls, answer="在的")

            await client._process_at_message(
                _at_message("在吗"), "GROUP_AT_MESSAGE_CREATE"
            )

            self.assertEqual(fetched, [])
            self.assertEqual(vision_calls, [])
            self.assertEqual(sent, [(1, "在的")])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
