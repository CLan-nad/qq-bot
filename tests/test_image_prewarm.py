"""收到图片就后台预识别：不用 @、不发消息、不重复调用视觉模型。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import maodou.bot as bot_module
from maodou.bot import MaodouClient
from maodou.images import UNRECOGNIZED_IMAGE_MARK
from maodou.store import ChatStore


GROUP = "prewarm-group"
OTHER_GROUP = "prewarm-other"
JPEG_BYTES = b"\xff\xd8\xff\xe0prewarm-stub"


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


class _LoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "prewarm.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()


class _Vision:
    """可计数、可卡住的假视觉模型。"""

    def __init__(self, text: str = "一只橘猫趴在键盘上") -> None:
        self.text = text
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.started = 0

    async def __call__(self, image_ref, hint=""):
        self.calls.append(hint)
        self.started += 1
        if self.gate is not None and self.release is not None:
            self.gate.set()
            await self.release.wait()
        return self.text


class _FakeLLM:
    def __init__(self, vision: _Vision) -> None:
        self.describe_image = vision
        self.judge_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.intent_calls = 0

    async def judge_active_reply(self, **kwargs):
        self.judge_calls.append(kwargs)
        return {"reply": False, "reason": "test"}

    async def generate_active_reply(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ""

    async def parse_intent(self, **kwargs):
        self.intent_calls += 1
        raise AssertionError("预识别和普通消息都不能进入 action 分发")


def _message(
    *,
    group: str = GROUP,
    message_id: str = "image-1",
    content: str = "[图片]",
    url: str = "https://example.com/cat.jpg",
    mentions: list[dict] | None = None,
):
    return SimpleNamespace(
        group_openid=group,
        id=message_id,
        msg_seq=1,
        content=content,
        timestamp="2026-10-30T15:00:00+08:00",
        username="CL",
        member_role="member",
        author=SimpleNamespace(member_openid="openid-CL"),
        raw={
            "mentions": list(mentions or []),
            "attachments": [
                {
                    "content_type": "image/jpeg",
                    "filename": "cat.jpg",
                    "url": url,
                }
            ],
        },
    )


class ImagePrewarmTests(_LoopTestCase):
    def _client(self, vision: _Vision, **overrides) -> MaodouClient:
        client = MaodouClient(
            settings=_settings(**overrides),
            store=self.store,
            llm=_FakeLLM(vision),
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

    def _wire_chat(
        self,
        client: MaodouClient,
        sent,
        answer="看到啦",
        topics: list[str] | None = None,
    ) -> None:
        async def parse_intent(**kwargs):
            if topics is not None:
                topics.append(kwargs["topic_context"])
            return {"action": "chat"}

        async def handle_intent(**kwargs):
            return answer

        async def post_group_message(**kwargs):
            sent.append((kwargs["msg_seq"], kwargs["content"]))

        client.llm.parse_intent = parse_intent
        client._handle_intent = handle_intent
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._identity_context = lambda group, requester: SimpleNamespace(
            prompt=lambda: ""
        )
        client._memory_context = lambda group: ""
        client._summaries_context = lambda group: ""
        client._remember_bot_reply = lambda *args, **kwargs: None
        client._queue_image_backfill = lambda *args, **kwargs: None

    def test_image_without_at_is_recognized_and_never_enters_actions(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            client = self._client(vision)

            await client.on_group_message_create(_message())
            await client._wait_image_prewarm(
                client._image_prewarm_events(GROUP), timeout=5
            )

            self.assertEqual(len(vision.calls), 1)
            stored = self.store.get_message(GROUP, "image-1:1")
            assert stored is not None
            self.assertIn("[图片内容：一只橘猫趴在键盘上]", stored.content)
            self.assertEqual(client.llm.intent_calls, 0)

        asyncio.run(scenario())

    def test_same_group_processes_images_in_order(self) -> None:
        async def scenario() -> None:
            fetched: list[str] = []
            self._stub_fetch(fetched)
            vision = _Vision()
            vision.gate = asyncio.Event()
            vision.release = asyncio.Event()
            client = self._client(vision)

            await client.on_group_message_create(
                _message(message_id="img-a", url="https://example.com/a.jpg")
            )
            await client.on_group_message_create(
                _message(message_id="img-b", url="https://example.com/b.jpg")
            )
            await asyncio.sleep(0)
            self.assertEqual(vision.started, 1)
            vision.release.set()
            await asyncio.gather(*list(client._image_prewarm_tasks))

            self.assertEqual(
                fetched,
                ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            )
            self.assertEqual(vision.started, 2)

        asyncio.run(scenario())

    def test_different_groups_run_in_parallel(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            vision.gate = asyncio.Event()
            vision.release = asyncio.Event()
            client = self._client(vision)

            await client.on_group_message_create(
                _message(group=GROUP, message_id="g1", url="https://example.com/1.jpg")
            )
            await client.on_group_message_create(
                _message(
                    group=OTHER_GROUP,
                    message_id="g2",
                    url="https://example.com/2.jpg",
                )
            )
            await asyncio.sleep(0)
            self.assertEqual(vision.started, 2)
            vision.release.set()
            await asyncio.gather(*list(client._image_prewarm_tasks))

        asyncio.run(scenario())

    def test_same_message_is_queued_only_once(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            vision.gate = asyncio.Event()
            vision.release = asyncio.Event()
            client = self._client(vision)

            message = _message()
            client._save_message(message, "GROUP_MESSAGE_CREATE")
            self.assertTrue(client._prewarm_image_message(message))
            self.assertFalse(client._prewarm_image_message(message))
            self.assertEqual(len(client._image_prewarm_items), 1)
            vision.release.set()
            await asyncio.gather(*list(client._image_prewarm_tasks))
            self.assertEqual(vision.started, 1)

        asyncio.run(scenario())

    def test_at_reuses_prewarm_result_without_second_vision_call(self) -> None:
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
            self.assertEqual(len(vision.calls), 1)

            at_message = SimpleNamespace(
                **{
                    **_message(message_id="at-1", content="@小毛豆 图里是什么").__dict__,
                    "raw": {
                        "mentions": [{"bot": True, "is_you": True, "id": "bot-1"}]
                    },
                }
            )
            await asyncio.wait_for(
                client._process_at_message(at_message, "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )

            self.assertEqual(len(vision.calls), 1)
            self.assertEqual(sent, [(1, "看到啦")])

        asyncio.run(scenario())

    def test_at_waits_for_inflight_prewarm_of_previous_image(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            vision.gate = asyncio.Event()
            vision.release = asyncio.Event()
            client = self._client(vision, ack_after_seconds=0.01)
            sent: list[tuple[int, str]] = []
            topics: list[str] = []
            self._wire_chat(client, sent, topics=topics)

            # 图先单独发，后台已经开始识别但还没回来。
            await client.on_group_message_create(_message(message_id="before-at"))
            await asyncio.sleep(0.02)
            self.assertEqual(vision.started, 1)

            at_message = SimpleNamespace(
                **{
                    **_message(
                        message_id="after-at", content="@小毛豆 图里是什么"
                    ).__dict__,
                    "raw": {
                        "mentions": [{"bot": True, "is_you": True, "id": "bot-1"}]
                    },
                }
            )
            task = asyncio.create_task(
                client._process_at_message(at_message, "GROUP_AT_MESSAGE_CREATE")
            )
            await asyncio.sleep(0.05)
            # 还没拿到图，所以先垫一句，也还没开始解析意图。
            self.assertEqual(sent, [(1, "我看看图～")])
            self.assertEqual(topics, [])

            vision.release.set()
            await asyncio.wait_for(task, timeout=5)

            self.assertEqual(len(vision.calls), 1)
            self.assertIn("一只橘猫趴在键盘上", topics[0])
            self.assertEqual(sent[-1], (2, "看到啦"))

        asyncio.run(scenario())

    def test_slow_current_image_acks_once_then_replies(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            vision.gate = asyncio.Event()
            vision.release = asyncio.Event()
            client = self._client(vision, ack_after_seconds=0.01)
            sent: list[tuple[int, str]] = []
            self._wire_chat(client, sent)

            message = _message(
                message_id="at-image",
                content="@小毛豆 看看这个",
                mentions=[{"bot": True, "is_you": True, "id": "bot-1"}],
            )
            task = asyncio.create_task(
                client._process_at_message(message, "GROUP_AT_MESSAGE_CREATE")
            )
            await asyncio.sleep(0.05)
            self.assertEqual(sent, [(1, "我看看图～")])
            vision.release.set()
            await asyncio.wait_for(task, timeout=5)

            self.assertEqual(sent, [(1, "我看看图～"), (2, "看到啦")])
            self.assertEqual(len(vision.calls), 1)

        asyncio.run(scenario())

    def test_fast_current_image_does_not_send_ack(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            client = self._client(vision, ack_after_seconds=5.0)
            sent: list[tuple[int, str]] = []
            self._wire_chat(client, sent)

            message = _message(
                message_id="at-fast",
                content="@小毛豆 看看这个",
                mentions=[{"bot": True, "is_you": True, "id": "bot-1"}],
            )
            await asyncio.wait_for(
                client._process_at_message(message, "GROUP_AT_MESSAGE_CREATE"),
                timeout=5,
            )

            self.assertEqual(sent, [(1, "看到啦")])

        asyncio.run(scenario())

    def test_active_reply_waits_for_the_image_description(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            client = self._client(vision, active_window_seconds=600)
            client._activate_active_group(GROUP)

            await client.on_group_message_create(_message(message_id="active-img"))
            await client._wait_image_prewarm(
                client._image_prewarm_events(GROUP), timeout=5
            )

            self.assertEqual(len(client.llm.judge_calls), 1)
            self.assertIn(
                "一只橘猫趴在键盘上",
                client.llm.judge_calls[0]["content"],
            )

        asyncio.run(scenario())

    def test_failed_description_is_marked_once_and_not_retried(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision(text="")
            client = self._client(vision)
            message = _message(message_id="bad-image")

            await client.on_group_message_create(message)
            await client._wait_image_prewarm(
                client._image_prewarm_events(GROUP), timeout=5
            )
            stored = self.store.get_message(GROUP, "bad-image:1")
            assert stored is not None
            self.assertIn(UNRECOGNIZED_IMAGE_MARK, stored.content)

            self.assertFalse(client._prewarm_image_message(message))
            await asyncio.gather(*list(client._image_prewarm_tasks))
            self.assertEqual(len(vision.calls), 1)

        asyncio.run(scenario())

    def test_prewarm_can_be_disabled(self) -> None:
        async def scenario() -> None:
            self._stub_fetch()
            vision = _Vision()
            client = self._client(vision, vision_prewarm=False)

            await client.on_group_message_create(_message())
            await asyncio.sleep(0.01)

            self.assertEqual(vision.calls, [])
            self.assertEqual(client._image_prewarm_items, {})

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
