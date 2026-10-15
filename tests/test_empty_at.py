"""空 @ 不许再变成一份循环播放的功能说明书。

线上（86FD26B0… 群，2026-10-15）一天里出现四次：用户只 @ 一下没带话，
机器人每次都回一份几乎一模一样的两百字功能清单。原因是旧代码把空消息硬填成
「你好，请介绍一下你能帮我们做什么。」——一句写死的假用户消息，把模型钉死在
「自我介绍」这一条路上，SYSTEM_PROMPT 里那句「不要重复自己」根本拦不住，
因为重复的指令是代码自己下的。

现在只如实告诉模型「对方没带话」，接什么由它照【当前话题】自己定。
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import xiaoxia.bot as bot_module
import xiaoxia.store as store_module
from xiaoxia.bot import EMPTY_AT_REQUEST, XiaoxiaClient
from xiaoxia.llm import SYSTEM_PROMPT
from xiaoxia.store import ChatStore


GROUP = "test-group"
BOT_ID = "bot-1"
OLD_SELF_INTRO = "请介绍一下你能帮我们做什么"
_IDS = itertools.count(1)


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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _at_message(content: str, *, attachments: list | None = None) -> SimpleNamespace:
    raw: dict = {"mentions": [{"bot": True, "is_you": True, "id": BOT_ID}]}
    if attachments:
        raw["attachments"] = attachments
    return SimpleNamespace(
        group_openid=GROUP,
        id=f"m-{next(_IDS)}",
        msg_seq="0",
        content=content,
        timestamp="2026-10-15T17:38:49+08:00",
        username="CL",
        member_role="member",
        author=SimpleNamespace(member_openid="openid-CL"),
        raw=raw,
    )


class EmptyAtRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "empty-at.sqlite3"))
        self.to_intent: list[str] = []
        self.to_reply: list[str] = []
        self.sent: list[tuple[int, str]] = []

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _client(self, **overrides) -> XiaoxiaClient:
        client = XiaoxiaClient(
            settings=_settings(**overrides), store=self.store, llm=SimpleNamespace()
        )

        async def parse_intent(**kwargs) -> dict:
            self.to_intent.append(kwargs["request"])
            return {"action": "chat"}

        async def handle_intent(**kwargs) -> str:
            self.to_reply.append(kwargs["request"])
            return "在呢，怎么啦？"

        async def post_group_message(**kwargs) -> None:
            self.sent.append((kwargs["msg_seq"], kwargs["content"]))

        client.llm = SimpleNamespace(parse_intent=parse_intent)
        client._handle_intent = handle_intent
        client.api = SimpleNamespace(post_group_message=post_group_message)
        client._topic_context = lambda group: ("CL：好累", "", "全量记录")
        client._identity_context = lambda group, requester: SimpleNamespace(prompt=lambda: "")
        client._memory_context = lambda group: ""
        client._remember_bot_reply = lambda group, text, source: None
        client._queue_image_backfill = lambda *args, **kwargs: None
        return client

    def _run(self, client: XiaoxiaClient, message: SimpleNamespace) -> None:
        self.loop.run_until_complete(
            client._process_at_message(message, "GROUP_AT_MESSAGE_CREATE")
        )

    def test_bare_at_becomes_an_honest_note_not_a_scripted_question(self) -> None:
        # QQ 的 AT 事件里，只 @ 不说话时 content 就是一个空格
        self._run(self._client(), _at_message(" "))
        self.assertEqual(self.to_reply, [EMPTY_AT_REQUEST])
        self.assertEqual(self.to_intent, [EMPTY_AT_REQUEST])
        self.assertNotIn(OLD_SELF_INTRO, self.to_reply[0])
        self.assertEqual(self.sent[-1][1], "在呢，怎么啦？")

    def test_mention_only_message_counts_as_empty_too(self) -> None:
        # 全量消息模式下 @ 走 GROUP_MESSAGE_CREATE，剥掉提及之后什么都不剩
        self._run(self._client(), _at_message(f"<@{BOT_ID}> "))
        self.assertEqual(self.to_reply, [EMPTY_AT_REQUEST])

    def test_real_text_is_passed_through_untouched(self) -> None:
        self._run(self._client(), _at_message(f"<@{BOT_ID}> 明天记得提醒我买花"))
        self.assertEqual(self.to_reply, ["明天记得提醒我买花"])
        self.assertEqual(self.to_intent, ["明天记得提醒我买花"])

    def test_an_image_beats_the_empty_note(self) -> None:
        client = self._client()

        async def fake_describe(attachments, hint="") -> list[str]:
            return ["一张橘猫瘫在桌上的表情包"]

        client._describe_attachments = fake_describe
        self._run(
            client,
            _at_message(
                " ",
                attachments=[
                    {"content_type": "image/jpeg", "filename": "cat.jpg", "url": "https://x/cat.jpg"}
                ],
            ),
        )
        self.assertIn("橘猫", self.to_reply[0])
        self.assertNotEqual(self.to_reply[0], EMPTY_AT_REQUEST)

    def test_the_hardcoded_self_intro_is_gone_from_the_code(self) -> None:
        # 那句话现在只允许出现在解释历史的注释里，绝不能再出现在代码里：
        # 一旦有人把它当兜底填回去，空 @ 又会变成循环播放的说明书。
        source = inspect.getsource(bot_module)
        code_only = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertFalse(OLD_SELF_INTRO in code_only, "空 @ 又被写死成一句假的用户消息了")
        self.assertIn(
            "EMPTY_AT_REQUEST", inspect.getsource(bot_module.XiaoxiaClient._process_at_message)
        )

    def test_the_stored_placeholder_does_not_smell_like_internals(self) -> None:
        # 线上真出现过：空 @ 入库成「[non-text message]」，模型读到这个英文标记，
        # 回了一句「它到我这边就只是个占位符，里面是啥我看不到」——SYSTEM_PROMPT
        # 第 7 条明写着不许编自己的内部实现，可这个标记本身就在暗示有个占位符。
        self._run(self._client(), _at_message(" "))
        stored = self.store.recent_messages(GROUP, 5)
        self.assertTrue(stored)
        self.assertEqual(stored[-1].content, "[无文字内容]")
        self.assertNotIn("non-text message", inspect.getsource(store_module))

    def test_system_prompt_forbids_recapping_the_feature_list(self) -> None:
        # 光改掉那句假消息还不够：模型得知道空 @ 该怎么接，否则它自己也会去背清单
        self.assertIn("只 @ 我没带话", SYSTEM_PROMPT)
        self.assertIn("别背功能清单", SYSTEM_PROMPT)
        self.assertIn("能力介绍只在被明确问起时说一次", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()