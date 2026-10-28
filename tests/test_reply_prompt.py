"""回复 prompt 的输入形状：谁在说话、哪行是看图的记录。

三条线上原样回放：
1. 把 A 的话算到 B 头上——记录里只有显示名，模型每轮都得猜；
2. 先发消息再 @ 它，它把「不是编程工具，是 agent」听成「在说我自己」；
3. 看图认出了白桃乌龙雪糕，被问却说「图片我是真看不了」。
前两条靠给每条消息挂【说话人】标签和规则约束，第三条靠把看图记录拆成独立的〔图〕行。
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from maodou.bot import IdentityContext, MaodouClient
from maodou.llm import MaodouLLM
from maodou.store import StoredMessage


GROUP = "test-group"


def _member(username: str, openid: str | None = None) -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid=openid if openid is not None else "openid-" + username,
        role="member",
        content="",
        timestamp="2026-10-16T15:24:45+08:00",
        source="GROUP_MESSAGE_CREATE",
    )


def _llm(reply_max_chars: int = 900) -> MaodouLLM:
    settings = SimpleNamespace(
        llm_api_key="k",
        llm_base_url="https://example.invalid",
        model="big-model",
        care_model="",
        vision_model="cheap-model",
        reply_max_chars=reply_max_chars,
    )
    return MaodouLLM(settings)


class GenerateReplyPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()

    def _capture(self, request: str = "这条可以不看", **kwargs) -> dict:
        captured: dict = {}

        async def fake_chat(**chat_kwargs):
            captured.update(chat_kwargs)
            return "好"

        llm = _llm()
        llm._chat = fake_chat
        self.loop.run_until_complete(
            llm.generate_reply(
                requester=_member("CL"),
                request=request,
                topic_context="2026-10-16 15:24:45 【主人】这条可以不看",
                **kwargs,
            )
        )
        return captured

    def test_user_prompt_explains_the_speaker_brackets(self) -> None:
        prompt = self._capture()["user_prompt"]
        self.assertIn("每条记录开头的【】里就是发那条消息的人", prompt)
        self.assertIn("【小毛豆】是你自己说过的话", prompt)

    def test_request_keeps_image_notes_out_of_the_user_text(self) -> None:
        prompt = self._capture(
            "八块，我去\n[图片内容：一张俯拍照片，手里的白桃乌龙雪糕]"
        )["user_prompt"]
        lines = prompt.splitlines()
        self.assertIn("八块，我去", lines)
        self.assertNotIn("八块，我去\n[图片内容：", prompt)
        self.assertTrue(any(line.strip() == "〔图〕一张俯拍照片，手里的白桃乌龙雪糕" for line in lines))

    def test_intent_prompt_keeps_image_notes_out_of_the_user_text(self) -> None:
        captured: dict = {}

        async def fake_chat(**chat_kwargs):
            captured.update(chat_kwargs)
            return "{}"

        llm = _llm()
        llm._chat = fake_chat
        self.loop.run_until_complete(
            llm.parse_intent(
                request="八块，我去\n[图片内容：一张俯拍照片，手里的白桃乌龙雪糕]",
                current_time="2026-10-16T15:24:45+08:00",
                tasks_summary="",
            )
        )
        self.assertNotIn("[图片内容：", captured["user_prompt"])
        self.assertIn("〔图〕一张俯拍照片，手里的白桃乌龙雪糕", captured["user_prompt"])

    def test_long_reply_is_no_longer_clipped_at_the_old_900_chars(self) -> None:
        captured: dict = {}

        async def fake_chat(**chat_kwargs):
            captured.update(chat_kwargs)
            return "长" * 2000

        llm = _llm(reply_max_chars=3000)
        llm._chat = fake_chat
        reply = self.loop.run_until_complete(
            llm.generate_reply(
                requester=_member("CL"),
                request="详细分析一下",
                topic_context="【主人】详细分析一下",
            )
        )
        self.assertEqual(len(reply), 2000)
        self.assertEqual(captured["max_tokens"], 2200)

    def test_voice_reply_declares_capability_and_spoken_style(self) -> None:
        prompt = self._capture("发一条语音", voice_reply=True)["user_prompt"]
        self.assertIn("【本轮输出方式：语音】", prompt)
        self.assertIn("你确实可以发语音", prompt)
        self.assertIn("口语、简短（1～3 句）、适合朗读", prompt)
        self.assertIn("不要说自己不会发语音、只能打字", prompt)

    def test_regular_chat_does_not_add_voice_delivery_context(self) -> None:
        prompt = self._capture("随便聊聊")["user_prompt"]
        self.assertNotIn("【本轮输出方式：语音】", prompt)


class IdentityLegendTests(unittest.TestCase):
    """身份块要把【】标签对准人：谁是「你」、谁是「另一位」。"""

    def test_legend_maps_labels_to_roles(self) -> None:
        identity = IdentityContext(
            requester=_member("CL", "openid-CL"),
            partner=_member("二十四桥明月夜", "openid-对方"),
            addresses={"CL": "主人", "二十四桥明月夜": "小虾主人"},
            speaker_labels={
                "openid-CL": "主人",
                "openid-对方": "小虾主人",
                "主人": "主人",
                "小虾主人": "小虾主人",
            },
        )
        prompt = identity.prompt()
        self.assertIn("【主人】= 当前触发者（就是「你」）", prompt)
        self.assertIn("【小虾主人】= 另一位聊天成员", prompt)
        self.assertIn("【小毛豆】= 你自己说的话", prompt)

    def test_legend_without_partner_still_names_the_requester(self) -> None:
        identity = IdentityContext(requester=_member("CL"), partner=None)
        prompt = identity.prompt()
        self.assertIn("【CL】= 当前触发者", prompt)
        self.assertNotIn("另一位聊天成员】", prompt)


class TopicContextLabelTests(unittest.TestCase):
    """_topic_context 渲染出来的记录必须挂上登记过的称呼。"""

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from maodou.store import ChatStore

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "labels.sqlite3"))
        self.client = MaodouClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                recent_messages=50,
                max_context_chars=4000,
            ),
            store=self.store,
            llm=SimpleNamespace(),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _save(self, content: str, username: str, member_openid: str) -> None:
        self.store.save_message(
            message_id=f"id-{content}",
            msg_seq="",
            group_openid=GROUP,
            member_openid=member_openid,
            username=username,
            role="member",
            content=content,
            timestamp="2026-10-16T23:03:17+08:00",
            source="GROUP_MESSAGE_CREATE",
            raw={},
        )

    def test_topic_context_uses_registered_addresses(self) -> None:
        self._save("都怪你叫叫叫", "二十四桥明月夜", "openid-小虾")
        self._save("那你说清楚点", "CL", "openid-CL")
        self.store.set_person_address(GROUP, "二十四桥明月夜", "小虾主人")
        self.store.set_person_address(GROUP, "CL", "主人")
        self.store.save_bot_message(
            group_openid=GROUP,
            content="我脑子一乱……",
            timestamp="2026-10-16T23:04:16+08:00",
        )

        topic, _background, _full = self.client._topic_context(GROUP)
        lines = topic.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("【小虾主人】都怪你叫叫叫", lines[0])
        self.assertIn("【主人】那你说清楚点", lines[1])
        self.assertIn("【小毛豆】我脑子一乱", lines[2])


if __name__ == "__main__":
    unittest.main()
