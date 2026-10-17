"""摘要与记忆分块：摘要不再混进长期记忆，本次请求紧跟当前话题。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from xiaoxia.bot import XiaoxiaClient
from xiaoxia.llm import XiaoxiaLLM
from xiaoxia.store import ChatStore, StoredMessage, clip_summary


GROUP = "test-group"


def _member(username: str) -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid="openid-" + username,
        role="member",
        content="",
        timestamp="2026-10-16T15:24:45+08:00",
        source="GROUP_MESSAGE_CREATE",
    )


class ClipSummaryTests(unittest.TestCase):
    def test_short_text_passes_through(self) -> None:
        self.assertEqual(clip_summary("一句话。"), "一句话。")

    def test_long_text_cuts_at_last_full_sentence(self) -> None:
        text = "。" .join(["第%d句" % i for i in range(50)])
        clipped = clip_summary(text, max_chars=40)
        self.assertLessEqual(len(clipped), 40)
        self.assertTrue(clipped.endswith("。"))
        self.assertNotIn("…", clipped)

    def test_no_sentence_boundary_hard_cuts(self) -> None:
        clipped = clip_summary("很长" * 100, max_chars=30)
        self.assertEqual(len(clipped), 31)
        self.assertTrue(clipped.endswith("…"))


class SummaryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "summary.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _save_summary(self, start: int, summary: str) -> None:
        self.store.save_memory_batch(
            group_openid=GROUP,
            start_row_id=start,
            end_row_id=start + 9,
            summary=summary,
            memories=[],
        )

    def test_clamp_long_summaries(self) -> None:
        self._save_summary(1, "正常长度。")
        self._save_summary(11, "。" .join(["第%d句" % i for i in range(80)]))
        self.assertEqual(self.store.clamp_long_summaries(GROUP, max_chars=50), 1)
        summaries = self.store.conversation_summaries(GROUP, limit=10)
        self.assertEqual(summaries[0], "正常长度。")
        self.assertLessEqual(len(summaries[1]), 50)

    def test_dedupe_memory_cards_keeps_lowest_id(self) -> None:
        # 正常写卡有 (group_openid, content) 唯一索引挡着，造不出重复。模拟历史脏数据：
        # 先把唯一索引摘掉，插两条重复的，再让 dedupe 收。
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="CL", content="重复的事实"
        )
        with self.store._conn:
            self.store._conn.execute("DROP INDEX idx_memory_cards_content")
            for _ in range(2):
                self.store._conn.execute(
                    "INSERT INTO memory_cards (group_openid, kind, person, content)"
                    " VALUES (?, 'fact', 'CL', '重复的事实')",
                    (GROUP,),
                )
        self.assertEqual(self.store.dedupe_memory_cards(GROUP), 2)
        contents = [c.content for c in self.store.memory_cards(GROUP)]
        self.assertEqual(contents, ["重复的事实"])


class SummariesContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "ctx.sqlite3"))
        self.client = XiaoxiaClient(
            settings=SimpleNamespace(timezone="Asia/Shanghai", max_context_chars=4000),
            store=self.store,
            llm=SimpleNamespace(),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_takes_latest_two(self) -> None:
        for start, text in ((1, "第一段"), (11, "第二段"), (21, "第三段")):
            self.store.save_memory_batch(
                group_openid=GROUP,
                start_row_id=start,
                end_row_id=start + 9,
                summary=text,
                memories=[],
            )
        rendered = self.client._summaries_context(GROUP, limit=2, max_chars=2000)
        self.assertIn("第二段", rendered)
        self.assertIn("第三段", rendered)
        self.assertNotIn("第一段", rendered)

    def test_caps_total_chars(self) -> None:
        self.store.save_memory_batch(
            group_openid=GROUP,
            start_row_id=1,
            end_row_id=10,
            summary="很长的摘要" * 100,
            memories=[],
        )
        rendered = self.client._summaries_context(GROUP, limit=2, max_chars=120)
        self.assertLessEqual(len(rendered), 120)

    def test_memory_context_has_no_summary_prefix(self) -> None:
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="CL", content="CL 爱吃辣"
        )
        rendered = self.client._memory_context(GROUP)
        self.assertNotIn("[摘要]", rendered)
        self.assertIn("CL 爱吃辣", rendered)


class ReplyPromptOrderTests(unittest.TestCase):
    def test_request_sits_right_after_topic_and_summary_is_separate(self) -> None:
        captured: dict = {}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def fake_chat(**kwargs):
            captured.update(kwargs)
            return "好"

        settings = SimpleNamespace(
            llm_api_key="k",
            llm_base_url="https://example.invalid",
            model="big-model",
            care_model="",
            vision_model="cheap-model",
            reply_max_chars=900,
        )
        llm = XiaoxiaLLM(settings)
        llm._chat = fake_chat
        try:
            loop.run_until_complete(
                llm.generate_reply(
                    requester=_member("CL"),
                    request="说说你主人",
                    topic_context="2026-10-17 19:00:23 【小虾主人】都不关心我",
                    summaries_context="[摘要] 一段剧情梗概",
                    memory_context="[fact] CL: 用户叫蔡乐",
                )
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        prompt = captured["user_prompt"]
        topic_at = prompt.index("【当前话题】")
        request_at = prompt.index("本次要回应的是")
        summary_at = prompt.index("【最近聊天概要】")
        memory_at = prompt.index("【长期记忆】")
        self.assertLess(topic_at, request_at)
        self.assertLess(request_at, summary_at)
        self.assertLess(summary_at, memory_at)
        self.assertIn("一段剧情梗概", prompt.split("【长期记忆】")[0])
        self.assertNotIn("一段剧情梗概", prompt.split("【长期记忆】")[1])


if __name__ == "__main__":
    unittest.main()
