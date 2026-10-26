"""长回复拆条：自然边界优先，正文不丢，也不把 QQ 表情标签劈开。"""
from __future__ import annotations

import unittest

from maodou.replies import split_reply_text


class SplitReplyTextTests(unittest.TestCase):
    def test_long_text_is_split_without_losing_content(self) -> None:
        text = "。".join(["甲" * 12, "乙" * 13, "丙" * 11, "丁" * 10]) + "。"
        chunks = split_reply_text(text, 20)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_natural_sentence_boundary_is_preferred(self) -> None:
        text = "一二三四五六七八九十。后面还有一段内容。"
        chunks = split_reply_text(text, 12)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "一二三四五六七八九十。")
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))

    def test_paragraph_break_has_priority(self) -> None:
        text = "第一段内容还在继续\n\n第二段内容也会继续"
        chunks = split_reply_text(text, 16)

        self.assertEqual(chunks[0], "第一段内容还在继续")
        self.assertTrue(chunks[1].startswith("第二段内容"))

    def test_hard_cut_does_not_split_a_qq_face_tag(self) -> None:
        tag = '<faceType=1,faceId="267">'
        text = ("甲" * 890) + tag
        chunks = split_reply_text(text, 900)

        self.assertEqual(chunks[0], "甲" * 890)
        self.assertEqual(chunks[1], tag)

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(split_reply_text(" \n ", 20), [])


if __name__ == "__main__":
    unittest.main()
