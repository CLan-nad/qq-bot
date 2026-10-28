from __future__ import annotations

import unittest

from maodou.completion import (
    completion_is_incomplete,
    continuation_prompt,
    merge_continuation,
)


class CompletionRepairTests(unittest.TestCase):
    def test_length_reason_is_always_incomplete(self) -> None:
        self.assertTrue(completion_is_incomplete("已经写完。", "length"))
        self.assertTrue(completion_is_incomplete("已经写完。", "max_tokens"))

    def test_normal_short_chat_is_complete_without_punctuation(self) -> None:
        self.assertFalse(completion_is_incomplete("好呀", "stop"))
        self.assertFalse(completion_is_incomplete("哈哈哈", None))

    def test_obvious_dangling_marks_are_incomplete(self) -> None:
        for text in ("时间线：", "先去吃饭，", "你 →", "引用（没闭合"):
            with self.subTest(text=text):
                self.assertTrue(completion_is_incomplete(text, "stop"))

    def test_merge_removes_repeated_overlap_only(self) -> None:
        self.assertEqual(
            merge_continuation("他去图书馆", "图书馆待了两小时。"),
            "他去图书馆待了两小时。",
        )
        self.assertEqual(
            merge_continuation("前半句。", "后半句。"),
            "前半句。后半句。",
        )

    def test_merge_strips_continuation_label(self) -> None:
        self.assertEqual(
            merge_continuation("前半句。", "续写：后半句。"),
            "前半句。后半句。",
        )

    def test_continuation_prompt_distinguishes_empty_output(self) -> None:
        self.assertIn("从断点直接续写", continuation_prompt("写到一半"))
        self.assertIn("从头完整回答", continuation_prompt(""))


if __name__ == "__main__":
    unittest.main()
