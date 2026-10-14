"""话题切分：决定每轮模型输入里哪些算【当前话题】、哪些只能当【更早的背景】。

话题边界同时是联网搜索能不能用好的前提——指代（“我对象/上次那个”）只有靠
当前话题才能还原成具体检索词，混进旧话题就会搜出垃圾结果。
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from xiaoxia.store import StoredMessage
from xiaoxia.topics import format_topic, split_topic


BASE = datetime(2026, 10, 14, 15, 0)


def _at(minutes: int, text: str, row_id: int, username: str = "CL") -> StoredMessage:
    return StoredMessage(
        username=username,
        member_openid="openid-" + username,
        role="member",
        content=text,
        timestamp=(BASE + timedelta(minutes=minutes)).isoformat(),
        source="GROUP_MESSAGE_CREATE",
        row_id=row_id,
    )


def _chain(count: int, *, step_minutes: int, start_row: int = 1) -> list[StoredMessage]:
    return [
        _at(step_minutes * index, f"m{index}", start_row + index)
        for index in range(count)
    ]


class SplitTopicTests(unittest.TestCase):
    def test_empty(self) -> None:
        slice_ = split_topic([])
        self.assertEqual(slice_.current, [])
        self.assertEqual(slice_.previous_tail, [])

    def test_time_gap_breaks_topic(self) -> None:
        messages = [
            _at(0, "旧话题1", 1),
            _at(5, "旧话题2", 2),
            _at(60, "新话题1", 3),
            _at(65, "新话题2", 4),
        ]
        slice_ = split_topic(messages)
        self.assertEqual([m.content for m in slice_.current], ["新话题1", "新话题2"])
        self.assertEqual([m.content for m in slice_.previous_tail], ["旧话题1", "旧话题2"])

    def test_max_messages_breaks_topic(self) -> None:
        messages = _chain(45, step_minutes=1)
        slice_ = split_topic(messages)
        self.assertEqual(len(slice_.current), 40)
        self.assertEqual(slice_.current[0].content, "m5")
        self.assertEqual(len(slice_.previous_tail), 5)

    def test_max_hours_breaks_topic(self) -> None:
        messages = _chain(14, step_minutes=60)
        slice_ = split_topic(messages, gap_minutes=90, max_hours=12)
        self.assertEqual(len(slice_.current), 13)
        self.assertEqual([m.content for m in slice_.previous_tail], ["m0"])

    def test_summary_boundary_breaks_topic(self) -> None:
        messages = _chain(5, step_minutes=2, start_row=10)
        slice_ = split_topic(messages, boundary_row_id=12)
        self.assertEqual([m.row_id for m in slice_.current], [13, 14])

    def test_single_message_borrows_previous_tail(self) -> None:
        messages = [
            _at(0, "上一个话题", 1),
            _at(6, "还在聊", 2),
            _at(120, "@ 小毛豆", 3),
        ]
        slice_ = split_topic(messages)
        self.assertEqual([m.content for m in slice_.current], ["上一个话题", "还在聊", "@ 小毛豆"])
        self.assertEqual(slice_.previous_tail, [])

    def test_all_messages_in_one_topic(self) -> None:
        messages = _chain(4, step_minutes=3)
        slice_ = split_topic(messages)
        self.assertEqual(len(slice_.current), 4)
        self.assertEqual(slice_.previous_tail, [])


class FormatTopicTests(unittest.TestCase):
    def test_line_shape_and_char_budget(self) -> None:
        messages = [
            _at(0, "第一条", 1, username="小虾"),
            _at(2, "第二条很长很长很长", 2),
        ]
        full = format_topic(messages, max_chars=1000)
        self.assertIn("小虾(member): 第一条", full)
        self.assertEqual(len(full.splitlines()), 2)

        tight = format_topic(messages, max_chars=60)
        self.assertEqual(len(tight.splitlines()), 1)
        self.assertTrue(len(tight) <= 60)


if __name__ == "__main__":
    unittest.main()
