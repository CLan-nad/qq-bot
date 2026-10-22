"""话题切分：决定每轮模型输入里哪些算【当前话题】、哪些只能当【更早的背景】。

话题边界同时是联网搜索能不能用好的前提——指代（“我对象/上次那个”）只有靠
当前话题才能还原成具体检索词，混进旧话题就会搜出垃圾结果。
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from maodou.store import StoredMessage
from maodou.topics import format_topic, speaker_label, split_topic


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

    def test_summary_boundary_does_not_cut_live_topic(self) -> None:
        messages = _chain(5, step_minutes=2, start_row=10)
        slice_ = split_topic(messages)
        self.assertEqual([m.row_id for m in slice_.current], [10, 11, 12, 13, 14])

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
        self.assertIn("【小虾】第一条", full)
        self.assertEqual(len(full.splitlines()), 2)

        tight = format_topic(messages, max_chars=60)
        self.assertEqual(len(tight.splitlines()), 1)
        self.assertTrue(len(tight) <= 60)


class SpeakerLabelTests(unittest.TestCase):
    """每条消息都要挂清是谁说的：线上出现过把 A 的话算到 B 头上。"""

    def test_address_wins_over_display_name(self) -> None:
        message = _at(0, "都怪你叫叫叫", 1, username="二十四桥明月夜")
        self.assertEqual(
            speaker_label(message, {"openid-二十四桥明月夜": "小虾主人"}),
            "小虾主人",
        )

    def test_labels_can_be_keyed_by_display_name(self) -> None:
        # 显示名对不上时的兜底：称呼表两个 key 都能查。
        message = _at(0, "在吗", 1, username="CL")
        self.assertEqual(speaker_label(message, {"CL": "主人"}), "主人")

    def test_falls_back_to_display_name(self) -> None:
        message = _at(0, "在吗", 1, username="路人")
        self.assertEqual(speaker_label(message, {}), "路人")

    def test_bot_messages_are_always_the_bot(self) -> None:
        bot = StoredMessage(
            username="小毛豆",
            member_openid="",
            role="bot",
            content="我查一下",
            timestamp=BASE.isoformat(),
            source="BOT_REPLY",
        )
        # 就算登记表里恰好有同名条目，机器人也不能被挂成别人
        self.assertEqual(speaker_label(bot, {"小毛豆": "主人"}), "小毛豆")

    def test_replay_two_messages_from_the_same_sender(self) -> None:
        # 新群 row 835/836 原样回放：两句都是小虾说的，不能算成主人。
        labels = {"openid-二十四桥明月夜": "小虾主人", "openid-CL": "主人"}
        messages = [
            _at(0, "都怪你叫叫叫", 1, username="二十四桥明月夜"),
            _at(1, "@小毛豆 气死我了", 2, username="二十四桥明月夜"),
        ]
        rendered = format_topic(messages, 1000, labels)
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn("【小虾主人】", line)
            self.assertNotIn("【主人】", line)


class ImageNoteRenderingTests(unittest.TestCase):
    """看图的记录要单独成行，不能混成「对方打出来的字」。"""

    def test_image_note_goes_on_its_own_line(self) -> None:
        messages = [
            _at(
                0,
                "八块，我去\n[图片内容：一张俯拍照片，手里的白桃乌龙雪糕]",
                1,
                username="二十四桥明月夜",
            )
        ]
        rendered = format_topic(messages, 1000, {"openid-二十四桥明月夜": "小虾主人"})
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("【小虾主人】八块，我去", lines[0])
        self.assertTrue(lines[1].strip().startswith("〔图〕"))
        self.assertIn("白桃乌龙雪糕", lines[1])
        # 入库标记不该原样进 prompt，否则又变成用户文字
        self.assertNotIn("[图片内容：", rendered)
        # 〔图〕行跟在说话人下面，缩进对齐
        self.assertTrue(lines[1].startswith(" " * 4))

    def test_bot_messages_render_as_the_bot(self) -> None:
        bot = StoredMessage(
            username="小毛豆",
            member_openid="",
            role="bot",
            content="我看看",
            timestamp=BASE.isoformat(),
            source="BOT_REPLY",
        )
        self.assertIn("【小毛豆】我看看", format_topic([bot], 1000))


if __name__ == "__main__":
    unittest.main()
