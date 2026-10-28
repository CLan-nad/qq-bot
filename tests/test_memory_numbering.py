"""记忆编号必须按群连续：展示序号与数据库主键解耦。"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from maodou.bot import MaodouClient
from maodou.store import ChatStore


GROUP_A = "group-A"
GROUP_B = "group-B"


def _message(group_openid: str = GROUP_A) -> SimpleNamespace:
    return SimpleNamespace(
        group_openid=group_openid,
        id="m-1",
        username="CL",
        author=SimpleNamespace(member_openid="openid-CL"),
    )


class MemoryNumberingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "numbering.sqlite3"))
        self.client = MaodouClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                task_max_count=20,
                max_context_chars=4000,
                reply_max_chars=500,
            ),
            store=self.store,
            llm=SimpleNamespace(),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def _add(self, group: str, content: str) -> int:
        card = self.store.add_memory_card(
            group_openid=group, kind="fact", person="两人", content=content
        )
        self.assertIsNotNone(card)
        return card.id

    def test_numbers_are_per_group_and_contiguous(self) -> None:
        # 两个群交错写入，数据库 id 是全局自增，展示编号必须各自从 1 开始。
        self._add(GROUP_A, "A1")
        self._add(GROUP_B, "B1")
        self._add(GROUP_A, "A2")
        self._add(GROUP_B, "B2")
        self._add(GROUP_A, "A3")

        self.assertEqual(
            sorted(self.store.memory_card_numbers(GROUP_A).values()), [1, 2, 3]
        )
        self.assertEqual(
            sorted(self.store.memory_card_numbers(GROUP_B).values()), [1, 2]
        )

        cards = self.store.find_memory_cards_by_numbers(GROUP_A, [2])
        self.assertEqual([card.content for card in cards], ["A2"])
        # 群 B 的第 2 条不能被群 A 的编号取到。
        self.assertEqual(
            [card.content for card in self.store.find_memory_cards_by_numbers(GROUP_B, [2])],
            ["B2"],
        )
        self.assertEqual(self.store.find_memory_cards_by_numbers(GROUP_A, [0, -1]), [])

    def test_memory_list_does_not_show_global_ids(self) -> None:
        for index in range(1, 7):
            self._add(GROUP_A, f"A{index}")
        for index in range(1, 21):
            self._add(GROUP_B, f"B{index}")
        self._add(GROUP_A, "A7")
        self._add(GROUP_A, "A8")

        reply = self.client._handle_memory_list({"action": "memory_list"}, _message())
        self.assertIn("共8条", reply)
        for number in range(1, 9):
            self.assertIn(f"\n{number}. [fact]", reply)
        self.assertNotIn("\n27.", reply)

    def test_forget_by_number_only_touches_current_group(self) -> None:
        for index in range(1, 4):
            self._add(GROUP_A, f"A{index}")
        for index in range(1, 3):
            self._add(GROUP_B, f"B{index}")

        reply = self.client._handle_memory_forget(
            {"action": "memory_forget", "memory_id": 2}, _message(GROUP_A)
        )
        self.assertIn("已忘记记忆 2", reply)
        self.assertEqual(
            [card.content for card in self.store.all_memory_cards(GROUP_A)], ["A1", "A3"]
        )
        self.assertEqual(
            [card.content for card in self.store.all_memory_cards(GROUP_B)], ["B1", "B2"]
        )
        # 删除后编号重新连续。
        self.assertEqual(
            sorted(self.store.memory_card_numbers(GROUP_A).values()), [1, 2]
        )


if __name__ == "__main__":
    unittest.main()
