from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xiaoxia.store import ChatStore


GROUP = "test-group"


class MemoryInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "mem.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _fill(self, count: int, *, kind: str = "fact") -> list[int]:
        # content 上有 UNIQUE(group_openid, content)，所以不同 kind 的占位内容也要不一样，
        # 否则第二遍开始就全被 INSERT OR IGNORE 吃掉了。
        ids = []
        for index in range(1, count + 1):
            card = self.store.add_memory_card(
                group_openid=GROUP,
                kind=kind,
                person="两人",
                content=f"{kind}第{index}条记忆",
            )
            ids.append(card.id if card else 0)
        return ids

    def test_oldest_cards_stay_in_the_prompt(self) -> None:
        # P0-1 的根因：老做法是 memory_cards(limit=30)，id 倒序取最新 30 条，
        # 卡片一多，最前面那批（真名、纪念日、称呼、人设）一条都进不了上下文。
        self._fill(1)
        base = self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        self._fill(62)
        self.assertEqual(self.store.count_live_memory_cards(GROUP), 63)

        legacy = [card.content for card in self.store.memory_cards(GROUP, limit=30)]
        self.assertNotIn("fact第1条记忆", legacy)
        self.assertNotIn("她的真名叫小虾", legacy)

        prompt_cards = self.store.memory_cards_for_prompt(GROUP, total=40, per_kind=6)
        contents = [card.content for card in prompt_cards]
        self.assertIn("fact第1条记忆", contents)
        self.assertIn("fact第62条记忆", contents)
        self.assertLessEqual(len(prompt_cards), 40)
        self.assertEqual([card.id for card in prompt_cards], sorted(card.id for card in prompt_cards))

    def test_pinned_beats_every_quota_and_cap(self) -> None:
        self._fill(60)
        target = self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        self.store.set_memory_pinned(GROUP, target.id, True)
        contents = [
            card.content
            for card in self.store.memory_cards_for_prompt(GROUP, total=4, per_kind=1)
        ]
        self.assertIn("她的真名叫小虾", contents)

    def test_kind_quotas(self) -> None:
        self._fill(20, kind="fact")
        self._fill(20, kind="preference")
        self._fill(20, kind="event")
        picked = self.store.memory_cards_for_prompt(GROUP, total=40, per_kind=6, event_cards=5)
        kinds: dict[str, int] = {}
        for card in picked:
            kinds[card.kind] = kinds.get(card.kind, 0) + 1
        self.assertEqual(kinds, {"fact": 6, "preference": 6, "event": 5})

    def test_pinned_cards_always_win(self) -> None:
        self._fill(50)
        for memory_id in (1, 2, 3):
            self.assertTrue(self.store.set_memory_pinned(GROUP, memory_id, True))
        picked = self.store.memory_cards_for_prompt(GROUP, total=10, per_kind=2)
        self.assertTrue(all(card.pinned for card in picked[:3]))
        self.assertEqual([card.id for card in picked[:3]], [1, 2, 3])
        self.assertEqual(
            self.store.pinned_memory_contents(GROUP),
            ["fact第1条记忆", "fact第2条记忆", "fact第3条记忆"],
        )

    def test_temporary_memories_expire(self) -> None:
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="两人", content="永久的",
        )
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="两人", content="临时的",
            scope="temporary", expires_at="2026-10-20T00:00:00+08:00",
        )
        before = self.store.memory_cards_for_prompt(GROUP, now_iso="2026-10-14T00:00:00+08:00")
        self.assertEqual({card.content for card in before}, {"永久的", "临时的"})
        after = self.store.memory_cards_for_prompt(GROUP, now_iso="2026-10-21T00:00:00+08:00")
        self.assertEqual([card.content for card in after], ["永久的"])

    def test_pinned_flag_shows_up_in_text(self) -> None:
        card = self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        self.store.set_memory_pinned(GROUP, card.id, True)
        text = ChatStore.format_memory_cards(self.store.memory_cards_for_prompt(GROUP))
        self.assertIn("★", text)

    def test_pagination_orders_pinned_first_then_oldest(self) -> None:
        self._fill(30)
        self.store.set_memory_pinned(GROUP, 27, True)
        first, current, pages, total = self.store.memory_cards_page(GROUP, page=1, page_size=12)
        self.assertEqual((current, pages, total), (1, 3, 30))
        self.assertEqual(len(first), 12)
        self.assertEqual(first[0].id, 27)
        self.assertEqual(first[1].id, 1)
        last, current, _, _ = self.store.memory_cards_page(GROUP, page=99, page_size=12)
        self.assertEqual(current, 3)
        self.assertEqual(len(last), 6)
