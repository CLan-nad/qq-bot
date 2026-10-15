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
        # 新做法按字符预算装（默认 6000 字/160 条），短卡片全进，且按 id 正序输出。
        self._fill(1)
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        self._fill(62)
        self.assertEqual(self.store.count_live_memory_cards(GROUP), 63)

        legacy = [card.content for card in self.store.memory_cards(GROUP, limit=30)]
        self.assertNotIn("fact第1条记忆", legacy)
        self.assertNotIn("她的真名叫小虾", legacy)

        prompt_cards = self.store.memory_cards_for_prompt(GROUP)
        contents = [card.content for card in prompt_cards]
        self.assertIn("fact第1条记忆", contents)
        self.assertIn("她的真名叫小虾", contents)
        self.assertIn("fact第62条记忆", contents)
        self.assertEqual(len(prompt_cards), 63)
        self.assertEqual([card.id for card in prompt_cards], sorted(card.id for card in prompt_cards))

    def test_pinned_beats_every_quota_and_cap(self) -> None:
        self._fill(60)
        target = self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        self.store.set_memory_pinned(GROUP, target.id, True)
        contents = [
            card.content
            for card in self.store.memory_cards_for_prompt(GROUP, total=4)
        ]
        self.assertIn("她的真名叫小虾", contents)

    def test_rules_beat_episodic_when_budget_is_tight(self) -> None:
        # 旧版按 kind 配额（preference 只给 6 条），可群里实际攒了 25 条，被挤掉的恰好是
        # 「别分点」「别引 4399」这些行为规则。新版不按配额，按「规则优先、流水账靠后」装：
        # 预算不够时先丢 fact/event（流水账，新的更相关），preference/agreement/relationship
        # （人设地基，老的更重要）必须留下。
        for index in range(1, 6):
            self.store.add_memory_card(
                group_openid=GROUP, kind="preference", person="两人", content=f"规则第{index}条"
            )
        for index in range(1, 21):
            self.store.add_memory_card(
                group_openid=GROUP, kind="fact", person="两人", content=f"流水第{index}条"
            )
        picked = self.store.memory_cards_for_prompt(GROUP, total=8)
        kinds: dict[str, int] = {}
        for card in picked:
            kinds[card.kind] = kinds.get(card.kind, 0) + 1
        self.assertEqual(kinds.get("preference"), 5)
        self.assertEqual(kinds.get("fact"), 3)
        fact_contents = [card.content for card in picked if card.kind == "fact"]
        self.assertIn("流水第20条", fact_contents)
        self.assertNotIn("流水第1条", fact_contents)

    def test_pinned_cards_always_win(self) -> None:
        self._fill(50)
        for memory_id in (1, 2, 3):
            self.assertTrue(self.store.set_memory_pinned(GROUP, memory_id, True))
        picked = self.store.memory_cards_for_prompt(GROUP, total=10)
        self.assertTrue(all(card.pinned for card in picked[:3]))
        self.assertEqual([card.id for card in picked[:3]], [1, 2, 3])
        self.assertEqual(
            self.store.pinned_memory_contents(GROUP),
            ["fact第1条记忆", "fact第2条记忆", "fact第3条记忆"],
        )

    def test_duplicate_cards_are_collapsed(self) -> None:
        # 抽取器把同一件事存了好几遍（标点/空格不同也算重复），注入前只留一张，置顶的那张优先。
        self.store.add_memory_card(
            group_openid=GROUP, kind="preference", person="两人", content="回复不要分点，直接说"
        )
        dup = self.store.add_memory_card(
            group_openid=GROUP, kind="preference", person="两人", content="回复不要分点直接说。"
        )
        self.assertIsNotNone(dup)
        self.store.set_memory_pinned(GROUP, dup.id, True)
        picked = self.store.memory_cards_for_prompt(GROUP)
        matches = [card for card in picked if "分点" in card.content]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].pinned)

    def test_existing_memory_lines_lists_everything(self) -> None:
        # 去重清单要给全量（不只置顶），否则抽取器看不到已存的条目会重复抽。
        self.store.add_memory_card(
            group_openid=GROUP, kind="preference", person="两人", content="回复别分点"
        )
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="CL", content="CL 爱吃辣"
        )
        lines = self.store.existing_memory_lines(GROUP)
        joined = "\n".join(lines)
        self.assertIn("回复别分点", joined)
        self.assertIn("CL 爱吃辣", joined)
        self.assertTrue(all(line.startswith("- [") for line in lines))

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

    def test_find_memory_cards_accepts_id_list(self) -> None:
        # 用户说“把 3、7、12 一起设为永久”时，intent 里的 memory_id 是数组。
        # 老代码只认单个 int，数组被当成“没给编号”，于是每次都答“没有找到这条记忆”。
        self._fill(15)
        cards = self.store.find_memory_cards(GROUP, memory_id=[3, 7, 12])
        self.assertEqual([card.id for card in cards], [3, 7, 12])
        self.assertEqual(
            [card.id for card in self.store.find_memory_cards(GROUP, memory_id=7)], [7]
        )
        self.assertEqual(self.store.find_memory_cards(GROUP, memory_id=[]), [])
        # 脏值（模型偶尔把编号写成字符串或塞 null）要丢掉，不能抛异常。
        mixed = self.store.find_memory_cards(GROUP, memory_id=["7", None, 9999])
        self.assertEqual([card.id for card in mixed], [7])

    def test_memory_cards_page_filters(self) -> None:
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="小虾", content="她不吃香菜", pinned=True
        )
        self.store.add_memory_card(
            group_openid=GROUP, kind="preference", person="小虾", content="她喜欢喝冰美式"
        )
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="蔡乐", content="蔡乐周三要值班"
        )

        rows, _, _, total = self.store.memory_cards_page(GROUP, pinned=True)
        self.assertEqual((total, [card.id for card in rows]), (1, [1]))

        rows, _, _, total = self.store.memory_cards_page(GROUP, pinned=False)
        self.assertEqual((total, sorted(card.id for card in rows)), (2, [2, 3]))

        rows, _, _, total = self.store.memory_cards_page(GROUP, kind="fact")
        self.assertEqual((total, sorted(card.id for card in rows)), (2, [1, 3]))

        rows, _, _, total = self.store.memory_cards_page(GROUP, person="小虾")
        self.assertEqual((total, sorted(card.id for card in rows)), (2, [1, 2]))

        # match 走 INSTR 子串匹配（记忆卡没有 FTS 分词），中文子串能直接命中。
        rows, _, _, total = self.store.memory_cards_page(GROUP, match="冰美式")
        self.assertEqual((total, [card.id for card in rows]), (1, [2]))

        rows, _, _, total = self.store.memory_cards_page(GROUP, match="小虾")
        self.assertEqual((total, sorted(card.id for card in rows)), (2, [1, 2]))

        rows, _, _, total = self.store.memory_cards_page(GROUP, match="没有这个内容")
        self.assertEqual((total, rows), (0, []))

    def test_person_filter_normalizes_alias(self) -> None:
        # 群里统一过叫法之后，用旧叫法筛选也要查得到（先归一再匹配）。
        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="二十四桥明月夜", content="她的真名叫小虾"
        )
        self.store.set_person_alias(GROUP, "二十四桥明月夜", "小虾")
        rows, _, _, total = self.store.memory_cards_page(GROUP, person="二十四桥明月夜")
        self.assertEqual((total, [card.person for card in rows]), (1, ["小虾"]))


class PersonAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "alias.sqlite3"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_set_person_alias_rewrites_existing_cards(self) -> None:
        # 群里说「以后把小虾和罗晓霞都叫二十四桥明月夜」，已存的卡片 person 要一起改，
        # 新卡在写库前也归一，别再往库里添第四种写法。
        self.store.add_memory_card(group_openid=GROUP, kind="fact", person="小虾", content="小虾爱吃辣")
        self.store.add_memory_card(group_openid=GROUP, kind="fact", person="罗晓霞", content="罗晓霞怕冷")
        self.store.add_memory_card(group_openid=GROUP, kind="fact", person="CL", content="CL 爱打游戏")
        self.assertEqual(self.store.set_person_alias(GROUP, "小虾", "二十四桥明月夜"), 1)
        self.assertEqual(self.store.set_person_alias(GROUP, "罗晓霞", "二十四桥明月夜"), 1)
        people = {card.person for card in self.store.memory_cards(GROUP)}
        self.assertEqual(people, {"二十四桥明月夜", "CL"})
        self.assertEqual(self.store.canonical_person(GROUP, "小虾"), "二十四桥明月夜")
        self.store.add_memory_card(group_openid=GROUP, kind="fact", person="小虾", content="小虾的新事")
        self.assertEqual(
            [c.person for c in self.store.memory_cards(GROUP) if c.content == "小虾的新事"],
            ["二十四桥明月夜"],
        )

    def test_alias_chain_is_flattened(self) -> None:
        # A→B→C 应压平成 A→C，不绕圈。
        self.store.set_person_alias(GROUP, "A", "B")
        self.store.set_person_alias(GROUP, "B", "C")
        self.assertEqual(self.store.canonical_person(GROUP, "A"), "C")
        self.assertEqual(self.store.person_aliases(GROUP).get("A"), "C")

    def test_clear_person_alias(self) -> None:
        self.store.set_person_alias(GROUP, "小虾", "二十四桥明月夜")
        self.assertTrue(self.store.clear_person_alias(GROUP, "小虾"))
        self.assertFalse(self.store.clear_person_alias(GROUP, "小虾"))
        self.assertEqual(self.store.canonical_person(GROUP, "小虾"), "小虾")


if __name__ == "__main__":
    unittest.main()
