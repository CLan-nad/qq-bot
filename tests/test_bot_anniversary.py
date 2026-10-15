"""机器人层的生日/纪念日、记忆分页、节日查询处理逻辑。

_now() 一律钉在 2026-10-14 15:00，否则这些断言会随真实日期漂。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from xiaoxia.bot import XiaoxiaClient
from xiaoxia.store import ChatStore


GROUP = "test-group"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=TZ)


def _message() -> SimpleNamespace:
    return SimpleNamespace(
        group_openid=GROUP,
        id="m-1",
        username="CL",
        author=SimpleNamespace(member_openid="openid-CL"),
    )


class BotAnniversaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(str(Path(self.tmp.name) / "bot.sqlite3"))
        self.client = XiaoxiaClient(
            settings=SimpleNamespace(
                timezone="Asia/Shanghai",
                task_max_count=20,
                max_context_chars=4000,
                reply_max_chars=500,
            ),
            store=self.store,
            llm=SimpleNamespace(),
        )
        self.client._now = lambda: NOW

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()
        asyncio.set_event_loop(None)
        self.loop.close()

    def test_solar_anniversary_creates_warmup_task(self) -> None:
        reply = self.client._handle_task_create(
            {
                "action": "task_create",
                "schedule_type": "yearly",
                "month": 12,
                "day": 13,
                "time": "09:00",
                "remind_days_ahead": 3,
                "content": "在一起两年啦",
                "title": "纪念日",
            },
            _message(),
        )
        self.assertIn("每年 12-13 09:00", reply)
        self.assertIn("提前 3 天", reply)

        tasks = {task.title: task for task in self.store.tasks(GROUP)}
        self.assertEqual(len(tasks), 2)
        main = tasks["纪念日"]
        warm = tasks["[预热] 纪念日"]
        self.assertEqual(main.next_run_at, "2026-12-13T09:00:00+08:00")
        self.assertEqual(main.remind_days_ahead, 3)
        self.assertEqual(warm.next_run_at, "2026-12-10T09:00:00+08:00")
        self.assertEqual(warm.offset_days, 3)
        self.assertEqual(warm.linked_task_id, main.id)
        # 预热文案是固定的，不走模型，省 token 也省得翻车
        self.assertEqual(warm.content, "纪念日 还有 3 天，就是 12月13日，记得提前准备一下。")
        self.assertIsNone(warm.generation_prompt)

        self.store.delete_task(GROUP, main.id)
        self.assertEqual([task.title for task in self.store.tasks(GROUP)], [])

    def test_lunar_birthday_creates_lunar_warmup_task(self) -> None:
        self.client._handle_task_create(
            {
                "action": "task_create",
                "schedule_type": "lunar_yearly",
                "month": 11,
                "day": 5,
                "time": "08:30",
                "remind_days_ahead": 3,
                "content": "生日快乐",
                "title": "小虾生日",
            },
            _message(),
        )
        tasks = {task.title: task for task in self.store.tasks(GROUP)}
        self.assertEqual(tasks["小虾生日"].next_run_at, "2026-12-13T08:30:00+08:00")
        self.assertEqual(tasks["[预热] 小虾生日"].next_run_at, "2026-12-10T08:30:00+08:00")
        self.assertIn("每年农历 冬月初五 08:30", self.store.format_tasks(list(tasks.values())))

    def test_once_task_never_gets_a_warmup_task(self) -> None:
        self.client._handle_task_create(
            {
                "action": "task_create",
                "schedule_type": "once",
                "datetime": "2026-10-20T09:00:00",
                "remind_days_ahead": 2,
                "content": "取快递",
                "title": "取快递",
            },
            _message(),
        )
        self.assertEqual(len(self.store.tasks(GROUP)), 1)

    def test_unknown_schedule_type_is_refused(self) -> None:
        reply = self.client._handle_task_create(
            {"action": "task_create", "schedule_type": "monthly", "time": "09:00", "content": "x"},
            _message(),
        )
        self.assertIn("只支持单次、每天、每周、每年、农历每年", reply)
        self.assertEqual(self.store.tasks(GROUP), [])

    def test_lunar_without_engine_reports_friendly_error(self) -> None:
        import xiaoxia.calendar as cal

        zhdate, lunardate = cal._ZhDate, cal._LunarDate
        try:
            cal._ZhDate = None
            cal._LunarDate = None
            reply = self.client._handle_task_create(
                {
                    "action": "task_create",
                    "schedule_type": "lunar_yearly",
                    "month": 11,
                    "day": 5,
                    "time": "08:30",
                    "content": "生日快乐",
                },
                _message(),
            )
        finally:
            cal._ZhDate, cal._LunarDate = zhdate, lunardate
        self.assertIn("任务时间有点问题", reply)

    def test_memory_remember_can_pin_and_expire(self) -> None:
        identity = SimpleNamespace(
            requester=SimpleNamespace(username="CL"),
            partner=SimpleNamespace(username="对象"),
        )
        self.client._handle_memory_remember(
            {
                "memories": [
                    {"kind": "fact", "person": "她", "content": "她的真名叫小虾", "pin": True},
                    {"kind": "fact", "person": "两人", "content": "周五要修电脑", "temporary": True},
                ]
            },
            _message(),
            identity,
        )
        cards = {card.content: card for card in self.store.memory_cards(GROUP)}
        self.assertTrue(cards["她的真名叫小虾"].pinned)
        self.assertEqual(cards["她的真名叫小虾"].scope, "permanent")
        self.assertIsNone(cards["她的真名叫小虾"].expires_at)
        self.assertEqual(cards["周五要修电脑"].scope, "temporary")
        self.assertEqual(cards["周五要修电脑"].expires_at, "2026-10-28T15:00:00+08:00")
        self.assertFalse(cards["周五要修电脑"].pinned)

    def test_memory_list_paginates(self) -> None:
        for index in range(1, 31):
            self.store.add_memory_card(
                group_openid=GROUP, kind="fact", person="两人", content=f"记忆{index:02d}"
            )
        self.store.set_memory_pinned(GROUP, 7, True)
        reply = self.client._handle_memory_list({"action": "memory_list"}, _message())
        self.assertIn("第1/3页，共30条，★=永久", reply)
        self.assertIn("★[fact]", reply)
        self.assertIn("第 2 页记忆", reply)
        page3 = self.client._handle_memory_list({"page": 3}, _message())
        self.assertIn("第3/3页", page3)
        self.assertNotIn("下一页", page3)

    def test_memory_pin_and_unpin_by_match(self) -> None:
        card = self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="她", content="她的真名叫小虾"
        )
        reply = self.client._handle_memory_pin(
            {"match": "真名"}, _message(), pinned=True
        )
        self.assertIn(f"已置顶，以后不会忘记忆 {card.id}", reply)
        self.assertTrue(self.store.find_memory_card_by_id(GROUP, card.id).pinned)
        reply = self.client._handle_memory_pin(
            {"memory_id": card.id}, _message(), pinned=False
        )
        self.assertIn("已取消置顶", reply)
        self.assertFalse(self.store.find_memory_card_by_id(GROUP, card.id).pinned)
        self.assertIn("没有找到这条记忆", self.client._handle_memory_pin(
            {"match": "不存在的东西"}, _message(), pinned=True))

    def test_calendar_query(self) -> None:
        named = self.client._handle_calendar_query({"query": "重阳"})
        self.assertIn("重阳节是 2026年10月18日", named)
        self.assertIn("还有 4 天", named)
        self.assertIn("农历九月初九", named)
        listing = self.client._handle_calendar_query({})
        self.assertIn("未来 30 天的节日：", listing)
        self.assertIn("双十一", listing)
        self.assertIn("没查到", self.client._handle_calendar_query({"query": "万圣节不存在"}))

    def test_memory_alias_show_set_clear(self) -> None:
        # 同一个人被记成「小虾/罗晓霞/二十四桥明月夜」时模型会当成三个人，称呼跟着乱。
        # memory_alias 统一叫法，且不写死任何名字，统一成什么全由群里 @ 指令决定。
        reply = self.client._handle_memory_alias({"action": "memory_alias"}, _message())
        self.assertIn("还没统一过叫法", reply)

        self.store.add_memory_card(
            group_openid=GROUP, kind="fact", person="小虾", content="小虾爱吃辣"
        )
        reply = self.client._handle_memory_alias(
            {
                "action": "memory_alias",
                "op": "set",
                "alias": ["小虾", "罗晓霞"],
                "canonical": "二十四桥明月夜",
            },
            _message(),
        )
        self.assertIn("都统一叫", reply)
        self.assertIn("二十四桥明月夜", reply)
        self.assertEqual(
            [c.person for c in self.store.memory_cards(GROUP) if c.content == "小虾爱吃辣"],
            ["二十四桥明月夜"],
        )

        reply = self.client._handle_memory_alias({"action": "memory_alias", "op": "show"}, _message())
        self.assertIn("小虾 → 二十四桥明月夜", reply)

        reply = self.client._handle_memory_alias(
            {"action": "memory_alias", "op": "clear", "alias": "小虾"}, _message()
        )
        self.assertIn("不再统一了", reply)
        self.assertEqual(self.store.canonical_person(GROUP, "小虾"), "小虾")

    def test_memory_alias_needs_canonical(self) -> None:
        reply = self.client._handle_memory_alias(
            {"action": "memory_alias", "op": "set", "alias": "小虾"}, _message()
        )
        self.assertIn("要统一成哪个叫法", reply)


if __name__ == "__main__":
    unittest.main()
