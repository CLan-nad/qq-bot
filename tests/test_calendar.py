from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from maodou import calendar as cal


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=TZ)


class CalendarTests(unittest.TestCase):
    def test_lunar_names(self) -> None:
        self.assertEqual(cal.lunar_month_name(11), "冬月")
        self.assertEqual(cal.lunar_month_name(12), "腊月")
        self.assertEqual(cal.lunar_month_name(4, leap=True), "闰四月")
        self.assertEqual(cal.lunar_day_name(1), "初一")
        self.assertEqual(cal.lunar_day_name(20), "二十")
        self.assertEqual(cal.lunar_day_name(25), "廿五")
        self.assertEqual(cal.lunar_label(11, 5), "冬月初五")

    def test_lunar_conversion_roundtrip(self) -> None:
        self.assertEqual(cal.lunar_to_solar(2026, 11, 5), date(2026, 12, 13))
        self.assertEqual(cal.solar_to_lunar(date(2026, 12, 13)), (11, 5, False, 2026))
        # 该年没有闰冬月时退回平月，不能直接报错
        self.assertEqual(cal.lunar_to_solar(2026, 11, 1, leap=True), date(2026, 12, 9))

    def test_upcoming_festivals(self) -> None:
        items = cal.upcoming_festivals(NOW, days=30)
        self.assertEqual(
            [(day.isoformat(), name) for day, name in items],
            [("2026-10-18", "重阳节"), ("2026-11-11", "双十一")],
        )
        self.assertEqual(
            cal.format_festivals(items, today=NOW.date()),
            "10月18日重阳节（还有4天）、11月11日双十一（还有28天）",
        )

    def test_next_festival_fuzzy_match(self) -> None:
        self.assertEqual(cal.next_festival("中秋", NOW), (date(2027, 9, 15), "中秋节"))
        self.assertEqual(cal.next_festival("除夕", NOW), (date(2027, 2, 5), "除夕"))
        self.assertEqual(cal.next_festival("国庆节", NOW), (date(2027, 10, 1), "国庆节"))
        self.assertIsNone(cal.next_festival("不存在的节日", NOW))
        self.assertIsNone(cal.next_festival("", NOW))

    def test_new_years_eve_follows_the_lunar_calendar(self) -> None:
        # 除夕必须在正月初一的前一天，跨公历年也不能漏（曾经的 off-by-one）
        self.assertIn("除夕", cal.festivals_on(date(2027, 2, 5)))
        self.assertIn("除夕", cal.festivals_on(date(2026, 2, 16)))
        self.assertIn("春节", cal.festivals_on(date(2026, 2, 17)))
        window = cal.upcoming_festivals(datetime(2026, 1, 20, tzinfo=TZ), days=30)
        self.assertIn(date(2026, 2, 16), [day for day, _ in window])

    def test_qingming_uses_the_static_table(self) -> None:
        self.assertIn("清明节", cal.festivals_on(date(2026, 4, 5)))
        self.assertEqual(cal.QINGMING_DAYS[2026], 5)

    def test_conversion_degrades_without_an_engine(self) -> None:
        zhdate, lunardate = cal._ZhDate, cal._LunarDate
        try:
            cal._ZhDate = None
            cal._LunarDate = None
            self.assertIsNone(cal.lunar_to_solar(2026, 11, 5))
            self.assertIsNone(cal.solar_to_lunar(date(2026, 12, 13)))
            self.assertEqual(cal.upcoming_festivals(NOW, days=7), cal.upcoming_festivals(NOW, days=7))
        finally:
            cal._ZhDate, cal._LunarDate = zhdate, lunardate
