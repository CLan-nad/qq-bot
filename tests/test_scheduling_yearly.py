from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from maodou.scheduling import advance_task, resolve_next_run, schedule_label
from maodou.store import ChatStore, ScheduledTask


TZ = "Asia/Shanghai"
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=ZoneInfo(TZ))


def _task(**overrides):
    base = dict(
        id=1,
        group_openid="g",
        title="t",
        content=None,
        generation_prompt=None,
        schedule_type="yearly",
        next_run_at="2026-12-13T09:00:00+08:00",
        timezone=TZ,
        time_of_day="09:00",
        weekday=None,
        status="active",
        skip_next=False,
        last_sent_at=None,
        failure_count=0,
        created_by_openid="o",
        created_by_name="n",
        month=12,
        day=13,
    )
    base.update(overrides)
    return ScheduledTask(**base)


class YearlySchedulingTests(unittest.TestCase):
    def test_solar_yearly(self) -> None:
        run = resolve_next_run(
            schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="09:00", month=12, day=13
        )
        self.assertEqual(run.isoformat(), "2026-12-13T09:00:00+08:00")
        advanced = advance_task(
            schedule_type="yearly", from_run_at=run.isoformat(), timezone=TZ,
            time_of_day="09:00", weekday=None, month=12, day=13,
        )
        self.assertEqual(advanced.isoformat(), "2027-12-13T09:00:00+08:00")

    def test_solar_yearly_skips_common_years_for_feb_29(self) -> None:
        run = resolve_next_run(
            schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="10:00", month=2, day=29
        )
        self.assertEqual(run.isoformat(), "2028-02-29T10:00:00+08:00")

    def test_lunar_yearly_birthday(self) -> None:
        run = resolve_next_run(
            schedule_type="lunar_yearly", now=NOW, timezone=TZ, time_of_day="08:30", month=11, day=5
        )
        self.assertEqual(run.isoformat(), "2026-12-13T08:30:00+08:00")
        # 农历生日绝不能简单 +1 年：2027 年的冬月初五在公历 12 月 2 日
        advanced = advance_task(
            schedule_type="lunar_yearly", from_run_at=run.isoformat(), timezone=TZ,
            time_of_day="08:30", weekday=None, month=11, day=5,
        )
        self.assertEqual(advanced.isoformat(), "2027-12-02T08:30:00+08:00")

    def test_offset_days_warms_up_before_the_anniversary(self) -> None:
        warm = resolve_next_run(
            schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="09:00",
            month=1, day=1, offset_days=3,
        )
        # 元旦纪念日提前 3 天，跨到上一年 12 月 29 日，年份必须回退成功
        self.assertEqual(warm.isoformat(), "2026-12-29T09:00:00+08:00")
        warm_lunar = resolve_next_run(
            schedule_type="lunar_yearly", now=NOW, timezone=TZ, time_of_day="08:30",
            month=11, day=5, offset_days=3,
        )
        self.assertEqual(warm_lunar.isoformat(), "2026-12-10T08:30:00+08:00")

    def test_offset_days_survives_advance(self) -> None:
        warm = resolve_next_run(
            schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="09:00",
            month=1, day=1, offset_days=3,
        )
        nxt = advance_task(
            schedule_type="yearly", from_run_at=warm.isoformat(), timezone=TZ,
            time_of_day="09:00", weekday=None, month=1, day=1, offset_days=3,
        )
        self.assertEqual(nxt.isoformat(), "2027-12-29T09:00:00+08:00")

    def test_missing_fields_raise_friendly_errors(self) -> None:
        with self.assertRaises(ValueError):
            resolve_next_run(schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="09:00")
        with self.assertRaises(ValueError):
            resolve_next_run(
                schedule_type="yearly", now=NOW, timezone=TZ, time_of_day="09:00", month=13, day=1
            )
        with self.assertRaises(ValueError):
            resolve_next_run(
                schedule_type="lunar_yearly", now=NOW, timezone=TZ, time_of_day="09:00",
                month=11, day=31,
            )
        with self.assertRaises(ValueError):
            resolve_next_run(
                schedule_type="monthly", now=NOW, timezone=TZ, time_of_day="09:00"
            )

    def test_schedule_labels(self) -> None:
        self.assertEqual(schedule_label(_task()), "每年 12-13 09:00")
        self.assertEqual(
            schedule_label(_task(schedule_type="lunar_yearly", month=11, day=5)),
            "每年农历 冬月初五 09:00",
        )
        self.assertEqual(
            schedule_label(_task(offset_days=3)),
            "提前 3 天 每年 12-13 09:00",
        )
        self.assertEqual(schedule_label(_task(schedule_type="daily")), "每天 09:00")
        self.assertEqual(schedule_label(_task(schedule_type="weekly", weekday=5)), "周五 09:00")
        self.assertEqual(
            schedule_label(_task(schedule_type="once", next_run_at="2026-11-01T09:00:00+08:00")),
            "单次 2026-11-01 09:00",
        )


if __name__ == "__main__":
    unittest.main()
