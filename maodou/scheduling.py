from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from .calendar import lunar_label, lunar_to_solar, solar_to_lunar

if TYPE_CHECKING:
    from .store import ScheduledTask


TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def schedule_label(task: "ScheduledTask") -> str:
    """把任务的重复规则翻译成人话。store.format_tasks 和 bot 的回复共用这一份。"""
    prefix = f"提前 {task.offset_days} 天 " if task.offset_days else ""
    at = task.time_of_day or ""
    if task.schedule_type == "once":
        return f"{prefix}单次 {task.next_run_at[:16].replace('T', ' ')}".strip()
    if task.schedule_type == "daily":
        return f"{prefix}每天 {at}".strip()
    if task.schedule_type == "weekly":
        weekday = WEEKDAY_NAMES[task.weekday - 1] if task.weekday and 1 <= task.weekday <= 7 else "每周"
        return f"{prefix}{weekday} {at}".strip()
    if task.schedule_type == "yearly":
        month_day = f"{task.month:02d}-{task.day:02d}" if task.month and task.day else "每年"
        return f"{prefix}每年 {month_day} {at}".strip()
    if task.schedule_type == "lunar_yearly":
        label = lunar_label(task.month or 1, task.day or 1, leap=task.leap_month) if task.month and task.day else "农历"
        return f"{prefix}每年农历 {label} {at}".strip()
    return f"{prefix}{task.schedule_type}".strip()


def parse_time_of_day(value: str | None) -> time | None:
    if not value:
        return None
    match = TIME_RE.match(value.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour, minute)


def parse_datetime(value: str, timezone: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return parsed.astimezone(ZoneInfo(timezone))


def _solar_anniversary(
    year: int, month: int, day: int, at: time, tz: ZoneInfo, offset_days: int
) -> datetime | None:
    """公历某年某月某日的那个时刻；2 月 29 日在平年返回 None，让它自然跳到下个闰年。"""
    try:
        moment = datetime(year, month, day, at.hour, at.minute, tzinfo=tz)
    except ValueError:
        return None
    return moment - timedelta(days=offset_days)


def _lunar_anniversary(
    lunar_year: int, month: int, day: int, leap: bool, at: time, tz: ZoneInfo, offset_days: int
) -> datetime | None:
    """农历某年的月日换算成公历时刻；闰月不存在时 lunar_to_solar 自己退回平月。"""
    solar = lunar_to_solar(lunar_year, month, day, leap=leap)
    if solar is None:
        return None
    moment = datetime(solar.year, solar.month, solar.day, at.hour, at.minute, tzinfo=tz)
    return moment - timedelta(days=offset_days)


def resolve_next_run(
    *,
    schedule_type: str,
    now: datetime,
    timezone: str,
    once_datetime: str | None = None,
    time_of_day: str | None = None,
    weekday: int | None = None,
    month: int | None = None,
    day: int | None = None,
    leap_month: bool = False,
    offset_days: int = 0,
) -> datetime:
    tz = ZoneInfo(timezone)
    now = now.astimezone(tz)

    if schedule_type == "once":
        if not once_datetime:
            raise ValueError("单次任务需要具体时间")
        return parse_datetime(once_datetime, timezone)

    parsed_time = parse_time_of_day(time_of_day)
    if parsed_time is None:
        raise ValueError("重复任务需要 HH:MM 格式的时间")

    if schedule_type == "daily":
        candidate = now.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    if schedule_type == "weekly":
        if weekday is None or not 1 <= weekday <= 7:
            raise ValueError("每周任务需要 1-7 的星期，1 表示周一")
        days_ahead = (weekday - 1 - now.weekday()) % 7
        candidate = now.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        ) + timedelta(days=days_ahead)
        if candidate <= now:
            candidate += timedelta(days=7)
        return candidate

    if schedule_type in {"yearly", "lunar_yearly"}:
        if month is None or day is None:
            raise ValueError("每年任务需要月和日")
        if not 1 <= month <= 12:
            raise ValueError("月份要在 1 到 12 之间")
        if schedule_type == "yearly":
            if not 1 <= day <= 31:
                raise ValueError("日期要在 1 到 31 之间")
            for year in range(now.year, now.year + 10):
                candidate = _solar_anniversary(year, month, day, parsed_time, tz, offset_days)
                if candidate and candidate > now:
                    return candidate
        else:
            current = solar_to_lunar(now.date())
            if current is None:
                raise ValueError("农历暂时算不出来，服务器上没装 zhdate")
            if not 1 <= day <= 30:
                raise ValueError("农历日期要在 1 到 30 之间")
            lunar_year_now = current[3]
            for lunar_year in range(lunar_year_now - 1, lunar_year_now + 6):
                candidate = _lunar_anniversary(
                    lunar_year, month, day, leap_month, parsed_time, tz, offset_days
                )
                if candidate and candidate > now:
                    return candidate
        raise ValueError("算不出这个每年任务的下一次时间")

    raise ValueError("暂不支持这种重复频率，当前支持单次、每天、每周、每年和农历每年")


def advance_task(
    *,
    schedule_type: str,
    from_run_at: str,
    timezone: str,
    time_of_day: str | None,
    weekday: int | None,
    month: int | None = None,
    day: int | None = None,
    leap_month: bool = False,
    offset_days: int = 0,
) -> datetime | None:
    if schedule_type == "once":
        return None

    previous = parse_datetime(from_run_at, timezone)
    parsed_time = parse_time_of_day(time_of_day)
    if parsed_time is None:
        raise ValueError("重复任务缺少发送时间")

    if schedule_type == "daily":
        return previous.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        ) + timedelta(days=1)

    if schedule_type == "weekly":
        if weekday is None or not 1 <= weekday <= 7:
            raise ValueError("每周任务缺少星期")
        days_ahead = (weekday - 1 - previous.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return previous.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        ) + timedelta(days=days_ahead)

    if schedule_type in {"yearly", "lunar_yearly"}:
        # 农历生日、2 月 29 日都不是“简单加一年”，从上次运行时间的次日起重算下一次
        return resolve_next_run(
            schedule_type=schedule_type,
            now=previous + timedelta(days=1),
            timezone=timezone,
            time_of_day=time_of_day,
            month=month,
            day=day,
            leap_month=leap_month,
            offset_days=offset_days,
        )

    raise ValueError("暂不支持这种重复频率")


def iso_now(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
