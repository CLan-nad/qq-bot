from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


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


def resolve_next_run(
    *,
    schedule_type: str,
    now: datetime,
    timezone: str,
    once_datetime: str | None = None,
    time_of_day: str | None = None,
    weekday: int | None = None,
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

    raise ValueError("暂不支持这种重复频率，当前支持单次、每天和每周")


def advance_task(
    *,
    schedule_type: str,
    from_run_at: str,
    timezone: str,
    time_of_day: str | None,
    weekday: int | None,
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

    raise ValueError("暂不支持这种重复频率")


def iso_now(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
