"""小毛豆用的节日表和农历换算。

只做两件事：
① 给意图解析注入“未来 30 天有哪些节日”，让它不用猜日期；
② 回答“中秋节是几号”“下个节日是什么”这类查询（calendar_query）。
不在后台自动发祝福，具体日期一律由群里 @ 指令写进记忆或任务。

农历换算优先用 zhdate（覆盖 1900-01-01 至 2100-12-29），装不上退 lunardate，
两个都没有就返回 None，由调用方回答“暂时算不出来”。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta


try:  # 两个库都没有时功能自动降级，不影响机器人启动
    from zhdate import ZhDate as _ZhDate
except ImportError:  # pragma: no cover
    _ZhDate = None

try:
    from lunardate import LunarDate as _LunarDate
except ImportError:  # pragma: no cover
    _LunarDate = None


# 公历固定节日（含 520、双十一这类网络节日）
SOLAR_FESTIVALS: dict[tuple[int, int], str] = {
    (1, 1): "元旦",
    (2, 14): "情人节",
    (3, 8): "妇女节",
    (5, 1): "劳动节",
    (5, 4): "青年节",
    (5, 20): "520",
    (6, 1): "儿童节",
    (9, 10): "教师节",
    (10, 1): "国庆节",
    (11, 11): "双十一",
    (12, 24): "平安夜",
    (12, 25): "圣诞节",
}

# 农历节日，key 是（农历月, 农历日）
LUNAR_FESTIVALS: dict[tuple[int, int], str] = {
    (1, 1): "春节",
    (1, 15): "元宵节",
    (2, 2): "龙抬头",
    (5, 5): "端午节",
    (7, 7): "七夕",
    (7, 15): "中元节",
    (8, 15): "中秋节",
    (9, 9): "重阳节",
    (12, 8): "腊八节",
    (12, 23): "小年",
}

# 清明是节气，公历在 4/4 与 4/5 之间摆动。2000-2099 的通用公式是
#   日 = int(年份末两位 * 0.2422 + 4.81) - int(年份末两位 / 4)
# 下面这张表按它算出，并和公开的 2020-2030 清明对照表核对过（2026=4/5，2030=4/5），
# 存成静态表就不必在运行时再算一遍。表外的年份直接跳过清明。
QINGMING_DAYS: dict[int, int] = {
    2026: 5, 2027: 5, 2028: 4, 2029: 4, 2030: 5,
    2031: 5, 2032: 4, 2033: 4, 2034: 5, 2035: 5,
    2036: 4, 2037: 4, 2038: 5, 2039: 5, 2040: 4,
}

_CN_NUMS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
_LUNAR_MONTH_NAMES = (
    "正", "二", "三", "四", "五", "六", "七", "八", "九", "十", "冬", "腊",
)


def lunar_engine() -> str:
    """当前可用的农历换算引擎名，空串表示一个都没装。"""
    if _ZhDate is not None:
        return "zhdate"
    if _LunarDate is not None:
        return "lunardate"
    return ""


def lunar_month_name(month: int, *, leap: bool = False) -> str:
    if not 1 <= month <= 12:
        return f"{month}月"
    return f"{'闰' if leap else ''}{_LUNAR_MONTH_NAMES[month - 1]}月"


def lunar_day_name(day: int) -> str:
    if not 1 <= day <= 30:
        return str(day)
    if day <= 10:
        return "初" + _CN_NUMS[day - 1]
    if day < 20:
        return "十" + _CN_NUMS[day - 11]
    if day == 20:
        return "二十"
    if day < 30:
        return "廿" + _CN_NUMS[day - 21]
    return "三十"


def lunar_label(month: int, day: int, *, leap: bool = False) -> str:
    return f"{lunar_month_name(month, leap=leap)}{lunar_day_name(day)}"


def _convert(year: int, month: int, day: int, leap: bool) -> date | None:
    if _ZhDate is not None:
        try:
            return _ZhDate(year, month, day, leap).to_datetime().date()
        except Exception:  # 日期不存在（比如该年没有这个闰月）或超出覆盖范围
            return None
    if _LunarDate is not None:
        try:
            return _LunarDate(year, month, day, leap).to_solar_date()
        except Exception:
            return None
    return None


def lunar_to_solar(year: int, month: int, day: int, *, leap: bool = False) -> date | None:
    """农历某月初一转成公历。leap=True 但那年没有这个闰月时，退回同名平月。"""
    result = _convert(year, month, day, leap)
    if result is None and leap:
        result = _convert(year, month, day, False)
    return result


def solar_to_lunar(day: date) -> tuple[int, int, int, bool] | None:
    """公历转农历，返回（农历月, 农历日, 是否闰月, 农历年）；算不出来返回 None。"""
    if _ZhDate is None and _LunarDate is None:
        return None
    try:
        if _ZhDate is not None:
            parsed = _ZhDate.from_datetime(datetime(day.year, day.month, day.day))
            return (int(parsed.lunar_month), int(parsed.lunar_day), bool(parsed.leap_month), int(parsed.lunar_year))
        parsed = _LunarDate.from_solar_date(day)
        return (int(parsed.month), int(parsed.day), bool(parsed.isLeapMonth), int(parsed.year))
    except Exception:
        return None


def _eve_of_spring_festival(lunar_year: int) -> date | None:
    """除夕 = 该农历年正月初一的前一天。"""
    spring = lunar_to_solar(lunar_year, 1, 1)
    if spring is None:
        return None
    return spring - timedelta(days=1)


def festivals_on(day: date) -> list[str]:
    """某一天叫什么节（可能同时有好几个，比如除夕和情人节撞一天）。"""
    names: list[str] = []
    solar = SOLAR_FESTIVALS.get((day.month, day.day))
    if solar:
        names.append(solar)
    if day.month == 4 and QINGMING_DAYS.get(day.year) == day.day:
        names.append("清明节")
    lunar = solar_to_lunar(day)
    if lunar:
        month, lday, leap, _ = lunar
        if not leap:
            name = LUNAR_FESTIVALS.get((month, lday))
            if name:
                names.append(name)
    # 除夕 = 第二天就是正月初一；用“次日的农历”倒推，不受公历年份边界影响
    following = solar_to_lunar(day + timedelta(days=1))
    if following and following[0] == 1 and following[1] == 1 and not following[2]:
        names.append("除夕")
    return names


def _candidates(start: date, end: date) -> list[tuple[date, str]]:
    """把区间内的公历/农历/节气节日全列出来，按日期排序。"""
    items: list[tuple[date, str]] = []
    for year in range(start.year, end.year + 2):
        for (month, day), name in SOLAR_FESTIVALS.items():
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if start <= candidate <= end:
                items.append((candidate, name))
        qingming = QINGMING_DAYS.get(year)
        if qingming:
            candidate = date(year, 4, qingming)
            if start <= candidate <= end:
                items.append((candidate, "清明节"))
        for (month, day), name in LUNAR_FESTIVALS.items():
            candidate = lunar_to_solar(year, month, day)
            if candidate and start <= candidate <= end:
                items.append((candidate, name))
        eve = _eve_of_spring_festival(year)
        if eve and start <= eve <= end:
            items.append((eve, "除夕"))
    items.sort(key=lambda pair: pair[0])
    return items


def upcoming_festivals(now: datetime, days: int = 30) -> list[tuple[date, str]]:
    """从今天起 days 天内的节日，按日期升序，同日合并成一个条目。"""
    start = now.date()
    end = start + timedelta(days=days)
    merged: dict[date, list[str]] = {}
    for day, name in _candidates(start, end):
        merged.setdefault(day, [])
        if name not in merged[day]:
            merged[day].append(name)
    return [(day, "、".join(names)) for day, names in sorted(merged.items())]


def next_festival(name: str, now: datetime, *, horizon_days: int = 400) -> tuple[date, str] | None:
    """按名字模糊找下一次发生的节日，用来回答“中秋几号”。"""
    needle = (name or "").strip()
    if not needle:
        return None
    start = now.date()
    items = _candidates(start, start + timedelta(days=horizon_days))
    for day, festival in items:
        for part in festival.split("、"):
            if needle in part or part in needle or needle.replace("节", "") == part.replace("节", ""):
                return (day, part)
    return None


def format_festivals(items: list[tuple[date, str]], *, today: date | None = None) -> str:
    """给意图解析用的一行文本：1月1日元旦（3天）……"""
    if not items:
        return ""
    today = today or items[0][0]
    parts: list[str] = []
    for day, name in items:
        parts.append(f"{day.month}月{day.day}日{name}（还有{(day - today).days}天）")
    return "、".join(parts)
