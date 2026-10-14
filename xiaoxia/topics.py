"""话题切分：把最近的聊天记录分成「当前话题」和「更早的背景」。

以前每轮都把最近几十条消息全量塞进 prompt，模型容易把上一个话题当成当前话题的
延续，于是出现两个问题：回复爱重提旧事；联网搜索时把旧话题的词、甚至「要有依据」
这类对回答形式的要求当成检索词，直接搜出垃圾结果。话题边界比「全量最近记录」更能
代表用户此刻在聊什么，所以补全检索词只能依赖当前话题。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .store import StoredMessage


# 相邻两条消息间隔超过这个分钟数，认为上一个话题已经结束。
DEFAULT_GAP_MINUTES = 45
# 当前话题最多带这么多条消息，再多也切。
DEFAULT_MAX_MESSAGES = 40
# 当前话题的时间跨度上限（小时），防止一个话题无限往前伸。
DEFAULT_MAX_HOURS = 12
# 当前话题只剩一条消息时，从上一个话题末尾借几条来补语境。
FALLBACK_TAIL_SIZE = 5


@dataclass(frozen=True)
class TopicSlice:
    current: list[StoredMessage]
    previous_tail: list[StoredMessage]


def parse_timestamp(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def split_topic(
    messages: list[StoredMessage],
    *,
    boundary_row_id: int = 0,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_hours: int = DEFAULT_MAX_HOURS,
    previous_tail_size: int = 5,
) -> TopicSlice:
    """按时间正序的消息切成当前话题 + 上一话题末尾若干条。

    从最新一条往前扫，命中任意一条就切断：相邻时间差超过 gap_minutes、已攒够
    max_messages 条、跨度超过 max_hours、或 row_id 不大于 boundary_row_id
    （那些消息已经被折叠进对话摘要，不该再算作「正在聊的话题」）。
    """
    if not messages:
        return TopicSlice(current=[], previous_tail=[])

    newest_ts = parse_timestamp(messages[-1].timestamp)
    current: list[StoredMessage] = []
    cursor_ts = newest_ts
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if current:
            if len(current) >= max_messages:
                break
            if boundary_row_id and message.row_id and message.row_id <= boundary_row_id:
                break
            stamp = parse_timestamp(message.timestamp)
            if stamp is not None:
                reference = cursor_ts if cursor_ts is not None else stamp
                if (reference - stamp).total_seconds() / 60.0 > gap_minutes:
                    break
                if newest_ts is not None:
                    span_hours = (newest_ts - stamp).total_seconds() / 3600.0
                    if span_hours > max_hours:
                        break
                cursor_ts = stamp
        current.append(message)

    current.reverse()
    remaining = messages[: len(messages) - len(current)]
    previous_tail = remaining[-previous_tail_size:] if remaining else []

    if len(current) <= 1 and previous_tail:
        # 只有一条消息时语境太薄，补上上一话题末尾，帮模型还原「我对象/上次那个」。
        current = previous_tail[-FALLBACK_TAIL_SIZE:] + current
        previous_tail = []

    return TopicSlice(current=current, previous_tail=previous_tail)


def format_topic(messages: Iterable[StoredMessage], max_chars: int) -> str:
    lines: list[str] = []
    total = 0
    for message in messages:
        timestamp = (message.timestamp or "")[:19].replace("T", " ")
        line = f"{timestamp} {message.username}({message.role}): {message.content}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


__all__ = ["TopicSlice", "format_topic", "parse_timestamp", "split_topic"]
