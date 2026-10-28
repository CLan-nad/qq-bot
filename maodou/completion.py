from __future__ import annotations

import re


_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
_DANGLING_ENDINGS = (
    "→",
    "->",
    "：",
    ":",
    "，",
    ",",
    "、",
    "；",
    ";",
    "（",
    "(",
    "【",
    "[",
    "「",
    "『",
    "“",
    "『",
)
_OPEN_TO_CLOSE = {
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "「": "」",
    "『": "』",
    "“": "”",
}
_LIST_MARKER_AT_END = re.compile(r"(?:^|\n)\s*(?:\d+[.、]|[一二三四五六七八九十]+、)\s*$")


def completion_is_incomplete(
    text: str,
    finish_reason: str | None = None,
) -> bool:
    """判断一次正文生成是否明显没写完。

    ``finish_reason`` 是最可靠信号。文本启发式只处理服务端没给原因、或模型在
    箭号/冒号/逗号等明显断点停下的情况，避免把正常的一句短聊天误判成截断。
    """
    value = (text or "").strip()
    reason = (finish_reason or "").strip().lower()
    if reason in _TRUNCATED_FINISH_REASONS:
        return True
    if not value:
        return True
    if value.endswith(_DANGLING_ENDINGS):
        return True
    if _LIST_MARKER_AT_END.search(value):
        return True

    # 只检查“成对符号是否闭合”。正常聊天即使没有句号，也不应因此被续写。
    for opening, closing in _OPEN_TO_CLOSE.items():
        if value.count(opening) > value.count(closing):
            return True
    return False


def merge_continuation(original: str, continuation: str) -> str:
    """拼回续写正文，去掉模型有时会重复的断点附近内容。"""
    base = (original or "").rstrip()
    extra = (continuation or "").strip()
    if not base:
        return extra
    if not extra:
        return base

    # 去掉模型偶尔加的“续写：”标签，但不要误伤正文自身的短前缀。
    extra = re.sub(r"^(?:续写|接着写|接上)[：:]\s*", "", extra)
    max_overlap = min(120, len(base), len(extra))
    for size in range(max_overlap, 0, -1):
        if base.endswith(extra[:size]):
            extra = extra[size:].lstrip()
            break
    return base + extra


def continuation_prompt(partial_text: str) -> str:
    if (partial_text or "").strip():
        return (
            "上一条回答在正文中途被截断了。不要重头回答，也不要重复已有内容；"
            "从断点直接续写剩余部分，只输出续写正文。"
        )
    return (
        "上一条回答还没有输出任何正文。请从头完整回答原问题；"
        "不要解释截断，只输出最终正文。"
    )
