from __future__ import annotations


_BREAK_LEVELS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    "，",
    ",",
    " ",
)


def _safe_hard_cut(text: str, limit: int) -> int:
    """硬切时不要把 QQ 表情标签从中间劈开。"""
    cut = max(1, min(limit, len(text)))
    while cut > 1:
        prefix = text[:cut]
        if prefix.rfind("<") <= prefix.rfind(">"):
            break
        cut -= 1
    return cut


def split_reply_text(text: str, max_chars: int) -> list[str]:
    """按自然边界把一条长回复拆成多条 QQ 消息。

    优先级：空行、换行、句末标点、逗号、空格，最后才硬切。只丢掉分条处的
    纯空白，不丢正文；每个返回段都不超过 ``max_chars``。
    """
    limit = max(1, int(max_chars))
    remaining = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not remaining:
        return []

    chunks: list[str] = []
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining.strip())
            break

        minimum = max(1, limit // 2)
        cut = 0
        for separator in _BREAK_LEVELS:
            search_end = min(len(remaining), limit + len(separator))
            position = remaining.rfind(separator, 0, search_end)
            while position >= 0:
                candidate = position + len(separator)
                if minimum <= candidate <= limit:
                    cut = candidate
                    break
                position = remaining.rfind(separator, 0, position)
            if cut:
                break
        if not cut:
            cut = _safe_hard_cut(remaining, limit)

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()

    return [chunk for chunk in chunks if chunk]
