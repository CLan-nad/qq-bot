"""主动关心的第一道闸门：不花钱的关键词预筛。

小毛豆平时得安静。只有群里这句话先命中信号词，才值得再花一次模型调用去判断
到底要不要开口；命中也不代表一定会说，最终判断在模型那一层。
"""
from __future__ import annotations

# 关心话最多这么长，再长就像在写小作文了。
CARE_TEXT_MAX_CHARS = 25

# 身体不舒服
BODY_SIGNALS = (
    "睡不好", "没睡好", "睡不着", "失眠", "熬了一夜", "通宵", "好困", "困死",
    "累了", "好累", "太累", "累瘫", "累坏了", "疲惫", "没精神",
    "难受", "不舒服", "头疼", "头痛", "头晕", "发烧", "低烧", "感冒", "咳嗽",
    "拉肚子", "腹泻", "肚子疼", "肚子痛", "胃疼", "胃痛", "腰痛", "背痛",
    "痛经", "恶心", "呕吐", "受伤", "磕到", "扭到", "牙疼", "生病",
)

# 情绪低落
MOOD_SIGNALS = (
    "心情不好", "心情差", "不开心", "难过", "委屈", "郁闷", "低落", "emo",
    "烦躁", "心烦", "焦虑", "压力好大", "压力大", "崩溃", "心态崩", "破防",
    "想哭", "哭了", "生气", "气死", "被骂", "挨批", "吵架", "冷战", "分手",
    "孤独", "害怕", "担心", "心慌", "无力", "抑郁", "提不起劲",
)

CARE_SIGNALS = BODY_SIGNALS + MOOD_SIGNALS

# 中文口语喜欢往词中间塞副词（“肚子好疼”“头有点痛”），只按整词匹配会漏掉。
# 所以再补一条：身体部位和痛感词同时出现也算信号。
BODY_PARTS = (
    "肚子", "腹部", "胃", "头", "脑袋", "嗓子", "喉咙", "腰", "背", "胸",
    "膝盖", "腿", "胳膊", "手", "脚", "关节", "眼睛", "耳朵", "心脏", "身体",
)
PAIN_WORDS = ("疼", "痛", "难受", "不舒服", "恶", "吐", "泻", "抽筋", "发颤", "没力气")

_MAX_CONTENT_CHARS = 80
_MIN_CONTENT_CHARS = 2


def matched_signals(content: str) -> tuple[str, ...]:
    """返回命中的信号词，身体类在前、情绪类在后，方便日志里看清楚。"""
    text = (content or "").strip().lower()
    if not text:
        return ()
    hits = tuple(signal for signal in BODY_SIGNALS if signal.lower() in text)
    hits += tuple(signal for signal in MOOD_SIGNALS if signal.lower() in text)
    combos = (
        f"{part}{pain}"
        for part in BODY_PARTS
        if part in text
        for pain in PAIN_WORDS
        if pain in text
    )
    first_combo = next(combos, None)
    if first_combo:
        hits += (first_combo,)
    return hits


def should_ask_model(content: str) -> bool:
    """第一道闸门：值不值得为这句话再花一次模型调用。"""
    text = (content or "").strip()
    if len(text) < _MIN_CONTENT_CHARS or len(text) > _MAX_CONTENT_CHARS:
        return False
    if "http://" in text or "https://" in text or "www." in text.lower():
        return False
    if "@" in text:
        # 明确在 @ 别人（@小毛豆 那种走主流程），不是在抒发情绪。
        return False
    if text.startswith("[") and text.endswith("]"):
        return False
    return bool(matched_signals(text))


def normalize_care_text(text: str) -> str:
    """把模型给的话收成一条能直接发群的短句；不合格就返回空串。"""
    cleaned = (text or "").replace("\r", " ").replace("\n", " ").strip()
    cleaned = cleaned.strip("\u201c\u201d\u2018\u2019 ").strip()
    for prefix in ("care:", "text:", "关心：", "回复："):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    if not cleaned:
        return ""
    if len(cleaned) > CARE_TEXT_MAX_CHARS:
        cleaned = cleaned[:CARE_TEXT_MAX_CHARS].rstrip()
    return cleaned
