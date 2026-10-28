from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _apply_legacy_prefix() -> None:
    """兼容旧版配置名：XIAOXIA_* 当作 MAODOU_* 的别名，新版优先。

    这样服务器上的旧 .env 不用一次性改完也不会启动失败；等 .env 全量换名后，
    这段兼容可以再删掉。
    """
    for key, value in list(os.environ.items()):
        if key.startswith("XIAOXIA_"):
            os.environ.setdefault("MAODOU_" + key[len("XIAOXIA_"):], value)


@dataclass(frozen=True)
class Settings:
    qq_app_id: str
    qq_app_secret: str
    llm_api_key: str
    llm_base_url: str
    model: str
    reasoning_effort: str
    intent_model: str
    intent_reasoning_effort: str
    deep_reasoning_effort: str
    vision_reasoning_effort: str
    allowed_group_openids: frozenset[str]
    db_path: str
    recent_messages: int
    max_context_chars: int
    reply_max_chars: int
    reply_chunk_chars: int
    deep_reply_max_chars: int
    timezone: str
    scheduler_interval_seconds: int
    memory_interval_seconds: int
    memory_batch_messages: int
    memory_search_messages: int
    task_max_count: int
    vision_model: str
    vision_max_images_per_request: int
    vision_recent_messages: int
    vision_max_image_bytes: int
    image_cache_enabled: bool
    image_cache_max_rows: int
    care_enabled: bool
    care_model: str
    care_daily_max: int
    care_cooldown_minutes: int
    active_msg_daily_max: int
    face_tags: bool
    stickers_enabled: bool
    sticker_dir: str
    ack_after_seconds: float
    voice_reply: bool
    tts_provider: str
    tts_base_url: str
    tts_api_key: str
    tts_model: str
    tts_voice: str
    tts_format: str
    tts_instruction: str
    tts_sample_rate: int
    tts_timeout_seconds: float
    tts_max_chars: int
    tts_max_bytes: int
    tts_edge_voice: str
    tts_edge_rate: str
    tts_edge_pitch: str
    tts_edge_volume: str
    asr_base_url: str
    asr_api_key: str
    asr_model: str
    asr_timeout_seconds: float
    asr_max_bytes: int
    asr_max_per_request: int
    search_max_results: int
    search_timeout_seconds: int
    web_fetch_max_chars: int
    shell_enabled: bool
    shell_mode: str
    shell_cwd: str
    shell_timeout_seconds: int
    shell_max_output_chars: int
    document_max_chars: int


def load_settings() -> Settings:
    load_dotenv()
    _apply_legacy_prefix()

    required = ("QQ_APP_ID", "QQ_APP_SECRET", "LLM_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    raw_groups = os.getenv("MAODOU_ALLOWED_GROUP_OPENIDS", "")
    groups = frozenset(item.strip() for item in raw_groups.split(",") if item.strip())
    def _effort(name: str, default: str) -> str:
        value = os.getenv(name, default).strip().lower()
        return value if value in {"none", "low", "high", "max"} else default

    # 日常闲聊默认关思考：实测同一个 flash 模型，带 low 思考会多烧 200+ 个推理
    # token、多花 1.5 秒，回一句 30 字的话完全用不上。需要推理的只有 deep 工具循环。
    reasoning_effort = _effort("MAODOU_REASONING_EFFORT", "none")
    intent_reasoning_effort = _effort("MAODOU_INTENT_REASONING_EFFORT", "none")
    deep_reasoning_effort = _effort("MAODOU_DEEP_REASONING_EFFORT", "high")
    vision_reasoning_effort = _effort("MAODOU_VISION_REASONING_EFFORT", "low")

    return Settings(
        qq_app_id=os.environ["QQ_APP_ID"],
        qq_app_secret=os.environ["QQ_APP_SECRET"],
        llm_api_key=os.environ["LLM_API_KEY"],
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        # 全站统一用 flash；deepseek-chat 是旧别名，也会被网关悄悄改成 flash。
        model=os.getenv("MAODOU_MODEL", "deepseek-flash"),
        reasoning_effort=reasoning_effort,
        # 意图解析只输出一个小 JSON，不需要写作能力，但要快：默认用 flash 且关掉思考。
        # 最终回复仍然走 MAODOU_MODEL，人设和文笔不受影响。
        intent_model=os.getenv("MAODOU_INTENT_MODEL", "deepseek-flash"),
        intent_reasoning_effort=intent_reasoning_effort,
        deep_reasoning_effort=deep_reasoning_effort,
        vision_reasoning_effort=vision_reasoning_effort,
        allowed_group_openids=groups,
        db_path=os.getenv("MAODOU_DB_PATH", "data/maodou.sqlite3"),
        recent_messages=int(os.getenv("MAODOU_RECENT_MESSAGES", "60")),
        max_context_chars=int(os.getenv("MAODOU_MAX_CONTEXT_CHARS", "24000")),
        # 模型可以先写完整长回答，发送层再按 reply_chunk_chars 拆成多条 QQ 消息。
        # 900 是旧版“整条硬截断”的上限；现在总上限放开，但单条仍默认 900。
        reply_max_chars=int(os.getenv("MAODOU_REPLY_MAX_CHARS", "3000")),
        reply_chunk_chars=int(os.getenv("MAODOU_REPLY_CHUNK_CHARS", "900")),
        deep_reply_max_chars=int(
            os.getenv("MAODOU_DEEP_REPLY_MAX_CHARS", "6000")
        ),
        timezone=os.getenv("MAODOU_TIMEZONE", "Asia/Shanghai"),
        scheduler_interval_seconds=int(os.getenv("MAODOU_SCHEDULER_INTERVAL_SECONDS", "30")),
        memory_interval_seconds=int(os.getenv("MAODOU_MEMORY_INTERVAL_SECONDS", "300")),
        memory_batch_messages=int(os.getenv("MAODOU_MEMORY_BATCH_MESSAGES", "80")),
        memory_search_messages=int(os.getenv("MAODOU_MEMORY_SEARCH_MESSAGES", "20")),
        task_max_count=int(os.getenv("MAODOU_TASK_MAX_COUNT", "20")),
        vision_model=os.getenv("MAODOU_VISION_MODEL", "deepseek-flash"),
        vision_max_images_per_request=int(
            os.getenv("MAODOU_VISION_MAX_IMAGES_PER_REQUEST", "9")
        ),
        vision_recent_messages=int(os.getenv("MAODOU_VISION_RECENT_MESSAGES", "20")),
        vision_max_image_bytes=int(
            os.getenv("MAODOU_VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
        ),
        image_cache_enabled=os.getenv(
            "MAODOU_IMAGE_CACHE", "1"
        ).strip().lower() not in {"0", "false", "no"},
        image_cache_max_rows=int(os.getenv("MAODOU_IMAGE_CACHE_MAX_ROWS", "2000")),
        care_enabled=os.getenv("MAODOU_CARE_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        care_model=os.getenv("MAODOU_CARE_MODEL", "").strip(),
        care_daily_max=int(os.getenv("MAODOU_CARE_DAILY_MAX", "2")),
        care_cooldown_minutes=int(os.getenv("MAODOU_CARE_COOLDOWN_MINUTES", "180")),
        active_msg_daily_max=int(os.getenv("MAODOU_ACTIVE_MSG_DAILY_MAX", "6")),
        # 模型写的 [头秃] 出站时要不要还原成 QQ 表情标签。实测 QQ 群消息不认这些标签
        # （2026-10-16 探针：三种写法都原样显示成尖括号），所以默认关：让它改用 emoji。
        face_tags=os.getenv("MAODOU_FACE_TAGS", "0").strip().lower()
        not in {"0", "false", "no"},
        # 表情包：模型写 [爱你]，出站时换成一张真的图发出去。图跟代码一起放在
        # assets/stickers/，换目录或临时关掉都只动这两个开关。
        stickers_enabled=os.getenv("MAODOU_STICKERS", "1").strip().lower()
        not in {"0", "false", "no"},
        sticker_dir=os.getenv("MAODOU_STICKER_DIR", "assets/stickers").strip()
        or "assets/stickers",
        ack_after_seconds=float(os.getenv("MAODOU_ACK_AFTER_SECONDS", "4")),
        # 语音回复是独立能力：auto 会优先使用 QwenCloud，其次完整
        # OpenAI-compatible 配置，最后退回无需 key 的 Edge 中文神经语音。
        voice_reply=os.getenv("MAODOU_VOICE_REPLY", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        tts_provider=os.getenv("MAODOU_TTS_PROVIDER", "auto").strip().lower()
        or "auto",
        tts_base_url=os.getenv("MAODOU_TTS_BASE_URL", "").strip(),
        tts_api_key=os.getenv("MAODOU_TTS_API_KEY", "").strip(),
        tts_model=os.getenv("MAODOU_TTS_MODEL", "").strip(),
        tts_voice=os.getenv("MAODOU_TTS_VOICE", "").strip(),
        tts_format=os.getenv("MAODOU_TTS_FORMAT", "mp3").strip() or "mp3",
        tts_instruction=os.getenv("MAODOU_TTS_INSTRUCTION", "").strip(),
        tts_sample_rate=int(os.getenv("MAODOU_TTS_SAMPLE_RATE", "24000")),
        tts_timeout_seconds=float(os.getenv("MAODOU_TTS_TIMEOUT_SECONDS", "30")),
        tts_max_chars=int(os.getenv("MAODOU_TTS_MAX_CHARS", "160")),
        tts_max_bytes=int(
            os.getenv("MAODOU_TTS_MAX_BYTES", str(20 * 1024 * 1024))
        ),
        tts_edge_voice=os.getenv(
            "MAODOU_TTS_EDGE_VOICE", "zh-CN-XiaoyiNeural"
        ).strip()
        or "zh-CN-XiaoyiNeural",
        tts_edge_rate=os.getenv("MAODOU_TTS_EDGE_RATE", "+5%").strip() or "+5%",
        tts_edge_pitch=os.getenv("MAODOU_TTS_EDGE_PITCH", "+5Hz").strip()
        or "+5Hz",
        tts_edge_volume=os.getenv("MAODOU_TTS_EDGE_VOLUME", "+0%").strip()
        or "+0%",
        # QQ 通常已经给出 asr_refer_text；这些配置只在该字段缺失时兜底。
        asr_base_url=os.getenv("MAODOU_ASR_BASE_URL", "").strip(),
        asr_api_key=os.getenv("MAODOU_ASR_API_KEY", "").strip(),
        asr_model=os.getenv("MAODOU_ASR_MODEL", "").strip(),
        asr_timeout_seconds=float(os.getenv("MAODOU_ASR_TIMEOUT_SECONDS", "30")),
        asr_max_bytes=int(
            os.getenv("MAODOU_ASR_MAX_BYTES", str(20 * 1024 * 1024))
        ),
        asr_max_per_request=int(os.getenv("MAODOU_ASR_MAX_PER_REQUEST", "5")),
        search_max_results=int(os.getenv("MAODOU_SEARCH_MAX_RESULTS", "5")),
        search_timeout_seconds=int(os.getenv("MAODOU_SEARCH_TIMEOUT_SECONDS", "12")),
        web_fetch_max_chars=int(os.getenv("MAODOU_WEB_FETCH_MAX_CHARS", "8000")),
        shell_enabled=os.getenv("MAODOU_SHELL_ENABLED", "1").strip() not in {"0", "false", "no"},
        shell_mode=os.getenv("MAODOU_SHELL_MODE", "readonly").strip().lower(),
        shell_cwd=os.getenv("MAODOU_SHELL_CWD", "toolbox"),
        shell_timeout_seconds=int(os.getenv("MAODOU_SHELL_TIMEOUT_SECONDS", "10")),
        shell_max_output_chars=int(os.getenv("MAODOU_SHELL_MAX_OUTPUT_CHARS", "50000")),
        document_max_chars=int(os.getenv("MAODOU_DOCUMENT_MAX_CHARS", "24000")),
    )

