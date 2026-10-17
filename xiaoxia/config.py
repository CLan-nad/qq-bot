from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    qq_app_id: str
    qq_app_secret: str
    llm_api_key: str
    llm_base_url: str
    model: str
    allowed_group_openids: frozenset[str]
    db_path: str
    recent_messages: int
    max_context_chars: int
    reply_max_chars: int
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
    search_provider: str
    tavily_api_key: str
    bocha_api_key: str
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

    required = ("QQ_APP_ID", "QQ_APP_SECRET", "LLM_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    raw_groups = os.getenv("XIAOXIA_ALLOWED_GROUP_OPENIDS", "")
    groups = frozenset(item.strip() for item in raw_groups.split(",") if item.strip())

    return Settings(
        qq_app_id=os.environ["QQ_APP_ID"],
        qq_app_secret=os.environ["QQ_APP_SECRET"],
        llm_api_key=os.environ["LLM_API_KEY"],
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("XIAOXIA_MODEL", "deepseek-chat"),
        allowed_group_openids=groups,
        db_path=os.getenv("XIAOXIA_DB_PATH", "data/xiaoxia.sqlite3"),
        recent_messages=int(os.getenv("XIAOXIA_RECENT_MESSAGES", "60")),
        max_context_chars=int(os.getenv("XIAOXIA_MAX_CONTEXT_CHARS", "24000")),
        reply_max_chars=int(os.getenv("XIAOXIA_REPLY_MAX_CHARS", "900")),
        timezone=os.getenv("XIAOXIA_TIMEZONE", "Asia/Shanghai"),
        scheduler_interval_seconds=int(os.getenv("XIAOXIA_SCHEDULER_INTERVAL_SECONDS", "30")),
        memory_interval_seconds=int(os.getenv("XIAOXIA_MEMORY_INTERVAL_SECONDS", "300")),
        memory_batch_messages=int(os.getenv("XIAOXIA_MEMORY_BATCH_MESSAGES", "80")),
        memory_search_messages=int(os.getenv("XIAOXIA_MEMORY_SEARCH_MESSAGES", "20")),
        task_max_count=int(os.getenv("XIAOXIA_TASK_MAX_COUNT", "20")),
        vision_model=os.getenv("XIAOXIA_VISION_MODEL", "deepseek-flash"),
        vision_max_images_per_request=int(
            os.getenv("XIAOXIA_VISION_MAX_IMAGES_PER_REQUEST", "3")
        ),
        vision_recent_messages=int(os.getenv("XIAOXIA_VISION_RECENT_MESSAGES", "20")),
        vision_max_image_bytes=int(
            os.getenv("XIAOXIA_VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
        ),
        image_cache_enabled=os.getenv(
            "XIAOXIA_IMAGE_CACHE", "1"
        ).strip().lower() not in {"0", "false", "no"},
        image_cache_max_rows=int(os.getenv("XIAOXIA_IMAGE_CACHE_MAX_ROWS", "2000")),
        care_enabled=os.getenv("XIAOXIA_CARE_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        care_model=os.getenv("XIAOXIA_CARE_MODEL", "").strip(),
        care_daily_max=int(os.getenv("XIAOXIA_CARE_DAILY_MAX", "2")),
        care_cooldown_minutes=int(os.getenv("XIAOXIA_CARE_COOLDOWN_MINUTES", "180")),
        active_msg_daily_max=int(os.getenv("XIAOXIA_ACTIVE_MSG_DAILY_MAX", "6")),
        # 模型写的 [头秃] 出站时要不要还原成 QQ 表情标签。实测 QQ 群消息不认这些标签
        # （2026-10-16 探针：三种写法都原样显示成尖括号），所以默认关：让它改用 emoji。
        face_tags=os.getenv("XIAOXIA_FACE_TAGS", "0").strip().lower()
        not in {"0", "false", "no"},
        # 表情包：模型写 [爱你]，出站时换成一张真的图发出去。图跟代码一起放在
        # assets/stickers/，换目录或临时关掉都只动这两个开关。
        stickers_enabled=os.getenv("XIAOXIA_STICKERS", "1").strip().lower()
        not in {"0", "false", "no"},
        sticker_dir=os.getenv("XIAOXIA_STICKER_DIR", "assets/stickers").strip()
        or "assets/stickers",
        ack_after_seconds=float(os.getenv("XIAOXIA_ACK_AFTER_SECONDS", "6")),
        search_provider=os.getenv("XIAOXIA_SEARCH_PROVIDER", "auto").strip().lower(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        bocha_api_key=os.getenv("BOCHA_API_KEY", "").strip(),
        search_max_results=int(os.getenv("XIAOXIA_SEARCH_MAX_RESULTS", "5")),
        search_timeout_seconds=int(os.getenv("XIAOXIA_SEARCH_TIMEOUT_SECONDS", "12")),
        web_fetch_max_chars=int(os.getenv("XIAOXIA_WEB_FETCH_MAX_CHARS", "8000")),
        shell_enabled=os.getenv("XIAOXIA_SHELL_ENABLED", "1").strip() not in {"0", "false", "no"},
        shell_mode=os.getenv("XIAOXIA_SHELL_MODE", "readonly").strip().lower(),
        shell_cwd=os.getenv("XIAOXIA_SHELL_CWD", "toolbox"),
        shell_timeout_seconds=int(os.getenv("XIAOXIA_SHELL_TIMEOUT_SECONDS", "10")),
        shell_max_output_chars=int(os.getenv("XIAOXIA_SHELL_MAX_OUTPUT_CHARS", "50000")),
        document_max_chars=int(os.getenv("XIAOXIA_DOCUMENT_MAX_CHARS", "24000")),
    )

