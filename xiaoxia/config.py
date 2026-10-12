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
    task_quiet_start: str
    task_quiet_end: str
    vision_model: str
    vision_max_images_per_request: int
    vision_recent_messages: int
    vision_max_image_bytes: int


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
        recent_messages=int(os.getenv("XIAOXIA_RECENT_MESSAGES", "200")),
        max_context_chars=int(os.getenv("XIAOXIA_MAX_CONTEXT_CHARS", "24000")),
        reply_max_chars=int(os.getenv("XIAOXIA_REPLY_MAX_CHARS", "900")),
        timezone=os.getenv("XIAOXIA_TIMEZONE", "Asia/Shanghai"),
        scheduler_interval_seconds=int(os.getenv("XIAOXIA_SCHEDULER_INTERVAL_SECONDS", "30")),
        memory_interval_seconds=int(os.getenv("XIAOXIA_MEMORY_INTERVAL_SECONDS", "300")),
        memory_batch_messages=int(os.getenv("XIAOXIA_MEMORY_BATCH_MESSAGES", "80")),
        memory_search_messages=int(os.getenv("XIAOXIA_MEMORY_SEARCH_MESSAGES", "20")),
        task_max_count=int(os.getenv("XIAOXIA_TASK_MAX_COUNT", "20")),
        task_quiet_start=os.getenv("XIAOXIA_TASK_QUIET_START", "23:30"),
        task_quiet_end=os.getenv("XIAOXIA_TASK_QUIET_END", "07:00"),
        vision_model=os.getenv("XIAOXIA_VISION_MODEL", "deepseek-flash"),
        vision_max_images_per_request=int(
            os.getenv("XIAOXIA_VISION_MAX_IMAGES_PER_REQUEST", "3")
        ),
        vision_recent_messages=int(os.getenv("XIAOXIA_VISION_RECENT_MESSAGES", "20")),
        vision_max_image_bytes=int(
            os.getenv("XIAOXIA_VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
        ),
    )

