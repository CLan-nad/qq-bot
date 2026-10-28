from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .images import format_message_content, normalize_face_tags
from .stickers import FaceToken, face_book, face_tokens
from .scheduling import schedule_label


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredMessage:
    username: str
    member_openid: str
    role: str
    content: str
    timestamp: str
    source: str
    message_key: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    row_id: int = 0


@dataclass(frozen=True)
class SearchResult:
    row_id: int
    username: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class MemberMessageCount:
    member_key: str
    username: str
    count: int
    is_bot: bool = False


@dataclass(frozen=True)
class MessageStats:
    total: int
    members: list[MemberMessageCount]


@dataclass(frozen=True)
class MemoryCard:
    id: int
    kind: str
    person: str
    content: str
    source_message_id: str
    source_row_id: int | None = None


@dataclass(frozen=True)
class ScheduledTask:
    id: int
    group_openid: str
    title: str
    content: str | None
    generation_prompt: str | None
    schedule_type: str
    next_run_at: str
    timezone: str
    time_of_day: str | None
    weekday: int | None
    status: str
    skip_next: bool
    last_sent_at: str | None
    failure_count: int
    created_by_openid: str
    created_by_name: str
    month: int | None = None
    day: int | None = None
    leap_month: bool = False
    remind_days_ahead: int = 0
    offset_days: int = 0
    linked_task_id: int | None = None


@dataclass(frozen=True)
class MemoryBatch:
    start_row_id: int
    end_row_id: int
    messages: list[StoredMessage]


# 任务表只有一份建表文本：_init_schema 建新库用，_rebuild_tasks_table 升级老库用，
# 两处共用同一个常量，避免线上 schema 和迁移后的 schema 不一致。
_SCHEDULED_TASKS_SCHEMA = """
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_openid TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    generation_prompt TEXT,
                    schedule_type TEXT NOT NULL
                        CHECK (schedule_type IN ('once', 'daily', 'weekly', 'yearly', 'lunar_yearly')),
                    next_run_at TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    time_of_day TEXT,
                    weekday INTEGER,
                    month INTEGER,
                    day INTEGER,
                    leap_month INTEGER NOT NULL DEFAULT 0,
                    remind_days_ahead INTEGER NOT NULL DEFAULT 0,
                    offset_days INTEGER NOT NULL DEFAULT 0,
                    linked_task_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'paused', 'done', 'deleted')),
                    skip_next INTEGER NOT NULL DEFAULT 0,
                    last_sent_at TEXT,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    created_by_openid TEXT NOT NULL,
                    created_by_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
"""

# 重建表时按这个顺序整表搬数据；缺的新列在重建前已经由 ALTER TABLE 补齐
_TASKS_COLUMNS = (
    "id", "group_openid", "title", "content", "generation_prompt", "schedule_type",
    "next_run_at", "timezone", "time_of_day", "weekday", "month", "day", "leap_month",
    "remind_days_ahead", "offset_days", "linked_task_id", "status", "skip_next",
    "last_sent_at", "failure_count", "created_by_openid", "created_by_name",
    "created_at", "updated_at",
)

# 旧库升级清单：只加列、不改写任何已有数据
_ADDITIVE_COLUMNS = {
    "memory_cards": (
        ("pinned", "INTEGER NOT NULL DEFAULT 0"),
        ("scope", "TEXT NOT NULL DEFAULT 'permanent'"),
        ("expires_at", "TEXT"),
        ("source_row_id", "INTEGER"),
    ),
    # 称呼原来是挂在显示名上的，对方一改昵称就认不出人了。补一列 openid 把称呼钉在人身上。
    "person_addresses": (("member_openid", "TEXT NOT NULL DEFAULT ''"),),
    "scheduled_tasks": (
        ("month", "INTEGER"),
        ("day", "INTEGER"),
        ("leap_month", "INTEGER NOT NULL DEFAULT 0"),
        ("remind_days_ahead", "INTEGER NOT NULL DEFAULT 0"),
        ("offset_days", "INTEGER NOT NULL DEFAULT 0"),
        ("linked_task_id", "INTEGER"),
    ),
}

# 记忆卡固定取这些列；SELECT 列表必须和 _card_from_row 配套
_MEMORY_COLUMNS = (
    "id, kind, person, content, source_message_id, source_row_id"
)

# 记忆卡注入上下文的字符预算。实测 92 张卡正文一共才 2860 字，所以这个预算的作用
# 是「正常情况下全塞进去，异常膨胀了才裁」，不是靠它省 token。
MEMORY_PROMPT_MAX_CHARS = 6000
MEMORY_PROMPT_MAX_CARDS = 160

# 预算装不下时先丢哪一类：流水账（哪天抓了哪只宠物）最先丢，
# 行为规则和身份类的卡最后才丢——被丢掉的那些正是模型天天违反的规则。
MEMORY_KIND_PRIORITY = ("preference", "agreement", "relationship", "fact", "event")
MEMORY_EPISODIC_KINDS = frozenset({"fact", "event"})

# 判断「同一句话被抽了两遍」时忽略的标点空白。
_MEMORY_NOISE = re.compile(r"[\s，。、；：,.;:!！?？~～“”\"'（）()【】\[\]《》—\-]+")

# 对话摘要注入/落库的长度上限。生成时提示词要求 150-300 字，但线上出现过 615 字，
# 这里在落库前再兜一道底，避免一段摘要就把上下文灌满。
SUMMARY_MAX_CHARS = 300


def clip_summary(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """把超长摘要压回预算内：优先在最后一个整句处截断，找不到就用硬截断。"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("。", 0, max_chars)
    if cut == -1:
        cut = text.rfind("；", 0, max_chars)
    if cut == -1:
        return text[:max_chars].rstrip() + "…"
    return text[: cut + 1]


class ChatStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_key TEXT NOT NULL UNIQUE,
                    message_id TEXT NOT NULL,
                    msg_seq TEXT,
                    group_openid TEXT NOT NULL,
                    member_openid TEXT,
                    username TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    source TEXT,
                    raw_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_group_recent
                    ON messages(group_openid, row_id DESC);

                CREATE TABLE IF NOT EXISTS processed_at_messages (
                    message_key TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    __SCHEDULED_TASKS_BODY__
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due
                    ON scheduled_tasks(status, next_run_at);

                CREATE TABLE IF NOT EXISTS memory_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_openid TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    person TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_cards_content
                    ON memory_cards(group_openid, content);

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_openid TEXT NOT NULL,
                    start_row_id INTEGER NOT NULL,
                    end_row_id INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(group_openid, start_row_id, end_row_id)
                );

                CREATE TABLE IF NOT EXISTS image_descriptions (
                    digest TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT '',
                    byte_size INTEGER NOT NULL DEFAULT 0,
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_hit_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS care_log (
                    message_key TEXT PRIMARY KEY,
                    group_openid TEXT NOT NULL,
                    member_openid TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_care_log_member
                    ON care_log(group_openid, member_openid, created_at DESC);

                CREATE TABLE IF NOT EXISTS group_preferences (
                    group_openid TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_openid, key)
                );

                CREATE TABLE IF NOT EXISTS person_aliases (
                    group_openid TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    canonical TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_openid, alias)
                );

                CREATE TABLE IF NOT EXISTS person_addresses (
                    group_openid TEXT NOT NULL,
                    person TEXT NOT NULL,
                    address TEXT NOT NULL,
                    member_openid TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_openid, person)
                );

                CREATE TABLE IF NOT EXISTS face_tokens (
                    group_openid TEXT NOT NULL,
                    name TEXT NOT NULL,
                    face_type INTEGER NOT NULL DEFAULT 1,
                    face_id TEXT NOT NULL DEFAULT '',
                    ext TEXT NOT NULL DEFAULT '',
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_openid, name)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    username,
                    content='messages',
                    content_rowid='row_id',
                    tokenize='trigram'
                );

                CREATE TRIGGER IF NOT EXISTS messages_fts_insert
                AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, username)
                    VALUES (new.row_id, new.content, new.username);
                END;

                CREATE TRIGGER IF NOT EXISTS messages_fts_delete
                AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, username)
                    VALUES ('delete', old.row_id, old.content, old.username);
                END;

                CREATE TRIGGER IF NOT EXISTS messages_fts_update
                AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, username)
                    VALUES ('delete', old.row_id, old.content, old.username);
                    INSERT INTO messages_fts(rowid, content, username)
                    VALUES (new.row_id, new.content, new.username);
                END;
                """.replace("__SCHEDULED_TASKS_BODY__", _SCHEDULED_TASKS_SCHEMA)
            )
            # Backfill existing rows after the FTS table and triggers are created.
            self._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        self._migrate()

    def _table_columns(self, table: str) -> list[str]:
        # 表名只来自本模块的常量，不来自用户输入
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [str(row["name"]) for row in rows]

    def _migrate(self) -> None:
        """把旧库补到当前 schema：缺列就 ALTER TABLE ADD COLUMN，CHECK 约束过时重建任务表。

        顺带把历史消息里残留的 QQ 表情标签解码成可读文本。除此之外只做加法，
        不重写其它已有数据；每次启动都会跑，重复执行是安全的。
        """
        with self._conn:
            for table, columns in _ADDITIVE_COLUMNS.items():
                existing = set(self._table_columns(table))
                if not existing:
                    continue
                for name, ddl in columns:
                    if name not in existing:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                        logger.info("migration: added column %s.%s", table, name)
                        existing.add(name)

            # 永久/临时记忆已经是历史功能。旧库里的列保留用于兼容，但启动时统一
            # 归一成普通长期记忆：不删任何卡片，只把标记清掉；重复执行没有副作用。
            cursor = self._conn.execute(
                """
                UPDATE memory_cards
                SET pinned = 0, scope = 'permanent', expires_at = NULL
                WHERE pinned <> 0 OR scope <> 'permanent' OR expires_at IS NOT NULL
                """
            )
            if cursor.rowcount:
                logger.info(
                    "migration: normalized %s memory cards to long-term memory",
                    cursor.rowcount,
                )

            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'scheduled_tasks'"
            ).fetchone()
            table_sql = str(row["sql"] or "") if row else ""
            if table_sql and "'yearly'" not in table_sql:
                self._rebuild_tasks_table()

        self._normalize_legacy_face_tags()
        # 称呼补齐 openid：老库里的称呼只挂了显示名，补一次就够，之后启动都是空转。
        for row in self._conn.execute(
            "SELECT DISTINCT group_openid FROM person_addresses"
        ).fetchall():
            self.backfill_address_openids(str(row["group_openid"]))
        # 表情：学习是在 face_tokens 上线之后才开始的，老库第一次跑得把历史补一遍，
        # 否则群里用了一年的表情一个都发不出来。补过就不再动（hits 会翻倍）。
        for row in self._conn.execute(
            "SELECT DISTINCT group_openid FROM messages"
        ).fetchall():
            self.backfill_face_tokens(str(row["group_openid"]))

    def _normalize_legacy_face_tags(self) -> int:
        """把历史消息里存量的 <faceType=...> 乱码解码成 [表情名]。

        这些标签在库里躺着的时候既进不了 FTS 检索，又会原样喂给模型当上下文。
        只改写命中的行，重复执行安全。FTS 由 messages_fts_update 触发器同步。
        """
        rows = self._conn.execute(
            "SELECT group_openid, message_key, content FROM messages"
            " WHERE content LIKE '%<faceType=%'"
        ).fetchall()
        changed = 0
        for row in rows:
            cleaned = normalize_face_tags(row["content"])
            if not cleaned or cleaned == row["content"]:
                continue
            with self._conn:
                self._conn.execute(
                    "UPDATE messages SET content = ? WHERE group_openid = ? AND message_key = ?",
                    (cleaned, row["group_openid"], row["message_key"]),
                )
            changed += 1
        if changed:
            logger.info("migration: decoded face tags in %d stored messages", changed)
        return changed

    def _rebuild_tasks_table(self) -> None:
        """SQLite 改不了 CHECK 约束，只能建新表、整表搬过去、再改名。id 保持不变。"""
        present = set(self._table_columns("scheduled_tasks"))
        common = [name for name in _TASKS_COLUMNS if name in present]
        joined = ", ".join(common)
        self._conn.execute(f"CREATE TABLE scheduled_tasks_new ({_SCHEDULED_TASKS_SCHEMA})")
        self._conn.execute(
            f"INSERT INTO scheduled_tasks_new ({joined}) SELECT {joined} FROM scheduled_tasks"
        )
        self._conn.execute("DROP TABLE scheduled_tasks")
        self._conn.execute("ALTER TABLE scheduled_tasks_new RENAME TO scheduled_tasks")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due"
            " ON scheduled_tasks(status, next_run_at)"
        )
        logger.info("migration: rebuilt scheduled_tasks to allow yearly schedules")

    @staticmethod
    def message_key(message_id: str | None, msg_seq: Any) -> str:
        return f"{message_id or ''}:{msg_seq or ''}"

    @staticmethod
    def _attachment_summary(raw: dict[str, Any]) -> str:
        attachments = raw.get("attachments") or []
        names = []
        for item in attachments:
            if not isinstance(item, dict):
                continue
            content_type = item.get("content_type") or "attachment"
            names.append(f"[{content_type}]")
        return " ".join(names)

    def save_message(
        self,
        *,
        message_id: str | None,
        msg_seq: Any,
        group_openid: str | None,
        member_openid: str | None,
        username: str | None,
        role: str | None,
        content: str | None,
        timestamp: str | None,
        source: str,
        raw: dict[str, Any],
    ) -> bool:
        if not group_openid:
            return False

        key = self.message_key(message_id, msg_seq)
        # 表情标签必须在入库前解码：库里的 content 就是模型每轮要读的上下文，
        # 存成 <faceType=1,faceId="267",ext="..."> 等于让它读乱码。
        self.remember_faces(group_openid, content)
        text = normalize_face_tags(content).strip()
        if not text:
            text = self._attachment_summary(raw)
        if not text:
            text = "[无文字内容]"

        display_name = (username or "").strip() or (member_openid or "unknown")[:12]
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO messages (
                    message_key, message_id, msg_seq, group_openid, member_openid,
                    username, role, content, timestamp, source, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    message_id or "",
                    str(msg_seq or ""),
                    group_openid,
                    member_openid,
                    display_name,
                    role or "member",
                    text,
                    timestamp or "",
                    source,
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        return cursor.rowcount == 1

    def mark_at_processed(self, message_key: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO processed_at_messages (message_key) VALUES (?)",
                (message_key,),
            )
        return cursor.rowcount == 1

    def save_bot_message(
        self,
        *,
        group_openid: str,
        content: str,
        timestamp: str,
        source: str = "BOT_REPLY",
        username: str = "小毛豆",
    ) -> bool:
        """记录机器人自己发出的消息，让它能记住上一轮说过什么。

        QQ 不会把机器人自己发的消息回推给机器人，所以必须自己落库；
        否则最近上下文里永远只有两个人的消息，模型看不到自己的上一轮回复。
        """
        text = (content or "").strip()
        if not group_openid or not text:
            return False
        key = f"bot:{uuid.uuid4().hex}"
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO messages (
                    message_key, message_id, msg_seq, group_openid, member_openid,
                    username, role, content, timestamp, source, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    "",
                    "",
                    group_openid,
                    "",
                    (username or "").strip() or "小毛豆",
                    "bot",
                    text,
                    timestamp or "",
                    source,
                    "{}",
                ),
            )
        return cursor.rowcount == 1

    def update_message_content(self, group_openid: str, message_key: str, content: str) -> bool:
        """回填图片描述等派生内容，FTS 索引由触发器同步更新。"""
        text = (content or "").strip()
        if not text or not message_key:
            return False
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE messages SET content = ? WHERE group_openid = ? AND message_key = ?",
                (text, group_openid, message_key),
            )
        return cursor.rowcount == 1

    def get_image_description(self, digest: str) -> str:
        """命中缓存就返回上次的图片描述，并把命中计数加一。"""
        if not digest:
            return ""
        row = self._conn.execute(
            "SELECT description FROM image_descriptions WHERE digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            return ""
        with self._conn:
            self._conn.execute(
                "UPDATE image_descriptions"
                " SET hits = hits + 1, last_hit_at = CURRENT_TIMESTAMP WHERE digest = ?",
                (digest,),
            )
        return row["description"] or ""

    def save_image_description(
        self,
        *,
        digest: str,
        description: str,
        content_type: str = "",
        byte_size: int = 0,
        max_rows: int = 2000,
    ) -> None:
        text = (description or "").strip()
        if not digest or not text:
            return
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO image_descriptions (
                    digest, description, content_type, byte_size, hits
                ) VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(digest) DO UPDATE SET
                    description = excluded.description,
                    content_type = excluded.content_type,
                    byte_size = excluded.byte_size,
                    last_hit_at = CURRENT_TIMESTAMP
                """,
                (digest, text[:500], content_type, byte_size),
            )
            if max_rows > 0:
                self._conn.execute(
                    """
                    DELETE FROM image_descriptions
                    WHERE digest NOT IN (
                        SELECT digest FROM image_descriptions
                        ORDER BY last_hit_at DESC, hits DESC
                        LIMIT ?
                    )
                    """,
                    (max_rows,),
                )

    def image_cache_stats(self) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits"
            " FROM image_descriptions"
        ).fetchone()
        return {"entries": int(row["entries"]), "hits": int(row["hits"])}

    def recent_messages_with_raw(
        self,
        group_openid: str,
        limit: int,
    ) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT row_id, message_key, username, member_openid, role, content, timestamp, source, raw_json
            FROM messages
            WHERE group_openid = ?
            ORDER BY row_id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        messages: list[StoredMessage] = []
        for row in reversed(rows):
            try:
                raw = json.loads(row["raw_json"] or "{}")
            except (TypeError, ValueError):
                raw = {}
            if not isinstance(raw, dict):
                raw = {}
            messages.append(
                StoredMessage(
                    username=row["username"] or "unknown",
                    member_openid=row["member_openid"] or "",
                    role=row["role"] or "member",
                    content=row["content"] or "",
                    timestamp=row["timestamp"] or "",
                    source=row["source"] or "",
                    message_key=row["message_key"] or "",
                    raw=raw,
                    row_id=int(row["row_id"] or 0),
                )
            )
        return messages

    def recent_messages(self, group_openid: str, limit: int) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT row_id, username, member_openid, role, content, timestamp, source
            FROM messages
            WHERE group_openid = ?
            ORDER BY row_id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [
            StoredMessage(
                username=row["username"] or "unknown",
                member_openid=row["member_openid"] or "",
                role=row["role"] or "member",
                content=row["content"] or "",
                timestamp=row["timestamp"] or "",
                source=row["source"] or "",
                row_id=int(row["row_id"] or 0),
            )
            for row in reversed(rows)
        ]

    def message_stats(
        self,
        group_openid: str,
        *,
        since: str | None = None,
        until: str | None = None,
        include_bot: bool = False,
    ) -> MessageStats:
        """按成员统计消息条数。

        人数统计必须走 SQL 聚合，不能拿 memory_search 的几十条片段让模型猜。
        同一个人按 member_openid 聚合，昵称改了也不会被拆成两个人；老数据没有
        openid 时才退回显示名。默认不统计机器人自己的消息。
        """
        member_key = """
            CASE
                WHEN COALESCE(role, '') = 'bot' THEN 'bot'
                WHEN TRIM(COALESCE(member_openid, '')) != ''
                    THEN 'openid:' || member_openid
                ELSE 'name:' || COALESCE(NULLIF(TRIM(username), ''), 'unknown')
            END
        """
        conditions = ["group_openid = ?"]
        params: list[Any] = [group_openid]
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp < ?")
            params.append(until)
        if not include_bot:
            conditions.append("COALESCE(role, '') != 'bot'")

        rows = self._conn.execute(
            f"""
            SELECT {member_key} AS member_key,
                   COUNT(*) AS count,
                   MAX(row_id) AS latest_row_id
            FROM messages
            WHERE {' AND '.join(conditions)}
            GROUP BY member_key
            ORDER BY count DESC, latest_row_id DESC
            """,
            params,
        ).fetchall()

        members: list[MemberMessageCount] = []
        for row in rows:
            latest = self._conn.execute(
                "SELECT username FROM messages WHERE row_id = ?",
                (row["latest_row_id"],),
            ).fetchone()
            key = str(row["member_key"])
            username = str((latest["username"] if latest else "") or "").strip()
            if key == "bot":
                username = username or "小毛豆"
            else:
                username = username or "unknown"
            members.append(
                MemberMessageCount(
                    member_key=key,
                    username=username,
                    count=int(row["count"] or 0),
                    is_bot=key == "bot",
                )
            )
        return MessageStats(
            total=sum(member.count for member in members),
            members=members,
        )

    def known_members(self, group_openid: str, limit: int = 20) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT m.username, m.member_openid, m.role, m.content, m.timestamp, m.source
            FROM messages AS m
            JOIN (
                SELECT member_openid, MAX(row_id) AS row_id
                FROM messages
                WHERE group_openid = ?
                  AND member_openid IS NOT NULL
                  AND member_openid != ''
                GROUP BY member_openid
            ) AS latest ON latest.row_id = m.row_id
            ORDER BY m.row_id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [
            StoredMessage(
                username=row["username"] or "unknown",
                member_openid=row["member_openid"] or "",
                role=row["role"] or "member",
                content=row["content"] or "",
                timestamp=row["timestamp"] or "",
                source=row["source"] or "",
            )
            for row in rows
        ]

    # ---------- 表情：群里出现过哪些，怎么把模型写的 [头秃] 发成真表情 ----------
    #
    # 名字↔faceId 的对应关系全部来自真实消息，代码里不写死任何一个。模型只看得到
    # 可读名字（[头秃]），要发真表情时得靠这张表还原成 <faceType=1,faceId="267">。

    def remember_faces(self, group_openid: str, content: str | None) -> int:
        """把消息里出现过的表情标签记下来。返回记下的个数。"""
        tokens = face_tokens(content)
        if not tokens or not group_openid:
            return 0
        with self._conn:
            for token in tokens:
                self._conn.execute(
                    """
                    INSERT INTO face_tokens (group_openid, name, face_type, face_id, ext, hits)
                    VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(group_openid, name) DO UPDATE SET
                        face_type = excluded.face_type,
                        face_id = excluded.face_id,
                        ext = excluded.ext,
                        hits = face_tokens.hits + 1,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (group_openid, token.name, token.face_type, token.face_id, token.ext),
                )
        return len(tokens)

    def face_book(self, group_openid: str, limit: int = 60) -> dict[str, FaceToken]:
        """群里见过的表情：名字 → 标签。按出现次数排序，常用的排在前面。"""
        rows = self._conn.execute(
            """
            SELECT name, face_type, face_id, ext FROM face_tokens
            WHERE group_openid = ?
            ORDER BY hits DESC, updated_at DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return face_book(
            FaceToken(
                name=str(row["name"]),
                face_type=int(row["face_type"] or 1),
                face_id=str(row["face_id"] or ""),
                ext=str(row["ext"] or ""),
            )
            for row in rows
        )

    def face_names(self, group_openid: str, limit: int = 60) -> list[str]:
        return list(self.face_book(group_openid, limit=limit))

    def format_recent_messages(self, group_openid: str, limit: int, max_chars: int) -> str:
        # 局部导入：topics 依赖 store 里的数据类型，模块级互导会成环。
        from .topics import format_topic

        return format_topic(
            self.recent_messages(group_openid, limit),
            max_chars,
            self.speaker_labels(group_openid),
        )

    @staticmethod
    def _search_term_weight(term: str) -> int:
        return min(max(len(term), 2), 6)

    @staticmethod
    def _search_parts(query: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(r"[\s，。、；：,.;:!！?？~～“”\"'（）()【】\[\]《》—\-]+", query)
            if part.strip()
        ]

    @classmethod
    def _search_terms(cls, query: str) -> list[str]:
        """把检索描述拆成可用于包含匹配的关键词。

        意图模型常把 query 写成“小虾主人 让小毛豆 说说主人”这样的描述，
        原文却可能只是“说说你主人”。整句短语匹配会漏掉，拆成两字片段后
        由命中数量和来源排序共同筛选，更接近人在聊天记录里找原话。
        """
        parts = cls._search_parts(query)
        terms: list[str] = []
        seen: set[str] = set()
        single_only = sum(len(part) for part in parts) == 1
        for part in parts:
            candidates = [part]
            if part.isascii():
                candidates = [part]
            elif len(part) >= 3:
                candidates = [
                    part[index : index + 2]
                    for index in range(len(part) - 1)
                ] + [part]
            for candidate in candidates:
                if not candidate or (len(candidate) == 1 and not single_only):
                    continue
                if candidate not in seen:
                    terms.append(candidate)
                    seen.add(candidate)
                if len(terms) >= 24:
                    return terms
        return terms

    @staticmethod
    def _part_priority(part: str, priority_text: str) -> float:
        if not priority_text or not part:
            return 1.0
        covered: set[int] = set()
        if part in priority_text:
            covered.update(range(len(part)))
        elif not part.isascii() and len(part) >= 3:
            for index in range(len(part) - 1):
                if part[index : index + 2] in priority_text:
                    covered.add(index)
                    covered.add(index + 1)
        return 1.0 + 4.0 * len(covered) / len(part)

    @classmethod
    def _search_coverage(
        cls,
        query: str,
        text: str,
        priority_text: str = "",
    ) -> tuple[float, int]:
        """按查询字符覆盖率排序，避免重叠二字片段被重复加分。

        priority_text 是用户本轮的原话。模型补出来的称呼、解释可能比原话更
        抢眼，把原话里出现过的片段加权，能让真正的原消息排上来。
        """
        covered: set[tuple[int, int]] = set()
        exact_parts = 0
        parts = cls._search_parts(query)
        for part_index, part in enumerate(parts):
            if part and part in text:
                exact_parts += 1
                covered.update((part_index, index) for index in range(len(part)))
                continue
            if part.isascii() or len(part) < 3:
                continue
            for index in range(len(part) - 1):
                if part[index : index + 2] in text:
                    covered.add((part_index, index))
                    covered.add((part_index, index + 1))
        weighted = sum(
            cls._part_priority(parts[part_index], priority_text)
            for part_index, _ in covered
        )
        return weighted, exact_parts

    def search_messages(
        self,
        group_openid: str,
        query: str,
        limit: int = 20,
        *,
        priority_text: str = "",
    ) -> list[SearchResult]:
        terms = self._search_terms(query)
        if not terms:
            return []

        term_rows = ", ".join("(?, ?)" for _ in terms)
        term_params = [
            value
            for term in terms
            for value in (term, self._search_term_weight(term))
        ]
        candidate_limit = min(max(limit * 20, 400), 1200)
        candidates = self._conn.execute(
            f"""
            WITH search_terms(term, weight) AS (VALUES {term_rows})
            SELECT m.row_id, m.username, m.content, m.timestamp, m.role,
                   (
                       SELECT COALESCE(SUM(weight), 0)
                       FROM search_terms
                       WHERE INSTR(m.content, term) > 0 OR INSTR(m.username, term) > 0
                   ) AS score
            FROM messages AS m
            WHERE m.group_openid = ?
              AND EXISTS (
                  SELECT 1
                  FROM search_terms
                  WHERE INSTR(m.content, term) > 0 OR INSTR(m.username, term) > 0
              )
            ORDER BY score DESC, m.row_id DESC
            LIMIT ?
            """,
            (*term_params, group_openid, candidate_limit),
        ).fetchall()

        ranked_rows = []
        for row in candidates:
            coverage, exact_parts = self._search_coverage(
                query,
                f"{row['content'] or ''}\n{row['username'] or ''}",
                priority_text,
            )
            ranked_rows.append(
                (row["role"] == "bot", -coverage, -exact_parts, -int(row["row_id"]), row)
            )
        ranked_rows.sort(key=lambda item: item[:4])
        return [
            SearchResult(
                row_id=row["row_id"],
                username=row["username"] or "unknown",
                content=row["content"] or "",
                timestamp=row["timestamp"] or "",
            )
            for row in (item[4] for item in ranked_rows[:limit])
        ]

    @staticmethod
    def format_search_results(results: list[SearchResult], max_chars: int = 12000) -> str:
        lines: list[str] = []
        total = 0
        for result in results:
            timestamp = (result.timestamp or "")[:19].replace("T", " ")
            # 看图记录拆成独立的〔图〕行：检索回放里也要看得出「这是我当时看到的图」。
            line = f"{timestamp} {result.username}: {format_message_content(result.content)}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    @staticmethod
    def _card_from_row(row: sqlite3.Row) -> MemoryCard:
        return MemoryCard(
            id=int(row["id"]),
            kind=row["kind"],
            person=row["person"],
            content=row["content"],
            source_message_id=row["source_message_id"] or "",
            source_row_id=(
                int(row["source_row_id"]) if row["source_row_id"] is not None else None
            ),
        )

    def add_memory_card(
        self,
        *,
        group_openid: str,
        kind: str,
        person: str,
        content: str,
        source_message_id: str | None = None,
        source_row_id: int | None = None,
    ) -> MemoryCard | None:
        content = content.strip()
        if not content:
            return None
        # 同一个人被记成「罗晓霞 / 二十四桥明月夜 / 小虾」三种写法时，模型会当成三个人，
        # 称呼也就跟着乱。群里统一过叫法之后，新卡的 person 在写库前就归一。
        person = self.canonical_person(group_openid, person)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO memory_cards (
                    group_openid, kind, person, content, source_message_id, source_row_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    group_openid,
                    kind,
                    person,
                    content,
                    source_message_id,
                    source_row_id,
                ),
            )
        row = self._conn.execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memory_cards WHERE group_openid = ? AND content = ?",
            (group_openid, content),
        ).fetchone()
        if row is None:
            return None
        return self._card_from_row(row)

    def memory_cards(self, group_openid: str, limit: int = 50) -> list[MemoryCard]:
        rows = self._conn.execute(
            f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memory_cards
            WHERE group_openid = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [self._card_from_row(row) for row in rows]

    def all_memory_cards(self, group_openid: str) -> list[MemoryCard]:
        """按写入顺序取出全部长期记忆。"""
        rows = self._conn.execute(
            f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memory_cards
            WHERE group_openid = ?
            ORDER BY id
            """,
            (group_openid,),
        ).fetchall()
        return [self._card_from_row(row) for row in rows]

    def count_memory_cards(self, group_openid: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM memory_cards WHERE group_openid = ?",
            (group_openid,),
        ).fetchone()
        return int(row["n"] if row else 0)

    def memory_cards_for_prompt(
        self,
        group_openid: str,
        *,
        max_chars: int = MEMORY_PROMPT_MAX_CHARS,
        total: int = MEMORY_PROMPT_MAX_CARDS,
    ) -> list[MemoryCard]:
        """挑出要塞进模型上下文的记忆卡。

        两个历史坑：
        1. 最早是 memory_cards(limit=30)，而它 ORDER BY id DESC LIMIT 30，于是最早的
           卡片（真名、纪念日、称呼、人设）永远进不了上下文。
        2. 改成按 kind 配额之后，preference 只给 6 条，可群里实际攒了 25 条——被挤掉的
           恰好是「别分点」「别引 4399」「讲话俏皮」这些行为规则，于是模型天天违反。
        所以这里不再按 kind 配额，改成按字符预算装：按「规则优先、流水账靠后」排序
        依次装，装满为止。最后按 id 正序输出。
        """
        cards = self.all_memory_cards(group_openid)
        if not cards:
            return []

        cards = self._dedupe_memory_cards(cards)
        ordered = sorted(cards, key=self._memory_prompt_rank)

        picked: list[MemoryCard] = []
        used = 0
        for card in ordered:
            if len(picked) >= total:
                break
            cost = self._memory_card_cost(card)
            if used + cost > max_chars:
                # 这张太长了就跳过，继续试后面更短的，把预算填满而不是直接截断。
                continue
            picked.append(card)
            used += cost

        picked.sort(key=lambda card: card.id)
        return picked

    @staticmethod
    def _memory_card_cost(card: MemoryCard) -> int:
        # format_memory_cards 渲染成 "[kind] person: content"，按这个长度估。
        return len(card.content) + len(card.person) + len(card.kind) + 6

    @staticmethod
    def _memory_prompt_rank(card: MemoryCard) -> tuple[int, int]:
        kind = card.kind if card.kind in MEMORY_KIND_PRIORITY else "fact"
        priority = MEMORY_KIND_PRIORITY.index(kind)
        # 流水账看新的（越近越相关），规则和身份看老的（那批是人设地基）。
        return (priority, -card.id if kind in MEMORY_EPISODIC_KINDS else card.id)

    @staticmethod
    def _dedupe_memory_cards(cards: list[MemoryCard]) -> list[MemoryCard]:
        """同一句话被存了好几遍时只留最早写入的一张。"""
        chosen: dict[tuple[str, str], MemoryCard] = {}
        for card in cards:
            key = (card.person, _MEMORY_NOISE.sub("", card.content).lower())
            if not key[1]:
                continue
            chosen.setdefault(key, card)
        return list(chosen.values())

    def existing_memory_lines(
        self,
        group_openid: str,
        *,
        max_chars: int = MEMORY_PROMPT_MAX_CHARS,
    ) -> list[str]:
        """给需要看已有记忆的地方一份清单，用来避免重复写入。"""
        lines: list[str] = []
        total = 0
        for card in self.memory_cards_for_prompt(group_openid):
            line = f"- [{card.kind}/{card.person}] {card.content}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return lines

    def memory_cards_page(
        self,
        group_openid: str,
        *,
        page: int = 1,
        page_size: int = 12,
        kind: str | None = None,
        person: str | None = None,
        match: str | None = None,
    ) -> tuple[list[MemoryCard], int, int, int]:
        """分页列记忆，按 id 正序。返回 (本页, 当前页, 总页数, 总条数)。

        可选过滤：kind 精确匹配；person 先归一成群里统一的叫法再做包含匹配；
        match 对内容或 person 做包含匹配。
        """
        cards = self.all_memory_cards(group_openid)
        if kind:
            cards = [card for card in cards if card.kind == kind]
        person_needle = self.canonical_person(group_openid, person) if person else ""
        if person_needle:
            cards = [card for card in cards if person_needle in (card.person or "")]
        if match:
            needle = match.strip().lower()
            if needle:
                cards = [
                    card
                    for card in cards
                    if needle in card.content.lower() or needle in (card.person or "").lower()
                ]
        total = len(cards)
        pages = max(1, (total + page_size - 1) // page_size)
        current = min(max(page, 1), pages)
        start = (current - 1) * page_size
        return cards[start:start + page_size], current, pages, total

    # ---------- 称呼归一：同一个人在记忆里只留一种写法 ----------

    def person_aliases(self, group_openid: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT alias, canonical FROM person_aliases WHERE group_openid = ? ORDER BY alias",
            (group_openid,),
        ).fetchall()
        return {str(row["alias"]): str(row["canonical"]) for row in rows}

    def canonical_person(self, group_openid: str, person: str | None) -> str:
        """把一个叫法翻译成群里统一过的叫法；没登记过就原样返回。"""
        name = (person or "").strip()
        if not name:
            return name
        aliases = self.person_aliases(group_openid)
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = (aliases[name] or "").strip() or name
        return name

    def set_person_alias(self, group_openid: str, alias: str, canonical: str) -> int:
        """登记 alias→canonical，并把已有记忆卡里的 person 一起改过来。返回改写条数。"""
        alias = (alias or "").strip()
        canonical = self.canonical_person(group_openid, canonical)
        if not alias or not canonical or alias == canonical:
            return 0
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO person_aliases (group_openid, alias, canonical)
                VALUES (?, ?, ?)
                ON CONFLICT(group_openid, alias) DO UPDATE SET
                    canonical = excluded.canonical,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (group_openid, alias, canonical),
            )
            # 链条压平：原来 X→alias 的登记直接指到最终叫法，免得以后绕圈。
            self._conn.execute(
                "UPDATE person_aliases SET canonical = ? WHERE group_openid = ? AND canonical = ?",
                (canonical, group_openid, alias),
            )
            cursor = self._conn.execute(
                "UPDATE memory_cards SET person = ? WHERE group_openid = ? AND person = ?",
                (canonical, group_openid, alias),
            )
            # 称呼登记跟着人走：老叫法上定过的称呼搬到新叫法名下（新叫法已有就保留新的）。
            self._conn.execute(
                """
                INSERT INTO person_addresses (group_openid, person, address)
                SELECT ?, ?, address FROM person_addresses
                WHERE group_openid = ? AND person = ?
                ON CONFLICT(group_openid, person) DO NOTHING
                """,
                (group_openid, canonical, group_openid, alias),
            )
            self._conn.execute(
                "DELETE FROM person_addresses WHERE group_openid = ? AND person = ?",
                (group_openid, alias),
            )
        rewritten = int(cursor.rowcount or 0)
        if rewritten:
            logger.info(
                "person alias %s -> %s rewrote %s memory cards", alias, canonical, rewritten
            )
        return rewritten

    def clear_person_alias(self, group_openid: str, alias: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM person_aliases WHERE group_openid = ? AND alias = ?",
                (group_openid, (alias or "").strip()),
            )
        return cursor.rowcount > 0

    # ---------- 称呼：机器人该怎么叫某个人 ----------
    #
    # 称呼属于人设，代码里不写死任何一个：叫什么、叫谁，全由群里 @ 我下指令决定。
    # 存的时候先把 person 归一，这样「小虾」和「二十四桥明月夜」不会各存一份。

    def person_addresses(self, group_openid: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT person, address FROM person_addresses WHERE group_openid = ? ORDER BY person",
            (group_openid,),
        ).fetchall()
        return {str(row["person"]): str(row["address"]) for row in rows}

    def speaker_labels(self, group_openid: str, limit: int = 50) -> dict[str, str]:
        """说话人标签表：openid / 显示名 → 该挂的名字。登记过称呼的用称呼。

        渲染聊天记录时用它给每条消息挂名字，模型不用再猜「这句话是谁说的」。
        两个 key 都能查（openid 优先），历史消息的显示名变了也还能命中。
        """
        labels: dict[str, str] = {}
        for member in self.known_members(group_openid, limit=limit):
            name = (member.username or "").strip()
            openid = (member.member_openid or "").strip()
            label = (
                (self.address_for_member(group_openid, openid) if openid else "")
                or (self.address_for(group_openid, name) if name else "")
                or name
                or openid
            )
            if not label:
                continue
            if openid:
                labels.setdefault(openid, label)
            if name:
                labels.setdefault(name, label)
        return labels

    def address_for_member(self, group_openid: str, member_openid: str | None) -> str:
        """按 openid 查称呼。对方改昵称也照样认得出来（同一 openid 多行时取最近登记的）。"""
        openid = (member_openid or "").strip()
        if not openid:
            return ""
        row = self._conn.execute(
            """
            SELECT address FROM person_addresses
            WHERE group_openid = ? AND member_openid = ?
            ORDER BY updated_at DESC, rowid DESC
            LIMIT 1
            """,
            (group_openid, openid),
        ).fetchone()
        return str(row["address"]) if row else ""

    def member_openid_named(self, group_openid: str, person: str) -> str:
        """按叫法反查 openid：显示名或它登记过的别名匹配上就算。查不到返回空串。"""
        name = (person or "").strip()
        if not name:
            return ""
        aliases = {
            alias
            for alias, canonical in self.person_aliases(group_openid).items()
            if self.canonical_person(group_openid, alias) == name
        }
        wanted = {name, *aliases}
        for member in self.known_members(group_openid, limit=50):
            if (member.username or "").strip() in wanted:
                return (member.member_openid or "").strip()
        return ""

    def address_for(self, group_openid: str, person: str | None) -> str:
        """拿到该怎么称呼这个人；没登记过就返回空串。"""
        name = self.canonical_person(group_openid, person)
        if not name:
            return ""
        row = self._conn.execute(
            "SELECT address FROM person_addresses WHERE group_openid = ? AND person = ?",
            (group_openid, name),
        ).fetchone()
        return str(row["address"]) if row else ""

    def set_person_address(self, group_openid: str, person: str, address: str) -> str:
        """登记「以后这么叫他」。返回实际落库的人名（已归一），没存成返回空串。"""
        name = self.canonical_person(group_openid, person)
        address = (address or "").strip().strip("「」“”\"'")
        if not name or not address:
            return ""
        openid = self.member_openid_named(group_openid, name)
        with self._conn:
            if openid:
                # 同一个人只挂一个称呼：openid 从旧叫法那行摘掉，免得两行抢同一个 openid。
                self._conn.execute(
                    """
                    UPDATE person_addresses SET member_openid = ''
                    WHERE group_openid = ? AND member_openid = ? AND person != ?
                    """,
                    (group_openid, openid, name),
                )
            self._conn.execute(
                """
                INSERT INTO person_addresses (group_openid, person, address, member_openid)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_openid, person) DO UPDATE SET
                    address = excluded.address,
                    member_openid = CASE
                        WHEN excluded.member_openid != '' THEN excluded.member_openid
                        ELSE person_addresses.member_openid
                    END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (group_openid, name, address, openid),
            )
        logger.info("person address set: %s -> %s (openid bound=%s)", name, address, bool(openid))
        return name

    def backfill_address_openids(self, group_openid: str) -> int:
        """给历史里没绑 openid 的称呼补上；每次启动跑一遍，重复执行安全。"""
        rows = self._conn.execute(
            """
            SELECT person FROM person_addresses
            WHERE group_openid = ? AND (member_openid IS NULL OR member_openid = '')
            """,
            (group_openid,),
        ).fetchall()
        bound = 0
        for row in rows:
            person = str(row["person"])
            openid = self.member_openid_named(group_openid, person)
            if not openid:
                continue
            with self._conn:
                self._conn.execute(
                    "UPDATE person_addresses SET member_openid = ? WHERE group_openid = ? AND person = ?",
                    (openid, group_openid, person),
                )
            bound += 1
            logger.info("person address %s bound to openid", person)
        return bound

    def backfill_face_tokens(self, group_openid: str) -> int:
        """从历史消息里补出群用过的表情，只在群里一个表情都还没记下时跑。

        表情名 ↔ faceId 只能来自真实消息（代码里不写死任何一个），而入库学习是
        face_tokens 上线之后才开始的。所以首次升级后补一次历史，之后每次启动都是空转。
        返回补下的表情个数。
        """
        if not group_openid:
            return 0
        existing = self._conn.execute(
            "SELECT COUNT(*) AS n FROM face_tokens WHERE group_openid = ?",
            (group_openid,),
        ).fetchone()
        if existing and int(existing["n"]) > 0:
            return 0
        rows = self._conn.execute(
            "SELECT raw_json FROM messages WHERE group_openid = ? AND raw_json LIKE '%faceType%'",
            (group_openid,),
        ).fetchall()
        learned = 0
        for row in rows:
            try:
                raw = json.loads(row["raw_json"] or "{}")
            except (TypeError, ValueError):
                continue
            content = raw.get("content") if isinstance(raw, dict) else None
            learned += self.remember_faces(group_openid, content)
        if learned:
            logger.info(
                "backfilled %d face tokens for group %s", learned, group_openid
            )
        return learned

    def clear_person_address(self, group_openid: str, person: str) -> bool:
        name = self.canonical_person(group_openid, person)
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM person_addresses WHERE group_openid = ? AND person = ?",
                (group_openid, name),
            )
        return cursor.rowcount > 0

    def search_memory_cards(self, group_openid: str, query: str, limit: int = 10) -> list[MemoryCard]:
        terms = self._search_terms(query)
        if not terms:
            return []
        term_rows = ", ".join("(?, ?)" for _ in terms)
        term_params = [
            value
            for term in terms
            for value in (term, self._search_term_weight(term))
        ]
        rows = self._conn.execute(
            f"""
            WITH search_terms(term, weight) AS (VALUES {term_rows})
            SELECT {_MEMORY_COLUMNS},
                   (
                       SELECT COALESCE(SUM(weight), 0)
                       FROM search_terms
                       WHERE INSTR(kind, term) > 0
                          OR INSTR(person, term) > 0
                          OR INSTR(content, term) > 0
                   ) AS score
            FROM memory_cards
            WHERE group_openid = ?
              AND EXISTS (
                  SELECT 1
                  FROM search_terms
                  WHERE INSTR(kind, term) > 0
                     OR INSTR(person, term) > 0
                     OR INSTR(content, term) > 0
              )
            ORDER BY score DESC, id DESC
            LIMIT ?
            """,
            (*term_params, group_openid, limit),
        ).fetchall()
        return [self._card_from_row(row) for row in rows]

    def find_memory_cards(
        self,
        group_openid: str,
        *,
        memory_id: int | list[int] | None = None,
        query: str | None = None,
    ) -> list[MemoryCard]:
        if memory_id is not None:
            # memory_id 可以是单个编号，也可以是一串编号（“把 3、7、12 一起删掉”）。
            ids = memory_id if isinstance(memory_id, (list, tuple)) else [memory_id]
            clean: list[int] = []
            for item in ids:
                try:
                    clean.append(int(item))
                except (TypeError, ValueError):
                    continue
            if not clean:
                return []
            placeholders = ",".join("?" for _ in clean)
            rows = self._conn.execute(
                f"""
                SELECT {_MEMORY_COLUMNS}
                FROM memory_cards
                WHERE group_openid = ? AND id IN ({placeholders})
                ORDER BY id
                """,
                (group_openid, *clean),
            ).fetchall()
            return [self._card_from_row(row) for row in rows]

        return self.search_memory_cards(group_openid, query or "")

    def find_memory_card_by_id(self, group_openid: str, memory_id: int) -> MemoryCard | None:
        rows = self.find_memory_cards(group_openid, memory_id=memory_id)
        return rows[0] if rows else None

    def delete_memory_card(self, group_openid: str, memory_id: int) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memory_cards WHERE group_openid = ? AND id = ?",
                (group_openid, memory_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def format_memory_cards(cards: list[MemoryCard], max_chars: int = 8000) -> str:
        if not cards:
            return ""
        lines: list[str] = []
        total = 0
        for card in cards:
            line = f"[{card.kind}] {card.person}: {card.content}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def groups(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT group_openid FROM messages ORDER BY group_openid"
        ).fetchall()
        return [row["group_openid"] for row in rows]

    def pending_memory_batch(
        self,
        group_openid: str,
        *,
        min_messages: int,
        limit: int = 200,
    ) -> MemoryBatch | None:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(end_row_id), 0) AS last_row_id FROM conversation_summaries WHERE group_openid = ?",
            (group_openid,),
        ).fetchone()
        start_row_id = int(row["last_row_id"] or 0) + 1
        rows = self._conn.execute(
            """
            SELECT row_id, username, member_openid, role, content, timestamp, source
            FROM messages
            WHERE group_openid = ? AND row_id >= ?
            ORDER BY row_id
            LIMIT ?
            """,
            (group_openid, start_row_id, limit),
        ).fetchall()
        if len(rows) < min_messages:
            return None
        messages = [
            StoredMessage(
                username=row["username"] or "unknown",
                member_openid=row["member_openid"] or "",
                role=row["role"] or "member",
                content=row["content"] or "",
                timestamp=row["timestamp"] or "",
                source=row["source"] or "",
                row_id=int(row["row_id"] or 0),
            )
            for row in rows
        ]
        return MemoryBatch(
            start_row_id=start_row_id,
            end_row_id=start_row_id + len(rows) - 1,
            messages=messages,
        )

    def save_memory_batch(
        self,
        *,
        group_openid: str,
        start_row_id: int,
        end_row_id: int,
        summary: str,
        memories: list[dict[str, str]],
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO conversation_summaries (
                    group_openid, start_row_id, end_row_id, summary
                ) VALUES (?, ?, ?, ?)
                """,
                (group_openid, start_row_id, end_row_id, summary),
            )
            for memory in memories:
                content = memory.get("content", "").strip()
                if not content:
                    continue
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_cards (
                        group_openid, kind, person, content, source_row_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        group_openid,
                        memory.get("kind", "fact"),
                        memory.get("person", "两人"),
                        content,
                        end_row_id,
                    ),
                )

    def latest_summary_row_id(self, group_openid: str) -> int:
        """最近一次折叠进摘要的最后一行，用来给话题切分画一条硬边界。"""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(end_row_id), 0) AS last_row_id"
            " FROM conversation_summaries WHERE group_openid = ?",
            (group_openid,),
        ).fetchone()
        return int(row["last_row_id"] or 0)

    def conversation_summaries(self, group_openid: str, limit: int = 5) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT summary FROM conversation_summaries
            WHERE group_openid = ?
            ORDER BY end_row_id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [row["summary"] for row in reversed(rows)]

    def clamp_long_summaries(
        self,
        group_openid: str,
        max_chars: int = SUMMARY_MAX_CHARS,
    ) -> int:
        """把库里超长的摘要截回预算内，返回更新条数。"""
        rows = self._conn.execute(
            """
            SELECT id, summary FROM conversation_summaries
            WHERE group_openid = ? AND length(summary) > ?
            """,
            (group_openid, max_chars),
        ).fetchall()
        with self._conn:
            for row in rows:
                self._conn.execute(
                    "UPDATE conversation_summaries SET summary = ? WHERE id = ?",
                    (clip_summary(row["summary"], max_chars), row["id"]),
                )
        return len(rows)

    def dedupe_memory_cards(self, group_openid: str) -> int:
        """删除 person+content 完全相同的重复卡，保留 id 最小的一张，返回删除条数。"""
        before = self._conn.execute(
            "SELECT COUNT(*) FROM memory_cards WHERE group_openid = ?", (group_openid,)
        ).fetchone()[0]
        with self._conn:
            self._conn.execute(
                """
                DELETE FROM memory_cards
                WHERE group_openid = ?
                  AND id NOT IN (
                      SELECT MIN(id) FROM memory_cards
                      WHERE group_openid = ?
                      GROUP BY person, content
                  )
                """,
                (group_openid, group_openid),
            )
        after = self._conn.execute(
            "SELECT COUNT(*) FROM memory_cards WHERE group_openid = ?", (group_openid,)
        ).fetchone()[0]
        return before - after

    def create_task(
        self,
        *,
        group_openid: str,
        title: str,
        content: str | None,
        generation_prompt: str | None,
        schedule_type: str,
        next_run_at: str,
        timezone: str,
        time_of_day: str | None,
        weekday: int | None,
        created_by_openid: str,
        created_by_name: str,
        month: int | None = None,
        day: int | None = None,
        leap_month: bool = False,
        remind_days_ahead: int = 0,
        offset_days: int = 0,
        linked_task_id: int | None = None,
    ) -> ScheduledTask:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO scheduled_tasks (
                    group_openid, title, content, generation_prompt, schedule_type,
                    next_run_at, timezone, time_of_day, weekday, month, day,
                    leap_month, remind_days_ahead, offset_days, linked_task_id,
                    created_by_openid, created_by_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_openid,
                    title,
                    content,
                    generation_prompt,
                    schedule_type,
                    next_run_at,
                    timezone,
                    time_of_day,
                    weekday,
                    month,
                    day,
                    1 if leap_month else 0,
                    int(remind_days_ahead or 0),
                    int(offset_days or 0),
                    linked_task_id,
                    created_by_openid,
                    created_by_name,
                ),
            )
            task_id = int(cursor.lastrowid)
        task = self.get_task(group_openid, task_id)
        assert task is not None
        return task

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> ScheduledTask:
        return ScheduledTask(
            id=row["id"],
            group_openid=row["group_openid"],
            title=row["title"],
            content=row["content"],
            generation_prompt=row["generation_prompt"],
            schedule_type=row["schedule_type"],
            next_run_at=row["next_run_at"],
            timezone=row["timezone"],
            time_of_day=row["time_of_day"],
            weekday=row["weekday"],
            status=row["status"],
            skip_next=bool(row["skip_next"]),
            last_sent_at=row["last_sent_at"],
            failure_count=row["failure_count"],
            created_by_openid=row["created_by_openid"],
            created_by_name=row["created_by_name"],
            month=row["month"],
            day=row["day"],
            leap_month=bool(row["leap_month"]),
            remind_days_ahead=int(row["remind_days_ahead"] or 0),
            offset_days=int(row["offset_days"] or 0),
            linked_task_id=row["linked_task_id"],
        )

    def get_task(self, group_openid: str, task_id: int) -> ScheduledTask | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE group_openid = ? AND id = ?",
            (group_openid, task_id),
        ).fetchone()
        return self._task_from_row(row) if row else None

    def tasks(self, group_openid: str) -> list[ScheduledTask]:
        rows = self._conn.execute(
            """
            SELECT * FROM scheduled_tasks
            WHERE group_openid = ? AND status IN ('active', 'paused')
            ORDER BY next_run_at, id
            """,
            (group_openid,),
        ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def active_task_count(self, group_openid: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM scheduled_tasks WHERE group_openid = ? AND status IN ('active', 'paused')",
            (group_openid,),
        ).fetchone()
        return int(row["count"])

    def find_task(
        self,
        group_openid: str,
        *,
        task_id: int | None = None,
        query: str | None = None,
    ) -> list[ScheduledTask]:
        if task_id is not None:
            task = self.get_task(group_openid, task_id)
            return [task] if task and task.status != "deleted" else []

        query = (query or "").strip()
        if not query:
            return []
        needle = self._search_text(query)
        rows = self._conn.execute(
            """
            SELECT * FROM scheduled_tasks
            WHERE group_openid = ?
              AND status IN ('active', 'paused')
              AND (
                INSTR(title, ?) > 0 OR INSTR(content, ?) > 0 OR INSTR(generation_prompt, ?) > 0
              )
            ORDER BY next_run_at, id
            """,
            (group_openid, needle, needle, needle),
        ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def set_task_status(self, group_openid: str, task_id: int, status: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (status, group_openid, task_id),
            )
        return cursor.rowcount == 1

    def set_task_next_run(self, group_openid: str, task_id: int, next_run_at: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET next_run_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (next_run_at, group_openid, task_id),
            )
        return cursor.rowcount == 1

    def set_task_skip_next(self, group_openid: str, task_id: int, skip_next: bool) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET skip_next = ?, updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (1 if skip_next else 0, group_openid, task_id),
            )
        return cursor.rowcount == 1

    def delete_task(self, group_openid: str, task_id: int) -> bool:
        removed = self.set_task_status(group_openid, task_id, "deleted")
        # 预热任务挂在主任务上，主任务取消时一起删掉，不然它会孤零零地继续提醒
        with self._conn:
            self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND linked_task_id = ? AND status != 'deleted'
                """,
                (group_openid, task_id),
            )
        return removed

    def due_tasks(self, now_iso: str) -> list[ScheduledTask]:
        rows = self._conn.execute(
            """
            SELECT * FROM scheduled_tasks
            WHERE status = 'active' AND next_run_at <= ?
            ORDER BY next_run_at, id
            """,
            (now_iso,),
        ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def complete_task_run(
        self,
        *,
        group_openid: str,
        task_id: int,
        success: bool,
        now_iso: str,
        next_run_at: str | None,
        finish: bool,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = CASE WHEN ? THEN 'done' ELSE status END,
                    last_sent_at = ?,
                    failure_count = failure_count + ?,
                    next_run_at = COALESCE(?, next_run_at),
                    skip_next = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (
                    1 if finish else 0,
                    now_iso if success else None,
                    0 if success else 1,
                    next_run_at,
                    group_openid,
                    task_id,
                ),
            )

    def skip_task_run(
        self,
        *,
        group_openid: str,
        task_id: int,
        next_run_at: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE scheduled_tasks
                SET next_run_at = ?, skip_next = 0, updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (next_run_at, group_openid, task_id),
            )

    @staticmethod
    def format_tasks(tasks: list[ScheduledTask]) -> str:
        if not tasks:
            return "当前没有定时任务。可以直接对我说：每天 22:00 提醒我们互道晚安。"
        lines: list[str] = []
        for task in tasks:
            schedule = schedule_label(task)
            body = task.content or task.generation_prompt or ""
            status = "，已暂停" if task.status == "paused" else ""
            skip = "，下次跳过" if task.skip_next else ""
            lines.append(f"{task.id}. {schedule} | {task.title} | {body}{status}{skip}")
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        message_row = self._conn.execute(
            "SELECT COUNT(*) AS messages, COUNT(DISTINCT group_openid) AS groups FROM messages"
        ).fetchone()
        memory_row = self._conn.execute("SELECT COUNT(*) AS memory_cards FROM memory_cards").fetchone()
        task_row = self._conn.execute(
            "SELECT COUNT(*) AS scheduled_tasks FROM scheduled_tasks WHERE status IN ('active', 'paused')"
        ).fetchone()
        return {
            "messages": message_row["messages"],
            "groups": message_row["groups"],
            "memory_cards": memory_row["memory_cards"],
            "scheduled_tasks": task_row["scheduled_tasks"],
        }

    # ---------- 主动关心：一条消息只准触发一次 ----------

    def reserve_care_message(
        self,
        *,
        message_key: str,
        group_openid: str,
        member_openid: str,
        content: str,
        created_at: str,
    ) -> bool:
        """给这条消息占坑；False 说明已经为它想过要不要关心了。

        先占坑再问模型，这样同一条消息被 QQ 重推时不会重复关心。
        """
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO care_log
                    (message_key, group_openid, member_openid, content, sent, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (message_key, group_openid, member_openid, content, created_at),
            )
            return cursor.rowcount > 0

    def mark_care_sent(self, message_key: str) -> None:
        """这条消息确实发出去关心了；只有发出去的才算进频率预算。"""
        with self._conn:
            self._conn.execute(
                "UPDATE care_log SET sent = 1 WHERE message_key = ?",
                (message_key,),
            )

    def sent_care_count(self, *, group_openid: str, member_openid: str, since_iso: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total FROM care_log
            WHERE group_openid = ? AND member_openid = ? AND sent = 1 AND created_at >= ?
            """,
            (group_openid, member_openid, since_iso),
        ).fetchone()
        return int(row["total"])

    def last_care_at(self, *, group_openid: str, member_openid: str) -> str | None:
        """上一次真的发出关心是什么时候（没发出去的不算）。"""
        row = self._conn.execute(
            """
            SELECT created_at FROM care_log
            WHERE group_openid = ? AND member_openid = ? AND sent = 1
            ORDER BY created_at DESC, message_key DESC LIMIT 1
            """,
            (group_openid, member_openid),
        ).fetchone()
        return str(row["created_at"]) if row else None

    def proactive_message_count(self, *, group_openid: str, since_iso: str) -> int:
        """今天已经自发过多少条（定时任务和主动关心共享这个预算）。"""
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total FROM messages
            WHERE group_openid = ? AND source IN ('BOT_TASK', 'BOT_CARE') AND timestamp >= ?
            """,
            (group_openid, since_iso),
        ).fetchone()
        return int(row["total"])

    # ---------- 群内偏好：优先级高于 .env ----------

    def get_preference(self, group_openid: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM group_preferences WHERE group_openid = ? AND key = ?",
            (group_openid, key),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_preference(self, group_openid: str, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO group_preferences (group_openid, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(group_openid, key)
                DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (group_openid, key, value),
            )

    def preferences(self, group_openid: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM group_preferences WHERE group_openid = ? ORDER BY key",
            (group_openid,),
        ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def clear_preference(self, group_openid: str, key: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM group_preferences WHERE group_openid = ? AND key = ?",
                (group_openid, key),
            )
            return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()

