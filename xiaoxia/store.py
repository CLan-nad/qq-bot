from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .images import normalize_face_tags
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
class MemoryCard:
    id: int
    kind: str
    person: str
    content: str
    source_message_id: str
    pinned: bool = False
    scope: str = "permanent"
    expires_at: str | None = None
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
    "id, kind, person, content, source_message_id, pinned, scope, expires_at, source_row_id"
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

            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'scheduled_tasks'"
            ).fetchone()
            table_sql = str(row["sql"] or "") if row else ""
            if table_sql and "'yearly'" not in table_sql:
                self._rebuild_tasks_table()

        self._normalize_legacy_face_tags()

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
        text = normalize_face_tags(content).strip()
        if not text:
            text = self._attachment_summary(raw)
        if not text:
            text = "[non-text message]"

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

    def format_recent_messages(self, group_openid: str, limit: int, max_chars: int) -> str:
        messages = self.recent_messages(group_openid, limit)
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

    @staticmethod
    def _search_text(query: str) -> str:
        return query.strip()

    @staticmethod
    def _fts_query(query: str) -> str:
        return '"' + query.replace('"', '""') + '"'

    def search_messages(self, group_openid: str, query: str, limit: int = 20) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        rows: list[sqlite3.Row] = []
        if len(query) >= 3:
            try:
                rows = self._conn.execute(
                    """
                    SELECT m.row_id, m.username, m.content, m.timestamp
                    FROM messages_fts
                    JOIN messages AS m ON m.row_id = messages_fts.rowid
                    WHERE messages_fts MATCH ? AND m.group_openid = ?
                    ORDER BY messages_fts.rank, m.row_id DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), group_openid, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            needle = self._search_text(query)
            rows = self._conn.execute(
                """
                SELECT row_id, username, content, timestamp
                FROM messages
                WHERE group_openid = ?
                  AND (INSTR(content, ?) > 0 OR INSTR(username, ?) > 0)
                ORDER BY row_id DESC
                LIMIT ?
                """,
                (group_openid, needle, needle, limit),
            ).fetchall()

        return [
            SearchResult(
                row_id=row["row_id"],
                username=row["username"] or "unknown",
                content=row["content"] or "",
                timestamp=row["timestamp"] or "",
            )
            for row in rows
        ]

    @staticmethod
    def format_search_results(results: list[SearchResult], max_chars: int = 12000) -> str:
        lines: list[str] = []
        total = 0
        for result in results:
            timestamp = (result.timestamp or "")[:19].replace("T", " ")
            line = f"{timestamp} {result.username}: {result.content}"
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
            pinned=bool(row["pinned"]),
            scope=str(row["scope"] or "permanent"),
            expires_at=row["expires_at"],
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
        pinned: bool = False,
        scope: str = "permanent",
        expires_at: str | None = None,
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
                    group_openid, kind, person, content, source_message_id,
                    pinned, scope, expires_at, source_row_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_openid,
                    kind,
                    person,
                    content,
                    source_message_id,
                    1 if pinned else 0,
                    scope,
                    expires_at,
                    source_row_id,
                ),
            )
            if pinned:
                # 这条早就记住了的话 INSERT OR IGNORE 不会动它，置顶要单独补一次
                self._conn.execute(
                    """
                    UPDATE memory_cards
                    SET pinned = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE group_openid = ? AND content = ?
                    """,
                    (group_openid, content),
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

    def live_memory_cards(self, group_openid: str, now_iso: str = "") -> list[MemoryCard]:
        """按时间正序取出"还活着"的记忆：置顶的、永久的、还没过期的。"""
        rows = self._conn.execute(
            f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memory_cards
            WHERE group_openid = ?
              AND (? = '' OR pinned = 1 OR expires_at IS NULL OR expires_at > ?)
            ORDER BY id
            """,
            (group_openid, now_iso, now_iso),
        ).fetchall()
        return [self._card_from_row(row) for row in rows]

    def count_live_memory_cards(self, group_openid: str, now_iso: str = "") -> int:
        return len(self.live_memory_cards(group_openid, now_iso))

    def memory_cards_for_prompt(
        self,
        group_openid: str,
        *,
        now_iso: str = "",
        max_chars: int = MEMORY_PROMPT_MAX_CHARS,
        total: int = MEMORY_PROMPT_MAX_CARDS,
    ) -> list[MemoryCard]:
        """挑出要塞进模型上下文的记忆卡。

        两个历史坑：
        1. 最早是 memory_cards(limit=30)，而它 ORDER BY id DESC LIMIT 30，于是最早的
           卡片（真名、纪念日、称呼、人设）永远进不了上下文。
        2. 改成按 kind 配额之后，preference 只给 6 条，可群里实际攒了 25 条——被挤掉的
           恰好是「别分点」「别引 4399」「讲话俏皮」这些行为规则，于是模型天天违反。
        所以这里不再按 kind 配额，改成按字符预算装：置顶的全进，其余按「规则优先、
        流水账靠后」排序依次装，装满为止。最后按 id 正序输出。
        """
        live = self.live_memory_cards(group_openid, now_iso)
        if not live:
            return []

        cards = self._dedupe_memory_cards(live)
        pinned = [card for card in cards if card.pinned]
        rest = sorted(
            (card for card in cards if not card.pinned),
            key=self._memory_prompt_rank,
        )

        picked: list[MemoryCard] = []
        used = 0
        for card in pinned + rest:
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
        # format_memory_cards 渲染成 "[置顶★][kind] person: content"，按这个长度估。
        return len(card.content) + len(card.person) + len(card.kind) + 12

    @staticmethod
    def _memory_prompt_rank(card: MemoryCard) -> tuple[int, int]:
        kind = card.kind if card.kind in MEMORY_KIND_PRIORITY else "fact"
        priority = MEMORY_KIND_PRIORITY.index(kind)
        # 流水账看新的（越近越相关），规则和身份看老的（那批是人设地基）。
        return (priority, -card.id if kind in MEMORY_EPISODIC_KINDS else card.id)

    @staticmethod
    def _dedupe_memory_cards(cards: list[MemoryCard]) -> list[MemoryCard]:
        """同一句话被抽取器存了好几遍时只留一张，置顶的那张优先。"""
        chosen: dict[tuple[str, str], MemoryCard] = {}
        for card in cards:
            key = (card.person, _MEMORY_NOISE.sub("", card.content).lower())
            if not key[1]:
                continue
            previous = chosen.get(key)
            if previous is None or (card.pinned and not previous.pinned):
                chosen[key] = card
        return list(chosen.values())

    def existing_memory_lines(
        self,
        group_openid: str,
        *,
        now_iso: str = "",
        max_chars: int = MEMORY_PROMPT_MAX_CHARS,
    ) -> list[str]:
        """给记忆抽取器看的「已经记住了什么」清单，用来避免重复抽取。

        以前只传置顶卡，而置顶卡常年是 0 张，等于完全没做去重，于是同一件事被抽了
        三四遍（「不要分点」「不要引 4399」各存了两条），白占注入预算。
        """
        lines: list[str] = []
        total = 0
        for card in self.memory_cards_for_prompt(group_openid, now_iso=now_iso):
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
        now_iso: str = "",
        page: int = 1,
        page_size: int = 12,
        kind: str | None = None,
        person: str | None = None,
        pinned: bool | None = None,
        match: str | None = None,
    ) -> tuple[list[MemoryCard], int, int, int]:
        """分页列记忆，置顶优先、其余按 id 正序。返回 (本页, 当前页, 总页数, 总条数)。

        可选过滤：kind 精确匹配；person 先归一成群里统一的叫法再做包含匹配；
        pinned=True 只看置顶、False 只看未置顶；match 对内容或 person 做包含匹配。
        """
        live = self.live_memory_cards(group_openid, now_iso)
        if kind:
            live = [card for card in live if card.kind == kind]
        if pinned is not None:
            live = [card for card in live if bool(card.pinned) == pinned]
        person_needle = self.canonical_person(group_openid, person) if person else ""
        if person_needle:
            live = [card for card in live if person_needle in (card.person or "")]
        if match:
            needle = match.strip().lower()
            if needle:
                live = [
                    card
                    for card in live
                    if needle in card.content.lower() or needle in (card.person or "").lower()
                ]
        ordered = sorted(live, key=lambda card: (not card.pinned, card.id))
        total = len(ordered)
        pages = max(1, (total + page_size - 1) // page_size)
        current = min(max(page, 1), pages)
        start = (current - 1) * page_size
        return ordered[start:start + page_size], current, pages, total

    def set_memory_pinned(self, group_openid: str, memory_id: int, pinned: bool) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE memory_cards
                SET pinned = ?, updated_at = CURRENT_TIMESTAMP
                WHERE group_openid = ? AND id = ?
                """,
                (1 if pinned else 0, group_openid, memory_id),
            )
        return cursor.rowcount == 1

    def pinned_memory_contents(self, group_openid: str, limit: int = 80) -> list[str]:
        rows = self._conn.execute(
            f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memory_cards
            WHERE group_openid = ? AND pinned = 1
            ORDER BY id
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [row["content"] for row in rows]

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

    def purge_expired_memories(self, cutoff_iso: str) -> int:
        """删掉过期很久的临时记忆（cutoff 之前就该失效的那些）。"""
        if not cutoff_iso:
            return 0
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memory_cards WHERE expires_at IS NOT NULL AND expires_at < ?",
                (cutoff_iso,),
            )
        if cursor.rowcount:
            logger.info("purged %s expired memory cards", cursor.rowcount)
        return int(cursor.rowcount)

    def search_memory_cards(self, group_openid: str, query: str, limit: int = 10) -> list[MemoryCard]:
        query = query.strip()
        if not query:
            return []
        needle = self._search_text(query)
        rows = self._conn.execute(
            f"""
            SELECT {_MEMORY_COLUMNS}
            FROM memory_cards
            WHERE group_openid = ? AND (
                INSTR(kind, ?) > 0 OR INSTR(person, ?) > 0 OR INSTR(content, ?) > 0
            )
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_openid, needle, needle, needle, limit),
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
            # memory_id 可以是单个编号，也可以是一串编号（“把 3、7、12 一起置顶”）。
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
            marker = "[置顶★]" if card.pinned else ""
            line = f"{marker}[{card.kind}] {card.person}: {card.content}"
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
                        group_openid, kind, person, content, scope, expires_at, source_row_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        group_openid,
                        memory.get("kind", "fact"),
                        memory.get("person", "两人"),
                        content,
                        memory.get("scope") or "permanent",
                        memory.get("expires_at"),
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

