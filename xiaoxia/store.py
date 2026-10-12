from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


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


@dataclass(frozen=True)
class MemoryBatch:
    start_row_id: int
    end_row_id: int
    messages: list[StoredMessage]


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
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_openid TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    generation_prompt TEXT,
                    schedule_type TEXT NOT NULL CHECK (schedule_type IN ('once', 'daily', 'weekly')),
                    next_run_at TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    time_of_day TEXT,
                    weekday INTEGER,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'paused', 'done', 'deleted')),
                    skip_next INTEGER NOT NULL DEFAULT 0,
                    last_sent_at TEXT,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    created_by_openid TEXT NOT NULL,
                    created_by_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
                """
            )
            # Backfill existing rows after the FTS table and triggers are created.
            self._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")

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
        text = (content or "").strip()
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

    def recent_messages_with_raw(
        self,
        group_openid: str,
        limit: int,
    ) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT message_key, username, member_openid, role, content, timestamp, source, raw_json
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
                )
            )
        return messages

    def recent_messages(self, group_openid: str, limit: int) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT username, member_openid, role, content, timestamp, source
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

    def add_memory_card(
        self,
        *,
        group_openid: str,
        kind: str,
        person: str,
        content: str,
        source_message_id: str | None = None,
    ) -> MemoryCard | None:
        content = content.strip()
        if not content:
            return None
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO memory_cards (
                    group_openid, kind, person, content, source_message_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (group_openid, kind, person, content, source_message_id),
            )
        row = self._conn.execute(
            """
            SELECT id, kind, person, content, source_message_id
            FROM memory_cards
            WHERE group_openid = ? AND content = ?
            """,
            (group_openid, content),
        ).fetchone()
        if row is None:
            return None
        return MemoryCard(
            id=row["id"],
            kind=row["kind"],
            person=row["person"],
            content=row["content"],
            source_message_id=row["source_message_id"] or "",
        )

    def memory_cards(self, group_openid: str, limit: int = 50) -> list[MemoryCard]:
        rows = self._conn.execute(
            """
            SELECT id, kind, person, content, source_message_id
            FROM memory_cards
            WHERE group_openid = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_openid, limit),
        ).fetchall()
        return [
            MemoryCard(
                id=row["id"],
                kind=row["kind"],
                person=row["person"],
                content=row["content"],
                source_message_id=row["source_message_id"] or "",
            )
            for row in rows
        ]

    def search_memory_cards(self, group_openid: str, query: str, limit: int = 10) -> list[MemoryCard]:
        query = query.strip()
        if not query:
            return []
        needle = self._search_text(query)
        rows = self._conn.execute(
            """
            SELECT id, kind, person, content, source_message_id
            FROM memory_cards
            WHERE group_openid = ? AND (
                INSTR(kind, ?) > 0 OR INSTR(person, ?) > 0 OR INSTR(content, ?) > 0
            )
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_openid, needle, needle, needle, limit),
        ).fetchall()
        return [
            MemoryCard(
                id=row["id"],
                kind=row["kind"],
                person=row["person"],
                content=row["content"],
                source_message_id=row["source_message_id"] or "",
            )
            for row in rows
        ]

    def find_memory_cards(
        self,
        group_openid: str,
        *,
        memory_id: int | None = None,
        query: str | None = None,
    ) -> list[MemoryCard]:
        if memory_id is not None:
            row = self._conn.execute(
                """
                SELECT id, kind, person, content, source_message_id
                FROM memory_cards
                WHERE group_openid = ? AND id = ?
                """,
                (group_openid, memory_id),
            ).fetchone()
            return [
                MemoryCard(
                    id=row["id"],
                    kind=row["kind"],
                    person=row["person"],
                    content=row["content"],
                    source_message_id=row["source_message_id"] or "",
                )
            ] if row else []

        return self.search_memory_cards(group_openid, query or "")

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
            SELECT username, member_openid, role, content, timestamp, source
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
                        group_openid, kind, person, content
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        group_openid,
                        memory.get("kind", "fact"),
                        memory.get("person", "两人"),
                        content,
                    ),
                )

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
    ) -> ScheduledTask:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO scheduled_tasks (
                    group_openid, title, content, generation_prompt, schedule_type,
                    next_run_at, timezone, time_of_day, weekday,
                    created_by_openid, created_by_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        return self.set_task_status(group_openid, task_id, "deleted")

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
        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        lines: list[str] = []
        for task in tasks:
            if task.schedule_type == "once":
                schedule = f"单次 {task.next_run_at[:16].replace('T', ' ')}"
            elif task.schedule_type == "daily":
                schedule = f"每天 {task.time_of_day or ''}"
            else:
                weekday = weekday_names[task.weekday - 1] if task.weekday and 1 <= task.weekday <= 7 else "每周"
                schedule = f"{weekday} {task.time_of_day or ''}"
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

    def close(self) -> None:
        self._conn.close()

