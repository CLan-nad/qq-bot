from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import botpy
from botpy.message import GroupMessage
from botpy.robot import Token

from .calendar import (
    format_festivals,
    lunar_label,
    next_festival,
    solar_to_lunar,
    upcoming_festivals,
)
from .care import matched_signals, normalize_care_text, should_ask_model
from .config import Settings
from .images import (
    ImageAttachment,
    fetch_image_bytes,
    image_attachments,
    message_with_image_descriptions,
    message_with_unrecognized_images,
    needs_image_description,
    normalize_face_tags,
    to_data_url,
)
from .llm import XiaoxiaLLM
from .scheduling import advance_task, parse_datetime, resolve_next_run, schedule_label
from .store import ChatStore, MemoryCard, ScheduledTask, StoredMessage
from .topics import format_topic, split_topic
from .tools import (
    ToolResult,
    document_attachments,
    eval_math,
    fetch_page,
    read_document,
    run_shell,
    search_web,
)


logger = logging.getLogger("xiaoxia")


class RichGroupMessage(GroupMessage):
    __slots__ = ("username", "member_role", "raw")

    def __init__(self, api, event_id: str | None, data: dict[str, Any]) -> None:
        super().__init__(api, event_id, data)
        author = data.get("author") or {}
        self.username = author.get("username") or author.get("member_openid") or "unknown"
        self.member_role = author.get("member_role") or "member"
        self.raw = data


ROLE_LABELS = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}

# 这几类动作要调外部服务，通常要几秒到十几秒，先应一句「我查一下」比让人干等好。
ACK_ACTIONS = frozenset(
    {"web_search", "web_fetch", "document_read", "shell_run", "memory_search"}
)
TASK_SCHEDULE_TYPES = frozenset({"once", "daily", "weekly", "yearly", "lunar_yearly"})
TEMPORARY_MEMORY_DAYS = 14

# 没被 @ 时的主动关心，这几项可以在群里直接 @ 我改，优先级高于 .env。
# (中文名, 类型, 对应的默认配置字段)
GROUP_PREFS: dict[str, tuple[str, str, str]] = {
    "care_enabled": ("主动关心", "bool", "care_enabled"),
    "care_daily_max": ("每人每天被关心次数", "int", "care_daily_max"),
    "care_cooldown_minutes": ("同一个人两次关心的间隔（分钟）", "int", "care_cooldown_minutes"),
    "active_msg_daily_max": ("群里每天主动消息上限", "int", "active_msg_daily_max"),
}
# QQ 对主动消息的频控错误码前缀，撞上就当天收手。
RATE_LIMIT_CODE = "40034"
ACK_TEXT = "稍等，我查一下～"
FALLBACK_TEXT = "小毛豆刚刚走神了，稍后再 @ 我一次试试。"


@dataclass(frozen=True)
class IdentityContext:
    requester: StoredMessage
    partner: StoredMessage | None

    @staticmethod
    def _label(member: StoredMessage) -> str:
        role = ROLE_LABELS.get(member.role, member.role or "成员")
        return f"{member.username}（{role}）"

    def prompt(self) -> str:
        partner = self._label(self.partner) if self.partner else "暂时未识别"
        return (
            f"当前触发者：{self._label(self.requester)}\n"
            f"另一位聊天成员：{partner}\n"
            "身份要求：当前触发者用“你”指代，另一位用显示名或“你对象”指代。"
            "最近记录中的“我”以每条消息开头的显示名为准，不要把两人的经历或偏好混在一起。"
        )


class XiaoxiaClient(botpy.Client):
    def __init__(
        self,
        *,
        settings: Settings,
        store: ChatStore,
        llm: XiaoxiaLLM,
    ) -> None:
        super().__init__(
            intents=botpy.Intents(public_messages=True),
            bot_log=False,
        )
        self.settings = settings
        self.store = store
        self.llm = llm
        self._reply_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        # 群 openid -> 被 QQ 频控关静音到哪天（只当天有效，重启即复位）
        self._proactive_blocked_until: dict[str, str] = {}
        self._background_task: asyncio.Task[None] | None = None

    async def _bot_login(self, token: Token) -> None:
        await super()._bot_login(token)
        # qq-botpy 1.2.1 does not parse the new full-mode group event yet.
        state = self._connection.state
        state.parsers["group_message_create"] = self._parse_group_message_create
        state.parsers["group_at_message_create"] = self._parse_group_at_message_create
        self._connection.parser = state.parsers

    def _parse_group_message_create(self, payload: dict[str, Any]) -> None:
        message = RichGroupMessage(self.api, payload.get("id"), payload.get("d") or {})
        self.ws_dispatch("group_message_create", message)

    def _parse_group_at_message_create(self, payload: dict[str, Any]) -> None:
        message = RichGroupMessage(self.api, payload.get("id"), payload.get("d") or {})
        self.ws_dispatch("group_at_message_create", message)

    def _group_allowed(self, group_openid: str | None) -> bool:
        if not group_openid:
            return False
        return not self.settings.allowed_group_openids or group_openid in self.settings.allowed_group_openids

    @staticmethod
    def _bot_mention(message: RichGroupMessage) -> dict[str, Any] | None:
        # In full-message mode, QQ sends GROUP_MESSAGE_CREATE with a mentions list
        # instead of a separate GROUP_AT_MESSAGE_CREATE event.
        for mention in message.raw.get("mentions") or []:
            if mention.get("bot") and mention.get("is_you"):
                return mention
        return None

    @staticmethod
    def _strip_bot_mention(content: str, mention: dict[str, Any]) -> str:
        mention_id = mention.get("id") or mention.get("member_openid")
        if not mention_id:
            return content.strip()
        return re.sub(rf"\s*<@{re.escape(str(mention_id))}>\s*", " ", content).strip()

    def _save_message(self, message: RichGroupMessage, source: str) -> None:
        self.store.save_message(
            message_id=message.id,
            msg_seq=message.msg_seq,
            group_openid=message.group_openid,
            member_openid=message.author.member_openid,
            username=message.username,
            role=message.member_role,
            content=message.content,
            timestamp=message.timestamp,
            source=source,
            raw=message.raw,
        )

    def _remember_bot_reply(self, group_openid: str, content: str, source: str) -> None:
        """把机器人自己的回复写进本地库，让它下一轮能看到自己说过什么。"""
        try:
            self.store.save_bot_message(
                group_openid=group_openid,
                content=content,
                timestamp=self._now().isoformat(timespec="seconds"),
                source=source,
                username=getattr(self.robot, "name", None) or "小毛豆",
            )
        except Exception:
            logger.exception("failed to persist bot reply")

    async def _describe_attachments(
        self,
        attachments: list[ImageAttachment],
        *,
        hint: str = "",
    ) -> list[str]:
        descriptions: list[str] = []
        for attachment in attachments:
            image_ref, digest, byte_size, content_type = await self._image_reference(attachment)
            use_cache = bool(digest) and self.settings.image_cache_enabled
            cached = self.store.get_image_description(digest) if use_cache else ""
            if cached:
                logger.info("image description cache hit %s", digest[:12])
                descriptions.append(cached)
                continue
            try:
                description = await self.llm.describe_image(image_ref, hint=hint)
            except Exception:
                logger.exception(
                    "failed to describe image %s",
                    attachment.filename or attachment.content_type,
                )
                description = ""
            if not description.strip():
                logger.warning(
                    "image description empty: %s",
                    attachment.filename or attachment.content_type,
                )
            elif use_cache:
                # 空描述不写缓存，留下次重新识别的机会。
                self.store.save_image_description(
                    digest=digest,
                    description=description,
                    content_type=content_type,
                    byte_size=byte_size,
                    max_rows=self.settings.image_cache_max_rows,
                )
            descriptions.append(description)
        return descriptions

    async def _image_reference(self, attachment: ImageAttachment) -> tuple[str, str, int, str]:
        """优先自己取图并内联，失败时退回让模型直接下载原始链接。

        返回 (给视觉模型的引用, 字节的 sha256, 字节数, content_type)。
        取图失败时 digest 为空串，表示这一次不参与描述缓存。
        """
        fetched = await fetch_image_bytes(
            attachment.url,
            max_bytes=self.settings.vision_max_image_bytes,
        )
        if fetched is None:
            logger.info("image download failed, passing url through: %s", attachment.filename)
            return attachment.url, "", 0, attachment.content_type
        data, content_type = fetched
        logger.debug(
            "inlined image %s (%s bytes, %s)",
            attachment.filename,
            len(data),
            content_type,
        )
        digest = hashlib.sha256(data).hexdigest()
        return to_data_url(data, content_type), digest, len(data), content_type

    async def _describe_recent_images(
        self,
        group_openid: str,
        *,
        exclude_key: str,
        budget: int,
    ) -> int:
        """懒加载识别最近还没识别过的图片，返回剩余可用额度。"""
        if budget <= 0:
            return 0
        for stored in self.store.recent_messages_with_raw(
            group_openid,
            self.settings.vision_recent_messages,
        ):
            if budget <= 0:
                break
            if stored.message_key == exclude_key or not needs_image_description(
                stored.content, stored.raw
            ):
                continue
            attachments = image_attachments(stored.raw)[:budget]
            if not attachments:
                continue
            descriptions = await self._describe_attachments(attachments)
            budget -= len(attachments)
            if not any(description.strip() for description in descriptions):
                logger.warning(
                    "image description empty for message %s, marking unrecognized",
                    stored.message_key,
                )
                self.store.update_message_content(
                    group_openid,
                    stored.message_key,
                    message_with_unrecognized_images(stored.content),
                )
                continue
            enriched = message_with_image_descriptions(stored.content, descriptions)
            if enriched != stored.content:
                self.store.update_message_content(group_openid, stored.message_key, enriched)
        return budget

    def _now(self) -> datetime:
        return datetime.now(ZoneInfo(self.settings.timezone))

    def _identity_context(
        self,
        group_openid: str,
        requester: StoredMessage,
    ) -> IdentityContext:
        members = self.store.known_members(group_openid)
        partner = next(
            (
                member
                for member in members
                if requester.member_openid
                and member.member_openid
                and member.member_openid != requester.member_openid
            ),
            None,
        )
        if partner is None:
            partner = next(
                (
                    member
                    for member in members
                    if member.username and member.username != requester.username
                ),
                None,
            )
        return IdentityContext(requester=requester, partner=partner)

    async def on_ready(self) -> None:
        stats = self.store.stats()
        logger.info(
            "小毛豆已上线：%s | model=%s | vision=%s | stored_messages=%s | memory_cards=%s | scheduled_tasks=%s",
            getattr(self.robot, "name", "小毛豆"),
            self.settings.model,
            self.settings.vision_model if self.settings.vision_max_images_per_request > 0 else "off",
            stats["messages"],
            stats["memory_cards"],
            stats["scheduled_tasks"],
        )
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._run_background_jobs())

    async def _run_background_jobs(self) -> None:
        last_memory_run = 0.0
        while True:
            await asyncio.sleep(self.settings.scheduler_interval_seconds)
            try:
                await self._send_due_tasks()
            except Exception:
                logger.exception("failed to send due scheduled tasks")

            if time.monotonic() - last_memory_run >= self.settings.memory_interval_seconds:
                last_memory_run = time.monotonic()
                try:
                    await self._compact_memories()
                except Exception:
                    logger.exception("failed to compact conversation memories")

    async def _send_due_tasks(self) -> None:
        now = self._now()
        now_iso = now.isoformat(timespec="seconds")
        for task in self.store.due_tasks(now_iso):
            next_run = advance_task(
                schedule_type=task.schedule_type,
                from_run_at=task.next_run_at,
                timezone=task.timezone,
                time_of_day=task.time_of_day,
                weekday=task.weekday,
                month=task.month,
                day=task.day,
                leap_month=task.leap_month,
                offset_days=task.offset_days,
            )
            next_run_iso = next_run.isoformat(timespec="seconds") if next_run else None

            if task.skip_next:
                self.store.skip_task_run(
                    group_openid=task.group_openid,
                    task_id=task.id,
                    next_run_at=next_run_iso or task.next_run_at,
                )
                logger.info("skipped next run for task %s", task.id)
                continue

            try:
                content = await self._task_content(task)
                await self.api.post_group_message(
                    group_openid=task.group_openid,
                    content=content,
                )
                self._remember_bot_reply(task.group_openid, content, "BOT_TASK")
                self.store.complete_task_run(
                    group_openid=task.group_openid,
                    task_id=task.id,
                    success=True,
                    now_iso=now_iso,
                    next_run_at=next_run_iso,
                    finish=task.schedule_type == "once",
                )
                logger.info("sent scheduled task %s", task.id)
            except Exception as exc:
                self._note_rate_limit(task.group_openid, exc)
                logger.exception("failed to send scheduled task %s", task.id)
                self.store.complete_task_run(
                    group_openid=task.group_openid,
                    task_id=task.id,
                    success=False,
                    now_iso=now_iso,
                    next_run_at=next_run_iso,
                    finish=task.schedule_type == "once",
                )

    async def _task_content(self, task: ScheduledTask) -> str:
        if task.content:
            return task.content[: self.settings.reply_max_chars]

        prompt = task.generation_prompt or task.title
        recent = self.store.format_recent_messages(
            task.group_openid,
            self.settings.recent_messages,
            self.settings.max_context_chars,
        )
        return await self.llm.generate_task_message(
            title=task.title,
            prompt=prompt,
            recent_context=recent,
            memory_context=self._memory_context(task.group_openid),
        )

    async def _compact_memories(self) -> None:
        async with self._memory_lock:
            cutoff = (self._now() - timedelta(days=90)).isoformat(timespec="seconds")
            purged = self.store.purge_expired_memories(cutoff)
            if purged:
                logger.info("purged %s expired temporary memories", purged)
            for group_openid in self.store.groups():
                batch = self.store.pending_memory_batch(
                    group_openid,
                    min_messages=self.settings.memory_batch_messages,
                )
                if batch is None:
                    continue

                message_context = self._format_stored_messages(
                    batch.messages,
                    self.settings.max_context_chars,
                )
                pinned = self.store.pinned_memory_contents(group_openid)
                result = await self.llm.process_memory_batch(message_context, pinned)
                summary = str(result.get("summary") or "").strip()
                raw_memories = result.get("memories")
                memories = raw_memories if isinstance(raw_memories, list) else []
                if not summary:
                    continue

                self.store.save_memory_batch(
                    group_openid=group_openid,
                    start_row_id=batch.start_row_id,
                    end_row_id=batch.end_row_id,
                    summary=summary,
                    memories=[
                        item
                        for item in memories
                        if isinstance(item, dict) and str(item.get("content") or "").strip()
                    ],
                )
                logger.info(
                    "saved memory batch for group %s: messages=%s memories=%s",
                    group_openid,
                    len(batch.messages),
                    len(memories),
                )

    @staticmethod
    def _format_stored_messages(messages: list[StoredMessage], max_chars: int) -> str:
        lines: list[str] = []
        total = 0
        for message in messages:
            timestamp = (message.timestamp or "")[:19].replace("T", " ")
            line = f"{timestamp} {message.username}: {message.content}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def _memory_context(self, group_openid: str) -> str:
        summaries = self.store.conversation_summaries(group_openid, limit=5)
        # 注意：这里必须是 memory_cards_for_prompt，按“置顶+分类配额+最旧的优先”挑卡。
        # 早先用 memory_cards(limit=30) 是 id 倒序，卡片一多，最早那批（真名、纪念日、
        # 人设）就永远挤不进上下文，小毛豆会突然“不记得”最重要的事。
        cards = self.store.memory_cards_for_prompt(
            group_openid,
            now_iso=self._now().isoformat(timespec="seconds"),
        )
        parts = [f"[摘要] {summary}" for summary in summaries]
        parts.extend(ChatStore.format_memory_cards(cards).splitlines())
        return "\n".join(parts)[: self.settings.max_context_chars]

    # ---------- 没被 @ 时的主动关心 ----------

    def _pref_int(self, group_openid: str, key: str, default: int) -> int:
        raw = self.store.get_preference(group_openid, key)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except ValueError:
            logger.info("ignoring bad group preference %s=%r", key, raw)
            return default

    def _pref_bool(self, group_openid: str, key: str, default: bool) -> bool:
        raw = self.store.get_preference(group_openid, key)
        if raw is None:
            return default
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on", "开", "要", "是"}:
            return True
        if text in {"0", "false", "no", "off", "关", "不要", "否"}:
            return False
        return default

    def _note_rate_limit(self, group_openid: str, error: BaseException) -> bool:
        """撞上 QQ 的主动消息频控就当天收手，别一直撞。"""
        if RATE_LIMIT_CODE not in str(error):
            return False
        self._proactive_blocked_until[group_openid] = self._now().date().isoformat()
        logger.warning("proactive messages muted for %s today: %s", group_openid, error)
        return True

    def _care_gate(self, group_openid: str, member_openid: str, now: datetime) -> bool:
        """节流全过才值得去问模型：开关、频控、群总上限、每人次数、冷却。"""
        if self._proactive_blocked_until.get(group_openid) == now.date().isoformat():
            return False
        if not self._pref_bool(group_openid, "care_enabled", self.settings.care_enabled):
            return False
        group_max = self._pref_int(
            group_openid, "active_msg_daily_max", self.settings.active_msg_daily_max
        )
        person_max = self._pref_int(group_openid, "care_daily_max", self.settings.care_daily_max)
        if group_max <= 0 or person_max <= 0:
            return False
        today_iso = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(
            timespec="seconds"
        )
        if (
            self.store.proactive_message_count(group_openid=group_openid, since_iso=today_iso)
            >= group_max
        ):
            return False
        if (
            self.store.sent_care_count(
                group_openid=group_openid, member_openid=member_openid, since_iso=today_iso
            )
            >= person_max
        ):
            return False
        cooldown = self._pref_int(
            group_openid, "care_cooldown_minutes", self.settings.care_cooldown_minutes
        )
        last = self.store.last_care_at(group_openid=group_openid, member_openid=member_openid)
        if cooldown > 0 and last:
            try:
                last_at = datetime.fromisoformat(last)
            except ValueError:
                return True
            if now - last_at < timedelta(minutes=cooldown):
                return False
        return True

    async def _maybe_care(self, message: RichGroupMessage) -> None:
        """有人说身体不舒服或情绪低落时，主动接一句关心。

        两级判定：关键词预筛不花钱，命中才让便宜模型判断到底要不要说、说什么。
        任何一步拿不准都不开口——在这个群里，安静比说错话好。
        """
        content = normalize_face_tags(message.content).strip()
        if not should_ask_model(content):
            return
        group_openid = message.group_openid
        member_openid = message.author.member_openid or ""
        now = self._now()
        if not self._care_gate(group_openid, member_openid, now):
            logger.info("care throttled for %s", message.username)
            return

        message_key = ChatStore.message_key(message.id, message.msg_seq)
        if not self.store.reserve_care_message(
            message_key=message_key,
            group_openid=group_openid,
            member_openid=member_openid,
            content=content,
            created_at=now.isoformat(timespec="seconds"),
        ):
            return

        recent = self.store.recent_messages(group_openid, 6)
        try:
            verdict = await self.llm.judge_care(
                content=content,
                recent_context=self._format_stored_messages(recent, 600),
            )
            if not verdict.get("care"):
                return
            text = normalize_care_text(str(verdict.get("text") or ""))
            if not text:
                return
            await self.api.post_group_message(group_openid=group_openid, content=text)
        except Exception as exc:
            self._note_rate_limit(group_openid, exc)
            logger.info("proactive care skipped: %s", exc)
            return

        self.store.mark_care_sent(message_key)
        self._remember_bot_reply(group_openid, text, "BOT_CARE")
        logger.info(
            "proactive care for %s (%s): %s",
            message.username,
            ",".join(matched_signals(content)),
            text,
        )

    async def on_group_message_create(self, message: RichGroupMessage) -> None:
        if not self._group_allowed(message.group_openid):
            return
        self._save_message(message, "GROUP_MESSAGE_CREATE")
        logger.debug("saved full message from %s", message.username)

        if self._bot_mention(message):
            await self._process_at_message(message, "GROUP_MESSAGE_CREATE")
            return
        try:
            await self._maybe_care(message)
        except Exception:
            logger.exception("proactive care failed")

    async def on_group_at_message_create(self, message: RichGroupMessage) -> None:
        if not self._group_allowed(message.group_openid):
            return
        await self._process_at_message(message, "GROUP_AT_MESSAGE_CREATE")

    async def _process_at_message(self, message: RichGroupMessage, source: str) -> None:
        if not self._group_allowed(message.group_openid):
            return

        key = ChatStore.message_key(message.id, message.msg_seq)
        if not self.store.mark_at_processed(key):
            logger.info("duplicate @ event ignored: %s", key)
            return

        self._save_message(message, source)

        # 图片描述写回本地库，最近上下文、长期记忆、历史检索都能看到图片内容。
        content = normalize_face_tags(message.content).strip()
        descriptions: list[str] = []
        budget = self.settings.vision_max_images_per_request
        if budget > 0:
            attachments = image_attachments(message.raw)[:budget]
            if attachments:
                descriptions = await self._describe_attachments(attachments, hint=content)
                budget -= len(attachments)
                if any(description.strip() for description in descriptions):
                    enriched = message_with_image_descriptions(content, descriptions)
                    if enriched != content:
                        self.store.update_message_content(message.group_openid, key, enriched)
                    content = enriched
            await self._describe_recent_images(
                message.group_openid,
                exclude_key=key,
                budget=budget,
            )

        requester = StoredMessage(
            username=message.username,
            member_openid=message.author.member_openid or "",
            role=message.member_role,
            content=content,
            timestamp=message.timestamp or "",
            source=source,
        )
        topic, background, full_recent = self._topic_context(message.group_openid)
        identity = self._identity_context(message.group_openid, requester)

        async with self._reply_lock:
            acked = False
            try:
                bot_mention = self._bot_mention(message)
                raw_request = normalize_face_tags(message.content).strip()
                request = (
                    self._strip_bot_mention(raw_request, bot_mention)
                    if bot_mention
                    else raw_request
                )
                if descriptions:
                    request = message_with_image_descriptions(request, descriptions)
                request = request or "你好，请介绍一下你能帮我们做什么。"

                intent = await self.llm.parse_intent(
                    request=request,
                    current_time=self._now().isoformat(timespec="seconds"),
                    tasks_summary=ChatStore.format_tasks(self.store.tasks(message.group_openid)),
                    identity_context=identity.prompt(),
                    topic_context=topic,
                    festival_context=format_festivals(
                        upcoming_festivals(self._now(), days=30),
                        today=self._now().date(),
                    ),
                )
                acked = await self._answer_with_ack(
                    message=message,
                    intent=intent,
                    requester=requester,
                    request=request,
                    topic=topic,
                    background=background,
                    full_recent=full_recent,
                    identity=identity,
                    source=source,
                )
                logger.info("handled @ request %s from %s", intent.get("action"), message.username)
            except Exception:
                logger.exception("failed to generate or send reply")
                await self._send_reply(
                    message, FALLBACK_TEXT, 2 if acked else 1, f"{source}:FALLBACK"
                )

    def _topic_context(self, group_openid: str) -> tuple[str, str, str]:
        """切成【当前话题】+【更早的背景】，另外留一份全量记录给历史检索兜底。"""
        recent = self.store.recent_messages(group_openid, self.settings.recent_messages)
        topic_slice = split_topic(
            recent,
            boundary_row_id=self.store.latest_summary_row_id(group_openid),
        )
        return (
            format_topic(topic_slice.current, self.settings.max_context_chars),
            format_topic(topic_slice.previous_tail, 4000),
            format_topic(recent, self.settings.max_context_chars),
        )

    async def _send_reply(
        self,
        message: RichGroupMessage,
        text: str,
        seq: int,
        source: str,
    ) -> bool:
        """按 msg_seq 回复群消息。同一条 msg_id 下 seq 不能重复，所以 ack 用 1、正文用 2。"""
        try:
            await self.api.post_group_message(
                group_openid=message.group_openid,
                msg_id=message.id,
                msg_seq=seq,
                content=text,
            )
        except Exception:
            logger.exception("failed to send group reply, seq=%s", seq)
            return False
        self._remember_bot_reply(message.group_openid, text, source)
        return True

    async def _answer_with_ack(
        self,
        *,
        message: RichGroupMessage,
        intent: dict,
        requester: StoredMessage,
        request: str,
        topic: str,
        background: str,
        full_recent: str,
        identity: IdentityContext,
        source: str,
    ) -> bool:
        """执行 intent，返回是否已经发过 ack（决定正式回复用哪个 msg_seq）。"""
        action = str(intent.get("action") or "chat")
        task = asyncio.create_task(
            self._handle_intent(
                intent=intent,
                message=message,
                requester=requester,
                request=request,
                topic=topic,
                background=background,
                full_recent=full_recent,
                identity=identity,
            )
        )
        seq = 1
        acked = False
        timeout = self.settings.ack_after_seconds
        if action in ACK_ACTIONS and timeout > 0:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
            if not done:
                acked = await self._send_reply(message, ACK_TEXT, 1, f"{source}:ACK")
                if acked:
                    seq = 2
        try:
            answer = await task
        except Exception:
            logger.exception("failed to build reply for action %s", action)
            answer = FALLBACK_TEXT
        await self._send_reply(message, answer, seq, f"{source}:REPLY")
        return acked

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _int_field(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _handle_intent(
        self,
        *,
        intent: dict,
        message: RichGroupMessage,
        requester: StoredMessage,
        request: str,
        topic: str,
        background: str,
        full_recent: str,
        identity: IdentityContext,
    ) -> str:
        action = str(intent.get("action") or "chat")

        if action == "task_create":
            return self._handle_task_create(intent, message)
        if action in {"task_list", "task_cancel", "task_pause", "task_resume", "task_skip"}:
            return self._handle_task_action(intent, message)
        if action == "memory_remember":
            return self._handle_memory_remember(intent, message, identity)
        if action == "memory_list":
            return self._handle_memory_list(intent, message)
        if action in {"memory_pin", "memory_unpin"}:
            return self._handle_memory_pin(intent, message, pinned=action == "memory_pin")
        if action == "memory_forget":
            return self._handle_memory_forget(intent, message)
        if action == "calendar_query":
            return self._handle_calendar_query(intent)
        if action == "bot_pref":
            return self._handle_bot_pref(intent, message)
        if action == "memory_search":
            return await self._handle_memory_search(
                intent=intent,
                message=message,
                requester=requester,
                request=request,
                recent=full_recent,
                identity=identity,
            )
        if action == "web_search":
            return await self._handle_web_search(request, intent)
        if action == "web_fetch":
            return await self._handle_web_fetch(request, intent)
        if action == "document_read":
            return await self._handle_document_read(request, intent, message)
        if action == "shell_run":
            return await self._handle_shell_run(request, intent)
        if action == "math_eval":
            return await self._handle_math_eval(request, intent)

        return await self.llm.generate_reply(
            requester=requester,
            request=request,
            topic_context=topic,
            background_context=background,
            memory_context=self._memory_context(message.group_openid),
            identity_context=identity.prompt(),
        )

    def _handle_task_create(self, intent: dict, message: RichGroupMessage) -> str:
        schedule_type = str(intent.get("schedule_type") or "").strip()
        if schedule_type not in TASK_SCHEDULE_TYPES:
            return "这任务我听懂了，但暂时只支持单次、每天、每周、每年、农历每年这几种。"

        content = self._optional_str(intent.get("content"))
        prompt = self._optional_str(intent.get("prompt"))
        if not content and not prompt:
            return "要发什么内容我还没听明白，再说一次具体提醒或建议吧。"

        title = self._optional_str(intent.get("title")) or (content or prompt or "定时任务")[:24]
        now = self._now()
        time_of_day = self._optional_str(intent.get("time"))
        weekday = self._int_field(intent.get("weekday"))
        month = self._int_field(intent.get("month"))
        day = self._int_field(intent.get("day"))
        leap_month = bool(intent.get("leap_month"))
        remind_days_ahead = max(0, self._int_field(intent.get("remind_days_ahead")) or 0)
        try:
            next_run = resolve_next_run(
                schedule_type=schedule_type,
                now=now,
                timezone=self.settings.timezone,
                once_datetime=self._optional_str(intent.get("datetime")),
                time_of_day=time_of_day,
                weekday=weekday,
                month=month,
                day=day,
                leap_month=leap_month,
            )
        except ValueError as exc:
            return f"任务时间有点问题：{exc}"

        if next_run <= now:
            return "这个时间已经过去了，要不要改成之后的时间？"

        if self.store.active_task_count(message.group_openid) >= self.settings.task_max_count:
            return f"进行中的任务最多 {self.settings.task_max_count} 个，先取消一个再加吧。"

        task = self.store.create_task(
            group_openid=message.group_openid,
            title=title,
            content=content,
            generation_prompt=prompt,
            schedule_type=schedule_type,
            next_run_at=next_run.isoformat(timespec="seconds"),
            timezone=self.settings.timezone,
            time_of_day=time_of_day,
            weekday=weekday,
            created_by_openid=message.author.member_openid or "",
            created_by_name=message.username,
            month=month,
            day=day,
            leap_month=leap_month,
            remind_days_ahead=remind_days_ahead,
        )
        reply = f"已设置任务 {task.id}：{self._schedule_label(task)}，{task.title}。"
        warm = self._create_warmup_task(
            task=task,
            schedule_type=schedule_type,
            time_of_day=time_of_day,
            weekday=weekday,
            month=month,
            day=day,
            leap_month=leap_month,
            remind_days_ahead=remind_days_ahead,
            message=message,
            now=now,
        )
        if warm is not None:
            reply += f"\n还建了任务 {warm.id}：{self._schedule_label(warm)}提醒提前准备。"
        return reply

    def _create_warmup_task(
        self,
        *,
        task: ScheduledTask,
        schedule_type: str,
        time_of_day: str | None,
        weekday: int | None,
        month: int | None,
        day: int | None,
        leap_month: bool,
        remind_days_ahead: int,
        message: RichGroupMessage,
        now: datetime,
    ) -> ScheduledTask | None:
        """生日/纪念日提前 N 天先提个醒，当天那条留给主任务。"""
        if remind_days_ahead <= 0 or schedule_type == "once":
            return None
        if self.store.active_task_count(message.group_openid) + 1 > self.settings.task_max_count:
            logger.info("skipped warm-up task, task limit reached")
            return None
        try:
            warm_run = resolve_next_run(
                schedule_type=schedule_type,
                now=now,
                timezone=self.settings.timezone,
                time_of_day=time_of_day,
                weekday=weekday,
                month=month,
                day=day,
                leap_month=leap_month,
                offset_days=remind_days_ahead,
            )
        except ValueError:
            logger.exception("failed to resolve warm-up task for %s", task.id)
            return None
        main_day = parse_datetime(task.next_run_at, task.timezone)
        content = (
            f"{task.title} 还有 {remind_days_ahead} 天，"
            f"就是 {main_day.month}月{main_day.day}日，记得提前准备一下。"
        )
        return self.store.create_task(
            group_openid=message.group_openid,
            title=f"[预热] {task.title}",
            content=content,
            generation_prompt=None,
            schedule_type=schedule_type,
            next_run_at=warm_run.isoformat(timespec="seconds"),
            timezone=self.settings.timezone,
            time_of_day=time_of_day,
            weekday=weekday,
            created_by_openid=message.author.member_openid or "",
            created_by_name=message.username,
            month=month,
            day=day,
            leap_month=leap_month,
            offset_days=remind_days_ahead,
            linked_task_id=task.id,
        )

    def _handle_task_action(self, intent: dict, message: RichGroupMessage) -> str:
        action = str(intent.get("action") or "")
        if action == "task_list":
            return ChatStore.format_tasks(self.store.tasks(message.group_openid))

        task_id = self._int_field(intent.get("task_id"))
        match = self._optional_str(intent.get("match"))
        tasks = self.store.find_task(
            message.group_openid,
            task_id=task_id,
            query=match,
        )
        if not tasks:
            return "没有找到这个任务。可以先说“查看任务”。"
        if len(tasks) > 1:
            return "匹配到多个任务，请再说一次编号：\n" + ChatStore.format_tasks(tasks)

        task = tasks[0]
        if action == "task_cancel":
            self.store.delete_task(message.group_openid, task.id)
            return f"已取消任务 {task.id}：{task.title}。"
        if action == "task_pause":
            self.store.set_task_status(message.group_openid, task.id, "paused")
            return f"已暂停任务 {task.id}：{task.title}。"
        if action == "task_skip":
            if task.schedule_type == "once":
                return "单次任务没有“下一次”，要取消就直接说取消。"
            self.store.set_task_skip_next(message.group_openid, task.id, True)
            return f"好的，任务 {task.id} 下一次会跳过。"
        if action == "task_resume":
            if task.schedule_type == "once":
                next_run = parse_datetime(task.next_run_at, task.timezone)
                if next_run <= self._now():
                    return "这个单次任务的时间已经过了，重新创建一个吧。"
            else:
                next_run = resolve_next_run(
                    schedule_type=task.schedule_type,
                    now=self._now(),
                    timezone=task.timezone,
                    time_of_day=task.time_of_day,
                    weekday=task.weekday,
                    month=task.month,
                    day=task.day,
                    leap_month=task.leap_month,
                    offset_days=task.offset_days,
                )
                self.store.set_task_next_run(
                    message.group_openid,
                    task.id,
                    next_run.isoformat(timespec="seconds"),
                )
            self.store.set_task_status(message.group_openid, task.id, "active")
            return f"已恢复任务 {task.id}：{task.title}。"

        return "这个任务操作我还没听明白。"

    @staticmethod
    def _schedule_label(task: ScheduledTask) -> str:
        return schedule_label(task)

    @staticmethod
    def _is_sensitive_memory(content: str) -> bool:
        keywords = ("密码", "身份证", "银行卡", "验证码", "门牌", "住址")
        return any(keyword in content for keyword in keywords)

    @staticmethod
    def _canonical_person(person: str | None, identity: IdentityContext) -> str:
        value = (person or "").strip()
        if value in {"我", "本人", "自己", "当前用户", "用户", "你"}:
            return identity.requester.username
        if value in {"她", "他", "对象", "另一半", "伴侣", "对方", "你的对象"}:
            return identity.partner.username if identity.partner else "两人"
        return value or "两人"

    def _handle_memory_remember(
        self,
        intent: dict,
        message: RichGroupMessage,
        identity: IdentityContext,
    ) -> str:
        raw_memories = intent.get("memories")
        if not isinstance(raw_memories, list) or not raw_memories:
            return "要记住什么信息，可以更具体地说一次吗？"

        # pin/temporary 既写在单条上，也允许整条指令一起说（“这条记住，别忘”）
        intent_pin = bool(intent.get("pin"))
        intent_temporary = bool(intent.get("temporary"))
        saved = 0
        skipped = 0
        pinned_saved = 0
        for item in raw_memories:
            if not isinstance(item, dict):
                skipped += 1
                continue
            content = self._optional_str(item.get("content"))
            if not content:
                skipped += 1
                continue
            if self._is_sensitive_memory(content):
                skipped += 1
                continue
            pin = intent_pin or bool(item.get("pin"))
            temporary = intent_temporary or bool(item.get("temporary"))
            scope = "temporary" if temporary else "permanent"
            expires_at = (
                (self._now() + timedelta(days=TEMPORARY_MEMORY_DAYS)).isoformat(timespec="seconds")
                if temporary
                else None
            )
            self.store.add_memory_card(
                group_openid=message.group_openid,
                kind=self._optional_str(item.get("kind")) or "fact",
                person=self._canonical_person(
                    self._optional_str(item.get("person")),
                    identity,
                ),
                content=content,
                source_message_id=message.id,
                pinned=pin,
                scope=scope,
                expires_at=expires_at,
            )
            saved += 1
            pinned_saved += 1 if pin else 0

        if not saved:
            return "这条信息比较敏感，我不保存。"
        tail = "（已置顶★）" if pinned_saved and pinned_saved == saved else ""
        if saved and not skipped:
            return f"记住了，共 {saved} 条。{tail}"
        return f"记住了 {saved} 条，另有 {skipped} 条涉及敏感信息，我没有保存。{tail}"

    def _handle_memory_list(self, intent: dict, message: RichGroupMessage) -> str:
        cards, current, pages, total = self.store.memory_cards_page(
            message.group_openid,
            now_iso=self._now().isoformat(timespec="seconds"),
            page=max(1, self._int_field(intent.get("page")) or 1),
        )
        if not total:
            return "我现在还没有长期记忆。可以直接说：记住她不吃香菜。"
        if not cards:
            return f"记忆一共 {total} 条，只有 {pages} 页，最后一页说“第 {pages} 页记忆”。"
        header = f"当前长期记忆（第{current}/{pages}页，共{total}条，★=永久）："
        if current < pages:
            header += f"\n想看更多就说“第 {current + 1} 页记忆”。"
        return header + "\n" + self._format_memory_cards(cards)

    def _handle_memory_pin(self, intent: dict, message: RichGroupMessage, *, pinned: bool) -> str:
        cards = self.store.find_memory_cards(
            message.group_openid,
            memory_id=self._int_field(intent.get("memory_id")),
            query=self._optional_str(intent.get("match")),
        )
        if not cards:
            return "没有找到这条记忆。可以说“查看记忆”先看看编号。"
        if len(cards) > 1:
            return "匹配到多条记忆，请说编号：\n" + self._format_memory_cards(cards)

        card = cards[0]
        if not self.store.set_memory_pinned(message.group_openid, card.id, pinned):
            return "这条记忆刚刚变动过，麻烦再说一次。"
        verb = "置顶，以后不会忘" if pinned else "取消置顶"
        return f"已{verb}记忆 {card.id}：{card.content}"

    def _handle_calendar_query(self, intent: dict) -> str:
        name = self._optional_str(intent.get("query"))
        now = self._now()
        if name:
            found = next_festival(name, now)
            if found is None:
                return f"我没查到“{name}”是哪天，换个常见节日名试试，比如中秋、除夕、清明。"
            day, festival = found
            suffix = ""
            lunar = solar_to_lunar(day)
            if lunar:
                suffix = f"，农历{lunar_label(lunar[0], lunar[1], leap=lunar[2])}"
            return (
                f"{festival}是 {day.year}年{day.month}月{day.day}日"
                f"（星期{'一二三四五六日'[day.weekday()]}）{suffix}，还有 {(day - now.date()).days} 天。"
            )
        items = upcoming_festivals(now, days=30)
        if not items:
            return "未来 30 天没有节日。"
        return "未来 30 天的节日：" + format_festivals(items, today=now.date())

    # ---------- 群内偏好：群里说一句话就能改，优先级高于 .env ----------

    def _default_pref(self, kind: str, attr: str) -> str:
        value = getattr(self.settings, attr, None)
        if kind == "bool":
            return "开" if value else "关"
        return str(value)

    def _format_preferences(self, group_openid: str) -> str:
        lines: list[str] = []
        for key, (label, kind, attr) in GROUP_PREFS.items():
            if kind == "bool":
                shown = (
                    "开"
                    if self._pref_bool(group_openid, key, bool(getattr(self.settings, attr)))
                    else "关"
                )
            else:
                shown = str(
                    self._pref_int(group_openid, key, int(getattr(self.settings, attr) or 0))
                )
            source = "群里改过" if self.store.get_preference(group_openid, key) is not None else "默认"
            lines.append(f"· {label}：{shown}（{source}）")
        tail = ""
        if self._proactive_blocked_until.get(group_openid) == self._now().date().isoformat():
            tail = "\n今天被 QQ 频控了，主动消息先自己停一天。"
        return "这个群现在的设置：\n" + "\n".join(lines) + tail

    def _handle_bot_pref(self, intent: dict, message: RichGroupMessage) -> str:
        group_openid = message.group_openid
        op = str(intent.get("op") or "show").strip().lower()
        key = str(intent.get("key") or "").strip()

        if op != "set" or not key:
            return self._format_preferences(group_openid)
        if key not in GROUP_PREFS:
            return "这个群只能改这几项：" + "、".join(GROUP_PREFS) + "。"

        label, kind, attr = GROUP_PREFS[key]
        # 注意 False 和 0 都是合法值，不能当成“没填”。
        raw_value = intent.get("value")
        if isinstance(raw_value, bool):
            raw = "true" if raw_value else "false"
        elif raw_value is None:
            raw = ""
        else:
            raw = str(raw_value).strip()
        lowered = raw.lower()
        if "default" in lowered or "默认" in raw or "恢复" in raw or "还原" in raw or "改回去" in raw:
            self.store.clear_preference(group_openid, key)
            return f"好，{label}恢复成默认值 {self._default_pref(kind, attr)}。"

        if kind == "int":
            try:
                value = int(raw)
            except ValueError:
                return f"{label}得是个数字，比如「{label}改成 3」。"
            if not 0 <= value <= 50:
                return f"{label}只能在 0 到 50 之间，0 就是不再主动发。"
            stored, shown = str(value), str(value)
        else:
            text = raw.lower()
            if text in {"1", "true", "yes", "on", "开", "要", "是"}:
                stored, shown = "true", "开"
            elif text in {"0", "false", "no", "off", "关", "不要", "否"}:
                stored, shown = "false", "关"
            else:
                return f"{label}要说开还是关，比如「以后别主动安慰我们」。"

        self.store.set_preference(group_openid, key, stored)
        if key == "care_enabled" and shown == "关":
            return "好，那我不主动插话了，你们 @ 我随时都在。想恢复就说「要主动关心」。"
        return f"记下了，这个群的{label}改为{shown}。"

    @staticmethod
    def _format_memory_cards(cards: list[MemoryCard]) -> str:
        return "\n".join(
            f"{card.id}. {'★' if card.pinned else ''}[{card.kind}] {card.person}: {card.content}"
            for card in cards
        )

    def _handle_memory_forget(self, intent: dict, message: RichGroupMessage) -> str:
        memory_id = self._int_field(intent.get("memory_id"))
        match = self._optional_str(intent.get("match"))
        cards = self.store.find_memory_cards(
            message.group_openid,
            memory_id=memory_id,
            query=match,
        )
        if not cards:
            return "没有找到这条记忆。可以说“查看记忆”。"
        if len(cards) > 1:
            return "匹配到多条记忆，请说编号：\n" + self._format_memory_cards(cards)

        card = cards[0]
        self.store.delete_memory_card(message.group_openid, card.id)
        return f"已忘记记忆 {card.id}：{card.content}"

    async def _handle_memory_search(
        self,
        *,
        intent: dict,
        message: RichGroupMessage,
        requester: StoredMessage,
        request: str,
        recent: str,
        identity: IdentityContext,
    ) -> str:
        query = self._optional_str(intent.get("query")) or request
        if not query:
            return "你想查什么？可以说“我们之前聊过哪家餐厅”。"

        results = self.store.search_messages(
            message.group_openid,
            query,
            self.settings.memory_search_messages,
        )
        cards = self.store.search_memory_cards(message.group_openid, query)
        if not results and not cards:
            return f"我没找到和“{query}”有关的记录。"

        search_parts = []
        if results:
            search_parts.append(
                "相关历史消息：\n" + ChatStore.format_search_results(results)
            )
        if cards:
            search_parts.append("相关长期记忆：\n" + self._format_memory_cards(cards))

        return await self.llm.generate_reply(
            requester=requester,
            request=request,
            topic_context=recent,
            memory_context=self._memory_context(message.group_openid),
            search_context="\n\n".join(search_parts),
            identity_context=identity.prompt(),
        )

    @staticmethod
    def _format_search_result(result: ToolResult) -> str:
        lines: list[str] = []
        if result.output:
            lines.append("摘要：" + result.output[:300])
        for index, item in enumerate(result.data, 1):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            site = urlparse(url).netloc if url else ""
            suffix = f"（来源：{site}）" if site else ""
            lines.append(f"{index}. {title}{suffix}\n   {url}\n   {snippet}")
        return "\n".join(lines)

    async def _handle_web_search(self, request: str, intent: dict) -> str:
        query = self._optional_str(intent.get("query")) or request
        logger.info("web_search query=%r provider=%s", query, self.settings.search_provider)
        result = await search_web(
            query,
            limit=self.settings.search_max_results,
            provider=self.settings.search_provider,
            tavily_api_key=self.settings.tavily_api_key,
            bocha_api_key=self.settings.bocha_api_key,
            timeout=self.settings.search_timeout_seconds,
        )
        if not result.ok:
            logger.warning("web search failed: %s", result.error)
            if "没有配置" in (result.error or ""):
                return "我还没配好联网搜索，需要先在服务器上填 BOCHA_API_KEY，配好后就能查了。"
            detail = (result.error or "未知错误").replace("联网搜索没有配置：", "")[:80]
            return f"搜索服务这次没返回可用结果：{detail}，要不换个说法再让我查一次？"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="web_search",
            tool_output=self._format_search_result(result),
        )

    async def _handle_web_fetch(self, request: str, intent: dict) -> str:
        url = self._optional_str(intent.get("url"))
        if not url:
            return await self._handle_web_search(request, intent)
        result = await fetch_page(url, max_chars=self.settings.web_fetch_max_chars)
        if not result.ok:
            return result.error or "网页读取失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="web_fetch",
            tool_output=result.output,
        )

    async def _handle_document_read(
        self,
        request: str,
        intent: dict,
        message: RichGroupMessage,
    ) -> str:
        url = self._optional_str(intent.get("url"))
        filename = ""
        if not url:
            attachments = document_attachments(message.raw)
            if attachments:
                url = attachments[0].url
                filename = attachments[0].filename
        if not url:
            return "我还没看到要读取的文档，请把文件发到群里，或者贴一个文档链接。"

        result = await read_document(
            url,
            filename=filename,
            max_chars=self.settings.document_max_chars,
        )
        if not result.ok:
            return result.error or "文档读取失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="document_read",
            tool_output=result.output,
        )

    async def _handle_shell_run(self, request: str, intent: dict) -> str:
        if not self.settings.shell_enabled:
            return "本地命令执行目前是关闭的。"
        command = self._optional_str(intent.get("command"))
        if not command:
            return "要执行什么命令，再说具体一点？"
        result = await run_shell(
            command,
            cwd=self.settings.shell_cwd,
            mode=self.settings.shell_mode,
            timeout=self.settings.shell_timeout_seconds,
            max_output_chars=self.settings.shell_max_output_chars,
        )
        if not result.ok and not result.output:
            return result.error or "命令执行失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="shell_run",
            tool_output=result.output,
            tool_error=result.error,
        )

    async def _handle_math_eval(self, request: str, intent: dict) -> str:
        expression = self._optional_str(intent.get("expression")) or request
        result = eval_math(expression)
        if not result.ok:
            return result.error or "计算失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="math_eval",
            tool_output=result.output,
        )

    async def on_group_msg_receive(self, event) -> None:
        logger.info("group full-message receiving enabled: %s", event.group_openid)

    async def on_group_msg_reject(self, event) -> None:
        logger.warning("group full-message receiving disabled: %s", event.group_openid)

    async def on_group_add_robot(self, event) -> None:
        logger.info("Xiaomaodou was added to group: %s", event.group_openid)

    async def on_group_del_robot(self, event) -> None:
        logger.warning("Xiaomaodou was removed from group: %s", event.group_openid)
