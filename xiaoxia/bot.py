from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import botpy
from botpy.message import GroupMessage
from botpy.robot import Token

from .config import Settings
from .llm import XiaoxiaLLM
from .scheduling import (
    advance_task,
    in_quiet_hours,
    parse_datetime,
    parse_quiet_time,
    resolve_next_run,
)
from .store import ChatStore, MemoryCard, ScheduledTask, StoredMessage


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
            "小毛豆已上线：%s | model=%s | stored_messages=%s | memory_cards=%s | scheduled_tasks=%s",
            getattr(self.robot, "name", "小毛豆"),
            self.settings.model,
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
        quiet_start = parse_quiet_time(self.settings.task_quiet_start)
        quiet_end = parse_quiet_time(self.settings.task_quiet_end)
        if in_quiet_hours(now, quiet_start, quiet_end):
            return

        now_iso = now.isoformat(timespec="seconds")
        for task in self.store.due_tasks(now_iso):
            next_run = advance_task(
                schedule_type=task.schedule_type,
                from_run_at=task.next_run_at,
                timezone=task.timezone,
                time_of_day=task.time_of_day,
                weekday=task.weekday,
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
                self.store.complete_task_run(
                    group_openid=task.group_openid,
                    task_id=task.id,
                    success=True,
                    now_iso=now_iso,
                    next_run_at=next_run_iso,
                    finish=task.schedule_type == "once",
                )
                logger.info("sent scheduled task %s", task.id)
            except Exception:
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
                result = await self.llm.process_memory_batch(message_context)
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
        cards = self.store.memory_cards(group_openid, limit=30)
        parts = [f"[摘要] {summary}" for summary in summaries]
        parts.extend(ChatStore.format_memory_cards(cards).splitlines())
        return "\n".join(parts)[: self.settings.max_context_chars]

    async def on_group_message_create(self, message: RichGroupMessage) -> None:
        if not self._group_allowed(message.group_openid):
            return
        self._save_message(message, "GROUP_MESSAGE_CREATE")
        logger.debug("saved full message from %s", message.username)

        if self._bot_mention(message):
            await self._process_at_message(message, "GROUP_MESSAGE_CREATE")

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
        requester = StoredMessage(
            username=message.username,
            member_openid=message.author.member_openid or "",
            role=message.member_role,
            content=message.content or "",
            timestamp=message.timestamp or "",
            source=source,
        )
        recent = self.store.format_recent_messages(
            message.group_openid,
            self.settings.recent_messages,
            self.settings.max_context_chars,
        )
        identity = self._identity_context(message.group_openid, requester)

        async with self._reply_lock:
            try:
                bot_mention = self._bot_mention(message)
                raw_request = (message.content or "").strip()
                request = (
                    self._strip_bot_mention(raw_request, bot_mention)
                    if bot_mention
                    else raw_request
                ) or "你好，请介绍一下你能帮我们做什么。"

                intent = await self.llm.parse_intent(
                    request=request,
                    current_time=self._now().isoformat(timespec="seconds"),
                    tasks_summary=ChatStore.format_tasks(self.store.tasks(message.group_openid)),
                    identity_context=identity.prompt(),
                )
                answer = await self._handle_intent(
                    intent=intent,
                    message=message,
                    requester=requester,
                    request=request,
                    recent=recent,
                    identity=identity,
                )
                await message.reply(content=answer)
                logger.info("handled @ request %s from %s", intent.get("action"), message.username)
            except Exception:
                logger.exception("failed to generate or send reply")
                await message.reply(content="小毛豆刚刚走神了，稍后再 @ 我一次试试。")

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
        recent: str,
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
            return self._handle_memory_list(message)
        if action == "memory_forget":
            return self._handle_memory_forget(intent, message)
        if action == "memory_search":
            return await self._handle_memory_search(
                intent=intent,
                message=message,
                requester=requester,
                request=request,
                recent=recent,
                identity=identity,
            )

        return await self.llm.generate_reply(
            requester=requester,
            request=request,
            recent_context=recent,
            memory_context=self._memory_context(message.group_openid),
            identity_context=identity.prompt(),
        )

    def _handle_task_create(self, intent: dict, message: RichGroupMessage) -> str:
        schedule_type = str(intent.get("schedule_type") or "").strip()
        if schedule_type not in {"once", "daily", "weekly"}:
            return "这任务我听懂了，但暂时只支持单次、每天和每周。"

        content = self._optional_str(intent.get("content"))
        prompt = self._optional_str(intent.get("prompt"))
        if not content and not prompt:
            return "要发什么内容我还没听明白，再说一次具体提醒或建议吧。"

        title = self._optional_str(intent.get("title")) or (content or prompt or "定时任务")[:24]
        now = self._now()
        try:
            next_run = resolve_next_run(
                schedule_type=schedule_type,
                now=now,
                timezone=self.settings.timezone,
                once_datetime=self._optional_str(intent.get("datetime")),
                time_of_day=self._optional_str(intent.get("time")),
                weekday=self._int_field(intent.get("weekday")),
            )
        except ValueError as exc:
            return f"任务时间有点问题：{exc}"

        if next_run <= now:
            return "这个时间已经过去了，要不要改成之后的时间？"
        if in_quiet_hours(
            next_run,
            parse_quiet_time(self.settings.task_quiet_start),
            parse_quiet_time(self.settings.task_quiet_end),
        ):
            return (
                "这个时间在免打扰时段里。"
                f"我默认 {self.settings.task_quiet_start} 到 {self.settings.task_quiet_end} 不主动发消息。"
            )
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
            time_of_day=self._optional_str(intent.get("time")),
            weekday=self._int_field(intent.get("weekday")),
            created_by_openid=message.author.member_openid or "",
            created_by_name=message.username,
        )
        return f"已设置任务 {task.id}：{self._schedule_label(task)}，{task.title}。"

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
        if task.schedule_type == "once":
            return f"单次 {task.next_run_at[:16].replace('T', ' ')}"
        if task.schedule_type == "daily":
            return f"每天 {task.time_of_day or ''}"
        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weekday_names[task.weekday - 1] if task.weekday and 1 <= task.weekday <= 7 else "每周"
        return f"{weekday} {task.time_of_day or ''}"

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

        saved = 0
        skipped = 0
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
            self.store.add_memory_card(
                group_openid=message.group_openid,
                kind=self._optional_str(item.get("kind")) or "fact",
                person=self._canonical_person(
                    self._optional_str(item.get("person")),
                    identity,
                ),
                content=content,
                source_message_id=message.id,
            )
            saved += 1

        if saved and not skipped:
            return f"记住了，共 {saved} 条。"
        if saved:
            return f"记住了 {saved} 条，另有 {skipped} 条涉及敏感信息，我没有保存。"
        return "这条信息比较敏感，我不保存。"

    def _handle_memory_list(self, message: RichGroupMessage) -> str:
        cards = self.store.memory_cards(message.group_openid, limit=30)
        if not cards:
            return "我现在还没有长期记忆。可以直接说：记住她不吃香菜。"
        return "当前长期记忆：\n" + self._format_memory_cards(cards)

    @staticmethod
    def _format_memory_cards(cards: list[MemoryCard]) -> str:
        return "\n".join(
            f"{card.id}. [{card.kind}] {card.person}: {card.content}"
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
            recent_context=recent,
            memory_context=self._memory_context(message.group_openid),
            search_context="\n\n".join(search_parts),
            identity_context=identity.prompt(),
        )

    async def on_group_msg_receive(self, event) -> None:
        logger.info("group full-message receiving enabled: %s", event.group_openid)

    async def on_group_msg_reject(self, event) -> None:
        logger.warning("group full-message receiving disabled: %s", event.group_openid)

    async def on_group_add_robot(self, event) -> None:
        logger.info("Xiaomaodou was added to group: %s", event.group_openid)

    async def on_group_del_robot(self, event) -> None:
        logger.warning("Xiaomaodou was removed from group: %s", event.group_openid)
