from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import botpy
from botpy.http import Route
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
from .llm import ACTION_REPLY_ADDENDUM, STRUCTURED_DATA_REPLY_ADDENDUM, MaodouLLM
from .replies import split_reply_text
from .scheduling import advance_task, parse_datetime, resolve_next_run, schedule_label
from .store import (
    ChatStore,
    MemberMessageCount,
    MemoryCard,
    MessageStats,
    ScheduledTask,
    StoredMessage,
    clip_summary,
)
from .stickers import Sticker, StickerLibrary, face_prompt, render_faces
from .topics import format_topic, split_topic
from .agent import AgentContext, agent_run_loop
from .tools import (
    ToolResult,
    document_attachments,
    eval_math,
    fetch_page,
    read_document,
    run_shell,
    search_web,
)
from .voice import (
    VoiceAttachment,
    VoiceReply,
    VoiceSynthesizer,
    VoiceTranscriber,
    message_with_unrecognized_voice,
    message_with_voice_transcripts,
    needs_voice_transcript,
    voice_attachments,
)


logger = logging.getLogger("maodou")


class RichGroupMessage(GroupMessage):
    __slots__ = ("username", "member_role", "raw")

    def __init__(self, api, event_id: str | None, data: dict[str, Any]) -> None:
        super().__init__(api, event_id, data)
        author = data.get("author") or {}
        self.username = author.get("username") or author.get("member_openid") or "unknown"
        self.member_role = author.get("member_role") or "member"
        self.raw = data


class ActionResult(str):
    """本地操作的模板回复，顺带记两件事：能不能交给模型润色、哪些事实必须原样保留。

    继承 str 的全部意义在于：所有按字符串断言的调用方（包括测试）一点感觉都没有，
    没标 persona 的地方行为和以前完全一样。
    """

    def __new__(
        cls,
        text: str,
        *,
        persona: bool = False,
        facts: Iterable[str] = (),
    ) -> "ActionResult":
        self = super().__new__(cls, text)
        self.persona = persona
        self.facts = tuple(str(fact).strip() for fact in facts if str(fact).strip())
        return self


ROLE_LABELS = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}

# 这几类动作要调外部服务，通常要几秒到十几秒，先应一句比让人干等好。
# chat 也在里面：它是占比最高的一类，带图或模型排队时同样会卡十几秒，
# 而 ack 只在超过 ack_after_seconds 才发，快的时候一条消息都不会多。
ACK_ACTIONS = frozenset(
    {"chat", "agent_run"}
)
TASK_SCHEDULE_TYPES = frozenset({"once", "daily", "weekly", "yearly", "lunar_yearly"})

# 工具回复限长：计算类只给答案不该长篇，命令输出给个中等上限；检索/网页/文档走默认回复上限。
TOOL_REPLY_MAX_CHARS: dict[str, int] = {
    "math_eval": 120,
    "shell_run": 600,
}

# 这几个本地操作以前直接回模板字符串，SYSTEM_PROMPT 里那套人设一个 token 都没参与，
# 于是「以后叫我主人」换来一句冷冰冰的「好，以后就叫CL「主人」。」——死板的根源在这。
# 现在把模板当「事实凭据」交给模型用人设说一遍。只有成功路径这么做：报错和
# 「匹配到多条，请说编号」这类仍然原样返回，那些话必须精确可预测，
# 模型一活泼就没人知道该报哪个编号了。
PERSONA_TOOL_LABELS: dict[str, str] = {
    "memory_remember": "记住信息",
    "memory_forget": "删除记忆",
    "address_set": "登记称呼",
    "task_create": "创建定时提醒",
    "task_cancel": "取消定时提醒",
    "task_pause": "暂停定时提醒",
    "task_resume": "恢复定时提醒",
    "task_skip": "让定时提醒跳过下一次",
    "memory_alias": "统一某个人的叫法",
    "bot_pref": "改这个群的设置",
}
# 就是一句确认话，给太长反而啰嗦。
PERSONA_REPLY_MAX_CHARS = 60

# 「记住了 2 条」被模型说成「这两条我都记牢啦」是最地道的中文。只按阿拉伯数字比对，
# 会把这种说法判成「丢了事实」而退回模板，那这个功能在最常用的场景下等于白做。
# 所以纯数字的事实同时接受中文写法。方向是放宽而不是收紧：万一「三」出现在
# 别的地方（周三），也只是多放过一次，不会把对的回复打成错的。
_CN_DIGIT_VARIANTS: dict[str, tuple[str, ...]] = {
    "1": ("一",),
    "2": ("二", "两"),
    "3": ("三",),
    "4": ("四",),
    "5": ("五",),
    "6": ("六",),
    "7": ("七",),
    "8": ("八",),
    "9": ("九",),
    "10": ("十",),
}


# 任务和记忆的方向词同理：模板之间只差两个字，模型说反了用户会以为提醒还在跑，
# 所以方向必须锁。但「撤掉」「删了」这种同义说法是最自然的中文，放行；
# 反方向词（把「取消」说成「恢复」）永远不在同义表里，照样拦下来。
_DIRECTION_VARIANTS: dict[str, tuple[str, ...]] = {
    "取消": ("撤掉", "撤了", "删掉", "去掉"),
    "暂停": ("先停", "停一下", "按住"),
    "恢复": ("重新开", "接着提醒", "继续提醒"),
    "跳过": ("不发", "略过"),
    "忘记": ("删掉", "删了", "清掉", "去掉"),
}


# 群设置的中文名是给「看看群设置」那张表用的，写全称才准确；可确认话里模型一定
# 会把它说短，实测两个长名字全被判成丢事实、白白退回模板。这里只放宽这几个长名字
# 的比对。放宽的代价是「万一别名出现在别处」，那种情况只是多放过一次；
# 收紧的代价是说对了的回复被打成错的，用户直接看到日志口气——两害相权取其轻。
_FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "每人每天被关心次数": ("每人每天关心次数", "每天关心次数", "关心次数"),
    "同一个人两次关心的间隔（分钟）": ("关心间隔", "两次关心的间隔", "间隔分钟数"),
    "群里每天主动消息上限": ("主动消息上限", "每天主动消息", "主动消息条数"),
}


def _squash(text: str) -> str:
    """中文里空格不承载语义，「30 分钟」和「30分钟」必须算同一个事实。"""
    return "".join(text.split())


def _fact_variants(fact: str) -> tuple[str, ...]:
    return (
        fact,
        *_CN_DIGIT_VARIANTS.get(fact, ()),
        *_DIRECTION_VARIANTS.get(fact, ()),
        *_FACT_ALIASES.get(fact, ()),
    )


def _missing_facts(facts: Iterable[str], reply: str) -> list[str]:
    haystack = _squash(reply)
    return [
        fact
        for fact in facts
        if not any(_squash(variant) in haystack for variant in _fact_variants(fact))
    ]


# 真实模型跑出来的翻车样本：模板是「已忘记记忆 2：不吃香菜」，模型回
# 「第一条记忆没删成，实际删掉的是第 2 条」。子串校验只看事实词在不在，
# 「删掉」「2」都在，于是这句话顺利通过——可它把办成的事说成了没办成。
# 语义反转比死板严重得多，所以再兜一道。只列真出现过的、语义明确的组合，
# 不列单独的「没」：「一条都没漏」是好话，不该被误杀。
_FAILURE_PHRASES = (
    "没办成", "办不成", "没做成", "做不成", "没删成", "没存上", "没记住",
    "没成功", "没设成", "没建成", "没标成", "没定上", "没改过来", "失败了",
)


def _contradicts_success(reply: str) -> bool:
    """确认话里出现「其实没办成」或者反过来追问用户，都算跟执行结果打架。"""
    if any(phrase in reply for phrase in _FAILURE_PHRASES):
        return True
    # 事情已经办完了，这时候反问「是不是这条」只会让人以为没办成
    return reply.endswith(("？", "?"))


def _persona_max_chars(template: str) -> int:
    """确认话默认 60 字够了；但带预热任务的模板本身就是两行，按 60 截断必然丢事实、
    必然退回模板，那这次模型调用等于白花。模板有多长就给多长，再留 30 字换语气。"""
    return max(PERSONA_REPLY_MAX_CHARS, len(template) + 30)

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
# 同一条 msg_id 下 msg_seq 重复时报这个码，意思是「这个槽已经被占了」。
DEDUP_CODE = "40054005"
ACK_TEXT = "稍等，我查一下～"
ACK_TEXT_CHAT = "我想想～"
ACK_TEXT_IMAGE = "我看看图～"
ACK_TEXT_IMAGES = "我一张张看～"
ACK_TEXT_VOICE = "我听听～"
ACK_TEXT_VOICE_IMAGES = "我看看图、听听语音～"
FALLBACK_TEXT = "小毛豆刚刚走神了，稍后再 @ 我一次试试。"

# 只 @ 一下不说话，该怎么接？以前的答案是代码替它决定的：把空消息硬填成
# 「你好，请介绍一下你能帮我们做什么。」，于是一天里连着四次空 @，每次都背回一份
# 几乎一模一样的两百字功能清单——用户看到的不是机器人，是一段循环播放的说明书。
# 现在只如实告诉模型「对方没带话」，接什么由它照【当前话题】自己定：
# 绝大多数时候一句「在呢，怎么啦」就够了。行为规则写在 SYSTEM_PROMPT 第 4 条。
EMPTY_AT_REQUEST = "（只 @ 了我，没带话）"
# 被动回复的重试次数与间隔。ack 不走这套（它只是垫场，发不出去就算了）。
SEND_ATTEMPTS = 3
SEND_RETRY_DELAY_SECONDS = 0.4


@dataclass(frozen=True)
class IdentityContext:
    requester: StoredMessage
    partner: StoredMessage | None
    # 显示名 -> 群里登记过的称呼。没登记过就是空 dict，照旧用显示名。
    addresses: dict[str, str] = field(default_factory=dict)
    # openid / 显示名 -> 记录里【】里挂的那个名字，跟 format_topic 用的是同一张表。
    speaker_labels: dict[str, str] = field(default_factory=dict)

    def _address(self, member: StoredMessage) -> str:
        return self.addresses.get((member.username or "").strip(), "")

    def _speaker(self, member: StoredMessage | None) -> str:
        """这条人在聊天记录里挂的名字（登记过称呼就是称呼，否则显示名）。"""
        if member is None:
            return "暂时未识别"
        name = (member.username or "").strip()
        openid = (member.member_openid or "").strip()
        return self.speaker_labels.get(openid) or self.speaker_labels.get(name) or name or "成员"

    def _label(self, member: StoredMessage) -> str:
        role = ROLE_LABELS.get(member.role, member.role or "成员")
        address = self._address(member)
        name = f"{member.username}，请叫「{address}」" if address else str(member.username)
        return f"{name}（{role}）"

    def prompt(self) -> str:
        partner = self._label(self.partner) if self.partner else "暂时未识别"
        lines = [
            f"当前触发者：{self._label(self.requester)}",
            f"另一位聊天成员：{partner}",
        ]
        # 称呼写死在身份块里，模型才不会一会儿叫显示名、一会儿自己编昵称。
        rules = "；".join(
            f"{name} → 「{address}」" for name, address in self.addresses.items() if address
        )
        if rules:
            lines.append(f"称呼要求（他们自己定的，每一轮都要照做）：{rules}。")
        # 记录里每条消息挂的是【称呼】，这里把标签对准身份：谁是「你」、谁是「另一位」。
        legend = f"【{self._speaker(self.requester)}】= 当前触发者（就是「你」）"
        if self.partner is not None:
            legend += f"；【{self._speaker(self.partner)}】= 另一位聊天成员"
        lines.append(
            f"聊天记录里的名字：{legend}；【小毛豆】= 你自己说的话。"
        )
        lines.append(
            "身份要求：当前触发者用“你”指代，另一位用他的称呼或“你对象”指代。"
            "最近记录中的“我”以每条消息开头的【】里那个人为准，不要把两人的经历或偏好混在一起。"
        )
        return "\n".join(lines)


@dataclass
class _ActiveWindow:
    """一次 @ 触发的群级“在场”状态，只保存在进程内。"""

    until: float
    last_message_at: float
    followup_done: bool = False


class MaodouClient(botpy.Client):
    def __init__(
        self,
        *,
        settings: Settings,
        store: ChatStore,
        llm: MaodouLLM,
    ) -> None:
        super().__init__(
            intents=botpy.Intents(public_messages=True),
            bot_log=False,
        )
        self.settings = settings
        self.store = store
        self.llm = llm
        # 语音是可插拔能力：QQ 自带 ASR 优先，TTS/ASR 没配好时分别退回
        # “只看文字”和“不调用外部服务”，不能让缺失配置拦住启动。
        self.voice_synthesizer = VoiceSynthesizer(settings)
        self.voice_transcriber = VoiceTranscriber(settings)
        # 表情包：清单在 assets/stickers/manifest.json。读不到就当没有这个能力，
        # 静默降级成纯文字，不能让一个坏掉的素材目录把机器人拦在启动之外。
        sticker_dir = str(getattr(settings, "sticker_dir", "") or "")
        self.stickers = (
            StickerLibrary.load(sticker_dir)
            if sticker_dir and getattr(settings, "stickers_enabled", False)
            else StickerLibrary(Path("."), {})
        )
        if self.stickers:
            logger.info("loaded %s stickers from %s", len(self.stickers), self.stickers.root)
        self._reply_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        # 后台补识别旧图片：正在跑的群 + 任务引用（防止被 GC 掉）
        self._image_backfill_running: set[str] = set()
        self._image_backfill_tasks: set[asyncio.Task] = set()
        # 群 openid -> 被 QQ 频控关静音到哪天（只当天有效，重启即复位）
        self._proactive_blocked_until: dict[str, str] = {}
        # 群 openid -> @ 后 10 分钟的活跃窗口。消息本身仍在 SQLite，这里只存临时在场状态。
        self._active_windows: dict[str, _ActiveWindow] = {}
        self._active_locks: dict[str, asyncio.Lock] = {}
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
            if not image_ref:
                # 图根本没下载下来。QQ 的 rkey 链接过一会儿就失效，把原始 url 丢给
                # 视觉模型它同样打不开，只会白烧一次调用再拿回一个空描述。
                # 直接留空，调用方会把这条消息标成 [图片未识别]，下次也不再重试。
                logger.warning(
                    "image download failed, skipping vision call: %s",
                    attachment.filename or attachment.content_type,
                )
                descriptions.append("")
                continue
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
        """自己取图并内联成 data url，交给视觉模型。

        返回 (给视觉模型的引用, 字节的 sha256, 字节数, content_type)。
        下载失败时四个值里前三个都是空/0，调用方据此跳过视觉模型。
        """
        fetched = await fetch_image_bytes(
            attachment.url,
            max_bytes=self.settings.vision_max_image_bytes,
        )
        if fetched is None:
            logger.info("image download failed: %s", attachment.filename)
            return "", "", 0, attachment.content_type
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

    def _pending_image_batch(
        self,
        group_openid: str,
        *,
        exclude_key: str,
        budget: int,
    ) -> list[tuple[StoredMessage, list[ImageAttachment]]]:
        """从最新消息往前收集「连续的一批待识别图片」。

        图片消息本身不会触发视觉模型，只有 @ 事件才会。用户一次连发好几张图再 @ 时，
        如果只同步最新一张，第一轮回答仍然会把前面几张当成没看到。这里按消息从新到旧
        扫，碰到普通文字就停，最多收满本轮额度；机器人自己的消息跳过，不打断图片批次。
        """
        if budget <= 0:
            return []
        selected: list[tuple[StoredMessage, list[ImageAttachment]]] = []
        remaining = budget
        for stored in reversed(
            self.store.recent_messages_with_raw(
                group_openid,
                self.settings.vision_recent_messages,
            )
        ):
            if stored.message_key == exclude_key or stored.role == "bot":
                continue
            attachments = image_attachments(stored.raw)
            if attachments and needs_image_description(stored.content, stored.raw):
                taken = attachments[:remaining]
                if taken:
                    selected.append((stored, taken))
                    remaining -= len(taken)
                if remaining <= 0:
                    break
                continue
            # 一张普通文字（或已经看过、没有待处理图片的消息）就是这一批的边界，
            # 不能因为它而继续往前翻，把很久以前的旧图也拖进来重看。
            break
        selected.reverse()
        return selected

    async def _describe_pending_image_batch(
        self,
        group_openid: str,
        *,
        exclude_key: str,
        budget: int,
        batch: list[tuple[StoredMessage, list[ImageAttachment]]] | None = None,
    ) -> int:
        """同步识别最近一批待看图，返回这次消耗的图片额度。

        每张消息识别完立刻写回消息表，全部处理完再让调用方组装当前话题和 prompt。
        """
        if budget <= 0:
            return 0
        items = (
            batch
            if batch is not None
            else self._pending_image_batch(
                group_openid,
                exclude_key=exclude_key,
                budget=budget,
            )
        )
        used = 0
        for stored, attachments in items:
            logger.info(
                "describing pending image batch synchronously: %s (%s image(s))",
                stored.message_key,
                len(attachments),
            )
            descriptions = await self._describe_attachments(attachments)
            used += len(attachments)
            if not any(description.strip() for description in descriptions):
                logger.warning(
                    "image description empty for pending message %s, marking unrecognized",
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
            logger.info("pending image described: %s", stored.message_key)
        return used

    async def _transcribe_voice_attachments(
        self,
        attachments: list[VoiceAttachment],
    ) -> list[str]:
        transcripts: list[str] = []
        for attachment in attachments:
            try:
                transcript = await self.voice_transcriber.transcribe(attachment)
            except Exception:
                logger.exception("failed to transcribe voice attachment")
                transcript = ""
            transcripts.append((transcript or "").strip())
        return transcripts

    def _pending_voice_batch(
        self,
        group_openid: str,
        *,
        exclude_key: str,
        budget: int,
    ) -> list[tuple[StoredMessage, list[VoiceAttachment]]]:
        """收集「先连续发语音、下一条文字再 @」时的那批语音。

        和图片批次一样，只从最新往回扫连续的一批；遇到普通文字或已经处理过的
        语音就停止，避免把很久以前、链接早已失效的语音重新拖进来。
        """
        if budget <= 0:
            return []
        selected: list[tuple[StoredMessage, list[VoiceAttachment]]] = []
        remaining = budget
        for stored in reversed(
            self.store.recent_messages_with_raw(
                group_openid,
                max(int(getattr(self.settings, "recent_messages", 60) or 60), budget),
            )
        ):
            if stored.message_key == exclude_key or stored.role == "bot":
                continue
            attachments = voice_attachments(stored.raw)
            if attachments and needs_voice_transcript(stored.content, stored.raw):
                taken = attachments[:remaining]
                if taken:
                    selected.append((stored, taken))
                    remaining -= len(taken)
                if remaining <= 0:
                    break
                continue
            break
        selected.reverse()
        return selected

    async def _transcribe_pending_voice_batch(
        self,
        group_openid: str,
        *,
        exclude_key: str,
        budget: int,
        batch: list[tuple[StoredMessage, list[VoiceAttachment]]] | None = None,
    ) -> int:
        """同步转写最近一批语音并立刻回写数据库，返回处理条数。"""
        if budget <= 0:
            return 0
        items = (
            batch
            if batch is not None
            else self._pending_voice_batch(
                group_openid,
                exclude_key=exclude_key,
                budget=budget,
            )
        )
        used = 0
        for stored, attachments in items:
            logger.info(
                "transcribing pending voice batch synchronously: %s (%s voice(s))",
                stored.message_key,
                len(attachments),
            )
            transcripts = await self._transcribe_voice_attachments(attachments)
            used += len(attachments)
            if any(transcripts):
                enriched = message_with_voice_transcripts(stored.content, transcripts)
            else:
                logger.warning(
                    "voice transcript empty for pending message %s, marking unrecognized",
                    stored.message_key,
                )
                enriched = message_with_unrecognized_voice(stored.content)
            if enriched != stored.content:
                self.store.update_message_content(
                    group_openid,
                    stored.message_key,
                    enriched,
                )
        return used

    def _queue_image_backfill(self, group_openid: str, *, exclude_key: str, budget: int) -> None:
        """把旧图片的补识别丢到后台去跑，不卡在 @ 回复路径上。

        它跟本轮回复没关系，却原来是同步 await 的：最多能串着识别好几张旧图，
        每张都要下载 + 一次视觉模型调用，用户干等十几秒才看到回复。
        同一个群同时只跑一份，免得两条消息挨着来就把额度重复烧一遍。
        """
        if budget <= 0 or group_openid in self._image_backfill_running:
            return
        self._image_backfill_running.add(group_openid)

        async def run() -> None:
            try:
                await self._describe_recent_images(
                    group_openid, exclude_key=exclude_key, budget=budget
                )
            except Exception:
                logger.exception("image backfill failed for %s", group_openid)
            finally:
                self._image_backfill_running.discard(group_openid)

        task = asyncio.create_task(run())
        self._image_backfill_tasks.add(task)
        task.add_done_callback(self._image_backfill_tasks.discard)

    def _now(self) -> datetime:
        return datetime.now(ZoneInfo(self.settings.timezone))

    def _monotonic(self) -> float:
        """活跃窗口用单调时钟，系统校时不会让 10 分钟忽然变短或变长。"""
        return time.monotonic()

    def _active_window_seconds(self) -> int:
        return max(0, int(getattr(self.settings, "active_window_seconds", 600) or 0))

    def _active_followup_seconds(self) -> int:
        return max(
            0, int(getattr(self.settings, "active_followup_seconds", 180) or 0)
        )

    def _activate_active_group(self, group_openid: str) -> None:
        seconds = self._active_window_seconds()
        if seconds <= 0:
            self._active_windows.pop(group_openid, None)
            return
        now = self._monotonic()
        self._active_windows[group_openid] = _ActiveWindow(
            until=now + seconds,
            last_message_at=now,
        )
        logger.info("active chat started for %s for %ss", group_openid, seconds)

    def _active_window(self, group_openid: str) -> _ActiveWindow | None:
        window = self._active_windows.get(group_openid)
        if window is None:
            return None
        if self._monotonic() >= window.until:
            self._active_windows.pop(group_openid, None)
            return None
        return window

    def _touch_active_group(self, group_openid: str) -> None:
        window = self._active_window(group_openid)
        if window is not None:
            window.last_message_at = self._monotonic()

    def _active_lock(self, group_openid: str) -> asyncio.Lock:
        lock = self._active_locks.get(group_openid)
        if lock is None:
            lock = asyncio.Lock()
            self._active_locks[group_openid] = lock
        return lock

    @staticmethod
    def _message_stats_range(
        intent: dict,
        now: datetime,
    ) -> tuple[datetime | None, datetime, str]:
        """把意图里的 period 换成 [since, until) 和给人看的范围说明。"""
        raw_period = str(intent.get("period") or "").strip().lower()
        raw_period = raw_period.replace("-", "_").replace(" ", "_")
        aliases = {
            "3_days": "last_3_days",
            "three_days": "last_3_days",
            "最近三天": "last_3_days",
            "这三天": "last_3_days",
            "7_days": "last_7_days",
            "seven_days": "last_7_days",
            "最近一周": "last_7_days",
            "这一周": "last_7_days",
            "这周": "last_7_days",
            "本周": "this_week",
            "上周": "last_week",
            "这个月": "this_month",
            "本月": "this_month",
            "全部": "all",
            "所有": "all",
            "今天的": "today",
            "昨天的": "yesterday",
        }
        period = aliases.get(raw_period, raw_period or "last_7_days")
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if period == "today":
            return today, now, "今天"
        if period == "yesterday":
            return today - timedelta(days=1), today, "昨天"
        if period == "last_3_days":
            return now - timedelta(days=3), now, "最近 3 天"
        if period in {"last_7_days", "this_week"}:
            return now - timedelta(days=7), now, "最近 7 天"
        if period == "last_week":
            this_monday = today - timedelta(days=today.weekday())
            return this_monday - timedelta(days=7), this_monday, "上周"
        if period == "this_month":
            month_start = today.replace(day=1)
            return month_start, now, "本月"
        if period == "all":
            return None, now, "全部历史记录"
        if period == "custom":
            start_text = str(intent.get("start_date") or "").strip()
            end_text = str(intent.get("end_date") or "").strip()
            if not start_text or not end_text:
                raise ValueError("自定义统计需要开始日期和结束日期")
            try:
                start_day = datetime.strptime(start_text, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
                end_day = datetime.strptime(end_text, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            except ValueError as exc:
                raise ValueError("日期格式应为 YYYY-MM-DD") from exc
            if end_day < start_day:
                raise ValueError("结束日期不能早于开始日期")
            until = end_day + timedelta(days=1)
            if until > now:
                until = now
            return start_day, until, f"{start_day:%Y-%m-%d} 至 {end_day:%Y-%m-%d}"
        return now - timedelta(days=7), now, "最近 7 天"

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
        # 称呼先归一再取：显示名是老叫法（比如「小虾」）时也能查到登记过的称呼。
        addresses: dict[str, str] = {}
        for member in [*members, requester]:
            name = (member.username or "").strip()
            if not name or name in addresses:
                continue
            address = self.store.address_for(group_openid, name)
            if address:
                addresses[name] = address
        return IdentityContext(
            requester=requester,
            partner=partner,
            addresses=addresses,
            speaker_labels=self.store.speaker_labels(group_openid),
        )

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

            try:
                await self._maybe_active_followup()
            except Exception:
                logger.exception("failed to consider an active-chat follow-up")

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
                    content=self._outgoing_text(task.group_openid, content),
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
        """把攒够的聊天压成一段摘要。

        这里**不**自动生产长期记忆：用户要求「只有我明确说记住 xx 才记」，
        长期记忆一律由 memory_remember（群里 @ 我下指令）写入，后台只出摘要。
        """
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
                    self.store.speaker_labels(group_openid),
                )
                result = await self.llm.process_memory_batch(message_context)
                summary = str(result.get("summary") or "").strip()
                if not summary:
                    continue
                summary = clip_summary(summary)

                self.store.save_memory_batch(
                    group_openid=group_openid,
                    start_row_id=batch.start_row_id,
                    end_row_id=batch.end_row_id,
                    summary=summary,
                    memories=[],
                )
                logger.info(
                    "saved chat summary for group %s: messages=%s",
                    group_openid,
                    len(batch.messages),
                )

    @staticmethod
    def _format_stored_messages(
        messages: list[StoredMessage],
        max_chars: int,
        labels: dict[str, str] | None = None,
    ) -> str:
        """归档/情绪判断看到的记录跟主回复同一套：时间 + 【说话人】 + 正文。"""
        return format_topic(messages, max_chars, labels)

    def _memory_context(self, group_openid: str) -> str:
        """长期记忆只装记忆卡，不再混入对话摘要。

        摘要属于「最近几天的剧情梗概」，和「稳定的事实」是两回事；混在一起后模型会把
        剧情当成人设，张口就用摘要里的评价复述人。这里用 memory_cards_for_prompt 按
        字符预算装卡片，摘要单独走 _summaries_context。
        """
        cards = self.store.memory_cards_for_prompt(group_openid)
        return ChatStore.format_memory_cards(cards)[: self.settings.max_context_chars]

    def _summaries_context(
        self,
        group_openid: str,
        *,
        limit: int = 2,
        max_chars: int = 900,
    ) -> str:
        """最近几天的对话概要，只取最新两段，整块截断到 max_chars。"""
        summaries = self.store.conversation_summaries(group_openid, limit=limit)
        text = "\n".join(f"[摘要] {summary}" for summary in summaries)
        return text[:max_chars]

    # ---------- @ 后的群级活跃窗口 ----------

    async def _send_active_message(
        self,
        group_openid: str,
        text: str,
        *,
        source: str,
    ) -> bool:
        """主动发送一条自然接话；有图时严格先文字、后单独发图。"""
        content, sticker = self._outgoing_parts(group_openid, text)
        chunks = split_reply_text(
            content,
            int(getattr(self.settings, "reply_chunk_chars", 900) or 900),
        )
        if not chunks and sticker is None:
            return False

        sent_chunks: list[str] = []
        try:
            for chunk in chunks:
                await self.api.post_group_message(
                    group_openid=group_openid,
                    content=chunk,
                )
                sent_chunks.append(chunk)

            if sticker is not None:
                file_info = await self._upload_group_image(group_openid, sticker)
                if not file_info:
                    raise RuntimeError(
                        f"sticker upload returned no file_info: {sticker.name}"
                    )
                await self._post_sticker_message(group_openid, file_info)
        except Exception as exc:
            self._note_rate_limit(group_openid, exc)
            if sent_chunks:
                # 文字已经送达时记下已经发生的内容；没送达的表情不写进上下文。
                self._remember_bot_reply(
                    group_openid, "\n".join(sent_chunks), source
                )
            logger.warning("active message send failed: %s", exc)
            # 已经送达部分正文时算“这轮已经接上”，不要让关键词关心再补一句；
            # 一个字都没送达才返回 False，交给原有关心流程兜底。
            return bool(sent_chunks)

        self._remember_bot_reply(group_openid, text, source)
        return True

    async def _maybe_active_reply(self, message: RichGroupMessage) -> bool:
        """活跃窗口里的非 @ 消息：只判断和生成自然聊天，不进入 action 分发。"""
        group_openid = message.group_openid
        if self._active_window(group_openid) is None:
            return False

        async with self._active_lock(group_openid):
            if self._active_window(group_openid) is None:
                return False

            content = normalize_face_tags(message.content).strip()
            requester = StoredMessage(
                username=message.username,
                member_openid=message.author.member_openid or "",
                role=message.member_role,
                content=content,
                timestamp=message.timestamp or "",
                source="GROUP_MESSAGE_CREATE",
            )
            topic, background, _full_recent = self._topic_context(group_openid)
            identity = self._identity_context(group_openid, requester)
            verdict = await self.llm.judge_active_reply(
                content=content,
                topic_context=topic,
                mode="message",
            )
            reason = str(verdict.get("reason") or "").strip()
            if not verdict.get("reply"):
                logger.info(
                    "active chat stayed quiet for %s: %s",
                    message.username,
                    reason or "model chose silence",
                )
                return False

            reply = await self.llm.generate_active_reply(
                requester=requester,
                request=content or "（只发了表情或附件，没有文字）",
                topic_context=topic,
                background_context=background,
                memory_context=self._memory_context(group_openid),
                summaries_context=self._summaries_context(group_openid),
                identity_context=identity.prompt(),
                face_context=face_prompt(
                    self.store.face_book(group_openid),
                    as_tags=self.settings.face_tags,
                    stickers=self.stickers,
                ),
                mode="message",
            )
            reply = (reply or "").strip()
            if not reply:
                return False

            if await self._send_active_message(
                group_openid,
                reply,
                source="BOT_ACTIVE",
            ):
                logger.info(
                    "active chat replied to %s (%s): %s",
                    message.username,
                    reason or "natural opening",
                    reply[:80],
                )
                # 已经真正接上话，就不再让关键词关心兜底，避免一轮两说。
                return True
            return False

    async def _maybe_active_followup(self) -> None:
        """活跃会话安静一段时间后，最多主动补一句话。"""
        followup_seconds = self._active_followup_seconds()
        if followup_seconds <= 0:
            return

        for group_openid in list(self._active_windows):
            window = self._active_window(group_openid)
            if window is None or window.followup_done:
                continue
            if self._monotonic() - window.last_message_at < followup_seconds:
                continue

            async with self._active_lock(group_openid):
                window = self._active_window(group_openid)
                if (
                    window is None
                    or window.followup_done
                    or self._monotonic() - window.last_message_at < followup_seconds
                ):
                    continue
                window.followup_done = True

                recent = self.store.recent_messages(
                    group_openid,
                    min(20, self.settings.recent_messages),
                )
                last_human = next(
                    (
                        item
                        for item in reversed(recent)
                        if not (item.source or "").startswith("BOT_")
                    ),
                    None,
                )
                if last_human is None:
                    continue

                topic, background, _full_recent = self._topic_context(group_openid)
                identity = self._identity_context(group_openid, last_human)
                verdict = await self.llm.judge_active_reply(
                    content=last_human.content,
                    topic_context=topic,
                    mode="followup",
                )
                reason = str(verdict.get("reason") or "").strip()
                if not verdict.get("reply"):
                    logger.info(
                        "active follow-up stayed quiet for %s: %s",
                        group_openid,
                        reason or "model chose silence",
                    )
                    continue

                idle_seconds = int(
                    max(0, self._monotonic() - window.last_message_at)
                )
                reply = await self.llm.generate_active_reply(
                    requester=last_human,
                    request=last_human.content,
                    topic_context=topic,
                    background_context=background,
                    memory_context=self._memory_context(group_openid),
                    summaries_context=self._summaries_context(group_openid),
                    identity_context=identity.prompt(),
                    face_context=face_prompt(
                        self.store.face_book(group_openid),
                        as_tags=self.settings.face_tags,
                        stickers=self.stickers,
                    ),
                    mode="followup",
                    idle_seconds=idle_seconds,
                )
                reply = (reply or "").strip()
                if not reply:
                    continue

                # 模型生成期间可能有新消息进来；这时把舞台还给真人，不再补话。
                current = self._active_window(group_openid)
                if (
                    current is None
                    or self._monotonic() - current.last_message_at < followup_seconds
                ):
                    continue
                await self._send_active_message(
                    group_openid,
                    reply,
                    source="BOT_ACTIVE",
                )
                logger.info(
                    "active follow-up sent for %s (%s): %s",
                    group_openid,
                    reason or "conversation still open",
                    reply[:80],
                )

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
                recent_context=self._format_stored_messages(
                    recent, 600, self.store.speaker_labels(group_openid)
                ),
            )
            if not verdict.get("care"):
                return
            text = normalize_care_text(str(verdict.get("text") or ""))
            if not text:
                return
            await self.api.post_group_message(
                group_openid=group_openid,
                content=self._outgoing_text(group_openid, text),
            )
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
            self._activate_active_group(message.group_openid)
            await self._process_at_message(message, "GROUP_MESSAGE_CREATE")
            return

        self._touch_active_group(message.group_openid)
        if self._active_window(message.group_openid) is not None:
            try:
                if await self._maybe_active_reply(message):
                    return
            except Exception:
                logger.exception("active-chat reply failed")
        try:
            await self._maybe_care(message)
        except Exception:
            logger.exception("proactive care failed")

    async def on_group_at_message_create(self, message: RichGroupMessage) -> None:
        if not self._group_allowed(message.group_openid):
            return
        self._activate_active_group(message.group_openid)
        await self._process_at_message(message, "GROUP_AT_MESSAGE_CREATE")

    async def _process_at_message(self, message: RichGroupMessage, source: str) -> None:
        if not self._group_allowed(message.group_openid):
            return
        started = time.perf_counter()
        intent_ms = 0.0
        action = "-"

        key = ChatStore.message_key(message.id, message.msg_seq)
        if not self.store.mark_at_processed(key):
            logger.info("duplicate @ event ignored: %s", key)
            return

        self._save_message(message, source)

        # 图片描述和语音转写都写回本地库，最近上下文、长期记忆、历史检索都能读到。
        content = normalize_face_tags(message.content).strip()
        descriptions: list[str] = []
        voice_transcripts: list[str] = []
        image_budget = max(
            0, int(getattr(self.settings, "vision_max_images_per_request", 0) or 0)
        )
        voice_budget = max(
            0, int(getattr(self.settings, "asr_max_per_request", 5) or 0)
        )
        pre_acked = False
        ack_task: asyncio.Task[bool] | None = None

        current_images = (
            image_attachments(message.raw)[:image_budget] if image_budget > 0 else []
        )
        pending_image_batch: list[tuple[StoredMessage, list[ImageAttachment]]] = []
        if image_budget > 0 and not current_images:
            pending_image_batch = self._pending_image_batch(
                message.group_openid,
                exclude_key=key,
                budget=image_budget,
            )
        planned_images = len(current_images) + sum(
            len(items) for _, items in pending_image_batch
        )

        current_voices = (
            voice_attachments(message.raw)[:voice_budget] if voice_budget > 0 else []
        )
        pending_voice_batch: list[tuple[StoredMessage, list[VoiceAttachment]]] = []
        if voice_budget > 0 and not current_voices:
            pending_voice_batch = self._pending_voice_batch(
                message.group_openid,
                exclude_key=key,
                budget=voice_budget,
            )
        planned_voices = len(current_voices) + sum(
            len(items) for _, items in pending_voice_batch
        )

        if planned_images > 0 or planned_voices > 0:
            # 下载识图或调 ASR 都可能卡几秒。图片和语音共用一次 ack，避免用户
            # 同时收到「我看看图」和「我听听」两条垫场；正式回复从 seq=2 开始。
            if planned_images > 0 and planned_voices > 0:
                ack_text = ACK_TEXT_VOICE_IMAGES
            elif planned_voices > 0:
                ack_text = ACK_TEXT_VOICE
            else:
                ack_text = ACK_TEXT_IMAGES if planned_images > 1 else ACK_TEXT_IMAGE
            ack_task = asyncio.create_task(
                self._send_reply(
                    message,
                    ack_text,
                    1,
                    f"{source}:ACK_MEDIA",
                    attempts=1,
                    proactive_fallback=False,
                    remember=False,
                )
            )

        if current_images:
            descriptions = await self._describe_attachments(
                current_images, hint=content
            )
            image_budget -= len(current_images)
            if any(description.strip() for description in descriptions):
                enriched = message_with_image_descriptions(content, descriptions)
                if enriched != content:
                    self.store.update_message_content(
                        message.group_openid, key, enriched
                    )
                content = enriched
        if not current_images and image_budget > 0:
            # 图片先单独发、下一条再 @ 问图：这一轮必须先把最近一批看完，
            # 否则 prompt 已经组装完，后台识别再快也赶不上当前回复。
            used = await self._describe_pending_image_batch(
                message.group_openid,
                exclude_key=key,
                budget=image_budget,
                batch=pending_image_batch,
            )
            image_budget -= used

        if current_voices:
            voice_transcripts = await self._transcribe_voice_attachments(
                current_voices
            )
            voice_budget -= len(current_voices)
            if any(voice_transcripts):
                enriched = message_with_voice_transcripts(
                    content, voice_transcripts
                )
            else:
                logger.warning(
                    "voice transcript empty for current message %s, marking unrecognized",
                    key,
                )
                enriched = message_with_unrecognized_voice(content)
            if enriched != content:
                self.store.update_message_content(
                    message.group_openid, key, enriched
                )
            content = enriched
        if not current_voices and voice_budget > 0:
            # 语音先单独发、下一条文字再 @：先同步转写最近一批，再组装话题。
            used = await self._transcribe_pending_voice_batch(
                message.group_openid,
                exclude_key=key,
                budget=voice_budget,
                batch=pending_voice_batch,
            )
            voice_budget -= used

        if ack_task is not None:
            try:
                pre_acked = await ack_task
            except Exception:
                logger.exception("failed to send media ack")
                pre_acked = False
        self._queue_image_backfill(
            message.group_openid, exclude_key=key, budget=image_budget
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
            acked = pre_acked
            try:
                bot_mention = self._bot_mention(message)
                raw_request = content
                request = (
                    self._strip_bot_mention(raw_request, bot_mention)
                    if bot_mention
                    else raw_request
                )
                request = request or EMPTY_AT_REQUEST

                intent_started = time.perf_counter()
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
                    memory_context=self._memory_context(message.group_openid),
                )
                intent_ms = (time.perf_counter() - intent_started) * 1000
                action = str(intent.get("action") or "chat")
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
                    pre_acked=pre_acked,
                )
                logger.info("handled @ request %s from %s", intent.get("action"), message.username)
                logger.info(
                    "perf at action=%s intent=%.0fms total=%.0fms images=%d",
                    action,
                    intent_ms,
                    (time.perf_counter() - started) * 1000,
                    len(descriptions),
                )
                logger.info(
                    "perf voice action=%s transcripts=%d",
                    action,
                    len(voice_transcripts),
                )
            except Exception:
                logger.exception("failed to generate or send reply")
                await self._send_reply(
                    message, FALLBACK_TEXT, 2 if acked else 1, f"{source}:FALLBACK"
                )

    def _topic_context(self, group_openid: str) -> tuple[str, str, str]:
        """切成【当前话题】+【更早的背景】，另外留一份全量记录给历史检索兜底。"""
        recent = self.store.recent_messages(group_openid, self.settings.recent_messages)
        labels = self.store.speaker_labels(group_openid)
        topic_slice = split_topic(recent)
        return (
            format_topic(topic_slice.current, self.settings.max_context_chars, labels),
            format_topic(topic_slice.previous_tail, 4000, labels),
            format_topic(recent, self.settings.max_context_chars, labels),
        )

    @staticmethod
    def _is_dedup_error(error: BaseException) -> bool:
        return DEDUP_CODE in str(error)

    def _outgoing_text(self, group_openid: str, text: str) -> str:
        """出站前处理表情：模型写的 [头秃] 还原成真表情标签。

        库里存的永远是可读名字（跟入库时把标签解码成 [头秃] 对称），所以只在出站这一处
        做转换，别让 <faceType=...> 这种标签回流进上下文。
        """
        if not text:
            return text
        return self._outgoing_parts(group_openid, text)[0]

    def _outgoing_parts(self, group_openid: str, text: str) -> tuple[str, Sticker | None]:
        """出站前的最后一道加工：先摘表情包，再处理小表情。

        顺序不能反。表情包的名字（[抱抱]）是要换成一张图的，得在 render_faces 之前
        摘走；否则它先被当成小表情名处理掉，图就发不出去了。
        """
        if not text:
            return text or "", None
        remainder, sticker = self.stickers.take(text)
        rendered = render_faces(
            remainder,
            self.store.face_book(group_openid),
            as_tags=self.settings.face_tags,
        )
        return rendered, sticker

    async def _upload_group_media(
        self,
        group_openid: str,
        data: bytes,
        *,
        file_type: int,
        label: str,
    ) -> str:
        """上传图片/语音等富媒体，返回下一条消息要用的 file_info。"""
        http = getattr(self.api, "_http", None)
        if http is None:
            return ""
        started = time.perf_counter()
        route = Route("POST", "/v2/groups/{group_openid}/files", group_openid=group_openid)
        result = await http.request(
            route,
            json={
                "file_type": file_type,
                "file_data": base64.b64encode(data).decode("ascii"),
                "srv_send_msg": False,
            },
        )
        file_info = str((result or {}).get("file_info") or "")
        logger.info(
            "perf media upload=%s type=%s %.0fms bytes=%d ok=%s",
            label,
            file_type,
            (time.perf_counter() - started) * 1000,
            len(data),
            bool(file_info),
        )
        return file_info

    async def _upload_group_image(self, group_openid: str, sticker: Sticker) -> str:
        """把本地表情图传上去换 file_info。

        qq-botpy 1.2.1 的 post_group_file 只收 url，本地文件塞不进去，所以直接打
        /v2/groups/{group_openid}/files 这个接口。
        """
        data = await asyncio.to_thread(sticker.read)
        return await self._upload_group_media(
            group_openid,
            data,
            file_type=1,
            label=sticker.name,
        )

    async def _upload_group_voice(self, group_openid: str, data: bytes) -> str:
        """上传一段语音，file_type=3 对应 QQ 群语音富媒体。"""
        return await self._upload_group_media(
            group_openid,
            data,
            file_type=3,
            label="voice",
        )

    async def _post_sticker_message(
        self,
        group_openid: str,
        file_info: str,
        *,
        msg_id: str | None = None,
        msg_seq: int | None = None,
    ) -> None:
        """file_info 已经上传好时，只发图片消息这一步。"""
        payload: dict[str, Any] = {
            "group_openid": group_openid,
            "msg_type": 7,
            # msg_type=7 时 content 必须为空，否则 QQ 会在图下面画出一行空白。
            "content": None,
            "media": {"file_info": file_info},
        }
        if msg_id is not None:
            payload["msg_id"] = msg_id
        if msg_seq is not None:
            payload["msg_seq"] = msg_seq
        started = time.perf_counter()
        await self.api.post_group_message(**payload)
        logger.info(
            "perf sticker send seq=%s %.0fms",
            msg_seq,
            (time.perf_counter() - started) * 1000,
        )

    async def _post_voice_message(
        self,
        group_openid: str,
        file_info: str,
        *,
        msg_id: str | None = None,
        msg_seq: int | None = None,
    ) -> None:
        """用 QQ 语音消息发送已上传的 file_info。"""
        payload: dict[str, Any] = {
            "group_openid": group_openid,
            "msg_type": 7,
            # 富媒体消息的 content 必须为空，否则语音条下面可能带出一条空白正文。
            "content": None,
            "media": {"file_info": file_info},
        }
        if msg_id is not None:
            payload["msg_id"] = msg_id
        if msg_seq is not None:
            payload["msg_seq"] = msg_seq
        started = time.perf_counter()
        await self.api.post_group_message(**payload)
        logger.info(
            "perf voice send seq=%s %.0fms",
            msg_seq,
            (time.perf_counter() - started) * 1000,
        )

    async def _send_sticker(
        self,
        group_openid: str,
        sticker: Sticker,
        *,
        msg_id: str | None = None,
        msg_seq: int | None = None,
    ) -> None:
        """单独发一张表情图；文字由调用方先走普通文本消息发出去。"""
        file_info = await self._upload_group_image(group_openid, sticker)
        if not file_info:
            raise RuntimeError(f"sticker upload returned no file_info: {sticker.name}")
        await self._post_sticker_message(
            group_openid, file_info, msg_id=msg_id, msg_seq=msg_seq
        )

    async def _send_reply_proactive(
        self,
        message: RichGroupMessage,
        text: str,
        source: str,
        *,
        remember: bool = True,
    ) -> bool:
        """被动回复通道彻底不通时，退回主动消息兜底，至少让人看到一句回复。"""
        content = self._outgoing_parts(message.group_openid, text)[0]
        chunks = split_reply_text(
            content,
            int(getattr(self.settings, "reply_chunk_chars", 900) or 900),
        )
        if not chunks:
            # 这条本来就只是一张表情图，没有能被文字兜住的正文，就别硬发一句空的。
            logger.warning("proactive fallback skipped, nothing but a sticker: %s", message.id)
            return False
        try:
            for chunk in chunks:
                await self.api.post_group_message(
                    group_openid=message.group_openid,
                    content=chunk,
                )
        except Exception as exc:
            self._note_rate_limit(message.group_openid, exc)
            logger.exception("proactive fallback also failed, msg_id=%s", message.id)
            return False
        if remember:
            self._remember_bot_reply(message.group_openid, content, f"{source}:FALLBACK")
        logger.info("group reply delivered via proactive fallback, msg_id=%s", message.id)
        return True

    async def _send_reply(
        self,
        message: RichGroupMessage,
        text: str,
        seq: int,
        source: str,
        *,
        attempts: int = SEND_ATTEMPTS,
        proactive_fallback: bool = True,
        remember: bool = True,
    ) -> bool:
        """按 msg_seq 回复群消息。同一条 msg_id 下每次实际发送都必须用不同的 seq。

        发送失败不再直接放弃。线上出现过一次 40054005「消息被去重」：ack 那次请求
        其实已经送到、只是本地报了异常，于是 acked=False、正文仍用 seq=1，
        被 QQ 当成重复请求拒掉——用户看到的就是「@ 了没反应」。现在的处理是：
        - 第一次就被判去重：这个 seq 槽被自己刚发的 ack 占了，换成 seq+1 重发正文；
        - 重试时才被判去重：说明上一次其实已经发出去了，直接算投递成功，不再重复发；
        - 其它错误按原 seq 重试，最多 attempts 次；
        - 全部失败退回主动消息通道兜底（ack 不兜底，它发不出去无所谓）。

        普通回复的长文字先按自然边界拆成多段，每段占一个独立 seq，全部文字发完后再用
        下一个 seq 单独发图。语音回复则只发语音，不先发文字；只有 TTS、上传或语音发送
        失败时才退回文字。中途失败只兜底尚未送达的剩余文字，绝不重发已送达的前文。
        """
        wants_voice = bool(getattr(text, "voice_reply", False))
        content, sticker = self._outgoing_parts(message.group_openid, text)
        if wants_voice:
            # 语音回复只发一条语音，不附文字或表情包。
            sticker = None
        current = seq

        def remember_delivery() -> None:
            if remember:
                self._remember_bot_reply(message.group_openid, text, source)
        max_attempts = max(1, attempts)
        chunks = split_reply_text(
            content,
            int(getattr(self.settings, "reply_chunk_chars", 900) or 900),
        )
        has_text = bool(chunks)

        if not has_text and sticker is None:
            logger.error("refusing to send an empty group reply, seq=%s", current)
            return False

        # 表情图要先读文件再 base64 上传，这一步往往比发一条文字还慢。用户要求
        # 「先文字、后单独发图」，所以可见顺序不能动，但上传可以提前：文字发送的
        # 同时在后台把图传好，文字一到，紧接着用现成的 file_info 发图。
        upload_task: asyncio.Task[str] | None = None
        pre_upload = sticker is not None and has_text
        if pre_upload:
            upload_task = asyncio.create_task(
                self._upload_group_image(message.group_openid, sticker)
            )
        # 语音成功时这就是本轮唯一可见的回复；失败时才把正文当兜底文字发送。
        voice_task: asyncio.Task[bytes] | None = None
        if wants_voice and has_text and self.voice_synthesizer.enabled:
            voice_task = asyncio.create_task(
                self.voice_synthesizer.synthesize(str(text))
            )

        try:
            if voice_task is not None:
                try:
                    voice_data = await voice_task
                except Exception as exc:
                    logger.warning("TTS failed, falling back to text: %s", exc)
                    voice_data = b""
                voice_task = None

                file_info = ""
                if voice_data:
                    try:
                        file_info = await self._upload_group_voice(
                            message.group_openid,
                            voice_data,
                        )
                    except Exception as exc:
                        logger.warning(
                            "voice upload failed, falling back to text: %s", exc
                        )

                if file_info:
                    for attempt in range(1, max_attempts + 1):
                        try:
                            await self._post_voice_message(
                                message.group_openid,
                                file_info,
                                msg_id=message.id,
                                msg_seq=current,
                            )
                        except Exception as exc:
                            if self._is_dedup_error(exc):
                                if attempt > 1:
                                    logger.info(
                                        "voice already delivered, seq=%s", current
                                    )
                                    remember_delivery()
                                    return True
                                logger.warning(
                                    "voice msg_seq %s already taken, resending as %s",
                                    current,
                                    current + 1,
                                )
                                current += 1
                                continue
                            self._note_rate_limit(message.group_openid, exc)
                            logger.warning(
                                "voice send failed (%s), seq=%s; falling back to text",
                                exc,
                                current,
                            )
                            break
                        else:
                            remember_delivery()
                            return True
                elif voice_data:
                    logger.warning(
                        "voice upload returned no file_info, falling back to text"
                    )

            if has_text:
                sent_chunks = 0
                for chunk_index, chunk in enumerate(chunks):
                    text_started = time.perf_counter()
                    text_seq = current
                    text_sent = False
                    for attempt in range(1, max_attempts + 1):
                        try:
                            await self.api.post_group_message(
                                group_openid=message.group_openid,
                                msg_id=message.id,
                                msg_seq=current,
                                content=chunk,
                            )
                        except Exception as exc:
                            if self._is_dedup_error(exc):
                                if attempt > 1:
                                    logger.info(
                                        "group reply chunk already delivered, seq=%s",
                                        current,
                                    )
                                    current += 1
                                    text_sent = True
                                    break
                                logger.warning(
                                    "msg_seq %s already taken (ack landed?), "
                                    "resending chunk as %s",
                                    current,
                                    current + 1,
                                )
                                current += 1
                                continue
                            logger.warning(
                                "failed to send group reply chunk %d/%d, "
                                "attempt %s/%s, seq=%s: %s",
                                chunk_index + 1,
                                len(chunks),
                                attempt,
                                max_attempts,
                                current,
                                exc,
                            )
                            if attempt < max_attempts:
                                await asyncio.sleep(SEND_RETRY_DELAY_SECONDS)
                                continue
                            if proactive_fallback:
                                remaining = "\n".join(chunks[chunk_index:])
                                if sent_chunks:
                                    fallback_ok = await self._send_reply_proactive(
                                        message,
                                        remaining,
                                        source,
                                        remember=False,
                                    )
                                    if fallback_ok:
                                        remember_delivery()
                                    return fallback_ok
                                return await self._send_reply_proactive(
                                    message, text, source
                                )
                            logger.error(
                                "giving up on group reply chunk, seq=%s", current
                            )
                            return False
                        else:
                            current += 1
                            text_sent = True
                            break
                    if not text_sent:
                        logger.error("giving up on group reply, seq=%s", current)
                        return False
                    sent_chunks += 1
                    logger.info(
                        "perf text send seq=%s %.0fms chars=%d chunk=%d/%d",
                        text_seq,
                        (time.perf_counter() - text_started) * 1000,
                        len(chunk),
                        chunk_index + 1,
                        len(chunks),
                    )

            if wants_voice:
                # 没配 TTS，或 TTS/上传/发送失败后，正文已作为兜底文字发出。
                remember_delivery()
                return True

            if sticker is None:
                remember_delivery()
                return True

            file_info = ""
            if upload_task is not None:
                try:
                    file_info = await upload_task
                except Exception as exc:
                    logger.warning(
                        "sticker pre-upload failed (%s): %s", sticker.name, exc
                    )
                upload_task = None
            if not file_info and pre_upload:
                # 上传已经试过一次并失败了：文字早就发出去了，再传一次只会让整轮更慢。
                logger.warning(
                    "sticker %s upload failed, keep the text only", sticker.name
                )
                remember_delivery()
                return True

            for attempt in range(1, max(1, attempts) + 1):
                try:
                    if file_info:
                        await self._post_sticker_message(
                            message.group_openid,
                            file_info,
                            msg_id=message.id,
                            msg_seq=current,
                        )
                    else:
                        await self._send_sticker(
                            message.group_openid,
                            sticker,
                            msg_id=message.id,
                            msg_seq=current,
                        )
                except Exception as exc:
                    if self._is_dedup_error(exc):
                        if attempt > 1:
                            logger.info("sticker already delivered, seq=%s", current)
                            remember_delivery()
                            return True
                        logger.warning(
                            "sticker msg_seq %s already taken, resending as %s",
                            current,
                            current + 1,
                        )
                        current += 1
                        continue
                    self._note_rate_limit(message.group_openid, exc)
                    logger.warning(
                        "sticker %s failed (%s), seq=%s", sticker.name, exc, current
                    )
                    if has_text:
                        # 文字已经发出去了，图失败不能再重发文字，也不能让人以为整轮没回复。
                        remember_delivery()
                        return True
                    logger.error(
                        "sticker failed and there is no text to send, seq=%s", current
                    )
                    return False
                else:
                    remember_delivery()
                    return True

            if has_text:
                remember_delivery()
                return True
            logger.error("giving up on sticker, seq=%s", current)
            return False
        finally:
            if voice_task is not None:
                voice_task.cancel()
                try:
                    await voice_task
                except BaseException:
                    pass
            if upload_task is not None:
                upload_task.cancel()
                try:
                    await upload_task
                except BaseException:
                    pass

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
        pre_acked: bool = False,
    ) -> bool:
        """执行 intent，返回是否已经发过 ack（决定正式回复用哪个 msg_seq）。"""
        action = str(intent.get("action") or "chat")
        handle_started = time.perf_counter()
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
        seq = 2 if pre_acked else 1
        acked = pre_acked
        timeout = self.settings.ack_after_seconds
        if not pre_acked and action in ACK_ACTIONS and timeout > 0:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
            if not done:
                # ack 只是垫场：发不出去就别重试也别兜底，把 seq=1 让给正文。
                acked = await self._send_reply(
                    message,
                    (
                        ACK_TEXT_CHAT
                        if action in {"chat", "voice_chat"}
                        else ACK_TEXT
                    ),
                    1,
                    f"{source}:ACK",
                    attempts=1,
                    proactive_fallback=False,
                )
                if acked:
                    seq = 2
        try:
            answer = await task
        except Exception:
            logger.exception("failed to build reply for action %s", action)
            answer = FALLBACK_TEXT
        handle_ms = (time.perf_counter() - handle_started) * 1000
        send_started = time.perf_counter()
        await self._send_reply(message, answer, seq, f"{source}:REPLY")
        logger.info(
            "perf reply action=%s handle=%.0fms send=%.0fms ack=%s",
            action,
            handle_ms,
            (time.perf_counter() - send_started) * 1000,
            acked,
        )
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

    def _memory_ids_field(self, value: Any) -> list[int] | None:
        """intent 里的 memory_id 可能是单个编号，也可能是一串。统一成 int 列表。

        这里拿到的是「查看记忆」显示的群内连续序号，不是数据库主键；
        后续必须走 find_memory_cards_by_numbers 解析。
        """
        if value is None:
            return None
        items = value if isinstance(value, (list, tuple)) else [value]
        ids: list[int] = []
        for item in items:
            parsed = self._int_field(item)
            if parsed is not None:
                ids.append(parsed)
        return ids or None

    def _resolve_memory_targets(
        self, intent: dict, message: RichGroupMessage
    ) -> tuple[list[MemoryCard], bool]:
        """拿出要操作的那批记忆，返回 (卡片, 是否是“全部”)。

        支持三种说法：报编号（单个或一串）、报内容关键词（match）、说“全部/所有”。
        早先只认单个 int，model 返回数组时 _int_field 直接返回 None，
        于是每次都答“没有找到这条记忆”。
        """
        group_openid = message.group_openid
        if bool(intent.get("all")):
            return self.store.all_memory_cards(group_openid), True
        ids = self._memory_ids_field(intent.get("memory_id"))
        if ids:
            # 用户说的编号是「查看记忆」里看到的群内连续序号，不是数据库主键。
            return self.store.find_memory_cards_by_numbers(group_openid, ids), False
        match = self._optional_str(intent.get("match"))
        if match:
            return self.store.find_memory_cards(group_openid, query=match), False
        return [], False

    async def _handle_agent_run(
        self,
        *,
        intent: dict,
        message: RichGroupMessage,
        request: str,
        topic: str,
        identity: "IdentityContext",
    ) -> str:
        """agent_run：工具循环已经拿着完整证据收口，结果直接发送。"""
        query = self._optional_str(intent.get("query")) or request
        mode = str(intent.get("mode") or "fast")
        if mode not in {"fast", "deep"}:
            mode = "fast"

        # fast 只是查一下，关思考最省时；deep 才值得让模型多想一会儿。
        agent_effort = (
            getattr(self.settings, "deep_reasoning_effort", "high")
            if mode == "deep"
            else getattr(self.settings, "reasoning_effort", "none")
        )
        ctx = AgentContext(
            llm_api_key=self.settings.llm_api_key,
            llm_base_url=self.settings.llm_base_url,
            model=self.settings.model,
            reasoning_effort=agent_effort,
            search_max_results=self.settings.search_max_results,
            search_timeout_seconds=self.settings.search_timeout_seconds,
            web_fetch_max_chars=self.settings.web_fetch_max_chars,
            document_max_chars=self.settings.document_max_chars,
            shell_enabled=self.settings.shell_enabled,
            shell_cwd=self.settings.shell_cwd,
            shell_mode=self.settings.shell_mode,
            shell_timeout_seconds=self.settings.shell_timeout_seconds,
            shell_max_output_chars=self.settings.shell_max_output_chars,
            memory_search_messages=self.settings.memory_search_messages,
            max_output_chars=(
                getattr(
                    self.settings,
                    "deep_reply_max_chars",
                    getattr(self.settings, "reply_max_chars", 3000),
                )
                if mode == "deep"
                else getattr(self.settings, "reply_max_chars", 3000)
            ),
            store=self.store,
            group_openid=message.group_openid,
            now=self._now(),
            speaker_labels=identity.speaker_labels or self.store.speaker_labels(message.group_openid),
        )

        result = await agent_run_loop(
            client=self.llm.client,
            ctx=ctx,
            query=query,
            mode=mode,
        )
        return result.get("text", "") or "抱歉，我没查到结果。"


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
        # 永久/临时记忆入口已经下线；旧模型若还返回这两个动作，按普通聊天处理。
        if action in {"memory_pin", "memory_unpin"}:
            action = "chat"

        async def speak(result: str) -> str:
            """办成一件事之后，让模型用小毛豆的语气把这句确认话说一遍。

            只有 handler 明确标了 persona=True 的成功模板会被改写；列表、报错、
            「匹配到多条，请说编号」这些原样返回——那些话必须精确可预测。
            """
            return await self._persona_or_raw(
                action, result, request, message, identity, topic
            )

        if action == "task_create":
            return await speak(self._handle_task_create(intent, message))
        if action in {"task_list", "task_cancel", "task_pause", "task_resume", "task_skip"}:
            return await speak(self._handle_task_action(intent, message))
        if action == "memory_remember":
            return await speak(self._handle_memory_remember(intent, message, identity))
        if action == "memory_list":
            return self._handle_memory_list(intent, message)
        if action == "memory_forget":
            return await speak(self._handle_memory_forget(intent, message))
        if action == "memory_alias":
            return await speak(self._handle_memory_alias(intent, message))
        if action == "address_set":
            return await speak(self._handle_address_set(intent, message, identity))
        if action == "calendar_query":
            return self._handle_calendar_query(intent)
        if action == "bot_pref":
            return await speak(self._handle_bot_pref(intent, message))
        if action == "agent_run":
            return await self._handle_agent_run(
                intent=intent,
                message=message,
                request=request,
                topic=topic,
                identity=identity,
            )

        face_context = face_prompt(
            self.store.face_book(message.group_openid),
            as_tags=self.settings.face_tags,
            stickers=self.stickers,
        )
        voice_reply = action == "voice_chat" and self.voice_synthesizer.enabled

        reply = await self.llm.generate_reply(
            requester=requester,
            request=request,
            topic_context=topic,
            background_context=background,
            memory_context=self._memory_context(message.group_openid),
            summaries_context=self._summaries_context(message.group_openid),
            identity_context=identity.prompt(),
            face_context=face_context,
            voice_reply=voice_reply,
        )
        if voice_reply:
            return VoiceReply(reply)
        return reply

    async def _persona_or_raw(
        self,
        action: str,
        result: str,
        request: str,
        message: RichGroupMessage,
        identity: IdentityContext,
        topic: str,
    ) -> str:
        """本地操作办成后，把模板文案交给模型用小毛豆的语气说一遍。

        模板是事实凭据，模型只许改语气：facts 里的条数、编号、人名、称呼、任务方向
        缺一个就退回模板。最怕的就是为了活泼把「记住了 3 条」说成「都记下啦」，
        数字一丢，用户根本没法核对到底存进去几条。
        """
        if not isinstance(result, ActionResult) or not result.persona:
            return result
        template = str(result)
        try:
            reply = await self.llm.generate_tool_reply(
                request=request,
                tool_name=PERSONA_TOOL_LABELS.get(action, action),
                tool_output=template,
                identity_context=identity.prompt(),
                memory_context=self._memory_context(message.group_openid),
                topic_context=topic,
                max_chars=_persona_max_chars(template),
                addendum=ACTION_REPLY_ADDENDUM,
            )
        except Exception as exc:
            # 一句确认话而已，模型挂了不该让整轮对话失败。
            logger.warning("persona reply failed for %s: %s", action, exc)
            return template
        reply = (reply or "").strip()
        if not reply:
            return template
        if _contradicts_success(reply):
            logger.info(
                "persona reply for %s contradicted the result, fell back to template", action
            )
            return template
        missing = _missing_facts(result.facts, reply)
        if missing:
            logger.info(
                "persona reply for %s dropped facts %s, fell back to template",
                action,
                missing,
            )
            return template
        return reply

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
        # 时间标签和任务名是硬事实：模型可以换语气，但「每天 09:00」和任务名
        # 必须原样带出来，否则用户没法核对到底定在了几点、定的是哪件事。
        label = self._schedule_label(task)
        facts = [str(task.id), label, task.title]
        reply = f"已设置任务 {task.id}：{label}，{task.title}。"
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
            warm_label = self._schedule_label(warm)
            facts += [str(warm.id), warm_label]
            reply += f"\n还建了任务 {warm.id}：{warm_label}提醒提前准备。"
        return ActionResult(reply, persona=True, facts=facts)

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
            return self._task_done("取消", f"已取消任务 {task.id}：{task.title}。", task)
        if action == "task_pause":
            self.store.set_task_status(message.group_openid, task.id, "paused")
            return self._task_done("暂停", f"已暂停任务 {task.id}：{task.title}。", task)
        if action == "task_skip":
            if task.schedule_type == "once":
                return "单次任务没有“下一次”，要取消就直接说取消。"
            self.store.set_task_skip_next(message.group_openid, task.id, True)
            return self._task_done(
                "跳过", f"好的，任务 {task.id} 下一次会跳过。", task, with_title=False
            )
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
            return self._task_done("恢复", f"已恢复任务 {task.id}：{task.title}。", task)

        return "这个任务操作我还没听明白。"

    @staticmethod
    def _task_done(
        direction: str,
        template: str,
        task: ScheduledTask,
        *,
        with_title: bool = True,
    ) -> ActionResult:
        """任务操作办成的那句确认话。方向词和编号锁死，语气交给模型。"""
        facts = [direction, str(task.id)]
        if with_title and task.title:
            facts.append(task.title)
        return ActionResult(template, persona=True, facts=facts)

    @staticmethod
    def _schedule_label(task: ScheduledTask) -> str:
        return schedule_label(task)

    @staticmethod
    def _is_sensitive_memory(content: str) -> bool:
        keywords = ("密码", "身份证", "银行卡", "验证码", "门牌", "住址")
        return any(keyword in content for keyword in keywords)

    def _canonical_person(
        self,
        person: str | None,
        identity: IdentityContext,
        group_openid: str,
    ) -> str:
        value = (person or "").strip()
        if value in {"我", "本人", "自己", "当前用户", "用户", "你"}:
            value = identity.requester.username
        elif value in {"她", "他", "对象", "另一半", "伴侣", "对方", "你的对象"}:
            value = identity.partner.username if identity.partner else "两人"
        value = value or "两人"
        # 群里统一过叫法就用统一后的那个，别再往库里添第四种写法。
        return self.store.canonical_person(group_openid, value)

    def _handle_memory_remember(
        self,
        intent: dict,
        message: RichGroupMessage,
        identity: IdentityContext,
    ) -> str:
        raw_memories = intent.get("memories")
        if not isinstance(raw_memories, list) or not raw_memories:
            return "要记住什么信息，可以更具体地说一次吗？"

        # 旧意图里可能还带 pin / temporary；现在记忆只有长期一种，直接忽略这两个字段。
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
                    message.group_openid,
                ),
                content=content,
                source_message_id=message.id,
            )
            saved += 1

        if not saved:
            return "这条信息比较敏感，我不保存。"
        # 只存进一条时不强制模型报数字，否则「记住啦～」这种最自然的说法会被判成丢事实、
        # 永远退回模板——那等于这个功能在最常用的场景下白做。
        facts = [] if saved == 1 and not skipped else [str(saved)]
        if skipped:
            facts.append(str(skipped))
        if saved and not skipped:
            return ActionResult(f"记住了，共 {saved} 条。", persona=True, facts=facts)
        return ActionResult(
            f"记住了 {saved} 条，另有 {skipped} 条涉及敏感信息，我没有保存。",
            persona=True,
            facts=facts,
        )

    def _handle_memory_list(self, intent: dict, message: RichGroupMessage) -> str:
        kind = self._optional_str(intent.get("kind"))
        person = self._optional_str(intent.get("person"))
        match = self._optional_str(intent.get("match"))
        filtered = bool(kind or person or match)
        cards, current, pages, total = self.store.memory_cards_page(
            message.group_openid,
            page=max(1, self._int_field(intent.get("page")) or 1),
            kind=kind,
            person=person,
            match=match,
        )
        if not total:
            if filtered:
                return "没有符合这个筛选条件的记忆。"
            return "我现在还没有长期记忆。可以直接说：记住她不吃香菜。"
        if not cards:
            return f"记忆一共 {total} 条，只有 {pages} 页，最后一页说“第 {pages} 页记忆”。"
        header = f"当前长期记忆（第{current}/{pages}页，共{total}条）："
        if current < pages:
            header += f"\n想看更多就说“第 {current + 1} 页记忆”。"
        numbers = self.store.memory_card_numbers(message.group_openid)
        return header + "\n" + self._format_memory_cards(cards, numbers)

    def _handle_memory_alias(self, intent: dict, message: RichGroupMessage) -> str:
        """统一某个人在记忆里的叫法。

        同一个人被记成「罗晓霞 / 二十四桥明月夜 / 小虾」时，模型会当成三个人，
        称呼跟着乱。这里不写死任何名字，统一成什么全由群里 @ 我下指令决定。
        """
        group_openid = message.group_openid
        op = str(intent.get("op") or "show").strip().lower()

        if op not in {"set", "clear"}:
            aliases = self.store.person_aliases(group_openid)
            if not aliases:
                return (
                    "还没统一过叫法。想统一就说：以后把「A」和「B」都叫「C」，"
                    "我会把已经记住的条目一起改过来。"
                )
            lines = "\n".join(f"· {alias} → {name}" for alias, name in sorted(aliases.items()))
            return "现在统一过的叫法：\n" + lines

        raw_alias = intent.get("alias")
        if raw_alias is None:
            raw_alias = intent.get("aliases")
        if isinstance(raw_alias, list):
            names = [str(item).strip() for item in raw_alias if str(item).strip()]
        else:
            text = str(raw_alias or "").strip()
            names = [text] if text else []
        if not names:
            return "要统一哪个叫法？说清楚一点，比如：以后把「A」和「B」都叫「C」。"

        if op == "clear":
            cleared = [name for name in names if self.store.clear_person_alias(group_openid, name)]
            if not cleared:
                return "这几个叫法我本来就没统一过：" + "、".join(names) + "。"
            return ActionResult(
                "好，不再统一了：" + "、".join(cleared) + "。已经存下的记忆我没动。",
                persona=True,
                facts=tuple(cleared),
            )

        canonical = self._optional_str(intent.get("canonical"))
        if not canonical:
            return "要统一成哪个叫法？比如：以后把「A」和「B」都叫「C」。"
        names = [name for name in names if name != canonical]
        if not names:
            return f"「{canonical}」就是它自己，不用统一。"

        rewritten = sum(self.store.set_person_alias(group_openid, name, canonical) for name in names)
        tail = f"，已经记住的 {rewritten} 条也一起改过来了" if rewritten else ""
        # 改了几条是用户唯一能核对的数字，必须活着走出润色。被统一的旧叫法本身
        # 不作硬性事实：模型说「以后都叫她小虾」同样准确，硬要求只会每次都退回模板。
        return ActionResult(
            f"好，以后「{'」「'.join(names)}」都统一叫「{canonical}」{tail}。",
            persona=True,
            facts=(canonical, *((str(rewritten),) if rewritten else ())),
        )

    def _handle_address_set(
        self, intent: dict, message: RichGroupMessage, identity: IdentityContext
    ) -> str:
        """登记「以后该怎么叫某个人」。

        称呼是人设的一部分，代码里不写死任何一个；叫什么、叫谁，全由群里 @ 我下指令决定。
        """
        group_openid = message.group_openid
        op = str(intent.get("op") or "show").strip().lower()

        if op not in {"set", "clear"}:
            addresses = self.store.person_addresses(group_openid)
            if not addresses:
                return "还没定过称呼。想定就直接说：以后叫我「主人」，叫他「小虾主人」。"
            lines = "\n".join(
                f"· {name} → 「{address}」" for name, address in sorted(addresses.items())
            )
            return "现在定好的称呼：\n" + lines

        person = self._optional_str(intent.get("person"))
        if not person:
            # “以后叫我X”这种没说清是谁的，默认就是当前触发者。
            person = (identity.requester.username or message.username or "").strip()
        if not person:
            return "你想让我怎么称呼谁？说清楚一点，比如：以后叫他「小虾主人」。"

        if op == "clear":
            if not self.store.clear_person_address(group_openid, person):
                return f"「{person}」本来就没定过称呼。"
            return ActionResult(f"好，以后就直接叫{person}了。", persona=True, facts=(person,))

        address = self._optional_str(intent.get("address"))
        if not address:
            return "想让我怎么称呼？把称呼说出来，比如：以后叫我「主人」。"
        name = self.store.set_person_address(group_openid, person, address)
        if not name:
            return "这个称呼我没存下来，再说一次试试。"
        clean = address.strip("「」“”")
        # 只锁称呼本身，不锁显示名。模型很自然会说「好嘞主人～」而不再重复一遍「CL」，
        # 逼它带上显示名，这句确认话就继续生硬——那等于白改。
        return ActionResult(f"好，以后就叫{name}「{clean}」。", persona=True, facts=(clean,))

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
            return ActionResult(
                f"好，{label}恢复成默认值 {self._default_pref(kind, attr)}。",
                persona=True,
                facts=(label, self._default_pref(kind, attr)),
            )

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
            # 「想恢复就说要主动关心」是这句话里唯一有用的操作提示，锁住别被润色掉。
            return ActionResult(
                "好，那我不主动插话了，你们 @ 我随时都在。想恢复就说「要主动关心」。",
                persona=True,
                facts=("要主动关心",),
            )
        return ActionResult(
            f"记下了，这个群的{label}改为{shown}。", persona=True, facts=(label, shown)
        )

    @staticmethod
    def _format_memory_cards(
        cards: list[MemoryCard], numbers: dict[int, int]
    ) -> str:
        """按群内连续编号展示；numbers 是 card.id -> 展示编号。"""
        return "\n".join(
            f"{numbers.get(card.id, card.id)}. [{card.kind}] {card.person}: {card.content}"
            for card in cards
        )

    def _handle_memory_forget(self, intent: dict, message: RichGroupMessage) -> str:
        by_id = self._memory_ids_field(intent.get("memory_id"))
        want_all = bool(intent.get("all"))
        cards, is_all = self._resolve_memory_targets(intent, message)
        if not cards:
            return "没有找到这条记忆。可以说“查看记忆”。"
        numbers = self.store.memory_card_numbers(message.group_openid)
        if not by_id and not want_all and len(cards) > 1:
            return (
                "匹配到多条记忆，请说编号：\n"
                + self._format_memory_cards(cards, numbers)
            )

        deleted = sum(
            1
            for card in cards
            if self.store.delete_memory_card(message.group_openid, card.id)
        )
        if not deleted:
            return "这些记忆刚刚变动过，麻烦再说一次。"
        if len(cards) == 1:
            number = numbers.get(cards[0].id, cards[0].id)
            return ActionResult(
                f"已忘记记忆 {number}：{cards[0].content}",
                persona=True,
                facts=("忘记", str(number)),
            )
        return ActionResult(
            f"已忘记{'全部 ' if is_all else ''}{deleted} 条记忆。",
            persona=True,
            facts=("忘记", str(deleted)),
        )

    @staticmethod
    def _message_stats_label(
        member: MemberMessageCount,
        labels: dict[str, str],
    ) -> str:
        if member.is_bot:
            return "小毛豆"
        openid = member.member_key.removeprefix("openid:")
        if member.member_key.startswith("openid:"):
            label = labels.get(openid)
            if label:
                return label
        return labels.get(member.username) or member.username or "未知成员"

    def _format_message_stats(
        self,
        stats: MessageStats,
        *,
        since: datetime | None,
        until: datetime,
        range_label: str,
        labels: dict[str, str],
        include_bot: bool,
    ) -> str:
        """生成一份事实锁死的统计文本，模型只负责把它说得像小毛豆。"""
        if range_label == "全部历史记录":
            range_line = "统计范围：全部历史记录"
        elif " 至 " in range_label:
            range_line = f"统计范围：{range_label}（含首尾两天）"
        elif since is not None:
            range_line = (
                f"统计范围：{range_label}"
                f"（{since:%Y-%m-%d %H:%M} 至 {until:%Y-%m-%d %H:%M}）"
            )
        else:
            range_line = f"统计范围：{range_label}"

        if not stats.members:
            return f"{range_line}\n这段时间没有聊天记录。"

        scope = "含小毛豆" if include_bot else "不含小毛豆"
        lines = [range_line, f"共 {stats.total} 条（{scope}）"]
        for member in stats.members:
            label = self._message_stats_label(member, labels)
            share = member.count / stats.total * 100 if stats.total else 0.0
            lines.append(f"{label}：{member.count} 条（{share:.1f}%）")

        top = stats.members[0]
        top_label = self._message_stats_label(top, labels)
        if len(stats.members) == 1:
            lines.append(f"只有 {top_label} 发了消息。")
        else:
            lines.append(f"发得最多的是 {top_label}，{top.count} 条。")
        return "\n".join(lines)

    async def _handle_message_stats(
        self,
        *,
        intent: dict,
        message: RichGroupMessage,
        request: str,
        topic: str,
        identity: IdentityContext,
    ) -> str:
        try:
            since, until, range_label = self._message_stats_range(intent, self._now())
        except ValueError as exc:
            return f"时间范围我没看懂：{exc}。可以说“最近三天”或“10月1日到10月3日”。"

        include_bot = bool(intent.get("include_bot"))
        stats = self.store.message_stats(
            message.group_openid,
            since=since.isoformat(timespec="seconds") if since else None,
            until=until.isoformat(timespec="seconds"),
            include_bot=include_bot,
        )
        labels = identity.speaker_labels or self.store.speaker_labels(message.group_openid)
        template = self._format_message_stats(
            stats,
            since=since,
            until=until,
            range_label=range_label,
            labels=labels,
            include_bot=include_bot,
        )
        if not stats.members:
            return template

        try:
            reply = await self.llm.generate_tool_reply(
                request=request,
                tool_name="message_stats",
                tool_output=template,
                identity_context=identity.prompt(),
                memory_context=self._memory_context(message.group_openid),
                topic_context=topic,
                addendum=STRUCTURED_DATA_REPLY_ADDENDUM,
            )
        except Exception as exc:
            logger.warning("message stats persona reply failed: %s", exc)
            return template

        reply = (reply or "").strip()
        # 条数必须原样出现。模型要是把 415 说成“四百多”或直接写反，退回事实模板。
        numbers = [stats.total, *(member.count for member in stats.members)]
        if not reply or any(
            re.search(rf"(?<!\d){number}(?!\d)", reply) is None for number in numbers
        ):
            logger.info("message stats reply dropped exact counts, fell back to template")
            return template
        return reply

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
            priority_text=request,
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
            numbers = self.store.memory_card_numbers(message.group_openid)
            search_parts.append(
                "相关长期记忆：\n" + self._format_memory_cards(cards, numbers)
            )

        return await self.llm.generate_reply(
            requester=requester,
            request=request,
            topic_context=recent,
            memory_context=self._memory_context(message.group_openid),
            search_context="\n\n".join(search_parts),
            identity_context=identity.prompt(),
            # 这一轮确实检索过，得写清楚，否则模型会照着“（无）”说“我没查”。
            executed_actions="在本地聊天记录和长期记忆里检索过（结果附在下面）",
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
            site = (item.get("site_name") or "").strip()
            if not site and url:
                site = urlparse(url).netloc
            meta: list[str] = []
            if site:
                meta.append(f"来源：{site}")
            date = str(item.get("date") or "").strip()
            if date:
                meta.append(date[:10])
            try:
                authority = int(item.get("authority"))
            except (TypeError, ValueError):
                authority = -1
            if authority >= 3:
                meta.append("权威度：高")
            suffix = f"（{'，'.join(meta)}）" if meta else ""
            lines.append(f"{index}. {title}{suffix}\n   {url}\n   {snippet}")
        return "\n".join(lines)

    async def _handle_web_search(
        self,
        request: str,
        intent: dict,
        *,
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
    ) -> str:
        query = self._optional_str(intent.get("query")) or request
        logger.info("web_search query=%r provider=deepseek", query)
        result = await search_web(
            query,
            limit=self.settings.search_max_results,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            original_question=request,
            timeout=self.settings.search_timeout_seconds,
        )
        if not result.ok:
            logger.warning("web search failed: %s", result.error)
            if "没有配置" in (result.error or ""):
                return "我还没配好联网搜索，需要先在服务器上填 LLM_API_KEY，配好后就能查了。"
            detail = (result.error or "未知错误").replace("联网搜索没有配置：", "")[:80]
            return f"搜索服务这次没返回可用结果：{detail}，要不换个说法再让我查一次？"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="web_search",
            tool_output=self._format_search_result(result),
            identity_context=identity_context,
            memory_context=memory_context,
            topic_context=topic_context,
        )

    async def _handle_web_fetch(
        self,
        request: str,
        intent: dict,
        *,
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
    ) -> str:
        url = self._optional_str(intent.get("url"))
        if not url:
            return await self._handle_web_search(
                request,
                intent,
                identity_context=identity_context,
                memory_context=memory_context,
                topic_context=topic_context,
            )
        result = await fetch_page(
            url,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            max_chars=self.settings.web_fetch_max_chars,
        )
        if not result.ok:
            return result.error or "网页读取失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="web_fetch",
            tool_output=result.output,
            identity_context=identity_context,
            memory_context=memory_context,
            topic_context=topic_context,
        )

    async def _handle_document_read(
        self,
        request: str,
        intent: dict,
        message: RichGroupMessage,
        *,
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
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
            identity_context=identity_context,
            memory_context=memory_context,
            topic_context=topic_context,
        )

    async def _handle_shell_run(
        self,
        request: str,
        intent: dict,
        *,
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
    ) -> str:
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
            identity_context=identity_context,
            memory_context=memory_context,
            topic_context=topic_context,
            max_chars=TOOL_REPLY_MAX_CHARS.get("shell_run"),
        )

    async def _handle_math_eval(
        self,
        request: str,
        intent: dict,
        *,
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
    ) -> str:
        expression = self._optional_str(intent.get("expression")) or request
        result = eval_math(expression)
        if not result.ok:
            return result.error or "计算失败。"
        return await self.llm.generate_tool_reply(
            request=request,
            tool_name="math_eval",
            tool_output=result.output,
            identity_context=identity_context,
            memory_context=memory_context,
            topic_context=topic_context,
            max_chars=TOOL_REPLY_MAX_CHARS.get("math_eval"),
        )

    async def on_group_msg_receive(self, event) -> None:
        logger.info("group full-message receiving enabled: %s", event.group_openid)

    async def on_group_msg_reject(self, event) -> None:
        logger.warning("group full-message receiving disabled: %s", event.group_openid)

    async def on_group_add_robot(self, event) -> None:
        logger.info("Xiaomaodou was added to group: %s", event.group_openid)

    async def on_group_del_robot(self, event) -> None:
        logger.warning("Xiaomaodou was removed from group: %s", event.group_openid)
