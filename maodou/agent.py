from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from .calendar import format_festivals, lunar_label, next_festival, upcoming_festivals
from .completion import (
    completion_is_incomplete,
    continuation_prompt,
    merge_continuation,
)
from .llm import (
    DEEP_RESEARCH_REPLY_ADDENDUM,
    FAST_TOOL_REPLY_ADDENDUM,
    SYSTEM_PROMPT,
)
from .store import ChatStore, MemoryCard
from .tools import (
    ToolResult,
    eval_math,
    fetch_page,
    read_document,
    run_shell,
    search_web,
)

logger = logging.getLogger(__name__)

# ── 工具定义（OpenAI function calling 格式） ──────────────────────────

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网，返回相关网页的标题、链接和摘要。适合查实时信息、新闻、百科、医学资料等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "2-5个中文关键词组成的检索词，必须自包含——把「上次说的」「那个」等指代替换成具体内容；不要把「依据」「权威」「来源」这些回答形式要求当检索词。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "打开并阅读指定网页的正文内容。适合在 web_search 之后深入阅读某篇具体文章。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要阅读的网页 URL"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "document_read",
            "description": "读取 PDF、Word、Excel 等文档的内容。给出文档的 URL 即可。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "文档的下载链接"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_run",
            "description": "在服务器上执行只读 shell 命令（ls/cat/grep/find/wc/head/tail/date/df/free 等），用于查看文件、统计信息或验证系统状态。严禁写入或删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的只读 shell 命令"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_eval",
            "description": "计算数学表达式，支持加减乘除、三角函数、对数、幂等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 2+3*4 或 sqrt(144)"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_query",
            "description": "查询节日日期。可以查指定节日的日期，也可以列出未来30天的节日。",
            "parameters": {
                "type": "object",
                "properties": {
                    "festival_name": {
                        "type": "string",
                        "description": "要查询的节日名（如「中秋」「除夕」），留空则列出未来30天所有节日",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "message_stats",
            "description": "统计群聊消息数量。可以统计某个时间段内每个人发了多少条消息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "开始日期，如 2026-10-01；不填则从最早记录开始",
                    },
                    "until": {
                        "type": "string",
                        "description": "结束日期，如 2026-10-25；不填则到当前时间",
                    },
                    "include_bot": {
                        "type": "boolean",
                        "description": "是否包含小毛豆自己的消息，默认 false",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "在聊天记录和长期记忆中搜索相关内容。适合回答「我们之前聊过什么」「有没有提到过XX」这类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

AGENT_OPERATION_PROMPT = """你正在执行 agent_run 工具任务。下面是操作层规则，和前面的全局人设一起使用。

可用工具：
- web_search：搜索互联网，获取最新信息
- web_fetch：打开并阅读指定网页
- document_read：读取 PDF/Word/Excel 等文档
- shell_run：在服务器上执行只读命令，查看文件或系统状态
- math_eval：计算数学表达式
- calendar_query：查节日日期
- message_stats：统计群聊消息数量
- memory_search：搜索聊天记录和长期记忆

工作方式：
1. 先理解用户真正要完成的任务，只调用能补足信息的工具。
2. 简单查询尽快收口；复杂任务先收集资料，再基于证据分析。不要为了显得努力而重复搜索同一个问题。
3. 工具轨迹就是你本轮实际执行的操作；没有调用的能力不能声称已经用过。
4. 结果与问题无关时直说没找到，不能拿无关内容凑答案，也不能编造来源。
5. 如果资料已经足够，直接给出最终回答，不要为了再确认一遍继续查；如果资料不足，说明缺什么，不要继续证明自己搜过。
6. 最终回答的形式服从当前模式追加的输出策略，不要套用固定客服模板。

预算：
- fast：最多 3 次工具调用、3 个模型回合，适合简单事实查询。
- deep：最多 12 次工具调用、8 个模型回合，适合调研、比较、周报、年报和需要多步分析的任务。
- 单回合最多并行 3 个工具；预算用完后必须停止查资料，基于已有结果回答。"""


AGENT_RUN_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + AGENT_OPERATION_PROMPT


FAST_MAX_STEPS = 3
DEEP_MAX_STEPS = 8
FAST_MAX_TOOL_CALLS = 3
DEEP_MAX_TOOL_CALLS = 12
MAX_PARALLEL_TOOLS = 3
FAST_STEP_MAX_TOKENS = 1200
DEEP_STEP_MAX_TOKENS = 1600
FAST_FINAL_MAX_TOKENS = 1800
DEEP_FINAL_MAX_TOKENS = 6000
FAST_CONTINUATION_MAX_TOKENS = 1200
DEEP_CONTINUATION_MAX_TOKENS = 3000
MAX_COMPLETION_CONTINUATIONS = 2
# 工具回合负责规划和查证，最终落笔不需要继续跑高档思考；否则推理 token
# 会和正文抢同一份 max_tokens，长回答很容易只写一半。
FINAL_REASONING_EFFORT = "low"
TOOL_RESULT_MAX_CHARS = 4000


def agent_system_prompt(mode: str) -> str:
    """全局人设 + 工具操作说明 + 当前模式的最终输出契约。"""
    addendum = (
        DEEP_RESEARCH_REPLY_ADDENDUM
        if mode == "deep"
        else FAST_TOOL_REPLY_ADDENDUM
    )
    return AGENT_RUN_SYSTEM_PROMPT + addendum


@dataclass
class AgentContext:
    """agent_run 所处的上下文环境，封装所有依赖。"""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    model: str = ""
    reasoning_effort: str = "none"
    search_max_results: int = 5
    search_timeout_seconds: float = 12.0
    web_fetch_max_chars: int = 8000
    document_max_chars: int = 24000
    shell_enabled: bool = False
    shell_cwd: str = "/tmp"
    shell_mode: str = "readonly"
    shell_timeout_seconds: float = 10.0
    shell_max_output_chars: int = 50000
    memory_search_messages: int = 10
    # 仅作为长回答的诊断阈值保留；agent_run 不会在发送前硬裁正文。
    max_output_chars: int = 6000
    store: Any = None  # ChatStore
    group_openid: str = ""
    now: datetime = field(default_factory=datetime.now)
    speaker_labels: dict[str, str] = field(default_factory=dict)


def _format_search_result(result: ToolResult) -> str:
    """把搜索结果格式化为文本。"""
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
    return "\n".join(lines) if lines else "（无结果）"


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    ctx: AgentContext,
) -> str:
    """执行单个工具调用，返回文本结果。"""
    try:
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return "错误：搜索词不能为空"
            result = await search_web(
                query,
                limit=ctx.search_max_results,
                api_key=ctx.llm_api_key,
                base_url=ctx.llm_base_url,
                timeout=ctx.search_timeout_seconds,
            )
            if not result.ok:
                return f"搜索失败：{result.error}"
            return _format_search_result(result)

        elif name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return "错误：URL 不能为空"
            result = await fetch_page(
                url,
                api_key=ctx.llm_api_key,
                base_url=ctx.llm_base_url,
                max_chars=ctx.web_fetch_max_chars,
            )
            if not result.ok:
                return f"网页读取失败：{result.error}"
            return result.output

        elif name == "document_read":
            url = str(arguments.get("url", "")).strip()
            if not url:
                return "错误：文档 URL 不能为空"
            result = await read_document(url, max_chars=ctx.document_max_chars)
            if not result.ok:
                return f"文档读取失败：{result.error}"
            return result.output

        elif name == "shell_run":
            if not ctx.shell_enabled:
                return "Shell 执行功能未开启"
            command = str(arguments.get("command", "")).strip()
            if not command:
                return "错误：命令不能为空"
            result = await run_shell(
                command,
                cwd=ctx.shell_cwd,
                mode=ctx.shell_mode,
                timeout=ctx.shell_timeout_seconds,
                max_output_chars=ctx.shell_max_output_chars,
            )
            if not result.ok and not result.output:
                return f"命令执行失败：{result.error}"
            out = result.output or "(无输出)"
            if result.error:
                out += f"\n[退出码：{result.error}]"
            return out

        elif name == "math_eval":
            expression = str(arguments.get("expression", "")).strip()
            if not expression:
                return "错误：表达式不能为空"
            result = eval_math(expression)
            if not result.ok:
                return f"计算失败：{result.error}"
            return result.output

        elif name == "calendar_query":
            name_str = str(arguments.get("festival_name", "")).strip()
            now = ctx.now
            if name_str:
                found = next_festival(name_str, now)
                if found is None:
                    return f"没查到「{name_str}」是哪天，换个常见节日名试试"
                day, festival = found
                suffix = ""
                from .calendar import solar_to_lunar as _s2l
                lunar = _s2l(day)
                if lunar:
                    suffix = f"，农历{lunar_label(lunar[0], lunar[1], leap=lunar[2])}"
                return (
                    f"{festival}是 {day.year}年{day.month}月{day.day}日"
                    f"（星期{'一二三四五六日'[day.weekday()]}）{suffix}，还有 {(day - now.date()).days} 天。"
                )
            else:
                items = upcoming_festivals(now, days=30)
                if not items:
                    return "未来 30 天没有节日。"
                return "未来 30 天的节日：" + format_festivals(items, today=now.date())

        elif name == "message_stats":
            store: ChatStore | None = ctx.store
            if store is None:
                return "消息统计功能不可用（数据库未连接）"

            since_str = str(arguments.get("since", "")).strip()
            until_str = str(arguments.get("until", "")).strip()
            include_bot = bool(arguments.get("include_bot", False))

            since = None
            if since_str:
                try:
                    since = datetime.strptime(since_str, "%Y-%m-%d")
                except ValueError:
                    return f"日期格式不正确：{since_str}，请用 YYYY-MM-DD"
            until = ctx.now
            if until_str:
                try:
                    until = datetime.strptime(until_str, "%Y-%m-%d")
                except ValueError:
                    return f"日期格式不正确：{until_str}，请用 YYYY-MM-DD"

            stats = store.message_stats(
                ctx.group_openid,
                since=since.isoformat(timespec="seconds") if since else None,
                until=until.isoformat(timespec="seconds"),
                include_bot=include_bot,
            )
            if not stats.members:
                return "这段时间没有聊天记录。"

            labels = ctx.speaker_labels or {}
            lines = [f"共 {stats.total} 条（{'含' if include_bot else '不含'}小毛豆）"]
            for member in stats.members:
                openid = member.member_key.removeprefix("openid:")
                label = labels.get(openid) or labels.get(member.username) or member.username or "未知成员"
                share = member.count / stats.total * 100 if stats.total else 0
                lines.append(f"{label}：{member.count} 条（{share:.1f}%）")
            return "\n".join(lines)

        elif name == "memory_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return "错误：搜索词不能为空"
            store: ChatStore | None = ctx.store
            if store is None:
                return "记忆搜索功能不可用"
            results = store.search_messages(
                ctx.group_openid,
                query,
                ctx.memory_search_messages,
            )
            cards = store.search_memory_cards(ctx.group_openid, query)
            if not results and not cards:
                return f"没找到和「{query}」有关的记录。"
            parts = []
            if results:
                parts.append("相关历史消息：\n" + ChatStore.format_search_results(results))
            if cards:
                parts.append("相关长期记忆：\n" + ChatStore.format_memory_cards(cards))
            return "\n\n".join(parts)

        else:
            return f"未知工具：{name}"

    except Exception as exc:
        logger.warning("tool %s failed: %r", name, exc)
        return f"工具 {name} 执行出错：{exc}"


async def agent_run_loop(
    *,
    client: Any,
    ctx: AgentContext,
    query: str,
    mode: str = "fast",
) -> dict[str, Any]:
    """有界工具调用循环。

    fast 最多 3 个模型回合 / 3 次工具调用；deep 最多 8 个回合 / 12 次工具调用。
    单回合最多并行 3 个工具。预算用完后不再暴露工具，强制基于已有证据收口。
    返回 {"text": "最终回复", "tool_calls_made": ["web_search", ...]}
    """
    mode = "deep" if mode == "deep" else "fast"
    max_steps = DEEP_MAX_STEPS if mode == "deep" else FAST_MAX_STEPS
    max_tool_calls = (
        DEEP_MAX_TOOL_CALLS if mode == "deep" else FAST_MAX_TOOL_CALLS
    )
    step_max_tokens = (
        DEEP_STEP_MAX_TOKENS if mode == "deep" else FAST_STEP_MAX_TOKENS
    )
    final_max_tokens = (
        DEEP_FINAL_MAX_TOKENS if mode == "deep" else FAST_FINAL_MAX_TOKENS
    )
    continuation_max_tokens = (
        DEEP_CONTINUATION_MAX_TOKENS
        if mode == "deep"
        else FAST_CONTINUATION_MAX_TOKENS
    )
    model = ctx.model or "deepseek-flash"
    tool_calls_made: list[str] = []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": agent_system_prompt(mode)},
        {"role": "user", "content": query},
    ]

    def _finish_reason(response: Any) -> str | None:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)

    def _usage_field(response: Any, name: str) -> Any:
        return getattr(getattr(response, "usage", None), name, None)

    async def complete_text(
        *,
        base_messages: list[dict[str, Any]],
        initial_content: str,
        initial_finish_reason: str | None,
        label: str,
    ) -> str:
        """把一次最终正文生成补完整；只续写，不重新润色已有内容。"""
        text = (initial_content or "").strip()
        finish_reason = initial_finish_reason
        working = list(base_messages)

        for attempt in range(MAX_COMPLETION_CONTINUATIONS + 1):
            if not completion_is_incomplete(text, finish_reason):
                break
            if attempt >= MAX_COMPLETION_CONTINUATIONS:
                logger.warning(
                    "agent_run %s still incomplete after %d continuations "
                    "(finish_reason=%s chars=%d)",
                    label,
                    MAX_COMPLETION_CONTINUATIONS,
                    finish_reason,
                    len(text),
                )
                break

            if text:
                working.append({"role": "assistant", "content": text})
            working.append(
                {"role": "user", "content": continuation_prompt(text)}
            )
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=working,
                    temperature=0.4,
                    max_tokens=continuation_max_tokens,
                    reasoning_effort=FINAL_REASONING_EFFORT,
                )
            except Exception as exc:
                logger.warning(
                    "agent_run %s continuation %d failed: %r",
                    label,
                    attempt + 1,
                    exc,
                )
                break

            choice = response.choices[0]
            addition = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
            text = merge_continuation(text, addition)
            logger.info(
                "agent_run %s continuation=%d finish_reason=%s "
                "completion_tokens=%s chars=%d",
                label,
                attempt + 1,
                finish_reason,
                _usage_field(response, "completion_tokens"),
                len(text),
            )
        return text

    async def finish(reason: str) -> dict[str, Any]:
        final_messages = [
            *messages,
            {"role": "user", "content": reason},
        ]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=final_messages,
                temperature=0.5,
                max_tokens=final_max_tokens,
                reasoning_effort=FINAL_REASONING_EFFORT,
            )
            final = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("agent_run final reply failed: %r", exc)
            return {
                "text": "我查完了，但最后整理答案时卡住了，要不换个问法再试一次？",
                "tool_calls_made": tool_calls_made,
            }
        finish_reason = _finish_reason(response)
        logger.info(
            "agent_run mode=%s final finish_reason=%s completion_tokens=%s chars=%d",
            mode,
            finish_reason,
            _usage_field(response, "completion_tokens"),
            len(final),
        )
        text = await complete_text(
            base_messages=final_messages,
            initial_content=final,
            initial_finish_reason=finish_reason,
            label="final",
        )
        if not text:
            text = "我查到了些资料，但这次没能整理成回答，要不我再试一次？"
        limit = max(0, int(ctx.max_output_chars or 0))
        if limit and len(text) > limit:
            logger.info(
                "agent_run long output chars=%d soft_budget=%d; "
                "leave full text for QQ chunked delivery",
                len(text),
                limit,
            )
        return {"text": text, "tool_calls_made": tool_calls_made}

    for _step in range(max_steps):
        if len(tool_calls_made) >= max_tool_calls:
            return await finish(
                "工具调用预算已经用完。不要再请求工具，直接基于上面已有结果完成最终回答。"
            )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.3,
                max_tokens=step_max_tokens,
                reasoning_effort=ctx.reasoning_effort,
            )
        except Exception as exc:
            logger.warning("agent_run LLM call failed at step %d: %r", _step + 1, exc)
            return {
                "text": "刚才查资料时出了点问题，要不换个说法再让我试一次？",
                "tool_calls_made": tool_calls_made,
            }

        msg = response.choices[0].message

        # 模型直接给了文本回复：它已经是在完整工具轨迹上收口。若这条正文被
        # token 上限截断，只续写缺失部分，不再交给第二个模型重新生成全文。
        if msg.content and not msg.tool_calls:
            finish_reason = _finish_reason(response)
            logger.info(
                "agent_run mode=%s direct-final finish_reason=%s "
                "completion_tokens=%s chars=%d",
                mode,
                finish_reason,
                _usage_field(response, "completion_tokens"),
                len(msg.content),
            )
            text = await complete_text(
                base_messages=messages,
                initial_content=msg.content,
                initial_finish_reason=finish_reason,
                label="direct-final",
            )
            return {
                "text": text,
                "tool_calls_made": tool_calls_made,
            }

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            remaining = max(0, max_tool_calls - len(tool_calls_made))
            executable = list(msg.tool_calls[: min(MAX_PARALLEL_TOOLS, remaining)])
            skipped = list(msg.tool_calls[len(executable):])
            parsed: list[tuple[Any, str, dict[str, Any]]] = []
            for tc in executable:
                func_name = str(tc.function.name or "")
                try:
                    loaded = json.loads(tc.function.arguments or "{}")
                    func_args = loaded if isinstance(loaded, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    func_args = {}
                parsed.append((tc, func_name, func_args))

            logger.info(
                "agent_run mode=%s step=%d tools=%s skipped=%d budget=%d/%d",
                mode,
                _step + 1,
                [name for _, name, _ in parsed],
                len(skipped),
                len(tool_calls_made) + len(parsed),
                max_tool_calls,
            )
            results = await asyncio.gather(
                *(
                    _execute_tool(func_name, func_args, ctx)
                    for _, func_name, func_args in parsed
                )
            )
            for (tc, func_name, _func_args), result_text in zip(parsed, results):
                tool_calls_made.append(func_name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text[:TOOL_RESULT_MAX_CHARS],
                    }
                )
            for tc in skipped:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "工具调用预算已用完，这次调用没有执行。"
                            "请不要再查新资料，基于已有结果完成回答。"
                        ),
                    }
                )
            if len(tool_calls_made) >= max_tool_calls:
                return await finish(
                    "工具调用预算已经用完。不要再请求工具，直接基于上面已有结果完成最终回答。"
                )
        else:
            return {
                "text": "抱歉，我没想好怎么回答，能换个说法吗？",
                "tool_calls_made": tool_calls_made,
            }

    return await finish(
        "模型回合已经用完。不要再请求工具，直接基于上面已有结果完成最终回答。"
    )
