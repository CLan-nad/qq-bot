from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from .calendar import format_festivals, lunar_label, next_festival, upcoming_festivals
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

AGENT_RUN_SYSTEM_PROMPT = """你是 QQ 情侣群里的“小毛豆”，外形和脾气取自洛克王国的大耳帽兜。你正在帮群友完成一个查询任务。

你有以下工具可以调用：
- web_search：搜索互联网，获取最新信息
- web_fetch：打开并阅读指定网页
- document_read：读取 PDF/Word/Excel 等文档
- shell_run：在服务器上执行只读命令，查看文件或系统状态
- math_eval：计算数学表达式
- calendar_query：查节日日期
- message_stats：统计群聊消息数量
- memory_search：搜索聊天记录和长期记忆

工作方式：
1. 先理解用户想要什么，想清楚需要哪些工具
2. 调用必要的工具，拿到结果
3. 如果结果不够，可以再调用一次；如果够了就给出最终回答
4. 最终回答要口语化、简洁、像小毛豆在说话，不要写成报告

规则：
- 结果与问题无关时必须直说没找到，不能编造
- 引用资料时写出来源站点名
- 不要输出「小毛豆：」、标题、代码块 Markdown
- 说话像熟络的死党，不像客服"""


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

    fast 模式最多 3 步（适合简单查询），deep 模式最多 8 步（适合复杂多步任务）。
    返回 {"text": "最终回复", "tool_calls_made": ["web_search", ...]}
    """
    max_steps = 3 if mode == "fast" else 8
    model = ctx.model or "deepseek-v4-pro"
    tool_calls_made: list[str] = []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_RUN_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for _step in range(max_steps):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=0.3,
                max_tokens=1200,
                reasoning_effort=ctx.reasoning_effort,
            )
        except Exception as exc:
            logger.warning("agent_run LLM call failed at step %d: %r", _step + 1, exc)
            return {"text": f"抱歉，我在处理时出了点问题（{exc}），要不换个说法再试试？", "tool_calls_made": tool_calls_made}

        msg = response.choices[0].message

        # 模型直接给了文本回复——任务完成
        if msg.content and not msg.tool_calls:
            return {"text": msg.content.strip(), "tool_calls_made": tool_calls_made}

        # 模型要调工具
        if msg.tool_calls:
            # 先把 assistant 消息加入历史
            messages.append({
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
            })

            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                logger.info("agent_run tool call: %s(%s)", func_name, func_args)
                tool_calls_made.append(func_name)

                result_text = await _execute_tool(func_name, func_args, ctx)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text[:4000],
                })
        else:
            # 既没有文本也没有工具调用——罕见，返回空
            return {"text": "抱歉，我没想好怎么回答，能换个说法吗？", "tool_calls_made": tool_calls_made}

    # 步数用尽，让模型给出最终回复
    try:
        messages.append({
            "role": "user",
            "content": "工具调用已经结束，请根据上面的结果给出最终回复。回答要口语、简洁，像小毛豆在说话。",
        })
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
            reasoning_effort=ctx.reasoning_effort,
        )
        final = response.choices[0].message.content or ""
        return {"text": final.strip(), "tool_calls_made": tool_calls_made}
    except Exception as exc:
        logger.warning("agent_run final reply failed: %r", exc)
        return {"text": "我查完了但组织语言时卡住了，要不换个问法？", "tool_calls_made": tool_calls_made}
