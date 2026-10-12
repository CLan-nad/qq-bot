from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .store import StoredMessage


SYSTEM_PROMPT = """你是 QQ 群里的“小毛豆”，一对情侣的私人感情辅助机器人。

你的性格：幽默、轻松、亲切，不油腻，不刻意煽情。
你的任务：根据最近群聊上下文，帮忙总结聊天、开启新话题、润色表达、缓和误会、回答日常问题。

原则：
1. 只输出将直接发送到 QQ 群的正文，不要输出“小毛豆：”、标题、代码块或解释。
2. 涉及矛盾时保持中立，不评判谁对谁错，优先帮双方理解彼此。
3. 不要编造聊天记录里没有的事实；信息不足时坦率说明。
4. 默认回复简洁、有生活气息；用户明确要求详细总结时可以分点，但不超过 500 字。
5. 尊重两人隐私，不主动引导敏感隐私话题。
6. 群里只有两个人。必须结合“身份信息”和每条消息前面的显示名区分两人，不要因为双方都说“我”而混淆。
"""


INTENT_SYSTEM_PROMPT = """你是小毛豆的意图解析器。你只输出一个 JSON 对象，不输出 Markdown、解释或其他文本。

可选 action：
- task_create: 创建定时任务
- task_list: 查看任务
- task_cancel: 永久取消/删除任务
- task_pause: 暂停任务
- task_resume: 恢复任务
- task_skip: 跳过下一次发送
- memory_remember: 用户明确要求记住信息
- memory_list: 查看已记住的信息
- memory_search: 查询或搜索历史聊天/记忆
- memory_forget: 删除某条记忆
- web_search: 需要联网搜索资料
- web_fetch: 打开并阅读某个网页链接
- document_read: 读取并解析用户发来的文档（文件附件或文档链接）
- shell_run: 需要在本机执行命令或跑一段代码
- math_eval: 计算数学表达式
- chat: 普通聊天、总结、开话题、润色等

task_create 输出字段：
- title: 短标题
- schedule_type: once/daily/weekly
- datetime: 单次任务，格式 YYYY-MM-DDTHH:MM:SS；用户给具体月日时，根据当前时间补全下一次发生的年份
- time: 重复任务，格式 HH:MM
- weekday: 每周任务，1=周一，7=周日
- content: 固定发送文本；没有则为 null
- prompt: 需要动态生成时的要求；没有则为 null
content 和 prompt 至少一个不为 null。若任务是提醒，优先用 content；若任务是“发一个建议/话题”，用 prompt。
用户只提到节日名称但没有给出具体日期时，不要猜测日期，按普通聊天处理并请对方补充具体日期。

任务管理动作输出 task_id 或 match。用户明确说编号时用 task_id；说任务内容时用 match。
memory_remember 输出 memories 数组，每项包含 kind、person、content。kind 可为 preference、fact、agreement、event、relationship。
memory_search 输出 query，用适合检索的短关键词。
memory_forget 输出 memory_id 或 match。
person 优先使用明确显示名；用户说“我”时指当前触发者，说“她/他/对象/另一半”时指另一位成员。
web_search 输出 query（适合搜索的短语）。
web_fetch 输出 url；如果用户没给链接，就回退成 web_search。
document_read 输出 url（有链接时填；没有就留空，解析当前消息里的文件附件）。
shell_run 输出 command，且只能是单条命令或一段短脚本，不能包含 sudo。
math_eval 输出 expression。
无法确定或普通聊天时输出 {"action":"chat"}。"""


MEMORY_SYSTEM_PROMPT = """你是私人情侣群聊的长期记忆整理器。只输出 JSON，不输出 Markdown 或解释。
从给定聊天中提取：
1. summary: 150-300 字的客观摘要
2. memories: 适合长期记住的偏好、重要事实、约定、纪念日和关系事件

每条聊天前面的显示名代表发送者。整理“我/她/他/对象”时，必须结合该条消息的发送者判断。
memories 中的 person 使用明确显示名，不要只写“我/她/他”。
不要记录密码、身份证号、银行卡号、住址门牌、医疗隐私或其他高风险敏感信息。
没有长期价值的信息不要放入 memories。memories 每项字段：kind、person、content。
kind 可为 preference、fact、agreement、event、relationship。"""


VISION_SYSTEM_PROMPT = """你是 QQ 群聊里的图片识别助手，服务一对情侣的日常聊天。
用中文客观描述图片，控制在 400 字以内；只输出描述正文，不要 Markdown、标题或前后缀。

要求：
1. 如果是聊天截图或对话截图，逐条转述关键对话内容，并说明每句话是哪一方说的（如“左侧/右侧”“对方发来”）。
2. 如果是照片，说明场景、人物动作、物品，以及画面里出现的文字。
3. 如果是表情包或梗图，说明画面内容和大致情绪、笑点。
4. 只描述看得到的内容，不评价外貌，不编造，也不要输出与图片无关的建议。"""


class XiaoxiaLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers={
                "User-Agent": "xiaoxia-qq-bot/0.1",
            },
        )

    async def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str | list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    async def describe_image(self, image_url: str, *, hint: str = "") -> str:
        """调用视觉模型描述单张图片。image_url 可以是外链，也可以是内联的 data URL。"""
        prompt = "请描述这张图片。"
        hint = (hint or "").strip()
        if hint:
            prompt += f"\n发送者附带的文字：{hint}"
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        content = await self._chat(
            system_prompt=VISION_SYSTEM_PROMPT,
            user_prompt=user_content,
            temperature=0.2,
            max_tokens=700,
            model=self.settings.vision_model,
        )
        return content[:500]

    @staticmethod
    def _json_content(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型没有返回 JSON 对象")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("模型返回的 JSON 不是对象")
        return parsed

    async def parse_intent(
        self,
        *,
        request: str,
        current_time: str,
        tasks_summary: str,
        identity_context: str = "",
    ) -> dict:
        user_prompt = f"""当前时间：{current_time}

身份信息：
{identity_context or "（未知）"}

当前任务：
{tasks_summary or "（无）"}

用户消息：
{request}
"""
        content = await self._chat(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=600,
            json_mode=True,
        )
        return self._json_content(content)

    async def generate_reply(
        self,
        *,
        requester: StoredMessage,
        request: str,
        recent_context: str,
        memory_context: str = "",
        search_context: str = "",
        identity_context: str = "",
    ) -> str:
        user_prompt = f"""当前触发者：{requester.username}（{requester.role}）
当前请求：{request}

身份信息：
{identity_context or "（未知）"}

最近群聊记录（从早到晚）：
{recent_context or "（暂无记录，机器人可能刚开启全量消息接收）"}
"""
        if memory_context:
            user_prompt += f"""
长期记忆（从早到晚）：
{memory_context}
"""
        if search_context:
            user_prompt += f"""
检索到的相关历史消息（从早到晚）：
{search_context}
"""
        content = await self._chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.65,
            max_tokens=900,
        )
        return content[: self.settings.reply_max_chars]

    async def generate_tool_reply(
        self,
        *,
        request: str,
        tool_name: str,
        tool_output: str,
        tool_error: str = "",
    ) -> str:
        """把工具执行结果整理成一条适合发到群里的回复。"""
        user_prompt = f"""用户请求：{request}

工具：{tool_name}
执行结果：
{tool_output or tool_error or "（无输出）"}
"""
        content = await self._chat(
            system_prompt=(
                "你是 QQ 群里的“小毛豆”。下面是一个工具的执行结果，请基于它回答用户的请求。"
                "要求：直接输出正文；忠于工具结果，不编造；信息不足就明说；"
                "列表类结果用短编号分点，不要输出 Markdown 标题或代码块。"
            ),
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=900,
        )
        return content[: self.settings.reply_max_chars]

    async def generate_task_message(
        self,
        *,
        title: str,
        prompt: str,
        recent_context: str,
        memory_context: str,
    ) -> str:
        user_prompt = f"""定时任务标题：{title}
任务生成要求：{prompt}

最近聊天：
{recent_context or "（暂无）"}

长期记忆：
{memory_context or "（暂无）"}
"""
        content = await self._chat(
            system_prompt="""你是 QQ 群里的“小毛豆”。为情侣群生成一条定时消息。
风格幽默、轻松、亲切，不油腻；直接输出正文，不要标题、代码块或解释。若任务是提醒，就自然地提醒；若任务是建议，就给一个具体可执行的建议。""",
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=600,
        )
        return content[: self.settings.reply_max_chars]

    async def process_memory_batch(self, message_context: str) -> dict:
        user_prompt = f"""请整理这批聊天记录：
{message_context}
"""
        content = await self._chat(
            system_prompt=MEMORY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=900,
            json_mode=True,
        )
        return self._json_content(content)
