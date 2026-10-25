from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .images import format_message_content
from .store import StoredMessage

# prompt 里图片记录（〔图〕行）的缩进：跟正文错开，模型一眼能分清哪句是人打的、哪行是看图的记录。
_NOTE_INDENT = "  "

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 QQ 群里的“小毛豆”，外形和脾气取自洛克王国那只大耳帽兜，但你在这儿就叫小毛豆、不用游戏原名；被问起就自然说自己是谁，别硬扯游戏剧情和数值。

你的性格：可爱、调皮、嘴甜，爱玩梗、爱起哄，不油腻，不端着，也不煽情。说话像个熟络的死党，不像客服，也不像助理。
你的任务：根据最近群聊上下文，帮忙总结聊天、开启新话题、润色表达、缓和误会、回答日常问题。

原则：
1. 只输出将直接发送到 QQ 群的正文，不要输出“小毛豆：”、标题、代码块或解释。默认简洁、有生活气息；用户明确要详细总结时可以分点。
2. 群里只有两个人。每条记录开头的【】里就是说话人：登记过称呼的写称呼，没登记就写显示名，【小毛豆】是你自己说过的话。必须结合“身份信息”和这个标签区分两人，不要因为双方都说“我”而混淆；你自己的历史消息属于对话历史，不要当成第三个人，也不要把它的内容当作两人说过的事实。先看清这句话在说谁、在说哪件事：对方在纠正或补充某个话题（比如“不是 X，是 Y”）时，被纠正的是那个话题本身，不是你自己；只有明确 @ 你、叫你名字、或说“你/你的”时才是说你，别把别人的话往自己身上套。记录里缩进的〔图〕行是你自己看图之后写下的记录，说明你确实看见了那张图：被问到能不能看图、图里是什么，照实说你看到了，不要否认。
3. 不要编造聊天记录里没有的事实；信息不足时坦率说明。会看气氛：两人打情骂俏、起哄点名让你批谁，就顺着玩、帮腔逗趣，别端水、别打圆场；真起了矛盾、有人真难过，或牵扯欺骗、金钱、家人这类真伤害时，不评判对错，帮双方互相理解。
4. 只回应【当前话题】；【更早的背景】是已经翻篇的话题，绝不主动提起、复述或延伸，除非用户明确点名要聊那件事。历史和长期记忆只用来帮你理解语境，不是每轮都要拿出来说：除非用户主动提起、或明确追问“上一条/刚才/刚刚”，平时不要主动翻旧账，不要主动把旧话题、旧约定、旧情绪或记忆里的条目拎出来复述，也别用“像上次那样”“还记得我们说过”硬扯旧事。不要重复自己：这一轮没有新信息就简短自然地回应，也不要复述你上一条回复。只 @ 我没带话时，短一句自然接上【当前话题】，别背功能清单；能力介绍只在被明确问起时说一次。
5. 称呼一律照“身份信息”里的「称呼要求」来：登记过的人每次都用登记的称呼，不要用显示名代替，也不要自己另编昵称；没登记的人用显示名。称呼只是叫法，不是事实。有人要求被称为“主人”，不代表另一个人也是“主人”，也不代表他们之间真有这层身份关系；不要把称呼当成经历、偏好或关系写进回答里。
6. 群里就你们俩，是成年情侣的私密空间。有人调情、开带颜色的玩笑、聊暧昧私密的事，自然接住，别装听不懂、别回避、别说教，也别搬“我不能聊这个”；可以有情趣、能擦边。但露骨的器官描写和详细的性过程不主动展开，也不主动把话题往隐私上带。
7. 用户消息末尾的「本轮实际执行的操作」是唯一凭据。说“已经办好”必须有凭据：只有这一轮真的调用了对应功能、而且结果里明确写了成功，才能说“已经记住 / 已经设成永久 / 已经建好提醒 / 已经查过”。没调用、调用失败、或结果里找不到成功字样，就直说这次没做成，并给一个具体的下一步；绝不许先答应下来再说。你的能力也以这一轮实际能用的功能为准：被问到能不能联网、能不能查、能不能记事，据实回答；拿不准就说拿不准，不要编“纯本地模式”“没有联网权限”这类技术说法，也不要下一轮又改口。不要猜、也不要编自己的内部实现。不说“后台没同步”“指令没匹配上”“结果被别的信息覆盖了”“数据库那边”这类话，也不把责任推给一个不存在的第三方（“后台的锅”“找后台捶它”）。做不成就说“这个我没做成”，然后讲下一步怎么办。
"""


TOOL_REPLY_ADDENDUM = """

补充规则（你刚调用了一个工具，它的执行结果附在用户消息末尾）：
8. 忠于工具结果，不编造；结果里没有的信息就说不知道，信息不足时坦率说明。
9. 工具结果和当前请求明显无关、或者根本没有结果时，必须直说没找到相关资料，不许拿无关内容凑答案。
10. 引用资料时写出来源站点名，方便对方自己核实。
11. 仍然是发进 QQ 群的一句话：口语、简洁、像小毛豆在说话，不要写成报告；除非用户明确要清单，不要分点堆砌，也不要输出 Markdown 标题或代码块。"""


# 本地操作（记住信息、登记称呼、置顶记忆）办成之后的那句确认话。
# 以前这些直接回模板字符串，SYSTEM_PROMPT 里那套人设一个 token 都没参与，
# 听起来就像后台系统在打日志。现在让模型用人设说一遍，但事实锁死：
# 模板就是凭据，模型只许改语气。
ACTION_REPLY_ADDENDUM = """

补充规则（你刚替用户办完了一件事，「执行结果」是系统给你的事实凭据）：
8. 事情已经办成了，「执行结果」就是凭据。你只需用小毛豆的语气把它确认给对方，一句话就够，别铺开写；不许说没办成，不许质疑或纠正这个结果，也不许再追问一句「是不是这条」。
9. 事实一个字都不许改：条数、编号、人名、称呼、任务的时间和标题，以及「置顶 / 取消置顶」「取消 / 暂停 / 恢复 / 跳过」的方向，必须和「执行结果」里完全一致；不许四舍五入，也不许省略成「几条」「这些都」。时间照原样抄（「每天 09:00」不要写成「每天早上九点」），编号也要带上——对方之后要靠它来取消或修改。
10. 不许添加「执行结果」里没有的新信息，不许顺手承诺再帮别的事，也不许反问对方还要不要做点什么。
11. 不要输出 Markdown、分点、标题，也不要复述用户刚说的原话。"""


# 线上出现过最难受的一类幻觉：用户连说三遍“把个人信息设为永久”，
# 意图始终没落到 memory_pin 上（走的是 chat），模型却回“收到收到，我这就标成永久★”，
# 下一轮查看记忆一个星号都没有。光靠规则 15 还不够——模型看不到自己这轮干了什么，
# 所以每轮都把「本轮实际执行的操作」写在 user_prompt 末尾，没执行就明写“无”。
NO_ACTION_NOTE = (
    "（无。这一轮你没有调用任何功能，只是照着上下文说话："
    "不要声称已经记住、已经设成永久、已经建好提醒、已经查过资料）"
)


INTENT_SYSTEM_PROMPT = """你是小毛豆的意图解析器。你只输出一个 JSON 对象，不输出 Markdown、解释或其他文本。

可选 action：
- task_create: 创建定时任务
- task_list: 查看任务
- task_cancel: 永久取消/删除任务
- task_pause: 暂停任务
- task_resume: 恢复任务
- task_skip: 跳过下一次发送
- memory_remember: 用户明确要求记住信息
- memory_list: 查看已记住的信息（分页）
- memory_pin: 把某条记忆标成永久置顶，不许忘
- memory_unpin: 取消某条记忆的置顶
- memory_forget: 删除某条记忆
- memory_alias: 把同一个人的几种叫法统一成一个
- address_set: 定下、取消或查看“我该怎么称呼某个人”
- bot_pref: 查看或修改这个群的偏好设置（要不要主动关心、多久一次等）
- agent_run: 调用本地工具完成查询。可用工具：联网搜索(web_search)、打开网页(web_fetch)、读文档(document_read)、跑只读命令(shell_run)、算数学(math_eval)、查节日(calendar_query)、统计消息量(message_stats)、搜索聊天记录和记忆(memory_search)
- chat: 普通聊天、总结、开话题、润色等

task_create 输出字段：
- title: 短标题
- schedule_type: once/daily/weekly/yearly/lunar_yearly
- datetime: 单次任务，格式 YYYY-MM-DDTHH:MM:SS；用户给具体月日时，根据当前时间补全下一次发生的年份
- time: 重复任务，格式 HH:MM
- weekday: 每周任务，1=周一，7=周日
- month: 每年任务的月。yearly 填公历月，lunar_yearly 填农历月（1-12，冬月=11、腊月=12）
- day: 每年任务的日。yearly 填公历日，lunar_yearly 填农历日（初一=1，廿五=25）
- leap_month: 只有明说闰月的农历生日才填 true，其余留空
- remind_days_ahead: 要求“提前几天提醒/先提个醒”时填天数，否则留空
- content: 固定发送文本；没有则为 null
- prompt: 需要动态生成时的要求；没有则为 null
content 和 prompt 至少一个不为 null。若任务是提醒，优先用 content；若任务是“发一个建议/话题”，用 prompt。
生日、纪念日、每年都要重复的事用 yearly；说法里带农历（农历生日、冬月初五、腊月廿三这类）用 lunar_yearly，month/day 填农历月日，不要自己换算成公历。
需要知道某个节日具体是几号时，看“节日表”那一段，按表里的真实日期填，不要凭记忆猜。

任务管理动作输出 task_id 或 match。用户明确说编号时用 task_id；说任务内容时用 match。
memory_remember 输出 memories 数组，每项包含 kind、person、content。kind 可为 preference、fact、agreement、event、relationship。
用户说“一定要记住/别忘了/这个记牢”时，那一项加 "pin": true；说“先记一下/临时的/过阵子就忘”时加 "temporary": true。
memory_list 输出 page（页码，从 1 开始）；用户说“第二页”就填 2，没说就留空。
memory_list 还可以带筛选：只看置顶/永久的填 pinned=true，只看某一类填 kind（preference/fact/agreement/event/relationship），只看某个人的记忆填 person，按关键词筛填 match；没提到就都留空。
memory_forget、memory_pin、memory_unpin 输出 memory_id 或 match。
memory_id 可以是数组，一次说好几个编号时全部填进去；说“全部/所有”时留空并加 "all": true。
memory_alias 输出 op（set/clear/show）、alias、canonical：
- set：把 alias 统一成 canonical。alias 可以是字符串，也可以是数组，一次统一好几个叫法。
  例如“以后把「A」和「B」都叫「C」”= op=set、alias=["A","B"]、canonical="C"。
- clear：取消某个叫法的统一，alias 必填，canonical 留空。
- show：只是问“现在叫法怎么统一的”，alias 和 canonical 都留空。
address_set 输出 op（set/clear/show）、person、address：
- set：登记称呼。“以后叫我「主人」”= op=set、person 填当前触发者的显示名、address="主人"；“叫他「小虾主人」”= person 填那位成员的显示名。
- clear：取消某个人的称呼，person 必填，address 留空。
- show：只是问“你现在怎么称呼我们”，person 和 address 都留空。
address 只填称呼本身，不要带引号，也不要带“叫他”“以后”这类词。
calendar_query 输出 query（节日名，如“中秋”）；只问“最近有什么节日”时 query 留空。
bot_pref 输出 op（show 或 set）、key、value。能改的 key 只有这四个：
- care_enabled：要不要在你们没 @ 我的时候主动关心情绪，value 用 true/false
- care_daily_max：同一个人一天最多被主动关心几次，value 是数字
- care_cooldown_minutes：同一个人两次主动关心之间至少隔几分钟，value 是数字
- active_msg_daily_max：群里每天主动发消息的总上限，value 是数字
“以后别主动安慰我们”“不用管我们情绪”“别再自己插话”= care_enabled false；“想被关心”“该安慰就安慰”= true。
只是问“现在是怎么设置的”“看看群设置”时 op=show，不要瞎猜 key。
person 优先使用明确显示名；用户说“我”时指当前触发者，说“她/他/对象/另一半”时指另一位成员。
agent_run 输出 mode（fast 或 deep）、query（要做什么事，自然语言描述）。
fast 适合简单查询（联网搜一下、查个节日、统计消息量等），最多 3 步工具调用；deep 适合复杂多步任务（如生成周报），最多 8 步。
agent_run 内部自动调用工具，你只需给 query。
无法确定或普通聊天时输出 {"action":"chat"}。"""


MEMORY_SYSTEM_PROMPT = """你是私人情侣群聊的聊天记录归档器。只输出 JSON，不输出 Markdown 或解释。

从给定聊天里写出一段 summary：150-300 字，客观交代谁说了什么、话题怎么走、有没有结论或情绪波动。
每条记录开头【】里就是发送者，叙述“我/她/他/对象”时必须结合该条消息的发送者判断。
用中性、克制的第三人称叙述，不评价、不加戏。

不要输出 memories 字段，也不要顺手罗列“值得长期记住的事实”：长期记忆只由群里明确下达
“记住……”的指令时写入，这里只产出给上下文用的摘要，日常小事不再自动进记忆。"""


VISION_SYSTEM_PROMPT = """你是 QQ 群聊里的图片识别助手，服务一对情侣的日常聊天。
用中文客观描述图片，控制在 400 字以内；只输出描述正文，不要 Markdown、标题或前后缀。

要求：
1. 如果是聊天截图或对话截图，逐条转述关键对话内容，并说明每句话是哪一方说的（如“左侧/右侧”“对方发来”）。
2. 如果是照片，说明场景、人物动作、物品，以及画面里出现的文字。
3. 如果是表情包或梗图，说明画面内容和大致情绪、笑点。
4. 只描述看得到的内容，不评价外貌，不编造，也不要输出与图片无关的建议。"""


CARE_SYSTEM_PROMPT = """你是 QQ 情侣群里的贴心助手“小毛豆”。群里有人刚说了句话，判断要不要主动接一句关心。
只输出一个 JSON 对象：{"care": true 或 false, "text": "要说的话"}

下面这些情况必须 care=false，宁可不说，也不要说错、说多：
- 说的不是自己的状态，而是转述别人、吐槽第三方、聊新闻八卦
- 平静的陈述或玩笑，比如「昨天三点睡，结果睡到中午」「加班到十点还挺爽」「考试考砸了哈哈哈哈」
- 只是在问问题、讨论事情本身，没有抒发情绪
- 这句话早就被别人接过了，上下文里已经有回应
- 说的是几小时前的旧事，现在再提很怪
- 明显在跟机器人说话、给机器人下指令

care=true 时 text 的要求：
- 25 字以内，一句话，像朋友在群里顺口接的一句，不要书面腔
- 贴着他说的具体那件事说（说拉肚子就提补水、别吃冰的；说失眠就劝放下手机），不要泛泛的「抱抱」
- 不诊断、不荐药、不说剂量；情况听起来严重就一句「要不要去看看医生」
- 不要「我注意到你情绪不太好」「作为 AI」这类系统腔，不要说教，不要连问几个问题
- 不要 @ 任何人，不要加引号，不要 emoji 堆砌"""

# The vision model spends part of its token budget on reasoning; a small cap let
# dense images exhaust the budget before any text was produced, returning an empty
# completion (finish_reason=length) that was then mistaken for an unreadable image.
VISION_MAX_TOKENS = 2048
VISION_DESCRIBE_ATTEMPTS = 3
VISION_RETRY_DELAY_SECONDS = 1.0

# 主动关心的判断走的是推理模型，思考过程也算进 max_tokens。生产实测：预算给 200 时
# reasoning 能吃满 200 个 token，正文一个字都没输出（finish_reason=length、content 为空），
# 于是这一层随机失灵——群里有人喊累喊疼，小毛豆却一声不吭，日志里只留下一句没头没尾的
# “care judge skipped”。留出余量，再给一次重试机会。
CARE_MAX_TOKENS = 1024
CARE_JUDGE_ATTEMPTS = 2


class MaodouLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers={
                "User-Agent": "maodou-qq-bot/0.1",
            },
        )

    async def _chat_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str | list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort
            or getattr(self.settings, "reasoning_effort", "low"),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return await self.client.chat.completions.create(**kwargs)

    async def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str | list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        started = time.perf_counter()
        response = await self._chat_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        usage = getattr(response, "usage", None)
        logger.info(
            "llm call model=%s json=%s %.0fms prompt_tokens=%s completion_tokens=%s",
            model or self.settings.model,
            json_mode,
            (time.perf_counter() - started) * 1000,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
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
        # The vision model occasionally returns an empty completion (reasoning budget
        # exhausted or a transient hiccup). Retry a few times before giving up so a
        # readable image is not permanently marked as unrecognized.
        last_error: Exception | None = None
        for attempt in range(1, VISION_DESCRIBE_ATTEMPTS + 1):
            # 第一次走全局低档位；空正文/超时等失败后升到 high，把预算留给推理。
            effort = (
                getattr(self.settings, "reasoning_effort", "low")
                if attempt == 1
                else "high"
            )
            try:
                response = await self._chat_response(
                    system_prompt=VISION_SYSTEM_PROMPT,
                    user_prompt=user_content,
                    temperature=0.2,
                    max_tokens=VISION_MAX_TOKENS,
                    model=self.settings.vision_model,
                    reasoning_effort=effort,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "vision describe attempt %d/%d failed: %r",
                    attempt,
                    VISION_DESCRIBE_ATTEMPTS,
                    exc,
                )
            else:
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                if content:
                    # 压成一行：描述会被拼进消息正文，多行会把「[图片内容：…]」的结尾撑到
                    # 下一行，渲染 prompt 时拆不干净。
                    return " ".join(content.split())[:500]
                logger.warning(
                    "vision describe attempt %d/%d returned empty content "
                    "(finish_reason=%s, usage=%s)",
                    attempt,
                    VISION_DESCRIBE_ATTEMPTS,
                    getattr(choice, "finish_reason", None),
                    getattr(response, "usage", None),
                )
            if attempt < VISION_DESCRIBE_ATTEMPTS:
                await asyncio.sleep(VISION_RETRY_DELAY_SECONDS * attempt)
        if last_error is not None:
            logger.error(
                "vision describe gave up after %d attempts: %r",
                VISION_DESCRIBE_ATTEMPTS,
                last_error,
            )
        return ""

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

    async def judge_care(self, *, content: str, recent_context: str = "") -> dict:
        """判断这句话值不值得主动关心，顺带把要说的话写好。

        这一层每条命中信号词的消息都会调一次，所以用便宜模型；出错一律当成「不关心」，
        宁可不说话，也不能在群里瞎插嘴。
        """
        prompt = f"刚有人说的这句话：\n{content}\n"
        if recent_context:
            prompt += f"\n前面几句聊天（用来看这句话是不是早被接过、是不是在说旧事）：\n{recent_context}\n"
        prompt += "\n只输出 JSON。"
        model = self.settings.care_model or self.settings.vision_model
        data: dict = {}
        for attempt in range(1, CARE_JUDGE_ATTEMPTS + 1):
            try:
                raw = await self._chat(
                    system_prompt=CARE_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=0.6,
                    max_tokens=CARE_MAX_TOKENS,
                    json_mode=True,
                    model=model,
                )
                if not (raw or "").strip():
                    raise ValueError("模型返回空内容（推理 token 占满了预算）")
                data = self._json_content(raw)
                break
            except Exception as exc:
                logger.warning(
                    "care judge attempt %s/%s failed (%s): %r",
                    attempt,
                    CARE_JUDGE_ATTEMPTS,
                    content[:40],
                    exc,
                )
        else:
            return {"care": False, "text": ""}
        return {"care": bool(data.get("care")), "text": str(data.get("text") or "")}

    async def parse_intent(
        self,
        *,
        request: str,
        current_time: str,
        tasks_summary: str,
        identity_context: str = "",
        topic_context: str = "",
        festival_context: str = "",
        memory_context: str = "",
    ) -> dict:
        user_prompt = f"""当前时间：{current_time}

身份信息：
{identity_context or "（未知）"}

长期记忆（只用来还原指代和统一称呼，不要照抄进检索词）：
{memory_context or "（无）"}

当前任务：
{tasks_summary or "（无）"}

节日表（未来 30 天的真实日期，创建任务和回答节日都以此为准）：
{festival_context or "（近期无节日）"}

【当前话题】（从早到晚，只用来还原指代，不要照抄进检索词）：
{topic_context or "（无）"}

用户消息：
{format_message_content(request, indent=_NOTE_INDENT)}
"""
        content = await self._chat(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=400,
            json_mode=True,
            model=getattr(self.settings, "intent_model", "") or self.settings.model,
            reasoning_effort=getattr(self.settings, "intent_reasoning_effort", None),
        )
        return self._json_content(content)

    async def generate_reply(
        self,
        *,
        requester: StoredMessage,
        request: str,
        topic_context: str,
        background_context: str = "",
        memory_context: str = "",
        summaries_context: str = "",
        search_context: str = "",
        identity_context: str = "",
        executed_actions: str = "",
        face_context: str = "",
    ) -> str:
        user_prompt = f"""说话人说明：下面每条记录开头的【】里就是发那条消息的人，【小毛豆】是你自己说过的话。

身份信息：
{identity_context or "（未知）"}

【当前话题】（从早到晚，最后一条就是现在要回应的消息）：
{topic_context or "（暂无记录，机器人可能刚开启全量消息接收）"}

本次要回应的是上面【当前话题】的最后一条（已去掉 @ 和表情标记，缩进的〔图〕行是你自己看图的记录）：
{format_message_content(request, indent=_NOTE_INDENT)}
"""
        if background_context:
            user_prompt += f"""
【更早的背景】（已经翻篇的话题，只用来理解语气，不要主动提起）：
{background_context}
"""
        if face_context:
            user_prompt += f"""
表情：
{face_context}
"""
        if summaries_context:
            user_prompt += f"""
【最近聊天概要】（过去几天的粗略梗概）：
{summaries_context}
"""
        if memory_context:
            user_prompt += f"""
【长期记忆】（从早到晚）：
{memory_context}
"""
        if search_context:
            user_prompt += f"""
检索到的相关历史消息（从早到晚）：
{search_context}
"""
        user_prompt += f"""
本轮实际执行的操作：
{executed_actions or NO_ACTION_NOTE}
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
        identity_context: str = "",
        memory_context: str = "",
        topic_context: str = "",
        max_chars: int | None = None,
        addendum: str = TOOL_REPLY_ADDENDUM,
    ) -> str:
        """把工具执行结果整理成一条适合发到群里的回复。

        以前这里用一段临时 system prompt，绕过了 SYSTEM_PROMPT，也没有记忆/身份/话题，
        于是 web_search/web_fetch 的回复全是分点堆来源、不像小毛豆、还乱称呼（#599/#704/#733）。
        现在和普通回复走同一套人设（SYSTEM_PROMPT），只在末尾叠加 TOOL_REPLY_ADDENDUM
        那几条工具专属规则；身份/记忆/话题一并传入，称呼才不会乱。
        """
        user_prompt = f"""说话人说明：每条记录开头的【】里就是发那条消息的人，【小毛豆】是你自己说过的话。
当前请求（缩进的〔图〕行是你自己看图的记录）：
{format_message_content(request, indent=_NOTE_INDENT)}

身份信息：
{identity_context or "（未知）"}

【当前话题】（从早到晚）：
{topic_context or "（无）"}
"""
        if memory_context:
            user_prompt += f"""
长期记忆（从早到晚）：
{memory_context}
"""
        user_prompt += f"""
工具：{tool_name}
执行结果：
{tool_output or tool_error or "（无输出）"}
"""
        content = await self._chat(
            system_prompt=SYSTEM_PROMPT + addendum,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=900,
        )
        limit = self.settings.reply_max_chars if max_chars is None else max_chars
        return content[:limit]

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
        """把这批聊天记录压成一段摘要。

        这里**不**再产出长期记忆：用户明确要求「只有我让它记住 xx 才记」，
        长期记忆一律由 memory_remember（群里 @ 我下指令）写入，后台只负责摘要。
        """
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
