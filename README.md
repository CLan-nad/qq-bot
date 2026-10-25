# 小毛豆

一个适合情侣私人群聊的 QQ 机器人，角色原型来自《洛克王国》的大耳帽兜。平时安静待在群里，被 @ 时才参与聊天。

## 能做什么

- **聊天**：接话、开新话题、总结刚才聊了什么
- **看图**：聊天截图、照片、表情包都能认；连发多张后 @ 它，会先看最近一批（默认最多 9 张），同一张图只识别一次
- **搜索与阅读**：联网搜索、总结网页，读 PDF / Word / Excel / 文本，算数或在受限沙箱里跑命令
- **聊天记录检索**：能按你的要求翻当前群的历史消息，找相关原话并结合记忆回答；例如「查一下我们之前聊火锅的那几条」。记录按群隔离，新群不会自动继承旧群数据
- **消息统计**：可以统计今天、最近几天、本周等范围内每个人发了多少条，由本地数据库直接聚合，不靠猜
- **提醒**：单次、每天、每周、每年、每年农历；重要日子可提前 N 天预热提醒
- **记忆**：明确说「记住 xx」才入库，可置顶（★）；「先记一下」只保留 14 天；日常闲聊只压成摘要，不会自己往记忆里塞
- **节日日历**：公历农历互查，「中秋是几号」「最近有什么节日」
- **主动关心**：没被 @ 时也能察觉「累了 / 心情不好 / 肚子疼」，接一句 25 字以内的关心
- **表情包**：8 张大耳帽兜表情图，先发文字再单独发图，一条回复最多一张
- **群内偏好**：@ 它说一声就能改，存在库里，优先级高于 `.env`

全部用自然语言，不用记命令：

```text
@小毛豆 总结一下我们刚才聊了什么
@小毛豆 每年 12 月 13 日 09:00 提醒我们纪念日，提前 3 天先提个醒
@小毛豆 每年农历冬月初五 08:30 给她过生日
@小毛豆 记住她的真名叫xx，一定要记住
@小毛豆 以后叫我「主人」，叫她「xx」
@小毛豆 中秋是几号
@小毛豆 搜一下周末上海有哪些展览
@小毛豆 查一下我们之前说火锅的那几条
@小毛豆 最近一周我们谁发得多
```

## 运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Windows 下也可以用 `scripts\start.ps1` / `status.ps1` / `stop.ps1`。
QQ 开放平台选 WebSocket 连接，并开启接收群消息。

联网搜索和网页抓取由 DeepSeek 提供，直接复用 `.env` 里的 `LLM_API_KEY`；没配好时会明确说查不了，不会编造结果。
模型思考强度默认 `low`（`MAODOU_REASONING_EFFORT`，可选 `none/low/high/max`）；图片识别失败时会自动升到 `high` 再试。

## 代码架构

小毛豆是一个单进程、全异步的 QQ 群机器人：QQ WebSocket 事件进来后，由 `bot.py` 跑完整条消息流水线，需要模型时调用 `llm.py`，需要查资料或翻记录时交给 `agent.py` 的工具循环，所有长期状态写在本地 SQLite。

```text
QQ 群消息
   │
   ▼
main.py                 装配配置 / 数据库 / 模型客户端 / botpy
   │
   ▼
maodou/bot.py          事件、去重、图片、上下文、意图分发、发送
   ├─ maodou/llm.py    提示词与模型调用
   ├─ maodou/agent.py  有界 function-calling 工具循环
   ├─ maodou/tools.py  搜索、网页、文档、shell、数学
   ├─ maodou/store.py  SQLite 持久层
   └─ maodou/topics.py 话题切分与消息渲染
```

### 模块职责

| 文件 | 职责 |
| --- | --- |
| `main.py` | 启动入口：读配置 → 打开数据库 → 创建模型客户端 → 启动 `MaodouClient`，退出时关闭数据库 |
| `maodou/config.py` | 把 `.env` 读成只读的 `Settings`；兼容旧的 `XIAOXIA_*` 配置名，但新配置统一用 `MAODOU_*` |
| `maodou/bot.py` | 最大的模块。QQ 事件适配、消息入库、图片批次、身份上下文、意图分发、本地 action、定时任务、主动关心、出站发送和后台循环都在这里 |
| `maodou/llm.py` | `MaodouLLM` 封装 OpenAI 兼容接口；集中管理聊天、意图解析、看图、摘要和关心判定的提示词与参数 |
| `maodou/agent.py` | 8 个工具的 JSON Schema、`AgentContext`、工具执行器和 `fast/deep` 两档有界循环 |
| `maodou/tools.py` | 工具的实际实现：DeepSeek 搜索/抓取、网页正文提取、PDF/Word/Excel 读取、受限 shell、数学计算 |
| `maodou/store.py` | SQLite 建表、增量迁移、消息检索、记忆卡、任务、称呼、群偏好、图片缓存等全部数据操作 |
| `maodou/topics.py` | 把最近消息切成【当前话题】和【更早的背景】，并统一渲染成「时间 + 【说话人】 + 正文」 |
| `maodou/images.py` | 解析 QQ 图片附件、下载并内联图片、清洗占位符、把看图结果与用户原话分开渲染 |
| `maodou/stickers.py` | 小表情名字映射，以及 8 张表情包的选择、摘取和出站处理 |
| `maodou/scheduling.py` | 单次 / 每天 / 每周 / 每年 / 农历每年的下次运行时间计算与文案 |
| `maodou/calendar.py` | 公历农历互转、固定节日和网络节日表、未来 30 天节日查询 |
| `maodou/care.py` | 主动关心的第一层关键词预筛；命中后才值得调用模型做第二层判断 |
| `scripts/` | 部署、回滚、换群迁移、记忆清理和服务器探针 |
| `tests/` | 全部离线单测；外部模型与 QQ 接口都在测试里替换掉 |

### 一次 @ 的完整链路

1. **收消息**：`on_group_message_create` 先把群消息落库，再判断有没有 @ 小毛豆；没 @ 就只走主动关心预筛。这样即使先发几句话、下一条才 @，前文也已经在数据库里。
2. **去重与补存**：`on_group_at_message_create` 或全量消息里的 mention 进入 `_process_at_message`；按 `message_id + msg_seq` 去重，避免 QQ 重推导致重复回答。
3. **处理图片**：只在本轮需要回复时识别图片。支持本条附件、向前回看连续的一批未识别图片，以及后台补识别旧图；同一张图按字节 SHA-256 缓存描述，不重复调用视觉模型。识图与回复并行，通常会先发一句「我看看图～」，正式答案随后用下一个 `msg_seq` 发出。
4. **组装上下文**：`_topic_context()` 从最近记录切出当前话题与背景，`_identity_context()` 绑定触发者、另一位成员、称呼和记录里的说话人标签。
5. **解析意图**：`llm.parse_intent()` 输出一个 JSON action。输入包含身份、当前话题、任务、节日表、长期记忆等；这里只做分类和抽取参数，不负责最终回复。
6. **执行 action**：任务、记忆、称呼、群偏好等本地操作走确定性 handler；`chat` 走 `generate_reply()`；`agent_run` 进入工具循环，拿到结果后再由 `generate_tool_reply()` 用小毛豆的语气整理。
7. **等待策略**：`chat` 和 `agent_run` 超过 `MAODOU_ACK_AFTER_SECONDS`（默认 6 秒）还没完成时，先用 `msg_seq=1` 发一句垫场，正式回复用 `msg_seq=2`。
8. **出站**：先处理小表情和表情包；文字用一条消息发送，表情图再单独发下一条。发送失败会重试，`msg_seq` 撞车会自动换槽，被动回复彻底失败时才退回主动消息。机器人自己的话也会入库，下一轮能看到。

### 上下文工程

普通聊天不会把整库记录一股脑塞给模型，而是按用途分层：

| 上下文块 | 来源 | 规则 |
| --- | --- | --- |
| 身份信息 | `IdentityContext` | 当前触发者、另一位成员、登记过的称呼、显示名与说话人标签的对应关系 |
| 【当前话题】 | `topics.split_topic()` | 从最新消息往回扫；相邻间隔超过 45 分钟、达到 40 条或跨度超过 12 小时就切断 |
| 本次要回应 | 当前这条 @ 消息 | 去掉 @ 和表情标记；看图结果单独标成〔图〕，不混进用户原话 |
| 【更早的背景】 | 上一话题末尾 | 最多 4000 字，只用来理解语气，不主动翻旧账 |
| 【最近聊天概要】 | `conversation_summaries` | 最近 2 段，总计最多 900 字；只由后台把日常闲聊压成摘要 |
| 【长期记忆】 | `memory_cards` | 只装稳定事实，按置顶、类型配额和字符预算选取，不混入对话摘要 |
| 表情 / 检索结果 / 本轮操作 | 运行时生成 | 明确告诉模型这轮到底做了什么，避免它声称「记住了」「查过了」但实际没执行 |

长期记忆不会从日常聊天自动抽取：只有群里明确说「记住 xx」才写 `memory_cards`；「先记一下」会带 14 天过期时间。后台摘要只是聊天梗概，和长期记忆是两套数据。

### 意图与工具

意图解析器共 16 个 action：

| 类别 | action | 执行方式 |
| --- | --- | --- |
| 定时任务 | `task_create`、`task_list`、`task_cancel`、`task_pause`、`task_resume`、`task_skip` | `bot.py` 本地 handler + SQLite |
| 长期记忆 | `memory_remember`、`memory_list`、`memory_pin`、`memory_unpin`、`memory_forget`、`memory_alias` | `bot.py` 本地 handler + SQLite |
| 身份与群设置 | `address_set`、`bot_pref` | 写称呼表 / 群偏好表，群偏好优先于 `.env` |
| 节日查询 | `calendar_query` | `calendar.py` 直接计算 |
| 工具任务 | `agent_run` | `agent.py` 的 function-calling 循环 |
| 普通聊天 | `chat` | `llm.generate_reply()` |

本地 action 先由代码算出可靠结果，再决定要不要让模型润色：成功确认可以换语气，但编号、报错和「匹配到多条」这类信息保持原样；模型若改写后丢了事实或把成功说成失败，会退回模板。

`agent_run` 内目前有 8 个工具：

| 工具 | 用途 |
| --- | --- |
| `web_search` | 联网搜索，返回标题、链接、来源和摘要 |
| `web_fetch` | 打开并阅读指定网页正文 |
| `document_read` | 读取 PDF / Word / Excel 等文档 |
| `shell_run` | 在服务器执行受限只读命令；默认禁止写操作和危险模式 |
| `math_eval` | 用 AST 校验后的表达式做数学计算 |
| `calendar_query` | 查某个节日或列出未来 30 天节日 |
| `message_stats` | 直接聚合数据库里的群消息数量 |
| `memory_search` | 同时搜索历史消息和长期记忆 |

循环使用 OpenAI 兼容的 `tools` 协议。`fast` 最多 3 步，`deep` 最多 8 步；模型不再调用工具、直接给出文本时就结束，步数耗尽后再补一次「根据结果作答」的调用。网页抓取会拦截私网地址，shell 有命令白名单、超时和输出上限，避免模型把服务器当成无限制执行环境。

> 旧版的 `_handle_web_search` / `_handle_web_fetch` / `_handle_document_read` / `_handle_shell_run` / `_handle_math_eval` / `_handle_memory_search` / `_handle_message_stats` 还留在 `bot.py` 里，但正常分发已经全部走 `agent_run`，不会再从这些旧入口进入。

### 数据与后台任务

数据库是单个 SQLite 文件，按群 `group_openid` 隔离数据。主要表：

| 表 | 内容 |
| --- | --- |
| `messages` | 群消息原文、发送者、时间、来源；另有 FTS5 全文索引用于历史检索 |
| `processed_at_messages` | @ 事件去重 |
| `memory_cards` | 长期记忆：内容、类型、人物、置顶、临时过期时间和来源消息 |
| `conversation_summaries` | 后台生成的聊天摘要，按消息 `row_id` 区间归档 |
| `scheduled_tasks` | 提醒与定时消息，包含周期、农历字段、提前天数和预热任务关联 |
| `person_aliases` / `person_addresses` | 同一个人的多种叫法，以及机器人该怎么称呼 |
| `group_preferences` | 每个群的主动关心和主动消息上限等设置 |
| `image_descriptions` | 图片字节指纹到文字描述的缓存 |
| `care_log` | 主动关心的发送记录与节流依据 |
| `face_tokens` | 群里见过的 QQ 小表情名字与标签 |

启动时 `store.py` 先建最终版 schema，再用 `PRAGMA table_info` 探测旧库缺列并 `ALTER TABLE`。只有约束本身变化（例如任务类型新增 `yearly`）时才整表重建，迁移过程不改写已有业务数据。

`on_ready` 后会启动一个后台循环：

- 每 `MAODOU_SCHEDULER_INTERVAL_SECONDS`（默认 30 秒）检查到期任务，发送后计算下一次运行时间。
- 每 `MAODOU_MEMORY_INTERVAL_SECONDS`（默认 300 秒）把攒够的消息压成摘要，并清理过期临时记忆。
- 未 @ 的消息先过 `care.py` 关键词预筛，再交给便宜模型判断要不要关心；同人冷却、每人每日上限、每群主动消息上限和 `care_log` 去重共同限流。

### 模型接口与扩展点

除 `agent_run` 使用原生 tool calling 外，其余结构化任务都用 JSON 模式和代码侧容错：

| 调用 | 用途 |
| --- | --- |
| `parse_intent()` | 自然语言 → 16 种 action 的 JSON |
| `generate_reply()` | 普通聊天，按上下文分层组装 prompt |
| `generate_tool_reply()` | 把工具结果整理成人设回复 |
| `describe_image()` | 视觉模型描述图片；失败自动重试并升到 `high` |
| `process_memory_batch()` | 聊天记录 → 150–300 字摘要，不产出长期记忆 |
| `judge_care()` | 主动关心的第二层判定，输出 `{"care": bool, "text": "..."}` |

扩展时通常只需要动这几处：

- **新增本地 action**：补 `INTENT_SYSTEM_PROMPT` 的 action 定义、`_handle_intent` 的分支和对应 handler。
- **新增 agent 工具**：在 `agent.py` 的 `AGENT_TOOLS` 加 schema，在 `_execute_tool` 加实现；真正的外部 I/O 放 `tools.py`。
- **新增数据字段**：同时修改 `_init_schema` 的最终建表语句和 `_ADDITIVE_COLUMNS`，保证新库与旧库迁移后的结构一致。
- **新增行为**：优先用提示词和上下文解决；只有需要确定性、权限控制或外部副作用时才写死在代码里。

## 部署

生产环境以 `maodou.service` 跑在 Linux 服务器的 `/opt/maodou`：

```powershell
.\scripts\deploy.ps1 -DryRun   # 先看要做什么，不动服务器
.\scripts\deploy.ps1 -Force    # fetch -> 跑测试 -> 备份数据库 -> 重启 -> 健康检查
.\scripts\rollback.ps1         # 出问题退回上一个已部署版本
```

部署会确认工作区干净且已 push，测试不过直接中止，健康检查失败自动回滚；版本记录在 `/var/lib/maodou-deploy/`。CI（GitHub Actions）在每次 push / PR 上跑 `compileall` 和全部单测。

换群后会拿到新的 `group_openid`，记忆等于清空；用 `scripts/migrate_group.py` 把旧群数据搬过去（加 `--list` 看群，`--dry-run` 预览，写库前自动留快照）。

## 隐私

聊天记录、图片描述和长期记忆都存在部署机器本地；生成回复时只把所需上下文发给配置的大模型服务。图片只在内存里识别，落盘的是内容指纹和一段文字描述，原图不保存。`.env` 已加入 `.gitignore`，密钥不会提交。

表情图由群主提供，版权归原权利人，仅供私人群内使用。
