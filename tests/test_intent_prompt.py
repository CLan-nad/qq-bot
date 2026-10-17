"""提示词约束：搜索能不能用，取决于意图模型写出的检索词质量。

实测过：用“权威医学资料 依据 来源”去搜，回来的全是课件垃圾；用“拉肚子 肚子疼 原因”
才是对的。所以这几条约束必须留在提示词里，并且有测试盯着，不然以后被顺手删掉就复发。
"""
from __future__ import annotations

import inspect
import unittest

from xiaoxia.llm import (
    CARE_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    XiaoxiaLLM,
)
from xiaoxia.bot import XiaoxiaClient


class IntentPromptTests(unittest.TestCase):
    def test_web_search_trigger_words(self) -> None:
        for signal in ("查一下", "来源", "出处", "是不是真的"):
            self.assertIn(signal, INTENT_SYSTEM_PROMPT)

    def test_query_must_be_self_contained(self) -> None:
        self.assertIn("query 要自包含", INTENT_SYSTEM_PROMPT)
        self.assertIn("【当前话题】", INTENT_SYSTEM_PROMPT)

    def test_form_of_answer_words_are_banned_from_query(self) -> None:
        self.assertIn("严禁把用户对回答形式的要求写进 query", INTENT_SYSTEM_PROMPT)
        for banned in ("要有依据", "找权威来源", "别瞎编"):
            self.assertIn(banned, INTENT_SYSTEM_PROMPT)

    def test_falls_back_to_chat_when_query_unclear(self) -> None:
        self.assertIn("不要硬猜检索词", INTENT_SYSTEM_PROMPT)

    def test_parse_intent_takes_topic_context(self) -> None:
        params = inspect.signature(XiaoxiaLLM.parse_intent).parameters
        self.assertIn("topic_context", params)

    def test_parse_intent_takes_memory_context(self) -> None:
        # 意图解析也要看长期记忆，才能把指代补全、把称呼统一，写出自包含的检索词。
        params = inspect.signature(XiaoxiaLLM.parse_intent).parameters
        self.assertIn("memory_context", params)
        self.assertIn("长期记忆", inspect.getsource(XiaoxiaLLM.parse_intent))

    def test_generate_reply_splits_topic_and_background(self) -> None:
        params = inspect.signature(XiaoxiaLLM.generate_reply).parameters
        self.assertIn("topic_context", params)
        self.assertIn("background_context", params)

    def test_generate_reply_takes_summaries_context(self) -> None:
        params = inspect.signature(XiaoxiaLLM.generate_reply).parameters
        self.assertIn("summaries_context", params)

    def test_memory_context_is_facts_only(self) -> None:
        # 摘要不能再混进「长期记忆」，否则模型把剧情当人设（“主人多次骂我”）。摘要单独走
        # _summaries_context，长期记忆只剩记忆卡。
        source = inspect.getsource(XiaoxiaClient._memory_context)
        self.assertNotIn("[摘要]", source)
        self.assertIn("format_memory_cards", source)
        self.assertIn("_summaries_context", inspect.getsource(XiaoxiaClient))

class AnniversaryPromptTests(unittest.TestCase):
    """生日/纪念日/农历这套要靠提示词里的字段说明才能真正被解析出来。"""

    def test_yearly_schedules_are_described(self) -> None:
        for field in ("yearly", "lunar_yearly", "month", "day", "leap_month", "remind_days_ahead"):
            self.assertIn(field, INTENT_SYSTEM_PROMPT)
        self.assertIn("冬月", INTENT_SYSTEM_PROMPT)

    def test_no_longer_refuses_to_guess_festival_dates(self) -> None:
        # 现在会把未来 30 天的真实节日表喂给意图模型，旧的那条“别猜，让人补日期”要删掉，
        # 否则“除夕晚上提醒我们吃饺子”永远建不成任务。
        self.assertNotIn("不要猜测日期", INTENT_SYSTEM_PROMPT)
        self.assertIn("节日表", INTENT_SYSTEM_PROMPT)

    def test_memory_actions_are_described(self) -> None:
        for action in ("memory_pin", "memory_unpin", "memory_alias", "calendar_query"):
            self.assertIn(action, INTENT_SYSTEM_PROMPT)
        for field in ('"pin": true', '"temporary": true', "memory_list 输出 page"):
            self.assertIn(field, INTENT_SYSTEM_PROMPT)
        # memory_alias 要说清 op/alias/canonical，且 alias 可为数组一次统一多个
        for field in ("canonical", 'alias=["A","B"]'):
            self.assertIn(field, INTENT_SYSTEM_PROMPT)

    def test_parse_intent_receives_festival_context(self) -> None:
        params = inspect.signature(XiaoxiaLLM.parse_intent).parameters
        self.assertIn("festival_context", params)
        self.assertIn("节日表", inspect.getsource(XiaoxiaLLM.parse_intent))

    def test_memory_batch_only_summarizes(self) -> None:
        # 用户要求：只有明确说「记住 xx」才写长期记忆。后台归档只出摘要。
        params = inspect.signature(XiaoxiaLLM.process_memory_batch).parameters
        self.assertNotIn("existing", params)
        self.assertNotIn("recent_context", params)
        self.assertIn("不要输出 memories", MEMORY_SYSTEM_PROMPT)
        self.assertNotIn("memories: 适合长期记住", MEMORY_SYSTEM_PROMPT)

    def test_compaction_never_writes_memory_cards(self) -> None:
        # bot 层也要守住：_compact_memories 里不能再出现自动落记忆卡的做法。
        source = inspect.getsource(XiaoxiaClient._compact_memories)
        self.assertIn("memories=[]", source)
        self.assertNotIn("existing_memory_lines", source)


class SystemPromptTests(unittest.TestCase):
    def test_rule_background_topic_must_not_be_raised(self) -> None:
        self.assertIn("【更早的背景】", SYSTEM_PROMPT)
        self.assertIn("绝不主动提起", SYSTEM_PROMPT)

    def test_rule_health_clause_is_gone(self) -> None:
        # 用户明确要求去掉「健康不下诊断/不荐药」那条，交给模型自己拿捏
        self.assertNotIn("不下诊断结论", SYSTEM_PROMPT)
        self.assertNotIn("不推荐具体药品和剂量", SYSTEM_PROMPT)

    def test_rule_length_cap_is_gone(self) -> None:
        # 「不超过 500 字」被要求去掉：需要长回答时别再自我截断
        self.assertNotIn("不超过 500 字", SYSTEM_PROMPT)

    def test_rule_playful_instead_of_dead_neutral(self) -> None:
        # 打情骂俏、起哄点名让批谁，要顺着玩，不许端水
        self.assertIn("打情骂俏", SYSTEM_PROMPT)
        self.assertIn("顺着玩", SYSTEM_PROMPT)
        self.assertIn("别端水", SYSTEM_PROMPT)
        # 但真矛盾时的中立限定必须留着，否则变成无条件帮腔、火上浇油
        self.assertIn("真起了矛盾", SYSTEM_PROMPT)
        self.assertIn("不评判对错", SYSTEM_PROMPT)

    def test_rule_intimacy_for_the_couple(self) -> None:
        # 情侣私密调情要配合，不能保守地闭口不提
        self.assertIn("成年情侣的私密空间", SYSTEM_PROMPT)
        self.assertIn("别装听不懂", SYSTEM_PROMPT)
        self.assertIn("别回避", SYSTEM_PROMPT)
        self.assertIn("能擦边", SYSTEM_PROMPT)
        # 安全阀：露骨内容不主动展开
        self.assertIn("不主动展开", SYSTEM_PROMPT)

    def test_persona_is_a_cute_hat_doudou(self) -> None:
        # 大耳帽兜只是它的样子和脾气来源，这里它就叫小毛豆，不用游戏原名
        self.assertIn("大耳帽兜", SYSTEM_PROMPT)
        self.assertIn("就叫小毛豆", SYSTEM_PROMPT)
        self.assertIn("不用游戏原名", SYSTEM_PROMPT)
        self.assertIn("可爱、调皮", SYSTEM_PROMPT)

    def test_rule_speaker_labels_are_spelled_out(self) -> None:
        # 说话人错位（把 A 的话算到 B 头上）的根治办法：prompt 里明说【】就是说话人
        self.assertIn("每条记录开头的【】里就是说话人", SYSTEM_PROMPT)
        self.assertIn("【小毛豆】是你自己说过的话", SYSTEM_PROMPT)

    def test_rule_correction_targets_the_topic_not_itself(self) -> None:
        # 「不是编程工具，是 agent」被它听成「在说我自己」，必须写清被纠正的是话题本身
        self.assertIn("被纠正的是那个话题本身，不是你自己", SYSTEM_PROMPT)
        self.assertIn("别把别人的话往自己身上套", SYSTEM_PROMPT)

    def test_rule_says_it_can_see_images(self) -> None:
        # 已经识别出雪糕图、被问却否认「看不了图片」，是 prompt 从没告诉它能看图
        self.assertIn("〔图〕", SYSTEM_PROMPT)
        self.assertIn("说明你确实看见了那张图", SYSTEM_PROMPT)
        self.assertIn("不要否认", SYSTEM_PROMPT)


class GroupPrefPromptTests(unittest.TestCase):
    def test_bot_pref_action_is_listed(self) -> None:
        self.assertIn("bot_pref", INTENT_SYSTEM_PROMPT)

    def test_only_the_four_tunable_keys_are_named(self) -> None:
        for key in (
            "care_enabled",
            "care_daily_max",
            "care_cooldown_minutes",
            "active_msg_daily_max",
        ):
            self.assertIn(key, INTENT_SYSTEM_PROMPT)

    def test_a_natural_way_of_turning_it_off_is_recognised(self) -> None:
        self.assertIn("以后别主动安慰我们", INTENT_SYSTEM_PROMPT)


class CarePromptTests(unittest.TestCase):
    def test_the_stay_quiet_cases_are_written_down(self) -> None:
        # 这几条是“宁可不说话”的判断依据，删掉任何一条都会变成在群里瞎插嘴。
        for phrase in ("转述别人", "平静的陈述", "早就被别人接过", "几小时前的旧事", "跟机器人说话"):
            self.assertIn(phrase, CARE_SYSTEM_PROMPT)

    def test_the_care_line_is_short_and_not_a_doctor(self) -> None:
        self.assertIn("25 字以内", CARE_SYSTEM_PROMPT)
        self.assertIn("不诊断、不荐药", CARE_SYSTEM_PROMPT)

    def test_ai_tone_is_banned(self) -> None:
        self.assertIn("作为 AI", CARE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
