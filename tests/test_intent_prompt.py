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

    def test_generate_reply_splits_topic_and_background(self) -> None:
        params = inspect.signature(XiaoxiaLLM.generate_reply).parameters
        self.assertIn("topic_context", params)
        self.assertIn("background_context", params)

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

    def test_memory_batch_is_told_not_to_repeat_existing_facts(self) -> None:
        params = inspect.signature(XiaoxiaLLM.process_memory_batch).parameters
        self.assertIn("existing", params)
        self.assertIn("不要重复输出这些", inspect.getsource(XiaoxiaLLM.process_memory_batch))
        self.assertIn("绝对不要再输出一次", MEMORY_SYSTEM_PROMPT)
        self.assertIn("同一个人只允许有一种叫法", MEMORY_SYSTEM_PROMPT)
        self.assertNotIn("recent_context", params)


class SystemPromptTests(unittest.TestCase):
    def test_rule_background_topic_must_not_be_raised(self) -> None:
        self.assertIn("【更早的背景】", SYSTEM_PROMPT)
        self.assertIn("绝不主动提起", SYSTEM_PROMPT)

    def test_rule_health_no_diagnosis(self) -> None:
        self.assertIn("不下诊断结论", SYSTEM_PROMPT)
        self.assertIn("不推荐具体药品和剂量", SYSTEM_PROMPT)
        self.assertIn("建议去医院", SYSTEM_PROMPT)


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
