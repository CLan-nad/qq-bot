"""提示词约束：搜索能不能用，取决于意图模型写出的检索词质量。

实测过：用“权威医学资料 依据 来源”去搜，回来的全是课件垃圾；用“拉肚子 肚子疼 原因”
才是对的。所以这几条约束必须留在提示词里，并且有测试盯着，不然以后被顺手删掉就复发。
"""
from __future__ import annotations

import inspect
import unittest

from xiaoxia.llm import INTENT_SYSTEM_PROMPT, SYSTEM_PROMPT, XiaoxiaLLM


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
        self.assertNotIn("recent_context", params)


class SystemPromptTests(unittest.TestCase):
    def test_rule_background_topic_must_not_be_raised(self) -> None:
        self.assertIn("【更早的背景】", SYSTEM_PROMPT)
        self.assertIn("绝不主动提起", SYSTEM_PROMPT)

    def test_rule_health_no_diagnosis(self) -> None:
        self.assertIn("不下诊断结论", SYSTEM_PROMPT)
        self.assertIn("不推荐具体药品和剂量", SYSTEM_PROMPT)
        self.assertIn("建议去医院", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
