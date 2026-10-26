from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from maodou import config as config_module
from maodou.llm import MaodouLLM


class _Completions:
    def __init__(self, content: str = "ok") -> None:
        self.calls: list[dict] = []
        self.content = content

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _llm(effort: str | None) -> tuple[MaodouLLM, _Completions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        vision_model="deepseek-flash",
        reply_max_chars=900,
    )
    if effort is not None:
        settings.reasoning_effort = effort
    llm = MaodouLLM(settings)
    completions = _Completions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


class ReasoningEffortTests(unittest.TestCase):
    def test_chat_defaults_to_none_when_setting_missing(self) -> None:
        llm, completions = _llm(None)
        asyncio.run(
            llm._chat(
                system_prompt="system",
                user_prompt="hi",
                temperature=0.5,
                max_tokens=64,
            )
        )
        self.assertEqual(completions.calls[0]["reasoning_effort"], "none")

    def test_chat_uses_configured_effort(self) -> None:
        llm, completions = _llm("none")
        asyncio.run(
            llm._chat(
                system_prompt="system",
                user_prompt="hi",
                temperature=0.5,
                max_tokens=64,
            )
        )
        self.assertEqual(completions.calls[0]["reasoning_effort"], "none")

    def test_chat_call_can_override_effort(self) -> None:
        llm, completions = _llm("low")
        asyncio.run(
            llm._chat(
                system_prompt="system",
                user_prompt="hi",
                temperature=0.5,
                max_tokens=64,
                reasoning_effort="high",
            )
        )
        self.assertEqual(completions.calls[0]["reasoning_effort"], "high")


class ReasoningEffortConfigTests(unittest.TestCase):
    def _load_settings(self, value: str | None):
        env = {
            "QQ_APP_ID": "app",
            "QQ_APP_SECRET": "secret",
            "LLM_API_KEY": "key",
        }
        if value is not None:
            env["MAODOU_REASONING_EFFORT"] = value
        with mock.patch.object(config_module, "load_dotenv"), mock.patch.dict(
            os.environ, env, clear=True
        ):
            return config_module.load_settings()

    def test_defaults_use_v4_pro_without_chat_reasoning(self) -> None:
        settings = self._load_settings(None)
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.reasoning_effort, "none")
        self.assertEqual(settings.deep_reasoning_effort, "low")
        self.assertEqual(settings.vision_reasoning_effort, "low")

    def test_valid_values_are_kept(self) -> None:
        for value in ("none", "low", "high", "max", "LOW"):
            with self.subTest(value=value):
                self.assertEqual(
                    self._load_settings(value).reasoning_effort, value.lower()
                )

    def test_invalid_value_falls_back_to_none(self) -> None:
        self.assertEqual(self._load_settings("turbo").reasoning_effort, "none")


class IntentModelTests(unittest.TestCase):
    """意图解析单独走快模型：它只输出一个小 JSON，不该拖慢每一条闲聊。"""

    def _llm_with_intent(self, **settings) -> tuple[MaodouLLM, _Completions]:
        llm, _ = _llm("low")
        completions = _Completions(content='{"action":"chat"}')
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        for key, value in settings.items():
            setattr(llm.settings, key, value)
        return llm, completions

    def test_parse_intent_uses_its_own_model_and_effort(self) -> None:
        llm, completions = self._llm_with_intent(
            intent_model="deepseek-flash", intent_reasoning_effort="none"
        )
        result = asyncio.run(
            llm.parse_intent(
                request="在吗",
                current_time="2026-10-25T23:00:00+08:00",
                tasks_summary="",
            )
        )
        self.assertEqual(result, {"action": "chat"})
        self.assertEqual(completions.calls[0]["model"], "deepseek-flash")
        self.assertEqual(completions.calls[0]["reasoning_effort"], "none")

    def test_parse_intent_falls_back_to_reply_model_without_setting(self) -> None:
        llm, completions = self._llm_with_intent()
        asyncio.run(
            llm.parse_intent(
                request="在吗",
                current_time="2026-10-25T23:00:00+08:00",
                tasks_summary="",
            )
        )
        self.assertEqual(completions.calls[0]["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
