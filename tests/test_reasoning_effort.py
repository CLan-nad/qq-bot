from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from maodou import config as config_module
from maodou.llm import MaodouLLM


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _llm(effort: str | None) -> tuple[MaodouLLM, _Completions]:
    settings = SimpleNamespace(
        llm_api_key="test-key",
        llm_base_url="https://api.deepseek.com",
        model="deepseek-chat",
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
    def test_chat_defaults_to_low_when_setting_missing(self) -> None:
        llm, completions = _llm(None)
        asyncio.run(
            llm._chat(
                system_prompt="system",
                user_prompt="hi",
                temperature=0.5,
                max_tokens=64,
            )
        )
        self.assertEqual(completions.calls[0]["reasoning_effort"], "low")

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
    def _load(self, value: str | None) -> str:
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
            return config_module.load_settings().reasoning_effort

    def test_default_is_low(self) -> None:
        self.assertEqual(self._load(None), "low")

    def test_valid_values_are_kept(self) -> None:
        for value in ("none", "low", "high", "max", "LOW"):
            with self.subTest(value=value):
                self.assertEqual(self._load(value), value.lower())

    def test_invalid_value_falls_back_to_low(self) -> None:
        self.assertEqual(self._load("turbo"), "low")


if __name__ == "__main__":
    unittest.main()
