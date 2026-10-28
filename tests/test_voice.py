from __future__ import annotations

import asyncio
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import maodou.voice as voice_module
from maodou.images import format_message_content
from maodou.voice import (
    UNRECOGNIZED_VOICE_MARK,
    VoiceAttachment,
    VoiceSynthesizer,
    VoiceTranscriber,
    clean_tts_text,
    message_with_unrecognized_voice,
    message_with_voice_transcripts,
    needs_voice_transcript,
    split_voice_notes,
    voice_attachments,
)


VOICE_RAW = {
    "attachments": [
        {
            "content_type": "voice",
            "filename": "voice.silk",
            "url": "https://example.com/voice.silk",
            "voice_wav_url": "https://example.com/voice.wav",
            "asr_refer_text": "今天下班好累",
        },
        {
            "content_type": "voice",
            "asr_refer_text": "只有识别文本也能用",
        },
        {
            "content_type": "image/jpeg",
            "url": "https://example.com/cat.jpg",
        },
        "not-a-dict",
    ]
}


class VoicePayloadTests(unittest.TestCase):
    def test_voice_attachment_fields_are_extracted_from_raw_payload(self) -> None:
        attachments = voice_attachments(VOICE_RAW)
        self.assertEqual(len(attachments), 2)
        self.assertEqual(attachments[0].wav_url, "https://example.com/voice.wav")
        self.assertEqual(attachments[0].asr_text, "今天下班好累")
        self.assertEqual(attachments[1].asr_text, "只有识别文本也能用")
        self.assertEqual(attachments[1].url, "")

    def test_unsupported_or_malformed_entries_are_ignored(self) -> None:
        self.assertEqual(voice_attachments(None), [])
        self.assertEqual(voice_attachments({}), [])
        self.assertEqual(
            voice_attachments(
                {
                    "attachments": [
                        {"content_type": "video/mp4", "url": "https://x/a.mp4"},
                        {"content_type": "audio/wav", "url": "file:///tmp/a.wav"},
                    ]
                }
            ),
            [],
        )

    def test_formatting_makes_transcript_a_separate_voice_note(self) -> None:
        enriched = message_with_voice_transcripts(
            "[语音]",
            ["今天下班好累", "晚上想吃面"],
        )
        self.assertEqual(
            enriched,
            "[语音内容：今天下班好累]\n[语音内容：晚上想吃面]",
        )
        self.assertEqual(
            format_message_content("听完啦\n" + enriched, indent="  "),
            "听完啦\n  〔语音〕今天下班好累\n  〔语音〕晚上想吃面",
        )
        self.assertEqual(split_voice_notes(enriched)[1], ["今天下班好累", "晚上想吃面"])
        self.assertFalse(needs_voice_transcript(enriched, VOICE_RAW))

    def test_unrecognized_voice_is_explicit_and_not_retried(self) -> None:
        marked = message_with_unrecognized_voice("[语音]")
        self.assertEqual(marked, UNRECOGNIZED_VOICE_MARK)
        self.assertFalse(needs_voice_transcript(marked, VOICE_RAW))
        self.assertEqual(
            format_message_content(marked),
            "〔语音〕（没听清）",
        )

    def test_pending_placeholder_is_not_mistaken_for_user_text(self) -> None:
        self.assertTrue(needs_voice_transcript("[语音]", VOICE_RAW))
        self.assertEqual(
            format_message_content("[语音]"),
            "〔语音〕（这条还没听）",
        )


class VoiceServiceTests(unittest.TestCase):
    def test_qq_asr_text_is_used_without_downloading_or_calling_external_asr(self) -> None:
        transcriber = VoiceTranscriber(
            SimpleNamespace(
                asr_base_url="https://asr.example/v1",
                asr_api_key="key",
                asr_model="whisper-test",
            )
        )
        attachment = VoiceAttachment(
            url="https://example.com/a.wav",
            wav_url="https://example.com/a.wav",
            asr_text="QQ 已经听懂了",
        )

        async def forbidden_fetch(*args, **kwargs):
            raise AssertionError("QQ ASR 已有结果时不该下载音频")

        original = voice_module.fetch_voice_bytes
        voice_module.fetch_voice_bytes = forbidden_fetch
        self.addCleanup(setattr, voice_module, "fetch_voice_bytes", original)
        self.assertEqual(asyncio.run(transcriber.transcribe(attachment)), "QQ 已经听懂了")

    def test_openai_tts_requires_complete_config(self) -> None:
        disabled = VoiceSynthesizer(
            SimpleNamespace(
                voice_reply=True,
                tts_provider="openai",
                tts_base_url="https://tts.example/v1",
                tts_api_key="key",
                tts_model="tts-1",
                tts_voice="",
            )
        )
        self.assertFalse(disabled.enabled)
        enabled = VoiceSynthesizer(
            SimpleNamespace(
                voice_reply=True,
                tts_provider="openai",
                tts_base_url="https://tts.example/v1",
                tts_api_key="key",
                tts_model="tts-1",
                tts_voice="cute",
            )
        )
        self.assertTrue(enabled.enabled)

    def test_auto_tts_prefers_openai_then_falls_back_to_edge(self) -> None:
        with patch.object(
            voice_module.importlib.util,
            "find_spec",
            return_value=object(),
        ):
            openai_ready = VoiceSynthesizer(
                SimpleNamespace(
                    voice_reply=True,
                    tts_provider="auto",
                    tts_base_url="https://tts.example/v1",
                    tts_api_key="key",
                    tts_model="tts-1",
                    tts_voice="cute",
                )
            )
            self.assertEqual(openai_ready.effective_provider, "openai")

            edge_fallback = VoiceSynthesizer(
                SimpleNamespace(
                    voice_reply=True,
                    tts_provider="auto",
                    tts_base_url="",
                    tts_api_key="",
                    tts_model="",
                    tts_voice="",
                )
            )
            self.assertEqual(edge_fallback.effective_provider, "edge")
            self.assertTrue(edge_fallback.enabled)

    def test_qwen_tts_requires_complete_config_and_is_preferred_by_auto(self) -> None:
        settings = SimpleNamespace(
            voice_reply=True,
            tts_provider="auto",
            tts_base_url="https://cn-beijing.maas.qwencloudapi.com/api/v1",
            tts_api_key="secret-key",
            tts_model="qwen-audio-3.1-tts-flash",
            tts_voice="longling_v3.1",
        )
        self.assertEqual(VoiceSynthesizer(settings).effective_provider, "qwen")
        settings.tts_voice = ""
        with patch.object(
            voice_module.importlib.util,
            "find_spec",
            return_value=object(),
        ):
            self.assertEqual(VoiceSynthesizer(settings).effective_provider, "edge")

    def test_qwen_synthesis_sends_instruction_and_downloads_audio(self) -> None:
        calls: list[dict] = []

        class FakeResponse:
            def __init__(self, *, body=None, content=b"", content_type="") -> None:
                self._body = body
                self.content = content
                self.headers = {"content-type": content_type}

            def raise_for_status(self) -> None:
                return None

            def json(self):
                return self._body

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, *, headers, json):
                calls.append(
                    {
                        "kind": "post",
                        "url": url,
                        "headers": headers,
                        "json": json,
                    }
                )
                return FakeResponse(
                    body={"output": {"audio": {"url": "https://audio.example/out.wav"}}},
                    content_type="application/json",
                )

            async def get(self, url):
                calls.append({"kind": "get", "url": url})
                return FakeResponse(
                    content=b"RIFF-wav-data",
                    content_type="audio/x-wav",
                )

        synth = VoiceSynthesizer(
            SimpleNamespace(
                voice_reply=True,
                tts_provider="qwen",
                tts_base_url="https://cn-beijing.maas.qwencloudapi.com/api/v1",
                tts_api_key="secret-key",
                tts_model="qwen-audio-3.1-tts-flash",
                tts_voice="longling_v3.1",
                tts_format="wav",
                tts_instruction="软糯女童声",
                tts_sample_rate=24000,
                tts_timeout_seconds=30,
                tts_max_chars=160,
                tts_max_bytes=1024,
            )
        )
        with patch.object(voice_module.httpx2, "AsyncClient", FakeClient):
            with self.assertLogs("maodou.voice", level=logging.INFO) as captured:
                audio = asyncio.run(synth.synthesize("**晚安**"))

        self.assertEqual(audio, b"RIFF-wav-data")
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0]["url"],
            "https://cn-beijing.maas.qwencloudapi.com/api/v1/"
            "services/audio/tts/SpeechSynthesizer",
        )
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(calls[0]["json"]["model"], "qwen-audio-3.1-tts-flash")
        self.assertEqual(calls[0]["json"]["input"]["text"], "晚安")
        self.assertEqual(calls[0]["json"]["input"]["voice"], "longling_v3.1")
        self.assertEqual(calls[0]["json"]["input"]["format"], "wav")
        self.assertEqual(calls[0]["json"]["input"]["sample_rate"], 24000)
        self.assertEqual(calls[0]["json"]["input"]["instruction"], "软糯女童声")
        self.assertEqual(calls[1], {"kind": "get", "url": "https://audio.example/out.wav"})
        self.assertNotIn("secret-key", "\n".join(captured.output))

    def test_qwen_synthesis_rejects_missing_audio_url(self) -> None:
        class FakeResponse:
            content = b""
            headers = {"content-type": "application/json"}

            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"output": {}}

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

        synth = VoiceSynthesizer(
            SimpleNamespace(
                voice_reply=True,
                tts_provider="qwen",
                tts_base_url="https://qwen.example/v1",
                tts_api_key="key",
                tts_model="model",
                tts_voice="voice",
            )
        )
        with patch.object(voice_module.httpx2, "AsyncClient", FakeClient):
            self.assertEqual(asyncio.run(synth.synthesize("晚安")), b"")

    def test_qwen_synthesis_handles_http_error_and_size_limit(self) -> None:
        class FakeClient:
            mode = "error"

            def __init__(self, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                if self.mode == "error":
                    raise voice_module.httpx2.HTTPError("boom")

                class Response:
                    def raise_for_status(self) -> None:
                        return None

                    def json(self):
                        return {
                            "output": {
                                "audio": {"url": "https://audio.example/out.wav"}
                            }
                        }

                return Response()

            async def get(self, url):
                class Response:
                    content = b"x" * 2048

                    def raise_for_status(self) -> None:
                        return None

                return Response()

        settings = SimpleNamespace(
            voice_reply=True,
            tts_provider="qwen",
            tts_base_url="https://qwen.example/v1",
            tts_api_key="key",
            tts_model="model",
            tts_voice="voice",
            tts_max_bytes=1024,
        )
        synth = VoiceSynthesizer(settings)
        with patch.object(voice_module.httpx2, "AsyncClient", FakeClient):
            self.assertEqual(asyncio.run(synth.synthesize("晚安")), b"")
            FakeClient.mode = "large"
            self.assertEqual(asyncio.run(synth.synthesize("晚安")), b"")

    def test_explicit_edge_provider_requires_the_optional_dependency(self) -> None:
        settings = SimpleNamespace(voice_reply=True, tts_provider="edge")
        with patch.object(
            voice_module.importlib.util,
            "find_spec",
            return_value=None,
        ):
            self.assertFalse(VoiceSynthesizer(settings).enabled)
        with patch.object(
            voice_module.importlib.util,
            "find_spec",
            return_value=object(),
        ):
            self.assertTrue(VoiceSynthesizer(settings).enabled)

    def test_edge_synthesis_uses_configured_voice_without_network(self) -> None:
        class FakeCommunicate:
            def __init__(self, text, voice, **kwargs) -> None:
                self.text = text
                self.voice = voice
                self.kwargs = kwargs

            async def stream(self):
                yield {"type": "audio", "data": b"first"}
                yield {"type": "WordBoundary", "data": b"ignored"}
                yield {"type": "audio", "data": b"second"}

        synth = VoiceSynthesizer(
            SimpleNamespace(
                voice_reply=True,
                tts_provider="edge",
                tts_edge_voice="zh-CN-XiaoyiNeural",
                tts_edge_rate="+5%",
                tts_edge_pitch="+5Hz",
                tts_edge_volume="+0%",
                tts_max_chars=40,
            )
        )
        with patch("edge_tts.Communicate", FakeCommunicate):
            audio = asyncio.run(synth.synthesize("**晚安**"))

        self.assertEqual(audio, b"firstsecond")

    def test_tts_text_is_shortened_and_markup_is_removed(self) -> None:
        text = "**晚安呀** [抱抱] https://example.com 〔语音〕不该念这个"
        cleaned = clean_tts_text(text, max_chars=40)
        self.assertEqual(cleaned, "晚安呀")
        long = clean_tts_text("第一句。" + "很长的内容" * 20, max_chars=12)
        self.assertLessEqual(len(long), 12)


if __name__ == "__main__":
    unittest.main()
