from __future__ import annotations

import base64
import importlib.util
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx2


logger = logging.getLogger("maodou.voice")


# QQ 的语音消息正文通常只有占位符，例如 "[语音]" 或 "[voice]"。
VOICE_PLACEHOLDER_PATTERN = re.compile(r"\[(?:语音|voice|audio)\]", re.IGNORECASE)

# 和图片记录一样，语音转写要作为单独的「机器人听到的内容」传给模型，
# 不能让模型误以为那是群友手打的文字。
VOICE_NOTE_LABEL = "〔语音〕"
VOICE_NOTE_PATTERN = re.compile(r"\[语音内容：(.*?)\]", re.S)
VOICE_DESCRIBED_MARKS = ("语音内容：", "语音未识别")
UNRECOGNIZED_VOICE_MARK = "[语音未识别]"
VOICE_PENDING_NOTE = "（这条还没听）"
VOICE_UNREADABLE_NOTE = "（没听清）"

DEFAULT_VOICE_MAX_BYTES = 20 * 1024 * 1024
VOICE_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_EDGE_VOICE = "zh-CN-XiaoyiNeural"
_VOICE_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) maodou-qq-bot/0.1",
    "Referer": "https://q.qq.com/",
}


@dataclass(frozen=True)
class VoiceAttachment:
    """QQ 原始语音附件中真正对回复有用的字段。"""

    url: str
    wav_url: str
    asr_text: str
    filename: str = ""


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _http_url(value: Any) -> str:
    text = _clean_text(value)
    if text.startswith(("http://", "https://")):
        return text
    return ""


def voice_attachments(raw: dict[str, Any] | None) -> list[VoiceAttachment]:
    """从 QQ 原始 payload 提取语音附件。

    qq-botpy 1.2.1 没有把 `voice_wav_url` / `asr_refer_text` 解析到
    Attachment 对象里，但项目已经保存了完整 raw payload，所以这里直接读原始
    `attachments`，不需要修改 SDK。
    """
    if not isinstance(raw, dict):
        return []
    result: list[VoiceAttachment] = []
    for item in raw.get("attachments") or []:
        if not isinstance(item, dict):
            continue
        content_type = _clean_text(item.get("content_type")).lower().split(";", 1)[0]
        url = _http_url(item.get("url"))
        wav_url = _http_url(item.get("voice_wav_url")) or url
        asr_text = _clean_text(item.get("asr_refer_text"))
        is_voice = (
            content_type == "voice"
            or content_type.startswith("audio/")
            or bool(item.get("voice_wav_url"))
        )
        if not is_voice or not (wav_url or asr_text):
            continue
        result.append(
            VoiceAttachment(
                url=url or wav_url,
                wav_url=wav_url,
                asr_text=asr_text,
                filename=_clean_text(item.get("filename")),
            )
        )
    return result


def strip_voice_placeholders(content: str | None) -> str:
    text = VOICE_PLACEHOLDER_PATTERN.sub(" ", content or "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def needs_voice_transcript(content: str | None, raw: dict[str, Any] | None = None) -> bool:
    """这条消息是否还带有尚未转写的语音。"""
    text = (content or "").strip()
    if any(mark in text for mark in VOICE_DESCRIBED_MARKS):
        return False
    if VOICE_PLACEHOLDER_PATTERN.search(text):
        return True
    return bool(voice_attachments(raw))


def message_with_voice_transcripts(content: str | None, transcripts: list[str]) -> str:
    text = strip_voice_placeholders(content)
    parts = [text] if text else []
    parts.extend(
        f"[语音内容：{_clean_text(transcript)}]"
        for transcript in transcripts
        if _clean_text(transcript)
    )
    return "\n".join(parts)


def message_with_unrecognized_voice(content: str | None) -> str:
    text = strip_voice_placeholders(content)
    parts = [text] if text else []
    parts.append(UNRECOGNIZED_VOICE_MARK)
    return "\n".join(parts)


def split_voice_notes(content: str | None) -> tuple[str, list[str]]:
    text = content or ""
    notes: list[str] = []
    for match in VOICE_NOTE_PATTERN.finditer(text):
        note = _clean_text(match.group(1))
        if note:
            notes.append(note)
    remainder = VOICE_NOTE_PATTERN.sub("", text)
    lines = [line.strip() for line in remainder.splitlines()]
    return "\n".join(line for line in lines if line).strip(), notes


def clean_tts_text(text: str, *, max_chars: int = 160) -> str:
    """把 QQ 正文收拾成适合朗读的短句。

    提示词里的表情、Markdown、链接和〔图〕/〔语音〕记录不适合朗读，先清掉。
    TTS 只负责把正文念出来，不会因为清理后为空就阻塞聊天。
    """
    max_chars = max(0, int(max_chars))
    if max_chars == 0:
        return ""
    value = str(text or "")
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`#>~]+", " ", value)
    value = re.sub(r"\[(?:图片内容|图片未识别|语音内容|语音未识别)[^\]]*\]", " ", value)
    value = re.sub(r"〔(?:图|语音)〕[^\n]*", " ", value)
    value = re.sub(r"\[[^\]]{1,12}\]", " ", value)
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value.strip()
    clipped = value[:max_chars]
    for punctuation in ("。", "！", "？", ".", "!", "?"):
        index = clipped.rfind(punctuation)
        if index >= max_chars // 2:
            return clipped[: index + 1].strip()
    return clipped[: max(0, max_chars - 1)].rstrip("，,、；;：: ") + "…"


async def fetch_voice_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_VOICE_MAX_BYTES,
    timeout: float = VOICE_DOWNLOAD_TIMEOUT_SECONDS,
) -> tuple[bytes, str] | None:
    if not url.startswith(("http://", "https://")) or max_bytes <= 0:
        return None
    try:
        async with httpx2.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url, headers=_VOICE_REQUEST_HEADERS)
            response.raise_for_status()
    except httpx2.HTTPError:
        return None
    data = response.content
    if not data or len(data) > max_bytes:
        return None
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    return data, content_type or "audio/wav"


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


class VoiceTranscriber:
    """可选的 OpenAI-compatible ASR 兜底。

    QQ 正常会直接给出 asr_refer_text，优先使用它；只有该字段缺失时才下载
    voice_wav_url 并调用这里配置的 /audio/transcriptions。没有配置时安静返回空，
    不会因为语音转写服务不可用而影响文字聊天。
    """

    def __init__(self, settings: Any) -> None:
        self.base_url = str(getattr(settings, "asr_base_url", "") or "").strip()
        self.api_key = str(getattr(settings, "asr_api_key", "") or "").strip()
        self.model = str(getattr(settings, "asr_model", "") or "").strip()
        self.timeout_seconds = float(
            getattr(settings, "asr_timeout_seconds", 30) or 30
        )
        self.max_bytes = int(
            getattr(settings, "asr_max_bytes", DEFAULT_VOICE_MAX_BYTES)
            or DEFAULT_VOICE_MAX_BYTES
        )

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    async def transcribe(self, attachment: VoiceAttachment) -> str:
        if attachment.asr_text:
            return attachment.asr_text
        if not self.enabled or not attachment.wav_url:
            return ""
        fetched = await fetch_voice_bytes(
            attachment.wav_url,
            max_bytes=self.max_bytes,
            timeout=self.timeout_seconds,
        )
        if fetched is None:
            logger.warning("voice download failed, skipping ASR")
            return ""
        data, content_type = fetched
        filename = attachment.filename or "voice.wav"
        try:
            async with httpx2.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    _endpoint(self.base_url, "audio/transcriptions"),
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data={"model": self.model},
                    files={
                        "file": (
                            filename,
                            data,
                            content_type or "audio/wav",
                        )
                    },
                )
                response.raise_for_status()
        except httpx2.HTTPError:
            logger.exception("ASR request failed")
            return ""
        try:
            payload = response.json()
        except (ValueError, TypeError):
            return ""
        if isinstance(payload, dict):
            return _clean_text(
                payload.get("text")
                or payload.get("transcript")
                or payload.get("result")
                or ""
            )
        return ""


class VoiceSynthesizer:
    """生成 QQ 可发送的语音。

    `auto` 依次选择 QwenCloud、OpenAI-compatible `/audio/speech`，最后才退回
    Microsoft Edge 的在线中文神经语音。任何一条路失败都只会保留文字，不会影响
    整轮回复。
    """

    def __init__(self, settings: Any) -> None:
        self.enabled_setting = str(
            getattr(settings, "voice_reply", "1") or "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.provider = (
            str(getattr(settings, "tts_provider", "auto") or "auto").strip().lower()
        )
        self.base_url = str(getattr(settings, "tts_base_url", "") or "").strip()
        self.api_key = str(getattr(settings, "tts_api_key", "") or "").strip()
        self.model = str(getattr(settings, "tts_model", "") or "").strip()
        self.voice = str(getattr(settings, "tts_voice", "") or "").strip()
        self.response_format = (
            str(getattr(settings, "tts_format", "mp3") or "mp3").strip().lower()
        )
        self.instruction = str(
            getattr(settings, "tts_instruction", "") or ""
        ).strip()
        self.sample_rate = int(
            getattr(settings, "tts_sample_rate", 24000) or 24000
        )
        self.timeout_seconds = float(
            getattr(settings, "tts_timeout_seconds", 30) or 30
        )
        self.max_chars = int(getattr(settings, "tts_max_chars", 160) or 160)
        self.max_bytes = int(
            getattr(settings, "tts_max_bytes", DEFAULT_VOICE_MAX_BYTES)
            or DEFAULT_VOICE_MAX_BYTES
        )
        self.edge_voice = (
            str(
                getattr(settings, "tts_edge_voice", DEFAULT_EDGE_VOICE)
                or DEFAULT_EDGE_VOICE
            ).strip()
            or DEFAULT_EDGE_VOICE
        )
        self.edge_rate = (
            str(getattr(settings, "tts_edge_rate", "+5%") or "+5%").strip()
            or "+5%"
        )
        self.edge_pitch = (
            str(getattr(settings, "tts_edge_pitch", "+5Hz") or "+5Hz").strip()
            or "+5Hz"
        )
        self.edge_volume = (
            str(getattr(settings, "tts_edge_volume", "+0%") or "+0%").strip()
            or "+0%"
        )

    @property
    def _qwen_ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model and self.voice)

    @property
    def _openai_ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model and self.voice)

    @property
    def _auto_qwen_ready(self) -> bool:
        return bool(
            self._qwen_ready
            and "qwencloudapi.com" in self.base_url.lower()
        )

    @property
    def _edge_ready(self) -> bool:
        try:
            return importlib.util.find_spec("edge_tts") is not None
        except (ImportError, ValueError):
            return False

    @property
    def effective_provider(self) -> str:
        if self.provider == "qwen":
            return "qwen" if self._qwen_ready else ""
        if self.provider == "openai":
            return "openai" if self._openai_ready else ""
        if self.provider == "edge":
            return "edge" if self._edge_ready else ""
        if self.provider == "auto":
            if self._auto_qwen_ready:
                return "qwen"
            if self._openai_ready:
                return "openai"
            return "edge" if self._edge_ready else ""
        return ""

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_setting and self.effective_provider)

    async def synthesize(self, text: str) -> bytes:
        if not self.enabled:
            return b""
        spoken = clean_tts_text(text, max_chars=self.max_chars)
        if not spoken:
            return b""
        if self.effective_provider == "edge":
            return await self._synthesize_edge(spoken)
        if self.effective_provider == "qwen":
            return await self._synthesize_qwen(spoken)
        return await self._synthesize_openai(spoken)

    async def _synthesize_edge(self, spoken: str) -> bytes:
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                spoken,
                self.edge_voice,
                rate=self.edge_rate,
                volume=self.edge_volume,
                pitch=self.edge_pitch,
            )
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and isinstance(
                    chunk.get("data"), (bytes, bytearray)
                ):
                    chunks.append(bytes(chunk["data"]))
            data = b"".join(chunks)
        except Exception:
            logger.exception("Edge TTS request failed")
            return b""
        if not data or len(data) > self.max_bytes:
            logger.warning(
                "Edge TTS returned empty or oversized audio (%d bytes)", len(data)
            )
            return b""
        logger.info(
            "edge tts generated %d bytes voice=%s chars=%d",
            len(data),
            self.edge_voice,
            len(spoken),
        )
        return data

    async def _synthesize_qwen(self, spoken: str) -> bytes:
        input_payload: dict[str, Any] = {
            "text": spoken,
            "voice": self.voice,
            "format": self.response_format,
            "sample_rate": self.sample_rate,
        }
        if self.instruction:
            input_payload["instruction"] = self.instruction
        payload = {
            "model": self.model,
            "input": input_payload,
        }
        try:
            async with httpx2.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    _endpoint(
                        self.base_url,
                        "services/audio/tts/SpeechSynthesizer",
                    ),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                try:
                    body = response.json()
                except (ValueError, TypeError):
                    logger.warning("Qwen TTS returned a non-JSON response")
                    return b""
                audio_url = ""
                if isinstance(body, dict):
                    output = body.get("output")
                    if isinstance(output, dict):
                        audio = output.get("audio")
                        if isinstance(audio, dict):
                            audio_url = _http_url(audio.get("url"))
                if not audio_url:
                    logger.warning("Qwen TTS response did not contain an audio URL")
                    return b""
                audio_response = await client.get(audio_url)
                audio_response.raise_for_status()
                data = audio_response.content
        except httpx2.HTTPError:
            logger.exception("Qwen TTS request failed")
            return b""
        if not data or len(data) > self.max_bytes:
            logger.warning(
                "Qwen TTS returned empty or oversized audio (%d bytes)", len(data)
            )
            return b""
        logger.info(
            "qwen tts generated %d bytes voice=%s format=%s chars=%d",
            len(data),
            self.voice,
            self.response_format,
            len(spoken),
        )
        return data

    async def _synthesize_openai(self, spoken: str) -> bytes:
        payload: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice,
            "input": spoken,
            "response_format": self.response_format,
        }
        try:
            async with httpx2.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    _endpoint(self.base_url, "audio/speech"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx2.HTTPError:
            logger.exception("TTS request failed")
            return b""
        data = response.content
        if not data:
            return b""
        # 少数兼容服务把音频放在 JSON 的 base64 字段里，顺手兼容一下。
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            try:
                body = response.json()
            except (ValueError, TypeError):
                body = {}
            encoded = ""
            if isinstance(body, dict):
                encoded = str(
                    body.get("audio")
                    or body.get("audio_base64")
                    or body.get("data")
                    or ""
                )
            if encoded:
                try:
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    return b""
        if not data or len(data) > self.max_bytes:
            logger.warning("TTS returned empty or oversized audio (%d bytes)", len(data))
            return b""
        logger.info(
            "openai tts generated %d bytes format=%s chars=%d",
            len(data),
            self.response_format,
            len(spoken),
        )
        return data


class VoiceReply(str):
    """标记一条回复只发送语音；语音失败时才由发送层退回文字。

    继承 str 是为了不影响现有大量把回复当字符串处理的 handler 和测试；
    发送层只读取 voice_reply 属性，不需要把调用链全部改成对象。
    """

    voice_reply = True

