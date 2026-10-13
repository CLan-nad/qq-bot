from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx2


SUPPORTED_IMAGE_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)

# QQ 在纯图片消息里只给出占位文本，例如 "[图片]" 或 "[image/jpeg]"。
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[(?:图片|image/[A-Za-z0-9.+-]+)\]", re.IGNORECASE)

# QQ 表情包（大表情，faceType=6）在消息体里是 <faceType=6,faceId="0",ext="..."> 标签，
# 真正的图片放在 attachments 里；系统小表情（faceType=1）没有附件。ext 是 base64 的
# JSON，里面的 text 字段是可读的表情名称，例如 [大胆]、[哈哈]。
FACE_TAG_PATTERN = re.compile(r'<faceType=(\d+),faceId="(\d+)",ext="([^"]*)">')

# 已经处理过的标记，避免对同一张图反复调用模型。
IMAGE_DESCRIBED_MARKS = ("图片内容：", "图片未识别")
UNRECOGNIZED_IMAGE_MARK = "[图片未识别]"

DEFAULT_IMAGE_MAX_BYTES = 8 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_IMAGE_REQUEST_HEADERS = {
    # QQ 图片 CDN 对默认 UA 不友好，这里伪装成普通浏览器请求。
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) xiaoxia-qq-bot/0.1",
    "Referer": "https://q.qq.com/",
}


@dataclass(frozen=True)
class ImageAttachment:
    url: str
    content_type: str
    filename: str


def image_attachments(raw: dict[str, Any]) -> list[ImageAttachment]:
    result: list[ImageAttachment] = []
    for item in raw.get("attachments") or []:
        if not isinstance(item, dict):
            continue
        content_type = str(item.get("content_type") or "").lower().split(";", 1)[0]
        url = str(item.get("url") or "").strip()
        if content_type not in SUPPORTED_IMAGE_TYPES or not url.startswith(("http://", "https://")):
            continue
        result.append(
            ImageAttachment(
                url=url,
                content_type=content_type,
                filename=str(item.get("filename") or ""),
            )
        )
    return result


def strip_image_placeholders(content: str | None) -> str:
    text = normalize_face_tags(content or "")
    text = IMAGE_PLACEHOLDER_PATTERN.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def needs_image_description(content: str | None) -> bool:
    """图片或表情包内容是否还没被识别过（用于避免重复调用视觉模型）。"""
    text = (content or "").strip()
    if any(mark in text for mark in IMAGE_DESCRIBED_MARKS):
        return False
    return bool(IMAGE_PLACEHOLDER_PATTERN.search(text) or FACE_TAG_PATTERN.search(text))


def message_with_image_descriptions(content: str | None, descriptions: list[str]) -> str:
    text = strip_image_placeholders(content)
    parts = [text] if text else []
    parts.extend(f"[图片内容：{description}]" for description in descriptions if description.strip())
    return "\n".join(parts) or "[图片]"


def message_with_unrecognized_images(content: str | None) -> str:
    """识别失败时留下的标记，避免每次 @ 都重试已失效的图片链接。"""
    text = strip_image_placeholders(content)
    parts = [text] if text else []
    parts.append(UNRECOGNIZED_IMAGE_MARK)
    return "\n".join(parts)


def _decode_face_text(ext: str) -> str:
    """解开 QQ 表情标签里的 ext，取出可读名称；失败时返回空串。"""
    if not ext:
        return ""
    try:
        padded = ext + "=" * (-len(ext) % 4)
        payload = json.loads(base64.b64decode(padded).decode("utf-8", "ignore"))
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("text") or "").strip()
    return ""


def normalize_face_tags(content: str | None) -> str:
    """把 QQ 表情标签还原成可读文本，例如 <faceType=6,...> 变成 [大胆]。"""
    if not content:
        return ""
    return FACE_TAG_PATTERN.sub(
        lambda match: _decode_face_text(match.group(3)) or "[表情]",
        content,
    )


def sniff_image_content_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def fetch_image_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
    timeout: float = IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
) -> tuple[bytes, str] | None:
    """把图片取到内存。QQ 图片链接自带时效签名，交给视觉模型自行下载经常失败。"""
    if not url.startswith(("http://", "https://")):
        return None
    if max_bytes <= 0:
        return None
    try:
        async with httpx2.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url, headers=_IMAGE_REQUEST_HEADERS)
            response.raise_for_status()
    except httpx2.HTTPError:
        return None

    data = response.content
    if not data or len(data) > max_bytes:
        return None
    header_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    content_type = sniff_image_content_type(data) or header_type
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return None
    return data, content_type


def to_data_url(data: bytes, content_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
