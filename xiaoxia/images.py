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
# ext 在个别客户端上会整个缺失，所以做成可选；缺了就退化成 [表情]。
FACE_TAG_PATTERN = re.compile(r'<faceType=(\d+),faceId="(\d+)"(?:,ext="([^"]*)")?\s*>')

# 已经处理过的标记，避免对同一张图反复调用模型。
IMAGE_DESCRIBED_MARKS = ("图片内容：", "图片未识别")
UNRECOGNIZED_IMAGE_MARK = "[图片未识别]"

# 图片描述入库时拼在正文里，形如「[图片内容：…]」。这条记录是「我看过这张图」的凭据，
# 但它混在用户正文里就成了同一行文字，模型会以为「这是对方打出来的字」——线上于是出现
# 「它都看出是白桃乌龙雪糕了，问它却说没识别出图片」。渲染 prompt 时统一拆成独立的〔图〕
# 行，跟用户原话分清楚：文字是对方说的，〔图〕是我自己看图留下的记录。
IMAGE_NOTE_LABEL = "〔图〕"
IMAGE_NOTE_PATTERN = re.compile(r"\[图片内容：(.*?)\]", re.S)
# 还没轮到识别（没有 @ 事件就不会跑视觉模型）和识别失败的图：渲染成人话，
# 别把 "[image/jpeg]" 这种占位符丢给模型——它只会困惑，或者干脆当作对方打的字。
IMAGE_PENDING_NOTE = "（这张还没识别）"
IMAGE_UNREADABLE_NOTE = "（没能看清）"

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


def needs_image_description(content: str | None, raw: dict[str, Any] | None = None) -> bool:
    """图片或表情包内容是否还没被识别过（用于避免重复调用视觉模型）。

    表情标签在入库时就已经解码成可读文本，光看 content 认不出「这条其实还带图」，
    所以再用原始 payload 里的图片附件兜一次底。
    """
    text = (content or "").strip()
    if any(mark in text for mark in IMAGE_DESCRIBED_MARKS):
        return False
    if IMAGE_PLACEHOLDER_PATTERN.search(text) or FACE_TAG_PATTERN.search(text):
        return True
    if not isinstance(raw, dict):
        return False
    return bool(image_attachments(raw))


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


def split_image_notes(content: str | None) -> tuple[str, list[str]]:
    """把「[图片内容：…]」从正文里拆出来，返回 (用户原话, [看图记录...])。"""
    text = content or ""
    notes: list[str] = []
    for match in IMAGE_NOTE_PATTERN.finditer(text):
        note = " ".join(match.group(1).split())
        if note:
            notes.append(note)
    remainder = IMAGE_NOTE_PATTERN.sub("", text)
    lines = [line.strip() for line in remainder.splitlines()]
    return "\n".join(line for line in lines if line).strip(), notes


def format_message_content(
    content: str | None,
    *,
    indent: str = "",
    note_label: str = IMAGE_NOTE_LABEL,
) -> str:
    """渲染一条消息的正文：用户原话在前，自己看图的记录缩进成独立的〔图〕行。"""
    text, notes = split_image_notes(content)
    text = IMAGE_PLACEHOLDER_PATTERN.sub(f"{note_label}{IMAGE_PENDING_NOTE}", text)
    text = text.replace(UNRECOGNIZED_IMAGE_MARK, f"{note_label}{IMAGE_UNREADABLE_NOTE}")
    lines = [text] if text else []
    lines.extend(f"{indent}{note_label}{note}" for note in notes)
    return "\n".join(lines)


def decode_face_text(ext: str) -> str:
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
    """把 QQ 表情标签还原成可读文本，例如 <faceType=6,...> 变成 [大胆]。

    系统小表情（faceType=1/3）解出来的名字不带方括号，这里统一补上，免得模型把
    「头秃」「偷感」这种表情名当成句子里的普通词；解不出来就退化成 [表情]。
    """
    if not content:
        return ""

    def render(match: re.Match[str]) -> str:
        name = decode_face_text(match.group(3) or "")
        if not name:
            return "[表情]"
        return name if name.startswith("[") else f"[{name}]"

    return FACE_TAG_PATTERN.sub(render, content)


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
