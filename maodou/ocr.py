"""DeepSeek 原生 OCR：把图片里的字读出来。

和视觉模型分工：视觉模型回答「这是什么」，OCR 回答「上面写了什么」。
课表、成绩单、长截图这类图的重点在文字，视觉模型容易读错漏行；OCR 逐页
返回原文，配 text_recognition 时是干净的 Markdown/HTML 表格，直接拼进
prompt 即可。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx2

logger = logging.getLogger(__name__)

OCR_PATH = "/v1/tools/ocr"
PURPOSE_TEXT = "text_recognition"

# rich_layout_recognition 会在文本外裹版面坐标：&#x3000;[[x1, y1, x2, y2]]
# text_recognition 一般没有，这里只是兜底清理。
_COORD_ARRAY_PATTERN = re.compile(r"\[\[\s*-?\d+(?:\s*,\s*-?\d+){3}\s*\]\]")
# OCR 会把表格输出成 HTML；逐行保留标签会白烧大量 token，先压成「a | b | c」。
_TABLE_PATTERN = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_TABLE_ROW_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TABLE_CELL_PATTERN = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# 用户明确要「读字」的说法。命中任意一个就把图送去 OCR。
_USER_TEXT_WORDS = (
    "文字", "原文", "念一下", "念给", "读一下", "读读", "念出来", "读出来",
    "提取", "翻译", "表格", "课表", "名单", "成绩", "截图", "写了什么",
    "写的是", "打出来", "列出来", "第几节", "几号", "几点",
)

# 视觉模型描述里出现这些词，说明图上确实有字值得读。
_DESCRIPTION_TEXT_WORDS = (
    "文字", "文本", "表格", "截图", "课表", "名单", "文档", "列表", "标题",
    "段落", "二维码", "数字", "成绩", "价格", "菜单", "账单", "行程",
    "车牌", "单号", "验证码", "字幕", "公告", "说明", "参数", "地址", "电话",
)


@dataclass(frozen=True)
class OcrResult:
    ok: bool
    text: str = ""
    error: str = ""
    page_count: int = 0


def compact_tables(text: str) -> str:
    """把 OCR 输出的 HTML 表格压成「单元格 | 单元格」，省 token 也更好读。"""
    if "<tr" not in text.lower():
        return text

    def _row(row_html: str) -> str:
        cells = [
            _HTML_TAG_PATTERN.sub("", cell).strip()
            for cell in _TABLE_CELL_PATTERN.findall(row_html)
        ]
        cells = [cell for cell in cells if cell]
        return " | ".join(cells)

    def _table(match: re.Match[str]) -> str:
        rows = [_row(row) for row in _TABLE_ROW_PATTERN.findall(match.group(1))]
        body = "\n".join(row for row in rows if row)
        return "\n" + body + "\n" if body else ""

    text = _TABLE_PATTERN.sub(_table, text)
    text = re.sub(r"</?table[^>]*>", "", text, flags=re.I)
    return text


def clean_ocr_text(raw: str) -> str:
    """去掉版面坐标、压缩表格、统一换行；正文原样保留。"""
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _COORD_ARRAY_PATTERN.sub("", text)
    text = compact_tables(text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_ocr_payload(payload: object, *, max_chars: int = 1200) -> str:
    """把 OCR 响应里的逐页文本拼起来；超长按字符截断。"""
    if not isinstance(payload, dict):
        return ""
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return ""
    chunks: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        text = page.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    text = clean_ocr_text("\n\n".join(chunks))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def ocr_precheck(
    *,
    hint: str = "",
    width: int = 0,
    height: int = 0,
    byte_size: int = 0,
    content_type: str = "",
) -> bool:
    """不依赖视觉模型就能判断「这张图该读字」。"""
    lowered = (hint or "").lower()
    if any(word in lowered for word in _USER_TEXT_WORDS):
        return True
    if width > 0 and height > 0:
        # 长截图：高度远大于宽度。
        if height >= width * 1.8 and height >= 600:
            return True
        # 高分辨率大图，多半是文档/表格截图。
        if width * height >= 1_500_000 and byte_size >= 300_000:
            return True
    # QQ 截图几乎都是 PNG，体积明显偏大时补一次 OCR。
    if content_type == "image/png" and byte_size >= 400_000:
        return True
    return False


def description_mentions_text(description: str) -> bool:
    """视觉模型说图里有字，就补一次 OCR。"""
    text = description or ""
    return any(word in text for word in _DESCRIPTION_TEXT_WORDS)


def flatten_ocr_text(text: str) -> str:
    """压成一行塞进「[图片内容：…]」。

    IMAGE_NOTE_PATTERN 以第一个 ] 收尾，所以正文里的方括号必须先换掉。
    """
    value = (text or "").replace("\r", "\n")
    value = value.replace("[", "【").replace("]", "】")
    value = re.sub(r"\s*\n\s*", " / ", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def compose_image_note(description: str, ocr_text: str) -> str:
    """把「看到的」和「读到的」拼成一条看图记录。"""
    parts: list[str] = []
    seen = " ".join((description or "").split())
    if seen:
        parts.append(seen)
    read = flatten_ocr_text(ocr_text)
    if read:
        parts.append("图内文字：" + read)
    return " ｜ ".join(parts)


def _ocr_url(base_url: str) -> str:
    base = (base_url or "https://api.deepseek.com").strip().rstrip("/")
    if base.endswith("/v1"):
        return base + OCR_PATH[len("/v1"):]
    return base + OCR_PATH


def _error_detail(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "").strip()[:160]
    return str(error or "").strip()[:160]


async def ocr_image(
    data: bytes,
    *,
    filename: str = "image.png",
    content_type: str = "image/png",
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
    timeout: float = 30.0,
    max_chars: int = 1200,
) -> OcrResult:
    """识别图片里的文字。失败只返回 ok=False，不抛异常。"""
    if not data:
        return OcrResult(ok=False, error="图片为空")
    key = (api_key or "").strip()
    if not key:
        return OcrResult(ok=False, error="OCR 没有配置：缺少 MAODOU_SEARCH_API_KEY")
    url = _ocr_url(base_url)
    try:
        async with httpx2.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                data={"purpose": PURPOSE_TEXT},
                files={
                    "file": (
                        filename or "image.png",
                        data,
                        content_type or "application/octet-stream",
                    )
                },
            )
    except httpx2.TimeoutException:
        return OcrResult(ok=False, error="OCR 超时")
    except httpx2.HTTPError as exc:
        return OcrResult(ok=False, error=f"OCR 请求失败：{exc}")

    if response.status_code >= 400:
        detail = ""
        try:
            detail = _error_detail(response.json())
        except Exception:
            detail = ""
        return OcrResult(
            ok=False,
            error=f"OCR 返回 {response.status_code}" + (f"：{detail}" if detail else ""),
        )

    try:
        payload = response.json()
    except Exception:
        return OcrResult(ok=False, error="OCR 返回了无法解析的内容")

    detail = _error_detail(payload)
    if detail:
        return OcrResult(ok=False, error=detail)

    text = parse_ocr_payload(payload, max_chars=max_chars)
    page_count = 0
    if isinstance(payload, dict):
        try:
            page_count = int(payload.get("page_count") or 0)
        except (TypeError, ValueError):
            page_count = 0
    if not text:
        return OcrResult(ok=False, error="图里没有识别到文字", page_count=page_count)
    return OcrResult(ok=True, text=text, page_count=page_count)
