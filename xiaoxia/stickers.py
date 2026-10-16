"""表情：群里见过哪些表情、怎么把模型写的 [头秃] 变成真的表情。

QQ 群里表情是二进制标签（<faceType=1,faceId="267",ext="...">），入库时已经解码成
可读的 [头秃]，所以模型只看得到名字。反过来发的时候就麻烦了：模型写 [头秃] 只是
四个字，发出去就是干巴巴的文本——线上出现过它写 [秃头]，还被自己复读了一整天。

这里只做两件事：
1. 记住群里出现过的表情名 → 标签（名字和 faceId 的对应关系全部来自真实消息，不写死）；
2. 出站时把模型写的 [名字] 换成标签；名字写错一个字就顺手纠回真实的那个。
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable

from .images import FACE_TAG_PATTERN, decode_face_text


# 模型表达情用的方括号写法：[头秃]、[好耶]。限制长度和字符，免得把 [[1]] 这种
# markdown 语法也当成表情名去做纠错。
FACE_NAME_PATTERN = re.compile(r"\[([^\[\]\s，。！？、；：,.!?;:]{1,8})\]")
# 纠错只在「差得很小」时动手，避免把别人真说的话改成别的表情名。
FACE_NAME_MATCH_CUTOFF = 0.5


@dataclass(frozen=True)
class FaceToken:
    """一个真实出现过的表情。"""

    name: str
    face_type: int
    face_id: str
    ext: str = ""

    @property
    def tag(self) -> str:
        """还原成 QQ 能认的原始标签。"""
        head = f'<faceType={self.face_type},faceId="{self.face_id}"'
        if self.ext:
            return f'{head},ext="{self.ext}">'
        return f"{head}>"

    @property
    def is_sticker(self) -> bool:
        """大表情（faceType=6）本质是一张图，发的时候得走富媒体。"""
        return self.face_type == 6


def face_tokens(content: str | None) -> list[FaceToken]:
    """从原始消息体里挖出所有表情标签。"""
    if not content:
        return []
    tokens: list[FaceToken] = []
    for match in FACE_TAG_PATTERN.finditer(content):
        name = decode_face_text(match.group(3) or "").strip().strip("[]")
        if not name:
            continue
        tokens.append(
            FaceToken(
                name=name,
                face_type=int(match.group(1)),
                face_id=match.group(2),
                ext=match.group(3) or "",
            )
        )
    return tokens


def face_book(tokens: Iterable[FaceToken]) -> dict[str, FaceToken]:
    """名字 → 表情。同名多个时优先留小表情：它不需要额外的图片通道。"""
    book: dict[str, FaceToken] = {}
    for token in tokens:
        current = book.get(token.name)
        if current is None or (current.is_sticker and not token.is_sticker):
            book[token.name] = token
    return book


def _closest(name: str, names: Iterable[str]) -> str:
    """模型把表情名写岔了（[秃头] vs [头秃]）时找回真实名字；差太多就返回空串。"""
    matches = difflib.get_close_matches(name, list(names), n=1, cutoff=FACE_NAME_MATCH_CUTOFF)
    return matches[0] if matches else ""


def render_faces(
    text: str | None,
    book: dict[str, FaceToken],
    *,
    as_tags: bool = True,
) -> str:
    """出站前处理方括号表情名。

    as_tags=True：认识的换成真表情标签，发出去就是表情；
    as_tags=False：换不成表情（平台不吃标签）时至少把名字纠正成真实存在的那个，
    别让 [秃头] 这种查无此表的名字又被复读一轮。
    """
    if not text or not book:
        return text or ""

    names = tuple(book)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        token = book.get(raw)
        if token is None:
            fixed = _closest(raw, names)
            if not fixed:
                return match.group(0)
            token = book[fixed]
        if not as_tags or token.is_sticker:
            # 大表情得走图片通道，纯文本里只保留可读名字，由调用方决定怎么发图。
            return f"[{token.name}]"
        return token.tag

    return FACE_NAME_PATTERN.sub(replace, text)


def face_names_in(text: str | None, book: dict[str, FaceToken]) -> list[str]:
    """模型这条回复里点了哪些表情（用来判断要不要补发图片）。"""
    if not text:
        return []
    found: list[str] = []
    for match in FACE_NAME_PATTERN.finditer(text):
        raw = match.group(1)
        name = raw if raw in book else _closest(raw, book)
        if name and name not in found:
            found.append(name)
    return found


def face_prompt(book: dict[str, FaceToken]) -> str:
    """告诉模型有哪些表情能用。一个都没见过时返回空串，不给它添乱。"""
    if not book:
        return ""
    names = "、".join(f"[{name}]" for name in book)
    return (
        f"群里出现过的表情：{names}。想表达情绪就直接把它写进回复里（比如：本豆要秃了[头秃]），"
        "发出去会变成真的表情；不确定名字就写你在聊天记录里见过的那一个，别自己造新名字。"
    )
