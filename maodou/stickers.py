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
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .images import FACE_TAG_PATTERN, decode_face_text


logger = logging.getLogger("maodou")


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
    as_tags=False：平台不吃标签（实测 QQ 群消息就是不吃），名字发出去只是文字，
    屏幕上就是一对方括号——所以干脆删掉，让模型改用 emoji 表达情绪。
    认不出来的方括号（[1]、[神秘代码]）一律不动。
    """
    if not text or not book:
        return text or ""

    # 只有小表情才换得成标签，纠错池也只放它们：大表情的名字长成 "/笑哭" 这样，
    # 拿它去纠 [哭] 只会把好端端的字改成 "/笑哭"。
    names = tuple(name for name, token in book.items() if not token.is_sticker)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        token = book.get(raw)
        if token is None:
            fixed = _closest(raw, names)
            if not fixed:
                return match.group(0)
            token = book[fixed]
        if not as_tags or token.is_sticker:
            # 大表情得走图片通道；纯文本里带上名字只是方括号，不如不给。
            return ""
        return token.tag

    rendered = FACE_NAME_PATTERN.sub(replace, text)
    # 整条回复只有表情名时，删干净就成空消息了（QQ 会直接拒收），退回原文保底。
    if not rendered.strip() and text.strip():
        return text
    return rendered


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


EMOJI_PROMPT = (
    "表情：想表达情绪就直接写 emoji（比如 本豆要秃了😭），发出去就是真的表情。"
    "别写 [头秃] 这类方括号名字——QQ 群消息不认这种标签，发出来只是一串方括号，很难看。"
)

# 有表情包时这句话得改口：方括号不是一律不能用，只是「只有下面列出的表情包名字能用」。
# 两段话各说各的会让模型无所适从——线上就是这么教出 [秃头] 复读的。
_EMOJI_ONLY_HALF = (
    "表情：想表达情绪就直接写 emoji（比如 本豆要秃了😭），发出去就是真的表情。"
    "群里的小表情名字（[头秃] 这类）发出来只是干巴巴几个方括号，别用。"
)


def face_prompt(
    book: dict[str, FaceToken],
    *,
    as_tags: bool = True,
    stickers: "StickerLibrary | None" = None,
) -> str:
    """告诉模型表情怎么发。标签通道不通时只教它用 emoji，外加能发的表情包。

    表情包和小表情得写在同一段里：模型的注意力只够记住一套规则，分开说它就会拿
    「别写方括号」去否掉表情包的名字。
    """
    names = [name for name, token in book.items() if not token.is_sticker]
    if as_tags and names:
        listed = "、".join(f"[{name}]" for name in names)
        head = (
            f"群里出现过的表情：{listed}。想表达情绪就直接把它写进回复里（比如：本豆要秃了[头秃]），"
            "发出去会变成真的表情。偶尔带一个就好，别每句都挂；不确定名字就写你在聊天记录里"
            "见过的那一个，别自己造新名字。"
        )
    else:
        head = _EMOJI_ONLY_HALF if stickers else EMOJI_PROMPT

    extra = stickers.prompt() if stickers else ""
    return f"{head}\n{extra}" if extra else head


# ---------------------------------------------------------------- 表情包（成图）
# 小表情的标签通道走不通（QQ 群消息不吃 <faceType>），大表情又只能靠图，所以小毛豆
# 想发表情的唯一出路是「发一张图」。这些图是我们自己挑好、跟代码一起进仓库的，模型
# 眼里就一个名字：写 [爱你]，出站时被摘出来换成图片。
_IMAGE_SUFFIX_TYPES = {
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@dataclass(frozen=True)
class Sticker:
    """一张能直接发出去的图。"""

    name: str
    path: Path
    emoji: str = ""
    tags: tuple[str, ...] = ()

    def read(self) -> bytes:
        return self.path.read_bytes()

    @property
    def content_type(self) -> str:
        return _IMAGE_SUFFIX_TYPES.get(self.path.suffix.lower(), "image/gif")


class StickerLibrary:
    """assets/stickers/ 里那批图。目录没了、清单坏了都只是空库，不该拖垮机器人。"""

    def __init__(self, root: Path, stickers: dict[str, Sticker]) -> None:
        self.root = root
        self._stickers = stickers

    def __bool__(self) -> bool:
        return bool(self._stickers)

    def __len__(self) -> int:
        return len(self._stickers)

    def names(self) -> list[str]:
        return list(self._stickers)

    def get(self, name: str) -> Sticker | None:
        return self._stickers.get(name)

    def take(self, text: str | None) -> tuple[str, Sticker | None]:
        """把回复里第一个认识的表情名摘出来，返回 (剩下的正文, 要发的图)。

        一条回复最多发一张：QQ 那边每张图都是一次上传 + 一条消息，发多了刷屏，也不像
        人在说话。名字只做精确匹配，不走纠错——纠错是给小表情的，套在图名上容易把
        好好一句话改成别的表情。
        """
        if not text or not self._stickers:
            return text or "", None

        found: Sticker | None = None

        def replace(match: re.Match[str]) -> str:
            nonlocal found
            sticker = self._stickers.get(match.group(1))
            if sticker is None or found is not None:
                return match.group(0)
            found = sticker
            return ""

        remainder = FACE_NAME_PATTERN.sub(replace, text)
        if found is None:
            return text, None
        # 摘掉中间那个名字会留下两个空格，顺手收干净；整条只有表情名时留空串，
        # 调用方会把它当「这条只发图」处理。
        return re.sub(r"[ \t]{2,}", " ", remainder).strip(), found

    def prompt(self) -> str:
        if not self._stickers:
            return ""
        listed = "、".join(
            f"[{sticker.name}]{sticker.emoji}" for sticker in self._stickers.values()
        )
        example = "爱你" if "爱你" in self._stickers else next(iter(self._stickers))
        return (
            f"表情包（你可以直接发图）：{listed}。想发就先写想说的话，再在句末带上名字，"
            f"比如「想你了[{example}]」；文字和图片会分成两条发。\n"
            "规矩：一条回复最多发一张；只在真的想表达那个情绪时发（开心、撒娇、害羞、委屈、"
            "生气、无语、困、想安慰人、想夸人都合适），平时照旧用文字和 emoji，别每条都挂图；"
            "名字必须从上面挑，别自己造，也别把方括号写成别的样子。"
        )

    @classmethod
    def load(cls, root: str | Path) -> "StickerLibrary":
        base = Path(root)
        empty = cls(base, {})
        manifest_path = base / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return empty
        except Exception:
            logger.exception("sticker manifest unreadable: %s", manifest_path)
            return empty

        stickers: dict[str, Sticker] = {}
        for item in (payload or {}).get("items") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            filename = str(item.get("file") or "").strip()
            if not name or not filename:
                continue
            path = base / filename
            if not path.is_file():
                logger.warning("sticker file missing, skipped: %s", path)
                continue
            tags = item.get("tags") or []
            stickers[name] = Sticker(
                name=name,
                path=path,
                emoji=str(item.get("emoji") or "").strip(),
                tags=tuple(str(tag) for tag in tags if str(tag).strip()),
            )
        return cls(base, stickers)
