"""把小毛豆本尊（大耳帽兜）的图装成表情包（本地跑，产物提交进仓库）。

开源 emoji 图源里没有这只精灵，图是 scripts/fetch_character_stickers.py 从图搜扒
下来的游戏图/官方图，所以只能先裁剪再入库：每张图配一个裁切框（原始像素坐标），
裁掉水印和 UI，缩到 240×240 再装进 assets/stickers/。

素材版权归腾讯《洛克王国》，只在我们自己的私人群里用，别外传。

用法：
  .venv\\Scripts\\python.exe scripts\\install_maodou_stickers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from sticker_manifest import write_collection


ROOT = Path(__file__).resolve().parents[1]
# 素材在 data/ 下（.gitignore 的地盘），装好的图在 assets/ 里跟代码走。
SOURCES = ROOT / "data" / "stickers"
LIBRARY = ROOT / "assets" / "stickers"

# 尺寸和 emoji 那批保持一致：QQ 群聊够看，体积也压得住。
TARGET = 240

COLLECTION = {
    "id": "maodou",
    "source": "《洛克王国》游戏内形象（大耳帽兜）",
    "license": "版权归腾讯，仅供私人群聊使用，请勿外传",
    "note": "由 scripts/fetch_character_stickers.py 扒图后人工裁切，未做重绘。",
}
PREFIX = "m_"

# (源图, 裁切框, 名字, emoji, 标签, 文件名)
# 名字都带「本尊」前缀：模型一眼知道这是它自己长什么样，不会跟那 40 张通用表情混。
# 文件名一律用英文：服务器上万一碰到非 UTF-8 的 locale，中文文件名会平白多一层坑。
PICKS: list[tuple[str, tuple[int, int, int, int], str, str, list[str], str]] = [
    # 手游版立绘：站着看你，最像「本尊」证件照的一张
    ("character2/c04.png", (150, 20, 460, 330), "本尊", "🐰", ["本尊", "帽兜"], "m_benzon.png"),
    # 手游版大头：笑眯了眼
    (
        "character2/c05.png",
        (115, 15, 455, 299),
        "本尊开心",
        "😄",
        ["本尊", "开心"],
        "m_benzon_happy.png",
    ),
    # 经典版：小跑着扑过来。右边界卡在 580：耳尖到这就完了，再往右是图站的水印
    (
        "character/c17.png",
        (55, 85, 580, 548),
        "本尊来啦",
        "🏃",
        ["本尊", "跑", "开心"],
        "m_benzon_run.png",
    ),
    # 经典版：委屈巴巴
    (
        "character/c21.png",
        (200, 330, 480, 620),
        "本尊委屈",
        "🥺",
        ["本尊", "委屈", "安慰"],
        "m_benzon_sad.png",
    ),
    # 手游版：乖乖坐着等夸
    (
        "character2/c14.png",
        (0, 0, 285, 269),
        "本尊坐好",
        "🐰",
        ["本尊", "乖乖", "等夸"],
        "m_benzon_sit.png",
    ),
    # 手游版半身：探头看一眼（这张原图背景是黑的，裁不掉，就这么用）
    (
        "character2/c11.png",
        (0, 0, 584, 292),
        "本尊探头",
        "👀",
        ["本尊", "探头", "好奇"],
        "m_benzon_peek.png",
    ),
]


def _square(image: Image.Image) -> Image.Image:
    """裁完的长方形补成正方形，补的底色取四角的中位色，看上去像原图的背景。"""
    image.thumbnail((TARGET, TARGET), Image.LANCZOS)
    if image.size == (TARGET, TARGET):
        return image
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    ]
    background = tuple(
        sorted(channel[index] for channel in corners)[len(corners) // 2] for index in range(3)
    )
    canvas = Image.new("RGB", (TARGET, TARGET), background)
    canvas.paste(image, ((TARGET - image.width) // 2, (TARGET - image.height) // 2))
    return canvas


def main() -> int:
    if not SOURCES.exists():
        print(f"没有素材目录：{SOURCES}", file=sys.stderr)
        return 1
    LIBRARY.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    missing: list[str] = []
    for relative, box, name, emoji, tags, filename in PICKS:
        source = SOURCES / relative
        if not source.is_file():
            missing.append(relative)
            continue
        frame = _square(Image.open(source).convert("RGB").crop(box))
        frame.save(LIBRARY / filename, optimize=True)
        items.append({"name": name, "file": filename, "emoji": emoji, "tags": tags})

    if missing:
        print(f"! 找不到这些素材，已跳过：{', '.join(missing)}", file=sys.stderr)
    if not items:
        print("一张都没装成", file=sys.stderr)
        return 1

    write_collection(
        LIBRARY / "manifest.json",
        collection=COLLECTION,
        prefix=PREFIX,
        items=items,
    )
    total = sum((LIBRARY / item["file"]).stat().st_size for item in items)
    print(f"装好 {len(items)} 张本尊图，合计 {total // 1024}KB -> {LIBRARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
