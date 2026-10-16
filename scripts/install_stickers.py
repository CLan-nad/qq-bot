"""把挑好的候选图装成正式表情库（本地跑一次，产物提交进仓库）。

素材不能只放在 data/ 里：那是 .gitignore 的地盘，服务器上 `git checkout -f` 拿不到，
又会出现「本地和服务器两套东西」。所以正式库放在 assets/stickers/，跟代码一起走 git。

用法：
  .venv\\Scripts\\python.exe scripts\\install_stickers.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from sticker_manifest import write_collection


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "stickers" / "candidates"
LIBRARY = ROOT / "assets" / "stickers"

COLLECTION = {
    "id": "emoji",
    "source": "Google Noto Animated Emoji",
    "license": "Apache-2.0",
    "note": "动图由 scripts/fetch_stickers.py 取自官方开源 emoji 素材，未改动内容，仅压缩尺寸。",
}
# 这一批的文件名前缀：删掉自己这一批旧条目时靠它认人。
PREFIX = "a_"

# 最终名单：(候选文件, 名字, 对应的 emoji, 标签)。
# 名字是模型唯一要记住的东西——写 [抱抱] 就发这张图，所以都得短、准、不打架：
# 不跟群里已经学会的小表情重名（头秃/笑哭/流泪/托腮这些），免得它以为在发小表情。
PICKS: list[tuple[str, str, str, list[str]]] = [
    ("a_1f970.gif", "爱心眼", "🥰", ["喜欢", "撒娇"]),
    ("a_1f618.gif", "飞吻", "😘", ["亲亲", "撒娇"]),
    ("a_1f60d.gif", "崇拜", "😍", ["喜欢", "夸"]),
    ("a_1f917.gif", "抱抱", "🤗", ["抱抱", "撒娇"]),
    ("a_1fac2.gif", "贴贴", "🫂", ["抱抱", "安慰"]),
    ("a_1faf6.gif", "比心", "🫶", ["喜欢", "比心"]),
    ("a_2764_fe0f.gif", "爱心", "❤️", ["喜欢"]),
    ("a_1f97a.gif", "求求了", "🥺", ["撒娇", "委屈"]),
    ("a_1f979.gif", "感动", "🥹", ["感动"]),
    ("a_1f62d.gif", "大哭", "😭", ["哭", "委屈"]),
    ("a_1f923.gif", "笑到打滚", "🤣", ["开心", "笑"]),
    ("a_1f601.gif", "嘿嘿", "😁", ["开心"]),
    ("a_1f60a.gif", "微笑", "😊", ["开心", "温和"]),
    ("a_1f389.gif", "撒花", "🎉", ["庆祝"]),
    ("a_1f44f.gif", "鼓掌", "👏", ["夸", "鼓励"]),
    ("a_1f44d.gif", "赞", "👍", ["夸", "同意"]),
    ("a_1f64c.gif", "好耶", "🙌", ["开心"]),
    ("a_1f92d.gif", "偷笑", "🤭", ["调皮", "笑"]),
    ("a_1f60f.gif", "坏笑", "😏", ["调皮", "得瑟"]),
    ("a_1f609.gif", "你懂的", "😉", ["调皮"]),
    ("a_1f633.gif", "害羞", "😳", ["害羞"]),
    ("a_1f971.gif", "困了", "🥱", ["困"]),
    ("a_1f634.gif", "睡觉", "😴", ["困", "晚安"]),
    ("a_1f914.gif", "思考", "🤔", ["思考", "装傻"]),
    ("a_1f644.gif", "无语", "🙄", ["无语", "嫌弃"]),
    ("a_1f62c.gif", "尴尬", "😬", ["尴尬"]),
    ("a_1fae0.gif", "融化了", "🫠", ["社死", "无语"]),
    ("a_1f92f.gif", "脑壳炸了", "🤯", ["头秃", "崩溃"]),
    ("a_1f631.gif", "震惊", "😱", ["震惊", "害怕"]),
    ("a_1f621.gif", "气鼓鼓", "😡", ["生气"]),
    ("a_1f620.gif", "哼", "😠", ["生气"]),
    ("a_1f480.gif", "我死了", "💀", ["崩溃", "无语"]),
    ("a_1f440.gif", "暗中观察", "👀", ["看戏", "调皮"]),
    ("a_1fae3.gif", "偷看", "🫣", ["调皮", "暗中观察"]),
    ("a_1f929.gif", "哇塞", "🤩", ["夸", "惊喜"]),
    ("a_1fae1.gif", "收到", "🫡", ["答应", "认真"]),
    ("a_1f407.gif", "蹦跶", "🐇", ["动物", "可爱"]),
    ("a_1f431.gif", "喵", "🐱", ["动物", "可爱"]),
    ("a_1f423.gif", "小鸡", "🐣", ["动物", "可爱"]),
    ("a_1f438.gif", "呱", "🐸", ["动物", "可爱"]),
]


def main() -> int:
    if not CANDIDATES.exists():
        print(f"没有候选目录：{CANDIDATES}，先跑 scripts/fetch_stickers.py", file=sys.stderr)
        return 1
    LIBRARY.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    missing: list[str] = []
    for filename, name, emoji, tags in PICKS:
        source = CANDIDATES / filename
        if not source.exists():
            missing.append(filename)
            continue
        shutil.copy2(source, LIBRARY / filename)
        items.append(
            {
                "name": name,
                "file": filename,
                "emoji": emoji,
                "tags": tags,
            }
        )

    if missing:
        print(f"! 候选里缺这些文件，已跳过：{', '.join(missing)}", file=sys.stderr)

    write_collection(
        LIBRARY / "manifest.json",
        collection=COLLECTION,
        prefix=PREFIX,
        items=items,
    )
    total = sum((LIBRARY / item["file"]).stat().st_size for item in items)
    print(f"装好 {len(items)} 张，合计 {total // 1024}KB -> {LIBRARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
