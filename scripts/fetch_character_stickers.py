"""给某个人物找素材图（本地跑一次，产物只用来过目挑选）。

开源 emoji 图源里没有「大耳帽兜本尊」，只能拿图搜去扒，所以单独一个脚本：查几条
关键词、把命中的图拉下来、去重、拼成带编号的预览总图，人看着挑。

用法：
  .venv\\Scripts\\python.exe scripts\\fetch_character_stickers.py --query "洛克王国 大耳帽兜"
产出：data/stickers/character/ 下的原图 + sheet.png 预览总图（data/ 不进仓库）。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import httpx2

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - 只有本地取材需要
    print("需要 Pillow：.venv\\Scripts\\python.exe -m pip install pillow", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parents[1]
SEARCH_URL = "https://api.bochaai.com/v1/web-search"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.baidu.com/"}

# 预览总图里每张图的格子。太小看不清脸，太大一张 sheet 装不下多少张。
CELL = 220
COLUMNS = 6


def _env_key() -> str:
    """BOCHA_API_KEY 从 .env 读，和机器人用同一把钥匙。"""
    key = os.getenv("BOCHA_API_KEY", "").strip()
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("BOCHA_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("BOCHA_API_KEY 没配：先写进 .env 或环境变量")


def _queries(base: str) -> list[str]:
    """同一个角色多换几种说法：光搜名字大多是攻略站截图，加「图片/立绘」才出素材图。"""
    return [base, f"{base} 图片", f"{base} 立绘", f"{base} 表情", f"{base} 精灵"]


def _search(client: httpx2.Client, key: str, query: str) -> list[str]:
    response = client.post(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "count": 10, "summary": False},
    )
    response.raise_for_status()
    images = (response.json().get("data", {}) or {}).get("images", {}) or {}
    urls: list[str] = []
    for item in images.get("value", []) or []:
        url = (item or {}).get("contentUrl")
        if url:
            urls.append(url)
    return urls


def _fetch_image(client: httpx2.Client, url: str) -> Image.Image | None:
    try:
        response = client.get(url, follow_redirects=True, headers=HEADERS)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except Exception as exc:  # 图源站挂图、防盗链、连接超时都只是这一张没了
        print(f"  ! {type(exc).__name__}: {url[:90]}", file=sys.stderr)
        return None
    return image


def _sheet(images: list[tuple[str, Image.Image]], out: Path) -> None:
    rows = (len(images) + COLUMNS - 1) // COLUMNS
    canvas = Image.new("RGB", (COLUMNS * CELL, rows * CELL), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (_, image) in enumerate(images):
        frame = image.copy()
        frame.thumbnail((CELL - 26, CELL - 26))
        x = (index % COLUMNS) * CELL
        y = (index // COLUMNS) * CELL
        canvas.paste(frame, (x + (CELL - frame.width) // 2, y + 22))
        draw.text((x + 4, y + 4), f"#{index}", fill="red")
        draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline="#cccccc")
    canvas.save(out)
    print(f"预览总图：{out}（共 {len(images)} 张）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="角色名，比如「洛克王国 大耳帽兜」")
    parser.add_argument("--out", default=str(ROOT / "data" / "stickers" / "character"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = _env_key()

    urls: list[str] = []
    with httpx2.Client(timeout=args.timeout) as client:
        for query in _queries(args.query):
            found = _search(client, key, query)
            print(f"搜索「{query}」→ {len(found)} 条")
            urls.extend(found)

        kept: list[tuple[str, Image.Image]] = []
        digests: set[str] = set()
        for url in urls:
            image = _fetch_image(client, url)
            if image is None or min(image.size) < 80:
                continue
            # 同一个站的同一张图会换域名重复出现，按像素算个指纹去重
            digest = hashlib.sha256(image.convert("RGB").resize((64, 64)).tobytes()).hexdigest()
            if digest in digests:
                continue
            digests.add(digest)
            kept.append((url, image))

    if not kept:
        print("一张都没拉到", file=sys.stderr)
        return 1

    manifest: list[dict] = []
    for index, (url, image) in enumerate(kept):
        name = f"c{index:02d}.png"
        image.convert("RGB").save(out / name)
        manifest.append({"index": index, "file": name, "size": list(image.size), "url": url})
    (out / "manifest.json").write_text(
        json.dumps({"query": args.query, "items": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _sheet(kept, out / "sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
