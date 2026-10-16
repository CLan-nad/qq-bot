"""素材取材脚本（只在本地跑，服务器不需要）。

小毛豆想发表情，QQ 群却只吃「图片」这一条通道，所以得先有一批图。这个脚本从两个
开源图源拉素材、压到适合群聊的体积，并生成一张带编号的预览总图，方便人过目挑选：

  A 批  Google Noto Animated Emoji（Apache-2.0，512x512 动图）
  B 批  Microsoft Fluent Emoji 3D（MIT，静态高清）

用法：
  .venv\\Scripts\\python.exe scripts\\fetch_stickers.py --out data\\stickers\\candidates
产出：candidates/ 下的图片 + manifest.json + sheet_A.png / sheet_B.png 预览总图。
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import httpx2

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - 只有本地取材需要
    print("需要 Pillow：.venv\\Scripts\\python.exe -m pip install pillow", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parents[1]

NOTO = "https://fonts.gstatic.com/s/e/notoemoji/latest/{cp}/512.gif"
FLUENT = "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/{folder}/3D/{stem}_3d.png"

NOTO_LICENSE = "Noto Emoji (Apache-2.0)"
FLUENT_LICENSE = "Fluent Emoji (MIT)"

# 大表情发出去是「一张图」，所以尺寸别贪大：240 够清楚，体积也压得住。
TARGET_SIZE = 240
# 单张上限，超过就抽帧。QQ 图片上限 20MB，这里留一大截余量。
MAX_BYTES = 400 * 1024


# ---------------------------------------------------------------- A 批：动图表情
# 覆盖一个「可爱、调皮、会撒娇」的搭子需要表达的情绪：开心、喜欢、撒娇、抱抱、
# 委屈、哭、生气、无语、困、震惊、比心、暗中观察……动物脸那几个是为大耳帽兜
# 这个形象挑的：兔子、小猫、小鸡、青蛙都在同一套画风里，不会打架。
NOTO_PICKS: list[tuple[str, str, str, list[str]]] = [
    ("1f970", "爱心眼", "喜欢", ["喜欢", "心动"]),
    ("1f618", "飞吻", "mua", ["亲亲", "撒娇"]),
    ("1f60d", "星星眼", "崇拜", ["喜欢", "崇拜"]),
    ("1f917", "抱抱脸", "贴贴", ["抱抱", "撒娇"]),
    ("1fac2", "两个人抱抱", "抱一下", ["抱抱", "安慰"]),
    ("1faf6", "比心", "比心", ["喜欢", "比心"]),
    ("2764_fe0f", "红心", "爱心", ["喜欢", "爱心"]),
    ("1f495", "两颗心", "爱心两颗", ["喜欢"]),
    ("1f496", "闪亮的心", "亮晶晶", ["喜欢", "撒娇"]),
    ("1f97a", "水汪汪的大眼", "求求了", ["委屈", "撒娇"]),
    ("1f979", "忍泪", "感动", ["感动", "委屈"]),
    ("1f62d", "大哭", "大哭", ["哭", "委屈"]),
    ("1f622", "流泪", "流泪", ["哭"]),
    ("1f602", "笑哭", "笑哭", ["开心", "笑"]),
    ("1f923", "翻滚大笑", "哈哈哈", ["开心", "笑"]),
    ("1f601", "咧嘴笑", "嘿嘿", ["开心"]),
    ("1f60a", "微笑", "微笑", ["开心", "温和"]),
    ("1f973", "撒花庆祝", "庆祝", ["开心", "庆祝"]),
    ("1f389", "礼炮", "撒花", ["庆祝"]),
    ("1f44f", "鼓掌", "鼓掌", ["夸", "鼓励"]),
    ("1f44d", "点赞", "赞", ["夸", "同意"]),
    ("1f64c", "举手欢呼", "好耶", ["开心"]),
    ("1f92d", "捂嘴笑", "偷笑", ["调皮", "笑"]),
    ("1f60f", "坏笑", "坏笑", ["调皮", "得瑟"]),
    ("1f609", "眨眼", "你懂的", ["调皮"]),
    ("1f633", "脸红", "害羞", ["害羞"]),
    ("1f971", "打哈欠", "困了", ["困"]),
    ("1f634", "睡觉", "睡觉", ["困", "晚安"]),
    ("1f914", "托腮思考", "思考", ["思考", "装傻"]),
    ("1f644", "翻白眼", "无语", ["无语", "嫌弃"]),
    ("1f62c", "龇牙尴尬", "尴尬", ["尴尬"]),
    ("1fae0", "融化", "融化了", ["社死", "无语"]),
    ("1f92f", "爆炸头", "脑壳炸了", ["头秃", "崩溃"]),
    ("1f631", "尖叫", "震惊", ["震惊", "害怕"]),
    ("1f621", "气鼓鼓", "生气", ["生气"]),
    ("1f620", "生气", "哼", ["生气"]),
    ("1f480", "骷髅", "我死了", ["崩溃", "无语"]),
    ("1f440", "眼睛", "暗中观察", ["看戏", "调皮"]),
    ("1fae3", "偷看", "偷看", ["调皮", "暗中观察"]),
    ("1f929", "星星眼惊叹", "哇塞", ["夸", "惊喜"]),
    ("1fae1", "敬礼", "收到", ["答应", "认真"]),
    ("1f407", "小兔子", "蹦跶", ["动物", "可爱"]),
    ("1f431", "猫脸", "喵", ["动物", "可爱"]),
    ("1f423", "破壳小鸡", "小鸡", ["动物", "可爱"]),
    ("1f438", "青蛙脸", "呱", ["动物", "可爱"]),
]

# ---------------------------------------------------------------- B 批：静态高清
# 动图之外留一档静态图：不会被抽帧、体积小，适合当「发张图给你看」的那种。
FLUENT_PICKS: list[tuple[str, str, str, str, list[str]]] = [
    ("Smiling face with hearts", "smiling_face_with_hearts", "爱心脸", "喜欢", ["喜欢"]),
    ("Smiling face with heart-eyes", "smiling_face_with_heart-eyes", "花痴脸", "喜欢", ["喜欢"]),
    ("Face blowing a kiss", "face_blowing_a_kiss", "亲亲", "mua", ["亲亲"]),
    ("Hugging face", "hugging_face", "抱抱", "抱抱", ["抱抱"]),
    ("Pleading face", "pleading_face", "狗狗眼", "求求", ["委屈", "撒娇"]),
    ("Face holding back tears", "face_holding_back_tears", "憋泪", "感动", ["感动"]),
    ("Grinning face with sweat", "grinning_face_with_sweat", "尴尬笑", "汗", ["尴尬"]),
    ("Melting face", "melting_face", "融化", "融了", ["社死"]),
    ("Smiling face with sunglasses", "smiling_face_with_sunglasses", "墨镜", "得瑟", ["调皮"]),
    ("Face with hand over mouth", "face_with_hand_over_mouth", "偷笑", "偷笑", ["调皮"]),
    ("Rabbit face", "rabbit_face", "兔兔脸", "兔兔", ["动物", "可爱"]),
    ("Rabbit", "rabbit", "小兔子", "兔子", ["动物", "可爱"]),
    ("Cat face", "cat_face", "猫猫", "喵喵", ["动物", "可爱"]),
    ("Dog face", "dog_face", "狗狗", "汪汪", ["动物", "可爱"]),
    ("Hatching chick", "hatching_chick", "破壳鸡", "小鸡", ["动物", "可爱"]),
    ("Kiss mark", "kiss_mark", "亲亲印", "啵", ["亲亲"]),
    ("Sleeping face", "sleeping_face", "睡脸", "睡着了", ["困", "晚安"]),
    ("Red heart", "red_heart", "红心", "爱心", ["爱心"]),
    ("Sparkling heart", "sparkling_heart", "闪光心", "爱心", ["爱心"]),
]


# GIF 只有 1 位透明度，留出最高索引当透明色。255 色量化之后这一格是空着的。
TRANSPARENT_INDEX = 255
ALPHA_CUTOFF = 128


def _to_palette_frame(frame: Image.Image) -> Image.Image:
    """RGBA 一帧 → 带透明索引的调色板帧（不然透明区会被填成黑色方块）。"""
    alpha = frame.getchannel("A")
    solid = Image.new("RGB", frame.size, (255, 255, 255))
    solid.paste(frame, (0, 0), alpha)
    paletted = solid.quantize(colors=TRANSPARENT_INDEX, method=Image.MEDIANCUT)
    paletted.paste(TRANSPARENT_INDEX, alpha.point(lambda value: 255 if value < ALPHA_CUTOFF else 0))
    paletted.info["transparency"] = TRANSPARENT_INDEX
    return paletted


def _render_gif(frames: list, duration: int) -> bytes:
    quantized = [_to_palette_frame(frame) for frame in frames]
    buffer = io.BytesIO()
    quantized[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=duration,
        loop=0,
        disposal=2,
        transparency=TRANSPARENT_INDEX,
        optimize=True,
    )
    return buffer.getvalue()


def encode_gif(raw: bytes) -> bytes:
    """把动图压到群聊友好：缩小到 TARGET_SIZE，去掉全透明首帧，太大了就抽帧。"""
    source = Image.open(io.BytesIO(raw))
    duration = int(source.info.get("duration") or 80) or 80
    frames = []
    try:
        while True:
            frames.append(source.convert("RGBA").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    except Exception:
        pass
    if not frames:
        raise ValueError("动图没有可用的帧")

    # Noto 有几张是淡入的，首帧整个是空的。QQ 里闪过一帧空白很难看，直接掐掉。
    while len(frames) > 1 and not frames[0].getchannel("A").getbbox():
        frames.pop(0)

    payload = b""
    for step in (1, 2, 3, 4, 6):
        picked = frames[::step][: max(6, 48 // step)]
        payload = _render_gif(picked, duration * step)
        if len(payload) <= MAX_BYTES:
            break
    return payload


def encode_png(raw: bytes) -> bytes:
    source = Image.open(io.BytesIO(raw)).convert("RGBA")
    source.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


async def download(client: httpx2.AsyncClient, url: str) -> bytes | None:
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        print(f"  ! 拉取失败 {url} ({type(exc).__name__})", file=sys.stderr)
        return None
    return response.content


async def collect(
    client: httpx2.AsyncClient,
    gate: asyncio.Semaphore,
    *,
    url: str,
    target: Path,
    encode,
    force: bool,
) -> bytes | None:
    """下载 + 转码。已经在本地且没要求重来，就直接用现成的，省得反复拉。"""
    if not force and target.exists() and target.stat().st_size > 0:
        return target.read_bytes()
    async with gate:
        raw = await download(client, url)
    if raw is None:
        return None
    try:
        payload = await asyncio.to_thread(encode, raw)
    except Exception as exc:
        print(f"  ! 转码失败 {target.name}: {exc}", file=sys.stderr)
        return None
    target.write_bytes(payload)
    return payload


def sheet(entries: list[tuple[str, Path]], out: Path, *, columns: int = 8) -> None:
    """带编号的预览总图，方便一句话点单（比如「A 3、7、12 要，18 不要」）。"""
    cell = TARGET_SIZE + 18
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, rows * (cell + 16)), (250, 250, 252))
    draw = ImageDraw.Draw(canvas)
    for position, (label, path) in enumerate(entries):
        column, row = position % columns, position // columns
        x, y = column * cell + 9, row * (cell + 16) + 9
        try:
            image = _best_frame(path)
            backdrop = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255, 255))
            backdrop.alpha_composite(image)
            canvas.paste(backdrop.convert("RGB"), (x, y))
        except Exception:
            pass
        draw.text((x, y + TARGET_SIZE + 2), label, fill=(30, 30, 40))
    canvas.save(out, format="PNG", optimize=True)


def _best_frame(path: Path) -> Image.Image:
    """预览用「内容最满」的那一帧：动图首帧常常是空的，拿它当封面等于一片白。"""
    source = Image.open(path)
    best: Image.Image | None = None
    best_area = -1
    try:
        while True:
            frame = source.convert("RGBA").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            box = frame.getchannel("A").getbbox()
            area = 0 if box is None else (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best, best_area = frame, area
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    except Exception:
        pass
    if best is None:
        best = source.convert("RGBA").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return best


async def run(out: Path, *, force: bool = False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    sheets: dict[str, list[tuple[str, Path]]] = {"A": [], "B": []}

    # 文件名按素材本身取（不带序号），这样改名单重跑时已经下好的不会被拉第二遍。
    jobs: list[dict] = []
    for codepoint, label, name, tags in NOTO_PICKS:
        jobs.append(
            {
                "batch": "A",
                "slug": f"a_{codepoint}",
                "ext": ".gif",
                "label": label,
                "name": name,
                "tags": tags,
                "url": NOTO.format(cp=codepoint),
                "license": NOTO_LICENSE,
                "encode": encode_gif,
            }
        )
    for folder, stem, label, name, tags in FLUENT_PICKS:
        jobs.append(
            {
                "batch": "B",
                "slug": f"b_{stem}",
                "ext": ".png",
                "label": label,
                "name": name,
                "tags": tags,
                "url": FLUENT.format(folder=folder.replace(" ", "%20"), stem=stem),
                "license": FLUENT_LICENSE,
                "encode": encode_png,
            }
        )

    gate = asyncio.Semaphore(6)
    async with httpx2.AsyncClient(timeout=60.0) as client:
        payloads = await asyncio.gather(
            *(
                collect(
                    client,
                    gate,
                    url=job["url"],
                    target=out / f"{job['slug']}{job['ext']}",
                    encode=job["encode"],
                    force=force,
                )
                for job in jobs
            )
        )

    counters = {"A": 0, "B": 0}
    for job, payload in zip(jobs, payloads):
        if not payload:
            continue
        counters[job["batch"]] += 1
        key = f"{job['batch'].lower()}{counters[job['batch']]:02d}"
        target = out / f"{job['slug']}{job['ext']}"
        sheets[job["batch"]].append((key, target))
        manifest.append(
            {
                "batch": job["batch"],
                "key": key,
                "label": job["label"],
                "name": job["name"],
                "tags": job["tags"],
                "source": job["url"],
                "license": job["license"],
                "file": target.name,
                "bytes": len(payload),
            }
        )
        print(f"  {key} {job['label']} {len(payload) // 1024}KB")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for batch, entries in sheets.items():
        if entries:
            sheet(entries, out / f"sheet_{batch}.png")
            print(f"  预览总图 sheet_{batch}.png（{len(entries)} 张）")
    total = sum(int(item["bytes"]) for item in manifest)
    print(f"共 {len(manifest)} 张，合计 {total // 1024}KB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "stickers" / "candidates"))
    parser.add_argument("--force", action="store_true", help="已经下过的也重新拉一遍")
    args = parser.parse_args()
    return asyncio.run(run(Path(args.out), force=args.force))


if __name__ == "__main__":
    sys.exit(main())
