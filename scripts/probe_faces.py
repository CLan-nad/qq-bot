"""一次性探针：验证 QQ 群消息能不能渲染表情标签 / 能不能发图片。

用法： .venv\\Scripts\\python.exe scripts\\probe_faces.py [--images]
只发消息，不改任何代码、不写数据库。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import httpx2


ROOT = Path(__file__).resolve().parents[1]
GROUP = "86FD26B0001F62602F66CF51E555C61E"
API = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """不依赖 Pillow，手搓一张纯色 PNG 当测试图。"""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


FACE_TEXT = "\n".join(
    [
        "【表情测试·忽略我】三种写法，看看哪个能变成真表情：",
        'A 小表情 <faceType=1,faceId="267">',
        'B 小表情 <faceType=1,faceId="344">',
        'C 大表情 <faceType=6,faceId="0",ext="eyJ0ZXh0IjoiW+Wkp+iDhl0ifQ==">',
    ]
)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", action="store_true", help="额外测一次图片发送")
    parser.add_argument("--group", default=GROUP)
    args = parser.parse_args()

    env = load_env()
    async with httpx2.AsyncClient(timeout=30.0) as client:
        token_response = await client.post(
            TOKEN_URL,
            json={"appId": env["QQ_APP_ID"], "clientSecret": env["QQ_APP_SECRET"]},
        )
        token = token_response.json().get("access_token")
        if not token:
            print("拿不到 access_token:", token_response.status_code, token_response.text)
            return 1
        headers = {"Authorization": f"QQBot {token}"}

        sent = await client.post(
            f"{API}/v2/groups/{args.group}/messages",
            headers=headers,
            json={"content": FACE_TEXT, "msg_type": 0},
        )
        print("[文本+表情标签]", sent.status_code, sent.text[:400])

        if not args.images:
            return 0

        png = solid_png(240, 240, (255, 179, 102))
        upload = await client.post(
            f"{API}/v2/groups/{args.group}/files",
            headers=headers,
            json={
                "file_type": 1,
                "file_data": base64.b64encode(png).decode("ascii"),
                "srv_send_msg": False,
            },
        )
        print("[上传图片]", upload.status_code, upload.text[:400])
        payload = upload.json()
        file_info = payload.get("file_info") or ""
        if not file_info:
            return 1

        media = await client.post(
            f"{API}/v2/groups/{args.group}/messages",
            headers=headers,
            json={"msg_type": 7, "content": " ", "media": {"file_info": file_info}},
        )
        print("[发图片]", media.status_code, media.text[:400])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
