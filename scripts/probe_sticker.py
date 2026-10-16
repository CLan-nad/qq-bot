"""一次性探针：把群里发过的大表情原图当图片发到群里，看 QQ 收不收、显示成什么样。

用法： .venv\\Scripts\\python.exe scripts\\probe_sticker.py --url "<附件 url>"
只发消息，不改代码、不写数据库。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
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


def sniff(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return "unknown"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="大表情附件地址")
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--caption", default="【表情包测试·忽略我】这是群里发过的那个表情的原图")
    args = parser.parse_args()

    env = load_env()
    async with httpx2.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        image = await client.get(args.url)
        print("[下载]", image.status_code, len(image.content), sniff(image.content))
        if image.status_code != 200 or not image.content:
            return 1

        token_response = await client.post(
            TOKEN_URL,
            json={"appId": env["QQ_APP_ID"], "clientSecret": env["QQ_APP_SECRET"]},
        )
        token = token_response.json().get("access_token")
        if not token:
            print("拿不到 access_token:", token_response.status_code, token_response.text)
            return 1
        headers = {"Authorization": f"QQBot {token}"}

        upload = await client.post(
            f"{API}/v2/groups/{args.group}/files",
            headers=headers,
            json={
                "file_type": 1,
                "file_data": base64.b64encode(image.content).decode("ascii"),
                "srv_send_msg": False,
            },
        )
        print("[上传]", upload.status_code, upload.text[:300])
        file_info = (upload.json() or {}).get("file_info") or ""
        if not file_info:
            return 1

        sent = await client.post(
            f"{API}/v2/groups/{args.group}/messages",
            headers=headers,
            json={
                "msg_type": 7,
                "content": args.caption,
                "media": {"file_info": file_info},
            },
        )
        print("[发送]", sent.status_code, sent.text[:300])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
