"""表情包清单的读写（两个取材脚本共用）。

清单是一份 items + 若干 collections：items 是机器人真正读的那部分（名字 → 文件），
collections 记的是每批素材从哪来、什么许可，方便以后翻账。

每批素材管自己文件名前缀（emoji 用 a_/b_，本尊用 m_）：重跑某一批只会换掉它自己
那些条目，不会把另一批的图从清单里抹掉。
"""
from __future__ import annotations

import json
from pathlib import Path


def load_items(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("items") or [])


def write_collection(
    manifest_path: Path,
    *,
    collection: dict,
    prefix: str,
    items: list[dict],
) -> list[dict]:
    """用新的一批条目替换掉同前缀的旧条目，其余原样留着。"""
    kept = [item for item in load_items(manifest_path) if not str(item.get("file", "")).startswith(prefix)]
    collections = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        collections = [
            entry
            for entry in payload.get("collections") or []
            if entry.get("id") != collection["id"]
        ]
    merged = kept + list(items)
    manifest_path.write_text(
        json.dumps(
            {"version": 1, "collections": collections + [collection], "items": merged},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return merged
