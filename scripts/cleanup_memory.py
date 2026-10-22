"""一次性清理脚本：删重复记忆卡、截断超长摘要。

用法（先停服再跑，避免和 bot 抢写锁）：
    python scripts/cleanup_memory.py --db /opt/maodou/data/maodou.sqlite3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/maodou.sqlite3")
    parser.add_argument("--backup-dir", default="")
    args = parser.parse_args()

    db = Path(args.db).resolve()
    if not db.exists():
        print(f"db not found: {db}", file=sys.stderr)
        return 1

    backup_dir = Path(args.backup_dir) if args.backup_dir else db.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"cleanup-{time.strftime('%Y%m%d-%H%M%S')}.sqlite3"
    shutil.copy2(db, backup)
    print(f"backup -> {backup}")

    from maodou.store import ChatStore

    store = ChatStore(str(db))
    try:
        for group_openid in store.groups():
            removed = store.dedupe_memory_cards(group_openid)
            clamped = store.clamp_long_summaries(group_openid)
            print(f"{group_openid}: dedup -{removed}, summaries clamped {clamped}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
