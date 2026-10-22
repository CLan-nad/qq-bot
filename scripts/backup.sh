#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db="$root/data/maodou.sqlite3"
backup_dir="$root/backups"

if [[ ! -f "$db" ]]; then
  echo "Database not found: $db" >&2
  exit 1
fi

mkdir -p "$backup_dir"
target="$backup_dir/maodou-$(date +%Y%m%d-%H%M%S).sqlite3"
sqlite3 "$db" ".backup '$target'"
find "$backup_dir" -type f \( -name 'maodou-*.sqlite3' -o -name 'xiaoxia-*.sqlite3' \) -mtime +14 -delete

echo "Backup created: $target"
