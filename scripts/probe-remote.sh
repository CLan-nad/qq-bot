#!/usr/bin/env bash
# 只读探测：不 checkout、不装依赖、不重启。供 deploy.ps1 -DryRun 使用。
set -uo pipefail

DIR="__REMOTE_DIR__"
SVC="__SERVICE__"
BRANCH="__BRANCH__"
TARGET="__TARGET__"

OWNER="$(stat -c '%U' "$DIR")"
RUNAS="sudo -u $OWNER -H"
STATE_DIR=/var/lib/xiaoxia-deploy
line() { printf -- '--- %s ---\n' "$1"; }

cd "$DIR" || { echo "FATAL: cannot enter $DIR"; exit 9; }

line "identity"
printf 'dir=%s owner=%s mode=%s git=%s\n' "$DIR" "$OWNER" \
  "$(stat -c '%a' "$DIR")" "$(command -v git || echo none)"
printf 'service=%s state=%s nrestarts=%s started=%s\n' "$SVC" \
  "$(systemctl is-active "$SVC.service" 2>/dev/null)" \
  "$(systemctl show -p NRestarts --value "$SVC.service" 2>/dev/null)" \
  "$(systemctl show -p ExecMainStartTimestamp --value "$SVC.service" 2>/dev/null)"
df -h "$DIR" | tail -n 1
printf 'gitignore=%s docs=%s\n' \
  "$(test -f "$DIR/.gitignore" && echo present || echo MISSING)" \
  "$(test -d "$DIR/docs" && echo present || echo absent)"

line "last deployed rev recorded by deploy.ps1"
for f in rev prev-rev; do
  if [ -f "$STATE_DIR/$f" ]; then printf '%s=%s\n' "$f" "$(cat "$STATE_DIR/$f")"; else printf '%s=(none)\n' "$f"; fi
done

line "files on disk"
ls -a1 "$DIR" | grep -vE '^\.\.?$' | tr '\n' ' '
echo

if [ ! -d "$DIR/.git" ]; then
  line "git"
  echo "NOT A GIT REPO - first deploy runs git init + fetch + reset --mixed (work tree untouched)"
  exit 0
fi
line "git"
printf 'HEAD=%s branch=%s\n' \
  "$($RUNAS git -C "$DIR" rev-parse HEAD 2>/dev/null)" \
  "$($RUNAS git -C "$DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)"
# 与 deploy-remote.sh 一致：origin 连不上就退回离线 bundle，探测不该被网络卡住
BUNDLE=/tmp/xiaoxia-deploy-bundle.bundle
if ! $RUNAS git -C "$DIR" fetch -q origin; then
  # 不猜 bundle 里的 ref 名，问它自己；优先取指向 TARGET 的那条
  bref="$($RUNAS git -C "$DIR" bundle list-heads "$BUNDLE" 2>/dev/null | awk -v want="$TARGET" '$1 == want { print $2; exit }')"
  [ -n "$bref" ] || bref="$($RUNAS git -C "$DIR" bundle list-heads "$BUNDLE" 2>/dev/null | awk 'NR == 1 { print $2 }')"
  if [ -n "$bref" ] && $RUNAS git -C "$DIR" fetch -q "$BUNDLE" "$bref"; then
    echo "WARN: origin unreachable, probed with the offline bundle"
  else
    echo "WARN: fetch failed (origin unreachable, no usable bundle)"
  fi
fi

line "target rev $TARGET"
if $RUNAS git -C "$DIR" cat-file -e "${TARGET}^{commit}" 2>/dev/null; then
  echo "PRESENT on server"
  $RUNAS git -C "$DIR" log --oneline -1 "$TARGET"
else
  echo "ABSENT - deploy would abort"
fi

line "tracked files changed on the server (this is what deploy.ps1 blocks on)"
dirty="$($RUNAS git -C "$DIR" status --porcelain --untracked-files=no)"
if [ -z "$dirty" ]; then
  echo "CLEAN"
else
  echo "$dirty"
  line "diff --stat"
  $RUNAS git -C "$DIR" diff --stat
fi

line "untracked files on the server (checkout -f leaves them alone)"
untracked="$($RUNAS git -C "$DIR" status --porcelain --untracked-files=normal | grep '^??' | head -n 20)"
if [ -z "$untracked" ]; then echo "none"; else echo "$untracked"; fi

line "commits to be deployed (HEAD..$TARGET)"
$RUNAS git -C "$DIR" log --oneline "HEAD..${TARGET}" 2>/dev/null | tail -n 30

line "requirements.txt delta"
if $RUNAS git -C "$DIR" diff --name-only HEAD "$TARGET" -- requirements.txt 2>/dev/null | grep -q requirements; then
  $RUNAS git -C "$DIR" diff HEAD "$TARGET" -- requirements.txt 2>/dev/null
else
  echo "requirements.txt unchanged - pip install would be skipped"
fi

line "db summary"
if [ -f "$DIR/data/xiaoxia.sqlite3" ]; then
  printf 'size=%s bytes\n' "$(stat -c '%s' "$DIR/data/xiaoxia.sqlite3")"
  $RUNAS sqlite3 "$DIR/data/xiaoxia.sqlite3" \
    "SELECT 'messages='||count(*) FROM messages; SELECT 'memory_cards='||count(*) FROM memory_cards; SELECT 'tasks='||count(*) FROM scheduled_tasks;" 2>/dev/null
else
  echo "(no database file)"
fi

line "env keys (names only, values never printed)"
if [ -f "$DIR/.env" ]; then
  grep -oE '^[A-Za-z_][A-Za-z0-9_]*' "$DIR/.env" | sort | tr '\n' ' '; echo
else
  echo "(no .env)"
fi

line "recent service log"
journalctl -u "$SVC.service" --no-pager --output cat -n 8 2>/dev/null