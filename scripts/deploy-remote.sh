#!/usr/bin/env bash
set -uo pipefail

DIR="__REMOTE_DIR__"
SVC="__SERVICE__"
URL="__REPO_URL__"
BRANCH="__BRANCH__"
TARGET="__TARGET__"
FORCE="__FORCE__"
SKIP_TESTS="__SKIP_TESTS__"

OWNER="$(stat -c '%U' "$DIR")"
RUNAS="sudo -u $OWNER -H"
STATE_DIR=/var/lib/xiaoxia-deploy
say() { printf '%s %s\n' "[$1]" "$2"; }

# 服务器到 github.com 的 git 通道时通时断，而且是"挂着不动"而不是干脆报错——
# 2026-10-16 就卡在这里十几分钟。所有对外 fetch 都套超时，超时了才走 bundle 兜底。
FETCH_TIMEOUT="${FETCH_TIMEOUT:-45}"
net_fetch() { timeout -k 10 "$FETCH_TIMEOUT" $RUNAS git -C "$DIR" "$@"; }

cd "$DIR" || { say fatal "cannot enter $DIR"; exit 9; }

# ---- 1. first run: attach the existing directory to git without touching the work tree ----
if [ ! -d "$DIR/.git" ]; then
  say init "git init $DIR (owner $OWNER)"
  $RUNAS git init -q "$DIR" || { say fatal "git init failed"; exit 9; }
  $RUNAS git -C "$DIR" remote add origin "$URL"
  $RUNAS git -C "$DIR" config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
  net_fetch fetch -q origin || { say fatal "git fetch failed (check server -> GitHub connectivity)"; exit 9; }
  $RUNAS git -C "$DIR" branch -f "$BRANCH" "origin/$BRANCH"
  $RUNAS git -C "$DIR" symbolic-ref HEAD "refs/heads/$BRANCH"
  # --mixed only rebuilds the index, the work tree stays untouched,
  # so the status check below still reports real hand edits made on the server
  $RUNAS git -C "$DIR" reset -q --mixed "origin/$BRANCH"
fi

# ---- fetch objects: origin first, offline bundle as the escape hatch ----
# 服务器在国内，api.github.com 通、github.com 的 git 传输通道却经常超时，
# 照原来的写法每次部署都要赌一次网络。deploy.ps1 会把同一分支的完整 bundle
# 一起传上来；正常路径不碰它，只有 fetch 不动时才拿来顶上。
BUNDLE=/tmp/xiaoxia-deploy-bundle.bundle
# bundle 里记的 ref 名随本地怎么打包而变（refs/heads/main 或 refs/remotes/origin/main），
# 所以不猜名字，直接问 bundle 自己要，优先取指向 TARGET 的那条。
bundle_ref_for_target() {
  $RUNAS git -C "$DIR" bundle list-heads "$BUNDLE" 2>/dev/null |
    awk -v want="$TARGET" '$1 == want { print $2; exit }'
}
bundle_any_ref() {
  $RUNAS git -C "$DIR" bundle list-heads "$BUNDLE" 2>/dev/null | awk 'NR == 1 { print $2 }'
}
fetch_from_bundle() {
  [ -f "$BUNDLE" ] || return 1
  local ref
  ref="$(bundle_ref_for_target)"
  [ -n "$ref" ] || ref="$(bundle_any_ref)"
  [ -n "$ref" ] || return 1
  $RUNAS git -C "$DIR" fetch -q "$BUNDLE" "$ref"
}
if net_fetch fetch -q origin; then
  say fetch "objects from origin"
elif fetch_from_bundle; then
  say fetch "origin unreachable, used offline bundle"
else
  say fatal "git fetch failed and no usable bundle at $BUNDLE"
  exit 9
fi
$RUNAS git -C "$DIR" cat-file -e "${TARGET}^{commit}" 2>/dev/null || {
  say fatal "target rev $TARGET not found on server"; exit 9; }

# ---- 2. protect hand edits on the server ----
# 只有"已跟踪文件被改"才需要人工确认；未跟踪的 data/ .venv/ 等本来就归 .gitignore 管
dirty="$($RUNAS git -C "$DIR" status --porcelain --untracked-files=no)"
untracked="$($RUNAS git -C "$DIR" status --porcelain --untracked-files=all | grep -c '^??' || true)"
if [ -n "$dirty" ] && [ "$FORCE" != "1" ]; then
  say dirty "server work tree differs from the repo; checkout -f would drop these:"
  echo "$dirty" | sed 's/^/    /'
  $RUNAS git -C "$DIR" diff --stat | sed 's/^/    /'
  printf '    (%s untracked files on server are NOT touched by checkout -f)\n' "$untracked"
  say dirty "review it, then re-run with -Force if dropping them is intended"
  exit 3
fi

PREV="$($RUNAS git -C "$DIR" rev-parse HEAD)"
say head "online rev $PREV -> target rev $TARGET"

restore_previous() {
  say rollback "rollback to $PREV"
  $RUNAS git -C "$DIR" checkout -f -q "$PREV" || say rollback "code rollback failed, fix manually"
  if [ -f "$STATE_DIR/req-changed" ]; then
    rm -f "$STATE_DIR/req-changed"
    $RUNAS git -C "$DIR" show "${PREV}:requirements.txt" > /tmp/rollback-req.txt 2>/dev/null
    if [ -s /tmp/rollback-req.txt ]; then
      say rollback "restore dependencies of $PREV"
      $RUNAS "$DIR/.venv/bin/pip" install -q -r /tmp/rollback-req.txt || say rollback "pip rollback failed, install manually"
    fi
  fi
  systemctl restart "$SVC.service"
}

# ---- 3. switch code ----
$RUNAS git -C "$DIR" checkout -f -q "$TARGET" || { say fatal "checkout failed"; exit 9; }
$RUNAS git -C "$DIR" rev-parse HEAD

# ---- 4. install deps only when requirements.txt changed ----
REQ_CHANGED=0
if [ "$PREV" != "$TARGET" ]; then
  old_md5="$($RUNAS git -C "$DIR" show "${PREV}:requirements.txt" 2>/dev/null | md5sum | cut -d' ' -f1)"
else
  old_md5="$(md5sum "$DIR/requirements.txt" | cut -d' ' -f1)"
fi
new_md5="$(md5sum "$DIR/requirements.txt" | cut -d' ' -f1)"
mkdir -p "$STATE_DIR"
if [ "$old_md5" != "$new_md5" ]; then
  say deps "requirements.txt changed, installing"
  REQ_CHANGED=1
  touch "$STATE_DIR/req-changed"
  if ! $RUNAS "$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"; then
    say fail "pip install failed"
    restore_previous
    exit 4
  fi
else
  say deps "requirements.txt unchanged, skip install"
fi

# ---- 5. run tests against the new code ----
if [ "$SKIP_TESTS" != "1" ]; then
  say test "running unit tests"
  if ! $RUNAS env PYTHONUTF8=1 "$DIR/.venv/bin/python" -m unittest discover -s "$DIR/tests" > /tmp/deploy-test.log 2>&1; then
    say fail "tests failed, service is not restarted"
    tail -n 25 /tmp/deploy-test.log | sed 's/^/    /'
    restore_previous
    exit 5
  fi
  tail -n 4 /tmp/deploy-test.log | sed 's/^/    /'
fi

# ---- 6. back up database and code ----
stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DIR/backups"
if [ -f "$DIR/data/xiaoxia.sqlite3" ]; then
  $RUNAS sqlite3 "$DIR/data/xiaoxia.sqlite3" ".backup '$DIR/backups/pre-deploy-$stamp.sqlite3'" \
    && say backup "database -> backups/pre-deploy-$stamp.sqlite3"
  $RUNAS bash -c "ls -t $DIR/backups/pre-deploy-*.sqlite3 2>/dev/null | tail -n +9 | xargs -r rm -f"
fi
$RUNAS tar -czf "$DIR/backups/code-$stamp.tar.gz" \
  --exclude=.venv --exclude=data --exclude=backups --exclude=.git -C "$DIR" . \
  && say backup "code -> backups/code-$stamp.tar.gz"
$RUNAS bash -c "ls -t $DIR/backups/code-*.tar.gz 2>/dev/null | tail -n +9 | xargs -r rm -f"

# ---- 7. restart and health check ----
nrest_before="$(systemctl show -p NRestarts --value "$SVC.service" 2>/dev/null || echo 0)"
since_ts="$(date +%s)"
say restart "systemctl restart $SVC"
systemctl restart "$SVC.service"

health=0
for i in $(seq 1 15); do
  sleep 2
  state="$(systemctl is-active "$SVC.service" 2>/dev/null)"
  nrest_now="$(systemctl show -p NRestarts --value "$SVC.service" 2>/dev/null || echo 0)"
  if [ "$state" != "active" ]; then continue; fi
  if [ "$nrest_now" -gt "$nrest_before" ]; then continue; fi
  if journalctl -u "$SVC.service" --since "@$since_ts" --no-pager --output cat 2>/dev/null | grep -qaE 'on_ready|Uvicorn|initialized'; then
    health=1; break
  fi
  # no crash but no clear startup log either: keep watching a bit longer
  if [ "$i" -ge 12 ]; then health=1; break; fi
done

if [ "$health" != "1" ]; then
  say fail "health check failed, last logs:"
  journalctl -u "$SVC.service" --since "@$since_ts" --no-pager --output cat 2>/dev/null | tail -n 30 | sed 's/^/    /'
  restore_previous
  exit 6
fi

echo "$TARGET" > "$STATE_DIR/rev"
echo "$PREV" > "$STATE_DIR/prev-rev"
say done "deploy finished: $TARGET"
