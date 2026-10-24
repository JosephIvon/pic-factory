#!/usr/bin/env bash
# ============================================================================
# .github/workflows/ci.yml 三条 job（+ 密钥扫描）的本地等价命令（Git Bash / WSL 可跑）。
#
# 用法：  bash tools/ci_local.sh [scan|test|docker|smoke|all]   （默认 all）
#        指定解释器：PY=/path/to/python bash tools/ci_local.sh test
#
# 与 CI 的口径差异（如实记录）：
#   1) 如果本机 venv 里 `pip install -e .` 指向的是另一份代码副本，直接跑 `aesthetics`
#      或 `-m aesthetics_pipeline` 会用到那份旧代码。所以本脚本一律用 PYTHONPATH 强制
#      指向本仓库 src/，等效于 CI 的 fresh checkout。
#   2) CI test job 用 python3.11；本机版本以实际为准（requires-python = ">=3.11"）。
#   3) CI 的 smoke job **故意不装 insightface**（人脸闸走显式 skipped 兜底，保证严格
#      离线）。本机若装了 insightface 且权重可用，跑出来是"带真人脸闸"的超集口径。
#   4) docker job 需要本机 Docker 在跑；没有 docker CLI 就跳过并说明。
# ============================================================================
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
DB='/tmp/ci.db'   # Git Bash 会把 /tmp 转换成 Windows 临时目录后传给原生 python
# 密钥扫描比 CI 版多排掉本地产物目录（.venv/out/__pycache__ 等不在 git 里，CI fresh checkout 没有它们）
SCAN_EXCLUDES=(--exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=out --exclude-dir=.pytest_cache)

# 让本仓库 src 压过 venv 里可能指向别处副本的 editable 安装（Windows python 认反斜杠/盘符路径）
export PYTHONPATH="$(cygpath -w "$REPO/src" 2>/dev/null || echo "$REPO/src")"

run_scan() {
  echo "== [scan] 密钥泄漏扫描（与 ci.yml 同一 PCRE 口径，误报说明见 yml 顶部注释）=="
  # grep 语义：命中→exit 1（失败）；无命中→echo clean
  grep -rPn "(?<![\w-])(ark-[A-Za-z0-9]{6,}|sk-ws-[A-Za-z0-9]{6,})" "${SCAN_EXCLUDES[@]}" "$REPO" \
    && { echo "!! 发现疑似密钥，门禁失败"; exit 1; } || echo clean
}

run_test() {
  echo "== [job1 test] 等价 CI: pip install -e \".[qc,panel,dev]\" && pytest -q =="
  echo "   （假设依赖已装；要重装：\"$PY\" -m pip install -e \"$REPO[qc,panel,dev]\" ——这会改写 venv 的 editable 指向，慎用）"
  cd "$REPO" && "$PY" -m pytest -q
}

run_docker() {
  echo "== [job2 docker] 等价 CI: docker build . （不跑容器）=="
  if command -v docker >/dev/null 2>&1; then
    cd "$REPO" && docker build .
  else
    echo "!! 本机无 docker CLI，跳过（CI 上 ubuntu-latest 自带）"
  fi
}

run_smoke() {
  echo "== [job3 replay-smoke] 离线端到端：replay 4 条 → qc → report（独立台账 $DB）=="
  rm -f "$DB"
  cd "$REPO" || exit 1
  "$PY" -m aesthetics_pipeline.cli generate --provider replay --replay-provider matrix --db "$DB" --limit 4 || { echo "!! generate 失败"; exit 1; }
  "$PY" -m aesthetics_pipeline.cli qc --db "$DB"       || { echo "!! qc 失败"; exit 1; }
  "$PY" -m aesthetics_pipeline.cli report --db "$DB"   || { echo "!! report 失败"; exit 1; }
  # 台账核验（与 ci.yml replay-smoke 最后一步同口径）。
  # $DB 走 argv 传入：Git Bash 对原生 exe 的参数做 POSIX→Windows 路径转换，
  # 与上面三条 CLI 命令写到的是同一个文件。
  "$PY" -c '
import sys, sqlite3
conn = sqlite3.connect(sys.argv[1])
rows = conn.execute("SELECT idempotency_key, qc_verdict FROM assets WHERE provider='"'"'matrix'"'"'").fetchall()
assert len(rows) == 4, f"期望 4 条回放落账，实际 {len(rows)}: {rows}"
assert all(v in ("accepted", "rejected") for _, v in rows), rows
print("replay-smoke ledger check OK:", rows)
' "$DB"
}

case "${1:-all}" in
  scan)   run_scan ;;
  test)   run_test ;;
  docker) run_docker ;;
  smoke)  run_smoke ;;
  all)    run_scan; run_test; run_docker; run_smoke ;;
  *)      echo "用法: bash tools/ci_local.sh [scan|test|docker|smoke|all]"; exit 2 ;;
esac
