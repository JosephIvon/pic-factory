#!/usr/bin/env bash
# ============================================================================
# .github/workflows/ci.yml 三条 job（+ 密钥扫描）的本地等价命令（Git Bash 可跑）。
#
# 用法：  bash tools/ci_local.sh [scan|test|docker|smoke|all]   （默认 all）
#
# 与 CI 的口径差异（如实记录）：
#   1) 本机 venv 里若装过 editable 模式（__editable__*.pth），`aesthetics` 入口
#      会指向**构建 venv 时的那份源码副本**，不是本仓库 src/ ——直接跑会用旧代码
#      （实测旧 cli.py 里 qc 缺 missing_face_reason 导入，NameError exit 1）。
#      所以本脚本一律用 PYTHONPATH 强制指向本仓库 src/，等效于 CI 的 fresh checkout。
#   2) CI test job 用 python3.11；本机 venv 版本以实际为准（≥3.11 即可，requires-python=">=3.11"）。
#   3) smoke job 在 CI 里**不装 insightface**（人脸闸走 skipped 兜底，保证离线）。
#      口径（2026-10-24 复审后改）：skipped = 闸门没跑成 = **不能记为合格**——
#      人物任务在质检阶段一律拒收（矩阵夹具 4 条全是 character=lin），
#      与 OCR 闸「文字闸未执行，不能记为合格」同口径。
#      本机 venv 装了且有权重：想模拟 CI 口径，先 export AP_SIMULATE_NO_INSIGHTFACE=1
#      并准备一个"raise ModuleNotFoundError"的假 insightface.py 放进 PYTHONPATH 前段，
#      或直接接受"本机带真人脸闸"这一超集口径（开发机两种都已实跑 exit 0）。
#   4) docker job 需要本机 Docker Desktop 在跑；未装则跳过并说明。
# ============================================================================
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"   # 默认用 PATH 里的 python；本机多版本时显式传带依赖的那个
DB='/tmp/ci.db'   # Git Bash 会自动转换成 %LOCALAPPDATA%\Temp\ci.db 传给 Windows python
# 密钥扫描比 CI 版多排掉本地产物目录（.venv/out/__pycache__ 等不在 git 里，CI fresh checkout 没有它们）
SCAN_EXCLUDES=(--exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=out --exclude-dir=.pytest_cache)

# 让本仓库 src 压过 venv 里指向旧副本的 editable 安装（Windows python 认反斜杠/盘符路径）
export PYTHONPATH="$(cygpath -w "$REPO/src" 2>/dev/null || echo "$REPO/src")"

run_scan() {
  echo "== [scan] 密钥泄漏扫描（与 ci.yml 同一 PCRE 口径，见 yml 顶部注释的误报说明）=="
  # grep 语义：命中→exit 1（失败）；无命中→echo clean
  grep -rPn "(?<![\w-])(ark-[A-Za-z0-9]{6,}|sk-ws-[A-Za-z0-9]{6,})" "${SCAN_EXCLUDES[@]}" "$REPO" \
    && { echo "!! 发现疑似密钥，门禁失败"; exit 1; } || echo clean
}

run_test() {
  echo "== [job1 test] 等价 CI: pip install -e \".[qc,panel,dev]\" && pytest -q =="
  echo "   （venv 已装依赖；若要重装：\"$PY\" -m pip install -e \"$REPO[qc,panel,dev]\" ——注意这会改写 venv 的 editable 指向，慎用）"
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
rows = conn.execute("SELECT idempotency_key, character, qc_verdict, qc_reasons FROM assets WHERE provider='"'"'matrix'"'"'").fetchall()
assert len(rows) == 4, f"期望 4 条回放落账，实际 {len(rows)}: {rows}"
assert all(v in ("accepted", "rejected") for _, _, v, _ in rows), rows
no_ocr = [k for k, _, _, r in rows if r and "OCR 引擎不可用" in r]
assert not no_ocr, f"文字闸未执行: {no_ocr}"
for key, character, verdict, reasons in rows:
    if character:
        assert verdict == "rejected", f"人物任务在人脸闸未执行时必须拒收: {(key, verdict, reasons)}"
        assert "人脸闸未执行" in (reasons or ""), f"拒收理由要写明闸门没跑: {(key, reasons)}"
print("replay-smoke ledger check OK:", [(k, v) for k, _, v, _ in rows])
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
