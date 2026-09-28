#!/usr/bin/env bash
set -uo pipefail
REPO="$1"
ROOT=~/repo-bench
cd "$ROOT/repos/$REPO" || exit 1
OUT="$ROOT/measurements"

# 记录起始 commit
git rev-parse HEAD > "$OUT/${REPO}_baseline_commit.txt"
git status > "$OUT/${REPO}_git_status_before.txt"

# 代码质量第一轮(|| true 是因为这些工具发现问题会返回非0,不能让脚本中断)
{
  echo "===== ruff check (Lint 告警密度) ====="; ruff check . || true
  echo "===== C901 (圈复杂度) ====="; ruff check --select C901 . || true
  echo "===== PLR0915 (语句数) ====="; ruff check --select PLR0915 . || true
  echo "===== PLR1702 (嵌套深度) ====="; ruff check --select PLR1702 . || true
  echo "===== ANN (类型注解) ====="; ruff check --select ANN . || true
  echo "===== F401/F841 (死代码) ====="; ruff check --select F401,F841 . || true
  echo "===== format --check (格式合规) ====="; ruff format --check . || true
  echo "===== radon mi (可维护性) ====="; radon mi . -s || true
  echo "===== radon raw (函数长度) ====="; radon raw . || true
  echo "===== vulture (死代码) ====="; vulture . || true
} > "$OUT/${REPO}_quality_before.txt" 2>&1

jscpd . --reporters json --output "$OUT/${REPO}_jscpd_before" >/dev/null 2>&1 || true
echo "[baseline done] $REPO"
