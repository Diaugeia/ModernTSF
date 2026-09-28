#!/usr/bin/env bash
set -uo pipefail
REPO="$1"
ROOT=~/repo-bench
cd "$ROOT/repos/$REPO" || exit 1
OUT="$ROOT/measurements"
BASE=$(cat "$OUT/${REPO}_baseline_commit.txt")

# 关键:Agent 可能没 commit,改动还在工作区/新文件里。
# 先 git add -A 把所有改动(含新文件)纳入索引,再和起始 commit 比,才能拿到完整增删量。
git add -A
git diff --cached --stat "$BASE"     > "$OUT/${REPO}_diffstat.txt"
git diff --cached --shortstat "$BASE" >> "$OUT/${REPO}_diffstat.txt"
git reset -q                          # 还原索引,不留副作用

# 代码质量第二轮(和 baseline 完全相同的扫描)
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
} > "$OUT/${REPO}_quality_after.txt" 2>&1

jscpd . --reporters json --output "$OUT/${REPO}_jscpd_after" >/dev/null 2>&1 || true
echo "[final done] $REPO"
