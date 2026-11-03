#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT}/影子记账_扣100_k1_尖端与0.62_20261102"
cd "$ROOT"
source ${VENV_DIR}/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p 结果 账本 中间
# 用法：
#   ./脚本/一键重跑.sh              # 回填 S1 全部可打分观察日
#   ./脚本/一键重跑.sh --day 2026-11-01
#   DAY=2026-11-01 ./脚本/一键重跑.sh
ARGS=()
if [[ "${DAY:-}" != "" ]]; then
  ARGS+=(--day "$DAY")
fi
ARGS+=("$@")
python -u 脚本/日更打分记账.py "${ARGS[@]}" 2>&1 | tee 结果/_stdout_定稿.log
