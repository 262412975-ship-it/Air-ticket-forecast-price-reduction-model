#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT}/价格日期回归对照_20261102"
cd "$ROOT"
source ${VENV_DIR}/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${REBUILD:-0}" == "1" ]]; then
  rm -f 中间/*.parquet 模型/model_*.txt 中间/*训练报告.json
fi
mkdir -p 结果 日志
python -u 脚本/价格日期回归对照_S3A1.py 2>&1 | tee 结果/_stdout_定稿.log
