#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT}/涨侧提前出手对照_20261102"
cd "$ROOT"
# shellcheck disable=SC1091
source ${VENV_DIR}/bin/activate
if [[ "${REBUILD:-0}" == "1" ]]; then
  echo "[REBUILD] 清理中间缓存与涨模型…"
  rm -f 中间/*.parquet 中间/*.json 模型/model_涨到线_A1特征_seed42.txt
  rm -f 结果/VAL选门槛过程表.csv 结果/VAL选择_先于测试集读取.json \
        结果/测试结果表_*.csv 结果/相对基线_*.csv 结果/测试逐观察日_*.csv \
        结果/测试按航司_*.csv 结果/涨侧提前出手报告_S3A1.json 结果/VAL入场池扫描_*.csv
fi
python -u 脚本/涨侧提前出手对照_S3A1.py
echo "OK: $ROOT"
