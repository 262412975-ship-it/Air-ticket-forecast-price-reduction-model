#!/usr/bin/env bash
# 例程入口包装：激活 venv 后调 python 主脚本；透传全部参数。
set -euo pipefail
source ${VENV_DIR}/bin/activate
exec python -u ${PROJECT_ROOT}/S0S1_日更并入/脚本/日更并入_S0S1.py "$@"
