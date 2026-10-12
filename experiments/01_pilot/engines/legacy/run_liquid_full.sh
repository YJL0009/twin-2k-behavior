#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIQUID_PYTHON="/root/autodl-tmp/venvs/liquid_tf52/bin/python"
cd "$PROJECT_DIR"
mkdir -p logs
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

"$LIQUID_PYTHON" -u train_liquid_full.py \
  --root "$PROJECT_DIR" \
  --epochs 3 --learning-rate 1e-5 \
  --micro-batch-size 1 --gradient-accumulation 16 --num-workers 4 \
  2>&1 | tee "logs/liquid_full_train_${RUN_STAMP}.log"

"$LIQUID_PYTHON" -u evaluate_liquid_full.py \
  --root "$PROJECT_DIR" \
  2>&1 | tee "logs/liquid_full_validation_${RUN_STAMP}.log"
