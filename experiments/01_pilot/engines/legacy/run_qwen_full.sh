#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${QWEN_PROJECT_ROOT:-/root/autodl-tmp/twin_qwen}"
QWEN_PYTHON="${QWEN_PYTHON:-/root/autodl-tmp/venvs/liquid_tf52/bin/python}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct}"
QWEN_RUN_DIR="${QWEN_RUN_DIR:-outputs/qwen_full_v1}"
cd "$PROJECT_DIR"
mkdir -p logs
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TRAIN_ARGS=(--root "$PROJECT_DIR" --model "$QWEN_MODEL_DIR"
  --source-run outputs/liquid_lora_v1 --out "$QWEN_RUN_DIR"
  --epochs 3 --learning-rate 1e-5 --micro-batch-size 1
  --gradient-accumulation 16 --preprocess-workers 4 --num-workers 4)

"$QWEN_PYTHON" -u "$SCRIPT_DIR/train_qwen_full.py" "${TRAIN_ARGS[@]}" --prepare-only \
  2>&1 | tee "logs/qwen_full_prepare_${RUN_STAMP}.log"

"$QWEN_PYTHON" -u "$SCRIPT_DIR/evaluate_qwen_full.py" \
  --root "$PROJECT_DIR" --train-run "$QWEN_RUN_DIR" --baseline-only \
  2>&1 | tee "logs/qwen_full_baseline_${RUN_STAMP}.log"

"$QWEN_PYTHON" -u "$SCRIPT_DIR/train_qwen_full.py" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "logs/qwen_full_train_${RUN_STAMP}.log"

"$QWEN_PYTHON" -u "$SCRIPT_DIR/evaluate_qwen_full.py" \
  --root "$PROJECT_DIR" --train-run "$QWEN_RUN_DIR" \
  2>&1 | tee "logs/qwen_full_validation_${RUN_STAMP}.log"
