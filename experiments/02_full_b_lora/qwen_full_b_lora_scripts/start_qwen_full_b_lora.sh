#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${QWEN_PROJECT_ROOT:-/root/autodl-tmp/twin_qwen}"
QWEN_PYTHON="${QWEN_PYTHON:-/root/autodl-tmp/venvs/liquid_tf52/bin/python}"
cd "$PROJECT_DIR"
mkdir -p logs
exec 9> "$PROJECT_DIR/logs/qwen_full_b_lora_launcher.lock"
if ! flock -n 9; then
  printf 'The full-B LoRA pipeline is already running.\n'
  exit 0
fi
# The earlier full-parameter launcher retains this lock while its job runs.
# Do not launch two competing jobs on the same GPU.
exec 8> "$PROJECT_DIR/logs/qwen_full_b_training_launcher.lock"
if ! flock -n 8; then
  printf 'The earlier full-parameter pipeline is still running. Stop that job before starting LoRA.\n' >&2
  exit 1
fi
"$QWEN_PYTHON" -u "$SCRIPT_DIR/run_qwen_full_b_lora_pipeline.py" --root "$PROJECT_DIR" "$@" --check-only
RUN_STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)"
RUN_LOG="$PROJECT_DIR/logs/qwen_full_b_lora_${RUN_STAMP}.log"
nohup "$QWEN_PYTHON" -u "$SCRIPT_DIR/run_qwen_full_b_lora_pipeline.py" --root "$PROJECT_DIR" \
  "$@" \
  > "$RUN_LOG" 2>&1 < /dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$PROJECT_DIR/logs/qwen_full_b_lora.pid"
ln -sfn "$(basename -- "$RUN_LOG")" "$PROJECT_DIR/logs/qwen_full_b_lora_latest.log"
sleep 2
if ! kill -0 "$RUN_PID" 2>/dev/null; then
  cat "$RUN_LOG"
  if wait "$RUN_PID"; then exit 0; else exit 1; fi
fi
printf 'Full-B LoRA training and validation started (PID %s). You can close this terminal.\n' "$RUN_PID"
printf 'Log: %s\n' "$RUN_LOG"
printf 'Monitor: tail -f %s/logs/qwen_full_b_lora_latest.log\n' "$PROJECT_DIR"
printf 'Default output: %s/outputs/qwen_full_b_lora_v4_b16_a2/ (see log if --out was supplied)\n' "$PROJECT_DIR"
