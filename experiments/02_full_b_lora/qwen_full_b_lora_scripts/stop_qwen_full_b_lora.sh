#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${QWEN_PROJECT_ROOT:-/root/autodl-tmp/twin_qwen}"
exec python3 "$SCRIPT_DIR/stop_qwen_full_b_lora.py" --root "$PROJECT_DIR" "$@"
