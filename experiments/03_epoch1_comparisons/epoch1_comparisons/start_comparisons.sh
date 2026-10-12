#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_PYTHON="${COMPARISON_PYTHON:-/root/autodl-tmp/venvs/liquid_tf52/bin/python}"
exec "$COMPARISON_PYTHON" "$SCRIPT_DIR/launch_background.py" "$@"
