#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVAL_PYTHON="${RETRIEVAL_PYTHON:-/root/autodl-tmp/venvs/liquid_tf52/bin/python}"
if [[ ! -x "$RETRIEVAL_PYTHON" ]]; then
  RETRIEVAL_PYTHON="$(command -v python)"
fi
exec "$RETRIEVAL_PYTHON" -u "$SCRIPT_DIR/launch_background.py" "$@"
