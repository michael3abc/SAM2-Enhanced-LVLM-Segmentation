#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/opt/venv}"

if [ -x "${VENV_DIR}/bin/pip" ] && [ -f "/workspace/pyproject.toml" ]; then
  "${VENV_DIR}/bin/pip" install --no-deps -e /workspace
fi

exec "$@"
