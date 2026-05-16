#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/opt/venv}"
HOME_DIR="${HOME:-/workspace}"

mkdir -p "${HOME_DIR}/.cache/torchinductor" "${HOME_DIR}/.cache/triton" /tmp/matplotlib /tmp/gradio

if [ -x "${VENV_DIR}/bin/pip" ] && [ -f "/workspace/pyproject.toml" ]; then
  "${VENV_DIR}/bin/pip" install --no-deps -e /workspace
fi

exec "$@"
