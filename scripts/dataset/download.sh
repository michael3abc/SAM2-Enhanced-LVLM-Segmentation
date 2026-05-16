#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"
if [[ -x "${root_dir}/.venv/bin/python" ]]; then
    default_python_bin="${root_dir}/.venv/bin/python"
else
    default_python_bin="python"
fi
python_bin="${PYTHON_BIN:-${default_python_bin}}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  bash scripts/dataset/download.sh [dataclass.py args]

Examples:
  bash scripts/dataset/download.sh \
    --datasets all \
    --root-dir . \
    --threads 8 \
    --http-tool auto

  bash scripts/dataset/download.sh \
    --datasets coco,ovseg \
    --root-dir . \
    --overwrite
EOF
    exit 0
fi

exec "${python_bin}" "${root_dir}/scripts/dataset/dataclass.py" "$@"

# CLI examples:
# bash scripts/dataset/download.sh \
#   --datasets all \
#   --root-dir . \
#   --threads 8 \
#   --http-tool auto
