#!/usr/bin/env bash
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
root_dir="$(realpath "$script_dir/../..")"

resolve_input_path() {
  local path_like="$1"
  if [[ "$path_like" = /* ]]; then
    echo "$path_like"
    return 0
  fi
  if [[ -e "$path_like" ]]; then
    realpath -m "$path_like"
    return 0
  fi
  if [[ -e "$root_dir/$path_like" ]]; then
    realpath -m "$root_dir/$path_like"
    return 0
  fi
  echo "$path_like"
}

CONFIG_YAML="$root_dir/xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_YAML="$(resolve_input_path "${1}")"
  shift
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [[ -x "/opt/venv/bin/python" ]]; then
  PYTHON_CMD="/opt/venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python"
fi

"${PYTHON_CMD}" "$root_dir/xsam/xsam/layer_analysis/spatial/layer_sweep_spatial.py" \
  --config-yaml "${CONFIG_YAML}" \
  "$@"
