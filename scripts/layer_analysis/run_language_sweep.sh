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

CONFIG_YAML="$root_dir/xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_YAML="$(resolve_input_path "${1}")"
  shift
fi

for ARG in "$@"; do
  if [[ "${ARG}" == "-h" || "${ARG}" == "--help" ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/layer_analysis/run_language_sweep.sh [EXTRA_ARGS...]
  bash scripts/layer_analysis/run_language_sweep.sh [CONFIG_YAML] [EXTRA_ARGS...]

Examples:
  bash scripts/layer_analysis/run_language_sweep.sh
  bash scripts/layer_analysis/run_language_sweep.sh xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml --dry-run
USAGE
    exit 0
  fi
done

PYTHON_ENTRY="$root_dir/xsam/xsam/layer_analysis/language/layer_sweep_language.py"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [[ -x "/opt/venv/bin/python" ]]; then
  PYTHON_CMD="/opt/venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python"
fi

BASE_ARGS=()
if [[ -n "${CONFIG_YAML}" ]]; then
  BASE_ARGS+=(--config-yaml "${CONFIG_YAML}")
fi
PLAN_ARGS=()
for ARG in "$@"; do
  if [[ "${ARG}" == "--dry-run" ]]; then
    PLAN_ARGS+=(--dry-run)
    break
  fi
done

"${PYTHON_CMD}" "${PYTHON_ENTRY}" \
  "${BASE_ARGS[@]}" \
  --phase phase1 \
  --mode train \
  "$@"

TOPK_CSV="$(
  "${PYTHON_CMD}" "${PYTHON_ENTRY}" \
    "${BASE_ARGS[@]}" \
    --phase phase2 \
    --mode plan \
    "${PLAN_ARGS[@]}" \
    --print-layer-ids
)"

IFS=',' read -r -a TOPK_LAYERS <<< "${TOPK_CSV}"
for LAYER_ID in "${TOPK_LAYERS[@]}"; do
  [[ -z "${LAYER_ID}" ]] && continue
  "${PYTHON_CMD}" "${PYTHON_ENTRY}" \
    "${BASE_ARGS[@]}" \
    --phase phase2 \
    --mode train \
    --single-layer-id "${LAYER_ID}" \
    "$@"
done

BEST_LAYER="$(
  "${PYTHON_CMD}" "${PYTHON_ENTRY}" \
    "${BASE_ARGS[@]}" \
    --phase phase3 \
    --mode plan \
    "${PLAN_ARGS[@]}" \
    --print-layer-ids
)"

"${PYTHON_CMD}" "${PYTHON_ENTRY}" \
  "${BASE_ARGS[@]}" \
  --phase phase3 \
  --mode train \
  --single-layer-id "${BEST_LAYER}" \
  "$@"
