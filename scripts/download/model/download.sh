#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir_default="$(cd "${script_dir}/../../.." && pwd)"
manifest_py="${script_dir}/manifest.py"

if [[ -x "${root_dir_default}/.venv/bin/python" ]]; then
    default_python_bin="${root_dir_default}/.venv/bin/python"
else
    default_python_bin="python3"
fi

python_bin="${PYTHON_BIN:-${default_python_bin}}"
root_dir="${root_dir_default}"
models="all"
tool="aria2c"
threads="8"
hf_username="${HF_USERNAME:-}"
hf_token="${HF_TOKEN:-}"
hf_endpoint="${HF_ENDPOINT:-}"
include_pattern=""
exclude_pattern=""
dry_run=0

show_help() {
    cat <<'EOF'
Usage:
  bash scripts/download/model/download.sh [options]

Options:
  --models <keys>        Comma-separated model keys. Default: all
  --root-dir <path>      Project root directory. Default: auto-detected repo root.
  --tool <aria2c|wget>   Download backend passed to docs/hfd.sh. Default: aria2c
  --threads <n>          Parallel threads for aria2c/wget mode. Default: 8
  --hf-username <user>   HuggingFace username for gated/private repos.
  --hf-token <token>     HuggingFace token for gated/private repos.
  --hf-endpoint <url>    HuggingFace endpoint mirror (exported as HF_ENDPOINT).
  --include <pattern>    Include pattern forwarded to docs/hfd.sh.
  --exclude <pattern>    Exclude pattern forwarded to docs/hfd.sh.
  --dry-run              Print commands only.
  -h, --help             Show this help.

Examples:
  bash scripts/download/model/download.sh --models all --threads 8
  bash scripts/download/model/download.sh --models sam3,xsam --hf-token "$HF_TOKEN"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --models)
        models="${2:-}"
        shift 2
        ;;
    --root-dir)
        root_dir="$(cd "${2:-}" && pwd)"
        shift 2
        ;;
    --tool)
        tool="${2:-}"
        shift 2
        ;;
    --threads | -x)
        threads="${2:-}"
        shift 2
        ;;
    --hf-username)
        hf_username="${2:-}"
        shift 2
        ;;
    --hf-token)
        hf_token="${2:-}"
        shift 2
        ;;
    --hf-endpoint)
        hf_endpoint="${2:-}"
        shift 2
        ;;
    --include)
        include_pattern="${2:-}"
        shift 2
        ;;
    --exclude)
        exclude_pattern="${2:-}"
        shift 2
        ;;
    --dry-run)
        dry_run=1
        shift
        ;;
    -h | --help)
        show_help
        exit 0
        ;;
    *)
        echo "[model-download] Unknown argument: $1" >&2
        show_help
        exit 2
        ;;
    esac
done

hfd_script="${root_dir}/docs/hfd.sh"

if [[ ! -f "${manifest_py}" ]]; then
    echo "[model-download] manifest not found: ${manifest_py}" >&2
    exit 2
fi

if [[ ! -f "${hfd_script}" ]]; then
    echo "[model-download] docs/hfd.sh not found: ${hfd_script}" >&2
    exit 2
fi

if [[ ! -x "${hfd_script}" ]]; then
    chmod +x "${hfd_script}"
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[model-download] python not found: ${python_bin}" >&2
    exit 2
fi

if [[ -n "${hf_endpoint}" ]]; then
    export HF_ENDPOINT="${hf_endpoint}"
fi

mapfile -t manifest_rows < <("${python_bin}" "${manifest_py}" --models "${models}" --format tsv)
if [[ "${#manifest_rows[@]}" -eq 0 ]]; then
    echo "[model-download] no model selected." >&2
    exit 2
fi

for row in "${manifest_rows[@]}"; do
    IFS=$'\t' read -r model_key model_repo model_relpath model_desc <<<"${row}"
    target_abs="${root_dir}/${model_relpath}"
    target_parent="$(dirname "${target_abs}")"
    expected_dir_name="$(basename "${target_abs}")"
    repo_dir_name="${model_repo##*/}"

    mkdir -p "${target_parent}"

    echo
    echo "[Model ${model_key}] ${model_desc}"
    echo "  repo   : ${model_repo}"
    echo "  target : ${target_abs}"

    cmd=(
        bash
        "${hfd_script}"
        "${model_repo}"
        --tool
        "${tool}"
        -x
        "${threads}"
        --save_dir
        "${target_parent}"
    )
    if [[ -n "${hf_username}" ]]; then
        cmd+=(--hf_username "${hf_username}")
    fi
    if [[ -n "${hf_token}" ]]; then
        cmd+=(--hf_token "${hf_token}")
    fi
    if [[ -n "${include_pattern}" ]]; then
        cmd+=(--include "${include_pattern}")
    fi
    if [[ -n "${exclude_pattern}" ]]; then
        cmd+=(--exclude "${exclude_pattern}")
    fi

    if [[ "${dry_run}" -eq 1 ]]; then
        printf '[DRY-RUN]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    "${cmd[@]}"

    downloaded_path="${target_parent}/${repo_dir_name}"
    if [[ "${repo_dir_name}" != "${expected_dir_name}" ]]; then
        if [[ -e "${target_abs}" ]]; then
            echo "[model-download] target already exists and repo basename differs: ${target_abs}" >&2
            exit 2
        fi
        mv "${downloaded_path}" "${target_abs}"
    fi
done

echo
echo "[model-download] done."

# CLI examples:
# bash scripts/download/model/download.sh \
#   --models all \
#   --root-dir . \
#   --threads 8
#
# bash scripts/download/model/download.sh \
#   --models sam3,xsam \
#   --hf-token "${HF_TOKEN}" \
#   --threads 8
