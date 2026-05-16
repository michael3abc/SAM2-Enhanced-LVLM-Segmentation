#!/usr/bin/env bash
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
if [[ -d "$script_dir/docker" ]]; then
    root_dir="$script_dir"
else
    root_dir="$(realpath "$script_dir/..")"
fi

compose_file="${DOCKER_COMPOSE_FILE:-$root_dir/docker/compose.yaml}"
env_file="${DOCKER_ENV_FILE:-$root_dir/docker/.env}"
service="${DOCKER_SERVICE:-dev}"
build_first=0
direct_cmd=0

print_help() {
    cat <<'EOF'
Usage:
  bash run_docker.sh [--build] [--service SERVICE] [--compose-file FILE] [--env-file FILE] -- <run.sh args>
  bash run_docker.sh [--build] [--service SERVICE] [--compose-file FILE] [--env-file FILE] <run.sh args>
  bash run_docker.sh [--build] [--service SERVICE] [--compose-file FILE] [--env-file FILE] -- bash -lc '<cmd>'

Options:
  --build                 Build image before run.
  --service SERVICE       Compose service name. Default: dev
  --compose-file FILE     Compose file path. Default: docker/compose.yaml
  --env-file FILE         Env file path. Default: docker/.env
  --help, -h              Show this help.

Examples:
  CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
  bash run_docker.sh --build \
    --modes sweep_spatial \
    --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
    --sweep-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml \
    --sweep-args "--layers=-1,-2,-4 --train-epochs 1"

  CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 MASTER_PORT=29601 \
  bash run_docker.sh \
    --modes train,segeval \
    --config xsam/xsam/configs/xsam/s1_seg_finetune/sam2/xsam_sam2_seg_finetune.py \
    --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam2/profiles/base_1024_e24_gpu2.yaml

  CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
  bash run_docker.sh \
    --modes sweep_language \
    --sweep-yaml xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml \
    --sweep-args "--dry-run"

  bash run_docker.sh -- bash -lc 'ls -lah /workspace/runs'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            build_first=1
            shift
            ;;
        --service)
            shift
            service="${1:-}"
            [[ -n "$service" ]] || { echo "Error: --service requires a value."; exit 1; }
            shift
            ;;
        --compose-file)
            shift
            compose_file="$(realpath -m "${1:-}")"
            [[ -n "${1:-}" ]] || { echo "Error: --compose-file requires a value."; exit 1; }
            shift
            ;;
        --env-file)
            shift
            env_file="$(realpath -m "${1:-}")"
            [[ -n "${1:-}" ]] || { echo "Error: --env-file requires a value."; exit 1; }
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ ! -f "$compose_file" ]]; then
    echo "Error: compose file not found: $compose_file"
    exit 1
fi
if [[ ! -f "$env_file" ]]; then
    echo "Error: env file not found: $env_file"
    echo "Hint: cp docker/.env.example docker/.env"
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Error: missing run.sh args."
    print_help
    exit 1
fi

case "${1:-}" in
    bash|sh|python|python3|/bin/bash|/bin/sh|/opt/venv/bin/python|/opt/venv/bin/python3)
        direct_cmd=1
        ;;
esac

compose_cmd=(docker compose -f "$compose_file" --env-file "$env_file")

if [[ "$build_first" -eq 1 ]]; then
    "${compose_cmd[@]}" build "$service"
fi

if [[ "$direct_cmd" -eq 1 ]]; then
    "${compose_cmd[@]}" run --rm "$service" "$@"
else
    "${compose_cmd[@]}" run --rm "$service" bash run.sh "$@"
fi

# CLI examples:
# CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
# bash run_docker.sh --build \
#   --modes sweep_spatial \
#   --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
#   --sweep-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml \
#   --sweep-args "--layers=-1,-2,-4 --train-epochs 1"
#
# CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 MASTER_PORT=29601 \
# bash run_docker.sh \
#   --modes train,segeval \
#   --config xsam/xsam/configs/xsam/s1_seg_finetune/sam2/xsam_sam2_seg_finetune.py \
#   --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam2/profiles/base_1024_e24_gpu2.yaml
#
# CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
# bash run_docker.sh \
#   --modes sweep_language \
#   --sweep-yaml xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml \
#   --sweep-args "--dry-run"
#
# bash run_docker.sh -- bash -lc 'find /workspace/runs -name "pytorch_model.bin" | head'
