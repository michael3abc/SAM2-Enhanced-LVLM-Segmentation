# SAM Enhanced LVLM Segmentation

## Overview
This repository extends [X-SAM](https://github.com/wanghao9610/X-SAM) with a reproducible, research-oriented framework for SAM-enhanced LVLM segmentation.

Key enhancements over the original X-SAM implementation include:

- **SAM2/SAM3 support**: unified integration of SAM2 and SAM3 backbones within the X-SAM training and evaluation stack.
- **Training-oriented execution pipeline**: standardized entry scripts (`run.sh`, `run_docker.sh`), improved resume semantics, and robust defaults for long-running experiments.
- **Layer-wise best-layer search**: a dedicated SAM3 probing workflow (`layer_sweep_spatial.py`) that trains per-layer probe heads and performs empirical layer selection.
- **Containerized reproducibility**: Docker-based environment management with pinned dependencies, GPU runtime configuration, and explicit Hugging Face cache mapping.

## 1. Prerequisites

- Linux host
- NVIDIA driver installed on host
- Docker + Docker Compose plugin
- NVIDIA Container Toolkit (for `gpus: all`)

## 2. Clone and Configure

```bash
git clone https://github.com/michael3abc/SAM-Enhanced-LVLM-Segmentation.git
cd SAM-Enhanced-LVLM-Segmentation

cp docker/.env.example docker/.env
```

Edit `docker/.env`:

- Set `HOST_HF_HOME` to your host cache path (example: `/home/<user>/.cache/huggingface`).
- Set `CONTAINER_HF_HOME` to the same absolute path unless you have a specific reason to change it.
- Adjust GPU and runtime fields if needed:
  - `CUDA_VISIBLE_DEVICES`
  - `GPU_PER_NODE`
  - `MASTER_PORT`
  - `SHM_SIZE`

## 3. Build Docker Environment

Build once (or after dependency/image changes):

```bash
bash run_docker.sh --build \
  --modes sweep_spatial \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
  --sweep-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml
```

Notes:

- The build installs dependencies inside `/opt/venv`.
- `flash_attn` is downloaded from `FLASH_ATTN_WHEEL_URL` in `docker/.env` and verified by `FLASH_ATTN_WHEEL_SHA256`.

## 4. Data and Model Layout

Host directories are mounted into the container by `docker/compose.yaml`:

- `../data` -> `/workspace/data`
- `../inits` -> `/workspace/inits`
- `../runs` -> `/workspace/runs`
- `HOST_HF_HOME` -> `CONTAINER_HF_HOME`

Expected core layout:

```text
data/
inits/
runs/
```

If you need to download datasets from inside Docker:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env run --rm dev \
  bash  \
    --datasets all \
    --root-dir /workspace \
    --threads 8
```

## 5. Unified Train/Sweep CLI

Both local and Docker workflows share the same `run.sh` interface:

- `--modes train|sweep|sweep_spatial|sweep_language|segeval|vlmeval|visualize|demo`
- `--config`: mmengine config (required for train/spatial sweep)
- `--yaml`: profile YAML for config
- `--sweep-yaml`: sweep YAML (`spatial_sweep.yaml` or `language_sweep.yaml`)
- `--sweep-args`: extra args forwarded to sweep entry

### 5.1 Local (`run.sh`)

```bash
# Train (SAM3)
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam3/xsam_sam3_1008_e12_gpu2_seg_finetune.py \
  --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam3/profiles/base_1008_e12_gpu2.yaml

# Spatial sweep (alias mode: sweep == sweep_spatial)
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run.sh \
  --modes sweep_spatial \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
  --sweep-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml \
  --sweep-args "--layers=-1,-2,-4,-6 --train-epochs 1 --train-ratio 0.05"

# Language sweep (phase1->phase2->phase3 scheduler)
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run.sh \
  --modes sweep_language \
  --sweep-yaml xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml \
  --sweep-args "--dry-run"
```

### 5.2 Docker (`run_docker.sh`)

`run_docker.sh` forwards args to in-container `run.sh`.

```bash
# Build image once
bash run_docker.sh --build \
  --modes sweep_spatial \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py

# Train (sam3)
# stage1
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run_docker.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam/xsam_sam_large_m2f_e36_gpu16_seg_finetune.py \
  --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam/profiles/large_1024_e36_gpu16.yaml

# stage2
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run_docker.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/s2_align_pretrain/sam3/xsam_sam3_align_pretrain.py \
  --yaml xsam/xsam/configs/xsam/s2_align_pretrain/sam3/profiles/sma3_s2.yaml

# stage3
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run_docker.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/s3_mixed_finetune/sam3/xsam_sam3_mixed_finetune.py \
  --yaml xsam/xsam/configs/xsam/s3_mixed_finetune/sam3/profiles/phi3_mini.yaml

# Spatial sweep
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run_docker.sh \
  --modes sweep_spatial \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
  --sweep-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml \
  --sweep-args "--layers=-1,-2,-4,-6 --train-epochs 1 --train-ratio 0.05"

# Language sweep
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run_docker.sh \
  --modes sweep_language \
  --sweep-yaml xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml \
  --sweep-args "--dry-run"
```

### 5.3 Optional Eval/Demo

```bash
YAML=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/profiles/base_plus_1024_gpu2.yaml
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_sam2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_sam2_mixed_finetune__base_plus_1024_gpu2

CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 MASTER_PORT=29601 \
bash run_docker.sh \
  --modes segeval \
  --config "$CFG" \
  --yaml "$YAML" \
  --work-dir "$WORK" \
  --resume

CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run_docker.sh \
  --modes demo \
  --config "$CFG" \
  --yaml "$YAML" \
  --work-dir "$WORK"
```

## 6. Direct Docker Compose Usage (Optional)

```bash
docker compose -f docker/compose.yaml --env-file docker/.env build dev
docker compose -f docker/compose.yaml --env-file docker/.env run --rm dev bash
```

Inside container:

```bash
bash run.sh --help
```
