# SAM Enhanced LVLM Segmentation

## Overview
This repository extends [X-SAM](https://github.com/wanghao9610/X-SAM) with a reproducible, research-oriented framework for SAM-enhanced LVLM segmentation.

Key enhancements over the original X-SAM implementation include:

- **SAM2/SAM3 support**: unified integration of SAM2 and SAM3 backbones within the X-SAM training and evaluation stack.
- **Training-oriented execution pipeline**: standardized entry scripts (`run.sh`, `run_docker.sh`), improved resume semantics, and robust defaults for long-running experiments.
- **Layer-wise best-layer search**: a dedicated SAM3 probing workflow (`sweep_L_spatial.py`) that trains per-layer probe heads and performs empirical layer selection.
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
  --modes sweep \
  --config xsam/xsam/configs/xsam/layer_analysis/xsam_sam3_spatial.py
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
  python scripts/dataset/datasets_preparing.py \
    --mode download \
    --root-dir . \
    --threads 8
```

## 5. Run Training/Eval/Sweep with Docker

Use `run_docker.sh` from host. It forwards arguments to in-container `run.sh`.

### 5.1 Layer Sweep

```bash
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run_docker.sh \
  --modes sweep \
  --config xsam/xsam/configs/xsam/layer_analysis/xsam_sam3_spatial.py
```

### 5.2 Train (Example: Stage 1)

```bash
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run_docker.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam3/xsam_sam3_1008_e12_gpu2_seg_finetune.py
```

### 5.3 Segmentation Eval

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 MASTER_PORT=29601 \
bash run_docker.sh \
  --modes segeval \
  --config "$CFG" \
  --work-dir "$WORK" \
  --resume
```

### 5.4 Demo

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run_docker.sh \
  --modes demo \
  --config "$CFG" \
  --work-dir "$WORK"
```

## 6. Direct Docker Compose Usage (Optional)

If you do not want the wrapper script:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env build dev
docker compose -f docker/compose.yaml --env-file docker/.env run --rm dev bash
```

Inside the container:

```bash
bash run.sh --help
```
