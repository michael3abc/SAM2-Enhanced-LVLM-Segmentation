# X-SAM CLI 快速手冊（Train / Eval / Demo）

這份文件只講你每天會用到的 CLI。

## 0) 先決條件

專案根目錄假設為：

```bash
cd /home/michael/projects/Research/X-SAM
```

`run.sh` 會優先找 `.venv/bin/python`，建議先確認：

```bash
test -x .venv/bin/python && echo "venv ok" || echo "venv missing"
```

---

## 1) 一句話總覽

```bash
bash run.sh --modes <train|segeval|demo> --config <config.py> [--work-dir <dir>] [--suffix <tag>]
```

`--config` 是必要參數。  
`--modes` 可單個或逗號串接，例如 `train,segeval`。  
`--work-dir` 不給就自動用 `runs/<stage>/<model_name>`。

---

## 2) Train

### 2.1 單純訓練

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py

CUDA_VISIBLE_DEVICES=0,1 \
GPU_PER_NODE=2 \
bash run.sh --modes train --config "$CFG"
```

### 2.2 指定輸出資料夾

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0,1 \
GPU_PER_NODE=2 \
bash run.sh --modes train --config "$CFG" --work-dir "$WORK"
```

### 2.3 常用訓練環境變數

- `CUDA_VISIBLE_DEVICES=0,1`：選 GPU。
- `GPU_PER_NODE=2`：`torchrun --nproc_per_node`。
- `MASTER_PORT=29601`：多次開不同 job 時避免 port 衝突。
- `DEEPSPEED_CFG=deepspeed_zero2|deepspeed_zero3|deepspeed_zero2_offload`：覆蓋 run.sh 預設。

---

## 3) Eval（Segmentation）

### 3.1 用 `run.sh` 跑完整 segeval（建議）

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0,1 \
GPU_PER_NODE=2 \
MASTER_PORT=29601 \
bash run.sh --modes segeval --config "$CFG" --work-dir "$WORK" --resume
```

### 3.2 直接呼叫 `eval.py`（只評特定 dataset）

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/torchrun --nproc_per_node=1 \
  xsam/xsam/tools/eval.py "$CFG" \
  --launcher pytorch \
  --work-dir "$WORK" \
  --pth_model latest \
  --data-names point_intseg box_intseg
```

### 3.3 輸出結果位置

- 預測檔：`$WORK/pred_data/<dataset_name>/...`
- 聚合 CSV：`$WORK/pred_data/results.csv`（每個 dataset 一筆）
- log：`$WORK/segeval-<timestamp>.log`

---

## 4) Demo（推論）

你有兩種方式：單張/資料夾 CLI，或 Gradio。

### 4.1 單張或資料夾 CLI 推論（`demo.py`）

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2
CKPT=$WORK/pytorch_model.bin

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python xsam/xsam/demo/demo.py "$CFG" \
  --pth_model "$CKPT" \
  --image xsam/xsam/demo/images/genseg.jpg \
  --prompt "ins: person, bird, boat; sem: water, sky" \
  --task_name genseg \
  --output_dir "$WORK/demo_out"
```

常見 `task_name`：

- `imgconv`
- `genseg`
- `refseg`
- `reaseg`
- `gcgseg`
- `intseg`
- `vgdseg`

### 4.2 啟動 Gradio Demo（`app.py`）

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0 \
GPU_PER_NODE=1 \
bash run.sh --modes demo --config "$CFG" --work-dir "$WORK"
```

預設 port 在 `run.sh` 是 `7862`。

---

## 5) Batch Size / Device 要在哪裡調

### 5.1 Device

最直接就是環境變數：

```bash
CUDA_VISIBLE_DEVICES=0
GPU_PER_NODE=1
```

### 5.2 Batch Size

- `train`：看 config 裡的 `train_dataloader.batch_size`。
- `eval.py`：目前程式內固定 `batch_size=1`。
- `demo.py`：單筆推論流程（不是 batched dataloader）。
- `visualize.py`：有 `--batch-size` 可調。

---

## 6) 常見工作流

### 6.1 Train -> Eval

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 MASTER_PORT=29601 \
bash run.sh --modes train,segeval --config "$CFG" --work-dir "$WORK"
```

### 6.2 只用現成 checkpoint 做 Demo

```bash
CFG=xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune.py
WORK=runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu2_mixed_finetune_v2

CUDA_VISIBLE_DEVICES=0 \
bash run.sh --modes demo --config "$CFG" --work-dir "$WORK"
```

---

## 7) 快速除錯

- checkpoint 不存在：先確認 `"$WORK/pytorch_model.bin"`。
- `--pth_model latest` 找不到：`--work-dir` 要對。
- NCCL 卡住：先單卡測（`CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1`）。
- port 衝突：換 `MASTER_PORT`。
- OOM：先降 GPU 數外的 batch/輸入大小，再考慮切 ZeRO。
