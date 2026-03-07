# Docker quickstart

```bash
cd /home/michael/projects/Research/X-SAM
cp docker/.env docker/.env

docker compose -f docker/compose.yaml --env-file docker/.env build dev
docker compose -f docker/compose.yaml --env-file docker/.env run --rm dev
```

Run `run.sh` inside container:

```bash
bash run.sh --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam3/xsam_sam3_1008_e12_gpu2_seg_finetune.py
```

Launch demo container profile:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env --profile demo run --rm demo
```
