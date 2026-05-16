import torch
import os
from pathlib import Path
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from torch.optim import AdamW
from xtuner.dataset.samplers import LengthGroupedSampler

from xsam.dataset import GenSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import genseg_postprocess_fn, process_map_fn_factory
from xsam.dataset.processors import Sam3ImageProcessor
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.evaluation.evaluators import GenSegEvaluator
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam3 import Sam3Model
from xsam.layer_analysis.common.sweep_cfg import SaptialSweepCfg

_DEFAULT_PROFILE = {
    "profile_name": "default",
    "batch_size": 2,
    "accumulative_counts": 32,
    "dataloader_num_workers": 8,
    "max_epochs": 12,
    "logging_interval": 50,
    "sweep_num_workers": 4,
    "sweep_train_epochs": 2,
    "sweep_grad_accum_steps": 24,
    "sweep_train_num_workers": 4,
    "sweep_resume": True,
    "sweep_resume_ckpt": "runs/sweep_spatial_best_layers/sam3_probe_ep2/checkpoints/latest.pt",
    "sweep_train_eval_interval": 0,
    "sweep_train_eval_max_samples": 128,
}
_config_dir = Path(__file__).resolve().parent


def _resolve_runtime_path(path_like: str, config_dir: Path) -> Path:
    """Resolve runtime path with cwd-first fallback strategy.

    Args:
        path_like: Runtime path from env.
        config_dir: Directory of the current config file.
    Returns:
        Resolved absolute path.
    """
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    cwd_candidate = path_obj.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (config_dir / path_obj).resolve()


def _load_mapping_from_config_file(path: Path) -> dict:
    """Load mapping config from a Python/YAML file with mmengine Config.

    Args:
        path: Config file path.
    Returns:
        Parsed mapping object.
    """
    loaded = Config.fromfile(str(path))._cfg_dict.to_dict()
    if not isinstance(loaded, dict):
        raise TypeError(f"Profile YAML must be a mapping. Got: {type(loaded)!r}")
    return loaded


def _load_runtime_profile(config_dir: Path) -> tuple[dict, str]:
    """Load runtime profile from env, sweep YAML, or defaults.

    Args:
        config_dir: Directory of the current config file.
    Returns:
        Tuple of merged profile mapping and profile source string.
    """
    profile_env = os.environ.get("XSAM_CONFIG_PROFILE_YAML", os.environ.get("XSAM_SEG_PROFILE_YAML"))
    if profile_env:
        profile_path = _resolve_runtime_path(profile_env, config_dir)
        if not profile_path.exists():
            raise FileNotFoundError(f"Cannot find profile yaml: {profile_env} -> {profile_path}")
        loaded = _load_mapping_from_config_file(profile_path)
        merged = dict(_DEFAULT_PROFILE)
        merged.update(loaded)
        return merged, str(profile_path)

    sweep_yaml_env = os.environ.get("XSAM_SPATIAL_SWEEP_YAML_PATH")
    if sweep_yaml_env:
        sweep_yaml_path = _resolve_runtime_path(sweep_yaml_env, config_dir)
        if not sweep_yaml_path.exists():
            raise FileNotFoundError(f"Cannot find sweep yaml: {sweep_yaml_env} -> {sweep_yaml_path}")
        sweep_loaded = _load_mapping_from_config_file(sweep_yaml_path)
        profile_loaded = sweep_loaded.get("mmengine_profile", {})
        if not isinstance(profile_loaded, dict):
            raise TypeError(
                f"`mmengine_profile` in sweep yaml must be mapping. Got: {type(profile_loaded)!r}"
            )
        merged = dict(_DEFAULT_PROFILE)
        merged.update(profile_loaded)
        return merged, f"{sweep_yaml_path}::mmengine_profile"

    return dict(_DEFAULT_PROFILE), "<built-in defaults>"


_profile, _profile_source = _load_runtime_profile(_config_dir)

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = __import__("os").environ.get("CODE_DIR", "./xsam/")
data_dir = __import__("os").environ.get("DATA_DIR", "./data/")
init_dir = __import__("os").environ.get("INIT_DIR", "./inits/")
work_dir = __import__("os").environ.get("WORK_DIR", "./runs/")

# Model
# 1. sam3
# Load trunk-only checkpoint from a single file.
# `strict=False` + trunk-only keys => trunk gets pretrained weight, FPN keeps random init.
seg_encoder_name_or_path = init_dir + "sam3/sam3_encoder.bin"
sam3_encoder_trunk = "sam3_encoder.bin"
sam3_simple_FPN = "sam3_fpn.bin"

# Split freeze control.
# - freeze_segmentor_encoder=True: whole encoder (trunk+fpn) no_grad.
# - freeze_segmentor_trunk=True + freeze_segmentor_fpn=False: trunk no_grad, fpn trainable.
freeze_segmentor_encoder = False
freeze_segmentor_trunk = True
freeze_segmentor_fpn = False

# 2. m2f decoder
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"
# Initialize only Mask2Former decoder/pixel_decoder weights.
s1_pretrained_pth = None

# Data
data_root = data_dir + "genseg_data/"
data_path = data_root + "coco2017/annotations/panoptic_train2017.json"
image_folder = data_root + "coco2017/train2017"
panseg_map_folder = data_root + "coco2017/panoptic_train2017"
image_size = int(1008)

# Scheduler & Optimizer
# Keep effective global batch close to gpu1 config when using 2 GPUs.
profile_name = str(_profile["profile_name"])
profile_yaml_path = str(_profile_source)
batch_size = int(_profile["batch_size"])  # per_device
accumulative_counts = int(_profile["accumulative_counts"])
dataloader_num_workers = int(_profile["dataloader_num_workers"])
max_epochs = int(_profile["max_epochs"])
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 0.01  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 5000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = int(_profile["logging_interval"])

# Sweep defaults
sweep_pth_model = None
sweep_layers = "-1,-2,-4,-6,-8,-10,-12,-16,-24,-32"
sweep_data_names = None
sweep_dense_keywords = "semantic_genseg,semantic_ovseg"
sweep_ref_keywords = "refseg"
sweep_num_workers = int(_profile.get("sweep_num_workers", dataloader_num_workers))
sweep_max_samples_per_task = 0
sweep_train_epochs = int(_profile.get("sweep_train_epochs", max_epochs))
sweep_train_ratio = 0.25
sweep_grad_accum_steps = int(_profile.get("sweep_grad_accum_steps", accumulative_counts))
sweep_train_num_workers = int(_profile.get("sweep_train_num_workers", dataloader_num_workers))
sweep_max_save = -save_total_limit
sweep_resume = bool(_profile.get("sweep_resume", False))
sweep_resume_ckpt = _profile.get("sweep_resume_ckpt")
sweep_probe_reinit = False
sweep_train_eval_interval = int(_profile.get("sweep_train_eval_interval", 200))
sweep_train_eval_max_samples = int(_profile.get("sweep_train_eval_max_samples", 256))
sweep_early_stop_patience_steps = 2000
sweep_early_stop_miou_eps = 0.1
sweep_seed_stride = 9973
sweep_output_csv = "sweep_L_spatial.csv"
sweep_output_root = "runs/sweep_spatial_best_layers"
sweep_run_name = "sam3_probe_ep2"
sweep_use_tqdm = False
sweep_eval_fail_fast = True
sweep_eval_fail_ratio_threshold = 0.05
sweep_eval_fail_check_min_samples = 64
sweep_eval_oom_empty_cache = True
sweep_eval_log_cuda_mem = True
sweep_seed = 1024
sweep_cfg_options = None

# Spatial layer sweep config (used by xsam/xsam/layer_analysis/spatial/layer_sweep_spatial.py).
sweep_spatial_cfg = dict(
    type=SaptialSweepCfg,
    config="",
    pth_model=sweep_pth_model,
    layers=sweep_layers,
    data_names=sweep_data_names,
    dense_keywords=sweep_dense_keywords,
    ref_keywords=sweep_ref_keywords,
    batch_size=batch_size,
    num_workers=sweep_num_workers,
    max_samples_per_task=sweep_max_samples_per_task,
    train_epochs=sweep_train_epochs,
    train_ratio=sweep_train_ratio,
    train_batch_size=batch_size,
    grad_accum_steps=sweep_grad_accum_steps,
    train_num_workers=sweep_train_num_workers,
    save_steps=save_steps,
    max_save=sweep_max_save,
    resume=sweep_resume,
    resume_ckpt=sweep_resume_ckpt,
    probe_lr=lr,
    probe_weight_decay=weight_decay,
    probe_reinit=sweep_probe_reinit,
    train_eval_interval=sweep_train_eval_interval,
    train_eval_max_samples=sweep_train_eval_max_samples,
    early_stop_patience_steps=sweep_early_stop_patience_steps,
    early_stop_miou_eps=sweep_early_stop_miou_eps,
    seed_stride=sweep_seed_stride,
    output_csv=sweep_output_csv,
    output_root=sweep_output_root,
    run_name=sweep_run_name,
    use_tqdm=sweep_use_tqdm,
    log_interval=logging_interval,
    eval_fail_fast=sweep_eval_fail_fast,
    eval_fail_ratio_threshold=sweep_eval_fail_ratio_threshold,
    eval_fail_check_min_samples=sweep_eval_fail_check_min_samples,
    eval_oom_empty_cache=sweep_eval_oom_empty_cache,
    eval_log_cuda_mem=sweep_eval_log_cuda_mem,
    seed=sweep_seed,
    cfg_options=sweep_cfg_options,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
extra_image_processor = dict(
    type=Sam3ImageProcessor,
    ignore_index=0,
    size={"height": image_size, "width": image_size},
    pad_size={"height": image_size, "width": image_size},
    mask_size={"height": image_size, "width": image_size},
    mask_pad_size={"height": image_size, "width": image_size},
)

model = dict(
    type=XSamModel,
    freeze_segmentor_encoder=freeze_segmentor_encoder,
    freeze_segmentor_trunk=freeze_segmentor_trunk,
    freeze_segmentor_fpn=freeze_segmentor_fpn,
    use_activation_checkpointing=False,
    s1_pretrained_pth=s1_pretrained_pth,
    postprocess_fn=genseg_postprocess_fn,
    connector_type=None,
    seg_select_layers=[0, 1, 2],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=Sam3Model.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            encoder_filename= sam3_encoder_trunk,
            fpn_filename=sam3_simple_FPN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            strict=False,
            map_location="cpu",
        ),
        decoder=dict(
            type=Mask2FormerModel.from_pretrained,
            pretrained_model_name_or_path=seg_decoder_name_or_path,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                image_size=image_size,
                # Keep channels aligned with SAM3 FPN output (256 each), so no extra bridge is needed.
                feature_channels=[256, 256, 256],
                feature_strides=[4, 8, 16],
                common_stride=4,
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=False,
        close_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_extra_image_processor = dict(
    type=Sam3ImageProcessor,
    ignore_index=0,
    size={"min_scale": 0.1, "max_scale": 2.0, "target_size": image_size},
    do_crop=True,
    crop_size={"height": image_size, "width": image_size},
)

pannoptic_genseg_dataset = dict(
    type=GenSegDataset,
    data_path=data_path,
    image_folder=image_folder,
    panseg_map_folder=panseg_map_folder,
    extra_image_processor=train_extra_image_processor,
    task_name="genseg",
    data_name="coco_panoptic_genseg",
    pad_image_to_square=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
    dataset=pannoptic_genseg_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

val_datasets = [
    dict(
        type=GenSegDataset,
        data_path=data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        panseg_map_folder=data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=data_root + "coco2017/panoptic_semseg_val2017",
        task_name="genseg",
        data_name="coco_panoptic_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=genseg_postprocess_fn, task_name="panoptic_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        panseg_map_folder=data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=data_root + "coco2017/panoptic_semseg_val2017",
        task_name="genseg",
        data_name="coco_panoptic_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=genseg_postprocess_fn, task_name="semantic_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=data_root + "coco2017/val2017",
        task_name="genseg",
        data_name="instance_genseg",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=genseg_postprocess_fn, task_name="instance_genseg"),
        extra_image_processor=extra_image_processor,
        pad_image_to_square=True,
    ),
]

val_evaluators = [
    dict(
        type=GenSegEvaluator,
        data_name="coco_panoptic_genseg",
        distributed=True,
    ),
    dict(
        type=GenSegEvaluator,
        data_name="coco_panoptic_semantic_genseg",
        distributed=True,
    ),
    dict(
        type=GenSegEvaluator,
        data_name="coco_instance_genseg",
        distributed=True,
    ),
]

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        custom_keys=(
            {
                "segmentor.encoder": dict(lr_mult=0.0, decay_mult=1.0),
            }
            if freeze_segmentor_encoder
            else (
                (
                    {
                        "segmentor.encoder.vision_backbone.trunk": dict(lr_mult=0.0, decay_mult=1.0),
                        "segmentor.encoder.vision_backbone.convs": dict(lr_mult=1.0, decay_mult=1.0),
                    }
                    if freeze_segmentor_trunk and not freeze_segmentor_fpn
                    else {
                        "segmentor.encoder.vision_backbone.trunk": dict(lr_mult=1.0, decay_mult=1.0),
                        "segmentor.encoder.vision_backbone.convs": dict(lr_mult=0.0, decay_mult=1.0),
                    }
                    if (not freeze_segmentor_trunk) and freeze_segmentor_fpn
                    else {
                        "segmentor.encoder": dict(
                            lr_mult=0.0 if (freeze_segmentor_trunk and freeze_segmentor_fpn) else 1.0,
                            decay_mult=1.0,
                        ),
                    }
                )
                if (freeze_segmentor_trunk or freeze_segmentor_fpn)
                else {
                    "segmentor.encoder": dict(lr_mult=1.0, decay_mult=1.0),
                }
            )
        ),
    ),
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=MultiStepLR,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 10],
        gamma=0.1,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = None

custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
    dict(type=PTCheckpointHook, clean_pth=False),
]

# configure default hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

"""
CUDA_VISIBLE_DEVICES=0 GPU_PER_NODE=1 \
bash run.sh \
  --modes sweep \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py
"""
