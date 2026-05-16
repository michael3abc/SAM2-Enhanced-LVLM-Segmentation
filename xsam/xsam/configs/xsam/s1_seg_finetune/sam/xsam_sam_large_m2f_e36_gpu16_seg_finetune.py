from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from torch.optim import AdamW
from xtuner.dataset.samplers import LengthGroupedSampler

from xsam.dataset import GenSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import genseg_postprocess_fn, process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.evaluation.evaluators import GenSegEvaluator
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel

_DEFAULT_PROFILE_YAML = "profiles/large_1024_e36_gpu16.yaml"


def _build_profile(profile_dict: dict[str, Any]) -> dict[str, Any]:
    """Build a validated profile dictionary from a plain dictionary.

    Args:
        profile_dict: Raw profile mapping loaded from YAML.

    Returns:
        Parsed and validated profile dictionary.
    """
    required_keys = (
        "profile_name",
        "seg_encoder_name",
        "image_size",
        "batch_size",
        "accumulative_counts",
        "dataloader_num_workers",
        "max_epochs",
        "save_steps",
        "save_total_limit",
        "logging_interval",
        "use_activation_checkpointing",
        "find_unused_parameters",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory",
    )
    missing_keys = [key for key in required_keys if key not in profile_dict]
    if missing_keys:
        raise KeyError(f"Profile YAML is missing keys: {missing_keys}")

    betas = profile_dict.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"`betas` must be a list/tuple with 2 values. Got: {betas}")

    milestones = profile_dict.get("milestones", [24, 30])
    if not isinstance(milestones, (list, tuple)) or len(milestones) == 0:
        raise ValueError(f"`milestones` must be a non-empty list/tuple. Got: {milestones}")

    return dict(
        profile_name=str(profile_dict["profile_name"]),
        seg_encoder_name=str(profile_dict["seg_encoder_name"]),
        image_size=int(profile_dict["image_size"]),
        batch_size=int(profile_dict["batch_size"]),
        accumulative_counts=int(profile_dict["accumulative_counts"]),
        dataloader_num_workers=int(profile_dict["dataloader_num_workers"]),
        max_epochs=int(profile_dict["max_epochs"]),
        optim_type=str(profile_dict.get("optim_type", "AdamW")),
        lr=float(profile_dict.get("lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(profile_dict.get("weight_decay", 0.05)),
        max_norm=float(profile_dict.get("max_norm", 0.01)),
        warmup_ratio=float(profile_dict.get("warmup_ratio", 0.03)),
        milestones=tuple(int(milestone) for milestone in milestones),
        save_steps=int(profile_dict["save_steps"]),
        save_total_limit=int(profile_dict["save_total_limit"]),
        logging_interval=int(profile_dict["logging_interval"]),
        use_activation_checkpointing=bool(profile_dict["use_activation_checkpointing"]),
        find_unused_parameters=bool(profile_dict["find_unused_parameters"]),
        persistent_workers=bool(profile_dict["persistent_workers"]),
        prefetch_factor=int(profile_dict["prefetch_factor"]),
        pin_memory=bool(profile_dict["pin_memory"]),
    )


def _resolve_optimizer(optim_name: str):
    """Resolve optimizer class from profile name.

    Args:
        optim_name: Optimizer type name from profile.

    Returns:
        Optimizer class object.
    """
    if optim_name == "AdamW":
        return AdamW

    raise ValueError(f"Unsupported optimizer in profile: {optim_name}")


_config_dir = __import__("pathlib").Path(__file__).resolve().parent
_profile_yaml = __import__("os").environ.get(
    "XSAM_CONFIG_PROFILE_YAML",
    __import__("os").environ.get("XSAM_SEG_PROFILE_YAML", _DEFAULT_PROFILE_YAML),
)
_profile_yaml_path = __import__("pathlib").Path(_profile_yaml)
if not _profile_yaml_path.is_absolute():
    _profile_yaml_path = (_config_dir / _profile_yaml_path).resolve()
if not _profile_yaml_path.exists():
    raise FileNotFoundError(f"Cannot find profile yaml: {_profile_yaml} -> {_profile_yaml_path}")
_profile_cfg = __import__("mmengine.config", fromlist=["Config"]).Config.fromfile(str(_profile_yaml_path))
_profile_dict = _profile_cfg._cfg_dict.to_dict()
if not isinstance(_profile_dict, dict):
    raise TypeError(f"Profile YAML must be a mapping. Got: {type(_profile_dict)!r}")
profile = _build_profile(_profile_dict)
segmentor_policy = __import__(
    "xsam.configs.xsam.profile_utils",
    fromlist=["build_segmentor_train_policy"],
).build_segmentor_train_policy(
    _profile_dict,
    default_trunk_mode="full_ft",
    default_fpn_mode="full_ft",
)
segmentor_lora_cfg = None
if segmentor_policy["segmentor_lora_kwargs"] is not None:
    segmentor_lora_cfg = dict(type=LoraConfig, **segmentor_policy["segmentor_lora_kwargs"])

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = __import__("os").environ.get("CODE_DIR", "./xsam/")
data_dir = __import__("os").environ.get("DATA_DIR", "./data/")
init_dir = __import__("os").environ.get("INIT_DIR", "./inits/")
work_dir = __import__("os").environ.get("WORK_DIR", "./runs/")

# Profile
profile_name = profile["profile_name"]
profile_yaml_path = str(_profile_yaml_path)

# Model
seg_encoder_name_or_path = init_dir + profile["seg_encoder_name"]
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Data
data_root = data_dir + "genseg_data/"
data_path = data_root + "coco2017/annotations/panoptic_train2017.json"
image_folder = data_root + "coco2017/train2017"
panseg_map_folder = data_root + "coco2017/panoptic_train2017"
image_size = int(profile["image_size"])

# Scheduler & Optimizer
batch_size = profile["batch_size"]  # per_device
accumulative_counts = profile["accumulative_counts"]
dataloader_num_workers = profile["dataloader_num_workers"]
max_epochs = profile["max_epochs"]
optim_type = _resolve_optimizer(profile["optim_type"])
lr = profile["lr"]
betas = profile["betas"]
weight_decay = profile["weight_decay"]
max_norm = profile["max_norm"]  # grad clip
warmup_ratio = profile["warmup_ratio"]
milestones = list(profile["milestones"])

# Save
save_steps = profile["save_steps"]
save_total_limit = profile["save_total_limit"]  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = profile["logging_interval"]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_segmentor_encoder=segmentor_policy["freeze_segmentor_encoder"],
    freeze_segmentor_trunk=segmentor_policy["freeze_segmentor_trunk"],
    freeze_segmentor_fpn=segmentor_policy["freeze_segmentor_fpn"],
    use_activation_checkpointing=profile["use_activation_checkpointing"],
    segmentor_lora=segmentor_lora_cfg,
    postprocess_fn=genseg_postprocess_fn,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=True,
        close_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": image_size},
        "do_crop": True,
        "crop_size": {"height": image_size, "width": image_size},
    }
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
    persistent_workers=profile["persistent_workers"] if dataloader_num_workers > 0 else False,
    pin_memory=profile["pin_memory"],
    dataset=pannoptic_genseg_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

if dataloader_num_workers > 0:
    train_dataloader["prefetch_factor"] = profile["prefetch_factor"]

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
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
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
        milestones=milestones,
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

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "connector", "segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
    dict(type=PTCheckpointHook, clean_pth=False),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

find_unused_parameters = profile["find_unused_parameters"]

del _config_dir
del _profile_cfg
del _profile_dict
del _profile_yaml
del _profile_yaml_path
del _build_profile
del _resolve_optimizer

"""
bash ./run.sh --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam/xsam_sam_large_m2f_e36_gpu16_seg_finetune.py \
  --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam/profiles/large_1024_e36_gpu16.yaml
"""
