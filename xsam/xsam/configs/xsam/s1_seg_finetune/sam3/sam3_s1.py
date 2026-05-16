from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, LinearLR, MultiStepLR
from peft import LoraConfig
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


_DEFAULT_PROFILE_YAML = "profiles/base_1008_e12_gpu2.yaml"


def _ensure_trailing_slash(path_value: str) -> str:
    """Append a trailing slash when needed.

    Args:
        path_value: Directory path string.

    Returns:
        Directory path ending with ``/``.
    """
    return path_value if path_value.endswith("/") else path_value + "/"


def _resolve_runtime_path(
    prefix: str,
    rel_value,
    *,
    root_prefix: str | None = None,
    keep_bare_name: bool = False,
):
    """Resolve an optional profile path against runtime directories.

    Args:
        prefix: Base directory for generic relative paths.
        rel_value: Path value from the profile.
        root_prefix: Optional project root for repo-relative paths.
        keep_bare_name: Whether bare filenames should be returned unchanged.

    Returns:
        Resolved path string, bare filename, or ``None``.
    """
    if rel_value in (None, "", "null"):
        return None

    raw = str(rel_value)
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return str(raw_path)

    if keep_bare_name and "/" not in raw and "\\" not in raw:
        return raw

    if root_prefix is not None:
        root_path = Path(root_prefix)
        if raw.startswith("./") or raw.startswith("../"):
            return str((root_path / raw).resolve())
        if raw.startswith(("runs/", "data/", "inits/", "xsam/", "external/", "scripts/")):
            return str((root_path / raw).resolve())

    return str((Path(prefix) / raw).resolve())


def _resolve_profile_yaml(config_dir: Path) -> Path:
    """Resolve the active profile YAML path.

    Args:
        config_dir: Directory containing this config file.

    Returns:
        Existing profile YAML path.
    """
    profile_yaml = __import__("os").environ.get(
        "XSAM_CONFIG_PROFILE_YAML",
        __import__("os").environ.get("XSAM_SEG_PROFILE_YAML", _DEFAULT_PROFILE_YAML),
    )
    profile_path = Path(profile_yaml)
    if not profile_path.is_absolute():
        profile_path_from_cwd = profile_path.resolve()
        if profile_path_from_cwd.exists():
            profile_path = profile_path_from_cwd
        else:
            profile_path = (config_dir / profile_path).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"Cannot find profile yaml: {profile_yaml} -> {profile_path}")
    return profile_path


def _build_profile(profile_dict: dict[str, Any]) -> dict[str, Any]:
    """Build a validated SAM3 Stage1 profile dictionary.

    Args:
        profile_dict: Raw profile mapping loaded from YAML.

    Returns:
        Parsed and normalized profile dictionary.
    """
    required_keys = (
        "profile_name",
        "seg_encoder_name",
        "image_size",
        "batch_size",
        "accumulative_counts",
        "dataloader_num_workers",
        "save_steps",
        "save_total_limit",
        "logging_interval",
        "use_activation_checkpointing",
        "find_unused_parameters",
        "persistent_workers",
        "pin_memory",
    )
    missing_keys = [key for key in required_keys if key not in profile_dict]
    if "max_epochs" not in profile_dict and "max_iters" not in profile_dict:
        missing_keys.append("max_epochs or max_iters")
    if missing_keys:
        raise KeyError(f"Profile YAML is missing keys: {missing_keys}")

    betas = profile_dict.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"`betas` must be a list/tuple with 2 values. Got: {betas}")

    max_iters = profile_dict.get("max_iters", None)
    max_epochs = profile_dict.get("max_epochs", None)

    return dict(
        profile_name=str(profile_dict["profile_name"]),
        seg_encoder_name=str(profile_dict["seg_encoder_name"]),
        seg_decoder_name=str(profile_dict.get("seg_decoder_name", "mask2former-swin-large-coco-panoptic")),
        sam3_encoder_trunk=str(profile_dict.get("sam3_encoder_trunk", "sam3_encoder.bin")),
        sam3_simple_fpn=str(profile_dict.get("sam3_simple_fpn", "sam3_fpn.bin")),
        s1_pretrained_pth=profile_dict.get(
            "s1_pretrained_pth",
            "extracted_weights/mask2former_decoder/xsam_mask2former_decoder.bin",
        ),
        image_size=int(profile_dict["image_size"]),
        batch_size=int(profile_dict["batch_size"]),
        accumulative_counts=int(profile_dict["accumulative_counts"]),
        dataloader_num_workers=int(profile_dict["dataloader_num_workers"]),
        max_iters=None if max_iters in (None, "", "null") else int(max_iters),
        max_epochs=None if max_epochs in (None, "", "null") else float(max_epochs),
        optim_type=str(profile_dict.get("optim_type", "AdamW")),
        lr=float(profile_dict.get("lr", 1e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(profile_dict.get("weight_decay", 0.05)),
        max_norm=float(profile_dict.get("max_norm", 0.01)),
        warmup_ratio=float(profile_dict.get("warmup_ratio", 0.03)),
        milestones=[int(value) for value in profile_dict.get("milestones", [8, 10])],
        save_steps=int(profile_dict["save_steps"]),
        save_total_limit=int(profile_dict["save_total_limit"]),
        logging_interval=int(profile_dict["logging_interval"]),
        use_activation_checkpointing=bool(profile_dict["use_activation_checkpointing"]),
        find_unused_parameters=bool(profile_dict["find_unused_parameters"]),
        persistent_workers=bool(profile_dict["persistent_workers"]),
        prefetch_factor=int(profile_dict.get("prefetch_factor", 2)),
        pin_memory=bool(profile_dict["pin_memory"]),
        data_root=str(profile_dict.get("data_root", "genseg_data/")),
        train_data_relpath=str(profile_dict.get("train_data_relpath", "coco2017/annotations/panoptic_train2017.json")),
        train_image_folder_relpath=str(profile_dict.get("train_image_folder_relpath", "coco2017/train2017")),
        train_panseg_map_relpath=profile_dict.get("train_panseg_map_relpath", "coco2017/panoptic_train2017"),
        train_semseg_map_relpath=profile_dict.get("train_semseg_map_relpath", None),
        train_task_name=str(profile_dict.get("train_task_name", "genseg")),
        train_data_name=str(profile_dict.get("train_data_name", "coco_panoptic_genseg")),
        val_datasets=profile_dict.get("val_datasets", None),
        pad_image_to_square=bool(profile_dict.get("pad_image_to_square", True)),
        train_min_scale=float(profile_dict.get("train_min_scale", 0.1)),
        train_max_scale=float(profile_dict.get("train_max_scale", 2.0)),
        sam3_neck_hidden_size=int(profile_dict.get("sam3_neck_hidden_size", 256)),
        sam3_neck_scale_factors=list(profile_dict.get("sam3_neck_scale_factors", [4.0, 2.0, 1.0, 0.5])),
        sam3_add_sam2_neck=bool(profile_dict.get("sam3_add_sam2_neck", False)),
        sam3_scalp=int(profile_dict.get("sam3_scalp", 1)),
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


def _infer_postprocess_task_name(data_name: str, explicit_value: str | None = None) -> str:
    """Infer the GenSeg postprocess task name from a data name.

    Args:
        data_name: Dataset name from config.
        explicit_value: Optional explicit task name from profile.

    Returns:
        Postprocess task name.
    """
    if explicit_value not in (None, "", "null"):
        return str(explicit_value)
    if "instance" in data_name:
        return "instance_genseg"
    if "semantic" in data_name:
        return "semantic_genseg"
    return "panoptic_genseg"


def _build_dataset_cfg(
    *,
    data_path: str,
    image_folder: str,
    extra_processor: dict,
    task_name: str,
    data_name: str,
    pad_image_to_square: bool,
    panseg_map_folder: str | None = None,
    semseg_map_folder: str | None = None,
    data_mode: str | None = None,
    postprocess_task_name: str | None = None,
) -> dict:
    """Build a GenSegDataset config dictionary.

    Args:
        data_path: COCO-style annotation file.
        image_folder: Image directory.
        extra_processor: Extra image processor config.
        task_name: X-SAM task name.
        data_name: Dataset name.
        pad_image_to_square: Whether to square-pad input images.
        panseg_map_folder: Optional panoptic RGB map directory.
        semseg_map_folder: Optional semantic map directory.
        data_mode: Optional dataset mode such as ``eval``.
        postprocess_task_name: Optional evaluation postprocess task.

    Returns:
        Dataset config dictionary.
    """
    dataset_cfg = dict(
        type=GenSegDataset,
        data_path=data_path,
        image_folder=image_folder,
        extra_image_processor=extra_processor,
        task_name=task_name,
        data_name=data_name,
        pad_image_to_square=pad_image_to_square,
    )
    if panseg_map_folder is not None:
        dataset_cfg["panseg_map_folder"] = panseg_map_folder
    if semseg_map_folder is not None:
        dataset_cfg["semseg_map_folder"] = semseg_map_folder
    if data_mode is not None:
        dataset_cfg["data_mode"] = data_mode
    if data_mode == "eval":
        dataset_cfg["postprocess_fn"] = dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name=_infer_postprocess_task_name(data_name, postprocess_task_name),
        )
    return dataset_cfg


def _build_val_dataset_cfgs(
    *,
    profile: dict[str, Any],
    data_root: str,
    data_dir: str,
    root_dir: str,
    eval_processor: dict,
) -> tuple[list[dict], list[str]]:
    """Build validation dataset configs from profile entries.

    Args:
        profile: Parsed profile dictionary.
        data_root: Resolved dataset root directory.
        data_dir: Runtime data directory.
        root_dir: Runtime project root directory.
        eval_processor: Evaluation image processor config.

    Returns:
        Tuple of validation dataset configs and evaluator dataset names.
    """
    raw_val_datasets = profile["val_datasets"]
    if raw_val_datasets in (None, "", "null"):
        raw_val_datasets = [
            dict(
                data_relpath="coco2017/annotations/panoptic_val2017.json",
                image_folder_relpath="coco2017/val2017",
                panseg_map_relpath="coco2017/panoptic_val2017",
                semseg_map_relpath="coco2017/panoptic_semseg_val2017",
                data_name="coco_panoptic_genseg",
                postprocess_task_name="panoptic_genseg",
                evaluator_data_name="coco_panoptic_genseg",
            ),
            dict(
                data_relpath="coco2017/annotations/panoptic_val2017.json",
                image_folder_relpath="coco2017/val2017",
                panseg_map_relpath="coco2017/panoptic_val2017",
                semseg_map_relpath="coco2017/panoptic_semseg_val2017",
                data_name="coco_panoptic_genseg",
                postprocess_task_name="semantic_genseg",
                evaluator_data_name="coco_panoptic_semantic_genseg",
            ),
            dict(
                data_relpath="coco2017/annotations/instances_val2017.json",
                image_folder_relpath="coco2017/val2017",
                data_name="instance_genseg",
                postprocess_task_name="instance_genseg",
                evaluator_data_name="coco_instance_genseg",
            ),
        ]

    val_cfgs = []
    evaluator_names = []
    for raw_cfg in raw_val_datasets:
        if not isinstance(raw_cfg, dict):
            raise TypeError(f"`val_datasets` entries must be mappings. Got: {type(raw_cfg)!r}")
        data_name = str(raw_cfg.get("data_name", "coco_panoptic_genseg"))
        postprocess_task_name = _infer_postprocess_task_name(data_name, raw_cfg.get("postprocess_task_name"))
        evaluator_names.append(str(raw_cfg.get("evaluator_data_name", data_name)))
        val_cfgs.append(
            _build_dataset_cfg(
                data_path=_resolve_runtime_path(data_root, raw_cfg["data_relpath"]),
                image_folder=_resolve_runtime_path(data_root, raw_cfg["image_folder_relpath"]),
                panseg_map_folder=_resolve_runtime_path(
                    data_root,
                    raw_cfg.get("panseg_map_relpath"),
                ),
                semseg_map_folder=_resolve_runtime_path(
                    data_root,
                    raw_cfg.get("semseg_map_relpath"),
                ),
                extra_processor=eval_processor,
                task_name=str(raw_cfg.get("task_name", "genseg")),
                data_name=data_name,
                data_mode="eval",
                postprocess_task_name=postprocess_task_name,
                pad_image_to_square=profile["pad_image_to_square"],
            )
        )
    return val_cfgs, evaluator_names


_config_dir = Path(__file__).resolve().parent
_profile_yaml_path = _resolve_profile_yaml(_config_dir)
_profile_cfg = Config.fromfile(str(_profile_yaml_path))
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
code_dir = _ensure_trailing_slash(__import__("os").environ.get("CODE_DIR", "./xsam/"))
data_dir = _ensure_trailing_slash(__import__("os").environ.get("DATA_DIR", "./data/"))
init_dir = _ensure_trailing_slash(__import__("os").environ.get("INIT_DIR", "./inits/"))
work_dir = _ensure_trailing_slash(__import__("os").environ.get("WORK_DIR", "./runs/"))
root_dir = _ensure_trailing_slash(__import__("os").environ.get("ROOT_DIR", "./"))

profile_name = profile["profile_name"]
profile_yaml_path = str(_profile_yaml_path)

seg_encoder_name_or_path = _resolve_runtime_path(init_dir, profile["seg_encoder_name"], root_prefix=root_dir)
seg_decoder_name_or_path = _resolve_runtime_path(init_dir, profile["seg_decoder_name"], root_prefix=root_dir)
sam3_encoder_trunk = _resolve_runtime_path(
    init_dir,
    profile["sam3_encoder_trunk"],
    root_prefix=root_dir,
    keep_bare_name=True,
)
sam3_simple_fpn = _resolve_runtime_path(
    init_dir,
    profile["sam3_simple_fpn"],
    root_prefix=root_dir,
    keep_bare_name=True,
)
s1_pretrained_pth = _resolve_runtime_path(init_dir, profile["s1_pretrained_pth"], root_prefix=root_dir)

data_root = _ensure_trailing_slash(_resolve_runtime_path(data_dir, profile["data_root"], root_prefix=root_dir))
data_path = _resolve_runtime_path(data_root, profile["train_data_relpath"])
image_folder = _resolve_runtime_path(data_root, profile["train_image_folder_relpath"])
panseg_map_folder = _resolve_runtime_path(data_root, profile["train_panseg_map_relpath"])
semseg_map_folder = _resolve_runtime_path(data_root, profile["train_semseg_map_relpath"])
image_size = int(profile["image_size"])

batch_size = profile["batch_size"]
accumulative_counts = profile["accumulative_counts"]
dataloader_num_workers = profile["dataloader_num_workers"]
max_iters = profile["max_iters"]
max_epochs = profile["max_epochs"]
optim_type = _resolve_optimizer(profile["optim_type"])
lr = profile["lr"]
betas = profile["betas"]
weight_decay = profile["weight_decay"]
max_norm = profile["max_norm"]
warmup_ratio = profile["warmup_ratio"]
milestones = profile["milestones"]

save_steps = profile["save_steps"]
save_total_limit = profile["save_total_limit"]
logging_interval = profile["logging_interval"]

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
    freeze_segmentor_encoder=segmentor_policy["freeze_segmentor_encoder"],
    freeze_segmentor_trunk=segmentor_policy["freeze_segmentor_trunk"],
    freeze_segmentor_fpn=segmentor_policy["freeze_segmentor_fpn"],
    use_activation_checkpointing=profile["use_activation_checkpointing"],
    segmentor_lora=segmentor_lora_cfg,
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
            encoder_filename=sam3_encoder_trunk,
            fpn_filename=sam3_simple_fpn,
            map_location="cpu",
            strict=False,
            torch_dtype=torch.bfloat16,
            config=dict(
                vision_config=dict(
                    img_size=image_size,
                    neck_hidden_size=profile["sam3_neck_hidden_size"],
                    neck_scale_factors=profile["sam3_neck_scale_factors"],
                    add_sam2_neck=profile["sam3_add_sam2_neck"],
                    scalp=profile["sam3_scalp"],
                )
            ),
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[256, 256, 256],
                feature_strides=[4, 8, 16],
                num_feature_levels=3,
                common_stride=4,
                image_size=image_size,
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
train_extra_image_processor = dict(
    type=Sam3ImageProcessor,
    ignore_index=0,
    size={
        "min_scale": profile["train_min_scale"],
        "max_scale": profile["train_max_scale"],
        "target_size": image_size,
    },
    do_crop=True,
    crop_size={"height": image_size, "width": image_size},
)

pannoptic_genseg_dataset = _build_dataset_cfg(
    data_path=data_path,
    image_folder=image_folder,
    panseg_map_folder=panseg_map_folder,
    semseg_map_folder=semseg_map_folder,
    extra_processor=train_extra_image_processor,
    task_name=profile["train_task_name"],
    data_name=profile["train_data_name"],
    pad_image_to_square=profile["pad_image_to_square"],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=profile["persistent_workers"],
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

val_datasets, _val_evaluator_names = _build_val_dataset_cfgs(
    profile=profile,
    data_root=data_root,
    data_dir=data_dir,
    root_dir=root_dir,
    eval_processor=extra_image_processor,
)
val_evaluators = [
    dict(
        type=GenSegEvaluator,
        data_name=data_name,
        distributed=True,
    )
    for data_name in _val_evaluator_names
]

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
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

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * (max_epochs if max_epochs is not None else 1),
        convert_to_iter_based=True,
    ),
    dict(
        type=MultiStepLR,
        begin=warmup_ratio * (max_epochs if max_epochs is not None else 1),
        end=(max_epochs if max_epochs is not None else 1),
        by_epoch=True,
        milestones=milestones,
        gamma=0.1,
        convert_to_iter_based=True,
    ),
]

if max_iters is not None:
    train_cfg = dict(type=TrainLoop, max_iters=max_iters)
else:
    train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
runner_type = "FlexibleRunner"
visualizer = None

custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=[
            "segmentor.encoder",
            "segmentor.pixel_decoder",
            "segmentor.decoder",
        ],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
    dict(type=PTCheckpointHook, clean_pth=False),
]

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

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=1024, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

find_unused_parameters = profile["find_unused_parameters"]

del _build_dataset_cfg
del _build_profile
del _build_val_dataset_cfgs
del _config_dir
del _ensure_trailing_slash
del _infer_postprocess_task_name
del _profile_cfg
del _profile_dict
del _profile_yaml_path
del _resolve_optimizer
del _resolve_profile_yaml
del _resolve_runtime_path
del _val_evaluator_names

"""
bash ./run.sh --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam3/sam3_s1.py \
  --yaml xsam/xsam/configs/xsam/s1_seg_finetune/sam3/profiles/bdd100k_sam3_small.yaml
"""
