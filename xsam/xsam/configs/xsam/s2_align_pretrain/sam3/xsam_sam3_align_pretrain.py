import os

import torch
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xsam.dataset import ImgConvDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import imgconv_map_fn, template_map_fn_factory
from xsam.dataset.processors import Sam3ImageProcessor
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.model import XSam3Model
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.sam3 import Sam3Model
from xsam.utils.template import PROMPT_TEMPLATE

_DEFAULT_PROFILE_YAML = "profiles/sma3_s2.yaml"
_config_dir = __import__("pathlib").Path(__file__).resolve().parent
_profile_yaml = os.environ.get(
    "XSAM_CONFIG_PROFILE_YAML",
    os.environ.get("XSAM_SEG_PROFILE_YAML", _DEFAULT_PROFILE_YAML),
)
_profile_yaml_path = __import__("pathlib").Path(_profile_yaml)
if not _profile_yaml_path.is_absolute():
    _profile_yaml_path_from_cwd = _profile_yaml_path.resolve()
    if _profile_yaml_path_from_cwd.exists():
        _profile_yaml_path = _profile_yaml_path_from_cwd
    else:
        _profile_yaml_path = (_config_dir / _profile_yaml_path).resolve()
if not _profile_yaml_path.exists():
    raise FileNotFoundError(f"Cannot find profile yaml: {_profile_yaml} -> {_profile_yaml_path}")
_profile_cfg = Config.fromfile(str(_profile_yaml_path))
_profile = _profile_cfg._cfg_dict.to_dict()
if not isinstance(_profile, dict):
    raise TypeError(f"Profile YAML must be a mapping. Got: {type(_profile)!r}")

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = os.environ.get("CODE_DIR", "./xsam/")
data_dir = os.environ.get("DATA_DIR", "./data/")
init_dir = os.environ.get("INIT_DIR", "./inits/")
work_dir = os.environ.get("WORK_DIR", "./runs/")
root_dir = os.environ.get("ROOT_DIR", "./")

# Profile
profile_name = str(_profile.get("profile_name", "sam3_align_phase1"))
profile_yaml_path = str(_profile_yaml_path)

# Model
llm_name_or_path = init_dir + str(_profile.get("llm_name", "Phi-3-mini-4k-instruct"))
seg_encoder_name_or_path = init_dir + str(_profile.get("seg_encoder_name", "sam3"))
sam3_encoder_trunk = str(_profile.get("sam3_encoder_trunk", "sam3_encoder.bin"))
sam3_simple_fpn = str(_profile.get("sam3_simple_fpn", "sam3_fpn.bin"))

def _optional_join(prefix: str, rel_value, root_prefix: str | None = None):
    """Join base directory and optional relative path.

    Args:
        prefix: Base directory for generic relative paths.
        rel_value: Relative or absolute path value from profile.
        root_prefix: Optional project root for repo-relative paths like `runs/...`.
    Returns:
        Resolved path or ``None``.
    """
    if rel_value in (None, "", "null"):
        return None
    raw = str(rel_value)
    raw_path = __import__("pathlib").Path(raw)
    if raw_path.is_absolute():
        return str(raw_path)
    if root_prefix is not None:
        root_path = __import__("pathlib").Path(root_prefix)
        if raw.startswith("./") or raw.startswith("../"):
            return str((root_path / raw).resolve())
        if raw.startswith(("runs/", "data/", "inits/", "xsam/", "external/", "scripts/")):
            return str((root_path / raw).resolve())
    return str((__import__("pathlib").Path(prefix) / raw).resolve())


s1_pretrained_relpath = _profile.get("s1_pretrained_relpath")
s1_pretrained_pth = None
if isinstance(s1_pretrained_relpath, dict):
    s1_encoder_trunk_relpath = s1_pretrained_relpath.get("encoder_trunk_relpath")
    s1_fpn_relpath = s1_pretrained_relpath.get("fpn_relpath")
    if (s1_encoder_trunk_relpath is None) != (s1_fpn_relpath is None):
        raise ValueError(
            "`s1_pretrained_relpath` must provide both `encoder_trunk_relpath` and `fpn_relpath` for SAM3 split "
            "weights."
        )
    if s1_encoder_trunk_relpath is not None:
        sam3_encoder_trunk = _optional_join(work_dir, s1_encoder_trunk_relpath, root_prefix=root_dir)
        sam3_simple_fpn = _optional_join(work_dir, s1_fpn_relpath, root_prefix=root_dir)
else:
    s1_pretrained_pth = _optional_join(work_dir, s1_pretrained_relpath, root_prefix=root_dir)
s2_pretrained_pth = _optional_join(init_dir, _profile.get("s2_pretrained_relpath"), root_prefix=root_dir)

# Data
data_root = data_dir + "imgconv_data/"
data_path = data_root + str(_profile.get("data_relpath", "llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"))
image_folder = data_root + str(_profile.get("image_folder_relpath", "llava/LLaVA-Pretrain/558k_images"))
prompt_template = getattr(PROMPT_TEMPLATE, str(_profile.get("prompt_template", "phi3_chat")))
max_length = int(_profile.get("max_length", 2319))
image_size = int(_profile.get("image_size", 1008))

# Scheduler & Optimizer
batch_size = int(_profile.get("batch_size", 2))
accumulative_counts = int(_profile.get("accumulative_counts", 64))
dataloader_num_workers = int(_profile.get("dataloader_num_workers", 8))
max_epochs = float(_profile.get("max_epochs", 1.0))
optim_type = AdamW
lr = float(_profile.get("lr", 5e-4))
betas = tuple(_profile.get("betas", [0.9, 0.999]))
weight_decay = float(_profile.get("weight_decay", 0.0))
max_norm = float(_profile.get("max_norm", 1.0))
warmup_ratio = float(_profile.get("warmup_ratio", 0.03))

# Runtime / training policy
freeze_llm = bool(_profile.get("freeze_llm", True))
segmentor_policy = __import__(
    "xsam.configs.xsam.profile_utils",
    fromlist=["build_segmentor_train_policy"],
).build_segmentor_train_policy(
    _profile,
    default_trunk_mode="freeze",
    default_fpn_mode="freeze",
)
freeze_segmentor_trunk = segmentor_policy["freeze_segmentor_trunk"]
freeze_segmentor_fpn = segmentor_policy["freeze_segmentor_fpn"]
freeze_segmentor_encoder = segmentor_policy["freeze_segmentor_encoder"]
use_activation_checkpointing = bool(_profile.get("use_activation_checkpointing", True))

# LoRA
enable_lora = bool(_profile.get("enable_lora", False))
llm_lora_rank = int(_profile.get("llm_lora_rank", 16))
llm_lora_alpha = int(_profile.get("llm_lora_alpha", 32))
llm_lora_dropout = float(_profile.get("llm_lora_dropout", 0.05))
llm_lora_target_modules = _profile.get("llm_lora_target_modules", None)
if llm_lora_target_modules is not None:
    llm_lora_target_modules = [str(module) for module in llm_lora_target_modules]
    if len(llm_lora_target_modules) == 0:
        llm_lora_target_modules = None

# Single-backbone projector routing
lang_bast_layers = [int(layer_id) for layer_id in _profile.get("lang_bast_layers", [-1])]
seg_bast_layers = [int(layer_id) for layer_id in _profile.get("seg_bast_layers", [-2])]
_layer_id_env = os.environ.get("XSAM_LAYER_ID")
if _layer_id_env not in (None, ""):
    _single_layer_id = int(_layer_id_env)
    lang_bast_layers = [_single_layer_id]
    if os.environ.get("XSAM_SEG_BAST_LAYERS") in (None, ""):
        seg_bast_layers = [_single_layer_id]

_lang_layers_env = os.environ.get("XSAM_LANG_BAST_LAYERS")
if _lang_layers_env not in (None, ""):
    lang_bast_layers = [int(token.strip()) for token in _lang_layers_env.split(",") if token.strip()]

_seg_layers_env = os.environ.get("XSAM_SEG_BAST_LAYERS")
if _seg_layers_env not in (None, ""):
    seg_bast_layers = [int(token.strip()) for token in _seg_layers_env.split(",") if token.strip()]

lang_downsample_ratio = float(_profile.get("lang_downsample_ratio", 1.0))
seg_downsample_ratio = float(_profile.get("seg_downsample_ratio", 1.0))
projector_depth = int(_profile.get("projector_depth", 2))

# 4-bit LLM loading
enable_4bit_llm = bool(_profile.get("enable_4bit_llm", False))
llm_quantization_config = None
if enable_4bit_llm:
    llm_quantization_config = dict(
        type=BitsAndBytesConfig,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# Save
save_steps = int(_profile.get("save_steps", 2000))
save_total_limit = int(_profile.get("save_total_limit", 2))

# Logging
logging_interval = int(_profile.get("logging_interval", 10))

# Evaluate sample generation
evaluation_freq = int(_profile.get("evaluation_freq", 2000))
SYSTEM = ""
evaluation_images = _optional_join(
    data_dir,
    _profile.get(
        "evaluation_image",
        data_dir + "imgconv_data/llava/LLaVA-Pretrain/558k_images/00001/000015879.jpg",
    ),
    root_prefix=root_dir,
)
evaluation_inputs = [
    "Can you describe this image in detail? Please elaborate in your response.",
]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
ignore_label = 255
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=Sam3ImageProcessor,
    ignore_index=0,
    size={"height": image_size, "width": image_size},
    pad_size={"height": image_size, "width": image_size},
    mask_size={"height": image_size, "width": image_size},
    mask_pad_size={"height": image_size, "width": image_size},
)

llm_cfg = dict(
    type=AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=False,
    torch_dtype=torch.bfloat16,
    attn_implementation=str(_profile.get("llm_attn_implementation", "flash_attention_2")),
)
if llm_quantization_config is not None:
    llm_cfg["quantization_config"] = llm_quantization_config
    llm_cfg["low_cpu_mem_usage"] = True

llm_lora_cfg = None
if enable_lora:
    llm_lora_cfg = dict(
        type=LoraConfig,
        task_type="CAUSAL_LM",
        r=llm_lora_rank,
        lora_alpha=llm_lora_alpha,
        lora_dropout=llm_lora_dropout,
        bias="none",
        target_modules=llm_lora_target_modules,
    )

segmentor_lora_cfg = None
if segmentor_policy["segmentor_lora_kwargs"] is not None:
    segmentor_lora_cfg = dict(type=LoraConfig, **segmentor_policy["segmentor_lora_kwargs"])

model = dict(
    type=XSam3Model,
    freeze_llm=freeze_llm,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=freeze_segmentor_encoder,
    freeze_segmentor_trunk=freeze_segmentor_trunk,
    freeze_segmentor_fpn=freeze_segmentor_fpn,
    use_activation_checkpointing=use_activation_checkpointing,
    llm_lora=llm_lora_cfg,
    segmentor_lora=segmentor_lora_cfg,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    lang_bast_layers=lang_bast_layers,
    seg_bast_layers=seg_bast_layers,
    lang_downsample_ratio=lang_downsample_ratio,
    seg_downsample_ratio=seg_downsample_ratio,
    projector_depth=projector_depth,
    llm=llm_cfg,
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=Sam3Model.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            encoder_filename=sam3_encoder_trunk,
            fpn_filename=sam3_simple_fpn,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            strict=False,
            map_location="cpu",
        ),
        torch_dtype=torch.bfloat16,
        drop_decoder=True,
    ),
)
if llm_lora_cfg is None:
    model.pop("llm_lora", None)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=data_path,
    tokenizer=tokenizer,
    image_folder=image_folder,
    task_name="imgconv",
    data_name="llava_imgconv",
    image_processor=image_processor,
    dataset_map_fn=imgconv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=False,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=bool(_profile.get("persistent_workers", True)) if dataloader_num_workers > 0 else False,
    pin_memory=bool(_profile.get("pin_memory", True)),
    dataset=imgconv_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=xsam_collate_fn),
)
if dataloader_num_workers > 0 and "prefetch_factor" in _profile:
    train_dataloader["prefetch_factor"] = int(_profile["prefetch_factor"])

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

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
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_projector", "seg_projector", "segmentor.encoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=PTCheckpointHook, clean_pth=False),
]

if evaluation_freq > 0:
    custom_hooks.insert(
        2,
        dict(
            type=EvaluateChatHook,
            tokenizer=tokenizer,
            image_processor=image_processor,
            every_n_iters=evaluation_freq,
            evaluation_inputs=evaluation_inputs,
            evaluation_images=evaluation_images,
            system=SYSTEM,
            prompt_template=prompt_template,
        ),
    )

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

visualizer = None
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)

log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

del os
del torch
del _config_dir
del _profile_cfg
del _profile_yaml
del _profile_yaml_path
del _optional_join

"""
bash run.sh --modes train \
  --config xsam/xsam/configs/xsam/s2_align_pretrain/sam3/xsam_sam3_align_pretrain.py \
  --yaml xsam/xsam/configs/xsam/s2_align_pretrain/sam3/profiles/phase1.yaml
"""
