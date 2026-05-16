from os import getenv
from pathlib import Path

import torch
from peft import LoraConfig
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

from xsam.dataset import ImgConvDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import imgconv_map_fn, template_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.sam import SamModel
from xsam.utils.template import PROMPT_TEMPLATE

_DEFAULT_PROFILE_YAML = "profiles/phi3_mini.yaml"
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
_profile = _profile_cfg._cfg_dict.to_dict()
if not isinstance(_profile, dict):
    raise TypeError(f"Profile YAML must be a mapping. Got: {type(_profile)!r}")

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./data/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./runs/")

# Profile
profile_name = str(_profile["profile_name"])
profile_yaml_path = str(_profile_yaml_path)
segmentor_policy = __import__(
    "xsam.configs.xsam.profile_utils",
    fromlist=["build_segmentor_train_policy"],
).build_segmentor_train_policy(
    _profile,
    default_trunk_mode="freeze",
    default_fpn_mode="freeze",
)
segmentor_lora_cfg = None
if segmentor_policy["segmentor_lora_kwargs"] is not None:
    segmentor_lora_cfg = dict(type=LoraConfig, **segmentor_policy["segmentor_lora_kwargs"])

# Model
llm_name_or_path = init_dir + str(_profile["llm_name"])
visual_encoder_name_or_path = init_dir + "siglip2-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"

# Specify the pretrained pth
s1_pretrained_pth = work_dir + str(_profile["s1_pretrained_relpath"])

# Data
data_root = data_dir + "imgconv_data/"
data_path = data_root + "llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
image_folder = data_root + "llava/LLaVA-Pretrain/images"
prompt_template = getattr(PROMPT_TEMPLATE, str(_profile["prompt_template"]))
max_length = int(_profile["max_length"])

# Scheduler & Optimizer
batch_size = int(_profile.get("batch_size", 4))  # per_device
accumulative_counts = int(_profile.get("accumulative_counts", 4))
dataloader_num_workers = int(_profile.get("dataloader_num_workers", 8))
max_epochs = int(_profile.get("max_epochs", 1))
optim_type = AdamW
lr = float(_profile.get("lr", 1e-3))
betas = tuple(_profile.get("betas", [0.9, 0.999]))
weight_decay = float(_profile.get("weight_decay", 0.0))
max_norm = float(_profile.get("max_norm", 1.0))  # grad clip
warmup_ratio = float(_profile.get("warmup_ratio", 0.03))

# Save
save_steps = int(_profile.get("save_steps", 2000))
save_total_limit = int(_profile.get("save_total_limit", 2))  # Maximum checkpoints to keep (-1 means unlimited)
# Logging
logging_interval = int(_profile.get("logging_interval", 10))

# Evaluate the generation performance during the training
evaluation_freq = int(_profile.get("evaluation_freq", 2000))
SYSTEM = ""
evaluation_images = code_dir + "xsam/configs/xsam/images/imgconv.jpg"
evaluation_inputs = ["Can you describe this image in detail? Please elaborate in your response."]

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
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=segmentor_policy["freeze_segmentor_encoder"],
    freeze_segmentor_trunk=segmentor_policy["freeze_segmentor_trunk"],
    freeze_segmentor_fpn=segmentor_policy["freeze_segmentor_fpn"],
    segmentor_lora=segmentor_lora_cfg,
    use_dual_encoder=True,
    s1_pretrained_pth=s1_pretrained_pth,
    tokenizer=tokenizer,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,  # from transformers
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        torch_dtype=torch.bfloat16,
        drop_decoder=True,
    ),
)

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
    extra_image_processor=extra_image_processor,
    dataset_map_fn=imgconv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=False,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=imgconv_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=xsam_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
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
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "segmentor.encoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
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

# set visualizer
visualizer = None

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

del _config_dir
del _profile_cfg
del _profile_yaml
del _profile_yaml_path

"""
bash run.sh --modes train \
  --config xsam/xsam/configs/xsam/s2_align_pretrain/sam/xsam_sam_align_pretrain.py \
  --yaml xsam/xsam/configs/xsam/s2_align_pretrain/sam/profiles/phi3_mini.yaml
"""
