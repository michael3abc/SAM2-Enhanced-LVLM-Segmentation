import os
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xsam.dataset import ConcatDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.process_fns import (
    gcgseg_postprocess_fn,
    genseg_postprocess_fn,
    intseg_postprocess_fn,
    reaseg_postprocess_fn,
    refseg_postprocess_fn,
    vgdseg_postprocess_fn,
)
from xsam.dataset.processors import Sam3ImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.model import XSam3Model
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam3 import Sam3Model
from xsam.utils.visualize import Visualizer


def _ensure_trailing_slash(path_value: str) -> str:
    """Append a trailing slash when needed.

    Args:
        path_value: Directory path string.
    Returns:
        Normalized directory path ending with ``/``.
    """
    return path_value if path_value.endswith("/") else path_value + "/"


def _optional_join(prefix: str, rel_value, root_prefix: str | None = None, keep_bare_name: bool = False):
    """Resolve optional relative paths from profile values.

    Args:
        prefix: Base directory for generic relative paths.
        rel_value: Relative or absolute path value from profile.
        root_prefix: Optional project root for repo-relative values.
        keep_bare_name: Whether to keep bare filenames unchanged.
    Returns:
        Resolved path string or ``None``.
    """
    if rel_value in (None, "", "null"):
        return None

    raw = str(rel_value)
    raw_path = __import__("pathlib").Path(raw)
    if raw_path.is_absolute():
        return str(raw_path)

    if keep_bare_name and "/" not in raw and "\\" not in raw:
        return raw

    if root_prefix is not None:
        root_path = __import__("pathlib").Path(root_prefix)
        if raw.startswith("./") or raw.startswith("../"):
            return str((root_path / raw).resolve())
        if raw.startswith(("runs/", "data/", "inits/", "xsam/", "external/", "scripts/")):
            return str((root_path / raw).resolve())

    return str((__import__("pathlib").Path(prefix) / raw).resolve())


def _replace_dict_in_place(target: dict, new_values: dict) -> None:
    """Replace a config dictionary in place.

    Args:
        target: Mutable config dictionary to overwrite.
        new_values: Replacement key-value pairs.
    Returns:
        None.
    """
    target.clear()
    target.update(new_values)


def _sync_dataset_cfg(node) -> None:
    """Recursively update dataset configs to SAM3 processors/tokenizer.

    Args:
        node: Dataset config node, list, or nested dict.
    Returns:
        None.
    """
    if isinstance(node, list):
        for item in node:
            _sync_dataset_cfg(item)
        return

    if not isinstance(node, dict):
        return

    if "tokenizer" in node:
        node["tokenizer"] = tokenizer
    if "image_processor" in node:
        node["image_processor"] = image_processor
    if "extra_image_processor" in node:
        node["extra_image_processor"] = extra_image_processor
    if "cond_type" in node:
        node["cond_type"] = cond_type
    if "special_tokens" in node:
        node["special_tokens"] = special_tokens
    if "max_length" in node:
        node["max_length"] = max_length
    if "ignore_label" in node:
        node["ignore_label"] = ignore_label
    if "template_map_fn" in node and isinstance(node["template_map_fn"], dict):
        node["template_map_fn"]["template"] = prompt_template

    for value in node.values():
        if isinstance(value, (dict, list)):
            _sync_dataset_cfg(value)


AdamW = __import__("torch.optim", fromlist=["AdamW"]).AdamW
AutoModelForCausalLM = __import__("transformers", fromlist=["AutoModelForCausalLM"]).AutoModelForCausalLM
AutoTokenizer = __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer
BitsAndBytesConfig = __import__("transformers", fromlist=["BitsAndBytesConfig"]).BitsAndBytesConfig
LoraConfig = __import__("peft", fromlist=["LoraConfig"]).LoraConfig
ConcatDataset = __import__("xsam.dataset", fromlist=["ConcatDataset"]).ConcatDataset
xsam_collate_fn = __import__("xsam.dataset.collate_fns", fromlist=["xsam_collate_fn"]).xsam_collate_fn
gcgseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["gcgseg_postprocess_fn"]).gcgseg_postprocess_fn
genseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["genseg_postprocess_fn"]).genseg_postprocess_fn
intseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["intseg_postprocess_fn"]).intseg_postprocess_fn
reaseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["reaseg_postprocess_fn"]).reaseg_postprocess_fn
refseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["refseg_postprocess_fn"]).refseg_postprocess_fn
vgdseg_postprocess_fn = __import__("xsam.dataset.process_fns", fromlist=["vgdseg_postprocess_fn"]).vgdseg_postprocess_fn
Sam3ImageProcessor = __import__("xsam.dataset.processors", fromlist=["Sam3ImageProcessor"]).Sam3ImageProcessor
SourceGroupedSampler = __import__("xsam.dataset.samplers", fromlist=["SourceGroupedSampler"]).SourceGroupedSampler
DatasetInfoHook = __import__("xsam.engine.hooks", fromlist=["DatasetInfoHook"]).DatasetInfoHook
EvaluateChatHook = __import__("xsam.engine.hooks", fromlist=["EvaluateChatHook"]).EvaluateChatHook
ModelInfoHook = __import__("xsam.engine.hooks", fromlist=["ModelInfoHook"]).ModelInfoHook
PTCheckpointHook = __import__("xsam.engine.hooks", fromlist=["PTCheckpointHook"]).PTCheckpointHook
TrainLoop = __import__("xsam.engine.runner", fromlist=["TrainLoop"]).TrainLoop
XSam3Model = __import__("xsam.model", fromlist=["XSam3Model"]).XSam3Model
XSegmentor = __import__("xsam.model.segmentors", fromlist=["XSegmentor"]).XSegmentor
Mask2FormerConfig = __import__(
    "xsam.model.segmentors.mask2former",
    fromlist=["Mask2FormerConfig"],
).Mask2FormerConfig
Mask2FormerModel = __import__(
    "xsam.model.segmentors.mask2former",
    fromlist=["Mask2FormerModel"],
).Mask2FormerModel
Sam3Model = __import__("xsam.model.segmentors.sam3", fromlist=["Sam3Model"]).Sam3Model
Visualizer = __import__("xsam.utils.visualize", fromlist=["Visualizer"]).Visualizer
CheckpointHook = __import__("mmengine.hooks", fromlist=["CheckpointHook"]).CheckpointHook
DistSamplerSeedHook = __import__("mmengine.hooks", fromlist=["DistSamplerSeedHook"]).DistSamplerSeedHook
IterTimerHook = __import__("mmengine.hooks", fromlist=["IterTimerHook"]).IterTimerHook
LoggerHook = __import__("mmengine.hooks", fromlist=["LoggerHook"]).LoggerHook
ParamSchedulerHook = __import__("mmengine.hooks", fromlist=["ParamSchedulerHook"]).ParamSchedulerHook
AmpOptimWrapper = __import__("mmengine.optim", fromlist=["AmpOptimWrapper"]).AmpOptimWrapper
CosineAnnealingLR = __import__("mmengine.optim", fromlist=["CosineAnnealingLR"]).CosineAnnealingLR
LinearLR = __import__("mmengine.optim", fromlist=["LinearLR"]).LinearLR


_base_config_path = __import__("pathlib").Path(__file__).resolve().parent.parent / "sam2" / "xsam_sam2_mixed_finetune.py"
_base_cfg = Config.fromfile(str(_base_config_path))
for _name, _value in _base_cfg._cfg_dict.to_dict().items():
    globals()[_name] = _value
del _base_cfg

_DEFAULT_PROFILE_YAML = "profiles/phi3_mini.yaml"
_config_dir = __import__("pathlib").Path(__file__).resolve().parent
_profile_yaml = __import__("os").environ.get(
    "XSAM_CONFIG_PROFILE_YAML",
    __import__("os").environ.get("XSAM_SEG_PROFILE_YAML", _DEFAULT_PROFILE_YAML),
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
del _profile_cfg

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
code_dir = _ensure_trailing_slash(__import__("os").environ.get("CODE_DIR", "./xsam/"))
data_dir = _ensure_trailing_slash(__import__("os").environ.get("DATA_DIR", "./data/"))
init_dir = _ensure_trailing_slash(__import__("os").environ.get("INIT_DIR", "./inits/"))
work_dir = _ensure_trailing_slash(__import__("os").environ.get("WORK_DIR", "./runs/"))
root_dir = _ensure_trailing_slash(__import__("os").environ.get("ROOT_DIR", "./"))

profile_name = str(_profile["profile_name"])
profile_yaml_path = str(_profile_yaml_path)
segmentor_policy = __import__(
    "xsam.configs.xsam.profile_utils",
    fromlist=["build_segmentor_train_policy"],
).build_segmentor_train_policy(
    _profile,
    default_trunk_mode="full_ft",
    default_fpn_mode="full_ft",
)
segmentor_lora_cfg = None
if segmentor_policy["segmentor_lora_kwargs"] is not None:
    segmentor_lora_cfg = dict(
        type=__import__("peft", fromlist=["LoraConfig"]).LoraConfig,
        **segmentor_policy["segmentor_lora_kwargs"],
    )

llm_name_or_path = _optional_join(init_dir, _profile["llm_name"], root_prefix=root_dir)
seg_encoder_name_or_path = _optional_join(init_dir, _profile.get("seg_encoder_name", "sam3"), root_prefix=root_dir)
seg_decoder_name_or_path = _optional_join(
    init_dir,
    _profile.get("seg_decoder_name", "mask2former-swin-large-coco-panoptic"),
    root_prefix=root_dir,
)
sam3_encoder_trunk = _optional_join(
    init_dir,
    _profile.get("sam3_encoder_trunk", "sam3_encoder.bin"),
    root_prefix=root_dir,
    keep_bare_name=True,
)
sam3_simple_fpn = _optional_join(
    init_dir,
    _profile.get("sam3_simple_fpn", "sam3_fpn.bin"),
    root_prefix=root_dir,
    keep_bare_name=True,
)

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
        sam3_encoder_trunk = _optional_join(
            work_dir,
            s1_encoder_trunk_relpath,
            root_prefix=root_dir,
            keep_bare_name=True,
        )
        sam3_simple_fpn = _optional_join(
            work_dir,
            s1_fpn_relpath,
            root_prefix=root_dir,
            keep_bare_name=True,
        )
else:
    s1_pretrained_pth = _optional_join(work_dir, s1_pretrained_relpath, root_prefix=root_dir)

s2_pretrained_pth = _optional_join(work_dir, _profile.get("s2_pretrained_relpath"), root_prefix=root_dir)

prompt_template = getattr(
    __import__("xtuner.utils", fromlist=["PROMPT_TEMPLATE"]).PROMPT_TEMPLATE,
    str(_profile.get("prompt_template", "phi3_chat")),
)
max_length = int(_profile.get("max_length", 2319))
image_size = int(_profile.get("image_size", 1008))

batch_size = int(_profile.get("batch_size", 4))
accumulative_counts = int(_profile.get("accumulative_counts", 1))
dataloader_num_workers = int(_profile.get("dataloader_num_workers", 4))
max_epochs = float(_profile.get("max_epochs", 1))
optim_type = __import__("torch.optim", fromlist=["AdamW"]).AdamW
lr = float(_profile.get("lr", 4e-5))
betas = tuple(_profile.get("betas", [0.9, 0.999]))
weight_decay = float(_profile.get("weight_decay", 0.05))
max_norm = float(_profile.get("max_norm", 1.0))
warmup_ratio = float(_profile.get("warmup_ratio", 0.03))

enable_lora = bool(_profile.get("enable_lora", True))
llm_lora_rank = int(_profile.get("llm_lora_rank", 64))
llm_lora_alpha = int(_profile.get("llm_lora_alpha", 128))
llm_lora_dropout = float(_profile.get("llm_lora_dropout", 0.05))
llm_lora_target_modules = _profile.get("llm_lora_target_modules")
if llm_lora_target_modules is not None:
    llm_lora_target_modules = [str(module) for module in llm_lora_target_modules]
    if len(llm_lora_target_modules) == 0:
        llm_lora_target_modules = None

enable_4bit_llm = bool(_profile.get("enable_4bit_llm", True))
llm_quantization_config = None
if enable_4bit_llm:
    llm_quantization_config = dict(
        type=__import__("transformers", fromlist=["BitsAndBytesConfig"]).BitsAndBytesConfig,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=__import__("torch").bfloat16,
    )

save_steps = int(_profile.get("save_steps", 2000))
save_total_limit = int(_profile.get("save_total_limit", 2))
logging_interval = int(_profile.get("logging_interval", 10))
evaluation_freq = int(_profile.get("evaluation_freq", 2000))
freeze_llm = bool(_profile.get("freeze_llm", True))
connector_type = _profile.get("connector_type")
llm_attn_implementation = str(_profile.get("llm_attn_implementation", "eager"))
llava_split_mode = str(_profile.get("llava_split_mode", "split"))
persistent_workers = bool(_profile.get("persistent_workers", True))
pin_memory = bool(_profile.get("pin_memory", True))
use_train_ratio = bool(_profile.get("use_train_ratio", True))
use_activation_checkpointing = bool(_profile.get("use_activation_checkpointing", True))

lang_bast_layers = [int(layer_id) for layer_id in _profile.get("lang_bast_layers", [-12])]
seg_bast_layers = [int(layer_id) for layer_id in _profile.get("seg_bast_layers", [-12])]
_layer_id_env = __import__("os").environ.get("XSAM_LAYER_ID")
if _layer_id_env not in (None, ""):
    _single_layer_id = int(_layer_id_env)
    lang_bast_layers = [_single_layer_id]
    if __import__("os").environ.get("XSAM_SEG_BAST_LAYERS") in (None, ""):
        seg_bast_layers = [_single_layer_id]

_lang_layers_env = __import__("os").environ.get("XSAM_LANG_BAST_LAYERS")
if _lang_layers_env not in (None, ""):
    lang_bast_layers = [int(token.strip()) for token in _lang_layers_env.split(",") if token.strip()]

_seg_layers_env = __import__("os").environ.get("XSAM_SEG_BAST_LAYERS")
if _seg_layers_env not in (None, ""):
    seg_bast_layers = [int(token.strip()) for token in _seg_layers_env.split(",") if token.strip()]

lang_downsample_ratio = float(_profile.get("lang_downsample_ratio", 0.5))
seg_downsample_ratio = float(_profile.get("seg_downsample_ratio", 0.5))
projector_depth = int(_profile.get("projector_depth", 2))

visual_encoder_name_or_path = None
visual_encoder_pretrained_pth = None
projector_pretrained_pth = None
llm_projector_pretrained_pth = None

#######################################################################
#            PART 2  Model & Tokenizer &  Processor                   #
#######################################################################
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"
ignore_label = 255
tokenizer = dict(
    type=__import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=__import__("xsam.dataset.processors", fromlist=["Sam3ImageProcessor"]).Sam3ImageProcessor,
    ignore_index=0,
    size={"height": image_size, "width": image_size},
    pad_size={"height": image_size, "width": image_size},
    mask_size={"height": image_size, "width": image_size},
    mask_pad_size={"height": image_size, "width": image_size},
)
extra_image_processor = dict(
    type=__import__("xsam.dataset.processors", fromlist=["Sam3ImageProcessor"]).Sam3ImageProcessor,
    ignore_index=0,
    size={"height": image_size, "width": image_size},
    pad_size={"height": image_size, "width": image_size},
    mask_size={"height": image_size, "width": image_size},
    mask_pad_size={"height": image_size, "width": image_size},
)

llm_cfg = dict(
    type=__import__("transformers", fromlist=["AutoModelForCausalLM"]).AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=False,
    torch_dtype=__import__("torch").bfloat16,
    attn_implementation=llm_attn_implementation,
)
if llm_quantization_config is not None:
    llm_cfg["quantization_config"] = llm_quantization_config
    llm_cfg["low_cpu_mem_usage"] = True

llm_lora_cfg = None
if enable_lora:
    llm_lora_cfg = dict(
        type=__import__("peft", fromlist=["LoraConfig"]).LoraConfig,
        task_type="CAUSAL_LM",
        r=llm_lora_rank,
        lora_alpha=llm_lora_alpha,
        lora_dropout=llm_lora_dropout,
        bias="none",
        target_modules=llm_lora_target_modules,
    )

model = dict(
    type=__import__("xsam.model", fromlist=["XSam3Model"]).XSam3Model,
    freeze_llm=freeze_llm,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=segmentor_policy["freeze_segmentor_encoder"],
    freeze_segmentor_trunk=segmentor_policy["freeze_segmentor_trunk"],
    freeze_segmentor_fpn=segmentor_policy["freeze_segmentor_fpn"],
    use_activation_checkpointing=use_activation_checkpointing,
    llm_lora=llm_lora_cfg,
    segmentor_lora=segmentor_lora_cfg,
    use_vision_sampler=True,
    connector_type=connector_type,
    cond_type=cond_type,
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=genseg_postprocess_fn,
    lang_bast_layers=lang_bast_layers,
    seg_bast_layers=seg_bast_layers,
    lang_downsample_ratio=lang_downsample_ratio,
    seg_downsample_ratio=seg_downsample_ratio,
    projector_depth=projector_depth,
    llm=llm_cfg,
    segmentor=dict(
        type=__import__("xsam.model.segmentors", fromlist=["XSegmentor"]).XSegmentor,
        encoder=dict(
            type=__import__("xsam.model.segmentors.sam3", fromlist=["Sam3Model"]).Sam3Model.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            encoder_filename=sam3_encoder_trunk,
            fpn_filename=sam3_simple_fpn,
            trust_remote_code=True,
            torch_dtype=__import__("torch").bfloat16,
            strict=False,
            map_location="cpu",
        ),
        decoder=dict(
            type=__import__(
                "xsam.model.segmentors.mask2former",
                fromlist=["Mask2FormerModel"],
            ).Mask2FormerModel.from_pretrained,
            pretrained_model_name_or_path=seg_decoder_name_or_path,
            config=dict(
                type=__import__(
                    "xsam.model.segmentors.mask2former",
                    fromlist=["Mask2FormerConfig"],
                ).Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[256, 256, 256],
                feature_strides=[4, 8, 16],
                common_stride=4,
                num_feature_levels=3,
                image_size=image_size,
                trust_remote_code=True,
            ),
            ignore_mismatched_sizes=True,
            torch_dtype=__import__("torch").bfloat16,
        ),
        torch_dtype=__import__("torch").bfloat16,
        reinit_decoder=False,
        open_cls=True,
    ),
)
if llm_lora_cfg is None:
    model.pop("llm_lora", None)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_ratio = dict(train_ratio)
if isinstance(_profile.get("train_ratio"), dict):
    train_ratio.update(_profile["train_ratio"])
if not use_train_ratio:
    for _ratio_key in train_ratio:
        train_ratio[_ratio_key] = 1.0

for _dataset_cfg, _ratio_key in [
    (llava_imgconv_coco_dataset, "llava_imgconv_coco"),
    (llava_imgconv_vg_dataset, "llava_imgconv_vg"),
    (llava_imgconv_gqa_dataset, "llava_imgconv_gqa"),
    (llava_imgconv_ocr_vqa_dataset, "llava_imgconv_ocr_vqa"),
    (llava_imgconv_textvqa_dataset, "llava_imgconv_textvqa"),
    (coco_genseg_dataset, "coco_panoptic_genseg"),
    (refcoco_refseg_dataset, "refcoco_refseg"),
    (refcocop_refseg_dataset, "refcoco_plus_refseg"),
    (refcocog_refseg_dataset, "refcocog_refseg"),
    (lisa_reaseg_dataset, "lisa_reaseg"),
    (grandf_gcgseg_dataset, "grandf_gcgseg"),
    (refcocog_gcgseg_dataset, "refcocog_gcgseg"),
    (psg_gcgseg_dataset, "psg_gcgseg"),
    (flickr_gcgseg_dataset, "flickr_gcgseg"),
    (coco_vgdseg_dataset, "coco_vgdseg"),
]:
    _dataset_cfg["train_ratio"] = train_ratio[_ratio_key]
llava_imgconv_dataset["train_ratio"] = train_ratio.get("llava_imgconv", 1.0)

for _dataset_cfg in [
    llava_imgconv_coco_dataset,
    llava_imgconv_vg_dataset,
    llava_imgconv_gqa_dataset,
    llava_imgconv_ocr_vqa_dataset,
    llava_imgconv_textvqa_dataset,
    llava_imgconv_dataset,
    coco_genseg_dataset,
    refcoco_refseg_dataset,
    refcocop_refseg_dataset,
    refcocog_refseg_dataset,
    lisa_reaseg_dataset,
    grandf_gcgseg_dataset,
    refcocog_gcgseg_dataset,
    psg_gcgseg_dataset,
    flickr_gcgseg_dataset,
    coco_vgdseg_dataset,
]:
    _sync_dataset_cfg(_dataset_cfg)
_sync_dataset_cfg(val_datasets)

if llava_split_mode == "single":
    _llava_train_datasets = [llava_imgconv_dataset]
else:
    _llava_train_datasets = [
        llava_imgconv_coco_dataset,
        llava_imgconv_vg_dataset,
        llava_imgconv_gqa_dataset,
        llava_imgconv_ocr_vqa_dataset,
        llava_imgconv_textvqa_dataset,
    ]

_train_dataset_list = _llava_train_datasets + [
    coco_genseg_dataset,
    refcoco_refseg_dataset,
    refcocop_refseg_dataset,
    refcocog_refseg_dataset,
    lisa_reaseg_dataset,
    grandf_gcgseg_dataset,
    refcocog_gcgseg_dataset,
    psg_gcgseg_dataset,
    flickr_gcgseg_dataset,
    coco_vgdseg_dataset,
]

train_datasets = dict(
    type=__import__("xsam.dataset", fromlist=["ConcatDataset"]).ConcatDataset,
    oversample_ratio=0.1,
    datasets=_train_dataset_list,
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=pin_memory,
    dataset=train_datasets,
    persistent_workers=persistent_workers if dataloader_num_workers > 0 else False,
    sampler=dict(
        type=__import__("xsam.dataset.samplers", fromlist=["SourceGroupedSampler"]).SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=__import__("xsam.dataset.collate_fns", fromlist=["xsam_collate_fn"]).xsam_collate_fn),
)
if dataloader_num_workers > 0 and "prefetch_factor" in _profile:
    train_dataloader["prefetch_factor"] = int(_profile["prefetch_factor"])

vis_datasets = val_datasets

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=__import__("mmengine.optim", fromlist=["AmpOptimWrapper"]).AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

param_scheduler = [
    dict(
        type=__import__("mmengine.optim", fromlist=["LinearLR"]).LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=__import__("mmengine.optim", fromlist=["CosineAnnealingLR"]).CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=__import__("xsam.engine.runner", fromlist=["TrainLoop"]).TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
visualizer = dict(
    type=__import__("xsam.utils.visualize", fromlist=["Visualizer"]).Visualizer,
    scale=1.0,
    font_size_scale=1.0,
)

custom_hooks = [
    dict(
        type=__import__("xsam.engine.hooks", fromlist=["ModelInfoHook"]).ModelInfoHook,
        module_names=["llm", "visual_projector", "seg_projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(
        type=__import__("xsam.engine.hooks", fromlist=["DatasetInfoHook"]).DatasetInfoHook,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
    ),
    dict(type=__import__("xsam.engine.hooks", fromlist=["PTCheckpointHook"]).PTCheckpointHook, clean_pth=False),
]

_enable_eval_chat_hook = evaluation_freq > 0 and evaluation_images is not None and len(evaluation_images) > 0
if _enable_eval_chat_hook:
    for _img in evaluation_images:
        if not __import__("os").path.exists(_img):
            _enable_eval_chat_hook = False
            break
if _enable_eval_chat_hook and vprompt_masks is not None:
    for _mask_group in vprompt_masks:
        if _mask_group[0] is None:
            continue
        for _mask_path in _mask_group:
            if not __import__("os").path.exists(_mask_path):
                _enable_eval_chat_hook = False
                break
        if not _enable_eval_chat_hook:
            break

if _enable_eval_chat_hook:
    custom_hooks.insert(
        2,
        dict(
            type=__import__("xsam.engine.hooks", fromlist=["EvaluateChatHook"]).EvaluateChatHook,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            image_processor=image_processor,
            postprocess_fns=[
                None,
                genseg_postprocess_fn,
                refseg_postprocess_fn,
                reaseg_postprocess_fn,
                gcgseg_postprocess_fn,
                intseg_postprocess_fn,
                intseg_postprocess_fn,
                intseg_postprocess_fn,
                intseg_postprocess_fn,
                vgdseg_postprocess_fn,
                vgdseg_postprocess_fn,
                vgdseg_postprocess_fn,
                vgdseg_postprocess_fn,
                vgdseg_postprocess_fn,
            ],
            extra_image_processor=extra_image_processor,
            visualizer=visualizer,
            every_n_iters=evaluation_freq,
            evaluation_inputs=evaluation_inputs,
            evaluation_images=evaluation_images,
            vprompt_masks=vprompt_masks,
            system=SYSTEM,
            prompt_template=prompt_template,
        ),
    )

default_hooks = dict(
    timer=dict(type=__import__("mmengine.hooks", fromlist=["IterTimerHook"]).IterTimerHook),
    logger=dict(
        type=__import__("mmengine.hooks", fromlist=["LoggerHook"]).LoggerHook,
        log_metric_by_epoch=False,
        interval=logging_interval,
    ),
    param_scheduler=dict(type=__import__("mmengine.hooks", fromlist=["ParamSchedulerHook"]).ParamSchedulerHook),
    checkpoint=dict(
        type=__import__("mmengine.hooks", fromlist=["CheckpointHook"]).CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=__import__("mmengine.hooks", fromlist=["DistSamplerSeedHook"]).DistSamplerSeedHook),
)

# Keep runtime-only modules out of the exported MMEngine config dict.
del os
