from copy import deepcopy
from pathlib import Path

import torch
from peft import LoraConfig
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, SiglipImageProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import (
    ConcatDataset,
    GCGSegDataset,
    GenSegDataset,
    ImgConvDataset,
    IntSegDataset,
    OVSegDataset,
    ReaSegDataset,
    RefSegDataset,
    VGDSegDataset,
)
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import (
    dataset_map_fn_factory,
    gcgseg_map_fn,
    genseg_map_fn,
    imgconv_map_fn,
    intseg_map_fn,
    ovseg_map_fn,
    reaseg_map_fn,
    refseg_map_fn,
    template_map_fn_factory,
    vgdseg_map_fn,
)
from xsam.dataset.process_fns import (
    gcgseg_postprocess_fn,
    genseg_postprocess_fn,
    intseg_postprocess_fn,
    ovseg_postprocess_fn,
    process_map_fn_factory,
    reaseg_postprocess_fn,
    refseg_postprocess_fn,
    vgdseg_postprocess_fn,
)
from xsam.dataset.processors import Sam2ImageProcessor, SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.evaluation.evaluators import (
    GCGSegEvaluator,
    GenSegEvaluator,
    IntSegEvaluator,
    OVSegEvaluator,
    ReaSegEvaluator,
    RefSegEvaluator,
    VGDSegEvaluator,
)
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam2 import Sam2Model
from xsam.utils.visualize import Visualizer

_DEFAULT_PROFILE_YAML = "profiles/base_plus_1024_gpu1.yaml"
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
# NOTE:
# mmengine lazy config does not allow calling imported functions (e.g. getenv) at parse time.
# Keep deterministic defaults here and override via `--cfg-options` when needed.
code_dir = __import__("os").environ.get("CODE_DIR", "./xsam/")
data_dir = __import__("os").environ.get("DATA_DIR", "./data/")
init_dir = __import__("os").environ.get("INIT_DIR", "./inits/")
work_dir = __import__("os").environ.get("WORK_DIR", "./runs/")
root_dir = __import__("os").environ.get("ROOT_DIR", "./")
root_dir = root_dir if root_dir.endswith("/") else root_dir + "/"
_cfg_dir = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
_image_dir_candidates = [
    __import__("os").path.normpath(__import__("os").path.join(_cfg_dir, "..", "images")),
    __import__("os").path.normpath(__import__("os").path.join(root_dir, "xsam/xsam/configs/xsam/images")),
    __import__("os").path.normpath(__import__("os").path.join(root_dir, "xsam/configs/xsam/images")),
    __import__("os").path.normpath(__import__("os").path.join(code_dir, "xsam/configs/xsam/images")),
    __import__("os").path.normpath(__import__("os").path.join(code_dir, "configs/xsam/images")),
]
images_dir = None
for _p in _image_dir_candidates:
    if __import__("os").path.exists(_p):
        images_dir = _p.rstrip("/") + "/"
        break
if images_dir is None:
    images_dir = _image_dir_candidates[0].rstrip("/") + "/"

# Profile
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
    segmentor_lora_cfg = dict(type=LoraConfig, **segmentor_policy["segmentor_lora_kwargs"])

# Model
llm_name_or_path = init_dir + str(_profile["llm_name"])
visual_encoder_name_or_path = init_dir + "siglip2-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + str(_profile["seg_encoder_name"])
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

_s1_rel = _profile.get("s1_pretrained_relpath")
s1_pretrained_pth = None if _s1_rel in (None, "null", "") else work_dir + str(_s1_rel)

_visual_encoder_rel = _profile.get("visual_encoder_pretrained_relpath")
visual_encoder_pretrained_pth = None if _visual_encoder_rel in (None, "null", "") else init_dir + str(_visual_encoder_rel)

projector_pretrained_pth = None
_projector_candidates = _profile.get("projector_pretrained_candidates")
if isinstance(_projector_candidates, list) and _projector_candidates:
    projector_pretrained_pth = work_dir + str(_projector_candidates[0])
    for _cand in _projector_candidates:
        _cand_abs = work_dir + str(_cand)
        if __import__("os").path.exists(_cand_abs):
            projector_pretrained_pth = _cand_abs
            break
else:
    _projector_rel = _profile.get("projector_pretrained_relpath")
    projector_pretrained_pth = None if _projector_rel in (None, "null", "") else init_dir + str(_projector_rel)

_llm_projector_rel = _profile.get("llm_projector_pretrained_relpath")
llm_projector_pretrained_pth = None if _llm_projector_rel in (None, "null", "") else init_dir + str(_llm_projector_rel)

_s2_rel = _profile.get("s2_pretrained_relpath")
if _s2_rel == "__projector__":
    s2_pretrained_pth = projector_pretrained_pth
elif _s2_rel in (None, "null", ""):
    s2_pretrained_pth = None
else:
    s2_pretrained_pth = work_dir + str(_s2_rel)

# Prompt
prompt_template = getattr(PROMPT_TEMPLATE, str(_profile.get("prompt_template", "phi3_chat")))
max_length = int(_profile.get("max_length", 2319))

# Scheduler & Optimizer
batch_size = int(_profile["batch_size"])  # per_device (single GPU)
accumulative_counts = int(_profile["accumulative_counts"])
dataloader_num_workers = int(_profile["dataloader_num_workers"])
max_epochs = float(_profile["max_epochs"])
optim_type = AdamW
lr = float(_profile.get("lr", 4e-5))
betas = tuple(_profile.get("betas", [0.9, 0.999]))
weight_decay = float(_profile.get("weight_decay", 0.05))
max_norm = float(_profile.get("max_norm", 1.0))  # grad clip
warmup_ratio = float(_profile.get("warmup_ratio", 0.03))

# LoRA (LLM only)
enable_lora = bool(_profile.get("enable_lora", True))
llm_lora_rank = int(_profile.get("llm_lora_rank", 64))
llm_lora_alpha = int(_profile.get("llm_lora_alpha", 128))
llm_lora_dropout = float(_profile.get("llm_lora_dropout", 0.05))

# QLoRA: 4-bit quantized base LLM + LoRA adapters
enable_4bit_llm = bool(_profile.get("enable_4bit_llm", True))
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
save_total_limit = int(_profile.get("save_total_limit", 2))  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = int(_profile.get("logging_interval", 10))

# Evaluate the generation performance during the training
evaluation_freq = int(_profile.get("evaluation_freq", 2000))
freeze_llm = bool(_profile.get("freeze_llm", True))
connector_type = _profile.get("connector_type")
llm_attn_implementation = str(_profile.get("llm_attn_implementation", "flash_attention_2"))
seg_attn_implementation = str(_profile.get("seg_attn_implementation", "flash_attention_2"))
llava_split_mode = str(_profile.get("llava_split_mode", "split"))
persistent_workers = bool(_profile.get("persistent_workers", True))
use_train_ratio = bool(_profile.get("use_train_ratio", True))
SYSTEM = ""
evaluation_images = [
    images_dir + "imgconv.jpg",
    images_dir + "genseg.jpg",
    images_dir + "refseg.jpg",
    images_dir + "reaseg.jpg",
    images_dir + "gcgseg.jpg",
    images_dir + "intseg.jpg",
    images_dir + "intseg.jpg",
    images_dir + "intseg.jpg",
    images_dir + "intseg.jpg",
    images_dir + "vgdseg.jpg",
    images_dir + "vgdseg.jpg",
    images_dir + "vgdseg.jpg",
    images_dir + "vgdseg.jpg",
    images_dir + "vgdseg.jpg",
]
evaluation_inputs = [
    "Can you describe this image in detail? Please elaborate in your response.",
    "Can you generate segmentation masks for this image based on the specified categories: <p>person</p>, <p>bicycle</p>, <p>car</p>, <p>motorcycle</p>, <p>airplane</p>, <p>bus</p>, <p>train</p>, <p>truck</p>, <p>boat</p>, <p>traffic light</p>, <p>fire hydrant</p>, <p>stop sign</p>, <p>parking meter</p>, <p>bench</p>, <p>bird</p>, <p>cat</p>, <p>dog</p>, <p>horse</p>, <p>sheep</p>, <p>cow</p>, <p>elephant</p>, <p>bear</p>, <p>zebra</p>, <p>giraffe</p>, <p>backpack</p>, <p>umbrella</p>, <p>handbag</p>, <p>tie</p>, <p>suitcase</p>, <p>frisbee</p>, <p>skis</p>, <p>snowboard</p>, <p>sports ball</p>, <p>kite</p>, <p>baseball bat</p>, <p>baseball glove</p>, <p>skateboard</p>, <p>surfboard</p>, <p>tennis racket</p>, <p>bottle</p>, <p>wine glass</p>, <p>cup</p>, <p>fork</p>, <p>knife</p>, <p>spoon</p>, <p>bowl</p>, <p>banana</p>, <p>apple</p>, <p>sandwich</p>, <p>orange</p>, <p>broccoli</p>, <p>carrot</p>, <p>hot dog</p>, <p>pizza</p>, <p>donut</p>, <p>cake</p>, <p>chair</p>, <p>couch</p>, <p>potted plant</p>, <p>bed</p>, <p>dining table</p>, <p>toilet</p>, <p>tv</p>, <p>laptop</p>, <p>mouse</p>, <p>remote</p>, <p>keyboard</p>, <p>cell phone</p>, <p>microwave</p>, <p>oven</p>, <p>toaster</p>, <p>sink</p>, <p>refrigerator</p>, <p>book</p>, <p>clock</p>, <p>vase</p>, <p>scissors</p>, <p>teddy bear</p>, <p>hair drier</p>, <p>toothbrush</p>, <p>banner</p>, <p>blanket</p>, <p>bridge</p>, <p>cardboard</p>, <p>counter</p>, <p>curtain</p>, <p>door</p>, <p>floor wood</p>, <p>flower</p>, <p>fruit</p>, <p>gravel</p>, <p>house</p>, <p>light</p>, <p>mirror</p>, <p>net</p>, <p>pillow</p>, <p>platform</p>, <p>playingfield</p>, <p>railroad</p>, <p>river</p>, <p>road</p>, <p>roof</p>, <p>sand</p>, <p>sea</p>, <p>shelf</p>, <p>snow</p>, <p>stairs</p>, <p>tent</p>, <p>towel</p>, <p>wall brick</p>, <p>wall stone</p>, <p>wall tile</p>, <p>wall wood</p>, <p>water</p>, <p>window blind</p>, <p>window</p>, <p>tree</p>, <p>fence</p>, <p>ceiling</p>, <p>sky</p>, <p>cabinet</p>, <p>table</p>, <p>floor</p>, <p>pavement</p>, <p>mountain</p>, <p>grass</p>, <p>dirt</p>, <p>paper</p>, <p>food</p>, <p>building</p>, <p>rock</p>, <p>wall</p>, <p>rug</p>? Please output the segmentation mask.",
    "Can you segment <p>the women with red coat</p> in this image? Please output the corresponding segmentation mask.",
    "<p>when enjoying an ice cream sundae, what can we use to scoop up the whipped cream and place it on top of the ice cream?</p> Please output the corresponding segmentation mask.",
    "Can you provide a brief description of the this image? Respond with interleaved segmentation masks for the corresponding phrases.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the <p><region></p> in this image? Please output the corresponding segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>, <p><region></p>? Please output the segmentation mask.",
    "Can you segment the image based on the following regions: <p><region></p>, <p><region></p>? Please output the segmentation mask.",
]
vprompt_masks = [
    (None,),
    (None,),
    (None,),
    (None,),
    (None,),
    (images_dir + "vprompt_masks/intseg_point0.png",),
    (images_dir + "vprompt_masks/intseg_scribble1.png",),
    (images_dir + "vprompt_masks/intseg_box0.png",),
    (images_dir + "vprompt_masks/intseg_mask1.png",),
    (images_dir + "vprompt_masks/vgdseg_point0.png",),
    (images_dir + "vprompt_masks/vgdseg_scribble1.png",),
    (images_dir + "vprompt_masks/vgdseg_box0.png",),
    (
        images_dir + "vprompt_masks/vgdseg_point0.png",
        images_dir + "vprompt_masks/vgdseg_scribble1.png",
    ),
    (
        images_dir + "vprompt_masks/vgdseg_box0.png",
        images_dir + "vprompt_masks/vgdseg_point1.png",
    ),
]

#######################################################################
#            PART 2  Model & Tokenizer &  Processor              #
#######################################################################
# TODO: add special tokens via import from xsam.utils
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"  # "phrase" "cls" "all"
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

# Use SAM2 processor when encoder is SAM2.
extra_image_processor = dict(
    type=Sam2ImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

llm_cfg = dict(
    type=AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=False,
    torch_dtype=torch.bfloat16,
    attn_implementation=llm_attn_implementation,
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
        target_modules=None,
    )

model = dict(
    type=XSamModel,
    freeze_llm=freeze_llm,
    freeze_visual_encoder=False,
    freeze_segmentor_encoder=segmentor_policy["freeze_segmentor_encoder"],
    freeze_segmentor_trunk=segmentor_policy["freeze_segmentor_trunk"],
    freeze_segmentor_fpn=segmentor_policy["freeze_segmentor_fpn"],
    llm_lora=llm_lora_cfg,
    segmentor_lora=segmentor_lora_cfg,
    use_dual_encoder=True,
    use_vision_sampler=True,
    visual_select_layer=-3,
    connector_type=connector_type,
    cond_type=cond_type,
    seg_select_layers=[1, 2, 3],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=genseg_postprocess_fn,
    llm=llm_cfg,
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=Sam2Model.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=seg_attn_implementation,
        ),
        decoder=dict(
            type=Mask2FormerModel.from_pretrained,
            pretrained_model_name_or_path=seg_decoder_name_or_path,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=False,
        open_cls=True,
    ),
)

if visual_encoder_pretrained_pth is not None:
    model["visual_encoder_pretrained_pth"] = visual_encoder_pretrained_pth
if projector_pretrained_pth is not None:
    model["projector_pretrained_pth"] = projector_pretrained_pth
if llm_projector_pretrained_pth is not None:
    model["llm_projector_pretrained_pth"] = llm_projector_pretrained_pth
if llm_lora_cfg is None:
    model.pop("llm_lora", None)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
imgconv_data_root = data_dir + "imgconv_data/"
genseg_data_root = data_dir + "genseg_data/"
ovseg_data_root = data_dir + "ovseg_data/"
refseg_data_root = data_dir + "refseg_data/"
reaseg_data_root = data_dir + "reaseg_data/"
gcgseg_data_root = data_dir + "gcgseg_data/"
intseg_data_root = data_dir + "intseg_data/"
vgdseg_data_root = data_dir + "vgdseg_data/"

# Per-dataset sampling ratio for training. Set value in [0, 1].
train_ratio = dict(
    llava_imgconv_coco=0.1,
    llava_imgconv_vg=0.30,
    llava_imgconv_gqa=0.30,
    llava_imgconv_ocr_vqa=0.30,
    llava_imgconv_textvqa=1.00,

    coco_panoptic_genseg=0.2,
    refcoco_refseg=0.25,
    refcoco_plus_refseg=0.25,
    refcocog_refseg=0.3,
    lisa_reaseg=1.0,
    grandf_gcgseg=1.0,
    refcocog_gcgseg=1.0,
    psg_gcgseg=1.0,
    flickr_gcgseg=0.2,
    coco_vgdseg=0.25,
)
if isinstance(_profile.get("train_ratio"), dict):
    train_ratio.update(_profile["train_ratio"])
if not use_train_ratio:
    for _ratio_key in train_ratio:
        train_ratio[_ratio_key] = 1.0

_llava_imgconv_base = dict(
    type=ImgConvDataset,
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=imgconv_data_root + "llava/llava_images",
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    task_name="imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=imgconv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=False,
)

llava_imgconv_coco_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_imgconv_coco.json",
    data_name="llava_imgconv_coco",
    train_ratio=train_ratio["llava_imgconv_coco"],
)

llava_imgconv_vg_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_imgconv_vg.json",
    data_name="llava_imgconv_vg",
    train_ratio=train_ratio["llava_imgconv_vg"],
)

llava_imgconv_gqa_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_imgconv_gqa.json",
    data_name="llava_imgconv_gqa",
    train_ratio=train_ratio["llava_imgconv_gqa"],
)

llava_imgconv_ocr_vqa_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_imgconv_ocr_vqa.json",
    data_name="llava_imgconv_ocr_vqa",
    train_ratio=train_ratio["llava_imgconv_ocr_vqa"],
)

llava_imgconv_textvqa_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_imgconv_textvqa.json",
    data_name="llava_imgconv_textvqa",
    train_ratio=train_ratio["llava_imgconv_textvqa"],
)

llava_imgconv_dataset = dict(
    **_llava_imgconv_base,
    data_path=imgconv_data_root + "llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
    data_name="llava_imgconv",
    train_ratio=train_ratio.get("llava_imgconv", 1.0),
)

coco_genseg_dataset = dict(
    type=GenSegDataset,
    data_path=genseg_data_root + "coco2017/annotations/panoptic_train2017.json",
    image_folder=genseg_data_root + "coco2017/train2017",
    panseg_map_folder=genseg_data_root + "coco2017/panoptic_train2017",
    tokenizer=tokenizer,
    task_name="genseg",
    data_name="coco_panoptic_genseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=genseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    use_variant_cat=True,
    pad_image_to_square=False,
    train_ratio=train_ratio["coco_panoptic_genseg"],
)

refcoco_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcoco",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcoco_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["refcoco_refseg"],
)

refcocop_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcoco+",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcoco+_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["refcoco_plus_refseg"],
)

refcocog_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcocog",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcocog_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["refcocog_refseg"],
)

_lisa_explain_path = reaseg_data_root + "lisa/explanatory/train.json"
if not __import__("os").path.exists(_lisa_explain_path):
    _lisa_explain_path = None

lisa_reaseg_dataset = dict(
    type=ReaSegDataset,
    data_root=reaseg_data_root + "lisa",
    image_folder=reaseg_data_root + "lisa/train",
    explain_path=_lisa_explain_path,
    data_mode="train",
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="lisa_reaseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reaseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reaseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["lisa_reaseg"],
)

grandf_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/GranDf_HA_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="grandf_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["grandf_gcgseg"],
)

refcocog_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/RefCOCOg_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/coco2014/train2014",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="refcocog_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["refcocog_gcgseg"],
)

psg_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/OpenPsgGCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/coco2017",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="psg_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["psg_gcgseg"],
)

flickr_gcgseg_dataset = dict(
    type=GCGSegDataset,
    data_path=gcgseg_data_root + "grand_f/annotations/train/flickr_mergedGT_GCG_train.json",
    data_root=gcgseg_data_root,
    image_folder=gcgseg_data_root + "grand_f/images/flickr30k/images/train",
    data_mode="train",
    tokenizer=tokenizer,
    task_name="gcgseg",
    data_name="flickr_gcgseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=gcgseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    train_ratio=train_ratio["flickr_gcgseg"],
)

coco_vgdseg_dataset = dict(
    type=VGDSegDataset,
    source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_train2017.json",
    data_path=vgdseg_data_root + "coco_vgd/annotations/coco_vgdseg_train.json",
    image_folder=vgdseg_data_root + "coco_vgd/coco2017/train2017",
    tokenizer=tokenizer,
    data_mode="train",
    task_name="vgdseg",
    data_name="coco_vgdseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=vgdseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_negative_sample=True,
    num_class=5,
    max_length=max_length,
    pad_image_to_square=False,
    train_ratio=train_ratio["coco_vgdseg"],
)

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

train_datasets = dict(type=ConcatDataset, oversample_ratio=0.1, datasets=_train_dataset_list)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_datasets,
    persistent_workers=persistent_workers,
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)

# False for predict mode, True for tensor mode
output_ids_with_output = True
val_datasets = [
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        panseg_map_folder=genseg_data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=genseg_data_root + "coco2017/panoptic_semseg_val2017",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="coco_panoptic_genseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        output_ids_with_output=output_ids_with_output,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="panoptic_genseg",
            threshold=0.0,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        panseg_map_folder=genseg_data_root + "coco2017/panoptic_val2017",
        semseg_map_folder=genseg_data_root + "coco2017/panoptic_semseg_val2017",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="genseg",
        data_name="coco_panoptic_semantic_genseg",  # semantic genseg shared with panoptic annotation
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="semantic_genseg",
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=GenSegDataset,
        data_path=genseg_data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=genseg_data_root + "coco2017/val2017",
        task_name="genseg",
        data_name="coco_instance_genseg",
        data_mode="eval",
        tokenizer=tokenizer,
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=genseg_postprocess_fn,
            task_name="instance_genseg",
            threshold=0.0,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=genseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory,
            template=prompt_template,
            output_suffix=output_ids_with_output,
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        panseg_map_folder=ovseg_data_root + "ade20k/ade20k_panoptic_val",
        semseg_map_folder=ovseg_data_root + "ade20k/annotations_detectron2/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_panoptic_ovseg",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory, fn=ovseg_postprocess_fn, threshold=0.0, task_name="panoptic_ovseg"
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_panoptic_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        panseg_map_folder=ovseg_data_root + "ade20k/ade20k_panoptic_val",
        semseg_map_folder=ovseg_data_root + "ade20k/annotations_detectron2/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_panoptic_semantic_ovseg",  # semantic ovseg shared with panoptic annotation
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(type=process_map_fn_factory, fn=ovseg_postprocess_fn, task_name="semantic_ovseg"),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=OVSegDataset,
        data_path=ovseg_data_root + "ade20k/ade20k_instance_val.json",
        image_folder=ovseg_data_root + "ade20k/images/validation",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="ovseg",
        data_name="ade20k_instance_ovseg",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=ovseg_map_fn,
            cond_type=cond_type,
        ),
        postprocess_fn=dict(
            type=process_map_fn_factory, fn=ovseg_postprocess_fn, task_name="instance_ovseg", threshold=0.0
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="testA",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_testA_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco",
        data_split="testB",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco_testB_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="testA",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_testA_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcoco+",
        data_split="testB",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcoco+_testB_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcocog",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcocog_val_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=RefSegDataset,
        data_root=refseg_data_root,
        image_folder=refseg_data_root + "images/train2014",
        dataset="refcocog",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="refseg",
        data_name="refcocog_test_refseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=refseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=refseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/val",
        data_split="val",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="val_reaseg",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_all_reaseg",
        query_type="all",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_sentence_reaseg",
        query_type="sentence",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=ReaSegDataset,
        data_root=reaseg_data_root + "lisa",
        image_folder=reaseg_data_root + "lisa/test",
        data_split="test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="reaseg",
        data_name="test_phrase_reaseg",
        query_type="phrase",
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        output_ids_with_output=output_ids_with_output,
        image_processor=image_processor,
        postprocess_fn=reaseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=reaseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_variant_cat=True,
        use_random_cat=True,
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=GCGSegDataset,
        data_path=gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_mask_gt.json",
        cap_data_path=gcgseg_data_root + "grand_f/annotations/val_test/val_gcg_coco_caption_gt.json",
        data_root=gcgseg_data_root,
        image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="gcgseg",
        data_name="val_gcgseg",
        output_ids_with_output=False,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=gcgseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=gcgseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=GCGSegDataset,
        data_path=gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_mask_gt.json",
        cap_data_path=gcgseg_data_root + "grand_f/annotations/val_test/test_gcg_coco_caption_gt.json",
        data_root=gcgseg_data_root,
        image_folder=gcgseg_data_root + "grand_f/images/GranDf_HA_images/val_test",
        data_mode="eval",
        tokenizer=tokenizer,
        task_name="gcgseg",
        data_name="test_gcgseg",
        output_ids_with_output=False,
        cond_type=cond_type,
        special_tokens=special_tokens,
        image_processor=image_processor,
        extra_image_processor=extra_image_processor,
        postprocess_fn=gcgseg_postprocess_fn,
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=gcgseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template, output_suffix=False),
        max_length=max_length,
        pad_image_to_square=True,
        ignore_label=ignore_label,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="point_vgdseg",
        data_mode="eval",
        visual_prompt_type="point_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_class=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="scribble_vgdseg",
        data_mode="eval",
        visual_prompt_type="scribble_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_class=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="box_vgdseg",
        data_mode="eval",
        visual_prompt_type="box_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_class=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=VGDSegDataset,
        source_data_path=vgdseg_data_root + "coco_vgd/coco2017/annotations/instances_val2017.json",
        data_path=vgdseg_data_root + "coco_vgd/annotations/vgdseg_val.json",
        image_folder=vgdseg_data_root + "coco_vgd/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="vgdseg",
        data_name="mask_vgdseg",
        data_mode="eval",
        visual_prompt_type="mask_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=vgdseg_postprocess_fn,
            threshold=0.0,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=dict(
            type=dataset_map_fn_factory,
            fn=vgdseg_map_fn,
            cond_type=cond_type,
        ),
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        use_negative_sample=False,
        num_class=5,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="point_intseg",
        data_mode="eval",
        visual_prompt_type="point_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="scribble_intseg",
        data_mode="eval",
        visual_prompt_type="scribble_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="box_intseg",
        data_mode="eval",
        visual_prompt_type="box_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=IntSegDataset,
        source_data_path=intseg_data_root + "coco_int/annotations/coco_interactive_val_psalm.json",
        data_path=intseg_data_root + "coco_int/annotations/intseg_val.json",
        image_folder=intseg_data_root + "coco_int/coco2017/val2017",
        tokenizer=tokenizer,
        task_name="intseg",
        data_name="mask_intseg",
        data_mode="eval",
        visual_prompt_type="mask_visual_prompt",
        output_ids_with_output=output_ids_with_output,
        cond_type=cond_type,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        image_processor=image_processor,
        postprocess_fn=dict(
            type=process_map_fn_factory,
            fn=intseg_postprocess_fn,
            threshold=0.5,
            return_contiguous_labels=True,
        ),
        dataset_map_fn=intseg_map_fn,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template, output_suffix=output_ids_with_output
        ),
        max_length=max_length,
        pad_image_to_square=True,
    ),
]

val_evaluators = [
    dict(
        type=GenSegEvaluator,
        distributed=True,
        data_name="coco_panoptic_genseg",
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
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_panoptic_ovseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_panoptic_semantic_ovseg",
        distributed=True,
    ),
    dict(
        type=OVSegEvaluator,
        data_name="ade20k_instance_ovseg",
        distributed=True,
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_testA_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco_testB_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_testA_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcoco+_testB_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcocog_val_refseg",
    ),
    dict(
        type=RefSegEvaluator,
        distributed=True,
        data_name="refcocog_test_refseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="val_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_all_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_sentence_reaseg",
    ),
    dict(
        type=ReaSegEvaluator,
        distributed=True,
        data_name="test_phrase_reaseg",
    ),
    dict(
        type=GCGSegEvaluator,
        distributed=True,
        data_name="val_gcgseg",
    ),
    dict(
        type=GCGSegEvaluator,
        distributed=True,
        data_name="test_gcgseg",
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="point_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="scribble_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="box_vgdseg",
        distributed=True,
    ),
    dict(
        type=VGDSegEvaluator,
        data_name="mask_vgdseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="point_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="scribble_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="box_intseg",
        distributed=True,
    ),
    dict(
        type=IntSegEvaluator,
        data_name="mask_intseg",
        distributed=True,
    ),
]

# Keep lazy-config parse safe: avoid parse-time copy/mutation on lazy objects.
vis_datasets = val_datasets

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
    paramwise_cfg=dict(
        # Avoid adding tied/shared parameters (e.g., embedding <-> lm_head) multiple times
        # when traversing complex HF modules
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
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
# set visualizer
visualizer = dict(
    type=Visualizer,
    scale=1.0,
    font_size_scale=1.0,
)

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=PTCheckpointHook, clean_pth=False),
]

_enable_eval_chat_hook = evaluation_images is not None and len(evaluation_images) > 0
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
            type=EvaluateChatHook,
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

del _config_dir
del _profile_cfg
del _profile_yaml
del _profile_yaml_path

"""
bash run.sh --modes train \
  --config xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/xsam_sam2_mixed_finetune.py \
  --yaml xsam/xsam/configs/xsam/s3_mixed_finetune/sam2/profiles/base_plus_1024_gpu1.yaml

"""
