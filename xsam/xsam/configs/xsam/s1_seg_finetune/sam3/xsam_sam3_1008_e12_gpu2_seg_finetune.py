import torch
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
seg_encoder_name_or_path = init_dir + "sam3"
sam3_encoder_trunk = "sam3_encoder.bin"
sam3_simple_FPN = "sam3_fpn.bin"

# Freeze SAM3 encoder or not.
freeze_segmentor_encoder = True

# 2. m2f decoder
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"
# Initialize only Mask2Former decoder/pixel_decoder weights.
s1_pretrained_pth = init_dir + "extracted_weights/mask2former_decoder/xsam_mask2former_decoder.bin"

# Data
data_root = data_dir + "genseg_data/"
data_path = data_root + "coco2017/annotations/panoptic_train2017.json"
image_folder = data_root + "coco2017/train2017"
panseg_map_folder = data_root + "coco2017/panoptic_train2017"
image_size = int(1008)

# Scheduler & Optimizer
# Keep effective global batch close to gpu1 config when using 2 GPUs.
batch_size = 2  # per_device
accumulative_counts = 16
dataloader_num_workers = 8
max_epochs = 12
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
logging_interval = 10

# Spatial layer sweep defaults (used by external/sam3/layer_analysis/sweep_L_spatial.py).
sweep_spatial_args = dict(
    layers="-1,-2,-4,-6,-8,-10,-12,-16,-24,-32",
    data_names=None,
    dense_keywords="semantic_genseg,semantic_ovseg",
    ref_keywords="refseg",
    batch_size=2,
    num_workers=4,
    max_samples_per_task=0,
    train_steps=500,
    train_batch_size=2,
    train_num_workers=4,
    probe_lr=1e-4,
    probe_weight_decay=0.05,
    seed_stride=9973,
    disable_llm=True,
    output_csv="sweep_L_spatial.csv",
    output_root="runs/sweep_spatial_best_layers",
    run_name="sam3_probe_step500",
    use_tqdm=False,
    log_interval=10,
    seed=1024,
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
            type=Mask2FormerModel._from_config,
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
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
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
CUDA_VISIBLE_DEVICES=0,1 GPU_PER_NODE=2 \
bash run.sh --modes train \
  --config xsam/xsam/configs/xsam/s1_seg_finetune/sam3/xsam_sam3_1008_e12_gpu2_seg_finetune.py

"""
