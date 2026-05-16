#!/usr/bin/env python
"""Layer-wise spatial sweep with frozen SAM3 trunk and per-layer probes."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import logging
import math
import os
import os.path as osp
import shlex
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from mmengine.config import Config, DictAction
from tabulate import tabulate
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

def _find_project_root(start_path: Path) -> Path:
    """Find project root by searching upward.

    Args:
        start_path: Starting path to search from.

    Returns:
        Resolved project root.
    """
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "xsam" / "xsam").exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate project root from: {start_path}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
XSAM_PKG_ROOT = PROJECT_ROOT / "xsam"
for _path in [PROJECT_ROOT, XSAM_PKG_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from xsam.layer_analysis.common.sweep_cfg import SaptialSweepCfg
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.model.segmentors.x_segmentor import XSegmentorOutput
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.logging import XSamLogger, print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device, data_sample_to_device
from xsam.utils.utils import register_function


set_default_logging_format()

_CONSOLE_LOG_HANDLE = None
DEFAULT_SWEEP_CONFIG_PATH = "xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py"
DEFAULT_SWEEP_YAML_PATH = "xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml"


class _TeeStream:
    """Duplicate stream writes to multiple targets."""

    def __init__(self, *streams) -> None:
        """Initialize tee stream.

        Args:
            *streams: Target stream objects.

        Returns:
            None.
        """
        self.streams = streams

    def write(self, data: str) -> int:
        """Write content to all streams.

        Args:
            data: Text payload.

        Returns:
            Number of written characters.
        """
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush all streams.

        Args:
            None.

        Returns:
            None.
        """
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """Check whether any stream is tty.

        Args:
            None.

        Returns:
            Whether any target stream is a tty.
        """
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@dataclass
class LayerSpec:
    """One sweep layer spec.

    Args:
        raw_id: Raw layer id from user input.
        norm_id: Normalized non-negative layer id.

    Returns:
        None.
    """

    raw_id: int
    norm_id: int


@dataclass
class EvalEntry:
    """Holder for one dataset/evaluator pair.

    Args:
        data_name: Dataset/evaluator name.
        task_type: Task category.
        dataset: Built dataset object.
        evaluator_cfg: Evaluator config dictionary.

    Returns:
        None.
    """

    data_name: str
    task_type: str
    dataset: Any
    evaluator_cfg: Dict[str, Any]


@dataclass
class ProbeTrainStats:
    """Train statistics for per-layer probe heads.

    Args:
        mean_total_loss: Mean total loss by normalized layer id.
        mean_loss_components: Mean component losses by normalized layer id.
        loss_keys: Sorted component loss keys.
        trace_rows: Per-step per-layer trace rows.

    Returns:
        None.
    """

    mean_total_loss: Dict[int, float]
    mean_loss_components: Dict[int, Dict[str, float]]
    loss_keys: List[str]
    trace_rows: List[Dict[str, Any]]


class LayerProbeHead(nn.Module):
    """Per-layer probe head: SimpleFPN + M2F pixel decoder + M2F decoder + class head."""

    def __init__(
        self,
        base_segmentor: nn.Module,
        simple_fpn_template: nn.ModuleList,
        init_seed: Optional[int] = None,
        reinit_weights: bool = True,
    ) -> None:
        """Initialize one probe head.

        Args:
            base_segmentor: Source XSegmentor with decoder config and losses.
            simple_fpn_template: Template SAM3 simple FPN conv blocks.
            init_seed: Optional seed for deterministic random initialization.
            reinit_weights: Whether to reinitialize probe-head weights from scratch.

        Returns:
            None.
        """
        super().__init__()

        self.dec_config = copy.deepcopy(base_segmentor.dec_config)
        self.enc_config = copy.deepcopy(base_segmentor.enc_config)
        self.prompt_enc_config = copy.deepcopy(base_segmentor.prompt_enc_config)

        self.simple_fpn = copy.deepcopy(simple_fpn_template)
        self.pixel_decoder = copy.deepcopy(base_segmentor.pixel_decoder)
        self.decoder = copy.deepcopy(base_segmentor.decoder)

        self.close_cls = bool(getattr(base_segmentor, "close_cls", False))
        self.open_cls = bool(getattr(base_segmentor, "open_cls", False))
        self.use_cls = bool(getattr(base_segmentor, "use_cls", False))

        self.class_predictor = None
        if self.close_cls:
            self.class_predictor = copy.deepcopy(base_segmentor.class_predictor)

        self.logit_scale = None
        if self.open_cls:
            self.logit_scale = nn.Parameter(base_segmentor.logit_scale.detach().clone(), requires_grad=True)

        self.criterion = copy.deepcopy(base_segmentor.criterion)
        self.weight_dict = copy.deepcopy(base_segmentor.weight_dict)
        self.num_feature_levels = int(getattr(self.dec_config, "num_feature_levels", 3))

        if reinit_weights:
            if init_seed is not None:
                cpu_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(init_seed)
                self.apply(base_segmentor._init_weights)
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
            else:
                self.apply(base_segmentor._init_weights)

        # `base_model.requires_grad_(False)` is used in sweep driver.
        # Deep-copied params may inherit frozen flags, so force probe head trainable.
        for param in self.parameters():
            param.requires_grad_(True)

    def _build_multiscale_features(self, trunk_feature: Tensor) -> Tuple[Tensor, ...]:
        """Build multi-scale FPN features from one trunk feature map.

        Args:
            trunk_feature: Trunk feature map in ``[B, C, H, W]``.

        Returns:
            Multi-scale feature tuple for M2F pixel decoder.
        """
        features = [layer(trunk_feature) for layer in self.simple_fpn]
        if len(features) > self.num_feature_levels:
            features = features[-self.num_feature_levels :]
        return tuple(features)

    def get_class_prediction(
        self,
        query_embeddings: Tensor,
        cond_embeddings: Optional[Tensor],
        embed_masks: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Compute class logits.

        Args:
            query_embeddings: Decoder query embeddings in ``[B, Q, D]``.
            cond_embeddings: Optional condition embeddings.
            embed_masks: Optional condition masks.

        Returns:
            Class logits tensor or ``None``.
        """
        if not self.use_cls:
            return None

        if cond_embeddings is None:
            if self.class_predictor is None:
                raise ValueError("close_cls path requires class_predictor.")
            return self.class_predictor(query_embeddings)

        if self.logit_scale is None:
            raise ValueError("open_cls path requires logit_scale.")

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        cond_embeddings = F.normalize(cond_embeddings, dim=-1)
        cls_pred = self.logit_scale.exp() * torch.einsum("bqd,bcd->bqc", query_embeddings, cond_embeddings)
        cls_pred = torch.clamp(cls_pred, min=-500, max=500)

        if embed_masks is not None:
            if embed_masks.ndim == 2:
                embed_masks = embed_masks[:, None, :]
            embed_masks = embed_masks.to(torch.bool)
            cls_pred = cls_pred.masked_fill(~embed_masks, -1e9)

        return cls_pred

    def get_auxiliary_logits(
        self,
        classes: Tuple[Optional[Tensor], ...],
        output_masks: Tuple[Optional[Tensor], ...],
    ) -> List[Dict[str, Optional[Tensor]]]:
        """Build auxiliary prediction list for deep supervision.

        Args:
            classes: Tuple of class logits from all decoder stages.
            output_masks: Tuple of mask logits from all decoder stages.

        Returns:
            Auxiliary prediction list.
        """
        auxiliary_logits: List[Dict[str, Optional[Tensor]]] = []
        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )
        return auxiliary_logits

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Optional[Tensor],
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        auxiliary_predictions: Optional[List[Dict[str, Optional[Tensor]]]],
    ) -> Dict[str, Tensor]:
        """Compute weighted segmentation loss dictionary.

        Args:
            masks_queries_logits: Final mask logits.
            class_queries_logits: Final class logits.
            mask_labels: Ground-truth mask labels.
            class_labels: Ground-truth class labels.
            auxiliary_predictions: Auxiliary prediction list.

        Returns:
            Weighted loss dictionary.
        """
        if class_queries_logits is None:
            raise ValueError("class_queries_logits cannot be None when computing loss.")

        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        for key, weight in self.weight_dict.items():
            for loss_key in list(loss_dict.keys()):
                if key in loss_key:
                    loss_dict[loss_key] = loss_dict[loss_key] * weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        """Sum all segmentation losses.

        Args:
            loss_dict: Weighted loss dictionary.

        Returns:
            Scalar loss tensor.
        """
        return sum(loss_dict.values())

    def postprocess_masks_preds(self, masks_preds: Tuple[Optional[Tensor], ...]) -> List[Optional[Tensor]]:
        """Upscale mask predictions to encoder image size.

        Args:
            masks_preds: Tuple of mask predictions from decoder stages.

        Returns:
            Upscaled mask prediction list.
        """
        image_size = getattr(self.enc_config, "image_size", None)
        if image_size is None:
            image_size = getattr(self.enc_config, "img_size", None)
        if image_size is None:
            image_size = getattr(self.prompt_enc_config, "image_size", None)
        if image_size is None:
            image_size = getattr(self.prompt_enc_config, "img_size", None)
        if image_size is None and hasattr(self.enc_config, "backbone_config"):
            image_size = getattr(self.enc_config.backbone_config, "image_size", None)
        if image_size is None and hasattr(self.enc_config, "backbone_config"):
            image_size = getattr(self.enc_config.backbone_config, "img_size", None)
        if image_size is None:
            raise ValueError("Cannot infer image_size from config.")

        if isinstance(image_size, (tuple, list)):
            target_size = tuple(int(x) for x in image_size)
            if len(target_size) == 1:
                target_size = (target_size[0], target_size[0])
        else:
            target_size = (int(image_size), int(image_size))

        new_masks_preds: List[Optional[Tensor]] = []
        for masks_pred in masks_preds:
            if masks_pred is None:
                new_masks_preds.append(None)
                continue
            up_masks_pred = F.interpolate(
                masks_pred,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            new_masks_preds.append(up_masks_pred)

        return new_masks_preds

    def forward(
        self,
        trunk_feature: Tensor,
        seg_embeddings: Optional[Tensor] = None,
        cond_embeddings: Optional[Tensor] = None,
        embed_masks: Optional[Tensor] = None,
        cond_lens: Optional[List[int]] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: bool = False,
        output_auxiliary_logits: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> XSegmentorOutput:
        """Run one probe head forward.

        Args:
            trunk_feature: Selected trunk feature map in ``[B, C, H, W]``.
            seg_embeddings: Optional segment query embeddings.
            cond_embeddings: Optional condition embeddings.
            embed_masks: Optional condition masks.
            cond_lens: Optional condition lengths.
            mask_labels: Optional GT masks (training mode).
            class_labels: Optional GT labels (training mode).
            output_hidden_states: Whether to output decoder hidden states.
            output_auxiliary_logits: Whether to output auxiliary logits.
            output_attentions: Whether to output attentions.
            return_dict: Whether to return dataclass.

        Returns:
            ``XSegmentorOutput``.
        """
        image_embeddings = self._build_multiscale_features(trunk_feature)

        pixel_decoder_outputs = self.pixel_decoder(
            image_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        decoder_outputs = self.decoder(
            multi_scale_features=pixel_decoder_outputs.multi_scale_features,
            mask_features=pixel_decoder_outputs.mask_features,
            seg_embeddings=seg_embeddings,
            cond_lens=cond_lens,
            output_attentions=output_attentions,
        )

        class_queries_logits: Tuple[Optional[Tensor], ...] = ()
        for decoder_output in decoder_outputs.intermediate_hidden_states:
            class_prediction = self.get_class_prediction(
                decoder_output.transpose(0, 1),
                cond_embeddings,
                embed_masks,
            )
            class_queries_logits += (class_prediction,)

        masks_queries_logits = decoder_outputs.masks_queries_logits
        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        loss = None
        loss_dict = None
        if mask_labels is not None and class_labels is not None and self.training:
            if class_queries_logits[-1] is None:
                raise ValueError("Class logits are required for training loss.")
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)
        else:
            if masks_queries_logits[-1] is not None:
                masks_queries_logits = tuple(self.postprocess_masks_preds(masks_queries_logits))

        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = XSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
        )

        if not return_dict:
            tuple_out = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                tuple_out = (loss,) + tuple_out
            return tuple_out  # type: ignore[return-value]
        return output


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_default_sweep_args(config_path: str) -> SaptialSweepCfg:
    """Build default sweep arguments without relying on config-side dataclass.

    Args:
        config_path: mmengine config file path.

    Returns:
        Default ``SaptialSweepCfg``.
    """
    return SaptialSweepCfg(
        config=config_path,
        pth_model=None,
        layers="-1,-2,-4,-6,-8,-10,-12,-16,-24,-32",
        data_names=None,
        dense_keywords="semantic_genseg,semantic_ovseg",
        ref_keywords="refseg",
        batch_size=1,
        num_workers=4,
        max_samples_per_task=0,
        train_epochs=2,
        train_ratio=0.25,
        train_batch_size=1,
        grad_accum_steps=1,
        train_num_workers=4,
        save_steps=5000,
        max_save=-2,
        resume=False,
        resume_ckpt=None,
        probe_lr=1e-4,
        probe_weight_decay=0.05,
        probe_reinit=False,
        train_eval_interval=200,
        train_eval_max_samples=256,
        early_stop_patience_steps=2000,
        early_stop_miou_eps=0.1,
        seed_stride=9973,
        output_csv="sweep_L_spatial.csv",
        output_root="runs/sweep_spatial_best_layers",
        run_name="sam3_probe_ep2",
        use_tqdm=False,
        log_interval=10,
        eval_fail_fast=True,
        eval_fail_ratio_threshold=0.05,
        eval_fail_check_min_samples=64,
        eval_oom_empty_cache=True,
        eval_log_cuda_mem=True,
        seed=1024,
        cfg_options=None,
    )


def _load_sweep_yaml(config_yaml_path: str, phase: Optional[str]) -> Dict[str, Any]:
    """Load sweep YAML and resolve optional phase section.

    Args:
        config_yaml_path: YAML path.
        phase: Optional phase name.

    Returns:
        Resolved YAML dictionary.
    """
    yaml_path = Path(config_yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = (PROJECT_ROOT / yaml_path).resolve()
    if not yaml_path.exists():
        return {}

    with open(yaml_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}
    if not isinstance(raw_cfg, dict):
        raise TypeError(f"Sweep YAML must be a mapping, got {type(raw_cfg)} in {yaml_path}.")

    resolved = copy.deepcopy(raw_cfg)
    has_phase_blocks = isinstance(raw_cfg.get("common"), dict) or isinstance(raw_cfg.get("phases"), dict)
    if has_phase_blocks:
        resolved = copy.deepcopy(raw_cfg.get("common", {}))
        phase_name = phase or raw_cfg.get("default_phase", None)
        if phase_name is not None:
            phase_cfg_map = raw_cfg.get("phases", {})
            if phase_name not in phase_cfg_map:
                raise KeyError(f"Phase `{phase_name}` not found in YAML `{yaml_path}`.")
            if not isinstance(phase_cfg_map[phase_name], dict):
                raise TypeError(f"`phases.{phase_name}` must be a mapping.")
            resolved = _deep_merge_dict(resolved, phase_cfg_map[phase_name])

        for top_key in ["paths", "runtime", "data", "train", "checkpoint", "eval"]:
            section = raw_cfg.get(top_key, None)
            if isinstance(section, dict):
                resolved = _deep_merge_dict(section, resolved)

        if "mmengine_config" in raw_cfg and "mmengine_config" not in resolved:
            resolved["mmengine_config"] = raw_cfg["mmengine_config"]
        if "version" in raw_cfg and "version" not in resolved:
            resolved["version"] = raw_cfg["version"]
        if "default_phase" in raw_cfg and "default_phase" not in resolved:
            resolved["default_phase"] = raw_cfg["default_phase"]
    return resolved


def _apply_yaml_overrides(base_args: SaptialSweepCfg, yaml_cfg: Dict[str, Any]) -> SaptialSweepCfg:
    """Apply YAML fields on top of default sweep args.

    Args:
        base_args: Default sweep args.
        yaml_cfg: Resolved YAML mapping.

    Returns:
        YAML-overridden sweep args.
    """
    if not yaml_cfg:
        return copy.deepcopy(base_args)

    resolved = copy.deepcopy(base_args)
    aliases = {"mmengine_config": "config"}

    for field in SaptialSweepCfg.__annotations__.keys():
        keys = [field]
        for alias_key, target_field in aliases.items():
            if target_field == field:
                keys.append(alias_key)
        selected_key = next((key for key in keys if key in yaml_cfg), None)
        if selected_key is None:
            continue
        value = yaml_cfg[selected_key]
        if field == "data_names":
            if value is None:
                resolved.data_names = None
            elif isinstance(value, str):
                resolved.data_names = _split_csv_text(value)
            elif isinstance(value, (list, tuple)):
                resolved.data_names = [str(v).strip() for v in value if str(v).strip()]
            else:
                raise TypeError(f"`data_names` must be null/string/list, got {type(value)}")
            continue
        if field == "cfg_options":
            resolved.cfg_options = dict(value) if value is not None else None
            continue
        setattr(resolved, field, value)

    return resolved


def _resolve_sweep_args(
    config_path: Optional[str],
    config_yaml_path: Optional[str],
    phase: Optional[str],
) -> Tuple[SaptialSweepCfg, Config]:
    """Resolve sweep args from defaults + YAML without config-side dataclass dependency.

    Args:
        config_path: Optional mmengine config path from CLI.
        config_yaml_path: Optional sweep YAML path.
        phase: Optional phase key used for YAML merge.

    Returns:
        Tuple of ``(resolved_args, cfg)``.
    """
    yaml_cfg: Dict[str, Any] = {}
    resolved_yaml_path: Optional[Path] = None
    if config_yaml_path:
        resolved_yaml_path = Path(config_yaml_path)
        if not resolved_yaml_path.is_absolute():
            resolved_yaml_path = (PROJECT_ROOT / resolved_yaml_path).resolve()
        yaml_cfg = _load_sweep_yaml(config_yaml_path=config_yaml_path, phase=phase)
    if resolved_yaml_path is not None:
        os.environ["XSAM_SPATIAL_SWEEP_YAML_PATH"] = str(resolved_yaml_path)
    else:
        os.environ.pop("XSAM_SPATIAL_SWEEP_YAML_PATH", None)

    mmengine_config_path = config_path or yaml_cfg.get("mmengine_config") or yaml_cfg.get("config")
    if not mmengine_config_path:
        mmengine_config_path = DEFAULT_SWEEP_CONFIG_PATH

    config_path_obj = Path(str(mmengine_config_path))
    if not config_path_obj.is_absolute():
        config_path_obj = (PROJECT_ROOT / config_path_obj).resolve()
    mmengine_config_path = str(config_path_obj)

    default_args = _build_default_sweep_args(config_path=mmengine_config_path)
    resolved = _apply_yaml_overrides(default_args, yaml_cfg)
    resolved.config = str(mmengine_config_path)

    cfg = Config.fromfile(resolved.config, lazy_import=False)
    return resolved, cfg


def _parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Layer-wise SAM3 spatial sweep")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to mmengine config file.",
    )
    parser.add_argument(
        "--config-yaml",
        dest="config_yaml",
        type=str,
        default=DEFAULT_SWEEP_YAML_PATH,
        help="Path to sweep YAML config.",
    )
    parser.add_argument(
        "--phase",
        dest="phase",
        type=str,
        default=None,
        help="Optional phase key used when sweep YAML contains phase blocks.",
    )
    parser.add_argument("--pth-model", dest="pth_model", type=str, default=None, help="Optional model checkpoint path.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer ids.")
    parser.add_argument(
        "--data-names",
        type=str,
        default=None,
        help="Comma-separated explicit evaluator data names.",
    )
    parser.add_argument("--dense-keywords", type=str, default=None, help="Comma-separated dense keywords.")
    parser.add_argument("--ref-keywords", type=str, default=None, help="Comma-separated ref keywords.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Eval batch size.")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None, help="Eval dataloader workers.")
    parser.add_argument(
        "--max-samples-per-task",
        dest="max_samples_per_task",
        type=int,
        default=None,
        help="Max eval samples per task; 0 means full dataset.",
    )
    parser.add_argument("--train-epochs", dest="train_epochs", type=int, default=None, help="Probe train epochs.")
    parser.add_argument("--train-ratio", dest="train_ratio", type=float, default=None, help="Probe train subset ratio.")
    parser.add_argument(
        "--train-batch-size",
        dest="train_batch_size",
        type=int,
        default=None,
        help="Probe train batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        dest="grad_accum_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps for probe training.",
    )
    parser.add_argument(
        "--train-num-workers",
        dest="train_num_workers",
        type=int,
        default=None,
        help="Probe train dataloader workers.",
    )
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=None, help="Checkpoint save interval.")
    parser.add_argument(
        "--max-save",
        dest="max_save",
        type=int,
        default=None,
        help="Checkpoint retention policy: -1=unlimited, -N keep latest N, N keep latest N.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable resume.",
    )
    parser.add_argument("--resume-ckpt", dest="resume_ckpt", type=str, default=None, help="Resume checkpoint path.")
    parser.add_argument("--probe-lr", dest="probe_lr", type=float, default=None, help="Probe optimizer learning rate.")
    parser.add_argument(
        "--probe-weight-decay",
        dest="probe_weight_decay",
        type=float,
        default=None,
        help="Probe optimizer weight decay.",
    )
    parser.add_argument(
        "--probe-reinit",
        dest="probe_reinit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable probe-head random reinitialization.",
    )
    parser.add_argument(
        "--train-eval-interval",
        dest="train_eval_interval",
        type=int,
        default=None,
        help="Train-time eval interval in steps. 0 disables train-time eval.",
    )
    parser.add_argument(
        "--train-eval-max-samples",
        dest="train_eval_max_samples",
        type=int,
        default=None,
        help="Max samples per train-time eval snapshot. 0 means full dataset.",
    )
    parser.add_argument(
        "--early-stop-patience-steps",
        dest="early_stop_patience_steps",
        type=int,
        default=None,
        help="Early-stop patience in steps without sufficient mIoU improvement. 0 disables early-stop.",
    )
    parser.add_argument(
        "--early-stop-miou-eps",
        dest="early_stop_miou_eps",
        type=float,
        default=None,
        help="Minimum mIoU gain (percentage points) to count as improvement for early-stop.",
    )
    parser.add_argument("--seed-stride", dest="seed_stride", type=int, default=None, help="Per-layer seed stride.")
    parser.add_argument("--output-csv", dest="output_csv", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--output-root", dest="output_root", type=str, default=None, help="Output root directory.")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None, help="Run name.")
    parser.add_argument(
        "--use-tqdm",
        dest="use_tqdm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable tqdm.",
    )
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None, help="Log interval.")
    parser.add_argument(
        "--eval-fail-fast",
        dest="eval_fail_fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable eval fail-fast.",
    )
    parser.add_argument(
        "--eval-fail-ratio-threshold",
        dest="eval_fail_ratio_threshold",
        type=float,
        default=None,
        help="Eval fail-fast ratio threshold.",
    )
    parser.add_argument(
        "--eval-fail-check-min-samples",
        dest="eval_fail_check_min_samples",
        type=int,
        default=None,
        help="Minimum processed samples before eval fail-fast check.",
    )
    parser.add_argument(
        "--eval-oom-empty-cache",
        dest="eval_oom_empty_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.cuda.empty_cache on eval OOM.",
    )
    parser.add_argument(
        "--eval-log-cuda-mem",
        dest="eval_log_cuda_mem",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable CUDA memory logs during eval.",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default=None,
        help="Override config options in key=value format.",
    )
    raw_argv = sys.argv[1:]
    normalized_argv: List[str] = []
    idx = 0
    while idx < len(raw_argv):
        token = raw_argv[idx]
        next_idx = idx + 1
        if token == "--layers" and next_idx < len(raw_argv):
            layer_value = raw_argv[next_idx]
            if layer_value.startswith("-") and not layer_value.startswith("--"):
                normalized_argv.append(f"--layers={layer_value}")
                idx += 2
                continue
        normalized_argv.append(token)
        idx += 1
    return parser.parse_args(normalized_argv)


def _apply_cli_overrides(base_args: SaptialSweepCfg, cli_args: argparse.Namespace) -> SaptialSweepCfg:
    """Apply CLI overrides on top of config defaults.

    Args:
        base_args: Config-resolved sweep arguments.
        cli_args: Parsed CLI namespace.

    Returns:
        Overridden sweep arguments.
    """
    resolved = copy.deepcopy(base_args)
    scalar_fields = [
        "pth_model",
        "layers",
        "dense_keywords",
        "ref_keywords",
        "batch_size",
        "num_workers",
        "max_samples_per_task",
        "train_epochs",
        "train_ratio",
        "train_batch_size",
        "grad_accum_steps",
        "train_num_workers",
        "save_steps",
        "max_save",
        "resume",
        "resume_ckpt",
        "probe_lr",
        "probe_weight_decay",
        "probe_reinit",
        "train_eval_interval",
        "train_eval_max_samples",
        "early_stop_patience_steps",
        "early_stop_miou_eps",
        "seed_stride",
        "output_csv",
        "output_root",
        "run_name",
        "use_tqdm",
        "log_interval",
        "eval_fail_fast",
        "eval_fail_ratio_threshold",
        "eval_fail_check_min_samples",
        "eval_oom_empty_cache",
        "eval_log_cuda_mem",
        "seed",
    ]
    for field in scalar_fields:
        value = getattr(cli_args, field, None)
        if value is not None:
            setattr(resolved, field, value)

    if cli_args.data_names is not None:
        resolved.data_names = _split_csv_text(cli_args.data_names)

    if cli_args.cfg_options is not None:
        resolved.cfg_options = dict(cli_args.cfg_options)

    return resolved


def _split_csv_text(text: str) -> List[str]:
    """Split comma-separated text.

    Args:
        text: Input comma-separated text.

    Returns:
        Stripped non-empty token list.
    """
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_layers(text: str) -> List[int]:
    """Parse layer text into integer list.

    Args:
        text: Comma-separated layer string.

    Returns:
        Parsed layer list.
    """
    layers = [int(token) for token in _split_csv_text(text)]
    if not layers:
        raise ValueError("Layer list is empty.")
    return layers


def _build_layer_specs(layer_ids: Sequence[int], total_levels: int) -> List[LayerSpec]:
    """Normalize layer ids and deduplicate by normalized id.

    Args:
        layer_ids: Raw layer ids.
        total_levels: Total number of trunk blocks.

    Returns:
        Layer spec list preserving input order.
    """
    specs: List[LayerSpec] = []
    seen = set()
    for raw_id in layer_ids:
        norm_id = raw_id if raw_id >= 0 else total_levels + raw_id
        if norm_id < 0 or norm_id >= total_levels:
            raise ValueError(f"Invalid layer id {raw_id}. Valid range: [{-total_levels}, {total_levels - 1}].")
        if norm_id in seen:
            continue
        seen.add(norm_id)
        specs.append(LayerSpec(raw_id=raw_id, norm_id=norm_id))
    return specs


def _is_dense_task(data_name: str, dense_keywords: Sequence[str]) -> bool:
    """Check whether data name is dense task.

    Args:
        data_name: Evaluator data name.
        dense_keywords: Dense keywords.

    Returns:
        Whether matched.
    """
    return any(keyword in data_name for keyword in dense_keywords)


def _is_ref_task(data_name: str, ref_keywords: Sequence[str]) -> bool:
    """Check whether data name is ref task.

    Args:
        data_name: Evaluator data name.
        ref_keywords: Ref keywords.

    Returns:
        Whether matched.
    """
    return any(keyword in data_name for keyword in ref_keywords)


def _safe_to_float(value: Any) -> float:
    """Convert value to finite float.

    Args:
        value: Any value.

    Returns:
        Finite float or NaN.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number):
        return math.nan
    return number


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    """Deduplicate strings while preserving input order.

    Args:
        items: Input string sequence.

    Returns:
        Deduplicated list.
    """
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_model_summary_table(model: nn.Module, module_names: Sequence[str]) -> str:
    """Build model summary table text.

    Args:
        model: Model module.
        module_names: Module prefixes to summarize.

    Returns:
        Formatted table text.
    """
    tracked_names = _dedupe_keep_order([name for name in module_names if isinstance(name, str) and len(name) > 0])
    module_stats: Dict[str, Dict[str, int]] = {name: {"num_params": 0, "num_trainable_params": 0} for name in tracked_names}
    module_stats["others"] = {"num_params": 0, "num_trainable_params": 0}

    for name, param in model.named_parameters():
        num_params = int(param.ds_numel if hasattr(param, "ds_numel") else param.numel())
        matched_name = "others"
        for module_name in tracked_names:
            if name == module_name or name.startswith(f"{module_name}."):
                matched_name = module_name
                break
        module_stats[matched_name]["num_params"] += num_params
        if param.requires_grad:
            module_stats[matched_name]["num_trainable_params"] += num_params

    module_stats = {name: stats for name, stats in module_stats.items() if stats["num_params"] > 0}
    if len(module_stats) == 0:
        return "No parameters found."

    module_stats = dict(sorted(module_stats.items(), key=lambda item: item[1]["num_params"], reverse=True))
    total_params = sum(stats["num_params"] for stats in module_stats.values())
    total_trainable = sum(stats["num_trainable_params"] for stats in module_stats.values())
    total_params = max(total_params, 1)

    headers = ["Module", "# Params", "Params %", "# Trainable", "Trainable %"]
    module_lens = [max(len(name), 1) for name in module_stats.keys()]
    rows: List[List[str]] = []
    for name, stats in module_stats.items():
        params_pct = stats["num_params"] / total_params * 100.0
        trainable_pct = (
            stats["num_trainable_params"] / stats["num_params"] * 100.0 if stats["num_params"] > 0 else 0.0
        )
        rows.append(
            [
                name,
                f"{stats['num_params']:,}",
                f"{params_pct:.2f}%",
                f"{stats['num_trainable_params']:,}",
                f"{trainable_pct:.2f}%",
            ]
        )

    scale_candidates = [max(module_lens) / max(len(headers[0]), 1), 1.8, 1.6, 1.2, 1.2]
    rows.append(["=" * max(int(len(header) * scale), 1) for header, scale in zip(headers, scale_candidates)])
    rows.append(
        [
            "Total",
            f"{total_params:,}",
            "100.00%",
            f"{total_trainable:,}",
            f"{(total_trainable / total_params) * 100.0:.2f}%",
        ]
    )

    return tabulate(
        rows,
        headers=headers,
        tablefmt="outline",
        colalign=("center", "right", "right", "right", "right"),
    )


def _log_model_summary(
    model: nn.Module,
    module_names: Sequence[str],
    title: str,
) -> None:
    """Log one model summary table.

    Args:
        model: Model module.
        module_names: Module prefixes to summarize.
        title: Log title prefix.

    Returns:
        None.
    """
    table = _build_model_summary_table(model, module_names)
    print_log(f"{title}:\n{table}", logger="current")


def _resolve_base_model_summary_modules(model: Any) -> List[str]:
    """Resolve module prefixes for base model summary.

    Args:
        model: Base model instance.

    Returns:
        Ordered module-prefix list.
    """
    module_names: List[str] = []
    if getattr(model, "segmentor", None) is not None:
        module_names.extend(
            [
                "segmentor.encoder",
                "segmentor.pixel_decoder",
                "segmentor.decoder",
                "segmentor.class_predictor",
                "segmentor.criterion",
            ]
        )

    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            module_names.append(name)

    return _dedupe_keep_order(module_names)


def _extract_dataset_summary_row(
    dataset: Any,
    data_name: str,
    task_name: str,
) -> Dict[str, Any]:
    """Extract dataset summary row fields.

    Args:
        dataset: Dataset object.
        data_name: Fallback data name.
        task_name: Fallback task name.

    Returns:
        Dataset summary row.
    """
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    resolved_data_name = str(getattr(base_dataset, "data_name", data_name))
    resolved_task_name = str(getattr(base_dataset, "task_name", task_name))
    repeats = float(getattr(base_dataset, "repeats", 1.0))
    base_data_length = int(getattr(base_dataset, "data_length", len(base_dataset)))
    sample_count = int(len(dataset))
    return {
        "dataset": resolved_data_name,
        "task": resolved_task_name,
        "repeats": repeats,
        "data_length": base_data_length,
        "sample_count": sample_count,
    }


def _build_dataset_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Build dataset summary table text.

    Args:
        rows: Dataset rows with canonical keys.

    Returns:
        Formatted table text.
    """
    if len(rows) == 0:
        return "No datasets selected."

    headers = ["#", "Dataset", "Task", "# Repeats", "# Data", "# Samples"]
    table_rows: List[List[Any]] = []
    for idx, row in enumerate(rows):
        table_rows.append(
            [
                idx,
                str(row["dataset"]),
                str(row["task"]),
                f"{float(row['repeats']):.2f}",
                f"{int(row['data_length']):,}",
                f"{int(row['sample_count']):,}",
            ]
        )

    table_rows.append(
        [
            "=" * int(len(header) * scale)
            for header, scale in zip(headers, [5.0, 2.5, 2.0, 1.4, 1.4, 1.4])
        ]
    )
    table_rows.append(
        [
            "Total",
            len(rows),
            len(set(str(row["task"]) for row in rows)),
            f"{sum(float(row['repeats']) for row in rows):.2f}",
            f"{sum(int(row['data_length']) for row in rows):,}",
            f"{sum(int(row['sample_count']) for row in rows):,}",
        ]
    )
    return tabulate(
        table_rows,
        headers=headers,
        tablefmt="outline",
        colalign=("center", "center", "center", "center", "right", "right"),
    )


def _log_dataset_summary(rows: Sequence[Dict[str, Any]], title: str) -> None:
    """Log dataset summary table.

    Args:
        rows: Dataset rows.
        title: Log title prefix.

    Returns:
        None.
    """
    table = _build_dataset_summary_table(rows)
    print_log(f"{title}:\n{table}", logger="current")


def _log_probe_head_overview(layer_specs: Sequence[LayerSpec], heads: Dict[int, LayerProbeHead]) -> None:
    """Log per-layer probe-head parameter overview.

    Args:
        layer_specs: Layer specifications.
        heads: Probe heads by normalized layer id.

    Returns:
        None.
    """
    headers = ["Layer", "Normalized", "# Params", "# Trainable", "Trainable %"]
    rows: List[List[str]] = []
    for spec in layer_specs:
        head = heads[spec.norm_id]
        total_params = sum(param.numel() for param in head.parameters())
        trainable_params = sum(param.numel() for param in head.parameters() if param.requires_grad)
        trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
        rows.append(
            [
                str(spec.raw_id),
                str(spec.norm_id),
                f"{total_params:,}",
                f"{trainable_params:,}",
                f"{trainable_pct:.2f}%",
            ]
        )

    table = tabulate(
        rows,
        headers=headers,
        tablefmt="outline",
        colalign=("right", "right", "right", "right", "right"),
    )
    print_log(f"Probe heads overview:\n{table}", logger="current")


def _format_cuda_memory(device: torch.device) -> str:
    """Format CUDA memory usage string.

    Args:
        device: Target device.

    Returns:
        CUDA memory summary string.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=n/a"

    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
    total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    return f"cuda_mem={allocated:.2f}G/{reserved:.2f}G/{total:.2f}G(alloc/resv/total)"


def _compute_semantic_miou(conf_matrix: np.ndarray) -> float:
    """Compute semantic mIoU from confusion matrix.

    Args:
        conf_matrix: Confusion matrix with ignore class as last index.

    Returns:
        mIoU percentage.
    """
    if conf_matrix.ndim != 2 or conf_matrix.shape[0] != conf_matrix.shape[1]:
        return math.nan
    num_classes = conf_matrix.shape[0] - 1
    if num_classes <= 0:
        return math.nan

    matrix = conf_matrix.astype(np.float64)
    tp = matrix.diagonal()[:-1]
    pos_gt = np.sum(matrix[:-1, :-1], axis=0)
    pos_pred = np.sum(matrix[:-1, :-1], axis=1)

    union = pos_gt + pos_pred - tp
    valid = np.logical_and(pos_gt > 0, union > 0)
    if not np.any(valid):
        return math.nan

    iou = np.full(num_classes, np.nan, dtype=np.float64)
    iou[valid] = tp[valid] / union[valid]
    return float(np.nanmean(iou) * 100.0)


def _compute_semantic_pacc(conf_matrix: np.ndarray) -> float:
    """Compute semantic pixel accuracy (pACC) from confusion matrix.

    Args:
        conf_matrix: Confusion matrix with ignore class as last index.

    Returns:
        pACC percentage.
    """
    if conf_matrix.ndim != 2 or conf_matrix.shape[0] != conf_matrix.shape[1]:
        return math.nan
    if conf_matrix.shape[0] <= 1:
        return math.nan

    matrix = conf_matrix.astype(np.float64)
    valid_matrix = matrix[:-1, :-1]
    total = float(np.sum(valid_matrix))
    if total <= 0:
        return math.nan
    correct = float(np.trace(valid_matrix))
    return (correct / total) * 100.0


def _mean_without_ignore(values: np.ndarray, cat_names: Sequence[str]) -> float:
    """Compute mean excluding categories containing "ignore".

    Args:
        values: Metric values.
        cat_names: Category names.

    Returns:
        Mean value.
    """
    if values.size == 0:
        return math.nan

    valid_indices = [idx for idx, cat_name in enumerate(cat_names) if "ignore" not in str(cat_name).lower()]
    if not valid_indices:
        valid_indices = list(range(values.size))

    selected = values[valid_indices]
    if selected.size == 0:
        return math.nan
    return float(np.nanmean(selected))


def _extract_metrics(evaluator: Any) -> Dict[str, float]:
    """Extract mIoU/gIoU/cIoU/pACC from evaluator internals.

    Args:
        evaluator: Evaluator object after ``evaluate()``.

    Returns:
        Metric dictionary.
    """
    data_name = str(getattr(evaluator, "data_name", ""))
    miou = math.nan
    giou = math.nan
    ciou = math.nan
    pacc = math.nan

    if hasattr(evaluator, "iou_stat") and evaluator.iou_stat is not None:
        iou_stat = evaluator.iou_stat
        cat_names = list(getattr(iou_stat, "cat_names", []))
        ciou_arr = np.asarray(getattr(iou_stat, "ciou", []), dtype=np.float64)
        giou_arr = np.asarray(getattr(iou_stat, "giou", []), dtype=np.float64)
        if ciou_arr.size > 0:
            ciou = _mean_without_ignore(ciou_arr, cat_names)
            miou = ciou
        if giou_arr.size > 0:
            giou = _mean_without_ignore(giou_arr, cat_names)

    if "semantic" in data_name and hasattr(evaluator, "_conf_matrix"):
        conf_matrix = np.asarray(evaluator._conf_matrix)
        sem_miou = _compute_semantic_miou(conf_matrix)
        sem_pacc = _compute_semantic_pacc(conf_matrix)
        if math.isfinite(sem_miou):
            miou = sem_miou
        if math.isfinite(sem_pacc):
            pacc = sem_pacc

    return {"miou": miou, "giou": giou, "ciou": ciou, "pacc": pacc}


def _select_eval_entries(
    cfg: Config,
    dense_keywords: Sequence[str],
    ref_keywords: Sequence[str],
    explicit_data_names: Optional[Sequence[str]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str, str]]:
    """Select evaluation dataset/evaluator configs.

    Args:
        cfg: Parsed config.
        dense_keywords: Dense task keywords.
        ref_keywords: Ref task keywords.
        explicit_data_names: Explicit data-name whitelist.

    Returns:
        Selected config tuples ``(dataset_cfg, evaluator_cfg, data_name, task_type)``.
    """
    if len(cfg.val_datasets) != len(cfg.val_evaluators):
        raise ValueError(
            f"len(val_datasets)={len(cfg.val_datasets)} != len(val_evaluators)={len(cfg.val_evaluators)}."
        )

    selected = []
    explicit_set = set(explicit_data_names) if explicit_data_names is not None else None
    for dataset_cfg, evaluator_cfg in zip(cfg.val_datasets, cfg.val_evaluators):
        data_name = evaluator_cfg.get("data_name", None) if isinstance(evaluator_cfg, dict) else None
        if data_name is None:
            continue

        if explicit_set is not None:
            if data_name not in explicit_set:
                continue
            task_type = "explicit"
        else:
            if _is_dense_task(data_name, dense_keywords):
                task_type = "dense"
            elif _is_ref_task(data_name, ref_keywords):
                task_type = "ref"
            else:
                continue

        selected.append((dataset_cfg, evaluator_cfg, data_name, task_type))

    return selected


def _build_eval_entries(
    selected_cfgs: Sequence[Tuple[Dict[str, Any], Dict[str, Any], str, str]],
) -> List[EvalEntry]:
    """Build evaluation entries from configs.

    Args:
        selected_cfgs: Selected config tuples.

    Returns:
        Built evaluation entries.
    """
    entries: List[EvalEntry] = []
    for dataset_cfg, evaluator_cfg, data_name, task_type in selected_cfgs:
        dataset = BUILDER.build(dataset_cfg)
        entries.append(
            EvalEntry(
                data_name=data_name,
                task_type=task_type,
                dataset=dataset,
                evaluator_cfg=copy.deepcopy(evaluator_cfg),
            )
        )
    return entries


def _build_evaluator(evaluator_cfg: Dict[str, Any], metadata: Any) -> Any:
    """Build evaluator instance.

    Args:
        evaluator_cfg: Evaluator config.
        metadata: Dataset metadata.

    Returns:
        Built evaluator.
    """
    evaluator = BUILDER.build(copy.deepcopy(evaluator_cfg))
    evaluator.metadata = metadata
    if hasattr(evaluator, "_distributed"):
        evaluator._distributed = False
    evaluator.reset()
    return evaluator


def _get_pixel_values_from_batch(data_dict: Dict[str, Any]) -> Tensor:
    """Fetch encoder pixel values from batch data.

    Args:
        data_dict: Batched input dictionary.

    Returns:
        Pixel tensor in ``[B, 3, H, W]``.
    """
    pixel_values = data_dict.get("extra_pixel_values", None)
    if pixel_values is None:
        pixel_values = data_dict.get("pixel_values", None)
    if pixel_values is None:
        raise ValueError("Batch has neither `extra_pixel_values` nor `pixel_values`.")
    if isinstance(pixel_values, list):
        if not pixel_values:
            raise ValueError("Empty pixel value list.")
        if all(isinstance(x, torch.Tensor) and x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        else:
            raise ValueError("List pixel values have inconsistent shapes.")
    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError(f"Unsupported pixel value type: {type(pixel_values)}")
    return pixel_values


def _forward_encoder_with_trunk_capture(
    encoder: nn.Module,
    pixel_values: Tensor,
    trunk_select_layers: Sequence[int],
    encoder_dtype: torch.dtype,
) -> Any:
    """Run encoder with trunk hidden-state capture.

    Args:
        encoder: SAM3 vision encoder.
        pixel_values: Input image tensor.
        trunk_select_layers: Normalized trunk layer ids.
        encoder_dtype: Encoder dtype.

    Returns:
        Encoder outputs.
    """
    encoder_kwargs = dict(
        pixel_values=pixel_values.to(dtype=encoder_dtype),
        output_hidden_states=False,
        output_attentions=False,
        output_trunk_hidden_states=True,
        trunk_select_layers=tuple(trunk_select_layers),
        return_dict=True,
    )
    try:
        outputs = encoder(**encoder_kwargs)
    except TypeError:
        encoder_kwargs.pop("output_trunk_hidden_states", None)
        encoder_kwargs.pop("trunk_select_layers", None)
        with contextlib.suppress(TypeError):
            return encoder(**encoder_kwargs)
        outputs = encoder(
            encoder_kwargs.pop("pixel_values"),
            output_hidden_states=encoder_kwargs.get("output_hidden_states", False),
            output_attentions=encoder_kwargs.get("output_attentions", False),
            return_dict=True,
        )
    return outputs


def _extract_trunk_feature_map(
    encoder_outputs: Any,
    layer_specs: Sequence[LayerSpec],
) -> Dict[int, Tensor]:
    """Map captured trunk features by normalized layer id.

    Args:
        encoder_outputs: Encoder outputs containing captured trunk states.
        layer_specs: Layer specs.

    Returns:
        Mapping ``norm_layer_id -> feature_map``.
    """
    trunk_hidden_states = getattr(encoder_outputs, "trunk_hidden_states", None)
    trunk_selected_layers = getattr(encoder_outputs, "trunk_selected_layers", None)
    if trunk_hidden_states is None or trunk_selected_layers is None:
        raise ValueError("Encoder output does not provide trunk hidden states.")

    layer_to_feature: Dict[int, Tensor] = {
        int(layer_id): feat for layer_id, feat in zip(trunk_selected_layers, trunk_hidden_states)
    }

    selected: Dict[int, Tensor] = {}
    for spec in layer_specs:
        if spec.norm_id not in layer_to_feature:
            raise ValueError(
                f"Requested layer {spec.norm_id} missing from captured layers {list(layer_to_feature.keys())}."
            )
        selected[spec.norm_id] = layer_to_feature[spec.norm_id]
    return selected


def _build_probe_heads(
    base_model: Any,
    layer_specs: Sequence[LayerSpec],
    seed: int,
    seed_stride: int,
    reinit_weights: bool,
) -> Tuple[Dict[int, LayerProbeHead], Dict[int, torch.optim.Optimizer]]:
    """Build per-layer probe heads and optimizers.

    Args:
        base_model: Built X-SAM model containing SAM3 segmentor.
        layer_specs: Layer specs.
        seed: Base random seed.
        seed_stride: Per-layer seed stride.
        reinit_weights: Whether to reinitialize probe-head weights from scratch.

    Returns:
        Tuple of ``(heads, optimizers)`` dictionaries keyed by normalized layer id.
    """
    if getattr(base_model, "segmentor", None) is None:
        raise ValueError("Model has no segmentor.")
    if getattr(base_model.segmentor, "encoder", None) is None:
        raise ValueError("Segmentor has no encoder.")
    if getattr(base_model.segmentor, "pixel_decoder", None) is None:
        raise ValueError("Segmentor has no pixel_decoder.")
    if getattr(base_model.segmentor, "decoder", None) is None:
        raise ValueError("Segmentor has no decoder.")
    if not bool(getattr(base_model.segmentor, "use_cls", False)):
        raise ValueError(
            "Current segmentor has `use_cls=False`. "
            "Layer-wise probe requires class prediction branch (close_cls/open_cls)."
        )

    simple_fpn_template = base_model.segmentor.encoder.vision_backbone.convs
    segmentor_dtype = base_model.segmentor.dtype
    device = torch.device(get_device())

    heads: Dict[int, LayerProbeHead] = {}
    optimizers: Dict[int, torch.optim.Optimizer] = {}
    for idx, spec in enumerate(layer_specs):
        init_seed = (seed + idx * seed_stride) if reinit_weights else None
        head = LayerProbeHead(
            base_segmentor=base_model.segmentor,
            simple_fpn_template=simple_fpn_template,
            init_seed=init_seed,
            reinit_weights=reinit_weights,
        ).to(device=device)
        head = head.to(dtype=segmentor_dtype)
        heads[spec.norm_id] = head
        trainable_params = [param for param in head.parameters() if param.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                f"Probe head for layer {spec.raw_id}/{spec.norm_id} has no trainable parameters."
            )
        optimizers[spec.norm_id] = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )

    return heads, optimizers


def _set_optimizer_hparams(
    optimizers: Dict[int, torch.optim.Optimizer],
    lr: float,
    weight_decay: float,
) -> None:
    """Apply optimizer hyperparameters.

    Args:
        optimizers: Probe optimizers by layer.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        None.
    """
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = weight_decay


def _build_train_dataset(cfg: Config) -> Any:
    """Build training dataset from config.

    Args:
        cfg: Parsed config.

    Returns:
        Built train dataset.
    """
    if not hasattr(cfg, "train_dataloader") or cfg.train_dataloader is None:
        raise ValueError("Config has no `train_dataloader`; cannot train probe heads.")
    dataset_cfg = cfg.train_dataloader.get("dataset", None)
    if dataset_cfg is None:
        raise ValueError("`train_dataloader.dataset` is missing.")
    dataset_cfg = copy.deepcopy(dataset_cfg)
    data_name = dataset_cfg.get("data_name", None)
    if isinstance(data_name, str) and len(data_name) > 0:
        dataset_cfg["data_name"] = f"{data_name}__sweep_train"
    return BUILDER.build(dataset_cfg)


def _build_train_subset(
    train_dataset: Any,
    train_ratio: float,
    seed: int,
) -> Any:
    """Build deterministic train subset by ratio.

    Args:
        train_dataset: Full training dataset.
        train_ratio: Ratio in ``(0, 1]``.
        seed: Random seed for deterministic sampling.

    Returns:
        Full dataset or ``Subset`` depending on ratio.
    """
    if train_ratio <= 0.0 or train_ratio > 1.0:
        raise ValueError(f"`train_ratio` must be in (0, 1], got {train_ratio}.")

    dataset_len = len(train_dataset)
    if dataset_len <= 0:
        raise ValueError("Training dataset is empty.")
    if math.isclose(train_ratio, 1.0):
        return train_dataset

    subset_len = max(1, int(dataset_len * train_ratio))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_len)[:subset_len].tolist()
    return Subset(train_dataset, indices)


def _select_train_eval_entry(entries: Sequence[EvalEntry]) -> Optional[EvalEntry]:
    """Select one eval entry for train-time mIoU/pACC monitoring.

    Args:
        entries: All selected eval entries.

    Returns:
        Preferred semantic eval entry, or first entry when semantic one is unavailable.
    """
    if len(entries) == 0:
        return None
    for entry in entries:
        if "semantic" in str(entry.data_name).lower():
            return entry
    return entries[0]


def _evaluate_probe_heads_snapshot(
    base_model: Any,
    heads: Dict[int, LayerProbeHead],
    layer_specs: Sequence[LayerSpec],
    entry: EvalEntry,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    eval_oom_empty_cache: bool,
) -> Dict[int, Dict[str, float]]:
    """Run lightweight train-time eval snapshot and return per-layer metrics.

    Args:
        base_model: Built X-SAM model.
        heads: Probe heads by normalized layer id.
        layer_specs: Layer specs.
        entry: Eval entry used for train-time monitoring.
        batch_size: Eval batch size.
        num_workers: Eval dataloader workers.
        max_samples: Max evaluated samples. 0 means full dataset.
        eval_oom_empty_cache: Whether to clear CUDA cache on OOM.

    Returns:
        Mapping ``norm_layer_id -> {'miou': float, 'pacc': float}``.
    """
    device = torch.device(get_device())
    encoder = base_model.segmentor.encoder
    encoder_dtype = base_model.segmentor.dtype

    encoder_mode = encoder.training
    head_modes = {norm_id: head.training for norm_id, head in heads.items()}
    encoder.eval()
    for head in heads.values():
        head.eval()

    try:
        evaluators = {
            spec.norm_id: _build_evaluator(entry.evaluator_cfg, entry.dataset.metadata) for spec in layer_specs
        }
        dataloader = DataLoader(
            dataset=entry.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=xsam_collate_fn,
        )

        global_processed = 0
        for batch in dataloader:
            if max_samples > 0 and global_processed >= max_samples:
                break

            data_dict = data_dict_to_device(batch["data_dict"], device=device, dtype=encoder_dtype)
            data_samples = data_sample_to_device(batch["data_samples"], device=device)
            image_infos = data_samples.metainfo["image_infos"]
            image_sizes = data_samples.metainfo.get("image_sizes", None)
            scaled_sizes = data_samples.metainfo.get("scaled_sizes", image_sizes)
            sampled_labels = getattr(data_samples, "sampled_labels", None)
            vprompt_masks = data_dict.get("vprompt_masks", None)
            batch_size_real = len(image_infos)

            try:
                pixel_values = _get_pixel_values_from_batch(data_dict)
                with torch.no_grad():
                    encoder_outputs = _forward_encoder_with_trunk_capture(
                        encoder=encoder,
                        pixel_values=pixel_values,
                        trunk_select_layers=[spec.norm_id for spec in layer_specs],
                        encoder_dtype=encoder_dtype,
                    )
                    feature_map_by_layer = _extract_trunk_feature_map(encoder_outputs, layer_specs)
            except Exception as exc:
                if eval_oom_empty_cache and "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                global_processed += batch_size_real
                continue

            for spec in layer_specs:
                norm_id = spec.norm_id
                head = heads[norm_id]
                try:
                    with torch.no_grad():
                        seg_outputs = head(
                            trunk_feature=feature_map_by_layer[norm_id],
                            seg_embeddings=None,
                            cond_embeddings=None,
                            embed_masks=None,
                            cond_lens=None,
                            mask_labels=None,
                            class_labels=None,
                            output_hidden_states=False,
                            output_auxiliary_logits=False,
                            output_attentions=False,
                            return_dict=True,
                        )
                    pred_outputs = entry.dataset.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_sizes,
                        scaled_sizes=scaled_sizes,
                        metadata=entry.dataset.metadata,
                        sampled_labels=sampled_labels,
                        vprompt_masks=vprompt_masks,
                    )
                    evaluators[norm_id].process(image_infos, pred_outputs)
                except Exception as exc:
                    if eval_oom_empty_cache and "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            global_processed += batch_size_real

        metrics_by_layer: Dict[int, Dict[str, float]] = {}
        for spec in layer_specs:
            norm_id = spec.norm_id
            metrics = _extract_metrics(evaluators[norm_id])
            metrics_by_layer[norm_id] = {
                "miou": _safe_to_float(metrics.get("miou", math.nan)),
                "pacc": _safe_to_float(metrics.get("pacc", math.nan)),
            }
        return metrics_by_layer
    finally:
        encoder.train(encoder_mode)
        for norm_id, mode in head_modes.items():
            heads[norm_id].train(mode)


def _to_cpu_obj(value: Any) -> Any:
    """Recursively move tensors to CPU.

    Args:
        value: Any nested python object.

    Returns:
        Object with all tensors moved to CPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_obj(item) for item in value)
    return value


def _format_eta(seconds: float) -> str:
    """Format ETA seconds into readable text.

    Args:
        seconds: Remaining seconds.

    Returns:
        ETA string.
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    return str(timedelta(seconds=int(seconds)))


def _resolve_max_keep_ckpts(max_save: int) -> Optional[int]:
    """Resolve checkpoint retention count from max_save policy.

    Args:
        max_save: Retention policy integer.

    Returns:
        Number of step checkpoints to keep, or ``None`` for unlimited.
    """
    if max_save == -1:
        return None
    if max_save < -1:
        return abs(max_save)
    return max(max_save, 0)


def _prune_step_checkpoints(ckpt_dir: Path, max_save: int) -> None:
    """Prune old step checkpoints according to retention policy.

    Args:
        ckpt_dir: Checkpoint directory.
        max_save: Retention policy integer.

    Returns:
        None.
    """
    keep_count = _resolve_max_keep_ckpts(max_save)
    if keep_count is None:
        return

    step_ckpts: List[Tuple[int, Path]] = []
    for ckpt_path in ckpt_dir.glob("step_*.pt"):
        stem = ckpt_path.stem
        try:
            step_id = int(stem.split("_")[-1])
        except ValueError:
            continue
        step_ckpts.append((step_id, ckpt_path))
    step_ckpts.sort(key=lambda x: x[0])

    if keep_count <= 0:
        to_delete = step_ckpts
    elif len(step_ckpts) > keep_count:
        to_delete = step_ckpts[:-keep_count]
    else:
        to_delete = []

    for _, ckpt_path in to_delete:
        with contextlib.suppress(FileNotFoundError):
            ckpt_path.unlink()


def _save_probe_checkpoint(
    run_dir: Path,
    layer_specs: Sequence[LayerSpec],
    heads: Dict[int, LayerProbeHead],
    optimizers: Dict[int, torch.optim.Optimizer],
    global_step: int,
    train_epochs: int,
    steps_per_epoch: int,
    running_loss: Dict[int, float],
    running_count: Dict[int, int],
    running_components: Dict[int, Dict[str, float]],
    loss_keys: Sequence[str],
    max_save: int,
) -> str:
    """Save resume checkpoint and latest exported FPN/M2F weights.

    Args:
        run_dir: Sweep run directory.
        layer_specs: Layer specs.
        heads: Probe heads.
        optimizers: Probe optimizers.
        global_step: Current global step.
        train_epochs: Target train epochs.
        steps_per_epoch: Steps per epoch.
        running_loss: Running total loss sums.
        running_count: Running sample counts.
        running_components: Running component loss sums.
        loss_keys: Observed loss keys.
        max_save: Checkpoint retention policy.

    Returns:
        Saved step checkpoint path.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_weights_dir = ckpt_dir / "latest_weights"
    latest_weights_dir.mkdir(parents=True, exist_ok=True)

    for spec in layer_specs:
        norm_id = spec.norm_id
        raw_id = spec.raw_id
        head = heads[norm_id]

        fpn_path = latest_weights_dir / f"layer_{raw_id}_{norm_id}_fpn.bin"
        torch.save(_to_cpu_obj(head.simple_fpn.state_dict()), fpn_path)

        m2f_payload = {
            "pixel_decoder": _to_cpu_obj(head.pixel_decoder.state_dict()),
            "decoder": _to_cpu_obj(head.decoder.state_dict()),
        }
        if head.class_predictor is not None:
            m2f_payload["class_predictor"] = _to_cpu_obj(head.class_predictor.state_dict())
        if head.logit_scale is not None:
            m2f_payload["logit_scale"] = _to_cpu_obj(head.logit_scale)
        m2f_path = latest_weights_dir / f"layer_{raw_id}_{norm_id}_m2f.bin"
        torch.save(m2f_payload, m2f_path)

    ckpt_state = {
        "global_step": int(global_step),
        "train_epochs": int(train_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "layer_specs": [{"raw_id": spec.raw_id, "norm_id": spec.norm_id} for spec in layer_specs],
        "heads": {int(norm_id): _to_cpu_obj(head.state_dict()) for norm_id, head in heads.items()},
        "optimizers": {int(norm_id): _to_cpu_obj(opt.state_dict()) for norm_id, opt in optimizers.items()},
        "running_loss": {int(key): float(val) for key, val in running_loss.items()},
        "running_count": {int(key): int(val) for key, val in running_count.items()},
        "running_components": {
            int(key): {str(k): float(v) for k, v in comp.items()} for key, comp in running_components.items()
        },
        "loss_keys": list(loss_keys),
    }
    step_ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
    torch.save(ckpt_state, step_ckpt_path)

    latest_ckpt = ckpt_dir / "latest.pt"
    shutil.copy2(step_ckpt_path, latest_ckpt)
    _prune_step_checkpoints(ckpt_dir=ckpt_dir, max_save=max_save)
    return str(step_ckpt_path)


def _resolve_resume_ckpt_path(
    run_dir: Path,
    resume: bool,
    resume_ckpt: Optional[str],
) -> Optional[Path]:
    """Resolve resume checkpoint path.

    Args:
        run_dir: Sweep run directory.
        resume: Whether resume is enabled.
        resume_ckpt: Optional explicit checkpoint path.

    Returns:
        Resolved checkpoint path or ``None``.
    """
    if not resume:
        return None

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.is_absolute():
            project_relative = (PROJECT_ROOT / ckpt_path).resolve()
            run_relative = (run_dir / ckpt_path).resolve()
            if project_relative.exists():
                ckpt_path = project_relative
            else:
                ckpt_path = run_relative
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    latest_step_ckpt: Optional[Tuple[int, Path]] = None
    for ckpt_file in ckpt_dir.glob("step_*.pt"):
        stem = ckpt_file.stem
        try:
            step_id = int(stem.split("_")[-1])
        except ValueError:
            continue
        if latest_step_ckpt is None or step_id > latest_step_ckpt[0]:
            latest_step_ckpt = (step_id, ckpt_file)
    if latest_step_ckpt is not None:
        return latest_step_ckpt[1]

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        return latest_ckpt
    return None


def _load_probe_checkpoint(
    ckpt_path: Path,
    layer_specs: Sequence[LayerSpec],
    heads: Dict[int, LayerProbeHead],
    optimizers: Dict[int, torch.optim.Optimizer],
    expected_steps_per_epoch: int,
    expected_train_epochs: int,
) -> Dict[str, Any]:
    """Load probe checkpoint for resume.

    Args:
        ckpt_path: Checkpoint path.
        layer_specs: Current layer specs.
        heads: Probe heads.
        optimizers: Probe optimizers.
        expected_steps_per_epoch: Current steps per epoch.
        expected_train_epochs: Current train epochs.

    Returns:
        Restored state dictionary.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    expected_norm_ids = [spec.norm_id for spec in layer_specs]
    stored_specs = checkpoint.get("layer_specs", [])
    stored_norm_ids = [int(item["norm_id"]) for item in stored_specs]
    if stored_norm_ids != expected_norm_ids:
        raise ValueError(
            "Resume checkpoint layer specs mismatch. "
            f"expected={expected_norm_ids}, stored={stored_norm_ids}, ckpt={ckpt_path}"
        )

    for norm_id in expected_norm_ids:
        heads[norm_id].load_state_dict(checkpoint["heads"][norm_id], strict=True)
        optimizers[norm_id].load_state_dict(checkpoint["optimizers"][norm_id])

    stored_steps_per_epoch = int(checkpoint.get("steps_per_epoch", expected_steps_per_epoch))
    stored_train_epochs = int(checkpoint.get("train_epochs", expected_train_epochs))
    if stored_steps_per_epoch != expected_steps_per_epoch:
        print_log(
            (
                "Resume warning: `steps_per_epoch` changed "
                f"(stored={stored_steps_per_epoch}, current={expected_steps_per_epoch})."
            ),
            logger="current",
        )
    if stored_train_epochs != expected_train_epochs:
        print_log(
            (
                "Resume warning: `train_epochs` changed "
                f"(stored={stored_train_epochs}, current={expected_train_epochs})."
            ),
            logger="current",
        )

    return {
        "global_step": int(checkpoint.get("global_step", 0)),
        "running_loss": {int(k): float(v) for k, v in checkpoint.get("running_loss", {}).items()},
        "running_count": {int(k): int(v) for k, v in checkpoint.get("running_count", {}).items()},
        "running_components": {
            int(k): {str(loss_key): float(loss_val) for loss_key, loss_val in comp.items()}
            for k, comp in checkpoint.get("running_components", {}).items()
        },
        "loss_keys": list(checkpoint.get("loss_keys", [])),
    }


def _train_probe_heads(
    base_model: Any,
    heads: Dict[int, LayerProbeHead],
    optimizers: Dict[int, torch.optim.Optimizer],
    layer_specs: Sequence[LayerSpec],
    train_dataset: Any,
    train_epochs: int,
    train_batch_size: int,
    grad_accum_steps: int,
    train_num_workers: int,
    run_dir: Path,
    save_steps: int,
    max_save: int,
    resume: bool,
    resume_ckpt: Optional[str],
    use_tqdm: bool,
    log_interval: int,
    train_eval_entry: Optional[EvalEntry],
    train_eval_batch_size: int,
    train_eval_num_workers: int,
    train_eval_interval: int,
    train_eval_max_samples: int,
    early_stop_patience_steps: int,
    early_stop_miou_eps: float,
    eval_oom_empty_cache: bool,
) -> ProbeTrainStats:
    """Train all probe heads with one trunk forward per batch.

    Args:
        base_model: Built X-SAM model.
        heads: Probe heads keyed by normalized layer id.
        optimizers: Probe optimizers keyed by normalized layer id.
        layer_specs: Layer specs.
        train_dataset: Training dataset.
        train_epochs: Number of training epochs.
        train_batch_size: Batch size.
        grad_accum_steps: Gradient accumulation steps.
        train_num_workers: DataLoader workers.
        run_dir: Sweep run directory for checkpoints.
        save_steps: Save checkpoint every N steps.
        max_save: Checkpoint retention policy.
        resume: Whether to resume from checkpoint.
        resume_ckpt: Optional explicit resume checkpoint path.
        use_tqdm: Whether to show tqdm progress bar.
        log_interval: MMEngine logging interval.
        train_eval_entry: Optional eval entry used for train-time mIoU/pACC monitoring.
        train_eval_batch_size: Batch size for train-time eval snapshots.
        train_eval_num_workers: DataLoader workers for train-time eval snapshots.
        train_eval_interval: Train-time eval interval in steps. 0 disables snapshots.
        train_eval_max_samples: Max samples per train-time eval snapshot. 0 means full dataset.
        early_stop_patience_steps: Early-stop patience in steps without sufficient mIoU gain.
        early_stop_miou_eps: Minimum mIoU gain (percentage points) to reset patience.
        eval_oom_empty_cache: Whether to clear CUDA cache on OOM during train-time eval.

    Returns:
        Probe training statistics.
    """
    if train_epochs <= 0:
        return ProbeTrainStats(
            mean_total_loss={spec.norm_id: math.nan for spec in layer_specs},
            mean_loss_components={spec.norm_id: {} for spec in layer_specs},
            loss_keys=[],
            trace_rows=[],
        )
    if grad_accum_steps <= 0:
        raise ValueError(f"`grad_accum_steps` must be >= 1, got {grad_accum_steps}.")

    device = torch.device(get_device())
    encoder = base_model.segmentor.encoder
    encoder_dtype = base_model.segmentor.dtype

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=xsam_collate_fn,
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch <= 0:
        raise ValueError("Probe train dataloader has zero steps per epoch.")
    total_steps = train_epochs * steps_per_epoch

    for head in heads.values():
        head.train()

    running_loss: Dict[int, torch.Tensor] = {
        spec.norm_id: torch.zeros((), device=device, dtype=torch.float32) for spec in layer_specs
    }
    running_count: Dict[int, int] = {spec.norm_id: 0 for spec in layer_specs}
    running_components: Dict[int, Dict[str, torch.Tensor]] = {spec.norm_id: {} for spec in layer_specs}
    all_loss_keys: set[str] = set()
    trace_rows: List[Dict[str, Any]] = []
    norm_to_raw = {spec.norm_id: spec.raw_id for spec in layer_specs}

    global_step = 0
    resume_state = None
    resume_path = _resolve_resume_ckpt_path(run_dir=run_dir, resume=resume, resume_ckpt=resume_ckpt)
    if resume_path is not None:
        resume_state = _load_probe_checkpoint(
            ckpt_path=resume_path,
            layer_specs=layer_specs,
            heads=heads,
            optimizers=optimizers,
            expected_steps_per_epoch=steps_per_epoch,
            expected_train_epochs=train_epochs,
        )
        loaded_global_step = int(resume_state["global_step"])
        if loaded_global_step > total_steps:
            print_log(
                (
                    f"Resume warning: checkpoint global_step={loaded_global_step} "
                    f"> total_steps={total_steps}, clamp to total_steps."
                ),
                logger="current",
            )
        global_step = min(loaded_global_step, total_steps)
        for spec in layer_specs:
            norm_id = spec.norm_id
            running_loss[norm_id] = torch.tensor(
                float(resume_state["running_loss"].get(norm_id, 0.0)),
                device=device,
                dtype=torch.float32,
            )
            running_count[norm_id] = int(resume_state["running_count"].get(norm_id, 0))
            running_components[norm_id] = {
                str(loss_key): torch.tensor(float(loss_val), device=device, dtype=torch.float32)
                for loss_key, loss_val in resume_state["running_components"].get(norm_id, {}).items()
            }
        all_loss_keys.update(str(key) for key in resume_state["loss_keys"])
        print_log(
            f"Resume enabled: loaded checkpoint {resume_path}, global_step={global_step}/{total_steps}",
            logger="current",
        )
    elif resume:
        print_log(
            "Resume enabled: no checkpoint found in current run directory, start from scratch.",
            logger="current",
        )

    progress_bar: Optional[Any] = None
    if use_tqdm:
        progress_bar = tqdm(total=total_steps, desc="ProbeTrain", leave=False, dynamic_ncols=True, initial=global_step)

    resume_base_step = global_step
    last_saved_step = global_step
    train_start_time = time.perf_counter()

    start_epoch = global_step // steps_per_epoch
    start_step_in_epoch = global_step % steps_per_epoch

    train_eval_enabled = train_eval_entry is not None and train_eval_interval > 0
    best_train_eval_miou = math.nan
    best_train_eval_step = global_step
    best_train_eval_norm_id: Optional[int] = None
    early_stop_triggered = False
    accum_counter = 0

    if train_eval_enabled:
        print_log(
            (
                f"Train-time eval enabled: task={train_eval_entry.data_name}, "
                f"interval={train_eval_interval}, max_samples={train_eval_max_samples}, "
                f"early_stop_patience_steps={early_stop_patience_steps}, "
                f"early_stop_miou_eps={early_stop_miou_eps:.4f}"
            ),
            logger="current",
        )

    for epoch in range(start_epoch, train_epochs):
        for step_in_epoch, batch in enumerate(dataloader):
            if epoch == start_epoch and step_in_epoch < start_step_in_epoch:
                continue
            if global_step >= total_steps:
                break
            global_step += 1
            if accum_counter == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

            try:
                data_dict = data_dict_to_device(batch["data_dict"], device=device, dtype=encoder_dtype)
                data_samples = data_sample_to_device(batch["data_samples"], device=device)

                pixel_values = _get_pixel_values_from_batch(data_dict)
                with torch.no_grad():
                    encoder_outputs = _forward_encoder_with_trunk_capture(
                        encoder=encoder,
                        pixel_values=pixel_values,
                        trunk_select_layers=[spec.norm_id for spec in layer_specs],
                        encoder_dtype=encoder_dtype,
                    )
                    feature_map_by_layer = _extract_trunk_feature_map(encoder_outputs, layer_specs)

                mask_labels = getattr(data_samples, "mask_labels", None)
                class_labels = getattr(data_samples, "class_labels", None)
                if mask_labels is None or class_labels is None:
                    raise ValueError("Training batch does not contain `mask_labels` or `class_labels`.")

                for spec in layer_specs:
                    norm_id = spec.norm_id
                    head = heads[norm_id]

                    outputs = head(
                        trunk_feature=feature_map_by_layer[norm_id],
                        seg_embeddings=None,
                        cond_embeddings=None,
                        embed_masks=None,
                        cond_lens=None,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                        output_hidden_states=False,
                        output_auxiliary_logits=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    if outputs.loss is None:
                        continue
                    loss = outputs.loss
                    (loss / float(grad_accum_steps)).backward()

                    running_loss[norm_id] = running_loss[norm_id] + loss.detach().to(dtype=torch.float32)
                    running_count[norm_id] += 1

                    loss_dict = outputs.loss_dict or {}
                    for loss_key, loss_tensor in loss_dict.items():
                        current_sum = running_components[norm_id].get(
                            loss_key,
                            torch.zeros((), device=device, dtype=torch.float32),
                        )
                        running_components[norm_id][loss_key] = current_sum + loss_tensor.detach().to(dtype=torch.float32)
                        all_loss_keys.add(loss_key)
            except torch.OutOfMemoryError as exc:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_log(
                    (
                        f"[ProbeTrainOOM] epoch={epoch}/{train_epochs}, step={global_step}/{total_steps}, "
                        "reset accumulated grads and skip current batch. "
                        f"error={exc}"
                    ),
                    logger="current",
                )
                if progress_bar is not None and hasattr(progress_bar, "update"):
                    progress_bar.update(1)
                continue

            accum_counter += 1
            if accum_counter >= grad_accum_steps:
                for optimizer in optimizers.values():
                    optimizer.step()
                accum_counter = 0

            if log_interval > 0 and global_step % log_interval == 0:
                done_steps = max(global_step - resume_base_step, 1)
                elapsed = time.perf_counter() - train_start_time
                avg_step_time = elapsed / done_steps
                eta_seconds = avg_step_time * max(total_steps - global_step, 0)
                eta_text = _format_eta(eta_seconds)

                desc_items = []
                for spec in layer_specs:
                    norm_id = spec.norm_id
                    count = max(running_count[norm_id], 1)
                    avg_total = float(running_loss[norm_id].item()) / count
                    component_avg = {
                        key: float(value.item()) / count for key, value in running_components[norm_id].items()
                    }
                    core_parts = []
                    for loss_key in ["loss_mask", "loss_dice", "loss_cls"]:
                        if loss_key in component_avg:
                            core_parts.append(f"{loss_key}={component_avg[loss_key]:.3f}")
                    core_text = ", ".join(core_parts)
                    if core_text:
                        desc_items.append(f"L{spec.raw_id}/{spec.norm_id}:total={avg_total:.3f}, {core_text}")
                    else:
                        desc_items.append(f"L{spec.raw_id}/{spec.norm_id}:total={avg_total:.3f}")
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "layer": norm_to_raw[norm_id],
                        "normalized_layer": norm_id,
                        "total_loss": avg_total,
                    }
                    row.update(component_avg)
                    trace_rows.append(row)
                desc_text = ", ".join(desc_items)
                print_log(
                    (
                        f"[ProbeTrain] epoch={epoch}/{train_epochs}, "
                        f"step={global_step}/{total_steps}, eta={eta_text}, {desc_text}"
                    ),
                    logger="current",
                )
                if progress_bar is not None and hasattr(progress_bar, "set_postfix_str"):
                    progress_bar.set_postfix_str(f"eta={eta_text}; {desc_text}")

            if train_eval_enabled and global_step % train_eval_interval == 0:
                snapshot_metrics = _evaluate_probe_heads_snapshot(
                    base_model=base_model,
                    heads=heads,
                    layer_specs=layer_specs,
                    entry=train_eval_entry,
                    batch_size=train_eval_batch_size,
                    num_workers=train_eval_num_workers,
                    max_samples=train_eval_max_samples,
                    eval_oom_empty_cache=eval_oom_empty_cache,
                )
                current_best_norm_id: Optional[int] = None
                current_best_miou = math.nan
                current_best_pacc = math.nan
                for spec in layer_specs:
                    norm_id = spec.norm_id
                    layer_miou = _safe_to_float(snapshot_metrics.get(norm_id, {}).get("miou", math.nan))
                    layer_pacc = _safe_to_float(snapshot_metrics.get(norm_id, {}).get("pacc", math.nan))
                    if not math.isfinite(layer_miou):
                        continue
                    if current_best_norm_id is None or layer_miou > current_best_miou:
                        current_best_norm_id = norm_id
                        current_best_miou = layer_miou
                        current_best_pacc = layer_pacc

                if current_best_norm_id is None:
                    print_log(
                        f"[TrainEval] step={global_step}/{total_steps}, task={train_eval_entry.data_name}, no valid mIoU.",
                        logger="current",
                    )
                else:
                    improved = (not math.isfinite(best_train_eval_miou)) or (
                        current_best_miou - best_train_eval_miou > early_stop_miou_eps
                    )
                    if improved:
                        best_train_eval_miou = current_best_miou
                        best_train_eval_step = global_step
                        best_train_eval_norm_id = current_best_norm_id

                    steps_since_best = global_step - best_train_eval_step
                    print_log(
                        (
                            f"[TrainEval] step={global_step}/{total_steps}, task={train_eval_entry.data_name}, "
                            f"best_now=L{norm_to_raw[current_best_norm_id]}/{current_best_norm_id}, "
                            f"mIoU={current_best_miou:.4f}, pACC={current_best_pacc:.4f}, "
                            f"best_global=L{norm_to_raw[best_train_eval_norm_id]}/{best_train_eval_norm_id} "
                            f"@{best_train_eval_step}, best_mIoU={best_train_eval_miou:.4f}, "
                            f"delta_steps={steps_since_best}"
                        ),
                        logger="current",
                    )

                    if early_stop_patience_steps > 0 and steps_since_best >= early_stop_patience_steps:
                        early_stop_triggered = True
                        print_log(
                            (
                                f"[EarlyStop] Triggered at step={global_step}/{total_steps}: "
                                f"no mIoU gain > {early_stop_miou_eps:.4f} for "
                                f"{early_stop_patience_steps} steps."
                            ),
                            logger="current",
                        )

            if progress_bar is not None and hasattr(progress_bar, "update"):
                progress_bar.update(1)

            if save_steps > 0 and global_step % save_steps == 0:
                ckpt_running_loss = {layer_id: float(loss_sum.item()) for layer_id, loss_sum in running_loss.items()}
                ckpt_running_components = {
                    layer_id: {loss_key: float(loss_sum.item()) for loss_key, loss_sum in comp.items()}
                    for layer_id, comp in running_components.items()
                }
                saved_ckpt = _save_probe_checkpoint(
                    run_dir=run_dir,
                    layer_specs=layer_specs,
                    heads=heads,
                    optimizers=optimizers,
                    global_step=global_step,
                    train_epochs=train_epochs,
                    steps_per_epoch=steps_per_epoch,
                    running_loss=ckpt_running_loss,
                    running_count=running_count,
                    running_components=ckpt_running_components,
                    loss_keys=sorted(all_loss_keys),
                    max_save=max_save,
                )
                last_saved_step = global_step
                print_log(
                    f"[ProbeCkpt] step={global_step}/{total_steps}, saved={saved_ckpt}",
                    logger="current",
                )
            if early_stop_triggered:
                break
        if global_step >= total_steps or early_stop_triggered:
            break

    if global_step > 0 and accum_counter > 0:
        for optimizer in optimizers.values():
            optimizer.step()
        print_log(
            (
                f"[ProbeTrain] Applied residual optimizer step with "
                f"{accum_counter}/{grad_accum_steps} accumulated micro-steps."
            ),
            logger="current",
        )

    mean_total_loss = {}
    mean_loss_components = {}
    for spec in layer_specs:
        norm_id = spec.norm_id
        if running_count[norm_id] == 0:
            mean_total_loss[norm_id] = math.nan
            mean_loss_components[norm_id] = {}
        else:
            count = float(running_count[norm_id])
            mean_total_loss[norm_id] = float(running_loss[norm_id].item()) / count
            mean_loss_components[norm_id] = {
                key: float(value.item()) / count for key, value in running_components[norm_id].items()
            }

    if save_steps > 0 and global_step > 0 and global_step != last_saved_step:
        ckpt_running_loss = {layer_id: float(loss_sum.item()) for layer_id, loss_sum in running_loss.items()}
        ckpt_running_components = {
            layer_id: {loss_key: float(loss_sum.item()) for loss_key, loss_sum in comp.items()}
            for layer_id, comp in running_components.items()
        }
        saved_ckpt = _save_probe_checkpoint(
            run_dir=run_dir,
            layer_specs=layer_specs,
            heads=heads,
            optimizers=optimizers,
            global_step=global_step,
            train_epochs=train_epochs,
            steps_per_epoch=steps_per_epoch,
            running_loss=ckpt_running_loss,
            running_count=running_count,
            running_components=ckpt_running_components,
            loss_keys=sorted(all_loss_keys),
            max_save=max_save,
        )
        print_log(
            f"[ProbeCkpt] final step={global_step}/{total_steps}, saved={saved_ckpt}",
            logger="current",
        )

    if progress_bar is not None and hasattr(progress_bar, "close"):
        progress_bar.close()

    return ProbeTrainStats(
        mean_total_loss=mean_total_loss,
        mean_loss_components=mean_loss_components,
        loss_keys=sorted(all_loss_keys),
        trace_rows=trace_rows,
    )


def _evaluate_probe_heads(
    base_model: Any,
    heads: Dict[int, LayerProbeHead],
    layer_specs: Sequence[LayerSpec],
    entries: Sequence[EvalEntry],
    batch_size: int,
    num_workers: int,
    max_samples_per_task: int,
    use_tqdm: bool,
    log_interval: int,
    train_loss_by_layer: Dict[int, float],
    train_loss_components_by_layer: Dict[int, Dict[str, float]],
    eval_fail_fast: bool,
    eval_fail_ratio_threshold: float,
    eval_fail_check_min_samples: int,
    eval_oom_empty_cache: bool,
    eval_log_cuda_mem: bool,
) -> List[Dict[str, Any]]:
    """Evaluate all probe heads on selected datasets.

    Args:
        base_model: Built X-SAM model.
        heads: Probe heads by normalized layer id.
        layer_specs: Layer specs.
        entries: Evaluation entries.
        batch_size: Eval dataloader batch size.
        num_workers: Eval dataloader workers.
        max_samples_per_task: Max evaluated samples per task. 0 means full.
        use_tqdm: Whether to show tqdm progress bar.
        log_interval: MMEngine logging interval.
        train_loss_by_layer: Mean train loss by layer.
        train_loss_components_by_layer: Mean train component losses by layer.
        eval_fail_fast: Whether to stop eval early when fail ratio is high.
        eval_fail_ratio_threshold: Per-layer fail ratio threshold for fail-fast.
        eval_fail_check_min_samples: Minimum processed samples before fail-fast check.
        eval_oom_empty_cache: Whether to clear CUDA cache on OOM.
        eval_log_cuda_mem: Whether to include CUDA memory usage in eval logs.

    Returns:
        CSV row list.
    """
    device = torch.device(get_device())
    encoder = base_model.segmentor.encoder
    encoder_dtype = base_model.segmentor.dtype

    encoder.eval()
    for head in heads.values():
        head.eval()

    all_rows: List[Dict[str, Any]] = []
    for entry in entries:
        norm_to_raw = {spec.norm_id: spec.raw_id for spec in layer_specs}
        evaluators = {
            spec.norm_id: _build_evaluator(entry.evaluator_cfg, entry.dataset.metadata) for spec in layer_specs
        }
        processed_count = {spec.norm_id: 0 for spec in layer_specs}
        failed_count = {spec.norm_id: 0 for spec in layer_specs}
        error_by_layer: Dict[int, str] = {spec.norm_id: "" for spec in layer_specs}

        dataloader = DataLoader(
            dataset=entry.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=xsam_collate_fn,
        )

        global_processed = 0
        eval_iter: Any = dataloader
        if use_tqdm:
            eval_iter = tqdm(dataloader, desc=f"Eval:{entry.data_name}", leave=False, dynamic_ncols=True)
        for batch_idx, batch in enumerate(eval_iter, start=1):
            if max_samples_per_task > 0 and global_processed >= max_samples_per_task:
                break

            data_dict = data_dict_to_device(batch["data_dict"], device=device, dtype=encoder_dtype)
            data_samples = data_sample_to_device(batch["data_samples"], device=device)

            image_infos = data_samples.metainfo["image_infos"]
            image_sizes = data_samples.metainfo.get("image_sizes", None)
            scaled_sizes = data_samples.metainfo.get("scaled_sizes", image_sizes)
            sampled_labels = getattr(data_samples, "sampled_labels", None)
            vprompt_masks = data_dict.get("vprompt_masks", None)
            batch_size_real = len(image_infos)

            try:
                pixel_values = _get_pixel_values_from_batch(data_dict)
                with torch.no_grad():
                    encoder_outputs = _forward_encoder_with_trunk_capture(
                        encoder=encoder,
                        pixel_values=pixel_values,
                        trunk_select_layers=[spec.norm_id for spec in layer_specs],
                        encoder_dtype=encoder_dtype,
                    )
                    feature_map_by_layer = _extract_trunk_feature_map(encoder_outputs, layer_specs)
            except Exception as exc:
                error_text = f"{exc.__class__.__name__}: {exc}"
                for spec in layer_specs:
                    failed_count[spec.norm_id] += batch_size_real
                    processed_count[spec.norm_id] += batch_size_real
                    if not error_by_layer[spec.norm_id]:
                        error_by_layer[spec.norm_id] = error_text
                if eval_oom_empty_cache and "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print_log(
                    f"[EvalBatchError] task={entry.data_name}, batch={batch_idx}, error={error_text}",
                    logger="current",
                )
                global_processed += batch_size_real
                continue

            for spec in layer_specs:
                norm_id = spec.norm_id
                head = heads[norm_id]
                try:
                    with torch.no_grad():
                        seg_outputs = head(
                            trunk_feature=feature_map_by_layer[norm_id],
                            seg_embeddings=None,
                            cond_embeddings=None,
                            embed_masks=None,
                            cond_lens=None,
                            mask_labels=None,
                            class_labels=None,
                            output_hidden_states=False,
                            output_auxiliary_logits=False,
                            output_attentions=False,
                            return_dict=True,
                        )

                    pred_outputs = entry.dataset.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_sizes,
                        scaled_sizes=scaled_sizes,
                        metadata=entry.dataset.metadata,
                        sampled_labels=sampled_labels,
                        vprompt_masks=vprompt_masks,
                    )
                    evaluators[norm_id].process(image_infos, pred_outputs)
                except Exception as exc:
                    failed_count[norm_id] += batch_size_real
                    if not error_by_layer[norm_id]:
                        error_by_layer[norm_id] = f"{exc.__class__.__name__}: {exc}"
                        print_log(
                            f"[EvalLayerError] task={entry.data_name}, "
                            f"layer={norm_to_raw[norm_id]}/{norm_id}, "
                            f"batch={batch_idx}, error={error_by_layer[norm_id]}",
                            logger="current",
                        )
                    if eval_oom_empty_cache and "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                processed_count[norm_id] += batch_size_real

            global_processed += batch_size_real

            if log_interval > 0 and batch_idx % log_interval == 0:
                fail_ratios = {
                    norm_id: (failed_count[norm_id] / processed_count[norm_id])
                    for norm_id in failed_count
                    if processed_count[norm_id] > 0
                }
                worst_norm_id: Optional[int] = None
                if fail_ratios:
                    worst_norm_id = max(fail_ratios, key=fail_ratios.get)
                    worst_fail_ratio = fail_ratios[worst_norm_id]
                    worst_text = (
                        f"L{norm_to_raw[worst_norm_id]}/{worst_norm_id}"
                        f" fail_ratio={worst_fail_ratio:.3f}"
                        f" ({failed_count[worst_norm_id]}/{processed_count[worst_norm_id]})"
                    )
                else:
                    worst_fail_ratio = 0.0
                    worst_text = "n/a"

                health_text = f"[EvalHealth] task={entry.data_name}, batch={batch_idx}, worst={worst_text}"
                if eval_log_cuda_mem:
                    health_text += f", {_format_cuda_memory(device)}"
                print_log(health_text, logger="current")

                print_log(
                    f"[EvalProgress] task={entry.data_name}, batch={batch_idx}, processed={global_processed}",
                    logger="current",
                )
                if (
                    eval_fail_fast
                    and global_processed >= eval_fail_check_min_samples
                    and worst_fail_ratio >= eval_fail_ratio_threshold
                    and worst_norm_id is not None
                ):
                    raise RuntimeError(
                        "Eval fail-fast triggered: "
                        f"task={entry.data_name}, batch={batch_idx}, "
                        f"layer={norm_to_raw[worst_norm_id]}/{worst_norm_id}, "
                        f"fail_ratio={worst_fail_ratio:.3f}, "
                        f"threshold={eval_fail_ratio_threshold:.3f}"
                    )

        for spec in layer_specs:
            norm_id = spec.norm_id
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "layer": spec.raw_id,
                "normalized_layer": norm_id,
                "task_type": entry.task_type,
                "data_name": entry.data_name,
                "train_loss": _safe_to_float(train_loss_by_layer.get(norm_id, math.nan)),
                "train_loss_items": json.dumps(
                    train_loss_components_by_layer.get(norm_id, {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "miou": math.nan,
                "giou": math.nan,
                "ciou": math.nan,
                "processed": processed_count[norm_id],
                "failed": failed_count[norm_id],
                "status": "ok",
                "error": error_by_layer.get(norm_id, ""),
            }
            try:
                evaluators[norm_id].evaluate()
                metrics = _extract_metrics(evaluators[norm_id])
                row["miou"] = _safe_to_float(metrics["miou"])
                row["giou"] = _safe_to_float(metrics["giou"])
                row["ciou"] = _safe_to_float(metrics["ciou"])
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{exc.__class__.__name__}: {exc}"
                print_log(
                    f"Eval error at layer={spec.raw_id}/{norm_id}, task={entry.data_name}\n{traceback.format_exc()}",
                    logger="current",
                )

            if row["status"] == "ok":
                if row["processed"] > 0 and row["failed"] == row["processed"]:
                    row["status"] = "error"
                elif row["failed"] > 0:
                    row["status"] = "partial"

            all_rows.append(row)
            print_log(
                f"[Result] layer={spec.raw_id}/{norm_id}, task={entry.data_name}, "
                f"mIoU={row['miou']}, gIoU={row['giou']}, cIoU={row['ciou']}, "
                f"train_loss={row['train_loss']}, status={row['status']}",
                logger="current",
            )

    return all_rows


def _write_csv(csv_path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows to CSV.

    Args:
        csv_path: Output path.
        rows: Row list.

    Returns:
        None.
    """
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    headers = [
        "timestamp",
        "layer",
        "normalized_layer",
        "task_type",
        "data_name",
        "train_loss",
        "train_loss_items",
        "miou",
        "giou",
        "ciou",
        "processed",
        "failed",
        "status",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_train_loss_trace_csv(
    csv_path: str,
    trace_rows: Sequence[Dict[str, Any]],
    loss_keys: Sequence[str],
) -> None:
    """Write per-step per-layer train loss trace CSV.

    Args:
        csv_path: Output trace CSV path.
        trace_rows: Trace row list.
        loss_keys: Loss component keys.

    Returns:
        None.
    """
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    ordered_keys = sorted(loss_keys)
    headers = ["step", "epoch", "layer", "normalized_layer", "total_loss"] + ordered_keys

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in trace_rows:
            out_row = {key: row.get(key, math.nan) for key in headers}
            writer.writerow(out_row)


def _write_train_loss_summary_csv(
    csv_path: str,
    layer_specs: Sequence[LayerSpec],
    mean_total_loss: Dict[int, float],
    mean_loss_components: Dict[int, Dict[str, float]],
    loss_keys: Sequence[str],
) -> None:
    """Write per-layer mean train loss summary CSV.

    Args:
        csv_path: Output summary CSV path.
        layer_specs: Layer specification list.
        mean_total_loss: Mean total loss by normalized layer id.
        mean_loss_components: Mean component losses by normalized layer id.
        loss_keys: Loss component keys.

    Returns:
        None.
    """
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    ordered_keys = sorted(loss_keys)
    headers = ["layer", "normalized_layer", "mean_total_loss"] + [f"mean_{key}" for key in ordered_keys]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for spec in layer_specs:
            component_dict = mean_loss_components.get(spec.norm_id, {})
            row = {
                "layer": spec.raw_id,
                "normalized_layer": spec.norm_id,
                "mean_total_loss": mean_total_loss.get(spec.norm_id, math.nan),
            }
            for key in ordered_keys:
                row[f"mean_{key}"] = component_dict.get(key, math.nan)
            writer.writerow(row)


def _resolve_run_dir(
    output_root: str,
    config_path: str,
    run_name: Optional[str],
) -> Tuple[Path, str]:
    """Resolve run directory under output root.

    Args:
        output_root: Root output directory path.
        config_path: Config file path.
        run_name: Optional run name.

    Returns:
        Tuple of ``(run_dir, run_name)``.
    """
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = PROJECT_ROOT / output_root_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = run_name or f"{Path(config_path).stem}_{timestamp}"
    run_dir = output_root_path / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve(), final_run_name


def _setup_run_logger(run_dir: Path, run_name: str) -> str:
    """Setup file logger for current sweep run.

    Args:
        run_dir: Run directory.
        run_name: Run name.

    Returns:
        Log file path string.
    """
    logger_name = f"sweep_L_spatial_{run_name}"
    log_file = run_dir / "sweep.log"
    if XSamLogger.check_instance_created(logger_name):
        XSamLogger.get_instance(logger_name)
    else:
        XSamLogger.get_instance(
            name=logger_name,
            logger_name=logger_name,
            log_file=str(log_file),
            log_level="INFO",
            file_mode="a",
            distributed=False,
        )
    return str(log_file)


def _setup_console_tee(run_dir: Path) -> str:
    """Duplicate stdout/stderr to a run-local console log file.

    Args:
        run_dir: Run directory.

    Returns:
        Console log file path string.
    """
    global _CONSOLE_LOG_HANDLE

    console_log = run_dir / "console.log"
    _CONSOLE_LOG_HANDLE = open(console_log, "a", encoding="utf-8", buffering=1)

    sys.stdout = _TeeStream(sys.__stdout__, _CONSOLE_LOG_HANDLE)
    sys.stderr = _TeeStream(sys.__stderr__, _CONSOLE_LOG_HANDLE)

    # Rebind existing root stream handlers to redirected stdout.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)

    return str(console_log)


def _resolve_output_csv(output_csv: Optional[str], run_dir: Path) -> str:
    """Resolve CSV output path.

    Args:
        output_csv: User-provided CSV path.
        run_dir: Run directory.

    Returns:
        Resolved CSV path string.
    """
    if output_csv is None:
        return str((run_dir / "sweep_L_spatial.csv").resolve())

    output_path = Path(output_csv)
    if output_path.is_absolute():
        return str(output_path)
    return str((run_dir / output_path).resolve())


def _save_run_artifacts(
    run_dir: Path,
    args: SaptialSweepCfg,
    source_config_path: str,
    source_yaml_path: Optional[str],
) -> None:
    """Save run metadata artifacts.

    Args:
        run_dir: Run directory.
        args: Parsed arguments.
        source_config_path: Source mmengine config path.
        source_yaml_path: Source sweep YAML path.

    Returns:
        None.
    """
    args_dict = asdict(args)
    args_json_path = run_dir / "args.json"
    with open(args_json_path, "w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=2, ensure_ascii=False)

    cmd_path = run_dir / "command.txt"
    with open(cmd_path, "w", encoding="utf-8") as handle:
        handle.write(shlex.join(sys.argv) + "\n")

    src_cfg = Path(source_config_path)
    if src_cfg.exists():
        shutil.copy2(src_cfg, run_dir / "config.py")
    if source_yaml_path is not None:
        src_yaml = Path(source_yaml_path)
        if not src_yaml.is_absolute():
            src_yaml = (PROJECT_ROOT / src_yaml).resolve()
        if src_yaml.exists():
            shutil.copy2(src_yaml, run_dir / "sweep.yaml")


def main() -> None:
    """Run layer-wise probe sweep.

    Args:
        None.

    Returns:
        None.
    """
    cli_args = _parse_cli_args()
    args, cfg = _resolve_sweep_args(
        config_path=cli_args.config,
        config_yaml_path=cli_args.config_yaml,
        phase=cli_args.phase,
    )
    args = _apply_cli_overrides(args, cli_args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir, run_name = _resolve_run_dir(args.output_root, args.config, args.run_name)
    console_log = _setup_console_tee(run_dir)
    log_file = _setup_run_logger(run_dir, run_name)
    _save_run_artifacts(run_dir, args, args.config, cli_args.config_yaml)
    print_log(f"Run directory: {run_dir}", logger="current")
    print_log(f"Console log: {console_log}", logger="current")
    print_log(f"Log file: {log_file}", logger="current")

    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)

    dense_keywords = _split_csv_text(args.dense_keywords)
    ref_keywords = _split_csv_text(args.ref_keywords)
    selected_cfgs = _select_eval_entries(
        cfg=cfg,
        dense_keywords=dense_keywords,
        ref_keywords=ref_keywords,
        explicit_data_names=args.data_names,
    )
    if not selected_cfgs:
        raise ValueError("No dataset/evaluator pair selected. Check data filters.")

    entries = _build_eval_entries(selected_cfgs)
    print_log(
        f"Selected {len(entries)} eval tasks: {[entry.data_name for entry in entries]}",
        logger="current",
    )
    eval_dataset_rows = [
        _extract_dataset_summary_row(entry.dataset, data_name=entry.data_name, task_name=entry.task_type)
        for entry in entries
    ]
    _log_dataset_summary(eval_dataset_rows, "Eval dataset summary")

    train_dataset_full = None
    train_dataset = None
    if args.train_epochs > 0:
        train_dataset_full = _build_train_dataset(cfg)
        train_dataset = _build_train_subset(
            train_dataset=train_dataset_full,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        full_train_row = _extract_dataset_summary_row(
            train_dataset_full,
            data_name=getattr(train_dataset_full, "data_name", "train_dataset"),
            task_name=getattr(train_dataset_full, "task_name", "train"),
        )
        subset_train_row = _extract_dataset_summary_row(
            train_dataset,
            data_name=f"{full_train_row['dataset']}__subset",
            task_name=full_train_row["task"],
        )
        subset_train_row["dataset"] = f"{full_train_row['dataset']}__subset"
        subset_train_row["repeats"] = (
            full_train_row["repeats"] * (len(train_dataset) / max(len(train_dataset_full), 1))
        )
        subset_train_row["data_length"] = len(train_dataset)
        _log_dataset_summary([full_train_row, subset_train_row], "Train dataset summary")

    model = BUILDER.build(cfg.model)
    model = model.to(get_device())
    model.eval()
    model.requires_grad_(False)
    _log_model_summary(model, _resolve_base_model_summary_modules(model), "Base model summary")

    if args.pth_model is not None:
        load_checkpoint(model, args.pth_model)
    else:
        print_log("No --pth-model provided, using weights from config init fields only.", logger="current")

    if getattr(model, "segmentor", None) is None or getattr(model.segmentor, "encoder", None) is None:
        raise ValueError("Model must contain `segmentor.encoder`.")
    if not hasattr(model.segmentor.encoder, "vision_backbone"):
        raise ValueError("Current segmentor encoder has no `vision_backbone`.")
    if not hasattr(model.segmentor.encoder.vision_backbone, "trunk"):
        raise ValueError("Current SAM3 encoder has no `vision_backbone.trunk`.")

    total_levels = len(model.segmentor.encoder.vision_backbone.trunk.blocks)
    layer_ids = _parse_layers(args.layers)
    layer_specs = _build_layer_specs(layer_ids, total_levels)
    print_log(
        f"Sweep layers (raw->norm): {[f'{spec.raw_id}->{spec.norm_id}' for spec in layer_specs]}",
        logger="current",
    )

    heads, optimizers = _build_probe_heads(
        base_model=model,
        layer_specs=layer_specs,
        seed=args.seed,
        seed_stride=args.seed_stride,
        reinit_weights=args.probe_reinit,
    )
    _set_optimizer_hparams(optimizers, lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    _log_probe_head_overview(layer_specs=layer_specs, heads=heads)
    for spec in layer_specs:
        head = heads[spec.norm_id]
        head_module_names = [name for name, module in head.named_children() if isinstance(module, nn.Module)]
        _log_model_summary(
            head,
            module_names=head_module_names,
            title=f"Probe head summary (layer={spec.raw_id}/{spec.norm_id})",
        )

    train_loss_by_layer = {spec.norm_id: math.nan for spec in layer_specs}
    train_loss_components_by_layer = {spec.norm_id: {} for spec in layer_specs}
    train_eval_entry = _select_train_eval_entry(entries)
    if args.train_epochs > 0:
        if train_dataset_full is None or train_dataset is None:
            raise RuntimeError("Training datasets should be prepared before entering probe training.")
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.train_batch_size))
        total_steps = steps_per_epoch * args.train_epochs
        print_log(
            (
                f"Probe training starts: epochs={args.train_epochs}, batch_size={args.train_batch_size}, "
                f"train_ratio={args.train_ratio:.3f}, dataset_len={len(train_dataset_full)}, "
                f"subset_len={len(train_dataset)}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
                f"probe_reinit={args.probe_reinit}, grad_accum_steps={args.grad_accum_steps}, "
                f"effective_batch={args.train_batch_size * args.grad_accum_steps}, max_save={args.max_save}"
            ),
            logger="current",
        )
        if train_eval_entry is not None and args.train_eval_interval > 0:
            print_log(
                f"Train-time eval task selected: {train_eval_entry.data_name}",
                logger="current",
            )
        train_stats = _train_probe_heads(
            base_model=model,
            heads=heads,
            optimizers=optimizers,
            layer_specs=layer_specs,
            train_dataset=train_dataset,
            train_epochs=args.train_epochs,
            train_batch_size=args.train_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            train_num_workers=args.train_num_workers,
            run_dir=run_dir,
            save_steps=args.save_steps,
            max_save=args.max_save,
            resume=args.resume,
            resume_ckpt=args.resume_ckpt,
            use_tqdm=args.use_tqdm,
            log_interval=args.log_interval,
            train_eval_entry=train_eval_entry,
            train_eval_batch_size=args.batch_size,
            train_eval_num_workers=args.num_workers,
            train_eval_interval=args.train_eval_interval,
            train_eval_max_samples=args.train_eval_max_samples,
            early_stop_patience_steps=args.early_stop_patience_steps,
            early_stop_miou_eps=args.early_stop_miou_eps,
            eval_oom_empty_cache=args.eval_oom_empty_cache,
        )
        train_loss_by_layer = train_stats.mean_total_loss
        train_loss_components_by_layer = train_stats.mean_loss_components

        train_trace_csv = str((run_dir / "train_loss_trace.csv").resolve())
        train_summary_csv = str((run_dir / "train_loss_summary.csv").resolve())
        _write_train_loss_trace_csv(
            csv_path=train_trace_csv,
            trace_rows=train_stats.trace_rows,
            loss_keys=train_stats.loss_keys,
        )
        _write_train_loss_summary_csv(
            csv_path=train_summary_csv,
            layer_specs=layer_specs,
            mean_total_loss=train_stats.mean_total_loss,
            mean_loss_components=train_stats.mean_loss_components,
            loss_keys=train_stats.loss_keys,
        )
        print_log(f"Saved train loss trace CSV to: {train_trace_csv}", logger="current")
        print_log(f"Saved train loss summary CSV to: {train_summary_csv}", logger="current")

    rows = _evaluate_probe_heads(
        base_model=model,
        heads=heads,
        layer_specs=layer_specs,
        entries=entries,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_task=args.max_samples_per_task,
        use_tqdm=args.use_tqdm,
        log_interval=args.log_interval,
        train_loss_by_layer=train_loss_by_layer,
        train_loss_components_by_layer=train_loss_components_by_layer,
        eval_fail_fast=args.eval_fail_fast,
        eval_fail_ratio_threshold=args.eval_fail_ratio_threshold,
        eval_fail_check_min_samples=args.eval_fail_check_min_samples,
        eval_oom_empty_cache=args.eval_oom_empty_cache,
        eval_log_cuda_mem=args.eval_log_cuda_mem,
    )

    output_csv = _resolve_output_csv(args.output_csv, run_dir)
    _write_csv(output_csv, rows)
    print_log(f"Saved CSV to: {output_csv}", logger="current")


if __name__ == "__main__":
    main()

"""
python xsam/xsam/layer_analysis/spatial/layer_sweep_spatial.py \
  --config-yaml xsam/xsam/layer_analysis/spatial/spatial_sweep.yaml \
  --config xsam/xsam/configs/xsam/layer_analysis/spatial/xsam_sam3_spatial.py \
  --layers=-1,-2,-4,-6,-8,-10,-12,-16,-24,-32 \
  --train-epochs 2 \
  --train-ratio 0.25 \
  --grad-accum-steps 24 \
  --max-save -2 \
  --probe-lr 1e-4 \
  --probe-weight-decay 0.05 \
  --no-probe-reinit \
  --train-eval-interval 200 \
  --train-eval-max-samples 256 \
  --early-stop-patience-steps 2000 \
  --early-stop-miou-eps 0.1

"""
