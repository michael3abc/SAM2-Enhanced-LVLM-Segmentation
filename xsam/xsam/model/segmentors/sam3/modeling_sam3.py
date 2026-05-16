"""PyTorch SAM3 model wrappers for X-SAM."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_sam3 import Sam3Config, Sam3PromptEncoderConfig, Sam3VisionConfig

logger = logging.get_logger(__name__)


def _resolve_external_sam3_dir(start_path: Path) -> Optional[Path]:
    """Resolve the `external/sam3` directory from multiple runtime hints.

    Args:
        start_path: Current file path.
    Returns:
        Path to `external/sam3` when found, otherwise None.
    """
    candidate_bases: list[Path] = []
    env_root = os.environ.get("ROOT_DIR", None)
    env_code = os.environ.get("CODE_DIR", None)

    if env_root:
        candidate_bases.append(Path(env_root).resolve())
    if env_code:
        code_dir = Path(env_code).resolve()
        candidate_bases.extend([code_dir, code_dir.parent])

    candidate_bases.extend([Path.cwd().resolve(), start_path, *start_path.parents])

    seen = set()
    for base in candidate_bases:
        if base in seen:
            continue
        seen.add(base)
        for external_dir in (base / "external" / "sam3", base.parent / "external" / "sam3"):
            if external_dir.exists():
                return external_dir.resolve()
    return None


def _ensure_external_sam3_in_path() -> None:
    """Add `external/sam3` to Python path when available.

    Args:
        None.
    Returns:
        None.
    """
    external_sam3 = _resolve_external_sam3_dir(Path(__file__).resolve())
    if external_sam3 is None:
        return

    ext_path = str(external_sam3)
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)


def _import_sam3_backbone_modules():
    """Import SAM3 backbone modules from submodule package.

    Args:
        None.
    Returns:
        Tuple of `(ViT, Sam3DualViTDetNeck, PositionEmbeddingSine)` classes.
    """
    _ensure_external_sam3_in_path()
    try:
        from sam3.model.necks import Sam3DualViTDetNeck
        from sam3.model.position_encoding import PositionEmbeddingSine
        from sam3.model.vitdet import ViT
    except Exception:
        # Fallback: avoid executing external `sam3/__init__.py` when optional deps
        # (e.g. pkg_resources from setuptools) are missing in runtime env.
        external_sam3 = _resolve_external_sam3_dir(Path(__file__).resolve())
        if external_sam3 is None:
            raise

        sam3_pkg_root = external_sam3 / "sam3"
        sam3_model_root = sam3_pkg_root / "model"
        if not sam3_model_root.exists():
            raise

        if "sam3" not in sys.modules:
            sam3_pkg = types.ModuleType("sam3")
            sam3_pkg.__path__ = [str(sam3_pkg_root)]
            sys.modules["sam3"] = sam3_pkg
        if "sam3.model" not in sys.modules:
            sam3_model_pkg = types.ModuleType("sam3.model")
            sam3_model_pkg.__path__ = [str(sam3_model_root)]
            sys.modules["sam3.model"] = sam3_model_pkg

        Sam3DualViTDetNeck = importlib.import_module("sam3.model.necks").Sam3DualViTDetNeck
        PositionEmbeddingSine = importlib.import_module("sam3.model.position_encoding").PositionEmbeddingSine
        ViT = importlib.import_module("sam3.model.vitdet").ViT

    return ViT, Sam3DualViTDetNeck, PositionEmbeddingSine


def _normalize_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize checkpoint dictionary to a flat state dict.

    Args:
        state_dict: Raw dictionary loaded from checkpoint.
    Returns:
        Flat state dictionary.
    """
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        return state_dict["model"]
    return state_dict


def _load_sam3_vision_state_dict(
    pretrained_model_name_or_path: str,
    encoder_filename: str,
    fpn_filename: str,
    map_location: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load SAM3 vision trunk+FPN state dict from file or directory.

    Args:
        pretrained_model_name_or_path: Source path, either a directory or `sam3.pt` file.
        encoder_filename: File name for split encoder checkpoint.
        fpn_filename: File name for split FPN checkpoint.
        map_location: Device map location for torch load.
    Returns:
        State dict with keys normalized to `vision_backbone.*`.
    """
    source = Path(pretrained_model_name_or_path)
    state_dict: dict[str, torch.Tensor] = {}

    if source.is_file():
        raw = torch.load(source, map_location=map_location, weights_only=True)
        flat = _normalize_checkpoint_state_dict(raw)
        for key, value in flat.items():
            if key.startswith("detector.backbone.vision_backbone."):
                state_dict[key.replace("detector.backbone.", "", 1)] = value
        return state_dict

    encoder_path = source / encoder_filename
    fpn_path = source / fpn_filename
    full_path = source / "sam3.pt"

    if encoder_path.exists() and fpn_path.exists():
        for path in [encoder_path, fpn_path]:
            raw = torch.load(path, map_location=map_location, weights_only=True)
            flat = _normalize_checkpoint_state_dict(raw)
            for key, value in flat.items():
                if key.startswith("detector.backbone.vision_backbone."):
                    state_dict[key.replace("detector.backbone.", "", 1)] = value
                elif key.startswith("vision_backbone."):
                    state_dict[key] = value
        return state_dict

    if full_path.exists():
        return _load_sam3_vision_state_dict(
            pretrained_model_name_or_path=str(full_path),
            encoder_filename=encoder_filename,
            fpn_filename=fpn_filename,
            map_location=map_location,
        )

    raise FileNotFoundError(
        f"Cannot locate SAM3 weights from {pretrained_model_name_or_path}. "
        f"Expected either file `sam3.pt` or files `{encoder_filename}` + `{fpn_filename}`."
    )


def _filter_state_dict_by_shape(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """Filter checkpoint tensors with incompatible shapes for non-strict loads.

    Args:
        model: Target model receiving the state dict.
        state_dict: Candidate checkpoint state dict.
        strict: Whether strict loading is requested.

    Returns:
        Tuple of filtered state dict and skipped shape-mismatch records.
    """
    if strict:
        return state_dict, []

    model_state = model.state_dict()
    filtered_state = OrderedDict()
    skipped = []
    for key, value in state_dict.items():
        target_value = model_state.get(key)
        if target_value is not None and target_value.shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        filtered_state[key] = value
    return filtered_state, skipped


@dataclass
class Sam3VisionEncoderOutput(ModelOutput):
    """Outputs of SAM3 vision encoder."""

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    fpn_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    fpn_position_encoding: Optional[Tuple[torch.FloatTensor, ...]] = None
    sam2_fpn_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    sam2_fpn_position_encoding: Optional[Tuple[torch.FloatTensor, ...]] = None
    trunk_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    trunk_selected_layers: Optional[Tuple[int, ...]] = None


@dataclass
class Sam3ImageSegmentationOutput(ModelOutput):
    """Outputs of SAM3 model wrapper."""

    last_hidden_state: torch.FloatTensor = None
    image_embeddings: Optional[Tuple[torch.FloatTensor, ...]] = None
    fpn_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    fpn_position_encoding: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    trunk_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    trunk_selected_layers: Optional[Tuple[int, ...]] = None


class Sam3PromptEncoder(nn.Module):
    """Placeholder prompt encoder for API compatibility with X-SAM."""

    def __init__(self, config: Sam3PromptEncoderConfig):
        """Initialize prompt encoder placeholder.

        Args:
            config: Prompt encoder config.
        Returns:
            None.
        """
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        """Forward prompt encoder placeholder.

        Args:
            *args: Unused args.
            **kwargs: Unused kwargs.
        Returns:
            Tuple `(None, None)` as sparse/dense prompt embeddings placeholder.
        """
        return None, None


class Sam3PreTrainedModel(PreTrainedModel):
    """Base class for SAM3 wrappers."""

    config_class = Sam3Config
    base_model_prefix = "sam3"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights.

        Args:
            module: Module to initialize.
        Returns:
            None.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class Sam3VisionModel(Sam3PreTrainedModel):
    """SAM3 vision trunk + simple FPN neck."""

    config_class = Sam3VisionConfig

    def __init__(self, config: Sam3VisionConfig):
        """Initialize SAM3 vision model.

        Args:
            config: Vision config.
        Returns:
            None.
        """
        super().__init__(config)
        self.config = config

        ViT, Sam3DualViTDetNeck, PositionEmbeddingSine = _import_sam3_backbone_modules()
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=config.neck_hidden_size,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=config.img_size,
        )

        trunk = ViT(
            img_size=config.img_size,
            pretrain_img_size=config.pretrain_img_size,
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            embed_dim=config.hidden_size,
            depth=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_ratio,
            norm_layer="LayerNorm",
            drop_path_rate=config.drop_path_rate,
            qkv_bias=config.qkv_bias,
            use_abs_pos=config.use_abs_pos,
            tile_abs_pos=config.tile_abs_pos,
            global_att_blocks=tuple(config.global_attn_indexes),
            rel_pos_blocks=tuple(config.rel_pos_blocks),
            use_rope=config.use_rope,
            use_interp_rope=config.use_interp_rope,
            window_size=config.window_size,
            pretrain_use_cls_token=config.pretrain_use_cls_token,
            retain_cls_token=config.retain_cls_token,
            ln_pre=config.ln_pre,
            ln_post=config.ln_post,
            return_interm_layers=config.return_interm_layers,
            bias_patch_embed=config.bias_patch_embed,
            compile_mode=None,
        )

        self.vision_backbone = Sam3DualViTDetNeck(
            trunk=trunk,
            position_encoding=position_encoding,
            d_model=config.neck_hidden_size,
            scale_factors=tuple(config.neck_scale_factors),
            add_sam2_neck=config.add_sam2_neck,
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return input embedding layer.

        Args:
            None.
        Returns:
            Patch embedding module.
        """
        return self.vision_backbone.trunk.patch_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        freeze_trunk: bool = False,
        freeze_fpn: bool = False,
        output_trunk_hidden_states: bool = False,
        trunk_select_layers: Optional[Tuple[int, ...] | list[int]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Sam3VisionEncoderOutput | tuple:
        """Forward SAM3 vision model.

        Args:
            pixel_values: Input image tensor `[B, 3, H, W]`.
            output_hidden_states: Whether to return hidden states.
            output_attentions: Unused, kept for API compatibility.
            freeze_trunk: Run trunk forward in ``torch.no_grad()``.
            freeze_fpn: Freeze FPN parameters hint. ``torch.no_grad()`` is only
                applied when both trunk and FPN are frozen.
            output_trunk_hidden_states: Whether to return selected trunk block outputs.
            trunk_select_layers: Trunk block indexes to capture, supports negative indexing.
            return_dict: Whether to return model output dataclass.
            **kwargs: Unused kwargs.
        Returns:
            SAM3 vision output with FPN feature maps.
        """
        del output_attentions, kwargs
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        trunk_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        trunk_selected_layers: Optional[Tuple[int, ...]] = None
        hooks = []
        captured_trunk_states: Dict[int, torch.Tensor] = OrderedDict()

        need_trunk_capture = output_trunk_hidden_states or trunk_select_layers is not None
        if need_trunk_capture:
            total_blocks = len(self.vision_backbone.trunk.blocks)
            if trunk_select_layers is None:
                normalized_layers = list(range(total_blocks))
            else:
                normalized_layers = []
                for layer_id in trunk_select_layers:
                    normalized_id = layer_id if layer_id >= 0 else total_blocks + layer_id
                    if normalized_id < 0 or normalized_id >= total_blocks:
                        raise ValueError(
                            f"Invalid trunk layer id {layer_id}. "
                            f"Valid range: [{-total_blocks}, {total_blocks - 1}]."
                        )
                    if normalized_id not in normalized_layers:
                        normalized_layers.append(normalized_id)

            for layer_id in normalized_layers:
                def make_hook(cur_layer_id):
                    """Create forward hook for one trunk block."""
                    def hook(_, __, output):
                        """Capture trunk block output in channels-first format."""
                        if isinstance(output, torch.Tensor):
                            if output.ndim == 4:
                                # trunk blocks output [B, H, W, C]
                                captured = output.permute(0, 3, 1, 2).contiguous()
                            elif output.ndim == 3:
                                # fallback path (e.g. with cls token retained)
                                side = int((output.shape[1]) ** 0.5)
                                captured = output.transpose(1, 2).reshape(output.shape[0], output.shape[2], side, side)
                            else:
                                raise ValueError(f"Unsupported trunk output ndim: {output.ndim}")
                            captured_trunk_states[cur_layer_id] = captured

                    return hook

                hooks.append(self.vision_backbone.trunk.blocks[layer_id].register_forward_hook(make_hook(layer_id)))

        try:
            if not freeze_trunk and not freeze_fpn:
                sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone(pixel_values)
            else:
                trunk_context = torch.no_grad if freeze_trunk else contextlib.nullcontext
                fpn_context = torch.no_grad if (freeze_trunk and freeze_fpn) else contextlib.nullcontext

                with trunk_context():
                    trunk_features = self.vision_backbone.trunk(pixel_values)
                x = trunk_features[-1]

                sam3_features, sam3_pos = [], []
                sam2_features, sam2_pos = None, None
                if self.vision_backbone.sam2_convs is not None:
                    sam2_features, sam2_pos = [], []

                with fpn_context():
                    for i in range(len(self.vision_backbone.convs)):
                        sam3_x_out = self.vision_backbone.convs[i](x)
                        sam3_pos_out = self.vision_backbone.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
                        sam3_features.append(sam3_x_out)
                        sam3_pos.append(sam3_pos_out)

                        if self.vision_backbone.sam2_convs is not None:
                            sam2_x_out = self.vision_backbone.sam2_convs[i](x)
                            sam2_pos_out = self.vision_backbone.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                            sam2_features.append(sam2_x_out)
                            sam2_pos.append(sam2_pos_out)
        finally:
            for hook in hooks:
                hook.remove()

        if need_trunk_capture:
            trunk_selected_layers = tuple(captured_trunk_states.keys())
            trunk_hidden_states = tuple(captured_trunk_states[k] for k in trunk_selected_layers)

        if self.config.scalp > 0:
            sam3_features = sam3_features[: -self.config.scalp]
            sam3_pos = sam3_pos[: -self.config.scalp]
            if sam2_features is not None and sam2_pos is not None:
                sam2_features = sam2_features[: -self.config.scalp]
                sam2_pos = sam2_pos[: -self.config.scalp]

        output = Sam3VisionEncoderOutput(
            last_hidden_state=sam3_features[-1],
            hidden_states=tuple(sam3_features) if output_hidden_states else None,
            fpn_hidden_states=tuple(sam3_features),
            fpn_position_encoding=tuple(sam3_pos),
            sam2_fpn_hidden_states=tuple(sam2_features) if sam2_features is not None else None,
            sam2_fpn_position_encoding=tuple(sam2_pos) if sam2_pos is not None else None,
            trunk_hidden_states=trunk_hidden_states,
            trunk_selected_layers=trunk_selected_layers,
        )
        if not return_dict:
            return tuple(output.values())
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Sam3VisionConfig | dict | None = None,
        encoder_filename: str = "sam3_encoder.bin",
        fpn_filename: str = "sam3_fpn.bin",
        strict: bool = False,
        map_location: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Sam3VisionModel":
        """Load SAM3 vision model from local checkpoint.

        Args:
            pretrained_model_name_or_path: Source path to directory or checkpoint file.
            *model_args: Additional model args.
            config: Vision config object or dict.
            encoder_filename: Split encoder filename.
            fpn_filename: Split FPN filename.
            strict: Whether to enforce strict state-dict loading.
            map_location: Device map location for torch load.
            torch_dtype: Optional target dtype of the model.
            **kwargs: Additional unused kwargs for API compatibility.
        Returns:
            Loaded SAM3 vision model.
        """
        del kwargs
        if config is None:
            config = Sam3VisionConfig()
        elif isinstance(config, dict):
            config = Sam3VisionConfig(**config)

        model = cls(config, *model_args)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        state_dict = _load_sam3_vision_state_dict(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            encoder_filename=encoder_filename,
            fpn_filename=fpn_filename,
            map_location=map_location,
        )
        state_dict, skipped_shape_keys = _filter_state_dict_by_shape(model, state_dict, strict=strict)
        if len(skipped_shape_keys) > 0:
            logger.warning("Skipped SAM3 keys with shape mismatch: %s", skipped_shape_keys)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        if len(unexpected_keys) > 0:
            logger.warning("Unexpected SAM3 keys: %s", unexpected_keys)
        if len(missing_keys) > 0:
            logger.warning("Missing SAM3 keys: %s", missing_keys)
        return model


class Sam3Model(Sam3PreTrainedModel):
    """SAM3 model wrapper that exposes vision encoder for X-SAM."""

    config_class = Sam3Config

    def __init__(self, config: Sam3Config):
        """Initialize SAM3 model wrapper.

        Args:
            config: SAM3 config.
        Returns:
            None.
        """
        super().__init__(config)
        self.config = config
        self.vision_encoder = Sam3VisionModel(config.vision_config)
        self.prompt_encoder = Sam3PromptEncoder(config.prompt_encoder_config)
        self.mask_decoder = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return input embedding layer.

        Args:
            None.
        Returns:
            Patch embedding module.
        """
        return self.vision_encoder.get_input_embeddings()

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Sam3ImageSegmentationOutput | tuple:
        """Forward SAM3 model wrapper.

        Args:
            pixel_values: Input image tensor `[B, 3, H, W]`.
            output_hidden_states: Whether to return hidden states.
            output_attentions: Unused, kept for API compatibility.
            return_dict: Whether to return model output dataclass.
            **kwargs: Additional unused kwargs.
        Returns:
            SAM3 wrapper output containing FPN image embeddings.
        """
        del output_attentions
        output_trunk_hidden_states = kwargs.pop("output_trunk_hidden_states", False)
        trunk_select_layers = kwargs.pop("trunk_select_layers", None)
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_trunk_hidden_states=output_trunk_hidden_states,
            trunk_select_layers=trunk_select_layers,
            return_dict=True,
        )

        output = Sam3ImageSegmentationOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            image_embeddings=vision_outputs.fpn_hidden_states,
            fpn_hidden_states=vision_outputs.fpn_hidden_states,
            fpn_position_encoding=vision_outputs.fpn_position_encoding,
            vision_hidden_states=vision_outputs.hidden_states,
            trunk_hidden_states=vision_outputs.trunk_hidden_states,
            trunk_selected_layers=vision_outputs.trunk_selected_layers,
        )
        if not return_dict:
            return tuple(output.values())
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Sam3Config | dict | None = None,
        encoder_filename: str = "sam3_encoder.bin",
        fpn_filename: str = "sam3_fpn.bin",
        strict: bool = False,
        map_location: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Sam3Model":
        """Load SAM3 wrapper from local checkpoint.

        Args:
            pretrained_model_name_or_path: Source path to directory or checkpoint file.
            *model_args: Additional model args.
            config: SAM3 config object or dict.
            encoder_filename: Split encoder filename.
            fpn_filename: Split FPN filename.
            strict: Whether to enforce strict state-dict loading.
            map_location: Device map location for torch load.
            torch_dtype: Optional target dtype.
            **kwargs: Additional unused kwargs for API compatibility.
        Returns:
            Loaded SAM3 model wrapper.
        """
        del kwargs
        if config is None:
            config = Sam3Config()
        elif isinstance(config, dict):
            config = Sam3Config(**config)

        model = cls(config, *model_args)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        vision_state = _load_sam3_vision_state_dict(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            encoder_filename=encoder_filename,
            fpn_filename=fpn_filename,
            map_location=map_location,
        )
        wrapped_state = {f"vision_encoder.{k}": v for k, v in vision_state.items()}
        wrapped_state, skipped_shape_keys = _filter_state_dict_by_shape(model, wrapped_state, strict=strict)
        if len(skipped_shape_keys) > 0:
            logger.warning("Skipped SAM3 keys with shape mismatch: %s", skipped_shape_keys)
        missing_keys, unexpected_keys = model.load_state_dict(wrapped_state, strict=strict)
        if len(unexpected_keys) > 0:
            logger.warning("Unexpected SAM3 keys: %s", unexpected_keys)
        if len(missing_keys) > 0:
            logger.warning("Missing SAM3 keys: %s", missing_keys)
        return model
