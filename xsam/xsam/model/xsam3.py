import logging
from typing import List, Optional

import torch
from mmengine import print_log

from ..utils.misc import data_sample_to_device
from .modules import DynamicProjectorConfig, DynamicProjectorModel
from .utils import prepare_inputs_labels_for_multimodal
from .xsam import XSamModel


class XSam3Model(XSamModel):
    """X-SAM model variant with a single SAM3 backbone."""

    def __init__(
        self,
        *args,
        lang_bast_layers: Optional[List[int]] = None,
        seg_bast_layers: Optional[List[int]] = None,
        lang_downsample_ratio: float = 1.0,
        seg_downsample_ratio: Optional[float] = None,
        **kwargs,
    ):
        """Initialize XSam3Model with single-backbone settings.

        Args:
            *args: Positional args forwarded to ``XSamModel``.
            lang_bast_layers: Trunk block indexes used to build ``H_v``.
            seg_bast_layers: Trunk block indexes used to build ``H_s``.
            lang_downsample_ratio: Pixel-shuffle downsample ratio for ``H_v``.
            seg_downsample_ratio: Pixel-shuffle downsample ratio for ``H_s``.
            **kwargs: Keyword args forwarded to ``XSamModel``.
        Returns:
            None.
        """
        visual_encoder = kwargs.pop("visual_encoder", None)
        kwargs.pop("use_dual_encoder", None)
        projector_pretrained_pth = kwargs.get("projector_pretrained_pth", None)
        projector_depth = kwargs.get("projector_depth", 2)
        downsample_ratio = kwargs.get("downsample_ratio", 0.5)

        if visual_encoder is not None:
            print_log(
                "XSam3Model uses single backbone. `visual_encoder` is ignored.",
                logger="current",
                level=logging.WARNING,
            )

        super().__init__(*args, visual_encoder=None, **kwargs)

        self.lang_bast_layers = list(lang_bast_layers) if lang_bast_layers is not None else [-1]
        self.seg_bast_layers = list(seg_bast_layers) if seg_bast_layers is not None else [-2]
        self.lang_downsample_ratio = lang_downsample_ratio
        self.seg_downsample_ratio = downsample_ratio if seg_downsample_ratio is None else seg_downsample_ratio
        self._warned_extra_pixel_values = False

        if self.segmentor is None:
            return

        segmentor_hidden_size = getattr(self.segmentor.enc_config, "hidden_size", None)
        if segmentor_hidden_size is None:
            raise ValueError("`segmentor.enc_config.hidden_size` is required for XSam3Model.")

        if self.llm is not None:
            visual_projector_config = DynamicProjectorConfig(
                visual_hidden_size=segmentor_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                downsample_ratio=self.lang_downsample_ratio,
                depth=projector_depth,
            )
            self.visual_projector = DynamicProjectorModel(visual_projector_config).to(self.segmentor.dtype)

        if self.llm is not None:
            seg_projector_config = DynamicProjectorConfig(
                visual_hidden_size=segmentor_hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                downsample_ratio=self.seg_downsample_ratio,
                depth=projector_depth,
            )
            self.seg_projector = DynamicProjectorModel(seg_projector_config).to(self.segmentor.dtype)

        if projector_pretrained_pth is not None and self.llm is not None:
            target_prefixes = ["visual_projector", "seg_projector"]
            self._load_partial_pretrained(
                checkpoint_path=projector_pretrained_pth,
                target_prefixes=target_prefixes,
                source_prefix_map={},
                ckpt_tag="projector_pretrained_pth_xsam3",
            )

    def activation_checkpointing_enable(self):
        """Enable activation checkpointing for submodules.

        Args:
            None.
        Returns:
            None.
        """
        super().activation_checkpointing_enable()
        if hasattr(self, "visual_projector"):
            self.visual_projector.gradient_checkpointing_enable()

    def activation_checkpointing_disable(self):
        """Disable activation checkpointing for submodules.

        Args:
            None.
        Returns:
            None.
        """
        super().activation_checkpointing_disable()
        if hasattr(self, "visual_projector"):
            self.visual_projector.gradient_checkpointing_disable()

    def _normalize_layer_ids(
        self,
        layer_ids: List[int],
        total_levels: int,
        attr_name: str,
    ) -> List[int]:
        """Normalize signed layer indexes to non-negative indexes.

        Args:
            layer_ids: Layer index list that may contain negative indexes.
            total_levels: Total number of available layers.
            attr_name: Attribute name used in error messages.
        Returns:
            Normalized non-negative index list.
        """
        if len(layer_ids) == 0:
            raise ValueError(f"`{attr_name}` cannot be empty.")

        normalized_layers = []
        for level_id in layer_ids:
            normalized_id = level_id if level_id >= 0 else total_levels + level_id
            if normalized_id < 0 or normalized_id >= total_levels:
                raise ValueError(
                    f"`{attr_name}` contains invalid index {level_id}. "
                    f"Valid range: [{-total_levels}, {total_levels - 1}]."
                )
            normalized_layers.append(normalized_id)
        return normalized_layers

    def _select_feature_levels(
        self,
        feature_levels: List[torch.Tensor] | tuple[torch.Tensor, ...],
        layer_ids: List[int],
        attr_name: str,
    ) -> List[torch.Tensor]:
        """Select feature maps by index list.

        Args:
            feature_levels: Multi-level feature maps in `[B, C, H, W]`.
            layer_ids: Feature level indexes, supports negative indexing.
            attr_name: Attribute name used in error messages.
        Returns:
            A list of selected feature maps.
        """
        if len(feature_levels) == 0:
            raise ValueError("Empty feature levels.")
        total_levels = len(feature_levels)
        normalized_layers = self._normalize_layer_ids(layer_ids, total_levels, attr_name)
        return [feature_levels[layer_id] for layer_id in normalized_layers]

    def _project_feature_levels(
        self,
        feature_levels: List[torch.Tensor],
        projector: DynamicProjectorModel,
        target_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Project selected feature levels to LLM token embeddings.

        Args:
            feature_levels: Selected feature maps in `[B, C, H, W]`.
            projector: Projector module.
            target_dtype: Output dtype.
        Returns:
            Projected token embeddings `[B, N, D]` or ``None``.
        """
        if len(feature_levels) == 0:
            return None

        projected = []
        for feat in feature_levels:
            projected_feat = projector(feat.permute(0, 2, 3, 1).contiguous())
            projected.append(projected_feat.to(target_dtype))

        return torch.cat(projected, dim=1)

    def _extract_segmentor_features(self, pixel_values: torch.Tensor, trunk_select_layers: Optional[List[int]] = None):
        """Extract SAM3 backbone features from input images.

        Args:
            pixel_values: Input image tensor `[B, 3, H, W]`.
            trunk_select_layers: Trunk block indexes to capture from one forward pass.
        Returns:
            Segmentor encoder outputs and normalized feature tuple.
        """
        encoder_kwargs = dict(
            pixel_values=pixel_values.to(self.segmentor.dtype),
            output_hidden_states=True,
            output_attentions=False,
        )
        if trunk_select_layers is not None and len(trunk_select_layers) > 0:
            encoder_kwargs["output_trunk_hidden_states"] = True
            encoder_kwargs["trunk_select_layers"] = tuple(trunk_select_layers)

        try:
            seg_visual_outputs = self.segmentor.encoder(**encoder_kwargs)
        except TypeError:
            # Backward compatibility for encoders that don't expose trunk capture args
            # or use positional `pixel_values`.
            encoder_kwargs.pop("output_trunk_hidden_states", None)
            encoder_kwargs.pop("trunk_select_layers", None)
            try:
                seg_visual_outputs = self.segmentor.encoder(**encoder_kwargs)
            except TypeError:
                seg_visual_outputs = self.segmentor.encoder(
                    encoder_kwargs.pop("pixel_values"),
                    output_hidden_states=encoder_kwargs.get("output_hidden_states", True),
                    output_attentions=encoder_kwargs.get("output_attentions", False),
                )

        if hasattr(seg_visual_outputs, "fpn_hidden_states"):
            seg_image_embeddings = seg_visual_outputs.fpn_hidden_states
        elif hasattr(seg_visual_outputs, "feature_maps"):
            seg_image_embeddings = seg_visual_outputs.feature_maps
        elif hasattr(seg_visual_outputs, "last_hidden_state"):
            seg_image_embeddings = (seg_visual_outputs.last_hidden_state,)
        else:
            raise ValueError("Cannot infer feature levels from segmentor encoder outputs.")

        return seg_visual_outputs, tuple(seg_image_embeddings)

    def _select_trunk_feature_levels(self, seg_visual_outputs, layer_ids: List[int], attr_name: str) -> List[torch.Tensor]:
        """Select captured trunk feature maps by layer indexes.

        Args:
            seg_visual_outputs: Segmentor encoder output object.
            layer_ids: Requested trunk layer indexes.
            attr_name: Attribute name used in error messages.
        Returns:
            Selected trunk feature maps in `[B, C, H, W]`.
        """
        trunk_hidden_states = getattr(seg_visual_outputs, "trunk_hidden_states", None)
        trunk_selected_layers = getattr(seg_visual_outputs, "trunk_selected_layers", None)
        if trunk_hidden_states is None or trunk_selected_layers is None:
            raise ValueError(
                "Segmentor encoder output does not provide trunk hidden states. "
                "Please use `Sam3VisionModel` with trunk capture support."
            )

        layer_to_feature = {layer_id: feat for layer_id, feat in zip(trunk_selected_layers, trunk_hidden_states)}
        total_levels = len(self.segmentor.encoder.vision_backbone.trunk.blocks)
        normalized_layers = self._normalize_layer_ids(layer_ids, total_levels, attr_name)

        selected = []
        for layer_id in normalized_layers:
            if layer_id not in layer_to_feature:
                raise ValueError(
                    f"Requested trunk layer {layer_id} is missing from captured layers {list(layer_to_feature.keys())}."
                )
            selected.append(layer_to_feature[layer_id])
        return selected

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        """Run model forward with single-backbone data flow.

        Args:
            data_dict: Batch input dictionary.
            data_samples: Optional data samples.
            mode: Forward mode in ``{'loss', 'predict', 'tensor'}``.
            **kwargs: Additional forward kwargs.
        Returns:
            Model outputs in the selected mode.
        """
        if self.segmentor is None:
            return super().forward(data_dict, data_samples=data_samples, mode=mode, **kwargs)

        if data_samples is not None:
            data_samples = data_sample_to_device(data_samples, device=self.device)

        if data_dict.get("extra_pixel_values", None) is not None and not self._warned_extra_pixel_values:
            print_log(
                "XSam3Model ignores `extra_pixel_values` and only uses `pixel_values` as backbone input.",
                logger="current",
                level=logging.WARNING,
            )
            self._warned_extra_pixel_values = True

        extra_data_dict = {}
        seg_projected_tokens = None

        if "pixel_values" in data_dict and data_dict["pixel_values"] is not None:
            trunk_capture_layers = None
            if self.llm is not None:
                merged_layers = list(self.lang_bast_layers)
                if hasattr(self, "seg_projector"):
                    merged_layers.extend(self.seg_bast_layers)
                total_trunk_levels = len(self.segmentor.encoder.vision_backbone.trunk.blocks)
                trunk_capture_layers = self._normalize_layer_ids(
                    layer_ids=merged_layers,
                    total_levels=total_trunk_levels,
                    attr_name="trunk_capture_layers",
                )

            seg_visual_outputs, seg_image_embeddings = self._extract_segmentor_features(
                data_dict["pixel_values"], trunk_select_layers=trunk_capture_layers
            )

            if self.llm is not None:
                lang_features = self._select_trunk_feature_levels(
                    seg_visual_outputs=seg_visual_outputs,
                    layer_ids=self.lang_bast_layers,
                    attr_name="lang_bast_layers",
                )
                pixel_values = self._project_feature_levels(
                    feature_levels=lang_features,
                    projector=self.visual_projector,
                    target_dtype=self.llm.dtype,
                )

                if hasattr(self, "seg_projector"):
                    seg_features = self._select_trunk_feature_levels(
                        seg_visual_outputs=seg_visual_outputs,
                        layer_ids=self.seg_bast_layers,
                        attr_name="seg_bast_layers",
                    )
                    seg_projected_tokens = self._project_feature_levels(
                        feature_levels=seg_features,
                        projector=self.seg_projector,
                        target_dtype=self.llm.dtype,
                    )
                    if seg_projected_tokens is not None:
                        pixel_values = (
                            seg_projected_tokens
                            if pixel_values is None
                            else torch.cat([pixel_values, seg_projected_tokens], dim=1)
                        )

                data_dict["pixel_values"] = pixel_values
            else:
                data_dict["pixel_values"] = None

            extra_data_dict = {
                "extra_pixel_values": None,
                "seg_image_embeddings": seg_image_embeddings,
            }
        else:
            data_dict["pixel_values"] = None
            extra_data_dict = {
                "extra_pixel_values": None,
                "seg_image_embeddings": None,
            }

        # Keep compatibility with vision sampler route.
        data_dict["extra_pixel_values"] = seg_projected_tokens
        if data_dict.get("vprompt_masks", None) is not None and hasattr(self, "vision_sampler"):
            vprompt_masks = data_dict.pop("vprompt_masks")
            class_labels, contiguous_labels = self._get_vgd_labels(data_samples)
            sampled_labels = self._get_attrs_from_data_samples(data_samples, ["sampled_labels"])[0]

            sampler_input = data_dict.get(self.sampler_input_feat, None)
            if sampler_input is None and self.sampler_input_feat == "extra_pixel_values":
                sampler_input = data_dict.get("pixel_values", None)
            if sampler_input is None:
                raise ValueError(
                    f"Vision sampler input `{self.sampler_input_feat}` is None. "
                    "Set `sampler_input_feat='pixel_values'` or ensure projected tokens exist."
                )

            sampled_feats = self.vision_sampler(sampler_input, vprompt_masks)
            assert all(sampled_feat is not None for sampled_feat in sampled_feats), f"{sampler_input}, {vprompt_masks}"
            vprompt_feats, vprompt_masks, _ = self._get_vprompt_feats_and_masks(
                sampled_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
            )
            data_dict["vprompt_feats"] = vprompt_feats
            kwargs["vprompt_masks"] = vprompt_masks
            kwargs["sampled_labels"] = sampled_labels

        # Disable legacy dual-stream concat into LLM inputs.
        data_dict["extra_pixel_values"] = None
        if self.llm is not None:
            data_dict = prepare_inputs_labels_for_multimodal(llm=self.llm, **data_dict)

        data_dict.update(extra_data_dict)

        if mode == "loss":
            return self.compute_loss(data_dict, data_samples, **kwargs)
        elif mode == "predict":
            return self.predict(data_dict, data_samples, **kwargs)
        elif mode == "tensor":
            return self._forward(data_dict, data_samples, **kwargs)
        else:
            raise NotImplementedError
