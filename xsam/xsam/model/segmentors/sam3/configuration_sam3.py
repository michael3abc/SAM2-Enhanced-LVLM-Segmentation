# coding=utf-8
"""SAM3 model configuration."""

from transformers.configuration_utils import PretrainedConfig


class Sam3PromptEncoderConfig(PretrainedConfig):
    """Configuration for SAM3 prompt encoder placeholder."""

    base_config_key = "prompt_encoder_config"
    model_type = "sam3_prompt_encoder"

    def __init__(
        self,
        hidden_size: int = 256,
        **kwargs,
    ) -> None:
        """Initialize SAM3 prompt encoder config.

        Args:
            hidden_size: Hidden size used by prompt-related embeddings.
            **kwargs: Additional config arguments.
        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class Sam3MaskDecoderConfig(PretrainedConfig):
    """Configuration for SAM3 mask decoder placeholder."""

    base_config_key = "mask_decoder_config"
    model_type = "sam3_mask_decoder"

    def __init__(
        self,
        hidden_size: int = 256,
        **kwargs,
    ) -> None:
        """Initialize SAM3 mask decoder config.

        Args:
            hidden_size: Hidden size used by decoder-related modules.
            **kwargs: Additional config arguments.
        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class Sam3VisionConfig(PretrainedConfig):
    """Configuration for SAM3 vision trunk and simple FPN neck."""

    base_config_key = "vision_config"
    model_type = "sam3_vision_model"

    def __init__(
        self,
        img_size: int = 1008,
        pretrain_img_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1024,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.625,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        global_attn_indexes: list[int] | None = None,
        rel_pos_blocks: list[int] | None = None,
        use_rope: bool = True,
        use_interp_rope: bool = True,
        window_size: int = 24,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = False,
        ln_pre: bool = True,
        ln_post: bool = False,
        return_interm_layers: bool = False,
        bias_patch_embed: bool = False,
        neck_hidden_size: int = 256,
        neck_scale_factors: list[float] | None = None,
        add_sam2_neck: bool = False,
        scalp: int = 1,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        """Initialize SAM3 vision config.

        Args:
            img_size: Input image size used by SAM3 trunk.
            pretrain_img_size: Pretraining image size used by SAM3 trunk.
            patch_size: Patch size of ViT trunk.
            num_channels: Number of input channels.
            hidden_size: ViT embedding dimension.
            num_hidden_layers: Number of ViT blocks.
            num_attention_heads: Number of attention heads in ViT.
            mlp_ratio: ViT MLP ratio.
            drop_path_rate: Drop-path rate in ViT.
            qkv_bias: Whether to use QKV bias in ViT.
            use_abs_pos: Whether to use absolute positional embedding.
            tile_abs_pos: Whether to tile absolute positional embedding.
            global_attn_indexes: Indices of global-attention blocks.
            rel_pos_blocks: Indices of relative-position blocks.
            use_rope: Whether to use RoPE in ViT.
            use_interp_rope: Whether to interpolate RoPE.
            window_size: Window size for window attention blocks.
            pretrain_use_cls_token: Whether pretraining uses cls token.
            retain_cls_token: Whether to retain cls token at output.
            ln_pre: Whether to apply pre LayerNorm.
            ln_post: Whether to apply post LayerNorm.
            return_interm_layers: Whether to return intermediate layers in trunk.
            bias_patch_embed: Whether patch embedding has bias.
            neck_hidden_size: Output hidden size of simple FPN neck.
            neck_scale_factors: Scale factors used by simple FPN neck.
            add_sam2_neck: Whether to clone an additional SAM2-style neck branch.
            scalp: Number of lowest-resolution levels to drop.
            layer_norm_eps: LayerNorm epsilon.
            **kwargs: Additional config arguments.
        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.img_size = img_size
        self.pretrain_img_size = pretrain_img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.tile_abs_pos = tile_abs_pos
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [7, 15, 23, 31]
        self.rel_pos_blocks = rel_pos_blocks if rel_pos_blocks is not None else []
        self.use_rope = use_rope
        self.use_interp_rope = use_interp_rope
        self.window_size = window_size
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.retain_cls_token = retain_cls_token
        self.ln_pre = ln_pre
        self.ln_post = ln_post
        self.return_interm_layers = return_interm_layers
        self.bias_patch_embed = bias_patch_embed
        self.neck_hidden_size = neck_hidden_size
        self.neck_scale_factors = (
            neck_scale_factors if neck_scale_factors is not None else [4.0, 2.0, 1.0, 0.5]
        )
        self.add_sam2_neck = add_sam2_neck
        self.scalp = scalp
        self.layer_norm_eps = layer_norm_eps


class Sam3Config(PretrainedConfig):
    """Main configuration for SAM3 model wrapper."""

    model_type = "sam3"

    def __init__(
        self,
        vision_config: Sam3VisionConfig | dict | None = None,
        prompt_encoder_config: Sam3PromptEncoderConfig | dict | None = None,
        mask_decoder_config: Sam3MaskDecoderConfig | dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize SAM3 config.

        Args:
            vision_config: Vision trunk+neck config.
            prompt_encoder_config: Prompt encoder config.
            mask_decoder_config: Mask decoder config.
            **kwargs: Additional config arguments.
        Returns:
            None.
        """
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            vision_config = Sam3VisionConfig(**vision_config)
        if isinstance(prompt_encoder_config, dict):
            prompt_encoder_config = Sam3PromptEncoderConfig(**prompt_encoder_config)
        if isinstance(mask_decoder_config, dict):
            mask_decoder_config = Sam3MaskDecoderConfig(**mask_decoder_config)

        self.vision_config = vision_config if vision_config is not None else Sam3VisionConfig()
        self.prompt_encoder_config = (
            prompt_encoder_config if prompt_encoder_config is not None else Sam3PromptEncoderConfig()
        )
        self.mask_decoder_config = mask_decoder_config if mask_decoder_config is not None else Sam3MaskDecoderConfig()
