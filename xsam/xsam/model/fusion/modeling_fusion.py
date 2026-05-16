"""Optional token fusion blocks for X-SAM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


_FUSION_TYPES = {"none", "concat", "joint_cross_attention"}


def _validate_rank3(name: str, value: Optional[torch.Tensor]) -> None:
    """Validate a token tensor with shape `[B, N, D]`.

    Args:
        name: Tensor name used in error messages.
        value: Tensor to validate, or ``None``.

    Returns:
        None.
    """
    if value is None:
        return
    if value.ndim != 3:
        raise ValueError(f"`{name}` must be rank-3 `[B, N, D]`. Got shape: {tuple(value.shape)}")


def _concat_context(
    visual_tokens: torch.Tensor,
    sam_tokens: Optional[torch.Tensor],
    question_tokens: Optional[torch.Tensor],
    *,
    include_visual: bool,
) -> torch.Tensor:
    """Concatenate available context tokens.

    Args:
        visual_tokens: Visual token tensor `[B, N, D]`.
        sam_tokens: Optional SAM token tensor `[B, M, D]`.
        question_tokens: Optional question token tensor `[B, Q, D]`.
        include_visual: Whether to prepend visual tokens to the context.

    Returns:
        Concatenated context tensor `[B, K, D]`.
    """
    pieces = [visual_tokens] if include_visual else []
    if sam_tokens is not None:
        pieces.append(sam_tokens)
    if question_tokens is not None:
        pieces.append(question_tokens)
    if not pieces:
        return visual_tokens.new_empty(visual_tokens.shape[0], 0, visual_tokens.shape[-1])
    return torch.cat(pieces, dim=1)


def _build_key_padding_mask(attention_mask: Optional[torch.Tensor], context_tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """Build a key padding mask for PyTorch multi-head attention.

    Args:
        attention_mask: Optional mask where truthy values mean keep.
        context_tokens: Context token tensor `[B, K, D]`.

    Returns:
        Boolean key padding mask where ``True`` means ignore, or ``None``.
    """
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(f"`attention_mask` must be rank-2 `[B, K]`. Got shape: {tuple(attention_mask.shape)}")
    if attention_mask.shape[0] != context_tokens.shape[0] or attention_mask.shape[1] != context_tokens.shape[1]:
        raise ValueError(
            "`attention_mask` shape must match context token shape. "
            f"Got mask={tuple(attention_mask.shape)}, context={tuple(context_tokens.shape[:2])}."
        )
    return ~attention_mask.to(dtype=torch.bool, device=context_tokens.device)


def _masked_mean(tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool tokens with an optional keep mask.

    Args:
        tokens: Token tensor `[B, K, D]`.
        attention_mask: Optional mask `[B, K]` where truthy values mean keep.

    Returns:
        Pooled token tensor `[B, D]`.
    """
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    if attention_mask is None:
        return tokens.mean(dim=1)
    if attention_mask.ndim != 2:
        raise ValueError(f"`attention_mask` must be rank-2 `[B, K]`. Got shape: {tuple(attention_mask.shape)}")
    if attention_mask.shape[0] != tokens.shape[0] or attention_mask.shape[1] != tokens.shape[1]:
        raise ValueError(
            "`attention_mask` shape must match token shape. "
            f"Got mask={tuple(attention_mask.shape)}, tokens={tuple(tokens.shape[:2])}."
        )
    weights = attention_mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class FusionModel(nn.Module):
    """Fuse visual, SAM, and optional question tokens while preserving visual shape."""

    def __init__(
        self,
        hidden_size: int,
        fusion_type: str = "none",
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        """Initialize a fusion module.

        Args:
            hidden_size: Token hidden dimension.
            fusion_type: One of ``none``, ``concat``, or ``joint_cross_attention``.
            num_attention_heads: Number of attention heads for cross-attention fusion.
            dropout: Dropout probability for projected fusion residuals.
            gate_init: Initial scalar gate value for non-identity fusion paths.

        Returns:
            None.
        """
        super().__init__()
        normalized_type = str(fusion_type).strip().lower()
        if normalized_type not in _FUSION_TYPES:
            raise ValueError(f"`fusion_type` must be one of {sorted(_FUSION_TYPES)}. Got: {fusion_type}")
        if hidden_size <= 0:
            raise ValueError(f"`hidden_size` must be positive. Got: {hidden_size}")
        if normalized_type == "joint_cross_attention" and hidden_size % num_attention_heads != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `num_attention_heads` for joint cross-attention. "
                f"Got hidden_size={hidden_size}, heads={num_attention_heads}."
            )

        self.hidden_size = int(hidden_size)
        self.fusion_type = normalized_type
        self.dropout = nn.Dropout(float(dropout))

        if self.fusion_type == "concat":
            self.context_norm = nn.LayerNorm(hidden_size)
            self.context_proj = nn.Linear(hidden_size, hidden_size)
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        elif self.fusion_type == "joint_cross_attention":
            self.query_norm = nn.LayerNorm(hidden_size)
            self.context_norm = nn.LayerNorm(hidden_size)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        sam_tokens: Optional[torch.Tensor] = None,
        question_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse input tokens and return visual-shaped tokens.

        Args:
            visual_tokens: Base visual tokens `[B, N, D]`.
            sam_tokens: Optional SAM tokens `[B, M, D]`.
            question_tokens: Optional text/question tokens `[B, Q, D]`.
            attention_mask: Optional keep mask for the fusion context.

        Returns:
            Fused visual tokens with shape `[B, N, D]`.
        """
        _validate_rank3("visual_tokens", visual_tokens)
        _validate_rank3("sam_tokens", sam_tokens)
        _validate_rank3("question_tokens", question_tokens)

        if sam_tokens is not None and sam_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError("`sam_tokens` batch size must match `visual_tokens`.")
        if question_tokens is not None and question_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError("`question_tokens` batch size must match `visual_tokens`.")
        if sam_tokens is not None and sam_tokens.shape[-1] != visual_tokens.shape[-1]:
            raise ValueError("`sam_tokens` hidden size must match `visual_tokens`.")
        if question_tokens is not None and question_tokens.shape[-1] != visual_tokens.shape[-1]:
            raise ValueError("`question_tokens` hidden size must match `visual_tokens`.")

        if self.fusion_type == "none":
            return visual_tokens
        if self.fusion_type == "concat":
            return self._forward_concat(visual_tokens, sam_tokens, question_tokens, attention_mask)
        return self._forward_joint_cross_attention(visual_tokens, sam_tokens, question_tokens, attention_mask)

    def _forward_concat(
        self,
        visual_tokens: torch.Tensor,
        sam_tokens: Optional[torch.Tensor],
        question_tokens: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pooled concat-style residual fusion.

        Args:
            visual_tokens: Base visual tokens `[B, N, D]`.
            sam_tokens: Optional SAM tokens `[B, M, D]`.
            question_tokens: Optional text/question tokens `[B, Q, D]`.
            attention_mask: Optional keep mask over SAM plus question context.

        Returns:
            Fused visual tokens `[B, N, D]`.
        """
        context_tokens = _concat_context(visual_tokens, sam_tokens, question_tokens, include_visual=False)
        pooled = _masked_mean(context_tokens, attention_mask)
        residual = self.context_proj(self.context_norm(pooled)).unsqueeze(1)
        return visual_tokens + torch.tanh(self.gate) * self.dropout(residual)

    def _forward_joint_cross_attention(
        self,
        visual_tokens: torch.Tensor,
        sam_tokens: Optional[torch.Tensor],
        question_tokens: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply HoP-style joint cross-attention fusion.

        Args:
            visual_tokens: Base visual tokens `[B, N, D]`.
            sam_tokens: Optional SAM tokens `[B, M, D]`.
            question_tokens: Optional text/question tokens `[B, Q, D]`.
            attention_mask: Optional keep mask over `[visual, sam, question]` context.

        Returns:
            Fused visual tokens `[B, N, D]`.
        """
        context_tokens = _concat_context(visual_tokens, sam_tokens, question_tokens, include_visual=True)
        key_padding_mask = _build_key_padding_mask(attention_mask, context_tokens)
        residual, _ = self.cross_attn(
            query=self.query_norm(visual_tokens),
            key=self.context_norm(context_tokens),
            value=context_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return visual_tokens + torch.tanh(self.gate) * self.dropout(residual)
