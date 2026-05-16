"""Shared helpers for profile-driven X-SAM config parsing."""

from __future__ import annotations

from typing import Any, Mapping


_TRUNK_MODES = {"freeze", "lora", "full_ft"}
_FPN_MODES = {"freeze", "full_ft"}


def _normalize_mode_value(raw_value: Any, allowed_values: set[str], field_name: str) -> str:
    """Normalize and validate a profile mode string.

    Args:
        raw_value: Raw profile value.
        allowed_values: Allowed string values.
        field_name: Field name for error messages.

    Returns:
        Normalized lower-case mode string.
    """
    mode = str(raw_value).strip().lower()
    if mode not in allowed_values:
        raise ValueError(f"`{field_name}` must be one of {sorted(allowed_values)}. Got: {raw_value}")
    return mode


def _resolve_trunk_mode(profile: Mapping[str, Any], default_mode: str) -> str:
    """Resolve segmentor trunk mode with backward-compatible fallbacks.

    Args:
        profile: Raw or parsed profile mapping.
        default_mode: Default mode when neither new nor legacy fields exist.

    Returns:
        Resolved trunk mode string.
    """
    if "segmentor_trunk_mode" in profile:
        return _normalize_mode_value(profile["segmentor_trunk_mode"], _TRUNK_MODES, "segmentor_trunk_mode")

    if "freeze_segmentor_trunk" in profile:
        return "freeze" if bool(profile["freeze_segmentor_trunk"]) else "full_ft"

    if "freeze_segmentor_encoder" in profile:
        return "freeze" if bool(profile["freeze_segmentor_encoder"]) else "full_ft"

    return _normalize_mode_value(default_mode, _TRUNK_MODES, "segmentor_trunk_mode")


def _resolve_fpn_mode(profile: Mapping[str, Any], default_mode: str) -> str:
    """Resolve segmentor FPN mode with backward-compatible fallbacks.

    Args:
        profile: Raw or parsed profile mapping.
        default_mode: Default mode when neither new nor legacy fields exist.

    Returns:
        Resolved FPN mode string.
    """
    if "segmentor_fpn_mode" in profile:
        return _normalize_mode_value(profile["segmentor_fpn_mode"], _FPN_MODES, "segmentor_fpn_mode")

    if "freeze_segmentor_fpn" in profile:
        return "freeze" if bool(profile["freeze_segmentor_fpn"]) else "full_ft"

    if "freeze_segmentor_encoder" in profile:
        return "freeze" if bool(profile["freeze_segmentor_encoder"]) else "full_ft"

    return _normalize_mode_value(default_mode, _FPN_MODES, "segmentor_fpn_mode")


def build_segmentor_train_policy(
    profile: Mapping[str, Any],
    *,
    default_trunk_mode: str,
    default_fpn_mode: str,
) -> dict[str, Any]:
    """Build normalized segmentor train policy from a profile mapping.

    Args:
        profile: Raw or parsed profile mapping.
        default_trunk_mode: Default trunk mode for this config/stage.
        default_fpn_mode: Default FPN mode for this config/stage.

    Returns:
        Normalized segmentor train policy dictionary.
    """
    trunk_mode = _resolve_trunk_mode(profile, default_trunk_mode)
    fpn_mode = _resolve_fpn_mode(profile, default_fpn_mode)

    target_modules = profile.get("segmentor_lora_target_modules")
    if target_modules is not None:
        if isinstance(target_modules, str):
            target_modules = [token.strip() for token in target_modules.split(",") if token.strip()]
        elif isinstance(target_modules, (list, tuple)):
            target_modules = [str(token).strip() for token in target_modules if str(token).strip()]
        else:
            raise TypeError(
                "`segmentor_lora_target_modules` must be null, string, or list/tuple of strings. "
                f"Got: {type(target_modules)!r}"
            )
        if len(target_modules) == 0:
            target_modules = None

    lora_kwargs = None
    if trunk_mode == "lora":
        lora_kwargs = dict(
            task_type="FEATURE_EXTRACTION",
            r=int(profile.get("segmentor_lora_rank", 16)),
            lora_alpha=int(profile.get("segmentor_lora_alpha", 32)),
            lora_dropout=float(profile.get("segmentor_lora_dropout", 0.05)),
            bias=str(profile.get("segmentor_lora_bias", "none")),
            target_modules=target_modules,
        )

    return dict(
        segmentor_trunk_mode=trunk_mode,
        segmentor_fpn_mode=fpn_mode,
        freeze_segmentor_trunk=(trunk_mode == "freeze"),
        freeze_segmentor_fpn=(fpn_mode == "freeze"),
        freeze_segmentor_encoder=(trunk_mode == "freeze" and fpn_mode == "freeze"),
        segmentor_lora_kwargs=lora_kwargs,
    )
