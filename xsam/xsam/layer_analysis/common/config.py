"""Config helpers for layer-sweep pipelines."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config file into a dictionary.

    Args:
        path: YAML file path.
    Returns:
        Parsed config dictionary.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(loaded)}.")
    return loaded


def resolve_phase_config(raw_cfg: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """Resolve merged phase config from ``common`` and ``phases`` sections.

    Args:
        raw_cfg: Raw config dictionary.
        phase: Phase name.
    Returns:
        Merged phase config dictionary.
    """
    common_cfg = raw_cfg.get("common", {})
    if not isinstance(common_cfg, dict):
        raise TypeError("`common` must be a mapping.")

    phase_map = raw_cfg.get("phases", {})
    if not isinstance(phase_map, dict):
        raise TypeError("`phases` must be a mapping.")
    if phase not in phase_map:
        raise KeyError(f"Phase `{phase}` is missing in config.")
    if not isinstance(phase_map[phase], dict):
        raise TypeError(f"`phases.{phase}` must be a mapping.")

    merged = deep_merge_dict(common_cfg, phase_map[phase])
    for key in ["paths", "runtime", "runner"]:
        section = raw_cfg.get(key)
        if isinstance(section, dict):
            merged = deep_merge_dict(section, merged)
    merged["phase"] = phase
    return merged
