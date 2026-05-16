"""Model download manifest for X-SAM project."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    """Model download specification.

    Args:
        key: Stable model key used in CLI.
        repo_id: HuggingFace repository id.
        target_relpath: Target path relative to repository root.
        description: Human-readable summary.
    Returns:
        None
    """

    key: str
    repo_id: str
    target_relpath: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "phi3": ModelSpec(
        key="phi3",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        target_relpath="inits/Phi-3-mini-4k-instruct",
        description="Phi-3-mini-4k-instruct base model",
    ),
    "mask2former": ModelSpec(
        key="mask2former",
        repo_id="facebook/mask2former-swin-large-coco-panoptic",
        target_relpath="inits/mask2former-swin-large-coco-panoptic",
        description="Mask2Former Swin-L COCO panoptic checkpoint",
    ),
    "sam": ModelSpec(
        key="sam",
        repo_id="facebook/sam-vit-large",
        target_relpath="inits/sam-vit-large",
        description="SAM ViT-L checkpoint",
    ),
    "sam2": ModelSpec(
        key="sam2",
        repo_id="facebook/sam2.1-hiera-base-plus",
        target_relpath="inits/sam2.1-hiera-base-plus",
        description="SAM2.1 hiera base-plus checkpoint",
    ),
    "sam3": ModelSpec(
        key="sam3",
        repo_id="facebook/sam3",
        target_relpath="inits/sam3",
        description="SAM3 repository and checkpoints",
    ),
    "siglip2": ModelSpec(
        key="siglip2",
        repo_id="google/siglip2-so400m-patch14-384",
        target_relpath="inits/siglip2-so400m-patch14-384",
        description="SigLIP2 SO400M patch14-384 checkpoint",
    ),
    "xsam": ModelSpec(
        key="xsam",
        repo_id="hao9610/X-SAM",
        target_relpath="inits/X-SAM",
        description="Official X-SAM released weights",
    ),
}

DEFAULT_MODEL_KEYS: tuple[str, ...] = (
    "phi3",
    "mask2former",
    "sam",
    "sam2",
    "sam3",
    "siglip2",
    "xsam",
)


def list_default_models() -> list[str]:
    """Return default model keys in download order.

    Args:
        None
    Returns:
        list[str]: Default model keys.
    """

    return list(DEFAULT_MODEL_KEYS)


def normalize_model_names(raw_names: Iterable[str]) -> list[str]:
    """Normalize model names while preserving order and uniqueness.

    Args:
        raw_names: Raw model name iterable.
    Returns:
        list[str]: Normalized model names.
    """

    normalized: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        value = name.strip().lower()
        if not value:
            continue
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def parse_models_arg(raw_models: str) -> list[str]:
    """Parse model argument string.

    Args:
        raw_models: Comma-separated model keys or all.
    Returns:
        list[str]: Parsed model keys.
    """

    names = normalize_model_names(raw_models.split(","))
    if not names:
        raise ValueError("No model is specified.")
    if "all" in names:
        return list_default_models()
    return names


def resolve_model_specs(model_names: Sequence[str]) -> list[ModelSpec]:
    """Resolve model keys to model specs.

    Args:
        model_names: Parsed model keys.
    Returns:
        list[ModelSpec]: Resolved model specs.
    """

    unknown = [name for name in model_names if name not in MODEL_SPECS]
    if unknown:
        valid = ",".join(list_default_models())
        raise KeyError(f"Unknown model keys: {unknown}. Valid keys: all,{valid}")
    return [MODEL_SPECS[name] for name in model_names]


def format_specs_tsv(model_specs: Sequence[ModelSpec]) -> str:
    """Format model specs to TSV string.

    Args:
        model_specs: Model specs.
    Returns:
        str: TSV formatted lines.
    """

    lines = [
        f"{spec.key}\t{spec.repo_id}\t{spec.target_relpath}\t{spec.description}"
        for spec in model_specs
    ]
    return "\n".join(lines)


def format_specs_json(model_specs: Sequence[ModelSpec]) -> str:
    """Format model specs to JSON string.

    Args:
        model_specs: Model specs.
    Returns:
        str: JSON formatted output.
    """

    payload = [asdict(spec) for spec in model_specs]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Args:
        None
    Returns:
        argparse.ArgumentParser: CLI parser.
    """

    parser = argparse.ArgumentParser(description="Model manifest for X-SAM downloads.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model keys. Use all for full list.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="tsv",
        choices=("tsv", "json"),
        help="Output format.",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print valid model keys and exit.",
    )
    return parser


def main() -> int:
    """Run CLI entrypoint.

    Args:
        None
    Returns:
        int: Exit code.
    """

    args = build_arg_parser().parse_args()
    if args.list_keys:
        print(",".join(list_default_models()))
        return 0

    specs = resolve_model_specs(parse_models_arg(args.models))
    if args.format == "json":
        print(format_specs_json(specs))
    else:
        print(format_specs_tsv(specs))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[manifest] {exc}", file=sys.stderr)
        raise


# CLI examples:
# python scripts/download/model/manifest.py --models all --format tsv
# python scripts/download/model/manifest.py --models phi3,sam3,xsam --format json
