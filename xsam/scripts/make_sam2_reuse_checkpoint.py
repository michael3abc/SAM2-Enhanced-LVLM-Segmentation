#!/usr/bin/env python3
"""Build a SAM2 warm-start checkpoint from a SAM mixed-finetune checkpoint."""

import argparse
import os
from collections import Counter
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_KEEP_PREFIXES = (
    "llm.",
    "visual_encoder.",
    "visual_projector.",
    "llm_projector.",
    "segmentor.decoder.",
    "segmentor.logit_scale",
    "bg_embeds.",
)

DEFAULT_DROP_PREFIXES = (
    "segmentor.encoder.",
    "segmentor.pixel_decoder.",
    "segmentor.shared_image_embedding",
    "seg_projector.",
    "seg_connector.",
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        None
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a reusable SAM2 warm-start checkpoint from a SAM checkpoint."
    )
    parser.add_argument("--src", required=True, help="Path to source checkpoint.")
    parser.add_argument("--dst", required=True, help="Path to output checkpoint.")
    parser.add_argument(
        "--include-pixel-decoder",
        action="store_true",
        help="Keep `segmentor.pixel_decoder.*` weights from source checkpoint.",
    )
    parser.add_argument(
        "--sam2-encoder-path",
        default=None,
        help="Optional SAM2 pretrained path (local dir or HF repo) to merge encoder/prompt_encoder weights.",
    )
    parser.add_argument(
        "--overwrite-sam2-keys",
        action="store_true",
        help="Overwrite existing keys when merging SAM2 encoder weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched key statistics without writing file.",
    )
    return parser.parse_args()


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint as a plain state dict.

    Args:
        path: Checkpoint path.
    Returns:
        Dict[str, torch.Tensor]: State dictionary.
    """
    raw_obj = torch.load(path, map_location="cpu")
    if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
        return raw_obj["state_dict"]
    if isinstance(raw_obj, dict):
        return raw_obj
    raise ValueError(f"Unsupported checkpoint object type: {type(raw_obj)} from {path}")


def _match_prefix(key: str, prefixes: Iterable[str]) -> bool:
    """Check whether a key starts with any given prefix.

    Args:
        key: Parameter key.
        prefixes: Prefix list.
    Returns:
        bool: True if any prefix matches.
    """
    return any(key.startswith(prefix) for prefix in prefixes)


def _filter_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter reusable keys for SAM2 migration.

    Args:
        state_dict: Source model state dictionary.
    Returns:
        Dict[str, torch.Tensor]: Filtered state dictionary.
    """
    output = {}
    for key, value in state_dict.items():
        if _match_prefix(key, DEFAULT_KEEP_PREFIXES) and not _match_prefix(key, DEFAULT_DROP_PREFIXES):
            output[key] = value
    return output


def _filter_state_dict_with_pixel_decoder(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter reusable keys for SAM2 migration and keep pixel decoder.

    Args:
        state_dict: Source model state dictionary.
    Returns:
        Dict[str, torch.Tensor]: Filtered state dictionary including pixel decoder.
    """
    output = _filter_state_dict(state_dict)
    for key, value in state_dict.items():
        if key.startswith("segmentor.pixel_decoder."):
            output[key] = value
    return output


def _print_stats(state_dict: Dict[str, torch.Tensor], title: str) -> None:
    """Print key count summary by first-level prefix.

    Args:
        state_dict: Model state dictionary.
        title: Label of this statistics block.
    Returns:
        None
    """
    counter = Counter(key.split(".", 1)[0] for key in state_dict)
    print(f"{title}:")
    print(f"  num_keys: {len(state_dict)}")
    print(f"  prefix_counts: {dict(counter)}")


def _merge_sam2_encoder_weights(
    state_dict: Dict[str, torch.Tensor], sam2_encoder_path: str, overwrite: bool
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Merge SAM2 vision/prompt encoder weights into X-SAM key space.

    Args:
        state_dict: Existing state dictionary in X-SAM key format.
        sam2_encoder_path: Path or repo id for `Sam2Model.from_pretrained`.
        overwrite: Whether to overwrite existing keys.
    Returns:
        Tuple[Dict[str, torch.Tensor], int]: Merged state dict and number of inserted/overwritten keys.
    """
    from transformers import Sam2Model

    sam2_model = Sam2Model.from_pretrained(
        sam2_encoder_path,
        local_files_only=os.path.isdir(sam2_encoder_path),
    )
    sam2_state_dict = sam2_model.state_dict()

    merged_state_dict = dict(state_dict)
    merged_count = 0
    for key, value in sam2_state_dict.items():
        if key.startswith("vision_encoder."):
            target_key = f"segmentor.encoder.{key[len('vision_encoder.'):]}"
        elif key.startswith("prompt_encoder."):
            target_key = f"segmentor.prompt_encoder.{key[len('prompt_encoder.'):]}"
        else:
            continue

        if overwrite or target_key not in merged_state_dict:
            merged_state_dict[target_key] = value.detach().cpu()
            merged_count += 1

    return merged_state_dict, merged_count


def main() -> None:
    """Run checkpoint conversion and save result.

    Args:
        None
    Returns:
        None
    """
    args = parse_args()
    if not os.path.isfile(args.src):
        raise FileNotFoundError(f"Source checkpoint does not exist: {args.src}")

    source_state_dict = _load_state_dict(args.src)
    if args.include_pixel_decoder:
        filtered_state_dict = _filter_state_dict_with_pixel_decoder(source_state_dict)
    else:
        filtered_state_dict = _filter_state_dict(source_state_dict)
    if args.sam2_encoder_path is not None:
        filtered_state_dict, merged_count = _merge_sam2_encoder_weights(
            filtered_state_dict,
            sam2_encoder_path=args.sam2_encoder_path,
            overwrite=args.overwrite_sam2_keys,
        )
        print(f"sam2_merged_keys: {merged_count}")

    _print_stats(source_state_dict, title="source")
    _print_stats(filtered_state_dict, title="filtered")

    if args.dry_run:
        return

    output_dir = os.path.dirname(args.dst)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(filtered_state_dict, args.dst)
    print(f"saved: {args.dst}")


if __name__ == "__main__":
    main()
