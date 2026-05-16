#!/usr/bin/env python3
"""Centralized dataset manifest for dataset download pipeline."""

from __future__ import annotations

import argparse
from typing import Iterable

COCO2017_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "panoptic_annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
}

COCO2014_URLS = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014.zip": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

REFSEG_URLS = {
    "refcoco.zip": "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    "refcoco+.zip": "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "refcocog.zip": "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
}

LLAVA_IMAGE_URLS = {
    "vg_images.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "vg_images2.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    "textvqa_train_val_images.zip": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "ocr_vqa_images_llava_v15.zip": "https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip",
    "gqa_images.zip": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
}

VGD_ANN_URLS = {
    "coco_vgdseg_train.json": "https://huggingface.co/hao9610/X-SAM/resolve/main/vgdseg_annotations/coco_vgdseg_train.json",
    "coco_vgdseg_val.json": "https://huggingface.co/hao9610/X-SAM/resolve/main/vgdseg_annotations/coco_vgdseg_val.json",
}

DEFAULT_DATASET_ORDER: tuple[str, ...] = (
    "coco",
    "ovseg",
    "refseg",
    "reaseg",
    "gcgseg",
    "intseg",
    "vgdseg",
    "llava",
    "lmu",
)


def list_default_datasets() -> list[str]:
    """Return default dataset order.

    Args:
        None.

    Returns:
        list[str]: Default dataset names.
    """

    return list(DEFAULT_DATASET_ORDER)


def normalize_dataset_names(dataset_names: Iterable[str]) -> list[str]:
    """Normalize dataset names and remove duplicates.

    Args:
        dataset_names: Raw dataset name iterable.

    Returns:
        list[str]: Lowercased deduplicated provider list.
    """

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_name in dataset_names:
        name = raw_name.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def list_default_providers() -> list[str]:
    """Backward-compatible alias for old provider API.

    Args:
        None.

    Returns:
        list[str]: Default dataset names.
    """

    return list_default_datasets()


def normalize_provider_names(provider_names: Iterable[str]) -> list[str]:
    """Backward-compatible alias for old provider API.

    Args:
        provider_names: Raw provider names.

    Returns:
        list[str]: Normalized dataset names.
    """

    return normalize_dataset_names(provider_names)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: Parser object.
    """

    parser = argparse.ArgumentParser(description="Show dataset manifest summary.")
    parser.add_argument("--list-datasets", action="store_true", help="List default dataset order.")
    parser.add_argument("--list-providers", action="store_true", help="Alias of --list-datasets.")
    parser.add_argument("--list-urls", action="store_true", help="List URL key summary for each dataset group.")
    return parser


def main() -> int:
    """Main entrypoint.

    Args:
        None.

    Returns:
        int: Exit code.
    """

    args = build_arg_parser().parse_args()
    if args.list_providers:
        args.list_datasets = True

    if not args.list_datasets and not args.list_urls:
        args.list_datasets = True
        args.list_urls = True

    if args.list_datasets:
        print("datasets:", ",".join(list_default_datasets()))

    if args.list_urls:
        print("coco2017:", ",".join(COCO2017_URLS.keys()))
        print("coco2014:", ",".join(COCO2014_URLS.keys()))
        print("refseg:", ",".join(REFSEG_URLS.keys()))
        print("llava:", ",".join(LLAVA_IMAGE_URLS.keys()))
        print("vgdseg:", ",".join(VGD_ANN_URLS.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/dataset/manifest.py --list-datasets --list-urls
"""
