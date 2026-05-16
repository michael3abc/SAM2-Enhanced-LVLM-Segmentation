#!/usr/bin/env python3
"""Split llava_v1_5_mix665k.json into per-source sub-dataset JSONs.

Reads the original 665k JSON once and writes 5 separate JSON files based on
the image path prefix: coco, vg, gqa, ocr_vqa, textvqa.
Entries without an 'image' field (pure text) are skipped.

Usage:
    python scripts/split_llava_mix665k.py \
        --input data/imgconv_data/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
        --output-dir data/imgconv_data/llava/LLaVA-Instruct-150K/
"""

import argparse
import json
import os
from collections import defaultdict


# Map image-path prefix -> output filename stem
PREFIX_MAP = {
    "coco": "llava_imgconv_coco",
    "vg": "llava_imgconv_vg",
    "gqa": "llava_imgconv_gqa",
    "ocr_vqa": "llava_imgconv_ocr_vqa",
    "textvqa": "llava_imgconv_textvqa",
}


def classify(image_path: str) -> str | None:
    """Classify an image path into one of the known sources.

    Args:
        image_path: The 'image' field value, e.g. 'coco/train2017/000000033471.jpg'.

    Returns:
        Source key string or None if unrecognized.
    """
    for prefix in PREFIX_MAP:
        if image_path.startswith(prefix + "/"):
            return prefix
    return None


def main():
    """Split the LLaVA mix665k JSON into per-source sub-dataset files."""
    parser = argparse.ArgumentParser(description="Split llava_v1_5_mix665k.json by source.")
    parser.add_argument("--input", required=True, help="Path to llava_v1_5_mix665k.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write output JSONs")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.")

    buckets: dict[str, list] = defaultdict(list)
    skipped = 0
    unknown = 0

    for entry in data:
        img = entry.get("image")
        if img is None:
            skipped += 1
            continue
        source = classify(img)
        if source is None:
            unknown += 1
            continue
        buckets[source].append(entry)

    print(f"Skipped (no image): {skipped}")
    print(f"Unknown prefix: {unknown}")

    os.makedirs(args.output_dir, exist_ok=True)
    for source, stem in PREFIX_MAP.items():
        out_path = os.path.join(args.output_dir, f"{stem}.json")
        entries = buckets.get(source, [])
        print(f"Writing {out_path} ... ({len(entries)} entries)")
        with open(out_path, "w") as f:
            json.dump(entries, f)

    print("Done!")


if __name__ == "__main__":
    main()
