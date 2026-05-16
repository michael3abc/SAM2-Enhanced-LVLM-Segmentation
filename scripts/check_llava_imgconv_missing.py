#!/usr/bin/env python3
"""Check missing image files referenced by LLaVA imgconv annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Check missing files in LLaVA imgconv json.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("data/imgconv_data/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"),
        help="Path to LLaVA annotation json.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/imgconv_data/llava/llava_images"),
        help="Path to llava_images root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max records to scan. 0 means full scan.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="Show first N missing files.",
    )
    return parser.parse_args()


def load_records(json_path: Path) -> List[Dict]:
    """Load records from annotation json.

    Args:
        json_path (Path): Json file path.
    Returns:
        List[Dict]: Annotation records.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list json, got: {type(data)}")
    return data


def image_source(image_rel_path: str) -> str:
    """Infer source name from relative image path.

    Args:
        image_rel_path (str): Relative image path in annotation.
    Returns:
        str: Source prefix (first path segment).
    """
    parts = image_rel_path.split("/")
    return parts[0] if parts else "unknown"


def scan_missing(
    records: Iterable[Dict], image_root: Path, max_samples: int = 0
) -> Tuple[int, int, Counter, List[str]]:
    """Scan missing files and summarize by source.

    Args:
        records (Iterable[Dict]): Annotation records.
        image_root (Path): Root image folder.
        max_samples (int): Max samples to scan, 0 for full scan.
    Returns:
        Tuple[int, int, Counter, List[str]]: total, missing count, source counter, missing paths.
    """
    total = 0
    missing = 0
    by_source: Counter = Counter()
    missing_paths: List[str] = []
    for rec in records:
        if max_samples > 0 and total >= max_samples:
            break
        total += 1
        image_rel = rec.get("image")
        if not image_rel:
            continue
        image_abs = image_root / image_rel
        if not image_abs.is_file():
            missing += 1
            src = image_source(str(image_rel))
            by_source[src] += 1
            missing_paths.append(str(image_abs))
    return total, missing, by_source, missing_paths


def main() -> None:
    """Run missing-file checker.

    Args:
        None
    Returns:
        None
    """
    args = parse_args()
    if not args.json.is_file():
        raise FileNotFoundError(f"Json not found: {args.json}")
    if not args.image_root.is_dir():
        raise NotADirectoryError(f"Image root not found: {args.image_root}")

    records = load_records(args.json)
    total, missing, by_source, missing_paths = scan_missing(records, args.image_root, args.max_samples)

    print(f"[INFO] json={args.json}")
    print(f"[INFO] image_root={args.image_root}")
    print(f"[INFO] scanned={total}")
    print(f"[INFO] missing={missing}")
    if total > 0:
        print(f"[INFO] missing_ratio={missing / total:.6f}")

    print("[INFO] missing_by_source:")
    if by_source:
        for src, cnt in by_source.most_common():
            print(f"  - {src}: {cnt}")
    else:
        print("  - none")

    n = max(0, args.show_missing)
    if n > 0 and missing_paths:
        print(f"[INFO] first_{min(n, len(missing_paths))}_missing_paths:")
        for p in missing_paths[:n]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
