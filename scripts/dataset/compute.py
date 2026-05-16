#!/usr/bin/env python3
"""CPU mapping tasks for dataset preparation pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


class BaseComputeTask:
    """Base class for CPU mapping tasks.

    Args:
        None.
    Returns:
        None.
    """

    task_name = "base"

    def run(self, ctx: Any) -> None:
        """Run compute task.

        Args:
            ctx: Runtime context with fields `root_dir`, `data_dir`, `overwrite`, `dry_run`.
        Returns:
            None.
        """

        raise NotImplementedError


@dataclass(frozen=True)
class _SplitSpec:
    """Container for dataset split spec.

    Args:
        split_name: Split key.
        expected_count: Expected number of files.
    Returns:
        None.
    """

    split_name: str
    expected_count: int


def _load_json_dict(path: Path) -> dict:
    """Load JSON file and enforce dict root.

    Args:
        path: JSON file path.
    Returns:
        dict: Parsed object.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return data


def _collect_file_stems(directory: Path, suffix: str) -> set[str]:
    """Collect file stems by suffix in directory.

    Args:
        directory: Target directory.
        suffix: File suffix including dot.
    Returns:
        set[str]: Stem set.
    """

    if not directory.exists() or not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob(f"*{suffix}") if path.is_file()}


def _panoptic_rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    """Convert panoptic RGB to segment id map.

    Args:
        rgb: RGB array in uint8 with shape [H, W, 3].
    Returns:
        np.ndarray: Segment id map in int32 shape [H, W].
    """

    rgb32 = rgb.astype(np.int32)
    return rgb32[:, :, 0] + 256 * rgb32[:, :, 1] + 256 * 256 * rgb32[:, :, 2]


class CocoSemanticMapCompute(BaseComputeTask):
    """Generate COCO panoptic semantic maps from panoptic PNG and JSON."""

    task_name = "coco_semantic_map"

    def _needs_generation(self, coco_root: Path) -> bool:
        """Check whether semantic map generation is required.

        Args:
            coco_root: COCO2017 root.
        Returns:
            bool: True if generation is needed.
        """

        split = "val2017"
        ann_path = coco_root / "annotations" / f"panoptic_{split}.json"
        panoptic_dir = coco_root / f"panoptic_{split}"
        semseg_dir = coco_root / f"panoptic_semseg_{split}"
        if not ann_path.exists() or not panoptic_dir.is_dir():
            return False
        data = _load_json_dict(ann_path)
        anns = data.get("annotations", [])
        if not isinstance(anns, list) or len(anns) == 0:
            return False
        expected_stems = {Path(item["file_name"]).stem for item in anns if "file_name" in item}
        existing_stems = _collect_file_stems(semseg_dir, ".png")
        return len(expected_stems - existing_stems) > 0

    def run(self, ctx: Any) -> None:
        """Run COCO semantic map generation.

        Args:
            ctx: Runtime context.
        Returns:
            None.
        """

        coco_root = Path(ctx.data_dir) / "coco" / "coco2017"
        if not coco_root.exists():
            return

        if not getattr(ctx, "overwrite", False) and not self._needs_generation(coco_root):
            return

        split = "val2017"
        ann_path = coco_root / "annotations" / f"panoptic_{split}.json"
        panoptic_dir = coco_root / f"panoptic_{split}"
        semseg_dir = coco_root / f"panoptic_semseg_{split}"
        if not ann_path.exists() or not panoptic_dir.is_dir():
            return

        if getattr(ctx, "dry_run", False):
            print(f"[DRY-RUN] compute {self.task_name} on {split}")
            return

        semseg_dir.mkdir(parents=True, exist_ok=True)
        data = _load_json_dict(ann_path)
        anns = data.get("annotations", [])
        if not isinstance(anns, list):
            raise RuntimeError(f"Invalid panoptic annotation schema: {ann_path}")

        iterator = tqdm(anns, desc=f"COCO semantic {split}", unit="img", dynamic_ncols=True)
        for ann in iterator:
            file_name = ann.get("file_name")
            if not isinstance(file_name, str):
                continue
            panoptic_path = panoptic_dir / file_name
            semseg_path = semseg_dir / file_name
            if semseg_path.exists() and not getattr(ctx, "overwrite", False):
                continue

            panoptic_rgb = np.asarray(Image.open(panoptic_path).convert("RGB"), dtype=np.uint8)
            panoptic_id_map = _panoptic_rgb_to_id(panoptic_rgb)
            semantic_map = np.zeros(panoptic_id_map.shape, dtype=np.uint8)
            for seg in ann.get("segments_info", []):
                segment_id = seg.get("id")
                category_id = seg.get("category_id")
                if segment_id is None or category_id is None:
                    continue
                semantic_map[panoptic_id_map == int(segment_id)] = int(category_id)
            Image.fromarray(semantic_map).save(semseg_path)


class Ade20KInstanceMapCompute(BaseComputeTask):
    """Rebuild ADE20K `annotations_instance` PNG maps from instance JSON."""

    task_name = "ade20k_instance_map"

    def _mapping_file(self, ctx: Any) -> Path:
        """Resolve ADE20K category mapping file path.

        Args:
            ctx: Runtime context.
        Returns:
            Path: Mapping file path.
        """

        return Path(ctx.root_dir) / "xsam" / "xsam" / "tools" / "dataset_tools" / "ade20k_instance_catid_mapping.txt"

    def _needs_rebuild(self, ade_root: Path) -> bool:
        """Check whether ADE instance png needs rebuild.

        Args:
            ade_root: ADE20K root.
        Returns:
            bool: True if rebuild is needed.
        """

        split_specs = [_SplitSpec("training", 20210), _SplitSpec("validation", 2000)]
        for spec in split_specs:
            image_dir = ade_root / "images" / spec.split_name
            instance_dir = ade_root / "annotations_instance" / spec.split_name
            if not image_dir.is_dir() or not instance_dir.is_dir():
                return True
            image_stems = _collect_file_stems(image_dir, ".jpg")
            instance_stems = _collect_file_stems(instance_dir, ".png")
            if len(image_stems) != spec.expected_count or len(instance_stems) != spec.expected_count:
                return True
            if len(image_stems - instance_stems) > 0:
                return True
        return False

    def rebuild_from_json(self, ctx: Any, ade_root: Path) -> None:
        """Rebuild ADE20K instance maps from json.

        Args:
            ctx: Runtime context.
            ade_root: ADE20K root.
        Returns:
            None.
        """

        try:
            from pycocotools import mask as mask_utils
        except ModuleNotFoundError as exc:
            raise RuntimeError("pycocotools is required for ADE20K instance rebuild.") from exc

        mapping_file = self._mapping_file(ctx)
        if not mapping_file.exists():
            raise FileNotFoundError(f"Cannot find ADE20K mapping file: {mapping_file}")

        semantic_to_instance: dict[int, int] = {}
        with mapping_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                ins_id, sem_id, _ = line.strip().split()
                semantic_to_instance[int(sem_id) - 1] = int(ins_id)

        split_alias = {"training": "train", "validation": "val"}
        for split, alias in split_alias.items():
            json_path = ade_root / f"ade20k_instance_{alias}.json"
            if not json_path.exists():
                continue
            data = _load_json_dict(json_path)
            images = data.get("images", [])
            annotations = data.get("annotations", [])
            if not isinstance(images, list) or not isinstance(annotations, list):
                raise RuntimeError(f"Invalid ADE20K instance json schema: {json_path}")

            out_dir = ade_root / "annotations_instance" / split
            if not getattr(ctx, "dry_run", False):
                out_dir.mkdir(parents=True, exist_ok=True)

            anns_by_image: dict[str, list[dict]] = {}
            for ann in annotations:
                image_id = str(ann.get("image_id"))
                anns_by_image.setdefault(image_id, []).append(ann)

            iterator = tqdm(images, desc=f"ADE instance {split}", unit="img", dynamic_ncols=True)
            for item in iterator:
                image_id = str(item.get("id"))
                file_name = str(item.get("file_name", ""))
                stem = Path(file_name).stem
                if not stem:
                    continue

                dst_path = out_dir / f"{stem}.png"
                if dst_path.exists() and not getattr(ctx, "overwrite", False):
                    continue
                if getattr(ctx, "dry_run", False):
                    continue

                height = int(item.get("height", 0))
                width = int(item.get("width", 0))
                if height <= 0 or width <= 0:
                    continue

                cat_map = np.zeros((height, width), dtype=np.uint8)
                ins_map = np.zeros((height, width), dtype=np.uint8)
                next_ins_id = 1
                for ann in anns_by_image.get(image_id, []):
                    segmentation = ann.get("segmentation")
                    if segmentation is None:
                        continue
                    decoded = mask_utils.decode(segmentation)
                    if decoded.ndim == 3:
                        decoded = decoded[:, :, 0]
                    mask = decoded.astype(bool)
                    if not np.any(mask):
                        continue

                    semantic_id = int(ann.get("category_id"))
                    instance_cat = semantic_to_instance.get(semantic_id)
                    if instance_cat is None:
                        continue
                    if next_ins_id > 255:
                        break

                    cat_map[mask] = np.uint8(instance_cat)
                    ins_map[mask] = np.uint8(next_ins_id)
                    next_ins_id += 1

                instance_png = np.stack([cat_map, ins_map], axis=-1)
                Image.fromarray(instance_png).save(dst_path)

    def run(self, ctx: Any) -> None:
        """Run ADE instance map rebuild.

        Args:
            ctx: Runtime context.
        Returns:
            None.
        """

        ade_root = Path(ctx.data_dir) / "ovseg_data" / "ade20k"
        if not ade_root.exists():
            return
        if not getattr(ctx, "overwrite", False) and not self._needs_rebuild(ade_root):
            return
        if getattr(ctx, "dry_run", False):
            print(f"[DRY-RUN] compute {self.task_name}")
            return
        self.rebuild_from_json(ctx, ade_root)


class Ade20KSemanticDetectron2Compute(BaseComputeTask):
    """Convert ADE20K semantic labels to detectron2 format."""

    task_name = "ade20k_semantic_detectron2"

    def _needs_convert(self, ade_root: Path) -> bool:
        """Check whether detectron2 semantic conversion is required.

        Args:
            ade_root: ADE20K root.
        Returns:
            bool: True if conversion is needed.
        """

        split_specs = [_SplitSpec("training", 20210), _SplitSpec("validation", 2000)]
        for spec in split_specs:
            src_dir = ade_root / "annotations" / spec.split_name
            dst_dir = ade_root / "annotations_detectron2" / spec.split_name
            if not src_dir.is_dir() or not dst_dir.is_dir():
                return True
            src_count = len(list(src_dir.glob("*.png")))
            dst_count = len(list(dst_dir.glob("*.png")))
            if src_count == 0 or dst_count < min(src_count, spec.expected_count):
                return True
        return False

    def run(self, ctx: Any) -> None:
        """Run detectron2 semantic conversion.

        Args:
            ctx: Runtime context.
        Returns:
            None.
        """

        ade_root = Path(ctx.data_dir) / "ovseg_data" / "ade20k"
        if not ade_root.exists():
            return
        if not getattr(ctx, "overwrite", False) and not self._needs_convert(ade_root):
            return
        if getattr(ctx, "dry_run", False):
            print(f"[DRY-RUN] compute {self.task_name}")
            return

        source_root = ade_root / "annotations"
        out_root = ade_root / "annotations_detectron2"
        for split in ["training", "validation"]:
            split_src = source_root / split
            split_dst = out_root / split
            split_dst.mkdir(parents=True, exist_ok=True)
            png_files = sorted(split_src.glob("*.png"))
            for src in tqdm(png_files, desc=f"ADE semantic {split}", unit="img", dynamic_ncols=True):
                dst = split_dst / src.name
                if dst.exists() and not getattr(ctx, "overwrite", False):
                    continue
                arr = np.asarray(Image.open(src), dtype=np.uint8)
                arr = (arr.astype(np.int16) - 1).astype(np.uint8)
                Image.fromarray(arr).save(dst)


def build_compute_tasks(dataset_name: str) -> list[BaseComputeTask]:
    """Build post-download compute tasks by dataset name.

    Args:
        dataset_name: Dataset key.
    Returns:
        list[BaseComputeTask]: Task list.
    """

    key = dataset_name.lower()
    if key == "coco":
        return [CocoSemanticMapCompute()]
    if key == "ovseg":
        return [Ade20KInstanceMapCompute(), Ade20KSemanticDetectron2Compute()]
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Parser object.
    """

    parser = argparse.ArgumentParser(description="Run dataset CPU compute tasks.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key, e.g. coco or ovseg.")
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[2], help="Project root directory.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory relative to root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapping outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    return parser


def main() -> int:
    """Main entrypoint.

    Args:
        None.
    Returns:
        int: Exit code.
    """

    args = build_arg_parser().parse_args()

    class _Ctx:
        """Lightweight runtime context.

        Args:
            None.
        Returns:
            None.
        """

        root_dir = args.root_dir.resolve()
        data_dir = (root_dir / args.data_dir).resolve()
        overwrite = bool(args.overwrite)
        dry_run = bool(args.dry_run)

    tasks = build_compute_tasks(args.dataset)
    if not tasks:
        raise SystemExit(f"No compute task bound to dataset: {args.dataset}")

    for task in tasks:
        print(f"[compute] start {task.task_name}")
        task.run(_Ctx)
        print(f"[compute] done  {task.task_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/dataset/compute.py \
  --dataset coco \
  --root-dir . \
  --data-dir data
"""
