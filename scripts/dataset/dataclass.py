#!/usr/bin/env python3
"""Dataset preparing utility for X-SAM.

This script merges dataset downloading and structure checking into one entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class PathCheckSpec:
    """描述單一路徑檢查規格。

    Args:
        rel_path: 相對於專案根目錄的路徑。
        kind: 路徑型別，`dir` 或 `file`。
        must_symlink: 是否必須是 symbolic link。
        symlink_target_rel: 若 must_symlink=True，預期指向的相對路徑。
    Returns:
        None
    """

    rel_path: str
    kind: str
    must_symlink: bool = False
    symlink_target_rel: Optional[str] = None


@dataclass(frozen=True)
class PrepareContext:
    """保存執行參數與環境設定。

    Args:
        root_dir: 專案根目錄。
        data_dir: 資料根目錄。
        python_bin: Python 可執行檔路徑。
        threads: 外部下載工具執行緒數。
        http_tool: HTTP 下載工具，`auto`、`aria2c` 或 `python`。
        overwrite: 是否覆寫既有檔案。
        keep_archives: 是否保留壓縮檔。
        delete_zip: 是否在最終檢查成功後刪除 zip 檔。
        skip_lmu: 是否跳過 LMUData 下載步驟。
        skip_imgconv_images: 是否跳過 LLaVA 額外影像下載。
        dry_run: 是否只顯示流程不真正執行。
    Returns:
        None
    """

    root_dir: Path
    data_dir: Path
    python_bin: str
    threads: int
    http_tool: str
    overwrite: bool
    keep_archives: bool
    delete_zip: bool
    skip_lmu: bool
    skip_imgconv_images: bool
    dry_run: bool


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

REQUIRED_STRUCTURE: tuple[PathCheckSpec, ...] = (
    PathCheckSpec("data/coco/coco2014", "dir"),
    PathCheckSpec("data/coco/coco2017", "dir"),
    PathCheckSpec("data/coco/coco2017/panoptic_val2017", "dir"),
    PathCheckSpec("data/coco/coco2017/panoptic_semseg_val2017", "dir"),
    PathCheckSpec("data/genseg_data/coco2017", "dir", True, "data/coco/coco2017"),
    PathCheckSpec("data/gcgseg_data/grand_f/annotations/train", "dir"),
    PathCheckSpec("data/gcgseg_data/grand_f/annotations/val_test", "dir"),
    PathCheckSpec("data/gcgseg_data/grand_f/images/coco2014", "dir", True, "data/coco/coco2014"),
    PathCheckSpec("data/gcgseg_data/grand_f/images/coco2017", "dir", True, "data/coco/coco2017"),
    PathCheckSpec("data/gcgseg_data/grand_f/images/flickr30k", "dir"),
    PathCheckSpec("data/gcgseg_data/grand_f/images/GranDf_HA_images", "dir"),
    PathCheckSpec("data/imgconv_data/llava/LLaVA-Instruct-150K", "dir"),
    PathCheckSpec("data/imgconv_data/llava/LLaVA-Pretrain/558k_images", "dir"),
    PathCheckSpec("data/imgconv_data/llava/llava_images/coco", "dir", True, "data/coco/coco2017"),
    PathCheckSpec("data/imgconv_data/llava/llava_images/gqa", "dir"),
    PathCheckSpec("data/imgconv_data/llava/llava_images/ocr_vqa", "dir"),
    PathCheckSpec("data/imgconv_data/llava/llava_images/text_vqa", "dir"),
    PathCheckSpec("data/imgconv_data/llava/llava_images/vg", "dir"),
    PathCheckSpec("data/intseg_data/coco_int/annotations", "dir"),
    PathCheckSpec("data/intseg_data/coco_int/coco2017", "dir", True, "data/coco/coco2017"),
    PathCheckSpec("data/LMUData/images/AI2D_TEST", "dir"),
    PathCheckSpec("data/LMUData/images/MMBench", "dir"),
    PathCheckSpec("data/LMUData/images/MME", "dir"),
    PathCheckSpec("data/LMUData/images/POPE", "dir"),
    PathCheckSpec("data/LMUData/images/SEEDBench_IMG", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/ade20k_panoptic_train", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/ade20k_panoptic_val", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations/training", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations/validation", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations_detectron2", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations_detectron2/training", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations_detectron2/validation", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations_instance/training", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/annotations_instance/validation", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/images", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/images/training", "dir"),
    PathCheckSpec("data/ovseg_data/ade20k/images/validation", "dir"),
    PathCheckSpec("data/reaseg_data/explanatory", "dir"),
    PathCheckSpec("data/reaseg_data/train", "dir"),
    PathCheckSpec("data/reaseg_data/val", "dir"),
    PathCheckSpec("data/reaseg_data/test", "dir"),
    PathCheckSpec("data/refseg_data/annotations", "dir", True, "data/coco/coco2014/annotations"),
    PathCheckSpec("data/refseg_data/images/train2014", "dir", True, "data/coco/coco2014/train2014"),
    PathCheckSpec("data/refseg_data/images/val2014", "dir", True, "data/coco/coco2014/val2014"),
    PathCheckSpec("data/refseg_data/refcoco", "dir"),
    PathCheckSpec("data/refseg_data/refcoco+", "dir"),
    PathCheckSpec("data/refseg_data/refcocog", "dir"),
    PathCheckSpec("data/vgdseg_data/coco_vgd/annotations", "dir"),
    PathCheckSpec("data/vgdseg_data/coco_vgd/coco2017", "dir", True, "data/coco/coco2017"),
    PathCheckSpec("data/vgdseg_data/coco_int", "dir", True, "data/vgdseg_data/coco_vgd"),
)

DOWNLOAD_CONNECT_TIMEOUT_SEC = 30
DOWNLOAD_READ_TIMEOUT_SEC = 1800
DOWNLOAD_RETRY_TIMES = 5
DOWNLOAD_RETRY_BASE_SLEEP_SEC = 3
DOWNLOAD_RETRY_MAX_SLEEP_SEC = 45
ADE20K_EXPECTED_SPLIT_COUNTS: dict[str, int] = {"training": 20210, "validation": 2000}
COCO_VAL2017_EXPECTED_COUNT = 5000
MAX_MISSING_REPORT = 20


def log(msg: str) -> None:
    """輸出帶時間戳的訊息。

    Args:
        msg: 要輸出的訊息。
    Returns:
        None
    """

    print(f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def _resolve_cmd_path(raw_path: str, cwd: Path) -> Path:
    """Resolve command path argument to absolute path.

    Args:
        raw_path: Raw path argument from command list.
        cwd: Command working directory.
    Returns:
        Path: Absolute normalized path.
    """

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (cwd / candidate).resolve()


def _copy_tree_python(src: Path, dst: Path, ignore_existing: bool) -> None:
    """Copy directory tree via Python with optional ignore-existing mode.

    Args:
        src: Source directory.
        dst: Destination directory.
        ignore_existing: Whether to skip existing files/symlinks.
    Returns:
        None
    """

    ensure_dir(dst)
    for item in src.iterdir():
        dst_item = dst / item.name
        if item.is_symlink():
            link_target = os.readlink(item)
            if dst_item.exists() or dst_item.is_symlink():
                if ignore_existing:
                    continue
                remove_path(dst_item)
            dst_item.symlink_to(link_target)
            continue

        if item.is_dir():
            _copy_tree_python(item, dst_item, ignore_existing=ignore_existing)
            continue

        if dst_item.exists():
            if ignore_existing:
                continue
            dst_item.unlink()
        shutil.copy2(item, dst_item)


def _run_rsync_fallback(cmd: list[str], cwd: Path) -> None:
    """Run limited rsync-compatible behavior in pure Python.

    Supported flags:
        - ``-a`` (ignored; archive-like behavior is default here)
        - ``--ignore-existing``
        - ``--delete``

    Args:
        cmd: Original rsync command argument list.
        cwd: Command working directory.
    Returns:
        None
    """

    if len(cmd) < 4:
        raise RuntimeError(f"Unsupported rsync command: {' '.join(cmd)}")

    ignore_existing = "--ignore-existing" in cmd
    delete_mode = "--delete" in cmd
    src = _resolve_cmd_path(cmd[-2], cwd)
    dst = _resolve_cmd_path(cmd[-1], cwd)
    if not src.exists():
        raise FileNotFoundError(f"rsync source not found: {src}")
    if not src.is_dir():
        raise RuntimeError(f"rsync fallback only supports directory source: {src}")

    if delete_mode:
        if dst.exists() or dst.is_symlink():
            remove_path(dst)
        ensure_dir(dst)
        _copy_tree_python(src, dst, ignore_existing=False)
        return

    _copy_tree_python(src, dst, ignore_existing=ignore_existing)


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool = False, env: Optional[dict[str, str]] = None) -> None:
    """執行外部命令。

    Args:
        cmd: 命令與參數列表。
        cwd: 命令執行目錄。
        dry_run: 是否僅列印命令。
        env: 額外環境變數。
    Returns:
        None
    """

    printable = " ".join(cmd)
    if dry_run:
        log(f"[DRY-RUN] {printable}")
        return
    log(printable)
    if cmd and cmd[0] == "rsync" and shutil.which("rsync") is None:
        _run_rsync_fallback(cmd, cwd)
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True)


def ensure_dir(path: Path) -> None:
    """確保目錄存在。

    Args:
        path: 目標目錄。
    Returns:
        None
    """

    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    """刪除檔案、資料夾或 symlink。

    Args:
        path: 要刪除的路徑。
    Returns:
        None
    """

    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def ensure_symlink(link_path: Path, target_path: Path, overwrite: bool = False, dry_run: bool = False) -> None:
    """建立或更新 symbolic link。

    Args:
        link_path: symlink 路徑。
        target_path: 目標路徑。
        overwrite: 既有路徑衝突時是否覆寫。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    if link_path.is_symlink():
        resolved = link_path.resolve()
        if resolved == target_path.resolve():
            return
        if not overwrite:
            raise RuntimeError(f"Symlink target mismatch: {link_path} -> {resolved}")
        if not dry_run:
            link_path.unlink()
    elif link_path.exists():
        if not overwrite:
            raise RuntimeError(f"Path exists and is not symlink: {link_path}")
        if link_path.is_dir():
            if dry_run:
                log(f"[DRY-RUN] migrate dir {link_path} -> {target_path}")
            else:
                ensure_dir(target_path.parent)
                if not target_path.exists():
                    shutil.move(str(link_path), str(target_path))
                else:
                    run_cmd(["rsync", "-a", "--ignore-existing", f"{link_path}/", f"{target_path}/"], cwd=Path.cwd())
                    remove_path(link_path)
        elif link_path.is_file():
            if dry_run:
                log(f"[DRY-RUN] migrate file {link_path} -> {target_path}")
            else:
                ensure_dir(target_path.parent)
                if not target_path.exists():
                    shutil.move(str(link_path), str(target_path))
                else:
                    shutil.copy2(link_path, target_path)
                    link_path.unlink()
        else:
            if not dry_run:
                remove_path(link_path)

    if dry_run:
        log(f"[DRY-RUN] ln -s {target_path} {link_path}")
        return

    ensure_dir(link_path.parent)
    rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))
    link_path.symlink_to(rel_target)


def directory_non_empty(path: Path) -> bool:
    """判斷目錄是否非空。

    Args:
        path: 目標目錄。
    Returns:
        bool: 非空回傳 True。
    """

    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def file_non_empty(path: Path) -> bool:
    """判斷檔案是否存在且大小大於 0。

    Args:
        path: 目標檔案。
    Returns:
        bool: 非空檔案回傳 True。
    """

    return path.is_file() and path.stat().st_size > 0


def path_has_content(path: Path) -> bool:
    """判斷路徑是否有有效內容。

    Args:
        path: 目標路徑，可為檔案、目錄或 symlink。
    Returns:
        bool: 路徑存在且有內容回傳 True。
    """

    if path.is_symlink():
        try:
            return path_has_content(path.resolve())
        except FileNotFoundError:
            return False
    if path.is_file():
        return file_non_empty(path)
    if path.is_dir():
        return directory_non_empty(path)
    return False


def llava_pretrain_has_real_images(images_root: Path) -> bool:
    """判斷 LLaVA-Pretrain 是否含有實際影像檔。

    Args:
        images_root: `LLaVA-Pretrain/558k_images` 根目錄。
    Returns:
        bool: 找到任一個非空影像檔回傳 True。
    """

    if images_root.is_symlink():
        try:
            return llava_pretrain_has_real_images(images_root.resolve())
        except FileNotFoundError:
            return False

    if not images_root.exists() or not images_root.is_dir():
        return False

    probe_paths = [
        images_root / "00001" / "000015879.jpg",
        images_root / "00000" / "000000000.jpg",
    ]
    for probe_path in probe_paths:
        if file_non_empty(probe_path):
            return True

    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for candidate in images_root.glob("*/*"):
        if candidate.is_file() and candidate.suffix.lower() in valid_suffixes and file_non_empty(candidate):
            return True

    return False


def paths_all_ready(paths: Iterable[Path]) -> bool:
    """判斷多個路徑是否全部可用且有內容。

    Args:
        paths: 路徑列表。
    Returns:
        bool: 全部可用回傳 True。
    """

    required = list(paths)
    if not required:
        return False
    return all(path_has_content(path) for path in required)


def load_json_dict(path: Path) -> dict:
    """讀取 JSON 檔並回傳 dict。

    Args:
        path: JSON 檔案路徑。
    Returns:
        dict: 解析後的 JSON 物件。
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return data


def collect_file_stems(directory: Path, suffix: str) -> set[str]:
    """收集指定副檔名檔案的 stem 集合。

    Args:
        directory: 目標目錄。
        suffix: 副檔名，例如 `.jpg` 或 `.png`。
    Returns:
        set[str]: stem 集合。
    """

    if not directory.exists() or not directory.is_dir():
        return set()
    pattern = f"*{suffix}"
    return {path.stem for path in directory.glob(pattern) if path.is_file()}


def format_missing_stems(stems: Iterable[str], suffix: str, limit: int = MAX_MISSING_REPORT) -> str:
    """將缺檔 stem 清單格式化為易讀字串。

    Args:
        stems: 缺失 stem 集合或列表。
        suffix: 副檔名，會加回輸出顯示。
        limit: 最多顯示幾筆。
    Returns:
        str: 人類可讀摘要。
    """

    items = sorted(stems)
    if not items:
        return ""
    preview = [f"{name}{suffix}" for name in items[:limit]]
    if len(items) > limit:
        preview.append(f"... (+{len(items) - limit} more)")
    return ", ".join(preview)


def get_ade20k_base_issues(ade_root: Path) -> list[str]:
    """檢查 ADE20K 基礎資料是否完整（影像 + semantic 標註）。

    Args:
        ade_root: `data/ovseg_data/ade20k` 路徑。
    Returns:
        list[str]: 問題清單，空列表代表通過。
    """

    issues: list[str] = []
    for split, expected_count in ADE20K_EXPECTED_SPLIT_COUNTS.items():
        image_dir = ade_root / "images" / split
        semantic_dir = ade_root / "annotations" / split

        if not image_dir.is_dir():
            issues.append(f"ADE20K_MISSING_DIR: {image_dir}")
            continue
        if not semantic_dir.is_dir():
            issues.append(f"ADE20K_MISSING_DIR: {semantic_dir}")
            continue

        image_stems = collect_file_stems(image_dir, ".jpg")
        semantic_stems = collect_file_stems(semantic_dir, ".png")

        if len(image_stems) != expected_count:
            issues.append(
                f"ADE20K_COUNT_MISMATCH[{split}]: images={len(image_stems)} expected={expected_count}"
            )
        if len(semantic_stems) != expected_count:
            issues.append(
                f"ADE20K_COUNT_MISMATCH[{split}]: semantic={len(semantic_stems)} expected={expected_count}"
            )

        missing_semantic = image_stems - semantic_stems
        if missing_semantic:
            issues.append(
                f"ADE20K_MISSING_SEMANTIC[{split}]: {format_missing_stems(missing_semantic, '.png')}"
            )

        extra_semantic = semantic_stems - image_stems
        if extra_semantic:
            issues.append(f"ADE20K_ORPHAN_SEMANTIC[{split}]: {format_missing_stems(extra_semantic, '.png')}")

    return issues


def get_ade20k_instance_annotation_issues(ade_root: Path) -> list[str]:
    """檢查 `annotations_instance` 是否與影像對齊且完整。

    Args:
        ade_root: `data/ovseg_data/ade20k` 路徑。
    Returns:
        list[str]: 問題清單，空列表代表通過。
    """

    issues: list[str] = []
    base_issues = get_ade20k_base_issues(ade_root)
    if base_issues:
        return base_issues

    for split, expected_count in ADE20K_EXPECTED_SPLIT_COUNTS.items():
        image_dir = ade_root / "images" / split
        instance_dir = ade_root / "annotations_instance" / split
        if not instance_dir.is_dir():
            issues.append(f"ADE20K_MISSING_DIR: {instance_dir}")
            continue

        image_stems = collect_file_stems(image_dir, ".jpg")
        instance_stems = collect_file_stems(instance_dir, ".png")

        if len(instance_stems) != expected_count:
            issues.append(
                f"ADE20K_COUNT_MISMATCH[{split}]: instance={len(instance_stems)} expected={expected_count}"
            )

        missing_instance = image_stems - instance_stems
        if missing_instance:
            issues.append(f"ADE20K_MISSING_INSTANCE[{split}]: {format_missing_stems(missing_instance, '.png')}")
        extra_instance = instance_stems - image_stems
        if extra_instance:
            issues.append(f"ADE20K_ORPHAN_INSTANCE[{split}]: {format_missing_stems(extra_instance, '.png')}")

    return issues


def rebuild_ade20k_instance_annotations_from_json(ctx: PrepareContext, ade_root: Path) -> None:
    """由 `ade20k_instance_{train,val}.json` 反建 `annotations_instance` PNG。

    Args:
        ctx: 執行上下文。
        ade_root: `data/ovseg_data/ade20k` 路徑。
    Returns:
        None
    """

    try:
        from pycocotools import mask as mask_utils
    except ModuleNotFoundError as exc:
        raise RuntimeError("pycocotools is required to rebuild ADE20K annotations_instance.") from exc

    mapping_file = ctx.root_dir / "xsam" / "xsam" / "tools" / "dataset_tools" / "ade20k_instance_catid_mapping.txt"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Cannot find ADE20K mapping file: {mapping_file}")

    # semantic_id(0-based) -> instance_category_id(1-based)
    semantic_to_instance_catid: dict[int, int] = {}
    with mapping_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            ins_id, sem_id, _ = line.strip().split()
            semantic_to_instance_catid[int(sem_id) - 1] = int(ins_id)

    split_alias = {"training": "train", "validation": "val"}
    for split, alias in split_alias.items():
        json_path = ade_root / f"ade20k_instance_{alias}.json"
        if not file_non_empty(json_path):
            raise RuntimeError(f"Missing ADE20K instance json for rebuilding annotations_instance: {json_path}")
        instance_dir = ade_root / "annotations_instance" / split
        ensure_dir(instance_dir)

        instance_data = load_json_dict(json_path)
        images = instance_data.get("images", [])
        annotations = instance_data.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise RuntimeError(f"Invalid ADE20K instance json schema: {json_path}")

        anns_by_image: dict[str, list[dict]] = {}
        for ann in annotations:
            image_id = str(ann.get("image_id"))
            anns_by_image.setdefault(image_id, []).append(ann)

        for image_item in tqdm(images, desc=f"ADE instance rebuild {split}", unit="img", leave=False):
            image_id = str(image_item.get("id"))
            file_name = image_item.get("file_name", "")
            stem = Path(file_name).stem
            if not stem:
                continue

            dst_path = instance_dir / f"{stem}.png"
            if dst_path.exists() and not ctx.overwrite:
                continue
            if ctx.dry_run:
                continue

            height = int(image_item.get("height", 0))
            width = int(image_item.get("width", 0))
            if height <= 0 or width <= 0:
                raise RuntimeError(f"Invalid image size in {json_path}: id={image_id}")

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
                instance_cat_id = semantic_to_instance_catid.get(semantic_id)
                if instance_cat_id is None:
                    continue
                if next_ins_id > 255:
                    raise RuntimeError(f"Too many instances (>255) in ADE image: {image_id}")

                cat_map[mask] = np.uint8(instance_cat_id)
                ins_map[mask] = np.uint8(next_ins_id)
                next_ins_id += 1

            instance_png = np.stack([cat_map, ins_map], axis=-1)
            Image.fromarray(instance_png).save(dst_path)

def get_ade20k_panoptic_issues(ade_root: Path) -> list[str]:
    """檢查 ADE20K panoptic/instance/semantic-detectron2 輸出是否完整。

    Args:
        ade_root: `data/ovseg_data/ade20k` 路徑。
    Returns:
        list[str]: 問題清單，空列表代表通過。
    """

    issues: list[str] = []
    split_alias = {"training": "train", "validation": "val"}
    base_issues = get_ade20k_base_issues(ade_root)
    if base_issues:
        return base_issues
    instance_png_issues = get_ade20k_instance_annotation_issues(ade_root)
    if instance_png_issues:
        return instance_png_issues

    for split in ADE20K_EXPECTED_SPLIT_COUNTS:
        short = split_alias[split]
        json_path = ade_root / f"ade20k_panoptic_{short}.json"
        panoptic_dir = ade_root / f"ade20k_panoptic_{short}"
        detectron_dir = ade_root / "annotations_detectron2" / split
        image_dir = ade_root / "images" / split

        if not file_non_empty(json_path):
            issues.append(f"ADE20K_MISSING_FILE: {json_path}")
            continue
        if not panoptic_dir.is_dir():
            issues.append(f"ADE20K_MISSING_DIR: {panoptic_dir}")
            continue
        if not detectron_dir.is_dir():
            issues.append(f"ADE20K_MISSING_DIR: {detectron_dir}")
            continue

        try:
            panoptic_data = load_json_dict(json_path)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"ADE20K_BAD_JSON: {json_path} ({type(exc).__name__}: {exc})")
            continue

        json_images = panoptic_data.get("images", [])
        json_anns = panoptic_data.get("annotations", [])
        if not isinstance(json_images, list) or not isinstance(json_anns, list):
            issues.append(f"ADE20K_BAD_JSON_SCHEMA: {json_path}")
            continue

        expected_count = ADE20K_EXPECTED_SPLIT_COUNTS[split]
        if len(json_images) != expected_count:
            issues.append(f"ADE20K_JSON_COUNT[{split}]: images={len(json_images)} expected={expected_count}")
        if len(json_anns) != expected_count:
            issues.append(f"ADE20K_JSON_COUNT[{split}]: annotations={len(json_anns)} expected={expected_count}")

        expected_stems = {Path(item["file_name"]).stem for item in json_images if "file_name" in item}
        panoptic_json_stems = {Path(item["file_name"]).stem for item in json_anns if "file_name" in item}

        image_stems = collect_file_stems(image_dir, ".jpg")
        panoptic_stems = collect_file_stems(panoptic_dir, ".png")
        detectron_stems = collect_file_stems(detectron_dir, ".png")

        missing_images = expected_stems - image_stems
        if missing_images:
            issues.append(f"ADE20K_MISSING_IMAGES[{split}]: {format_missing_stems(missing_images, '.jpg')}")
        missing_panoptic = panoptic_json_stems - panoptic_stems
        if missing_panoptic:
            issues.append(f"ADE20K_MISSING_PANOPTIC[{split}]: {format_missing_stems(missing_panoptic, '.png')}")
        missing_detectron = expected_stems - detectron_stems
        if missing_detectron:
            issues.append(f"ADE20K_MISSING_DETECTRON2[{split}]: {format_missing_stems(missing_detectron, '.png')}")

    for split_alias_name in ["train", "val"]:
        instance_json = ade_root / f"ade20k_instance_{split_alias_name}.json"
        if not file_non_empty(instance_json):
            issues.append(f"ADE20K_MISSING_FILE: {instance_json}")
            continue
        try:
            instance_data = load_json_dict(instance_json)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"ADE20K_BAD_JSON: {instance_json} ({type(exc).__name__}: {exc})")
            continue
        images = instance_data.get("images", [])
        annotations = instance_data.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            issues.append(f"ADE20K_BAD_JSON_SCHEMA: {instance_json}")
            continue
        split = "training" if split_alias_name == "train" else "validation"
        expected_count = ADE20K_EXPECTED_SPLIT_COUNTS[split]
        if len(images) != expected_count:
            issues.append(
                f"ADE20K_INSTANCE_JSON_COUNT[{split_alias_name}]: images={len(images)} expected={expected_count}"
            )
        if len(annotations) == 0:
            issues.append(f"ADE20K_INSTANCE_JSON_EMPTY_ANNOTATIONS: {instance_json}")

    return issues


def get_coco_semantic_issues(coco_root: Path, split: str = "val2017") -> list[str]:
    """檢查 COCO panoptic semantic map 是否與 annotation 對齊。

    Args:
        coco_root: `data/coco/coco2017` 路徑。
        split: COCO split 名稱，預設 `val2017`。
    Returns:
        list[str]: 問題清單，空列表代表通過。
    """

    issues: list[str] = []
    annotation_path = coco_root / "annotations" / f"panoptic_{split}.json"
    panoptic_dir = coco_root / f"panoptic_{split}"
    semseg_dir = coco_root / f"panoptic_semseg_{split}"
    image_dir = coco_root / split

    if not file_non_empty(annotation_path):
        issues.append(f"COCO_MISSING_FILE: {annotation_path}")
        return issues
    if not panoptic_dir.is_dir():
        issues.append(f"COCO_MISSING_DIR: {panoptic_dir}")
        return issues
    if not semseg_dir.is_dir():
        issues.append(f"COCO_MISSING_DIR: {semseg_dir}")
        return issues
    if not image_dir.is_dir():
        issues.append(f"COCO_MISSING_DIR: {image_dir}")
        return issues

    try:
        panoptic_data = load_json_dict(annotation_path)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"COCO_BAD_JSON: {annotation_path} ({type(exc).__name__}: {exc})")
        return issues

    json_images = panoptic_data.get("images", [])
    json_anns = panoptic_data.get("annotations", [])
    if not isinstance(json_images, list) or not isinstance(json_anns, list):
        issues.append(f"COCO_BAD_JSON_SCHEMA: {annotation_path}")
        return issues

    if split == "val2017" and len(json_images) != COCO_VAL2017_EXPECTED_COUNT:
        issues.append(f"COCO_JSON_COUNT[{split}]: images={len(json_images)} expected={COCO_VAL2017_EXPECTED_COUNT}")
    if split == "val2017" and len(json_anns) != COCO_VAL2017_EXPECTED_COUNT:
        issues.append(
            f"COCO_JSON_COUNT[{split}]: annotations={len(json_anns)} expected={COCO_VAL2017_EXPECTED_COUNT}"
        )

    expected_image_stems = {Path(item["file_name"]).stem for item in json_images if "file_name" in item}
    expected_panoptic_stems = {Path(item["file_name"]).stem for item in json_anns if "file_name" in item}

    image_stems = collect_file_stems(image_dir, ".jpg")
    panoptic_stems = collect_file_stems(panoptic_dir, ".png")
    semantic_stems = collect_file_stems(semseg_dir, ".png")

    missing_images = expected_image_stems - image_stems
    if missing_images:
        issues.append(f"COCO_MISSING_IMAGES[{split}]: {format_missing_stems(missing_images, '.jpg')}")
    missing_panoptic = expected_panoptic_stems - panoptic_stems
    if missing_panoptic:
        issues.append(f"COCO_MISSING_PANOPTIC[{split}]: {format_missing_stems(missing_panoptic, '.png')}")
    missing_semantic = expected_image_stems - semantic_stems
    if missing_semantic:
        issues.append(f"COCO_MISSING_SEMANTIC[{split}]: {format_missing_stems(missing_semantic, '.png')}")

    return issues


def panoptic_rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    """將 panoptic RGB map 轉成 segment id map。

    Args:
        rgb: 形狀為 `(H, W, 3)` 的 uint8 RGB 陣列。
    Returns:
        np.ndarray: 形狀 `(H, W)` 的 int32 id map。
    """

    return rgb[:, :, 0].astype(np.int32) + 256 * rgb[:, :, 1].astype(np.int32) + 256 * 256 * rgb[:, :, 2].astype(
        np.int32
    )


def generate_coco_panoptic_semantic_maps(ctx: PrepareContext, split: str = "val2017") -> None:
    """由 COCO panoptic 標註生成 semantic png map。

    Args:
        ctx: 執行上下文。
        split: COCO split 名稱，預設 `val2017`。
    Returns:
        None
    """

    coco_root = ctx.data_dir / "coco" / "coco2017"
    annotation_path = coco_root / "annotations" / f"panoptic_{split}.json"
    panoptic_dir = coco_root / f"panoptic_{split}"
    semseg_dir = coco_root / f"panoptic_semseg_{split}"

    if not file_non_empty(annotation_path):
        raise RuntimeError(f"Missing COCO panoptic json: {annotation_path}")
    if not panoptic_dir.is_dir():
        raise RuntimeError(f"Missing COCO panoptic folder: {panoptic_dir}")

    panoptic_data = load_json_dict(annotation_path)
    annotations = panoptic_data.get("annotations", [])
    if not isinstance(annotations, list):
        raise RuntimeError(f"Invalid panoptic annotation schema: {annotation_path}")

    ensure_dir(semseg_dir)
    generated = 0
    skipped = 0
    for ann in tqdm(annotations, desc=f"COCO semantic {split}", unit="img", leave=False):
        panoptic_file = ann.get("file_name")
        segments_info = ann.get("segments_info", [])
        if not panoptic_file or not isinstance(segments_info, list):
            continue

        panoptic_path = panoptic_dir / panoptic_file
        semseg_path = semseg_dir / panoptic_file
        if semseg_path.exists() and not ctx.overwrite:
            skipped += 1
            continue
        if ctx.dry_run:
            generated += 1
            continue
        if not panoptic_path.is_file():
            raise RuntimeError(f"Missing panoptic map while generating semantic map: {panoptic_path}")

        panoptic_rgb = np.asarray(Image.open(panoptic_path).convert("RGB"), dtype=np.uint8)
        panoptic_id_map = panoptic_rgb_to_id(panoptic_rgb)
        semantic_map = np.full(panoptic_id_map.shape, 255, dtype=np.uint8)

        for segment in segments_info:
            segment_id = segment.get("id")
            category_id = segment.get("category_id")
            if segment_id is None or category_id is None:
                continue
            semantic_map[panoptic_id_map == int(segment_id)] = int(category_id)

        Image.fromarray(semantic_map).save(semseg_path)
        generated += 1

    log(
        f"COCO panoptic semantic generation done ({split}): generated={generated}, skipped={skipped}, dir={semseg_dir}"
    )

    semantic_issues = get_coco_semantic_issues(coco_root, split=split)
    if semantic_issues:
        raise RuntimeError("COCO semantic map integrity check failed: " + " | ".join(semantic_issues[:5]))


def build_download_url_candidates(url: str) -> list[str]:
    """建立下載 URL 候選清單。

    Args:
        url: 原始下載網址。
    Returns:
        list[str]: 依序嘗試的 URL。
    """

    candidates: list[str] = [url]
    coco_prefix_http = "http://images.cocodataset.org/"
    coco_prefix_https = "https://images.cocodataset.org/"

    if url.startswith(coco_prefix_http):
        https_url = "https://" + url[len("http://") :]
        candidates = [url, https_url]
    elif url.startswith(coco_prefix_https):
        http_url = "http://" + url[len("https://") :]
        candidates = [url, http_url]
    return candidates


def stream_download_with_resume(url: str, tmp_path: Path, display_name: str) -> None:
    """下載檔案到暫存路徑，支援續傳。

    Args:
        url: 下載網址。
        tmp_path: 暫存檔路徑。
        display_name: tqdm 顯示名稱。
    Returns:
        None
    """

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers: dict[str, str] = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(
        url,
        stream=True,
        timeout=(DOWNLOAD_CONNECT_TIMEOUT_SEC, DOWNLOAD_READ_TIMEOUT_SEC),
        headers=headers,
    ) as response:
        if response.status_code == 416:
            return
        response.raise_for_status()

        append_mode = resume_from > 0 and response.status_code == 206
        if resume_from > 0 and not append_mode:
            tmp_path.unlink(missing_ok=True)
            resume_from = 0

        content_length = int(response.headers.get("content-length", 0))
        if append_mode and content_length > 0:
            total = resume_from + content_length
            initial = resume_from
            mode = "ab"
        else:
            total = content_length if content_length > 0 else None
            initial = 0
            mode = "wb"

        with (
            open(tmp_path, mode) as f,
            tqdm(
                total=total,
                initial=initial,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {display_name}",
                leave=True,
                dynamic_ncols=True,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def resolve_http_download_tool(http_tool: str) -> str:
    """Resolve HTTP download backend.

    Args:
        http_tool: Requested backend name.
    Returns:
        str: Resolved backend name.
    """

    normalized = http_tool.strip().lower()
    if normalized not in {"auto", "aria2c", "python"}:
        raise ValueError(f"Unsupported http tool: {http_tool}")
    if normalized == "auto":
        return "aria2c" if shutil.which("aria2c") is not None else "python"
    if normalized == "aria2c" and shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is not installed. Install it or use --http-tool python.")
    return normalized


def build_aria2_download_cmd(url: str, output_path: Path, threads: int) -> list[str]:
    """Build aria2c download command.

    Args:
        url: Download URL.
        output_path: Output file path.
        threads: Concurrent split/thread count.
    Returns:
        list[str]: Command argument list.
    """

    worker_count = max(1, int(threads))
    return [
        "aria2c",
        "--check-certificate=false",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "-x",
        str(worker_count),
        "-s",
        str(worker_count),
        "-k",
        "1M",
        url,
        "-d",
        str(output_path.parent),
        "-o",
        output_path.name,
    ]


def download_http_file_with_aria2(url: str, output_path: Path, threads: int, dry_run: bool = False) -> None:
    """Download HTTP file with aria2c.

    Args:
        url: Download URL.
        output_path: Output file path.
        threads: Concurrent split/thread count.
        dry_run: Whether only print commands.
    Returns:
        None
    """

    run_cmd(build_aria2_download_cmd(url, output_path, threads), cwd=output_path.parent, dry_run=dry_run)


def download_http_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    required_paths: Optional[Iterable[Path]] = None,
    threads: int = 8,
    http_tool: str = "auto",
    dry_run: bool = False,
) -> None:
    """下載 HTTP/HTTPS 檔案並顯示 tqdm 進度。

    Args:
        url: 下載網址。
        output_path: 輸出檔案路徑。
        overwrite: 是否覆寫既有檔案。
        required_paths: 若提供且目標皆已就緒，則跳過下載。
        threads: aria2c 下載分片/連線數。
        http_tool: HTTP 下載工具，`auto`、`aria2c` 或 `python`。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    if not overwrite and required_paths is not None and paths_all_ready(required_paths):
        log(f"Skip existing extracted targets for: {output_path}")
        return

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        log(f"Skip existing file: {output_path}")
        return

    ensure_dir(output_path.parent)
    if dry_run:
        log(f"[DRY-RUN] download {url} -> {output_path}")
        return

    resolved_http_tool = resolve_http_download_tool(http_tool)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    aria2_state_path = output_path.with_suffix(output_path.suffix + ".aria2")
    url_candidates = build_download_url_candidates(url)
    last_error: Optional[Exception] = None

    if overwrite:
        tmp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        aria2_state_path.unlink(missing_ok=True)

    for candidate_idx, candidate in enumerate(url_candidates):
        for attempt in range(1, DOWNLOAD_RETRY_TIMES + 1):
            try:
                if resolved_http_tool == "aria2c":
                    download_http_file_with_aria2(candidate, output_path, threads=threads, dry_run=False)
                else:
                    stream_download_with_resume(candidate, tmp_path, output_path.name)
                    tmp_path.replace(output_path)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < DOWNLOAD_RETRY_TIMES:
                    sleep_sec = min(DOWNLOAD_RETRY_BASE_SLEEP_SEC * attempt, DOWNLOAD_RETRY_MAX_SLEEP_SEC)
                    log(
                        f"Download failed ({attempt}/{DOWNLOAD_RETRY_TIMES}): {candidate} | "
                        f"{type(exc).__name__}: {exc}. Retry in {sleep_sec}s."
                    )
                    time.sleep(sleep_sec)
                else:
                    log(f"Download failed ({attempt}/{DOWNLOAD_RETRY_TIMES}): {candidate}")
        if candidate_idx < len(url_candidates) - 1:
            log(f"Switch to fallback URL: {url_candidates[candidate_idx + 1]}")

    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download file: {url}") from last_error


def unzip_archive(zip_path: Path, output_dir: Path, dry_run: bool = False) -> None:
    """解壓縮 zip 檔案並顯示 tqdm 進度。

    Args:
        zip_path: 壓縮檔路徑。
        output_dir: 解壓目錄。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    if dry_run:
        log(f"[DRY-RUN] unzip {zip_path} -> {output_dir}")
        return
    if not zip_path.exists():
        log(f"Skip unzip (archive missing): {zip_path}")
        return

    ensure_dir(output_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            for member in tqdm(
                members,
                desc=f"Extract {zip_path.name}",
                unit="file",
                leave=True,
                dynamic_ncols=True,
            ):
                zf.extract(member, path=output_dir)
    except zipfile.BadZipFile as exc:
        # Keep workspace clean: corrupted archive should be removed and re-downloaded.
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupted zip removed: {zip_path}. Re-run to download again.") from exc


def unzip_archive_if_needed(
    zip_path: Path,
    output_dir: Path,
    required_paths: Optional[Iterable[Path]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """依需求解壓 zip，避免重複解壓或被壞檔阻塞。

    Args:
        zip_path: 壓縮檔路徑。
        output_dir: 解壓目錄。
        required_paths: 若提供且已就緒則跳過解壓。
        overwrite: 是否強制重解壓。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    required_list = list(required_paths or [])
    if required_list and not overwrite and paths_all_ready(required_list):
        log(f"Skip unzip: extracted targets already exist for {zip_path}")
        return
    unzip_archive(zip_path, output_dir, dry_run=dry_run)


def gdown_download(
    url: str,
    output: Path,
    python_bin: str,
    is_folder: bool = False,
    overwrite: bool = False,
    required_paths: Optional[Iterable[Path]] = None,
    dry_run: bool = False,
) -> None:
    """透過 gdown 下載 Google Drive 檔案或資料夾。

    Args:
        url: Google Drive 檔案或資料夾網址。
        output: 輸出位置。
        python_bin: Python 可執行檔。
        is_folder: 是否為資料夾下載。
        overwrite: 是否強制重新下載。
        required_paths: 檢查是否缺檔的必要路徑列表，全部存在時會略過下載。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    required_list = list(required_paths or [])
    if not overwrite:
        if required_list:
            missing = [path for path in required_list if not path_has_content(path)]
            if not missing:
                log(f"Skip existing gdown target: {output}")
                return
            log(f"gdown missing {len(missing)} path(s), continue download: {output}")
        elif is_folder and directory_non_empty(output):
            log(f"Skip existing folder: {output}")
            return
        elif (not is_folder) and file_non_empty(output):
            log(f"Skip existing file: {output}")
            return

    cmd = [python_bin, "-m", "gdown"]
    if is_folder:
        cmd.append("--folder")
    else:
        cmd.append("--fuzzy")
    cmd.extend([url, "-O", str(output)])
    run_cmd(cmd, cwd=Path.cwd(), dry_run=dry_run)


def clean_archives(paths: Iterable[Path], keep_archives: bool, dry_run: bool = False) -> None:
    """依設定刪除壓縮檔。

    Args:
        paths: 壓縮檔路徑列表。
        keep_archives: 是否保留壓縮檔。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    if keep_archives:
        return
    for path in paths:
        if path.exists():
            if dry_run:
                log(f"[DRY-RUN] rm {path}")
            else:
                path.unlink()


def refresh_coco_panoptic_archives(ctx: PrepareContext, coco2017_dir: Path) -> None:
    """強制重抓並解壓 COCO panoptic annotations 檔案。

    Args:
        ctx: 執行上下文。
        coco2017_dir: `data/coco/coco2017` 路徑。
    Returns:
        None
    """

    archive_name = "panoptic_annotations_trainval2017.zip"
    archive = coco2017_dir / archive_name
    url = COCO2017_URLS[archive_name]

    download_http_file(
        url,
        archive,
        overwrite=True,
        required_paths=None,
        threads=ctx.threads,
        http_tool=ctx.http_tool,
        dry_run=ctx.dry_run,
    )
    unzip_archive(archive, coco2017_dir, dry_run=ctx.dry_run)

    nested_archives = [
        coco2017_dir / "annotations" / "panoptic_train2017.zip",
        coco2017_dir / "annotations" / "panoptic_val2017.zip",
    ]
    for nested in nested_archives:
        if nested.exists():
            unzip_archive(nested, coco2017_dir, dry_run=ctx.dry_run)
    clean_archives([archive, *nested_archives], keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)


def known_zip_paths(ctx: PrepareContext) -> list[Path]:
    """列出本腳本流程會產生的 zip 檔路徑。

    Args:
        ctx: 執行上下文。
    Returns:
        list[Path]: zip 路徑清單。
    """

    coco2017_dir = ctx.data_dir / "coco" / "coco2017"
    coco2014_dir = ctx.data_dir / "coco" / "coco2014"
    ovseg_root = ctx.data_dir / "ovseg_data"
    refseg_root = ctx.data_dir / "refseg_data"
    reaseg_root = ctx.data_dir / "reaseg_data" / "lisa"
    gcg_root = ctx.data_dir / "gcgseg_data"
    intseg_root = ctx.data_dir / "intseg_data" / "coco_int"
    llava_root = ctx.data_dir / "imgconv_data" / "llava"

    paths: list[Path] = []
    paths.extend(coco2017_dir / filename for filename in COCO2017_URLS)
    paths.extend(coco2014_dir / filename for filename in COCO2014_URLS)
    paths.extend([coco2017_dir / "annotations" / "panoptic_train2017.zip", coco2017_dir / "annotations" / "panoptic_val2017.zip"])
    paths.append(ovseg_root / "ADEChallengeData2016.zip")
    paths.extend(refseg_root / filename for filename in REFSEG_URLS)
    paths.extend([reaseg_root / "train.zip", reaseg_root / "val.zip", reaseg_root / "test.zip"])
    paths.extend([gcg_root / "GranDf_HA_images.zip", gcg_root / "flickr30k-images.zip"])
    paths.append(intseg_root / "PSALM_data.zip")
    paths.extend(llava_root / filename for filename in LLAVA_IMAGE_URLS)

    dedup: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dedup.append(path)
    return dedup


def delete_zip_archives_after_success(ctx: PrepareContext) -> None:
    """在檢查成功後刪除流程中產生的 zip 檔。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    zip_paths = known_zip_paths(ctx)
    removed = 0
    for path in zip_paths:
        if path.suffix != ".zip":
            continue
        if path.exists():
            if ctx.dry_run:
                log(f"[DRY-RUN] rm {path}")
            else:
                path.unlink()
            removed += 1
    log(f"--delete-zip removed {removed} zip file(s).")


def ensure_common_coco_symlinks(ctx: PrepareContext) -> None:
    """建立所有依賴 COCO 的 symlink。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    coco2014 = ctx.data_dir / "coco" / "coco2014"
    coco2017 = ctx.data_dir / "coco" / "coco2017"

    ensure_symlink(ctx.data_dir / "genseg_data" / "coco2017", coco2017, overwrite=True, dry_run=ctx.dry_run)
    ensure_symlink(
        ctx.data_dir / "refseg_data" / "images" / "train2014",
        coco2014 / "train2014",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "refseg_data" / "images" / "val2014",
        coco2014 / "val2014",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "refseg_data" / "annotations",
        coco2014 / "annotations",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "gcgseg_data" / "grand_f" / "images" / "coco2014",
        coco2014,
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "gcgseg_data" / "grand_f" / "images" / "coco2017",
        coco2017,
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "intseg_data" / "coco_int" / "coco2017",
        coco2017,
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "vgdseg_data" / "coco_vgd" / "coco2017",
        coco2017,
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "vgdseg_data" / "coco_int",
        ctx.data_dir / "vgdseg_data" / "coco_vgd",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ctx.data_dir / "imgconv_data" / "llava" / "llava_images" / "coco",
        coco2017,
        overwrite=True,
        dry_run=ctx.dry_run,
    )


def split_llava_mix665k_if_needed(ctx: PrepareContext) -> None:
    """將 LLaVA mix665k 依來源切分為訓練用 json。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    out_dir = ctx.data_dir / "imgconv_data" / "llava" / "LLaVA-Instruct-150K"
    mix_path = out_dir / "llava_v1_5_mix665k.json"
    needed = [
        out_dir / "llava_imgconv_coco.json",
        out_dir / "llava_imgconv_vg.json",
        out_dir / "llava_imgconv_gqa.json",
        out_dir / "llava_imgconv_ocr_vqa.json",
        out_dir / "llava_imgconv_textvqa.json",
    ]
    if not mix_path.exists() or all(path.exists() for path in needed):
        return

    script_path = ctx.root_dir / "scripts" / "split_llava_mix665k.py"
    cmd = [
        ctx.python_bin,
        str(script_path),
        "--input",
        str(mix_path),
        "--output-dir",
        str(out_dir),
    ]
    run_cmd(cmd, cwd=ctx.root_dir, dry_run=ctx.dry_run)


def normalize_hfd_repo_layout(base_dir: Path, owner: str, repo: str, overwrite: bool = False, dry_run: bool = False) -> None:
    """把 hfd 下載後的 owner/repo 目錄搬到預期 repo 目錄。

    Args:
        base_dir: 下載根目錄。
        owner: HuggingFace owner 名稱。
        repo: HuggingFace repo 名稱。
        overwrite: 目的目錄衝突時是否覆寫。
        dry_run: 是否僅列印操作。
    Returns:
        None
    """

    nested = base_dir / owner / repo
    target = base_dir / repo
    if not nested.exists():
        return

    if not target.exists():
        if dry_run:
            log(f"[DRY-RUN] mv {nested} {target}")
            return
        ensure_dir(target.parent)
        nested.rename(target)
        owner_dir = base_dir / owner
        if owner_dir.exists() and not any(owner_dir.iterdir()):
            owner_dir.rmdir()
        return

    if not overwrite:
        return

    if dry_run:
        log(f"[DRY-RUN] rsync -a --delete {nested}/ {target}/")
        return

    run_cmd(["rsync", "-a", "--delete", f"{nested}/", f"{target}/"], cwd=Path.cwd(), dry_run=False)
    shutil.rmtree(nested)
    owner_dir = base_dir / owner
    if owner_dir.exists() and not any(owner_dir.iterdir()):
        owner_dir.rmdir()


def download_generic_segmentation(ctx: PrepareContext) -> None:
    """下載 Generic Segmentation (COCO2017/2014) 並建立共用 symlink。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    coco2017_dir = ctx.data_dir / "coco" / "coco2017"
    coco2014_dir = ctx.data_dir / "coco" / "coco2014"
    ensure_dir(coco2017_dir)
    ensure_dir(coco2014_dir)

    coco2017_archives: list[Path] = []
    coco2017_required = {
        "train2017.zip": [coco2017_dir / "train2017"],
        "val2017.zip": [coco2017_dir / "val2017"],
        "annotations_trainval2017.zip": [
            coco2017_dir / "annotations" / "instances_train2017.json",
            coco2017_dir / "annotations" / "instances_val2017.json",
            coco2017_dir / "annotations" / "captions_train2017.json",
            coco2017_dir / "annotations" / "captions_val2017.json",
            coco2017_dir / "annotations" / "person_keypoints_train2017.json",
            coco2017_dir / "annotations" / "person_keypoints_val2017.json",
        ],
        "panoptic_annotations_trainval2017.zip": [
            coco2017_dir / "annotations" / "panoptic_train2017.json",
            coco2017_dir / "annotations" / "panoptic_val2017.json",
            coco2017_dir / "panoptic_train2017",
            coco2017_dir / "panoptic_val2017",
        ],
    }
    for filename, url in COCO2017_URLS.items():
        archive = coco2017_dir / filename
        download_http_file(
            url,
            archive,
            overwrite=ctx.overwrite,
            required_paths=coco2017_required.get(filename, []),
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        coco2017_archives.append(archive)

    for archive in coco2017_archives:
        unzip_archive(archive, coco2017_dir, dry_run=ctx.dry_run)

    nested_panoptic_archives = [
        coco2017_dir / "annotations" / "panoptic_train2017.zip",
        coco2017_dir / "annotations" / "panoptic_val2017.zip",
    ]
    for archive in nested_panoptic_archives:
        marker = coco2017_dir / archive.stem
        if paths_all_ready([marker]) and not ctx.overwrite:
            log(f"Skip existing extracted targets for: {archive}")
            continue
        if archive.exists():
            unzip_archive(archive, coco2017_dir, dry_run=ctx.dry_run)

    clean_archives(coco2017_archives + nested_panoptic_archives, keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)

    coco2014_archives: list[Path] = []
    coco2014_required = {
        "train2014.zip": [coco2014_dir / "train2014"],
        "val2014.zip": [coco2014_dir / "val2014"],
        "annotations_trainval2014.zip": [
            coco2014_dir / "annotations" / "instances_train2014.json",
            coco2014_dir / "annotations" / "instances_val2014.json",
            coco2014_dir / "annotations" / "captions_train2014.json",
            coco2014_dir / "annotations" / "captions_val2014.json",
            coco2014_dir / "annotations" / "person_keypoints_train2014.json",
            coco2014_dir / "annotations" / "person_keypoints_val2014.json",
        ],
    }
    for filename, url in COCO2014_URLS.items():
        archive = coco2014_dir / filename
        download_http_file(
            url,
            archive,
            overwrite=ctx.overwrite,
            required_paths=coco2014_required.get(filename, []),
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        coco2014_archives.append(archive)

    for archive in coco2014_archives:
        unzip_archive(archive, coco2014_dir, dry_run=ctx.dry_run)

    clean_archives(coco2014_archives, keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)
    ensure_common_coco_symlinks(ctx)

    coco_semantic_issues = get_coco_semantic_issues(coco2017_dir, split="val2017")
    needs_panoptic_refresh = any(issue.startswith("COCO_MISSING_PANOPTIC[val2017]") for issue in coco_semantic_issues)
    if needs_panoptic_refresh:
        log("COCO panoptic_val2017 missing/incomplete. Force refresh panoptic annotations archive once.")
        refresh_coco_panoptic_archives(ctx, coco2017_dir)
        coco_semantic_issues = get_coco_semantic_issues(coco2017_dir, split="val2017")

    if coco_semantic_issues and not ctx.overwrite:
        log("COCO semantic map missing/incomplete, regenerate panoptic_semseg_val2017.")
        for issue in coco_semantic_issues[:5]:
            log(f"  - {issue}")
    if ctx.overwrite or coco_semantic_issues:
        try:
            generate_coco_panoptic_semantic_maps(ctx, split="val2017")
        except RuntimeError as exc:
            if "Missing panoptic map while generating semantic map:" not in str(exc):
                raise
            log("COCO panoptic map missing during semantic generation. Force refresh panoptic annotations and retry once.")
            refresh_coco_panoptic_archives(ctx, coco2017_dir)
            generate_coco_panoptic_semantic_maps(ctx, split="val2017")
    else:
        log("Skip COCO panoptic semantic generation: outputs already complete.")


def convert_ade20k_semantic(ctx: PrepareContext) -> None:
    """轉換 ADE20K semantic 標註到 detectron2 格式。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    source_root = ctx.data_dir / "ovseg_data" / "ade20k" / "annotations"
    out_root = ctx.data_dir / "ovseg_data" / "ade20k" / "annotations_detectron2"

    for split in ["training", "validation"]:
        split_src = source_root / split
        split_dst = out_root / split
        ensure_dir(split_dst)
        png_files = sorted(split_src.glob("*.png"))
        for src in tqdm(png_files, desc=f"ADE semantic {split}", unit="img", leave=False):
            dst = split_dst / src.name
            if dst.exists() and not ctx.overwrite:
                continue
            if ctx.dry_run:
                continue
            arr = np.asarray(Image.open(src), dtype=np.uint8)
            arr = (arr.astype(np.int16) - 1).astype(np.uint8)
            Image.fromarray(arr).save(dst)


def normalize_ade20k_layout(ctx: PrepareContext, ovseg_root: Path) -> Path:
    """將 `ADEChallengeData2016` 佈局整理為 `ade20k`。

    Args:
        ctx: 執行上下文。
        ovseg_root: `data/ovseg_data` 路徑。
    Returns:
        Path: 正規化後 `ade20k` 目錄路徑。
    """

    raw_dir = ovseg_root / "ADEChallengeData2016"
    target_dir = ovseg_root / "ade20k"
    if not raw_dir.exists():
        return target_dir

    if not target_dir.exists():
        if ctx.dry_run:
            log(f"[DRY-RUN] mv {raw_dir} {target_dir}")
        else:
            raw_dir.rename(target_dir)
        return target_dir

    run_cmd(
        ["rsync", "-a", "--ignore-existing", f"{raw_dir}/", f"{target_dir}/"],
        cwd=ctx.root_dir,
        dry_run=ctx.dry_run,
    )
    if not ctx.dry_run:
        remove_path(raw_dir)
    return target_dir


def download_ovseg(ctx: PrepareContext) -> None:
    """下載 Open-Vocabulary Segmentation (ADE20K) 並完成轉檔。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    ovseg_root = ctx.data_dir / "ovseg_data"
    ensure_dir(ovseg_root)

    archive = ovseg_root / "ADEChallengeData2016.zip"
    ade_root = ovseg_root / "ade20k"
    base_issues = get_ade20k_base_issues(ade_root)

    if ctx.overwrite or base_issues:
        if base_issues and not ctx.overwrite:
            log("ADE20K base dataset incomplete, start auto repair (download/extract).")
            for issue in base_issues[:5]:
                log(f"  - {issue}")
        download_http_file(
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            archive,
            overwrite=ctx.overwrite,
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        unzip_archive(archive, ovseg_root, dry_run=ctx.dry_run)
        ade_root = normalize_ade20k_layout(ctx, ovseg_root)
        base_issues = get_ade20k_base_issues(ade_root)

        if base_issues and not ctx.dry_run:
            log("ADE20K base still incomplete after extraction. Force refresh archive and retry once.")
            archive.unlink(missing_ok=True)
            download_http_file(
                "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
                archive,
                overwrite=True,
                threads=ctx.threads,
                http_tool=ctx.http_tool,
                dry_run=False,
            )
            unzip_archive(archive, ovseg_root, dry_run=False)
            ctx_retry = PrepareContext(
                root_dir=ctx.root_dir,
                data_dir=ctx.data_dir,
                python_bin=ctx.python_bin,
                threads=ctx.threads,
                http_tool=ctx.http_tool,
                overwrite=ctx.overwrite,
                keep_archives=ctx.keep_archives,
                delete_zip=ctx.delete_zip,
                skip_lmu=ctx.skip_lmu,
                skip_imgconv_images=ctx.skip_imgconv_images,
                dry_run=False,
            )
            ade_root = normalize_ade20k_layout(ctx_retry, ovseg_root)
            base_issues = get_ade20k_base_issues(ade_root)

        if base_issues:
            raise RuntimeError("ADE20K base dataset incomplete: " + " | ".join(base_issues[:5]))
    else:
        log("Skip ADE20K base download: images/annotations already complete.")

    instance_png_issues = get_ade20k_instance_annotation_issues(ade_root)
    if instance_png_issues:
        instance_json_ready = paths_all_ready(
            [
                ade_root / "ade20k_instance_train.json",
                ade_root / "ade20k_instance_val.json",
            ]
        )
        if instance_json_ready:
            log("ADE20K annotations_instance missing/incomplete. Rebuild from existing instance JSON.")
            rebuild_ade20k_instance_annotations_from_json(ctx, ade_root)
            instance_png_issues = get_ade20k_instance_annotation_issues(ade_root)
        if instance_png_issues:
            raise RuntimeError(
                "ADE20K annotations_instance incomplete and cannot be rebuilt: " + " | ".join(instance_png_issues[:5])
            )

    tools_dir = ctx.root_dir / "xsam" / "xsam" / "tools" / "dataset_tools"
    panoptic_issues = get_ade20k_panoptic_issues(ade_root)
    if ctx.overwrite or panoptic_issues:
        if panoptic_issues and not ctx.overwrite:
            log("ADE20K derived annotations incomplete, regenerate panoptic/instance/semantic outputs.")
            for issue in panoptic_issues[:5]:
                log(f"  - {issue}")

        run_cmd(
            [ctx.python_bin, str(tools_dir / "prepare_ade20k_panoptic.py")],
            cwd=ctx.root_dir,
            dry_run=ctx.dry_run,
            env={"root_dir": str(ctx.root_dir)},
        )
        run_cmd(
            [ctx.python_bin, str(tools_dir / "prepare_ade20k_instance.py")],
            cwd=ctx.root_dir,
            dry_run=ctx.dry_run,
            env={"root_dir": str(ctx.root_dir)},
        )
        convert_ade20k_semantic(ctx)

        panoptic_issues = get_ade20k_panoptic_issues(ade_root)
        if panoptic_issues and not ctx.dry_run:
            log("ADE20K derived annotations still incomplete. Retry generation once.")
            run_cmd(
                [ctx.python_bin, str(tools_dir / "prepare_ade20k_panoptic.py")],
                cwd=ctx.root_dir,
                dry_run=False,
                env={"root_dir": str(ctx.root_dir)},
            )
            run_cmd(
                [ctx.python_bin, str(tools_dir / "prepare_ade20k_instance.py")],
                cwd=ctx.root_dir,
                dry_run=False,
                env={"root_dir": str(ctx.root_dir)},
            )
            convert_ade20k_semantic(
                PrepareContext(
                    root_dir=ctx.root_dir,
                    data_dir=ctx.data_dir,
                    python_bin=ctx.python_bin,
                    threads=ctx.threads,
                    http_tool=ctx.http_tool,
                    overwrite=True,
                    keep_archives=ctx.keep_archives,
                    delete_zip=ctx.delete_zip,
                    skip_lmu=ctx.skip_lmu,
                    skip_imgconv_images=ctx.skip_imgconv_images,
                    dry_run=False,
                )
            )
            panoptic_issues = get_ade20k_panoptic_issues(ade_root)

        if panoptic_issues:
            raise RuntimeError("ADE20K derived annotation incomplete: " + " | ".join(panoptic_issues[:5]))
    else:
        log("Skip ADE20K panoptic/instance/semantic generation: outputs already complete.")

    clean_archives([archive], keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)


def download_refseg(ctx: PrepareContext) -> None:
    """下載 Referring Segmentation 資料集。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    refseg_root = ctx.data_dir / "refseg_data"
    ensure_dir(refseg_root)
    ensure_dir(refseg_root / "images")

    archives: list[Path] = []
    refseg_required = {
        "refcoco.zip": [refseg_root / "refcoco"],
        "refcoco+.zip": [refseg_root / "refcoco+"],
        "refcocog.zip": [refseg_root / "refcocog"],
    }
    for filename, url in REFSEG_URLS.items():
        archive = refseg_root / filename
        download_http_file(
            url,
            archive,
            overwrite=ctx.overwrite,
            required_paths=refseg_required.get(filename, []),
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        archives.append(archive)

    for archive in archives:
        unzip_archive(archive, refseg_root, dry_run=ctx.dry_run)

    clean_archives(archives, keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)
    ensure_common_coco_symlinks(ctx)


def download_reaseg(ctx: PrepareContext) -> None:
    """下載 Reasoning Segmentation (LISA) 並整理目錄。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    lisa_root = ctx.data_dir / "reaseg_data" / "lisa"
    ensure_dir(lisa_root)
    ensure_dir(lisa_root / "explanatory")

    folder_url = "https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy"
    reaseg_ready = all(path_has_content(lisa_root / name) for name in ["train", "val", "test"])
    reaseg_archives_ready = all(file_non_empty(lisa_root / name) for name in ["train.zip", "val.zip", "test.zip"])
    if (reaseg_ready or reaseg_archives_ready) and not ctx.overwrite:
        log(f"Skip existing folder: {lisa_root}")
    else:
        gdown_download(
            folder_url,
            lisa_root,
            python_bin=ctx.python_bin,
            is_folder=True,
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )

    archives = [lisa_root / "train.zip", lisa_root / "val.zip", lisa_root / "test.zip"]
    for archive in archives:
        if archive.exists():
            unzip_archive(archive, lisa_root, dry_run=ctx.dry_run)

    root_train_json = lisa_root / "train.json"
    if root_train_json.exists():
        if ctx.dry_run:
            log(f"[DRY-RUN] mv {root_train_json} {lisa_root / 'explanatory' / 'train.json'}")
        else:
            ensure_dir(lisa_root / "explanatory")
            root_train_json.replace(lisa_root / "explanatory" / "train.json")

    clean_archives(archives, keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)

    # Make top-level aliases to satisfy documented structure.
    reaseg_root = ctx.data_dir / "reaseg_data"
    ensure_symlink(reaseg_root / "train", lisa_root / "train", overwrite=True, dry_run=ctx.dry_run)
    ensure_symlink(reaseg_root / "val", lisa_root / "val", overwrite=True, dry_run=ctx.dry_run)
    ensure_symlink(reaseg_root / "test", lisa_root / "test", overwrite=True, dry_run=ctx.dry_run)
    ensure_symlink(
        reaseg_root / "explanatory", lisa_root / "explanatory", overwrite=True, dry_run=ctx.dry_run
    )


def download_gcgseg(ctx: PrepareContext) -> None:
    """下載 GCG Segmentation 資料集與附屬影像。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    gcg_root = ctx.data_dir / "gcgseg_data"
    grand_f_root = gcg_root / "grand_f"
    ensure_dir(grand_f_root / "annotations")
    ensure_dir(grand_f_root / "images")

    hfd_script = ctx.root_dir / "docs" / "hfd.sh"
    if not hfd_script.exists():
        raise FileNotFoundError(f"Cannot find hfd script: {hfd_script}")

    anno_candidates = [gcg_root / "GranD-f", gcg_root / "MBZUAI" / "GranD-f"]
    grandf_ready = any(path_has_content(path) for path in anno_candidates)
    if grandf_ready and not ctx.overwrite:
        log("Skip GranD-f download: existing dataset detected.")
    else:
        if ctx.overwrite:
            for stale in anno_candidates:
                if stale.exists() or stale.is_symlink():
                    if ctx.dry_run:
                        log(f"[DRY-RUN] rm -rf {stale}")
                    else:
                        remove_path(stale)
        run_cmd(
            [
                "bash",
                str(hfd_script),
                "MBZUAI/GranD-f",
                "--tool",
                "aria2c",
                "-x",
                str(ctx.threads),
                "--save_dir",
                str(gcg_root),
                "--dataset",
            ],
            cwd=ctx.root_dir,
            dry_run=ctx.dry_run,
        )

    anno_src = next((p for p in anno_candidates if p.exists()), None)
    if anno_src is not None and not ctx.dry_run:
        run_cmd(["rsync", "-a", "--ignore-existing", f"{anno_src}/", f"{(grand_f_root / 'annotations')}/"], cwd=ctx.root_dir)

    gran_df_zip = gcg_root / "GranDf_HA_images.zip"
    gdown_download(
        "https://drive.google.com/file/d/1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_/view",
        gran_df_zip,
        python_bin=ctx.python_bin,
        is_folder=False,
        overwrite=ctx.overwrite,
        required_paths=[grand_f_root / "images" / "GranDf_HA_images"],
        dry_run=ctx.dry_run,
    )
    if gran_df_zip.exists():
        unzip_archive(gran_df_zip, grand_f_root / "images", dry_run=ctx.dry_run)

    flickr_zip = gcg_root / "flickr30k-images.zip"
    download_http_file(
        "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip",
        flickr_zip,
        overwrite=ctx.overwrite,
        required_paths=[grand_f_root / "images" / "flickr30k"],
        threads=ctx.threads,
        http_tool=ctx.http_tool,
        dry_run=ctx.dry_run,
    )
    if flickr_zip.exists():
        unzip_archive(flickr_zip, grand_f_root / "images", dry_run=ctx.dry_run)

    old_flickr_dir = grand_f_root / "images" / "flickr30k-images"
    flickr_dir = grand_f_root / "images" / "flickr30k"
    if old_flickr_dir.exists() and not flickr_dir.exists() and not ctx.dry_run:
        old_flickr_dir.rename(flickr_dir)

    # Ensure compatibility with config path: grand_f/images/flickr30k/images/train
    if flickr_dir.exists() and not ctx.dry_run:
        images_dir = flickr_dir / "images"
        ensure_dir(images_dir)
        train_link = images_dir / "train"
        if not train_link.exists() and not train_link.is_symlink():
            rel_target = os.path.relpath(str(flickr_dir), start=str(images_dir))
            train_link.symlink_to(rel_target)

    clean_archives([gran_df_zip, flickr_zip], keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)
    ensure_common_coco_symlinks(ctx)


def download_intseg(ctx: PrepareContext) -> None:
    """下載 Interactive Segmentation (PSALM) 資料。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    intseg_root = ctx.data_dir / "intseg_data" / "coco_int"
    ann_dir = intseg_root / "annotations"
    ensure_dir(ann_dir)

    archive = intseg_root / "PSALM_data.zip"
    gdown_download(
        "https://drive.google.com/file/d/1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3/view?usp=sharing",
        archive,
        python_bin=ctx.python_bin,
        is_folder=False,
        overwrite=ctx.overwrite,
        required_paths=[
            ann_dir / "coco_interactive_train_psalm.json",
            ann_dir / "coco_interactive_val_psalm.json",
        ],
        dry_run=ctx.dry_run,
    )
    if archive.exists():
        unzip_archive(archive, intseg_root, dry_run=ctx.dry_run)

    if not ctx.dry_run:
        for name in ["coco_interactive_train_psalm.json", "coco_interactive_val_psalm.json"]:
            dst = ann_dir / name
            found = list(intseg_root.rglob(name))
            if not found:
                continue

            src = None
            for candidate in found:
                if dst.exists():
                    try:
                        if os.path.samefile(candidate, dst):
                            continue
                    except FileNotFoundError:
                        pass
                src = candidate
                break

            if src is None:
                # Only found destination itself; nothing to copy.
                continue

            if dst.exists():
                try:
                    if os.path.samefile(src, dst):
                        continue
                except FileNotFoundError:
                    pass
            shutil.copy2(src, dst)

    ensure_symlink(
        ann_dir / "intseg_val.json",
        ann_dir / "coco_interactive_val_psalm.json",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    clean_archives([archive], keep_archives=ctx.keep_archives, dry_run=ctx.dry_run)
    ensure_common_coco_symlinks(ctx)


def download_vgdseg(ctx: PrepareContext) -> None:
    """下載 VGD Segmentation 標註並整理目錄。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    vgd_root = ctx.data_dir / "vgdseg_data" / "coco_vgd"
    ann_dir = vgd_root / "annotations"
    ensure_dir(ann_dir)

    for filename, url in VGD_ANN_URLS.items():
        download_http_file(
            url,
            ann_dir / filename,
            overwrite=ctx.overwrite,
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )

    ensure_symlink(
        ann_dir / "vgdseg_train.json",
        ann_dir / "coco_vgdseg_train.json",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_symlink(
        ann_dir / "vgdseg_val.json",
        ann_dir / "coco_vgdseg_val.json",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_common_coco_symlinks(ctx)


def download_llava_training_dataset(ctx: PrepareContext) -> None:
    """下載 LLaVA 訓練資料與對應影像資料。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    llava_root = ctx.data_dir / "imgconv_data" / "llava"
    ensure_dir(llava_root)

    hfd_script = ctx.root_dir / "docs" / "hfd.sh"
    if not hfd_script.exists():
        raise FileNotFoundError(f"Cannot find hfd script: {hfd_script}")

    instruct_target = llava_root / "LLaVA-Instruct-150K" / "llava_v1_5_mix665k.json"
    instruct_nested = llava_root / "liuhaotian" / "LLaVA-Instruct-150K" / "llava_v1_5_mix665k.json"
    instruct_ready = path_has_content(instruct_target) or path_has_content(instruct_nested)
    if instruct_ready and not ctx.overwrite:
        log("Skip LLaVA-Instruct-150K download: existing dataset detected.")
    else:
        if ctx.overwrite:
            for stale in [
                llava_root / "LLaVA-Instruct-150K",
                llava_root / "liuhaotian" / "LLaVA-Instruct-150K",
            ]:
                if stale.exists() or stale.is_symlink():
                    if ctx.dry_run:
                        log(f"[DRY-RUN] rm -rf {stale}")
                    else:
                        remove_path(stale)
        run_cmd(
            [
                "bash",
                str(hfd_script),
                "liuhaotian/LLaVA-Instruct-150K",
                "--tool",
                "aria2c",
                "-x",
                str(ctx.threads),
                "--save_dir",
                str(llava_root),
                "--dataset",
            ],
            cwd=ctx.root_dir,
            dry_run=ctx.dry_run,
        )

    pretrain_target = llava_root / "LLaVA-Pretrain" / "558k_images"
    pretrain_nested = llava_root / "liuhaotian" / "LLaVA-Pretrain" / "558k_images"
    pretrain_ready = llava_pretrain_has_real_images(pretrain_target) or llava_pretrain_has_real_images(pretrain_nested)
    if pretrain_ready and not ctx.overwrite:
        log("Skip LLaVA-Pretrain download: found non-empty image files.")
    else:
        if ctx.overwrite:
            for stale in [
                llava_root / "LLaVA-Pretrain",
                llava_root / "liuhaotian" / "LLaVA-Pretrain",
            ]:
                if stale.exists() or stale.is_symlink():
                    if ctx.dry_run:
                        log(f"[DRY-RUN] rm -rf {stale}")
                    else:
                        remove_path(stale)
        run_cmd(
            [
                "bash",
                str(hfd_script),
                "liuhaotian/LLaVA-Pretrain",
                "--tool",
                "aria2c",
                "-x",
                str(ctx.threads),
                "--save_dir",
                str(llava_root),
                "--dataset",
            ],
            cwd=ctx.root_dir,
            dry_run=ctx.dry_run,
        )

    normalize_hfd_repo_layout(llava_root, owner="liuhaotian", repo="LLaVA-Instruct-150K", overwrite=ctx.overwrite, dry_run=ctx.dry_run)
    normalize_hfd_repo_layout(llava_root, owner="liuhaotian", repo="LLaVA-Pretrain", overwrite=ctx.overwrite, dry_run=ctx.dry_run)

    llava_images = llava_root / "llava_images"
    ensure_dir(llava_images)

    if not ctx.skip_imgconv_images:
        vg_dir = llava_images / "vg"
        ensure_dir(vg_dir)
        vg_zip = llava_root / "vg_images.zip"
        vg_zip2 = llava_root / "vg_images2.zip"
        download_http_file(
            LLAVA_IMAGE_URLS["vg_images.zip"],
            vg_zip,
            overwrite=ctx.overwrite,
            required_paths=[vg_dir],
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        download_http_file(
            LLAVA_IMAGE_URLS["vg_images2.zip"],
            vg_zip2,
            overwrite=ctx.overwrite,
            required_paths=[vg_dir],
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        unzip_archive_if_needed(
            vg_zip,
            vg_dir,
            required_paths=[vg_dir],
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )
        unzip_archive_if_needed(
            vg_zip2,
            vg_dir,
            required_paths=[vg_dir],
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )

        textvqa_zip = llava_root / "textvqa_train_val_images.zip"
        download_http_file(
            LLAVA_IMAGE_URLS["textvqa_train_val_images.zip"],
            textvqa_zip,
            overwrite=ctx.overwrite,
            required_paths=[llava_images / "textvqa" / "train_images"],
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        unzip_archive_if_needed(
            textvqa_zip,
            llava_images / "textvqa",
            required_paths=[llava_images / "textvqa" / "train_images"],
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )

        ocr_zip = llava_root / "ocr_vqa_images_llava_v15.zip"
        download_http_file(
            LLAVA_IMAGE_URLS["ocr_vqa_images_llava_v15.zip"],
            ocr_zip,
            overwrite=ctx.overwrite,
            required_paths=[llava_images / "ocr_vqa" / "images"],
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        unzip_archive_if_needed(
            ocr_zip,
            llava_images / "ocr_vqa",
            required_paths=[llava_images / "ocr_vqa" / "images"],
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )

        gqa_zip = llava_root / "gqa_images.zip"
        download_http_file(
            LLAVA_IMAGE_URLS["gqa_images.zip"],
            gqa_zip,
            overwrite=ctx.overwrite,
            required_paths=[llava_images / "gqa" / "images"],
            threads=ctx.threads,
            http_tool=ctx.http_tool,
            dry_run=ctx.dry_run,
        )
        unzip_archive_if_needed(
            gqa_zip,
            llava_images / "gqa",
            required_paths=[llava_images / "gqa" / "images"],
            overwrite=ctx.overwrite,
            dry_run=ctx.dry_run,
        )

        if not ctx.dry_run:
            images_cands = list((llava_images / "ocr_vqa").rglob("images"))
            if images_cands and not (llava_images / "ocr_vqa" / "images").exists():
                ensure_dir(llava_images / "ocr_vqa")
                shutil.copytree(images_cands[0], llava_images / "ocr_vqa" / "images", dirs_exist_ok=True)

        clean_archives(
            [vg_zip, vg_zip2, textvqa_zip, ocr_zip, gqa_zip],
            keep_archives=ctx.keep_archives,
            dry_run=ctx.dry_run,
        )

    ensure_symlink(
        llava_images / "text_vqa",
        llava_images / "textvqa",
        overwrite=True,
        dry_run=ctx.dry_run,
    )
    ensure_common_coco_symlinks(ctx)
    split_llava_mix665k_if_needed(ctx)


def download_lmu_eval_datasets(ctx: PrepareContext) -> None:
    """下載或初始化 VLM Eval 所需的 LMUData。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    lmu_root = ctx.data_dir / "LMUData" / "images"
    for name in ["AI2D_TEST", "MMBench", "MME", "POPE", "SEEDBench_IMG"]:
        ensure_dir(lmu_root / name)

    if ctx.skip_lmu:
        log("Skip LMUData auto-download by --skip-lmu")
        return

    try:
        __import__("vlmeval")
    except ModuleNotFoundError:
        log("vlmeval is not installed. LMUData cannot be auto-downloaded in this environment.")
        log("Install VLMEvalKit via git, then rerun.")
        log("  git clone -b v0.3rc1 https://github.com/open-compass/VLMEvalKit.git")
        log("  cd VLMEvalKit && pip install -e .")
        log("Or use one-line install:")
        log("  pip install \"git+https://github.com/open-compass/VLMEvalKit.git@v0.3rc1\"")
        log("Then run: bash run.sh --modes vlmeval --config <your_config>")
        return

    code = (
        "from vlmeval.dataset import build_dataset\n"
        "datasets=['MME','MMBench_DEV_EN','SEEDBench_IMG','POPE','AI2D_TEST']\n"
        "for n in datasets:\n"
        "    print(f'[LMU] preparing {n}')\n"
        "    build_dataset(n)\n"
    )
    run_cmd(
        [ctx.python_bin, "-c", code],
        cwd=ctx.root_dir,
        dry_run=ctx.dry_run,
        env={"LMUData": str(ctx.data_dir / "LMUData")},
    )


def run_download_pipeline(ctx: PrepareContext) -> None:
    """依文件順序執行資料下載流程。

    Args:
        ctx: 執行上下文。
    Returns:
        None
    """

    steps = [
        ("Generic Segmentation Dataset", download_generic_segmentation),
        ("Open-Vocabulary Segmentation Dataset", download_ovseg),
        ("Referring Segmentation Dataset", download_refseg),
        ("Reasoning Segmentation Dataset", download_reaseg),
        ("GCG Segmentation Dataset", download_gcgseg),
        ("Interactive Segmentation Dataset", download_intseg),
        ("VGD Segmentation Dataset", download_vgdseg),
        ("LLaVA Training Dataset", download_llava_training_dataset),
        ("VLM Evaluation Dataset", download_lmu_eval_datasets),
    ]

    for idx, (name, fn) in enumerate(steps, start=1):
        print(f"\n[Downloading {idx}. {name}]")
        with tqdm(total=1, desc=f"{idx}. {name}", unit="step") as pbar:
            fn(ctx)
            pbar.update(1)


def check_dataset_structure(ctx: PrepareContext) -> bool:
    """檢查 Dataset Structure 的路徑存在性、非空與 symlink 正確性。

    Args:
        ctx: 執行上下文。
    Returns:
        bool: 全部通過回傳 True。
    """

    failures: list[str] = []
    for spec in REQUIRED_STRUCTURE:
        abs_path = ctx.root_dir / spec.rel_path
        if not abs_path.exists() and not abs_path.is_symlink():
            failures.append(f"MISSING: {spec.rel_path}")
            continue

        if spec.must_symlink:
            if not abs_path.is_symlink():
                failures.append(f"NOT_SYMLINK: {spec.rel_path}")
            elif spec.symlink_target_rel:
                expected = (ctx.root_dir / spec.symlink_target_rel).resolve()
                actual = abs_path.resolve()
                if actual != expected:
                    failures.append(
                        f"SYMLINK_TARGET_MISMATCH: {spec.rel_path} -> {actual} (expected {expected})"
                    )

        if spec.kind == "dir":
            if not abs_path.is_dir():
                failures.append(f"NOT_DIR: {spec.rel_path}")
            elif not directory_non_empty(abs_path):
                failures.append(f"EMPTY_DIR: {spec.rel_path}")
        elif spec.kind == "file":
            if not file_non_empty(abs_path):
                failures.append(f"EMPTY_FILE: {spec.rel_path}")
        else:
            failures.append(f"INVALID_SPEC_KIND: {spec.rel_path} ({spec.kind})")

    coco_root = ctx.data_dir / "coco" / "coco2017"
    if coco_root.exists():
        for issue in get_coco_semantic_issues(coco_root, split="val2017"):
            failures.append(f"INTEGRITY: {issue}")

    ade_root = ctx.data_dir / "ovseg_data" / "ade20k"
    if ade_root.exists():
        for issue in get_ade20k_panoptic_issues(ade_root):
            failures.append(f"INTEGRITY: {issue}")

    llava_root = ctx.data_dir / "imgconv_data" / "llava"
    llava_pretrain_candidates = [
        llava_root / "LLaVA-Pretrain" / "558k_images",
        llava_root / "liuhaotian" / "LLaVA-Pretrain" / "558k_images",
    ]
    if any(path.exists() or path.is_symlink() for path in llava_pretrain_candidates):
        if not any(llava_pretrain_has_real_images(path) for path in llava_pretrain_candidates):
            failures.append("INTEGRITY: data/imgconv_data/llava/LLaVA-Pretrain/558k_images has no real image files")

    if failures:
        log("Dataset structure check FAILED:")
        for item in failures:
            print(f"  - {item}")
        return False

    log("Dataset structure check PASSED.")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    """建立 CLI 參數解析器。

    Args:
        None
    Returns:
        argparse.ArgumentParser: 解析器物件。
    """

    parser = argparse.ArgumentParser(description="Download/check datasets for X-SAM.")
    parser.add_argument("--mode", choices=["download", "check"], required=True, help="Run mode")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory relative to root")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable path")
    parser.add_argument("--threads", type=int, default=8, help="Download thread setting for hfd/aria2c")
    parser.add_argument("--http-tool", choices=["auto", "aria2c", "python"], default="auto", help="HTTP download backend")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files/symlinks")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded zip files")
    parser.add_argument(
        "--delete-zip",
        "--delete_zip",
        action="store_true",
        help="Delete downloaded zip files at the end only if dataset check passes",
    )
    parser.add_argument("--skip-lmu", action="store_true", help="Skip LMUData auto download")
    parser.add_argument(
        "--skip-imgconv-images",
        action="store_true",
        help="Skip extra image downloads for gqa/ocr_vqa/text_vqa/vg",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return parser


def main() -> int:
    """主程式入口。

    Args:
        None
    Returns:
        int: 0 代表成功，非 0 代表失敗。
    """

    args = build_arg_parser().parse_args()
    if args.keep_archives and args.delete_zip:
        raise SystemExit("--keep-archives and --delete-zip cannot be used together.")

    root_dir = args.root_dir.resolve()
    data_dir = (root_dir / args.data_dir).resolve()
    ensure_dir(data_dir)

    keep_archives = args.keep_archives or args.delete_zip
    ctx = PrepareContext(
        root_dir=root_dir,
        data_dir=data_dir,
        python_bin=args.python_bin,
        threads=args.threads,
        http_tool=args.http_tool,
        overwrite=args.overwrite,
        keep_archives=keep_archives,
        delete_zip=args.delete_zip,
        skip_lmu=args.skip_lmu,
        skip_imgconv_images=args.skip_imgconv_images,
        dry_run=args.dry_run,
    )

    if args.mode == "download":
        run_download_pipeline(ctx)

    ok = check_dataset_structure(ctx)
    if args.mode == "download" and ctx.delete_zip:
        if ok:
            delete_zip_archives_after_success(ctx)
        else:
            log("Skip --delete-zip because dataset check failed.")
    return 0 if ok else 2



# -----------------------------------------------------------------------------
# Dataset-provider CLI (new entrypoint)
# -----------------------------------------------------------------------------
from typing import Callable, Sequence

try:
    from scripts.dataset import compute
    from scripts.dataset import manifest
except Exception:
    import compute  # type: ignore
    import manifest  # type: ignore


@dataclass(frozen=True)
class DatasetStep:
    """Execution step for a dataset provider.

    Args:
        idx: Step index.
        dataset_name: Dataset key.
        display_name: Human-readable dataset name.
        runner: Callable runner.
    Returns:
        None.
    """

    idx: int
    dataset_name: str
    display_name: str
    runner: Callable[[PrepareContext], None]


def inject_manifest_to_runtime() -> None:
    """Inject manifest constants into this module runtime.

    Args:
        None.
    Returns:
        None.
    """

    global COCO2017_URLS, COCO2014_URLS, REFSEG_URLS, LLAVA_IMAGE_URLS, VGD_ANN_URLS
    COCO2017_URLS = dict(manifest.COCO2017_URLS)
    COCO2014_URLS = dict(manifest.COCO2014_URLS)
    REFSEG_URLS = dict(manifest.REFSEG_URLS)
    LLAVA_IMAGE_URLS = dict(manifest.LLAVA_IMAGE_URLS)
    VGD_ANN_URLS = dict(manifest.VGD_ANN_URLS)


def build_dataset_registry() -> dict[str, tuple[str, Callable[[PrepareContext], None]]]:
    """Build dataset registry.

    Args:
        None.
    Returns:
        dict[str, tuple[str, callable]]: Dataset name -> (display name, runner).
    """

    return {
        "coco": ("Generic Segmentation Dataset", download_generic_segmentation),
        "ovseg": ("Open-Vocabulary Segmentation Dataset", download_ovseg),
        "refseg": ("Referring Segmentation Dataset", download_refseg),
        "reaseg": ("Reasoning Segmentation Dataset", download_reaseg),
        "gcgseg": ("GCG Segmentation Dataset", download_gcgseg),
        "intseg": ("Interactive Segmentation Dataset", download_intseg),
        "vgdseg": ("VGD Segmentation Dataset", download_vgdseg),
        "llava": ("LLaVA Training Dataset", download_llava_training_dataset),
        "lmu": ("VLM Evaluation Dataset", download_lmu_eval_datasets),
    }


def parse_dataset_names(raw_dataset_names: str) -> list[str]:
    """Parse dataset names from comma-separated string.

    Args:
        raw_dataset_names: Raw comma-separated dataset names.
    Returns:
        list[str]: Normalized dataset list.
    """

    names = manifest.normalize_dataset_names(raw_dataset_names.split(","))
    if "all" in names:
        return manifest.list_default_datasets()
    return names


def resolve_dataset_steps(dataset_names: Sequence[str]) -> list[DatasetStep]:
    """Resolve dataset names to executable steps.

    Args:
        dataset_names: Dataset names.
    Returns:
        list[DatasetStep]: Resolved steps.
    """

    registry = build_dataset_registry()
    steps: list[DatasetStep] = []
    for idx, dataset_name in enumerate(dataset_names, start=1):
        entry = registry.get(dataset_name)
        if entry is None:
            valid = ",".join(sorted(registry.keys()))
            raise KeyError(f"Unknown dataset '{dataset_name}'. Valid datasets: all,{valid}")
        display_name, runner = entry
        steps.append(DatasetStep(idx=idx, dataset_name=dataset_name, display_name=display_name, runner=runner))
    return steps


def run_dataset_download(ctx: PrepareContext, dataset_steps: Sequence[DatasetStep], do_check: bool) -> bool:
    """Run dataset download flow.

    Args:
        ctx: Preparation context.
        dataset_steps: Dataset steps.
        do_check: Whether run final structure check.
    Returns:
        bool: True if flow passes.
    """

    for step in dataset_steps:
        print(f"\n[Downloading {step.idx}. {step.display_name}]")
        step.runner(ctx)
        # Execute post-download CPU mapping tasks per dataset provider.
        for task in compute.build_compute_tasks(step.dataset_name):
            log(f"[compute:{step.dataset_name}] start {task.task_name}")
            task.run(ctx)
            log(f"[compute:{step.dataset_name}] done  {task.task_name}")

    if do_check:
        ok = check_dataset_structure(ctx)
    else:
        ok = True
        log("Skip dataset structure check by --skip-check.")

    if ctx.delete_zip:
        if ok:
            delete_zip_archives_after_success(ctx)
        else:
            log("Skip --delete-zip because dataset check failed.")
    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Parser object.
    """

    parser = argparse.ArgumentParser(description="Dataset downloader/checker for X-SAM.")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset list. Use all to download all datasets.")
    parser.add_argument("--providers", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["download", "check"], default="download", help="Run mode.")
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[2], help="Project root directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory relative to root")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable path")
    parser.add_argument("--threads", type=int, default=8, help="Download thread setting for hfd/aria2c")
    parser.add_argument("--http-tool", choices=["auto", "aria2c", "python"], default="auto", help="HTTP download backend")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files/symlinks")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded zip files")
    parser.add_argument("--delete-zip", "--delete_zip", action="store_true", help="Delete downloaded zip files only if final check passes")
    parser.add_argument("--skip-lmu", action="store_true", help="Skip LMUData auto download")
    parser.add_argument("--skip-imgconv-images", action="store_true", help="Skip extra image downloads for gqa/ocr_vqa/text_vqa/vg")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--skip-check", action="store_true", help="Skip final full dataset structure check")
    return parser


def main() -> int:
    """Main entrypoint.

    Args:
        None.
    Returns:
        int: Exit code.
    """

    args = build_arg_parser().parse_args()
    if args.keep_archives and args.delete_zip:
        raise SystemExit("--keep-archives and --delete-zip cannot be used together.")

    inject_manifest_to_runtime()

    root_dir = args.root_dir.resolve()
    data_dir = (root_dir / args.data_dir).resolve()
    ensure_dir(data_dir)

    keep_archives = args.keep_archives or args.delete_zip
    ctx = PrepareContext(
        root_dir=root_dir,
        data_dir=data_dir,
        python_bin=args.python_bin,
        threads=args.threads,
        http_tool=args.http_tool,
        overwrite=args.overwrite,
        keep_archives=keep_archives,
        delete_zip=args.delete_zip,
        skip_lmu=args.skip_lmu,
        skip_imgconv_images=args.skip_imgconv_images,
        dry_run=args.dry_run,
    )

    raw_datasets = args.datasets
    if args.providers:
        raw_datasets = args.providers
    dataset_names = parse_dataset_names(raw_datasets)

    if args.mode == "download":
        steps = resolve_dataset_steps(dataset_names)
        ok = run_dataset_download(ctx, steps, do_check=not args.skip_check)
    else:
        ok = check_dataset_structure(ctx)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/dataset/dataclass.py \
  --datasets all \
  --root-dir . \
  --threads 8 \
  --http-tool auto

python scripts/dataset/dataclass.py \
  --mode check \
  --root-dir .
"""
