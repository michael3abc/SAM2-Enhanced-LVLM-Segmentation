"""I/O helpers for layer-sweep pipelines."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Target directory path.
    Returns:
        Resolved directory path.
    """
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def parse_layer_ids(value: str | List[int]) -> List[int]:
    """Parse layer id input.

    Args:
        value: CSV string or integer list.
    Returns:
        Parsed layer id list.
    """
    if isinstance(value, list):
        return [int(v) for v in value]
    if not isinstance(value, str):
        raise TypeError(f"Unsupported layer id input type: {type(value)}")
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def read_summary_rows(summary_csv_path: str | Path) -> List[Dict[str, str]]:
    """Read summary CSV rows.

    Args:
        summary_csv_path: Summary CSV path.
    Returns:
        CSV row list.
    """
    path = Path(summary_csv_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def select_layers_by_metric(
    rows: List[Dict[str, str]],
    metric_key: str,
    metric_mode: str,
    topk: int,
) -> List[int]:
    """Select top-k layers by metric from summary rows.

    Args:
        rows: Summary row list.
        metric_key: Metric key in rows.
        metric_mode: ``min`` or ``max``.
        topk: Number of layers to select.
    Returns:
        Selected layer id list.
    """
    if metric_mode not in {"min", "max"}:
        raise ValueError(f"`metric_mode` must be `min` or `max`, got {metric_mode}.")
    candidates = []
    for row in rows:
        if row.get("status", "") != "ok":
            continue
        if "layer_id" not in row or metric_key not in row:
            continue
        try:
            layer_id = int(row["layer_id"])
            metric_val = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        candidates.append((layer_id, metric_val))
    if not candidates:
        return []
    reverse = metric_mode == "max"
    candidates.sort(key=lambda item: item[1], reverse=reverse)
    return [layer_id for layer_id, _ in candidates[: max(topk, 1)]]


def append_summary_row(summary_csv_path: str | Path, row: Dict[str, Any]) -> None:
    """Append one row to summary CSV.

    Args:
        summary_csv_path: Summary CSV path.
        row: Row data.
    Returns:
        None.
    """
    path = Path(summary_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    existing_rows: List[Dict[str, Any]] = []
    existing_fieldnames: List[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    merged_fieldnames = list(existing_fieldnames)
    for fieldname in fieldnames:
        if fieldname not in merged_fieldnames:
            merged_fieldnames.append(fieldname)

    if not merged_fieldnames:
        merged_fieldnames = fieldnames

    if not path.exists() or merged_fieldnames != existing_fieldnames:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                normalized_existing_row = {key: existing_row.get(key, "") for key in merged_fieldnames}
                writer.writerow(normalized_existing_row)
            normalized_row = {key: row.get(key, "") for key in merged_fieldnames}
            writer.writerow(normalized_row)
        return

    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        normalized_row = {key: row.get(key, "") for key in merged_fieldnames}
        writer.writerow(normalized_row)
