#!/usr/bin/env python3
"""Build a markdown table from X-SAM segeval logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LOG_LOAD_DATASET_RE = re.compile(r" - INFO - Loading ([^\s]+) dataset")
LOG_EVAL_RESULTS_RE = re.compile(r" - INFO - ([^\s]+) evaluation results:")
LOG_AP_ALL_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
)
LOG_PANOPTIC_ALL_ROW_RE = re.compile(r"^\|\s*All\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
LOG_MIOU_ROW_RE = re.compile(r"^\|\s*mIoU\s*\|\s*([0-9.]+)\s*\|")
LOG_IOU_ROW_RE = re.compile(r"^\|\s*(refer|reason|inter)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
LOG_ERROR_RE = re.compile(r" - INFO - Error evaluating ([^\s]+)")


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns:
        argparse.Namespace: 解析後的命令列參數。
    """
    parser = argparse.ArgumentParser(description="Parse X-SAM segeval logs and output a markdown table.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more segeval log paths. Later logs can fill missing metrics.",
    )
    parser.add_argument(
        "--output",
        default="~/results.md",
        help="Output markdown path. Default: ~/results.md",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Method name shown in table. Default: parent folder name of first log.",
    )
    parser.add_argument(
        "--iou-metric",
        choices=["ciou", "giou"],
        default="giou",
        help="Use cIoU or gIoU for Ref/Rea/Inter columns.",
    )
    return parser.parse_args()


def _empty_metrics() -> Dict[str, Optional[float]]:
    """建立預設空指標字典。

    Returns:
        Dict[str, Optional[float]]: 全部欄位皆為 None 的指標字典。
    """
    keys = [
        "gen_pan",
        "gen_ins",
        "gen_sem",
        "ov_pan",
        "ov_ins",
        "ov_sem",
        "ref_refcoco",
        "ref_refcocop",
        "ref_refcocog",
        "rea_val",
        "rea_test",
        "gcg_val",
        "gcg_test",
        "inter_point",
        "inter_box",
        "vgd_point",
        "vgd_box",
    ]
    return {k: None for k in keys}


def _pick_iou_value(ciou: str, giou: str, iou_metric: str) -> float:
    """依設定選擇 cIoU 或 gIoU。

    Args:
        ciou: cIoU 字串值。
        giou: gIoU 字串值。
        iou_metric: 指定要使用的 IoU 指標（ciou/giou）。

    Returns:
        float: 選擇後的數值。
    """
    return float(ciou) if iou_metric == "ciou" else float(giou)


def _fill_ap_metric(metrics: Dict[str, Optional[float]], dataset: str, ap01: float) -> None:
    """把 AP(0~1) 寫入對應欄位（轉百分比）。

    Args:
        metrics: 目標指標字典。
        dataset: 當前資料集名稱。
        ap01: AP 原始值（0~1）。

    Returns:
        None.
    """
    ap_pct = ap01 * 100.0
    mapping = {
        "coco_instance_genseg": "gen_ins",
        "ade20k_instance_ovseg": "ov_ins",
        "point_vgdseg": "vgd_point",
        "box_vgdseg": "vgd_box",
        "point_intseg": "inter_point",
        "box_intseg": "inter_box",
    }
    target = mapping.get(dataset)
    if target is not None and metrics[target] is None:
        metrics[target] = ap_pct


def _fill_panoptic_metric(metrics: Dict[str, Optional[float]], dataset: str, pq: float) -> None:
    """把 panoptic 的 All/PQ 寫入對應欄位。

    Args:
        metrics: 目標指標字典。
        dataset: 當前資料集名稱。
        pq: Panoptic Quality（All）。

    Returns:
        None.
    """
    mapping = {
        "coco_panoptic_genseg": "gen_pan",
        "ade20k_panoptic_ovseg": "ov_pan",
    }
    target = mapping.get(dataset)
    if target is not None and metrics[target] is None:
        metrics[target] = pq


def _fill_miou_metric(metrics: Dict[str, Optional[float]], dataset: str, miou: float) -> None:
    """把 mIoU 寫入對應欄位。

    Args:
        metrics: 目標指標字典。
        dataset: 當前資料集名稱。
        miou: mIoU 百分比數值。

    Returns:
        None.
    """
    mapping = {
        "coco_panoptic_semantic_genseg": "gen_sem",
        "ade20k_panoptic_semantic_ovseg": "ov_sem",
        "val_gcgseg": "gcg_val",
        "test_gcgseg": "gcg_test",
    }
    target = mapping.get(dataset)
    if target is not None and metrics[target] is None:
        metrics[target] = miou


def _fill_iou_metric(
    metrics: Dict[str, Optional[float]],
    dataset: str,
    row_name: str,
    value: float,
) -> None:
    """把 iou 表格中的 refer/reason/inter 寫入對應欄位。

    Args:
        metrics: 目標指標字典。
        dataset: 當前資料集名稱。
        row_name: iou 表格列名（refer/reason/inter）。
        value: 解析後的 IoU 數值。

    Returns:
        None.
    """
    if row_name == "refer":
        mapping = {
            "refcoco_val_refseg": "ref_refcoco",
            "refcoco+_val_refseg": "ref_refcocop",
            "refcocog_val_refseg": "ref_refcocog",
        }
        target = mapping.get(dataset)
        if target is not None and metrics[target] is None:
            metrics[target] = value
        return

    if row_name == "reason":
        if dataset == "val_reaseg" and metrics["rea_val"] is None:
            metrics["rea_val"] = value
            return
        if dataset == "test_all_reaseg" and metrics["rea_test"] is None:
            metrics["rea_test"] = value
            return
        if dataset in {"test_sentence_reaseg", "test_phrase_reaseg"} and metrics["rea_test"] is None:
            metrics["rea_test"] = value
            return
        return

    if row_name == "inter":
        if dataset == "point_intseg" and metrics["inter_point"] is None:
            metrics["inter_point"] = value
            return
        if dataset == "box_intseg" and metrics["inter_box"] is None:
            metrics["inter_box"] = value
            return


def parse_log_metrics(log_path: Path, iou_metric: str) -> Dict[str, Optional[float]]:
    """從單一 log 解析指標。

    Args:
        log_path: segeval log 檔案路徑。
        iou_metric: 指定 iou 欄位使用 cIoU 或 gIoU。

    Returns:
        Dict[str, Optional[float]]: 解析後的指標字典。
    """
    metrics = _empty_metrics()
    current_dataset: Optional[str] = None
    active_dataset: Optional[str] = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        m = LOG_LOAD_DATASET_RE.search(raw_line)
        if m:
            current_dataset = m.group(1)
            active_dataset = current_dataset
            continue

        m = LOG_EVAL_RESULTS_RE.search(raw_line)
        if m:
            active_dataset = m.group(1)
            continue

        m = LOG_ERROR_RE.search(raw_line)
        if m:
            active_dataset = m.group(1)
            continue

        m = LOG_AP_ALL_RE.search(raw_line)
        if m and active_dataset is not None:
            _fill_ap_metric(metrics, active_dataset, float(m.group(1)))
            continue

        m = LOG_PANOPTIC_ALL_ROW_RE.match(line)
        if m and active_dataset is not None:
            _fill_panoptic_metric(metrics, active_dataset, float(m.group(1)))
            continue

        m = LOG_MIOU_ROW_RE.match(line)
        if m and active_dataset is not None:
            _fill_miou_metric(metrics, active_dataset, float(m.group(1)))
            continue

        m = LOG_IOU_ROW_RE.match(line)
        if m and active_dataset is not None:
            row_name, ciou, giou = m.groups()
            value = _pick_iou_value(ciou, giou, iou_metric=iou_metric)
            _fill_iou_metric(metrics, active_dataset, row_name, value)
            continue

    return metrics


def merge_metrics(metrics_list: Iterable[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """把多個 log 的指標合併（前面缺值由後面補）。

    Args:
        metrics_list: 多個指標字典的可迭代物件。

    Returns:
        Dict[str, Optional[float]]: 合併後指標字典。
    """
    merged = _empty_metrics()
    for metrics in metrics_list:
        for key, value in metrics.items():
            if merged[key] is None and value is not None:
                merged[key] = value
    return merged


def _fmt(v: Optional[float], digits: int = 2) -> str:
    """格式化數值，缺值回傳 N/A。

    Args:
        v: 原始數值。
        digits: 小數位數。

    Returns:
        str: 格式化後字串。
    """
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _triplet(a: Optional[float], b: Optional[float], c: Optional[float]) -> str:
    """輸出三欄合併字串。

    Args:
        a: 第一欄數值。
        b: 第二欄數值。
        c: 第三欄數值。

    Returns:
        str: 三欄合併字串。
    """
    return f"{_fmt(a)} / {_fmt(b)} / {_fmt(c)}"


def _pair(a: Optional[float], b: Optional[float]) -> str:
    """輸出雙欄合併字串。

    Args:
        a: 第一欄數值。
        b: 第二欄數值。

    Returns:
        str: 雙欄合併字串。
    """
    return f"{_fmt(a)} / {_fmt(b)}"


def render_markdown_table(method: str, metrics: Dict[str, Optional[float]]) -> str:
    """產生 markdown 表格字串。

    Args:
        method: 表格中的方法名稱。
        metrics: 指標字典。

    Returns:
        str: markdown 表格內容。
    """
    headers = [
        "Method",
        "Gen. Seg.<br>Pan. / Ins. / Sem.",
        "OV Seg.<br>Pan. / Ins. / Sem.",
        "Ref. Seg.<br>RefCOCO / + / g",
        "Rea. Seg.<br>Val / Test",
        "GCG Seg.<br>Val / Test",
        "Inter. Seg.<br>Point / Box",
        "VGD Seg.<br>Point / Box",
    ]

    row = [
        method,
        _triplet(metrics["gen_pan"], metrics["gen_ins"], metrics["gen_sem"]),
        _triplet(metrics["ov_pan"], metrics["ov_ins"], metrics["ov_sem"]),
        _triplet(metrics["ref_refcoco"], metrics["ref_refcocop"], metrics["ref_refcocog"]),
        _pair(metrics["rea_val"], metrics["rea_test"]),
        _pair(metrics["gcg_val"], metrics["gcg_test"]),
        _pair(metrics["inter_point"], metrics["inter_box"]),
        _pair(metrics["vgd_point"], metrics["vgd_box"]),
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(row) + " |",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """主程式：讀取 log，輸出 markdown。

    Returns:
        None.
    """
    args = parse_args()
    log_paths = [Path(p).expanduser().resolve() for p in args.logs]
    for log_path in log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"Log not found: {log_path}")

    parsed_list: List[Dict[str, Optional[float]]] = []
    for log_path in log_paths:
        parsed_list.append(parse_log_metrics(log_path, iou_metric=args.iou_metric))
    merged = merge_metrics(parsed_list)

    method = args.method
    if method is None:
        method = log_paths[0].parent.name

    md = render_markdown_table(method=method, metrics=merged)
    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    """
    python scripts/build_segeval_table.py \
      --logs runs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_plus_1024_m2f_gpu1_mixed_finetune/segeval-20260221-142642.log \
      --output ~/results.md \
      --iou-metric giou
    """

    main()
