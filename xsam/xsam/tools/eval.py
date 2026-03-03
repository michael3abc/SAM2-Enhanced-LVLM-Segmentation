#!/usr/bin/env python

import argparse
import csv
import json
import math
import os
import os.path as osp
import re
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

RESULTS_CSV_HEADERS = ["timestamp", "dataset", "status", "error", "metrics_json", "result_summary"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument(
        "--data-names",
        nargs="+",
        default=None,
        help="optional subset of dataset names to evaluate, e.g. point_intseg box_intseg",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume evaluation by skipping datasets with complete prediction files",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def _sanitize_single_line(text: str) -> str:
    """把多行文字壓成單行，方便寫入 CSV。

    Args:
        text: 原始字串。

    Returns:
        壓成單行後的字串。
    """
    return " ".join(text.split())


def _load_json_safely(file_path: str) -> Tuple[bool, Optional[Any], str]:
    """Load JSON file safely.

    Args:
        file_path: JSON file path.

    Returns:
        Tuple[bool, Optional[Any], str]:
            (is_valid, loaded_object_or_none, reason_message).
    """
    if not osp.isfile(file_path):
        return False, None, f"missing file: {file_path}"

    try:
        if osp.getsize(file_path) <= 2:
            return False, None, f"file too small: {file_path}"
    except OSError as e:
        return False, None, f"failed to stat file: {e}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return False, None, f"invalid json: {e}"

    return True, payload, "ok"


def _is_complete_predictions(
    payload: Any,
    data_name: str,
    expected_count: Optional[int] = None,
) -> Tuple[bool, str]:
    """Check whether prediction payload is complete for resume.

    Args:
        payload: Loaded predictions JSON object.
        data_name: Dataset/evaluator name.
        expected_count: Expected number of samples from dataset length.

    Returns:
        Tuple[bool, str]:
            (is_complete, reason_message).
    """
    if "semantic" in data_name:
        if not isinstance(payload, dict):
            return False, "semantic predictions must be a dict"
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            return False, "semantic predictions missing 'annotations' list"
        if len(annotations) == 0:
            return False, "semantic annotations empty"
        return True, f"semantic annotations={len(annotations)}"

    if "panoptic" in data_name:
        if not isinstance(payload, dict):
            return False, "panoptic predictions must be a dict"
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            return False, "panoptic predictions missing 'annotations' list"
        if len(annotations) == 0:
            return False, "panoptic annotations empty"
        if expected_count is not None and len(annotations) < expected_count:
            return False, f"panoptic annotations incomplete: {len(annotations)} < {expected_count}"
        return True, f"panoptic annotations={len(annotations)}"

    if not isinstance(payload, list):
        return False, "instance/ref/int/rea/vgd/gcg predictions must be a list"
    if len(payload) == 0:
        return False, "prediction list is empty"
    if expected_count is not None and len(payload) < expected_count:
        return False, f"prediction list incomplete: {len(payload)} < {expected_count}"
    return True, f"prediction list={len(payload)}"


def _should_skip_dataset_for_resume(
    evaluator: Any,
    dataset: Any,
    work_dir: str,
) -> Tuple[bool, str]:
    """Check whether current dataset can be skipped when resume is enabled.

    Args:
        evaluator: Built evaluator instance.
        dataset: Built dataset instance.
        work_dir: Evaluation work directory.

    Returns:
        Tuple[bool, str]:
            (should_skip, reason_message).
    """
    pred_file = osp.join(work_dir, "pred_data", evaluator.data_name, "predictions.json")
    ok, payload, reason = _load_json_safely(pred_file)
    if not ok:
        return False, reason

    expected_count = None
    try:
        expected_count = len(dataset)
    except Exception:
        expected_count = None

    complete, reason = _is_complete_predictions(payload, evaluator.data_name, expected_count=expected_count)
    if not complete:
        return False, reason
    return True, reason


def _collect_numeric_metrics(eval_result: Any, evaluator: Any) -> Dict[str, float]:
    """盡量抽出可序列化的數值指標。

    Args:
        eval_result: evaluator.evaluate() 的回傳值。
        evaluator: evaluator 物件本身。

    Returns:
        指標字典（僅保留有限數值）。
    """
    metrics: Dict[str, float] = {}

    if isinstance(eval_result, dict):
        for key, value in eval_result.items():
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fvalue):
                continue
            metrics[key] = fvalue

    if hasattr(evaluator, "iou_stat") and evaluator.iou_stat is not None:
        iou_stat = evaluator.iou_stat
        cat_names = getattr(iou_stat, "cat_names", [])
        ciou = getattr(iou_stat, "ciou", None)
        giou = getattr(iou_stat, "giou", None)
        if cat_names is not None and ciou is not None and giou is not None:
            for idx, cat_name in enumerate(cat_names):
                cname = str(cat_name).replace(" ", "_")
                try:
                    cval = float(ciou[idx])
                    gval = float(giou[idx])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(cval):
                    metrics[f"cIoU_{cname}"] = cval
                if math.isfinite(gval):
                    metrics[f"gIoU_{cname}"] = gval

    return metrics


def _summarize_eval_result(eval_result: Any, metrics: Dict[str, float]) -> str:
    """把 evaluator 結果整理成摘要字串。

    Args:
        eval_result: evaluator.evaluate() 的回傳值。
        metrics: 已抽出的數值指標。

    Returns:
        可寫入 CSV 的單行摘要。
    """
    if isinstance(eval_result, str):
        return _sanitize_single_line(eval_result)

    if isinstance(eval_result, dict):
        return _sanitize_single_line(json.dumps(eval_result, ensure_ascii=False, sort_keys=True))

    if eval_result is None and metrics:
        return _sanitize_single_line(json.dumps(metrics, ensure_ascii=False, sort_keys=True))

    return _sanitize_single_line(str(eval_result))


def _append_results_csv(
    csv_path: str,
    dataset_name: str,
    status: str,
    eval_result: Any = None,
    evaluator: Any = None,
    error_message: str = "",
) -> None:
    """把單一 dataset 的結果 append 到 results.csv。

    Args:
        csv_path: CSV 輸出路徑。
        dataset_name: dataset 名稱。
        status: 狀態（ok/error）。
        eval_result: evaluator.evaluate() 回傳值。
        evaluator: evaluator 物件。
        error_message: 錯誤訊息。

    Returns:
        None.
    """
    metrics = _collect_numeric_metrics(eval_result, evaluator)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "status": status,
        "error": _sanitize_single_line(error_message),
        "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        "result_summary": _summarize_eval_result(eval_result, metrics),
    }

    output_dir = osp.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = osp.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
) -> Tuple[bool, Optional[torch.Tensor]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model

    Returns:
        Tuple of (success status, segmentation outputs)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "extra_pixel_values": data["data_dict"].get("extra_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

    with torch.no_grad():
        llm_outputs, seg_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
        )

    if seg_outputs is None:
        llm_generation_output = ""
        if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
            llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)

        print_log(
            rf"Failed to get segmentation outputs: {image_files}, "
            rf"llm question_input: {repr(llm_question_input)}, "
            rf"llm generation_output: {repr(llm_generation_output)}",
            logger="current",
        )
        return False, None

    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)
        gcg_phrases = [
            get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in llm_outputs.sequences
        ]
        gcg_captions = [get_gcg_caption(output) for output in llm_generation_output]
        for i, segmentation_output in enumerate(seg_outputs):
            segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    return True, seg_outputs


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
) -> Any:
    """Evaluate model on a single dataset."""
    data_name = evaluator.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output
    mode = "tensor" if output_ids_with_output else "predict"

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=4, sampler=sampler, shuffle=False, collate_fn=xsam_collate_fn
    )

    # Evaluation loop
    failed_cnt = 0
    evaluator.reset()
    print_log(f"Evaluating {data_name}...", logger="current")

    for data in tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0):
        success, seg_outputs = process_batch(model, data, data_name, metadata, generation_config, stop_criteria, mode)
        if not success:
            failed_cnt += 1
            continue

        image_infos = data["data_samples"].metainfo["image_infos"]
        evaluator.process(image_infos, seg_outputs)

    print_log(f"Failed number of {data_name}: {failed_cnt}", logger="current")
    print_log(f"Evaluating {data_name} done!", logger="current")
    return evaluator.evaluate()


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Build and setup model
    model = BUILDER.build(cfg.model)
    if "llm" in cfg.model:
        model.llm.to(cfg.model.llm.torch_dtype)
    model.eval()
    model = model.to(get_device())
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank]).module

    load_checkpoint(model, args.pth_model)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Evaluate on all datasets
    assert len(cfg.val_datasets) == len(
        cfg.val_evaluators
    ), f"len(cfg.val_datasets) = {len(cfg.val_datasets)}, len(cfg.val_evaluators) = {len(cfg.val_evaluators)}"
    selected_data_names = set(args.data_names) if args.data_names is not None else None
    if selected_data_names is None:
        print_log(f"Evaluating {len(cfg.val_datasets)} datasets...", logger="current")
    else:
        print_log(
            f"Evaluating selected datasets: {sorted(selected_data_names)} (from total {len(cfg.val_datasets)})",
            logger="current",
        )
    results_csv_path = osp.join(args.work_dir, "pred_data", "results.csv")
    for dataset_cfg, evaluator_cfg in zip(cfg.val_datasets, cfg.val_evaluators):
        dataset_name = evaluator_cfg.get("data_name", None) if isinstance(evaluator_cfg, dict) else None
        if selected_data_names is not None and dataset_name not in selected_data_names:
            continue
        try:
            dataset = BUILDER.build(dataset_cfg)
            model.postprocess_fn = dataset.postprocess_fn

            evaluator = BUILDER.build(evaluator_cfg)
            evaluator.metadata = dataset.metadata
            evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)
            if args.resume:
                should_skip, skip_reason = _should_skip_dataset_for_resume(evaluator, dataset, args.work_dir)
                if should_skip:
                    print_log(
                        f"[resume] Skip {evaluator.data_name}: found complete predictions ({skip_reason})",
                        logger="current",
                    )
                    if rank == 0:
                        _append_results_csv(
                            csv_path=results_csv_path,
                            dataset_name=evaluator.data_name,
                            status="skip",
                            eval_result=None,
                            evaluator=None,
                            error_message=skip_reason,
                        )
                    continue
                print_log(
                    f"[resume] Re-run {evaluator.data_name}: predictions not complete ({skip_reason})",
                    logger="current",
                )
            eval_result = evaluate_dataset(model, dataset, evaluator, rank, world_size, generation_config, stop_criteria)
            if rank == 0:
                _append_results_csv(
                    csv_path=results_csv_path,
                    dataset_name=evaluator.data_name,
                    status="ok",
                    eval_result=eval_result,
                    evaluator=evaluator,
                )
        except Exception as e:
            print_log(f"Error evaluating {dataset_cfg.data_name}\n: {e}\n{traceback.format_exc()}", logger="current")
            if rank == 0:
                cur_name = dataset_name or getattr(dataset_cfg, "data_name", "unknown_dataset")
                _append_results_csv(
                    csv_path=results_csv_path,
                    dataset_name=cur_name,
                    status="error",
                    eval_result=None,
                    evaluator=None,
                    error_message=str(e),
                )
            continue


if __name__ == "__main__":
    main()
