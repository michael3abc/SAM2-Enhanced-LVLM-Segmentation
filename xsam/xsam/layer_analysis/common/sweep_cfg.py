"""Shared config dataclass for SAM3 spatial layer sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SaptialSweepCfg:
    """Typed runtime config for SAM3 spatial layer sweep.

    Args:
        config: Absolute/relative path to the mmengine config file.
        pth_model: Optional path to X-SAM checkpoint.
        layers: Comma-separated trunk layer ids to evaluate.
        data_names: Optional explicit evaluator data-name allowlist.
        dense_keywords: Comma-separated keywords for dense-task evaluator filtering.
        ref_keywords: Comma-separated keywords for ref-task evaluator filtering.
        batch_size: Eval dataloader batch size.
        num_workers: Eval dataloader workers.
        max_samples_per_task: Eval sample cap per task; 0 means full dataset.
        train_epochs: Probe training epochs.
        train_ratio: Ratio of train dataset used for probe training.
        train_batch_size: Probe train dataloader batch size.
        grad_accum_steps: Gradient accumulation steps for probe training.
        train_num_workers: Probe train dataloader workers.
        save_steps: Save checkpoint every N steps. 0 means no periodic save.
        max_save: Checkpoint retention policy.
        resume: Whether to resume probe training from latest/specified checkpoint.
        resume_ckpt: Optional checkpoint path for resume.
        probe_lr: Probe optimizer learning rate.
        probe_weight_decay: Probe optimizer weight decay.
        probe_reinit: Whether to reinitialize probe-head weights from scratch.
        train_eval_interval: Train-time eval interval in steps. 0 disables train-time eval.
        train_eval_max_samples: Max samples for each train-time eval snapshot. 0 means full dataset.
        early_stop_patience_steps: Early-stop patience in steps without sufficient mIoU improvement. 0 disables.
        early_stop_miou_eps: Minimum mIoU gain (percentage points) to count as improvement.
        seed_stride: Per-layer random init seed stride.
        output_csv: Output CSV path (relative paths are resolved under run dir).
        output_root: Root directory for sweep run artifacts.
        run_name: Optional run name.
        use_tqdm: Whether to enable tqdm progress bars.
        log_interval: MMEngine logging interval.
        eval_fail_fast: Whether to stop evaluation early if failure ratio is too high.
        eval_fail_ratio_threshold: Per-layer failure ratio threshold for fail-fast.
        eval_fail_check_min_samples: Minimum processed samples before fail-fast check.
        eval_oom_empty_cache: Whether to call ``torch.cuda.empty_cache()`` on OOM during eval.
        eval_log_cuda_mem: Whether to include CUDA memory info in eval progress logs.
        seed: Global random seed.
        cfg_options: Optional mmengine merge options.
    Returns:
        None.
    """

    config: str
    pth_model: Optional[str]
    layers: str
    data_names: Optional[List[str]]
    dense_keywords: str
    ref_keywords: str
    batch_size: int
    num_workers: int
    max_samples_per_task: int
    train_epochs: int
    train_ratio: float
    train_batch_size: int
    grad_accum_steps: int
    train_num_workers: int
    save_steps: int
    max_save: int
    resume: bool
    resume_ckpt: Optional[str]
    probe_lr: float
    probe_weight_decay: float
    probe_reinit: bool
    train_eval_interval: int
    train_eval_max_samples: int
    early_stop_patience_steps: int
    early_stop_miou_eps: float
    seed_stride: int
    output_csv: Optional[str]
    output_root: str
    run_name: Optional[str]
    use_tqdm: bool
    log_interval: int
    eval_fail_fast: bool
    eval_fail_ratio_threshold: float
    eval_fail_check_min_samples: int
    eval_oom_empty_cache: bool
    eval_log_cuda_mem: bool
    seed: int
    cfg_options: Optional[Dict[str, str]]
