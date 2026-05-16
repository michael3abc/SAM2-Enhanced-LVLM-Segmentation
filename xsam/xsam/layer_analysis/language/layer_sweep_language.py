#!/usr/bin/env python
"""Language layer-sweep scheduler entrypoint."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Bootstrap import paths for direct script execution.
_THIS_PATH = Path(__file__).resolve()
_PROJECT_ROOT = None
for _candidate in [_THIS_PATH, *_THIS_PATH.parents]:
    if (_candidate / "xsam" / "xsam").exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    raise FileNotFoundError(f"Cannot locate project root from: {_THIS_PATH}")
for _path in [_PROJECT_ROOT, _PROJECT_ROOT / "xsam"]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from xsam.layer_analysis.common.config import deep_merge_dict, load_yaml_config, resolve_phase_config
from xsam.layer_analysis.common.io import (
    append_summary_row,
    ensure_dir,
    parse_layer_ids,
    read_summary_rows,
    select_layers_by_metric,
)
from xsam.layer_analysis.common.runtime import ensure_project_paths, resolve_path


@dataclass
class LanguageSweepDefaults:
    """Built-in defaults for language layer sweep.

    Args:
        run_root: Sweep run root directory.
        mmengine_config: MMEngine config used for training.
        metric_key: Metric key for top-k/best selection.
        metric_mode: Metric optimization direction.
        profile_root: Profile directory root.
        layers_all: Candidate layer ids for phase1.
        topk: Number of layers kept for phase2.
        train_command: Training command template.
    Returns:
        None.
    """

    run_root: str = "runs/sweep_language"
    mmengine_config: str = "xsam/xsam/configs/xsam/layer_analysis/language/xsam_sam3_language.py"
    metric_key: str = "loss"
    metric_mode: str = "min"
    profile_root: str = "xsam/xsam/configs/xsam/layer_analysis/language/profiles"
    layers_all: List[int] = field(default_factory=lambda: [-1, -2, -4, -6, -8, -10, -12, -16, -24, -32])
    topk: int = 5
    train_command: str = (
        "XSAM_LAYER_ID={layer_id} bash run.sh --modes train --config {mmengine_config} "
        "--yaml {profile_yaml} --work-dir {layer_work_dir}"
    )

    def to_raw_cfg(self) -> Dict[str, Any]:
        """Convert defaults into raw sweep config mapping.

        Args:
            None.
        Returns:
            Raw config dictionary.
        """
        return {
            "version": 1,
            "paths": {
                "run_root": self.run_root,
                "mmengine_config": self.mmengine_config,
            },
            "runtime": {
                "metric_key": self.metric_key,
                "metric_mode": self.metric_mode,
            },
            "runner": {
                "train_command": self.train_command,
            },
            "common": {
                "profile_root": self.profile_root,
                "layers_all": list(self.layers_all),
            },
            "default_phase": "phase1",
            "phases": {
                "phase1": {
                    "profile_yaml": "{profile_root}/phase1.yaml",
                    "layers": "all",
                },
                "phase2": {
                    "profile_yaml": "{profile_root}/phase2.yaml",
                    "layers": "topk",
                    "topk": int(self.topk),
                    "topk_from_phase": "phase1",
                },
                "phase3": {
                    "profile_yaml": "{profile_root}/phase3.yaml",
                    "layers": "best",
                    "best_from_phase": "phase2",
                },
            },
        }


def _load_effective_raw_cfg(project_root: Path, config_yaml: Optional[str]) -> tuple[Dict[str, Any], str]:
    """Load effective sweep config from built-in defaults and optional override YAML.

    Args:
        project_root: Project root path.
        config_yaml: Optional override YAML path.
    Returns:
        Tuple of merged raw config and source tag/path.
    """
    default_cfg = LanguageSweepDefaults().to_raw_cfg()
    if config_yaml in (None, ""):
        return default_cfg, "<builtin-defaults>"

    config_yaml_path = resolve_path(project_root, str(config_yaml))
    if not config_yaml_path.exists():
        raise FileNotFoundError(f"Sweep override YAML not found: {config_yaml_path}")
    override_cfg = load_yaml_config(str(config_yaml_path))
    merged_cfg = deep_merge_dict(default_cfg, override_cfg)
    return merged_cfg, str(config_yaml_path)


def _load_phase_profile_cfg(profile_yaml_path: Path) -> Dict[str, Any]:
    """Load phase profile YAML.

    Args:
        profile_yaml_path: Phase profile YAML path.
    Returns:
        Parsed profile mapping.
    """
    profile_cfg = load_yaml_config(str(profile_yaml_path))
    if not isinstance(profile_cfg, dict):
        raise TypeError(f"Phase profile YAML must be a mapping: {profile_yaml_path}")
    return profile_cfg


def _resolve_effective_profile_yaml(
    raw_cfg: Dict[str, Any],
    phase: str,
    phase_dir: Path,
    phase_profile_yaml_path: Path,
    dry_run: bool,
) -> Path:
    """Resolve effective profile YAML path for training.

    Args:
        raw_cfg: Raw sweep config mapping.
        phase: Phase name.
        phase_dir: Phase output directory.
        phase_profile_yaml_path: Phase-specific profile YAML path.
        dry_run: Whether current run is dry-run.
    Returns:
        Effective profile YAML path.
    """
    profile_common = raw_cfg.get("profile_common", {})
    if profile_common in (None, {}):
        return phase_profile_yaml_path
    if not isinstance(profile_common, dict):
        raise TypeError("`profile_common` must be a mapping in sweep yaml.")

    phase_profile_cfg = _load_phase_profile_cfg(phase_profile_yaml_path)
    merged_profile_cfg = deep_merge_dict(profile_common, phase_profile_cfg)
    merged_profile_yaml_path = phase_dir / f"{phase}_merged_profile.yaml"

    if not dry_run:
        with open(merged_profile_yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(merged_profile_cfg, handle, sort_keys=False, allow_unicode=False)
    return merged_profile_yaml_path


def _format_template(template: str, context: Dict[str, Any], field_name: str) -> str:
    """Render template string with runtime context.

    Args:
        template: Template string with ``str.format`` placeholders.
        context: Placeholder dictionary.
        field_name: Human-readable field name for error messages.
    Returns:
        Rendered string.
    """
    try:
        return template.format(**context)
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise KeyError(f"`{field_name}` template references missing key: `{missing_key}`.") from exc


def _resolve_phase_profile_yaml(
    project_root: Path,
    phase_cfg: Dict[str, Any],
    phase: str,
    run_root: Path,
) -> Path:
    """Resolve per-phase profile YAML path.

    Args:
        project_root: Project root path.
        phase_cfg: Resolved phase config.
        phase: Phase name.
        run_root: Sweep run root directory.
    Returns:
        Absolute profile YAML path.
    """
    raw_profile_yaml = phase_cfg.get("profile_yaml")
    if raw_profile_yaml in (None, ""):
        raise ValueError(f"`phases.{phase}.profile_yaml` is required.")

    profile_root_raw = phase_cfg.get("profile_root", "")
    profile_root = resolve_path(project_root, str(profile_root_raw)) if str(profile_root_raw) else project_root
    rendered_profile_yaml = _format_template(
        template=str(raw_profile_yaml),
        context=dict(
            phase=phase,
            project_root=str(project_root),
            run_root=str(run_root),
            profile_root=str(profile_root),
        ),
        field_name=f"phases.{phase}.profile_yaml",
    )
    profile_yaml_path = resolve_path(project_root, rendered_profile_yaml)
    if not profile_yaml_path.exists():
        raise FileNotFoundError(f"Profile YAML not found for phase `{phase}`: {profile_yaml_path}")
    return profile_yaml_path


def _parse_metric_from_text(metric_key: str, text: str, metric_regex: Optional[str] = None) -> Optional[float]:
    """Parse metric value from log text.

    Args:
        metric_key: Metric key name.
        text: Log file text.
        metric_regex: Optional explicit regex with one capturing group.
    Returns:
        Last matched metric value or ``None``.
    """
    regex = metric_regex
    if regex in (None, ""):
        key_escaped = re.escape(metric_key)
        regex = rf"{key_escaped}\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    matches = re.findall(regex, text)
    if not matches:
        return None
    last_value = matches[-1]
    if isinstance(last_value, tuple):
        last_value = last_value[-1]
    try:
        return float(last_value)
    except (TypeError, ValueError):
        return None


def _extract_metric_from_layer_logs(
    layer_work_dir: Path,
    metric_key: str,
    metric_regex: Optional[str] = None,
) -> Optional[float]:
    """Extract metric from latest train log under a layer work directory.

    Args:
        layer_work_dir: Layer work directory path.
        metric_key: Metric key name.
        metric_regex: Optional explicit regex with one capturing group.
    Returns:
        Parsed metric value or ``None``.
    """
    if not layer_work_dir.exists():
        return None
    log_files = sorted(layer_work_dir.glob("train-*.log"), key=lambda path: path.stat().st_mtime)
    if not log_files:
        return None
    latest_log = log_files[-1]
    log_text = latest_log.read_text(encoding="utf-8", errors="ignore")
    return _parse_metric_from_text(metric_key=metric_key, text=log_text, metric_regex=metric_regex)


def _select_layers_from_phase_logs(
    phase_dir: Path,
    metric_key: str,
    metric_mode: str,
    topk: int,
    metric_regex: Optional[str] = None,
) -> List[int]:
    """Fallback-select layers directly from existing per-layer train logs.

    Args:
        phase_dir: Phase directory that contains ``layer_*`` work dirs.
        metric_key: Metric key name.
        metric_mode: ``min`` or ``max``.
        topk: Number of layers to keep.
        metric_regex: Optional explicit regex with one capturing group.
    Returns:
        Selected layer ids.
    """

    rows: List[Dict[str, str]] = []
    if not phase_dir.exists():
        return []

    for layer_work_dir in sorted(phase_dir.glob("layer_*")):
        try:
            layer_id = int(layer_work_dir.name.replace("layer_", "", 1))
        except ValueError:
            continue
        metric_value = _extract_metric_from_layer_logs(
            layer_work_dir=layer_work_dir,
            metric_key=metric_key,
            metric_regex=metric_regex,
        )
        if metric_value is None:
            continue
        rows.append(
            {
                "layer_id": str(layer_id),
                "status": "ok",
                metric_key: str(metric_value),
            }
        )

    return select_layers_by_metric(rows, metric_key=metric_key, metric_mode=metric_mode, topk=topk)


def _parse_cli_args() -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        None.
    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Language layer-sweep scheduler")
    parser.add_argument(
        "--config-yaml",
        dest="config_yaml",
        type=str,
        default=None,
        help="Optional sweep override YAML path. Defaults are built into this script.",
    )
    parser.add_argument("--phase", type=str, required=True, help="Phase name, e.g. phase1/phase2/phase3.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["plan", "train"],
        required=True,
        help="Mode: plan layer ids or run train commands.",
    )
    parser.add_argument(
        "--single-layer-id",
        dest="single_layer_id",
        type=int,
        default=None,
        help="Override target layer for train mode.",
    )
    parser.add_argument(
        "--print-layer-ids",
        dest="print_layer_ids",
        action="store_true",
        help="Print resolved layer ids as CSV.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Do not execute train command; only print it.",
    )
    return parser.parse_args()


def _resolve_layer_ids(
    project_root: Path,
    raw_cfg: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    phase: str,
    dry_run: bool = False,
) -> List[int]:
    """Resolve layer ids for a given phase.

    Args:
        project_root: Project root path.
        raw_cfg: Raw YAML config.
        phase_cfg: Resolved phase config.
        phase: Phase name.
        dry_run: Whether to allow fallback planning without prior summaries.
    Returns:
        Resolved layer id list.
    """
    strategy = str(phase_cfg.get("layers", "all"))
    run_root = resolve_path(project_root, str(phase_cfg.get("run_root", raw_cfg.get("paths", {}).get("run_root"))))
    metric_key = str(phase_cfg.get("metric_key", raw_cfg.get("runtime", {}).get("metric_key", "val_ce_loss")))
    metric_mode = str(phase_cfg.get("metric_mode", raw_cfg.get("runtime", {}).get("metric_mode", "min")))
    metric_regex = phase_cfg.get("metric_regex", raw_cfg.get("runtime", {}).get("metric_regex"))
    if metric_regex is not None:
        metric_regex = str(metric_regex)

    if strategy == "all":
        return parse_layer_ids(phase_cfg.get("layers_all", raw_cfg.get("common", {}).get("layers_all", [])))

    if strategy == "topk":
        from_phase = str(phase_cfg.get("topk_from_phase", "phase1"))
        topk = int(phase_cfg.get("topk", 5))
        summary_csv = run_root / from_phase / "summary.csv"
        rows = read_summary_rows(summary_csv)
        layer_ids = select_layers_by_metric(rows, metric_key=metric_key, metric_mode=metric_mode, topk=topk)
        if not layer_ids:
            layer_ids = _select_layers_from_phase_logs(
                phase_dir=run_root / from_phase,
                metric_key=metric_key,
                metric_mode=metric_mode,
                topk=topk,
                metric_regex=metric_regex,
            )
        if not layer_ids:
            if dry_run:
                fallback_layers = parse_layer_ids(phase_cfg.get("layers_all", raw_cfg.get("common", {}).get("layers_all", [])))
                if fallback_layers:
                    return fallback_layers[: max(topk, 1)]
            raise RuntimeError(f"No available layers from {summary_csv}. Run `{from_phase}` first.")
        return layer_ids

    if strategy == "best":
        from_phase = str(phase_cfg.get("best_from_phase", "phase2"))
        summary_csv = run_root / from_phase / "summary.csv"
        rows = read_summary_rows(summary_csv)
        layer_ids = select_layers_by_metric(rows, metric_key=metric_key, metric_mode=metric_mode, topk=1)
        if not layer_ids:
            layer_ids = _select_layers_from_phase_logs(
                phase_dir=run_root / from_phase,
                metric_key=metric_key,
                metric_mode=metric_mode,
                topk=1,
                metric_regex=metric_regex,
            )
        if not layer_ids:
            if dry_run:
                fallback_layers = parse_layer_ids(phase_cfg.get("layers_all", raw_cfg.get("common", {}).get("layers_all", [])))
                if fallback_layers:
                    return [fallback_layers[0]]
            raise RuntimeError(f"No available best layer from {summary_csv}. Run `{from_phase}` first.")
        return layer_ids

    if strategy == "single":
        if "single_layer_id" not in phase_cfg:
            raise ValueError("`layers=single` requires `single_layer_id` in phase config.")
        return [int(phase_cfg["single_layer_id"])]

    raise ValueError(f"Unsupported layer strategy `{strategy}` for phase `{phase}`.")


def _build_train_command(
    command_template: str,
    template_context: Dict[str, Any],
) -> str:
    """Build concrete train command from a template string.

    Args:
        command_template: Command template.
        template_context: Placeholder context dictionary.
    Returns:
        Concrete shell command string.
    """
    return _format_template(
        template=command_template,
        context=template_context,
        field_name="runner.train_command",
    )


def _run_phase_train(
    project_root: Path,
    raw_cfg: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    phase: str,
    config_yaml: str,
    layer_ids: List[int],
    dry_run: bool,
) -> None:
    """Run train commands for all target layers and append summary rows.

    Args:
        project_root: Project root path.
        raw_cfg: Raw YAML config.
        phase_cfg: Resolved phase config.
        phase: Phase name.
        config_yaml: YAML path.
        layer_ids: Target layer ids.
        dry_run: Whether to skip actual subprocess execution.
    Returns:
        None.
    """
    run_root = resolve_path(project_root, str(phase_cfg.get("run_root", raw_cfg.get("paths", {}).get("run_root"))))
    phase_dir = run_root / phase
    if not dry_run:
        phase_dir = ensure_dir(phase_dir)
    summary_csv = phase_dir / "summary.csv"
    mmengine_config = phase_cfg.get("mmengine_config")
    if mmengine_config in (None, ""):
        raise ValueError("`paths.mmengine_config` is required in language sweep yaml.")
    mmengine_config_path = resolve_path(project_root, str(mmengine_config))
    if not mmengine_config_path.exists():
        raise FileNotFoundError(f"Cannot find mmengine config: {mmengine_config_path}")
    profile_yaml_path = _resolve_phase_profile_yaml(project_root, phase_cfg, phase=phase, run_root=run_root)
    effective_profile_yaml_path = _resolve_effective_profile_yaml(
        raw_cfg=raw_cfg,
        phase=phase,
        phase_dir=phase_dir,
        phase_profile_yaml_path=profile_yaml_path,
        dry_run=dry_run,
    )
    metric_key = str(phase_cfg.get("metric_key", raw_cfg.get("runtime", {}).get("metric_key", "val_ce_loss")))
    metric_regex = phase_cfg.get("metric_regex", raw_cfg.get("runtime", {}).get("metric_regex"))
    if metric_regex is not None:
        metric_regex = str(metric_regex)

    runner_cfg = phase_cfg.get("runner", {})
    if not isinstance(runner_cfg, dict):
        runner_cfg = {}
    command_template = runner_cfg.get("train_command")
    if command_template in (None, ""):
        command_template = phase_cfg.get("train_command")
    if command_template is None:
        raise ValueError(
            "`runner.train_command` is required for train mode. "
            "Set it in sweep overrides or built-in defaults."
        )

    for layer_id in layer_ids:
        layer_work_dir = phase_dir / f"layer_{int(layer_id)}"
        if not dry_run:
            layer_work_dir = ensure_dir(layer_work_dir)
        template_context = dict(
            phase=phase,
            layer_id=int(layer_id),
            config_yaml=config_yaml,
            mmengine_config=str(mmengine_config_path),
            profile_yaml=str(effective_profile_yaml_path),
            run_root=str(run_root),
            phase_dir=str(phase_dir),
            layer_work_dir=str(layer_work_dir),
            project_root=str(project_root),
        )
        command = _build_train_command(
            command_template=str(command_template),
            template_context=template_context,
        )
        start_time = time.time()
        if dry_run:
            print(command)
            continue

        completed = subprocess.run(command, shell=True, cwd=str(project_root), check=False)
        elapsed = time.time() - start_time
        metric_value = None
        if completed.returncode == 0:
            metric_value = _extract_metric_from_layer_logs(
                layer_work_dir=layer_work_dir,
                metric_key=metric_key,
                metric_regex=metric_regex,
            )
        row = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            phase=phase,
            layer_id=int(layer_id),
            status="ok" if completed.returncode == 0 else "failed",
            return_code=int(completed.returncode),
            elapsed_sec=round(float(elapsed), 3),
            command=command,
            profile_yaml=str(effective_profile_yaml_path),
            mmengine_config=str(mmengine_config_path),
            layer_work_dir=str(layer_work_dir),
            **({metric_key: metric_value} if metric_value is not None else {}),
        )
        append_summary_row(summary_csv, row)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Language layer train failed at phase={phase}, layer={layer_id}, "
                f"return_code={completed.returncode}"
            )


def main() -> None:
    """Run language layer-sweep scheduler.

    Args:
        None.
    Returns:
        None.
    """
    project_root = ensure_project_paths(__file__)
    cli_args = _parse_cli_args()

    raw_cfg, config_source = _load_effective_raw_cfg(project_root=project_root, config_yaml=cli_args.config_yaml)
    phase_cfg = resolve_phase_config(raw_cfg, cli_args.phase)
    layer_ids = _resolve_layer_ids(project_root, raw_cfg, phase_cfg, cli_args.phase, dry_run=bool(cli_args.dry_run))

    if cli_args.single_layer_id is not None:
        layer_ids = [int(cli_args.single_layer_id)]

    if cli_args.mode == "plan" or cli_args.print_layer_ids:
        print(",".join(str(layer_id) for layer_id in layer_ids))
        if cli_args.mode == "plan":
            return

    if cli_args.mode == "train":
        _run_phase_train(
            project_root=project_root,
            raw_cfg=raw_cfg,
            phase_cfg=phase_cfg,
            phase=cli_args.phase,
            config_yaml=str(config_source),
            layer_ids=layer_ids,
            dry_run=bool(cli_args.dry_run),
        )


if __name__ == "__main__":
    main()

"""
python xsam/xsam/layer_analysis/language/layer_sweep_language.py \
  --phase phase2 \
  --mode plan \
  --print-layer-ids

python xsam/xsam/layer_analysis/language/layer_sweep_language.py \
  --config-yaml xsam/xsam/configs/xsam/layer_analysis/language/profiles/language_sweep.yaml \
  --phase phase2 \
  --mode train \
  --single-layer-id -12
"""
