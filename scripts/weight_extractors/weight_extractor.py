#!/usr/bin/env python3
"""Extract module-specific weights from .bin/.pt/HF checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:  # pragma: no cover
    safetensors_load_file = None


MODULE_PREFIX_MAP: Mapping[str, Sequence[str]] = {
    "seg_encoder": ("segmentor.encoder.",),
    "seg_connector": ("seg_connector.",),
    "seg_decoder_wproj": (
        "segmentor.decoder.",
        "segmentor.pixel_decoder.",
        "segmentor.sam2_fpn_bridge.",
        "segmentor.class_predictor.",
        "segmentor.prompt_encoder.",
    ),
    "seg_decoder": (
        "segmentor.decoder.",
        "segmentor.pixel_decoder.",
    ),
    "llm": ("llm.",),
    "img_encoder": ("visual_encoder.",),
    "encoder_llm_porjector": (
        "visual_projector.",
        "seg_projector.",
        "llm_projector.",
    ),
}

MODULE_ALIAS_MAP: Mapping[str, str] = {
    "encoder_llm_projector": "encoder_llm_porjector",
}

SUPPORTED_INPUT_EXTS = {".bin", ".pt", ".pth", ".safetensors"}
SUPPORTED_OUTPUT_NAME_EXTS = {".bin", ".pt", ".hf"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        None
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Input model path: .bin/.pt/.pth/.safetensors file, or HF directory.",
    )
    parser.add_argument(
        "--extract_moudle",
        required=True,
        help=(
            "Module name to extract. "
            "Supported: seg_encoder, seg_connector, seg_decoder_wproj, seg_decoder, "
            "llm, img_encoder, encoder_llm_porjector."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="inits/extracted_weights/",
        help="Directory to place extracted weights. Default: inits/extracted_weights/",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help=(
            "Output file/folder name. Default: <extract_moudle>.bin. "
            "Use suffix .bin/.pt/.hf to choose output type."
        ),
    )
    parser.add_argument(
        "--hf_base",
        default=None,
        help=(
            "Optional HF base directory for copying config/tokenizer/remote-code files "
            "when output is HF and extract_moudle=llm."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file/directory.",
    )
    return parser.parse_args()


def normalize_extract_module(extract_module: str) -> str:
    """Normalize extract module name with aliases.

    Args:
        extract_module (str): User requested module name.
    Returns:
        str: Canonical module name.
    """
    module = extract_module.strip()
    module = MODULE_ALIAS_MAP.get(module, module)
    if module not in MODULE_PREFIX_MAP:
        supported = ", ".join(sorted(set(MODULE_PREFIX_MAP.keys()) | set(MODULE_ALIAS_MAP.keys())))
        raise ValueError(f"Unsupported extract_moudle={extract_module}. Supported: {supported}")
    return module


def infer_is_hf_dir(model_path: Path) -> bool:
    """Infer whether path points to an HF-style directory.

    Args:
        model_path (Path): Input model path.
    Returns:
        bool: True if path looks like HF directory.
    """
    if not model_path.is_dir():
        return False
    marker_files = (
        "config.json",
        "tokenizer_config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    return any((model_path / name).exists() for name in marker_files)


def normalize_checkpoint_payload(payload: object) -> Dict[str, torch.Tensor]:
    """Normalize raw payload to pure state_dict.

    Args:
        payload (object): Object loaded from checkpoint.
    Returns:
        Dict[str, torch.Tensor]: State dictionary.
    """
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        elif "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return payload


def load_single_state_file(path: Path) -> Dict[str, torch.Tensor]:
    """Load one checkpoint file.

    Args:
        path (Path): Checkpoint file path.
    Returns:
        Dict[str, torch.Tensor]: Loaded state dict.
    """
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError("safetensors is not installed, cannot load .safetensors")
        return safetensors_load_file(str(path), device="cpu")
    if suffix not in SUPPORTED_INPUT_EXTS:
        raise ValueError(f"Unsupported checkpoint file extension: {path.suffix}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return normalize_checkpoint_payload(payload)


def load_indexed_hf_state_dict(model_dir: Path, index_file: Path) -> Dict[str, torch.Tensor]:
    """Load sharded HF state dict via index json.

    Args:
        model_dir (Path): HF model directory.
        index_file (Path): Index json path.
    Returns:
        Dict[str, torch.Tensor]: Merged state dict.
    """
    with index_file.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid HF index file: {index_file}")

    shard_files = sorted(set(weight_map.values()))
    merged: Dict[str, torch.Tensor] = {}
    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard {shard_name} referenced by {index_file}")
        merged.update(load_single_state_file(shard_path))
    return merged


def load_hf_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load state dict from HF model directory.

    Args:
        model_dir (Path): HF directory path.
    Returns:
        Dict[str, torch.Tensor]: Loaded state dict.
    """
    safetensors_index = model_dir / "model.safetensors.index.json"
    pytorch_index = model_dir / "pytorch_model.bin.index.json"
    if safetensors_index.exists():
        return load_indexed_hf_state_dict(model_dir, safetensors_index)
    if pytorch_index.exists():
        return load_indexed_hf_state_dict(model_dir, pytorch_index)

    for single_name in ("model.safetensors", "pytorch_model.bin"):
        single_file = model_dir / single_name
        if single_file.exists():
            return load_single_state_file(single_file)

    shard_patterns = ("model-*.safetensors", "pytorch_model-*.bin")
    shard_files: List[Path] = []
    for pattern in shard_patterns:
        shard_files.extend(sorted(model_dir.glob(pattern)))
    if shard_files:
        merged: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            merged.update(load_single_state_file(shard))
        return merged

    raise FileNotFoundError(
        f"No supported HF weight file found in {model_dir}. "
        "Expected model.safetensors / pytorch_model.bin / index json / shard files."
    )


def load_model_state_dict(model_path: Path) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Load state dict from a file or HF directory.

    Args:
        model_path (Path): Input model path.
    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: (state_dict, is_hf_input)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    is_hf_input = infer_is_hf_dir(model_path)
    if model_path.is_file():
        state_dict = load_single_state_file(model_path)
    elif is_hf_input:
        state_dict = load_hf_state_dict(model_path)
    else:
        raise ValueError(
            f"Unsupported model_path directory: {model_path}. "
            "Only HF-style directories are supported for directory input."
        )

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Loaded empty state dict from {model_path}")
    return state_dict, is_hf_input


def is_probably_unprefixed_llm(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Check if state dict likely contains unprefixed plain LLM keys.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Input state dict.
    Returns:
        bool: True if keys look like unprefixed LLM keys.
    """
    keys = list(state_dict.keys())
    if any(key.startswith("llm.") for key in keys):
        return False
    llm_markers = (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
        "lm_head.",
        "transformer.h.",
        "transformer.wte.",
    )
    return any(any(key.startswith(marker) for marker in llm_markers) for key in keys)


def extract_by_module(
    state_dict: Mapping[str, torch.Tensor], extract_module: str
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Extract module-specific weights by prefixes.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Source state dict.
        extract_module (str): Canonical extract module.
    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: (extracted state dict, used_llm_fallback)
    """
    prefixes = tuple(MODULE_PREFIX_MAP[extract_module])
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefixes):
            extracted[key] = value

    used_llm_fallback = False
    if extract_module == "llm" and not extracted and is_probably_unprefixed_llm(state_dict):
        extracted = dict(state_dict)
        used_llm_fallback = True

    if not extracted:
        raise ValueError(
            f"No weights matched extract_moudle={extract_module}. "
            f"Checked prefixes={list(prefixes)}"
        )
    return extracted, used_llm_fallback


def strip_prefix_if_present(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Strip prefix from keys when all keys share that prefix.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Input state dict.
        prefix (str): Prefix to strip.
    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: (possibly stripped state dict, stripped flag)
    """
    keys = list(state_dict.keys())
    if not keys:
        return {}, False
    if not all(key.startswith(prefix) for key in keys):
        return dict(state_dict), False
    return {key[len(prefix) :]: value for key, value in state_dict.items()}, True


def normalize_output_name(output_name: str | None, extract_module: str) -> str:
    """Build normalized output name.

    Args:
        output_name (str | None): User-provided output name.
        extract_module (str): Canonical module name.
    Returns:
        str: Normalized output name.
    """
    if output_name is None or output_name.strip() == "":
        return f"{extract_module}.bin"
    return Path(output_name.strip()).name


def infer_output_target(output_dir: Path, output_name: str) -> Tuple[str, Path]:
    """Infer output type/path from output_name suffix.

    Args:
        output_dir (Path): Output base directory.
        output_name (str): Output name with suffix.
    Returns:
        Tuple[str, Path]: (output_form, output_path)
    """
    suffix = Path(output_name).suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_NAME_EXTS:
        raise ValueError(
            f"output_name must end with one of {sorted(SUPPORTED_OUTPUT_NAME_EXTS)}; "
            f"got {output_name}"
        )
    if suffix == ".hf":
        return "hf", output_dir / Path(output_name).stem
    return suffix, output_dir / output_name


def ensure_overwrite(output_path: Path, output_form: str, overwrite: bool) -> None:
    """Validate overwrite policy and cleanup existing target if needed.

    Args:
        output_path (Path): Target path.
        output_form (str): Output form, one of .bin/.pt/hf.
        overwrite (bool): Whether overwrite is allowed.
    Returns:
        None
    """
    if not output_path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace.")
    if output_form == "hf":
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    else:
        output_path.unlink()


def infer_remote_code_filenames(hf_dir: Path) -> List[str]:
    """Infer remote-code companion file names in HF directory.

    Args:
        hf_dir (Path): HF directory path.
    Returns:
        List[str]: Candidate support file names.
    """
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "modeling_phi3.py",
        "configuration_phi3.py",
        "tokenization_phi3.py",
        "tokenization_phi3_fast.py",
    ]
    for meta_name in ("config.json", "tokenizer_config.json"):
        meta_path = hf_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        auto_map = meta.get("auto_map", {})
        if not isinstance(auto_map, dict):
            continue
        for value in auto_map.values():
            values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
            for item in values:
                if not isinstance(item, str):
                    continue
                module_name = item.split(".")[0].strip()
                if module_name:
                    names.append(f"{module_name}.py")
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def copy_hf_support_files(src_dirs: Iterable[Path], dst_dir: Path) -> List[str]:
    """Copy HF config/tokenizer/remote-code files.

    Args:
        src_dirs (Iterable[Path]): Candidate source HF directories.
        dst_dir (Path): Destination HF directory.
    Returns:
        List[str]: Copied file names.
    """
    copied: List[str] = []
    for src in src_dirs:
        if not src or not src.is_dir():
            continue
        for name in infer_remote_code_filenames(src):
            src_file = src / name
            dst_file = dst_dir / name
            if not src_file.exists() or dst_file.exists():
                continue
            dst_file.write_bytes(src_file.read_bytes())
            copied.append(name)
    return copied


def save_tensor_state_dict(state_dict: Mapping[str, torch.Tensor], output_path: Path) -> None:
    """Save extracted state dict to .bin/.pt.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Extracted weights.
        output_path (Path): Output file path.
    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), output_path)


def save_as_hf_dir(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: Path,
    extract_module: str,
    model_path: Path,
    is_hf_input: bool,
    hf_base: Path | None,
) -> None:
    """Save extracted state dict as HF-style directory.

    Args:
        state_dict (Mapping[str, torch.Tensor]): Extracted weights.
        output_dir (Path): Output HF directory.
        extract_module (str): Canonical module name.
        model_path (Path): Source model path.
        is_hf_input (bool): Whether source path is HF directory.
        hf_base (Path | None): Optional base HF dir for support files.
    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    to_save = dict(state_dict)
    stripped = False
    if extract_module == "llm":
        to_save, stripped = strip_prefix_if_present(to_save, "llm.")
    torch.save(to_save, output_dir / "pytorch_model.bin")

    copied_files: List[str] = []
    if extract_module == "llm":
        src_dirs: List[Path] = []
        if is_hf_input and model_path.is_dir():
            src_dirs.append(model_path)
        if hf_base is not None:
            src_dirs.append(hf_base)
        copied_files = copy_hf_support_files(src_dirs, output_dir)

    meta = {
        "extract_moudle": extract_module,
        "source_model_path": str(model_path),
        "saved_weight_keys": len(to_save),
        "llm_prefix_stripped": stripped,
        "copied_hf_support_files": copied_files,
    }
    (output_dir / "extractor_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Run extraction entrypoint.

    Args:
        None
    Returns:
        None
    """
    args = parse_args()
    model_path = Path(args.model_path)
    extract_module = normalize_extract_module(args.extract_moudle)
    output_name = normalize_output_name(args.output_name, extract_module)
    output_form, output_path = infer_output_target(Path(args.output_dir), output_name)
    hf_base = Path(args.hf_base) if args.hf_base is not None else None

    state_dict, is_hf_input = load_model_state_dict(model_path)
    extracted_state_dict, used_llm_fallback = extract_by_module(state_dict, extract_module)

    ensure_overwrite(output_path, output_form, args.overwrite)

    if output_form in {".bin", ".pt"}:
        save_tensor_state_dict(extracted_state_dict, output_path)
        print(f"[OK] model_path={model_path}")
        print(f"[OK] extract_moudle={extract_module}")
        print(f"[OK] output={output_path}")
        print(f"[INFO] output_form={output_form}")
        print(f"[INFO] extracted_keys={len(extracted_state_dict)}")
        print(f"[INFO] llm_unprefixed_fallback={used_llm_fallback}")
        return

    save_as_hf_dir(
        state_dict=extracted_state_dict,
        output_dir=output_path,
        extract_module=extract_module,
        model_path=model_path,
        is_hf_input=is_hf_input,
        hf_base=hf_base,
    )
    print(f"[OK] model_path={model_path}")
    print(f"[OK] extract_moudle={extract_module}")
    print(f"[OK] output={output_path}")
    print("[INFO] output_form=hf")
    print(f"[INFO] extracted_keys={len(extracted_state_dict)}")
    print(f"[INFO] llm_unprefixed_fallback={used_llm_fallback}")


if __name__ == "__main__":
    main()

"""
CLI 使用範例
1) 抽 seg_encoder 成 .bin（預設 output_dir=inits/extracted_weights）
./.venv/bin/python scripts/weight_extractors/weight_extractor.py \
  --model_path inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin \
  --extract_moudle img_encoder

2) 抽 encoder_llm_porjector 成 .pt
./.venv/bin/python scripts/weight_extractors/weight_extractor.py \
  --model_path runs/s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam2_base_e1_gpu1_align_pretrain_768/pytorch_model.bin \
  --extract_moudle encoder_llm_porjector \
  --output_dir inits/extracted_weights/projectors \
  --output_name encoder_llm_projectors.pt \
  --overwrite

3) 抽 llm 成 HF 目錄（output_name 用 .hf 作為格式標記）
./.venv/bin/python scripts/weight_extractors/weight_extractor.py \
  --model_path inits/extracted_weights/lvlm/xsam_siglip2_hf \
  --extract_moudle llm \
  --output_dir inits/extracted_weights/lvlm \
  --output_name xsam_siglip2_only_llm.hf \
  --hf_base inits/Phi-3-mini-4k-instruct \
  --overwrite
"""