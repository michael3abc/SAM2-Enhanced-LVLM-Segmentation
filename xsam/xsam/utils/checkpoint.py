import logging
import os.path as osp

from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint

from xsam.utils.logging import print_log


def _filter_state_dict_by_shape(model_state_dict, checkpoint_state_dict):
    """Filter checkpoint parameters by key existence and tensor shape.

    Args:
        model_state_dict (dict): Current model state dict.
        checkpoint_state_dict (dict): Loaded checkpoint state dict.
    Returns:
        tuple[dict, list, list]: Filtered state dict, unexpected keys, shape-mismatched keys.
    """
    filtered_state_dict = {}
    unexpected_keys = []
    shape_mismatched_keys = []

    for key, value in checkpoint_state_dict.items():
        if key not in model_state_dict:
            unexpected_keys.append(key)
            continue
        if model_state_dict[key].shape != value.shape:
            shape_mismatched_keys.append(key)
            continue
        filtered_state_dict[key] = value

    return filtered_state_dict, unexpected_keys, shape_mismatched_keys


def load_checkpoint(model, pth_model: str) -> None:
    """Load checkpoint with partial shape-safe matching.

    Args:
        model: Target model.
        pth_model (str): Checkpoint path.
    Returns:
        None
    """
    if not osp.exists(pth_model):
        return

    backend = get_file_backend(pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio

        with patch_fileio():
            state_dict = guess_load_checkpoint(pth_model)
    else:
        state_dict = guess_load_checkpoint(pth_model)

    model_state_dict = model.state_dict()
    filtered_state_dict, unexpected_keys, shape_mismatched_keys = _filter_state_dict_by_shape(model_state_dict, state_dict)
    model.load_state_dict(filtered_state_dict, strict=False)
    matched_keys = list(filtered_state_dict.keys())
    missed_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict.keys()]

    print_log(f"Load checkpoint from {pth_model}", logger="current")
    print_log(f"Matched keys: {len(matched_keys)} / {len(state_dict.keys())}", logger="current")
    if len(unexpected_keys) > 0:
        preview = unexpected_keys[:20]
        suffix = " ..." if len(unexpected_keys) > 20 else ""
        print_log(
            f"Unexpected keys ({len(unexpected_keys)}): {preview}{suffix}",
            logger="current",
            level=logging.WARNING,
        )
    if len(shape_mismatched_keys) > 0:
        preview = shape_mismatched_keys[:20]
        suffix = " ..." if len(shape_mismatched_keys) > 20 else ""
        print_log(
            f"Shape-mismatched keys ({len(shape_mismatched_keys)}): {preview}{suffix}",
            logger="current",
            level=logging.WARNING,
        )
    if len(missed_keys) > 0:
        preview = missed_keys[:20]
        suffix = " ..." if len(missed_keys) > 20 else ""
        print_log(f"Missed keys ({len(missed_keys)}): {preview}{suffix}", logger="current", level=logging.WARNING)
