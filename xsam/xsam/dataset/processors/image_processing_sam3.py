# coding=utf-8
"""Image processor class for SAM3 training/evaluation."""

from typing import Dict, List, Optional, Union

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .image_processing_sam2 import Sam2ImageProcessor


class Sam3ImageProcessor(Sam2ImageProcessor):
    """SAM3 image processor with SAM3-friendly defaults."""

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = False,
        pad_size: Optional[Dict[str, int]] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: bool = True,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        """Initialize SAM3 image processor defaults.

        Args:
            do_resize: Whether to resize inputs.
            size: Target image size config.
            mask_size: Target mask size config.
            do_rescale: Whether to rescale image values.
            rescale_factor: Multiplicative factor for rescaling.
            do_normalize: Whether to normalize image values.
            image_mean: Channel mean for normalization.
            image_std: Channel std for normalization.
            do_pad: Whether to pad image and masks.
            pad_size: Target pad size for images.
            mask_pad_size: Target pad size for masks.
            do_convert_rgb: Whether to convert input image to RGB.
            ignore_index: Fill value used when padding masks.
            **kwargs: Additional keyword arguments.
        Returns:
            None.
        """
        size = size if size is not None else {"height": 1008, "width": 1008}
        mask_size = mask_size if mask_size is not None else {"height": 1008, "width": 1008}
        pad_size = pad_size if pad_size is not None else {"height": 1008, "width": 1008}
        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 1008, "width": 1008}

        # SAM3 repo uses normalize(mean=0.5, std=0.5) by default.
        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        # Keep fallback for compatibility when caller explicitly passes IMAGENET stats.
        if image_mean == IMAGENET_DEFAULT_MEAN and image_std == IMAGENET_DEFAULT_STD:
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]

        super().__init__(
            do_resize=do_resize,
            size=size,
            mask_size=mask_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            mask_pad_size=mask_pad_size,
            do_convert_rgb=do_convert_rgb,
            ignore_index=ignore_index,
            **kwargs,
        )
