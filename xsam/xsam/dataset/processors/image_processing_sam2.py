# coding=utf-8
"""Image processor class for SAM2 with SAM-compatible outputs."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, filter_out_non_signature_kwargs, logging


logger = logging.get_logger(__name__)


class Sam2ImageProcessor(BaseImageProcessor):
    """Standalone SAM2 image processor with SAM-style output fields."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # type: ignore
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
        """Initialize the SAM2 image processor.

        Args:
            do_resize: Whether to resize inputs.
            size: Output image size config.
            mask_size: Output mask size config.
            resample: Interpolation method for resizing.
            do_rescale: Whether to rescale image values.
            rescale_factor: Multiplicative factor for rescaling.
            do_normalize: Whether to normalize image values.
            image_mean: Mean for normalization.
            image_std: Std for normalization.
            do_pad: Whether to pad image and masks.
            pad_size: Target pad size for images.
            mask_pad_size: Target pad size for masks.
            do_convert_rgb: Whether to convert input image to RGB.
            ignore_index: Fill value used when padding masks.
            **kwargs: Additional keyword arguments.

        Returns:
            None.
        """
        super().__init__(**kwargs)

        size = size if size is not None else {"height": 1024, "width": 1024}
        if not isinstance(size, dict):
            size = get_size_dict(size=size, default_to_square=False)
        size = self._canonicalize_sam2_size(size)

        mask_size = mask_size if mask_size is not None else {"height": 256, "width": 256}
        if not isinstance(mask_size, dict):
            mask_size = get_size_dict(size=mask_size, default_to_square=False, param_name="mask_size")
        mask_size = self._canonicalize_sam2_size(mask_size)

        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        pad_size = get_size_dict(size=pad_size, default_to_square=True, param_name="pad_size")

        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 256, "width": 256}
        mask_pad_size = get_size_dict(size=mask_pad_size, default_to_square=True, param_name="mask_pad_size")

        self.do_resize = do_resize
        self.size = size
        self.mask_size = mask_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.mask_pad_size = mask_pad_size
        self.do_convert_rgb = do_convert_rgb
        self.ignore_index = ignore_index

    def _canonicalize_sam2_size(self, size: Dict[str, int]) -> Dict[str, int]:
        """Canonicalize legacy size formats to SAM2 fixed-size format.

        Args:
            size: Input size dictionary.

        Returns:
            Canonicalized size dictionary.
        """
        if "height" in size and "width" in size:
            return {"height": int(size["height"]), "width": int(size["width"])}
        if "longest_edge" in size:
            edge = int(size["longest_edge"])
            return {"height": edge, "width": edge}
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return {"height": edge, "width": edge}
        if "target_size" in size and isinstance(size["target_size"], int):
            edge = int(size["target_size"])
            return {"height": edge, "width": edge}
        return size

    def _ensure_size_dict(self, size: Optional[Dict[str, int]], *, param_name: str) -> Dict[str, int]:
        """Convert size-like input to a validated dictionary.

        Args:
            size: Size configuration.
            param_name: Name used in validation errors.

        Returns:
            A validated size dictionary.
        """
        if size is None:
            return {}
        if isinstance(size, dict):
            return self._canonicalize_sam2_size(size)
        normalized = get_size_dict(size=size, default_to_square=False, param_name=param_name)
        return self._canonicalize_sam2_size(normalized)

    def _get_longest_edge_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int) -> Tuple[int, int]:
        """Compute output shape from longest-edge resizing.

        Args:
            old_shape: Input `(height, width)`.
            longest_edge: Target longest edge.

        Returns:
            Output `(height, width)`.
        """
        old_h, old_w = old_shape
        scale = longest_edge * 1.0 / max(old_h, old_w)
        new_h, new_w = int(old_h * scale + 0.5), int(old_w * scale + 0.5)
        return new_h, new_w

    def _get_shortest_edge_preprocess_shape(self, old_shape: Tuple[int, int], shortest_edge: int) -> Tuple[int, int]:
        """Compute output shape from shortest-edge resizing.

        Args:
            old_shape: Input `(height, width)`.
            shortest_edge: Target shortest edge.

        Returns:
            Output `(height, width)`.
        """
        old_h, old_w = old_shape
        scale = shortest_edge * 1.0 / min(old_h, old_w)
        new_h, new_w = int(old_h * scale + 0.5), int(old_w * scale + 0.5)
        return new_h, new_w

    def _get_random_scale_preprocess_shape(
        self,
        old_shape: Tuple[int, int],
        min_scale: float,
        max_scale: float,
        target_size: Union[int, Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Compute output shape using random-scale augmentation.

        Args:
            old_shape: Input `(height, width)`.
            min_scale: Minimum random scale.
            max_scale: Maximum random scale.
            target_size: Base target size.

        Returns:
            Output `(height, width)`.
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        old_h, old_w = old_shape
        scale = np.random.uniform(min_scale, max_scale)
        scaled_target = (int(target_size[0] * scale + 0.5), int(target_size[1] * scale + 0.5))
        target_scale = min(scaled_target[0] / old_h, scaled_target[1] / old_w)
        new_h, new_w = int(old_h * target_scale + 0.5), int(old_w * target_scale + 0.5)
        return new_h, new_w

    def _get_output_size(
        self,
        image: ImageInput,
        size: Dict[str, int],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict[str, int]:
        """Resolve output size for one input image.

        Args:
            image: Input image.
            size: Requested size dictionary.
            input_data_format: Optional input channel format.

        Returns:
            A dictionary with `height` and `width`.
        """
        image_np = to_numpy_array(image)
        input_size = get_image_size(image_np, channel_dim=input_data_format)

        if "height" in size and "width" in size:
            out_h, out_w = int(size["height"]), int(size["width"])
        elif "longest_edge" in size:
            out_h, out_w = self._get_longest_edge_preprocess_shape(input_size, int(size["longest_edge"]))
        elif "shortest_edge" in size:
            out_h, out_w = self._get_shortest_edge_preprocess_shape(input_size, int(size["shortest_edge"]))
        elif "target_size" in size:
            out_h, out_w = self._get_random_scale_preprocess_shape(
                input_size,
                float(size["min_scale"]),
                float(size["max_scale"]),
                size["target_size"],
            )
        else:
            raise ValueError(
                "The `size` dictionary must contain `height`/`width`, `longest_edge`, "
                f"`shortest_edge`, or `target_size`. Got {size.keys()}"
            )

        return {"height": out_h, "width": out_w}

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Pad image to target shape on bottom and right.

        Args:
            image: Input image array.
            pad_size: Target size dictionary.
            data_format: Output channel format.
            input_data_format: Input channel format.
            **kwargs: Extra arguments for `pad`.

        Returns:
            Padded image.
        """
        output_h, output_w = int(pad_size["height"]), int(pad_size["width"])
        input_h, input_w = get_image_size(image, channel_dim=input_data_format)
        pad_w = max(output_w - input_w, 0)
        pad_h = max(output_h - input_h, 0)

        return pad(
            image,
            ((0, pad_h), (0, pad_w)),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # type: ignore
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize image with SAM2-compatible size handling.

        Args:
            image: Input image array.
            size: Size configuration.
            resample: Resampling method.
            data_format: Output channel format.
            input_data_format: Input channel format.
            **kwargs: Extra args for resize.

        Returns:
            Resized image.
        """
        size = self._ensure_size_dict(size, param_name="size")
        out_size = self._get_output_size(image=image, size=size, input_data_format=input_data_format)
        return resize(
            image,
            size=(out_size["height"], out_size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess(
        self,
        image: np.ndarray,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        rescale_factor: float,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        do_pad: bool,
        pad_size: Dict[str, int],
        constant_values: Union[float, Iterable[float]],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Apply core transforms to image-like tensor.

        Args:
            image: Input image array.
            do_resize: Whether to resize.
            do_rescale: Whether to rescale.
            do_normalize: Whether to normalize.
            size: Resize config.
            resample: Resize interpolation.
            rescale_factor: Rescale factor.
            image_mean: Normalization mean.
            image_std: Normalization std.
            do_pad: Whether to pad.
            pad_size: Target pad size.
            constant_values: Pad fill value.
            input_data_format: Input channel format.

        Returns:
            Tuple of `(processed_image, reshaped_input_size)`.
        """
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        reshaped_input_size = get_image_size(image, channel_dim=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        if do_pad:
            image = self.pad_image(
                image=image,
                pad_size=pad_size,
                input_data_format=input_data_format,
                constant_values=constant_values,
            )

        return image, reshaped_input_size

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        do_pad: bool,
        pad_size: Dict[str, int],
        do_convert_rgb: bool,
        data_format: Optional[Union[str, ChannelDimension]],
        input_data_format: Optional[Union[str, ChannelDimension]],
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Preprocess one image and return metadata.

        Args:
            image: Input image.
            do_resize: Whether to resize.
            size: Resize config.
            resample: Resize interpolation.
            do_rescale: Whether to rescale.
            rescale_factor: Rescale factor.
            do_normalize: Whether to normalize.
            image_mean: Normalization mean.
            image_std: Normalization std.
            do_pad: Whether to pad.
            pad_size: Pad target size.
            do_convert_rgb: Whether to convert to RGB.
            data_format: Desired output channel format.
            input_data_format: Input channel format.

        Returns:
            Tuple of `(processed_image, original_size, reshaped_input_size)`.
        """
        image = to_numpy_array(image)
        if do_convert_rgb:
            image = convert_to_rgb(image)

        image = to_numpy_array(image)

        if do_rescale and is_scaled_image(image):
            logger.warning_once(
                "Input appears to be already rescaled to [0, 1]. Set `do_rescale=False` to avoid double scaling."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        original_size = get_image_size(image, channel_dim=input_data_format)
        image, reshaped_input_size = self._preprocess(
            image=image,
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            size=size,
            resample=resample,
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            constant_values=0.0,
            input_data_format=input_data_format,
        )

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image, original_size, reshaped_input_size

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool,
        mask_size: Dict[str, int],
        do_pad: bool,
        mask_pad_size: Dict[str, int],
        input_data_format: Optional[Union[str, ChannelDimension]],
        ignore_index: int,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess one segmentation/condition mask.

        Args:
            segmentation_map: Input mask.
            do_resize: Whether to resize.
            mask_size: Resize config for masks.
            do_pad: Whether to pad.
            mask_pad_size: Pad target size for masks.
            input_data_format: Input channel format.
            ignore_index: Fill value for padded regions.

        Returns:
            Tuple of `(processed_mask, original_size)`.
        """
        segmentation_map = to_numpy_array(segmentation_map)

        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        original_size = get_image_size(segmentation_map, channel_dim=input_data_format)

        segmentation_map, _ = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            do_rescale=False,
            do_normalize=False,
            size=mask_size,
            resample=PILImageResampling.NEAREST,
            rescale_factor=1.0,
            image_mean=0.0,
            image_std=1.0,
            do_pad=do_pad,
            pad_size=mask_pad_size,
            constant_values=ignore_index,
            input_data_format=input_data_format,
        )

        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)

        return segmentation_map.astype(np.int64), original_size

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        condition_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,  # type: ignore
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        ignore_index: Optional[int] = None,
    ) -> BatchFeature:
        """Preprocess images and optional masks with SAM2 defaults.

        Args:
            images: Input image or batch of images.
            segmentation_maps: Optional segmentation maps.
            condition_maps: Optional visual-prompt masks.
            do_resize: Override resize flag.
            size: Override image size.
            mask_size: Override mask size.
            resample: Override resample mode.
            do_rescale: Override rescale flag.
            rescale_factor: Override rescale factor.
            do_normalize: Override normalize flag.
            image_mean: Override image mean.
            image_std: Override image std.
            do_pad: Override pad flag.
            pad_size: Override image pad size.
            mask_pad_size: Override mask pad size.
            do_convert_rgb: Override RGB conversion flag.
            return_tensors: Output tensor type.
            data_format: Output channel format.
            input_data_format: Input channel format.
            ignore_index: Ignore value for mask padding.

        Returns:
            A `BatchFeature` with SAM-compatible keys.
        """
        do_resize = self.do_resize if do_resize is None else do_resize
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        do_pad = self.do_pad if do_pad is None else do_pad
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb

        size = self.size if size is None else size
        size = self._ensure_size_dict(size, param_name="size")

        mask_size = self.mask_size if mask_size is None else mask_size
        mask_size = self._ensure_size_dict(mask_size, param_name="mask_size")

        pad_size = self.pad_size if pad_size is None else pad_size
        pad_size = get_size_dict(size=pad_size, default_to_square=True, param_name="pad_size")

        mask_pad_size = self.mask_pad_size if mask_pad_size is None else mask_pad_size
        mask_pad_size = get_size_dict(size=mask_pad_size, default_to_square=True, param_name="mask_pad_size")

        resample = self.resample if resample is None else resample
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None and len(segmentation_maps) > 0:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)
            if not valid_images(segmentation_maps):
                raise ValueError(
                    "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
                )

        if condition_maps is not None and len(condition_maps) > 0:
            condition_maps = make_list_of_images(condition_maps, expected_ndims=2)
            if not valid_images(condition_maps):
                raise ValueError(
                    "Invalid condition map type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
                )

        reference_image = images[0]
        output_size = self._get_output_size(reference_image, size=size, input_data_format=input_data_format)
        size = output_size
        mask_size = output_size

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        images, original_sizes, scaled_sizes = zip(
            *(
                self._preprocess_image(
                    image=image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=float(rescale_factor),
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    do_pad=do_pad,
                    pad_size=pad_size,
                    do_convert_rgb=do_convert_rgb,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            )
        )

        data = {
            "pixel_values": images,
            "original_sizes": original_sizes,
            "scaled_sizes": scaled_sizes,
        }

        if segmentation_maps is not None and len(segmentation_maps) > 0:
            segmentation_maps, _ = zip(
                *(
                    self._preprocess_mask(
                        segmentation_map=mask,
                        do_resize=do_resize,
                        mask_size=mask_size,
                        do_pad=do_pad,
                        mask_pad_size=mask_pad_size,
                        ignore_index=ignore_index,
                        input_data_format=input_data_format,
                    )
                    for mask in segmentation_maps
                )
            )
            data["mask_labels"] = segmentation_maps

        if condition_maps is not None and len(condition_maps) > 0:
            condition_maps, original_mask_sizes = zip(
                *(
                    self._preprocess_mask(
                        segmentation_map=mask,
                        do_resize=do_resize,
                        mask_size=mask_size,
                        do_pad=do_pad,
                        mask_pad_size=mask_pad_size,
                        ignore_index=ignore_index,
                        input_data_format=input_data_format,
                    )
                    for mask in condition_maps
                )
            )

            if not all(
                original_im_size == original_mask_size
                for original_im_size, original_mask_size in zip(original_sizes, original_mask_sizes)
            ):
                raise AssertionError("Condition maps should be the same size as input images.")

            data["vprompt_masks"] = condition_maps

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Sam2ImageProcessor"]
