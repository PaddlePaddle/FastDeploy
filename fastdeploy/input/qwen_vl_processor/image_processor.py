"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import math
from typing import List, Optional, Union

import numpy as np
import paddle
import PIL
from paddleformers.transformers.feature_extraction_utils import BatchFeature
from paddleformers.transformers.image_processing_utils import BaseImageProcessor
from paddleformers.transformers.image_transforms import (
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from paddleformers.transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from paddleformers.transformers.tokenizer_utils_base import TensorType
from PIL import Image

from fastdeploy.utils import data_processor_logger

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "paddle.Tensor",
    List["np.ndarray"],
    List["paddle.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarray"]],
    List[List["paddle.Tensor"]],
]


def round_by_factor(number: int, factor: int) -> int:
    """
    Round number to nearest multiple of factor.

    Args:
        number: Input number to round
        factor: Rounding factor

    Returns:
        int: Rounded number
    """
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """
    Round number up to nearest multiple of factor.

    Args:
        number: Input number to round
        factor: Rounding factor

    Returns:
        int: Rounded number
    """
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """
    Round number down to nearest multiple of factor.

    Args:
        number: Input number to round
        factor: Rounding factor

    Returns:
        int: Rounded number
    """
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int, max_ratio: int = 200):
    """
    Smart image resizing that maintains aspect ratio and respects constraints.

    Args:
        height: Original image height
        width: Original image width
        factor: Patch size factor
        min_pixels: Minimum allowed pixels
        max_pixels: Maximum allowed pixels
        max_ratio: Maximum allowed aspect ratio

    Returns:
        tuple: (new_height, new_width)

    Raises:
        ValueError: If calculated dimensions are invalid
    """
    if max(height, width) / min(height, width) > max_ratio:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * max_ratio, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * max_ratio, factor)

        data_processor_logger.info(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)},\
              resize to {max(new_height, new_width) / min(new_height, new_width)}"
        )

        height = new_height
        width = new_width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

    return h_bar, w_bar


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Check if image pixel values are already normalized to [0, 1] range.

    Args:
        image: Input image array

    Returns:
        bool: True if image is already scaled
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


class ImageProcessor(BaseImageProcessor):
    """
    Adaptive image processor for dynamic image resizing and preprocessing.

    This processor handles image resizing, rescaling, normalization and format conversion.
    It dynamically adjusts image dimensions based on original size and specified constraints.
    """

    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        image_mean: Union[float, List[float]] = OPENAI_CLIP_MEAN,
        image_std: Union[float, List[float]] = OPENAI_CLIP_STD,
        rescale_factor: float = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ) -> None:
        """
        Initialize image processor with configuration parameters.

        Args:
            patch_size (int): Spatial patch size for vision encoder
            merge_size (int): Merge size between vision and LLM encoders
            temporal_patch_size (int): Temporal patch size for video processing
            min_pixels (int): Minimum allowed pixels in resized image
            max_pixels (int): Maximum allowed pixels in resized image
            image_mean (float/list): Mean values for normalization per channel
            image_std (float/list): Std values for normalization per channel
            rescale_factor (float): Scaling factor for pixel values (default 1/255)
            do_rescale (bool): Whether to rescale images
            do_normalize (bool): Whether to normalize images
            resample: Resampling method for image resizing
            **kwargs: Additional base class arguments
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

        self.resample = resample

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        min_pixels: int,
        max_pixels: int,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        rescale_factor: float,
        do_rescale: bool,
        do_normalize: bool,
        resample: PILImageResampling,
        data_format: Optional[ChannelDimension],
        input_data_format: Optional[Union[str, ChannelDimension]],
    ):
        """
        Internal method for image preprocessing pipeline.

        Args:
            images: Input image or batch of images
            min_pixels: Minimum allowed pixels in output
            max_pixels: Maximum allowed pixels in output
            image_mean: Normalization mean values
            image_std: Normalization std values
            rescale_factor: Pixel value scaling factor
            do_rescale: Whether to rescale pixel values
            do_normalize: Whether to normalize pixel values
            resample: Resampling method
            data_format: Output channel format
            input_data_format: Input channel format

        Returns:
            tuple: (flatten_patches, grid_dimensions)
                - flatten_patches: Flattened image patches
                - grid_dimensions: Grid dimensions [t, h, w]
        """
        images = make_list_of_images(images)

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            data_processor_logger.warning(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # Get original dimensions and calculate optimal resize dimensions
        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,  # Combine patch and merge factors
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        processed_images = []
        for image in images:
            if height != resized_height or width != resized_width:
                # Convert to uint8 before resizing to avoid double scaling
                image = image.astype("uint8")
                # Convert to PIL Image and resize
                image = Image.fromarray(image)
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    data_format=input_data_format,
                )

            if do_rescale and do_normalize:
                # Adjust mean and std for combined rescale+normalize
                image_mean = np.array(image_mean, dtype=np.float32) * (1.0 / rescale_factor)
                image_std = np.array(image_std, dtype=np.float32) * (1.0 / rescale_factor)
                do_rescale = False  # Skip separate rescale step

            if do_rescale:
                image = image.astype(np.float32)
                image = rescale(image, scale=rescale_factor, data_format=input_data_format)

            if do_normalize:
                image = image.astype(np.float32)
                image = normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    data_format=input_data_format,
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)  # [C, H, W]
            processed_images.append(image)

        # Convert processed images to numpy array
        patches = np.array(processed_images)

        # Pad temporal dimension if needed
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        # Convert to channels-first format if needed
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])  # [N, H, W, C] -> [N, C, H, W]

        grid_t, channel = patches.shape[:2]
        grid_t = grid_t // self.temporal_patch_size

        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        # Reshape into hierarchical patch structure
        patches = patches.reshape(
            [
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            ]
        )
        # Reorder dimensions for better memory access pattern
        # [grid_t, grid_h/merge_size, grid_w/merge_size, merge_size, merge_size, C, temporal_patch_size, psz, psz]
        patches = patches.transpose([0, 3, 6, 4, 7, 2, 1, 5, 8])

        flatten_patches = patches.reshape(
            [
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            ]
        )

        return flatten_patches, np.array([grid_t, grid_h, grid_w])

    def preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        rescale_factor: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.LAST,
    ):
        """
        Main preprocessing method for images/videos.

        Args:
            images: Input image/video data
            min_pixels: Override for minimum pixels
            max_pixels: Override for maximum pixels
            image_mean: Override for normalization mean
            image_std: Override for normalization std
            rescale_factor: Override for rescaling factor
            do_rescale: Override for rescaling flag
            do_normalize: Override for normalization flag
            resample: Override for resampling method
            return_tensors: Desired output tensor format
            data_format: Output channel dimension format
            input_data_format: Input channel dimension format

        Returns:
            BatchFeature: Processed features containing:
                - pixel_values: Preprocessed pixel data
                - grid_thw: Grid dimensions [temporal, height, width]

        Raises:
            ValueError: For invalid image types or dimensions
        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample

        if images is not None and not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, " "paddle.Tensor.")

        pixel_values, grid_thw = self._preprocess(
            images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        data = {"pixel_values": pixel_values, "grid_thw": grid_thw}
        return BatchFeature(data=data, tensor_type=return_tensors)
