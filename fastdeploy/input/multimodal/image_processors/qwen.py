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

"""QwenImageProcessor for Qwen2.5-VL model."""

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
from paddleformers.transformers.legacy.tokenizer_utils_base import TensorType
from PIL import Image

from fastdeploy.input.multimodal.common import is_scaled_image, smart_resize
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


class QwenImageProcessor(BaseImageProcessor):
    """Image processor for Qwen2.5-VL. patch_size=14, CLIP mean/std."""

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
        images = make_list_of_images(images)
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            data_processor_logger.warning(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        processed_images = []
        for image in images:
            if height != resized_height or width != resized_width:
                image = image.astype("uint8")
                image = Image.fromarray(image)
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    data_format=input_data_format,
                )

            if do_rescale and do_normalize:
                image_mean = np.array(image_mean, dtype=np.float32) * (1.0 / rescale_factor)
                image_std = np.array(image_std, dtype=np.float32) * (1.0 / rescale_factor)
                do_rescale = False

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

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])

        grid_t, channel = patches.shape[:2]
        grid_t = grid_t // self.temporal_patch_size

        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
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
