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

"""Qwen3ImageProcessor — inherits QwenImageProcessor with different defaults."""

from typing import List, Union

from paddleformers.transformers.image_utils import PILImageResampling

from .qwen import QwenImageProcessor

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

MIN_PIXELS = 65536
MAX_PIXELS = 16777216


class Qwen3ImageProcessor(QwenImageProcessor):
    """Image processor for Qwen3-VL. patch_size=16, mean/std=[0.5,0.5,0.5]."""

    def __init__(
        self,
        patch_size: int = 16,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        image_mean: Union[float, List[float]] = IMAGE_MEAN,
        image_std: Union[float, List[float]] = IMAGE_STD,
        rescale_factor: float = 1 / 255,
        do_rescale: bool = True,
        do_normalize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            resample=resample,
            **kwargs,
        )
