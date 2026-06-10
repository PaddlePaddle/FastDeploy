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

"""Shared image utility functions for all VL image processors."""

import math

import numpy as np

from fastdeploy.utils import data_processor_logger

__all__ = [
    "round_by_factor",
    "ceil_by_factor",
    "floor_by_factor",
    "is_scaled_image",
    "smart_resize",
    "smart_resize_qwen",
    "smart_resize_paddleocr",
]


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def is_scaled_image(image: np.ndarray) -> bool:
    """Check if image pixel values are already normalized to [0, 1] range."""
    if image.dtype == np.uint8:
        return False
    return np.min(image) >= 0 and np.max(image) <= 1


def smart_resize_qwen(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
) -> tuple:
    """Smart image resizing for ERNIE / Qwen2.5 / Qwen3 models."""
    if max(height, width) / min(height, width) > max_ratio:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * max_ratio, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * max_ratio, factor)

        data_processor_logger.info(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}, "
            f"resize to {max(new_height, new_width) / min(new_height, new_width)}"
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


def smart_resize_paddleocr(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
) -> tuple:
    """Smart image resizing for PaddleOCR-VL model."""
    if height < factor:
        data_processor_logger.debug(f"smart_resize_paddleocr: height={height} < factor={factor}, reset height=factor")
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        data_processor_logger.debug(f"smart_resize_paddleocr: width={width} < factor={factor}, reset width=factor")
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, " f"got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
    variant: str = "qwen",
) -> tuple:
    """Unified smart_resize dispatcher."""
    if variant == "paddleocr":
        return smart_resize_paddleocr(height, width, factor, min_pixels, max_pixels)
    return smart_resize_qwen(height, width, factor, min_pixels, max_pixels, max_ratio)
