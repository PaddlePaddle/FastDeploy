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

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt

from .base import MediaIO


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    """
    对视频帧进行缩放，将每一帧的大小调整为指定的高度和宽度。

    Args:
        frames (npt.NDArray, shape=(N, H, W, C)): 包含N个帧的三维数组，其中H是高度，W是宽度，C是通道数。
            所有帧都应该具有相同的通道数。
        size (tuple[int, int], required): 一个元组，包含两个整数，分别表示目标高度和宽度。

    Returns:
        npt.NDArray, shape=(N, new_height, new_width, C): 返回一个新的三维数组，其中每一帧已经被缩放到指定的高度和宽度。
        新数组的通道数与输入数组相同。

    Raises:
        None
    """
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty((num_frames, new_height, new_width, channels), dtype=frames.dtype)
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    """
    对视频帧进行缩放，将每个帧的高度和宽度都乘以一个因子。

    Args:
        frames (npt.NDArray): 形状为（T，H，W，C）的四维numpy数组，表示T个帧，高度为H，宽度为W，通道数为C。
        size_factor (float): 用于缩放视频帧的因子，新的高度和宽度将分别是原来的高度和宽度的size_factor倍。

    Returns:
        npt.NDArray: 形状为（T，new_H，new_W，C）的四维numpy数组，表示T个帧，高度为new_H，宽度为new_W，通道数为C。
        其中new_H和new_W是根据size_factor计算出来的。

    Raises:
        None
    """
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    """
    从视频中随机选取指定数量的帧，并返回一个包含这些帧的numpy数组。

    Args:
        frames (npt.NDArray): 形状为（T，H，W，C）的ndarray，表示视频的所有帧，其中T是帧的总数，H、W是每个帧的高度和宽度，C是通道数。
        num_frames (int, optional): 要从视频中选取的帧数。如果设置为-1，则将返回所有帧。默认为-1。

    Returns:
        npt.NDArray: 形状为（num_frames，H，W，C）的ndarray，表示选取的帧。如果num_frames=-1，则返回原始的frames。
    """
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoMediaIO(MediaIO[bytes]):
    def __init__(self) -> None:
        """
            初始化一个 VideoMediaIO 对象。

        Args:
            无。

        Raises:
            无。

        Returns:
            无。
        """
        super().__init__()

    def load_bytes(self, data: bytes) -> bytes:
        """
            ERNIE-45-VL模型的前处理中包含抽帧操作，如果将视频帧加载为npt.NDArray格式会丢失FPS信息，因此目前
        不对字节数据做任何操作。

        Args:
            data (bytes): 包含视频帧数据的字节对象。

        Returns:
            bytes，字节数据原样返回。

        Raises:
            无。
        """
        return data

    def load_base64(self, media_type: str, data: str) -> bytes:
        """
        加载 base64 编码的数据，并返回bytes。

        Args:
            media_type (str): 媒体类型，目前不支持 "video/jpeg"。
            data (str): base64 编码的字符串数据。

        Returns:
            bytes, optional: 如果 media_type 不为 "video/jpeg"，则返回字节数据。

        Raises:
            ValueError: 如果media_type是"video/jpeg"。
        """
        if media_type.lower() == "video/jpeg":
            raise ValueError("Video in JPEG format is not supported")

        return base64.b64decode(data)

    def load_file(self, filepath: str) -> bytes:
        """
            读取文件内容，并返回bytes。

        Args:
            filepath (str): 文件路径，表示要读取的文件。

        Returns:
            bytes, optional: 返回字节数据，包含了文件内容。

        Raises:
            无。
        """
        with open(filepath, "rb") as f:
            data = f.read()

        return data
