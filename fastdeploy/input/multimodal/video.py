"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from .base import MediaIO
from .image import ImageMediaIO


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
    resized_frames = np.empty((num_frames, new_height, new_width, channels),
                              dtype=frames.dtype)
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


def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
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


class VideoMediaIO(MediaIO[npt.NDArray]):

    def __init__(
        self,
        image_io: ImageMediaIO,
        *,
        num_frames: int = 32,
    ) -> None:
        """
            初始化一个 VideoMediaIO 对象。
        
        Args:
            image_io (ImageMediaIO): 用于读取和写入图像的 ImageMediaIO 对象。
            num_frames (int, optional): 视频中帧数，默认为 32。
                ImageMediaIO 对象必须支持指定帧数。
        
        Raises:
            TypeError: 如果 image_io 不是 ImageMediaIO 类型。
            ValueError: 如果 num_frames 小于等于 0。
        
        Returns:
            None: 无返回值，直接初始化并设置属性。
        """
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames

    def load_bytes(self, data: bytes) -> npt.NDArray:
        """
            从字节数据加载视频帧，并返回一个 numpy ndarray。
        如果字节数据中的视频帧数量大于指定的 `num_frames`，则将其平均分布到这些帧上；否则，返回所有帧。
        
        Args:
            data (bytes): 包含视频帧数据的字节对象。
        
        Returns:
            npt.NDArray, shape=(num_frames, height, width, channels): 返回一个 numpy ndarray，其中包含了视频帧数据。
            如果 `num_frames` 小于视频帧数量，则返回前 `num_frames` 帧；否则，返回所有帧。
        
        Raises:
            None.
        """
        import decord
        vr = decord.VideoReader(BytesIO(data), num_threads=1)
        total_frame_num = len(vr)

        num_frames = self.num_frames
        if total_frame_num > num_frames:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frame_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = list(range(0, total_frame_num))

        return vr.get_batch(frame_idx).asnumpy()

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        """
        加载 base64 编码的数据，并返回 numpy ndarray。
        
            Args:
                media_type (str): 媒体类型，目前仅支持 "video/jpeg"。
                当为 "video/jpeg" 时，将解析每一帧的 base64 编码数据，并转换成 numpy ndarray。
                data (str): base64 编码的字符串数据。
        
            Returns:
                npt.NDArray, optional: 如果 media_type 为 "video/jpeg"，则返回 numpy ndarray 格式的视频数据；否则返回 None。
        
            Raises:
                None.
        """
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack([
                np.array(load_frame(frame_data))
                for frame_data in data.split(",")
            ])

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> npt.NDArray:
        """
            读取文件内容，并将其转换为numpy数组。
        
        Args:
            filepath (Path): 文件路径对象，表示要读取的文件。
        
        Returns:
            npt.NDArray, optional: 返回一个numpy数组，包含了文件内容。如果无法解析文件内容，则返回None。
        
        Raises:
            无。
        """
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        """
            将视频编码为Base64字符串，每一帧都是一个Base64字符串。
        如果视频格式为"JPEG"，则每一帧都会被转换成JPEG图片并进行编码。
        
        Args:
            media (npt.NDArray): 要编码的视频，形状为（H，W，C）或者（T，H，W，C），其中T为时间步长，H和W分别为高度和宽度，C为通道数。
                当前仅支持JPEG格式。
            video_format (str, optional, default="JPEG"): 视频格式，只支持"JPEG"。 Default to "JPEG".
        
        Raises:
            NotImplementedError: 当前仅支持JPEG格式。
        
        Returns:
            str: Base64字符串，每一帧都是一个Base64字符串，用","连接起来。
        """
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(
                encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
