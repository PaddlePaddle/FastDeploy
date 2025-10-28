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

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .base import MediaIO

# TODO 多模数据处理
# try:
#     import librosa
# except ImportError:
#     librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

# try:
#     import soundfile
# except ImportError:
#     soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


def resample_audio(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    """
    将音频数据从原始采样率（`orig_sr`）重采样到目标采样率（`target_sr`）。

    Args:
        audio (npt.NDArray[np.floating]): 带有单通道浮点型音频数据的 numpy ndarray，形状为 `(samples,)`。
        orig_sr (float): 音频数据的原始采样率。
        target_sr (float): 需要转换到的目标采样率。

    Returns:
        npt.NDArray[np.floating]: 带有单通道浮点型音频数据的 numpy ndarray，形状为 `(samples,)`，已经被重采样到目标采样率。

    Raises:
        None.
    """
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        """
            加载字节数据，返回音频信号和采样率。
        参数：
            data (bytes) - 字节数据，包含音频文件的内容。
        返回值（tuple）：
            (array, float) - 第一个元素是一个numpy数组，表示音频信号，第二个元素是一个浮点数，表示采样率。
            如果解码失败，则返回 None。
        """
        import librosa

        return librosa.load(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        """
            将 base64 编码的字符串转换为 numpy 数组和尺度。

        Args:
            media_type (str): 媒体类型，例如 'image/jpeg'、'image/png' 等。
            data (str): base64 编码的字符串，表示图像或其他二进制数据。

        Returns:
            tuple[npt.NDArray, float]: 包含以下两个元素：
                - npt.NDArray: 形状为（H，W，C）的 numpy 数组，表示图像或其他二进制数据。
                - float: 图像的尺度，单位为像素。

        Raises:
            ValueError: 当 media_type 不是有效的媒体类型时引发。
        """
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        """
            加载音频文件，返回音频数据和采样率。
        参数：
            filepath (Path): 音频文件路径（Path类型）。
        返回值：
            tuple[npt.NDArray, float]：包含两个元素的元组，第一个是音频数据（npt.NDArray类型），
            第二个是采样率（float类型）。
        """
        import librosa

        return librosa.load(filepath, sr=None)

    def encode_base64(self, media: tuple[npt.NDArray, float]) -> str:
        """
            将音频数据和采样率转换为Base64编码的字符串。
        参数：
            media (tuple[numpy.ndarray, float]): 包含音频数据和采样率的元组，其中音频数据是一个numpy数组，采样率是一个浮点数。
            返回值 (str): Base64编码的字符串，表示音频数据和采样率。
        """
        audio, sr = media

        with BytesIO() as buffer:
            import soundfile

            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode("utf-8")
