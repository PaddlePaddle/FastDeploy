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

"""Shared video utilities: VideoReaderWrapper, read_video_decord, and sample_frames."""

import io
import math
import os
from tempfile import NamedTemporaryFile as ntf
from typing import Optional, Union

import numpy as np

from fastdeploy.input.image_processors.common import ceil_by_factor, floor_by_factor
from fastdeploy.utils import data_processor_logger

__all__ = [
    "VideoReaderWrapper",
    "read_video_decord",
    "sample_frames",
    "sample_frames_qwen",
    "sample_frames_paddleocr",
]


# ---------------------------------------------------------------------------
# VideoReaderWrapper
# ---------------------------------------------------------------------------


def _is_gif(data: bytes) -> bool:
    """Check if bytes represent a GIF based on magic header."""
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper:
    """decord.VideoReader wrapper that fixes a memory leak and adds GIF support.

    Reference: https://github.com/dmlc/decord/issues/208
    """

    def __init__(self, video_path, *args, **kwargs):
        import decord

        try:
            # moviepy 1.0
            import moviepy.editor as mp
        except Exception:
            # moviepy 2.0
            import moviepy as mp

        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None  # only set when we create a temp file

            if isinstance(video_path, str):
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if _is_gif(video_path):
                    gif_file.write(video_path)
                    gif_file.flush()
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if _is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_file.flush()
                    gif_input = gif_file.name

            if gif_input is not None:
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                mp4_path = mp4_file.name
                mp4_file.close()  # close before moviepy writes
                clip.write_videofile(mp4_path, verbose=False, logger=None)
                clip.close()
                video_path = mp4_path
                self.original_file = video_path  # temp mp4, cleaned up in __del__

            self._reader = decord.VideoReader(video_path, *args, **kwargs)
            self._reader.seek(0)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, key):
        frames = self._reader[key]
        self._reader.seek(0)
        return frames

    def get_avg_fps(self):
        return self._reader.get_avg_fps()

    def seek(self, pos):
        return self._reader.seek(pos)

    def __del__(self):
        original_file = getattr(self, "original_file", None)
        if original_file:
            try:
                os.remove(original_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# read_video_decord
# ---------------------------------------------------------------------------


def read_video_decord(video_path, save_to_disk: bool = False):
    """Load a video file and return (video_reader, video_meta, video_path).

    video_meta contains keys: "fps", "duration", "num_of_frame".
    """
    if isinstance(video_path, VideoReaderWrapper):
        video_reader = video_path
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)
        video_reader = VideoReaderWrapper(video_path, num_threads=1)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    video_meta = {"fps": fps, "duration": duration, "num_of_frame": vlen}
    return video_reader, video_meta, video_path


# ---------------------------------------------------------------------------
# sample_frames — qwen_vl variant
# ---------------------------------------------------------------------------


def sample_frames_qwen(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = -1,
    num_frames: Optional[int] = -1,
) -> np.ndarray:
    """Sample frame indices — qwen_vl variant.

    Sentinel defaults are -1. Applies ceil_by_factor on min_frames and ensures
    num_frames is divisible by 4.
    """
    if fps > 0 and num_frames > 0:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if metadata is None:
        raise ValueError("metadata is required for sample_frames_qwen")

    total_num_frames = metadata["num_of_frame"]

    if num_frames > 0:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps > 0:
        min_frames = ceil_by_factor(min_frames, frame_factor)
        max_frames = floor_by_factor(min(max_frames, total_num_frames), frame_factor)

        num_frames = total_num_frames / metadata["fps"] * fps

        if num_frames > total_num_frames:
            data_processor_logger.warning(f"smart_nframes: nframes[{num_frames}] > total_frames[{total_num_frames}]")

        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)
        num_frames = floor_by_factor(num_frames, frame_factor)

    if num_frames > total_num_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds "
            f"`total_num_frames={total_num_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )

    # num_frames must be divisible by 4
    if num_frames > 2 and num_frames % 4 != 0:
        num_frames = (num_frames // 4) * 4
        total_num_frames = (total_num_frames // 4) * 4
        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)

    if num_frames > 0:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        indices = np.arange(0, total_num_frames).astype(np.int32)

    return indices


# ---------------------------------------------------------------------------
# sample_frames — paddleocr_vl / ernie4_5_vl variant
# ---------------------------------------------------------------------------


def sample_frames_paddleocr(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
) -> np.ndarray:
    """Sample frame indices — paddleocr_vl / ernie4_5_vl variant.

    Sentinel defaults are None. Uses plain math.floor/ceil; no %4 correction.
    """
    fps = fps or 0
    num_frames = num_frames or 0
    if fps > 0 and num_frames > 0:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if metadata is None:
        raise ValueError("metadata is required for sample_frames_paddleocr")

    total_num_frames = metadata["num_of_frame"]

    if num_frames > 0:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps > 0:
        max_frames = math.floor(min(max_frames, total_num_frames) / frame_factor) * frame_factor
        num_frames = total_num_frames / metadata["fps"] * fps
        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)
        num_frames = math.floor(num_frames / frame_factor) * frame_factor

    if num_frames > total_num_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds "
            f"`total_num_frames={total_num_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )

    if num_frames > 0:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        indices = np.arange(0, total_num_frames).astype(np.int32)

    return indices


def sample_frames(
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
    variant: str = "paddleocr",
) -> np.ndarray:
    """Dispatch to sample_frames_qwen or sample_frames_paddleocr based on variant."""
    if variant == "qwen":
        _fps = fps if fps is not None else -1
        _num_frames = num_frames if num_frames is not None else -1
        return sample_frames_qwen(frame_factor, min_frames, max_frames, metadata, _fps, _num_frames)
    if variant == "paddleocr":
        return sample_frames_paddleocr(frame_factor, min_frames, max_frames, metadata, fps, num_frames)
    raise ValueError(f"Unknown variant {variant!r}. Expected 'paddleocr' or 'qwen'.")
