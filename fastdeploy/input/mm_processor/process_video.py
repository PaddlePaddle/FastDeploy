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

import io
import os
import decord
from tempfile import NamedTemporaryFile as ntf
from typing import Union, Optional
import numpy as np
import math

try:
    # moviepy 1.0
    import moviepy.editor as mp
except:
    # moviepy 2.0
    import moviepy as mp

from fastdeploy.utils import data_processor_logger


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper(decord.VideoReader):
    """
    Solving memory leak bug

    https://github.com/dmlc/decord/issues/208
    """

    def __init__(self, video_path, *args, **kwargs):
        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None
            if isinstance(video_path, str):
                self.original_file = video_path
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if is_gif(video_path):
                    gif_file.write(video_path)
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_input = gif_file.name

            if gif_input is not None:
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                clip.write_videofile(mp4_file.name, verbose=False, logger=None)
                clip.close()
                video_path = mp4_file.name
                self.original_file = video_path

            super().__init__(video_path, *args, **kwargs)
            self.seek(0)

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames

    def __del__(self):
        if self.original_file and os.path.exists(self.original_file):
            os.remove(self.original_file)


def read_video_decord(video_path):
    """get reader and meta by decord"""
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
    return video_reader, video_meta


def sample_frames(
    video: np.ndarray,
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
):
    if fps is not None and num_frames is not None:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if fps is None and num_frames is None:
        return video

    total_num_frames = video.shape[0]

    # If num_frames is not given but fps is, calculate num_frames from fps
    if num_frames is not None:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps is not None:
        if metadata is None:
            raise ValueError(
                "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
            )
        max_frames = math.floor(min(max_frames, total_num_frames) / frame_factor) * frame_factor
        num_frames = total_num_frames / metadata["fps"] * fps
        num_frames = min(min(max(num_frames, min_frames), max_frames), total_num_frames)
        num_frames = math.floor(num_frames / frame_factor) * frame_factor

    if num_frames > total_num_frames:
        raise ValueError(
            f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
            "Decrease `num_frames` or `fps` for sampling."
        )

    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype("int")
    else:
        indices = np.arange(0, total_num_frames).astype("int")
    video = video[indices]

    return video
