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
from typing import Optional, Union

import numpy as np
from PIL import Image

from fastdeploy.input.mm_processor import read_video_decord


def read_frames(video_path):
    """
    Read and decode video frames from the given path

    This function reads a video file and decodes it into individual RGB frames
    using decord video reader. It also extracts video metadata including fps,
    duration and frame count.

    Args:
        video_path (str): Path to the video file or bytes object containing video data

    Returns:
        tuple: A tuple containing:
            frames (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                containing decoded RGB video frames
            meta (dict): Dictionary containing video metadata:
                - fps (float): Frames per second
                - duration (float): Video duration in seconds
                - num_of_frame (int): Total number of frames
                - width (int): Frame width in pixels
                - height (int): Frame height in pixels

    Note:
        - The function uses decord library for efficient video reading
        - All frames are converted to RGB format regardless of input format
    """
    reader, meta, _ = read_video_decord(video_path, save_to_disk=False)

    frames = []
    for i in range(meta["num_of_frame"]):
        frame = reader[i].asnumpy()
        image = Image.fromarray(frame, "RGB")
        frames.append(image)
    frames = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
    return frames, meta


def sample_frames(
    video: np.ndarray,
    frame_factor: int,
    min_frames: int,
    max_frames: int,
    metadata: Optional[dict] = None,
    fps: Optional[Union[int, float]] = None,
    num_frames: Optional[int] = None,
):
    """
    Sample frames from video according to specified criteria.

    Args:
        video: Input video frames as numpy array
        frame_factor: Ensure sampled frames are multiples of this factor
        min_frames: Minimum number of frames to sample
        max_frames: Maximum number of frames to sample
        metadata: Video metadata containing fps information
        fps: Target frames per second for sampling
        num_frames: Exact number of frames to sample

    Returns:
        np.ndarray: Sampled video frames

    Raises:
        ValueError: If both fps and num_frames are specified,
                   or if required metadata is missing,
                   or if requested frames exceed available frames
    """
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

    # Calculate frame indices based on sampling strategy
    if num_frames is not None:
        # Evenly spaced sampling for target frame count
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        # Keep all frames if no sampling requested
        indices = np.arange(0, total_num_frames).astype(np.int32)

    # Apply frame selection
    video = video[indices]

    return video
