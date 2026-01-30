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


def sample_frames(
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
    if fps > 0 and num_frames > 0:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    total_num_frames = metadata["num_of_frame"]

    # If num_frames is not given but fps is, calculate num_frames from fps
    if num_frames > 0:
        num_frames = round(num_frames / frame_factor) * frame_factor
    elif fps > 0:
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
    if num_frames > 0:
        # Evenly spaced sampling for target frame count
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(np.int32)
    else:
        # Keep all frames if no sampling requested
        indices = np.arange(0, total_num_frames).astype(np.int32)

    return indices
