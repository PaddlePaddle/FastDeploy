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
    Check if given bytes data is a GIF file by examining magic number.
    
    Args:
        data: Binary data to check
        
    Returns:
        bool: True if data is a GIF file (GIF87a or GIF89a format)
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class VideoReaderWrapper(decord.VideoReader):
    """
    Wrapper around decord.VideoReader to handle GIF files and fix memory leaks.
    
    This wrapper converts GIF inputs to MP4 format to work around decord's limitations,
    and implements proper cleanup to prevent memory leaks (https://github.com/dmlc/decord/issues/208).
    
    Attributes:
        original_file (str): Path to the original video file (for cleanup)
    """

    def __init__(self, video_path, *args, **kwargs):
        """
        Initialize the video reader wrapper.
        
        Args:
            video_path: Can be one of:
                - str: Path to video file
                - bytes: Raw video bytes
                - io.BytesIO: Video data stream
            *args: Additional arguments for decord.VideoReader
            **kwargs: Additional keyword arguments for decord.VideoReader
            
        Note:
            Automatically converts GIF files to MP4 format for compatibility.
        """
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
                # Convert GIF to MP4 for decord compatibility
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                clip.write_videofile(mp4_file.name, verbose=False, logger=None)
                clip.close()
                video_path = mp4_file.name
                self.original_file = video_path  # Store path for cleanup

            super().__init__(video_path, *args, **kwargs)
            self.seek(0)

    def __getitem__(self, key):
        """
        Get video frames by index/slice and reset reader position.
        
        Args:
            key: Index or slice of frames to retrieve
            
        Returns:
            decord.ndarray.NDArray: Requested video frames
            
        Note:
            Resets read position to start after frame retrieval
        """
        frames = super().__getitem__(key)
        self.seek(0)
        return frames

    def __del__(self):
        """
        Clean up temporary files when object is destroyed.
        
        Note:
            Removes any temporary MP4 files created from GIF conversions
        """
        if self.original_file and os.path.exists(self.original_file):
            os.remove(self.original_file)


def read_video_decord(video_path):
    """
    Read video file using decord video reader and get metadata.
    
    Args:
        video_path: Can be one of:
            - str: Path to video file
            - bytes: Raw video bytes
            - io.BytesIO: Video data stream
            - VideoReaderWrapper: Existing video reader instance
            
    Returns:
        tuple: (video_reader, video_meta) where:
            - video_reader: VideoReaderWrapper instance
            - video_meta: Dictionary containing:
                - fps: Frames per second
                - duration: Video duration in seconds
                - num_of_frame: Total number of frames
    """
    if isinstance(video_path, VideoReaderWrapper):
        video_reader = video_path  # Reuse existing reader if provided
    else:
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_path)  # Convert bytes to BytesIO
        video_reader = VideoReaderWrapper(video_path, num_threads=1)

    # Extract video metadata
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # Package metadata
    video_meta = {
        "fps": fps,            # Frames per second
        "duration": duration,  # Total duration in seconds
        "num_of_frame": vlen   # Total frame count
    }
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
