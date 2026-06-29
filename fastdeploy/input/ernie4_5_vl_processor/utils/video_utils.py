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
from tempfile import NamedTemporaryFile as ntf

import numpy as np
import paddle

from fastdeploy.utils import get_logger

try:
    # moviepy 1.0
    import moviepy.editor as mp
except:
    # moviepy 2.0
    import moviepy as mp

logger = get_logger("video_utils")


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class _NumpyFrame:
    """Wrapper so that frame[idx].asnumpy() keeps working with paddlecodec."""

    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


class VideoReaderWrapper:
    """paddlecodec VideoDecoder wrapper with GIF support."""

    def __init__(self, video_path, *args, **kwargs):
        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None
            if isinstance(video_path, str):
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

            with paddle.use_compat_guard(enable=True, scope={"torchcodec"}):
                try:
                    import sys

                    from torchcodec.decoders import VideoDecoder

                    sys.modules["torchcodec"] = None
                except (ImportError, RuntimeError) as e:
                    logger.error(
                        f"Failed to load 'torchcodec' backend via Paddle proxy.\n"
                        f"  - Common Causes:\n"
                        f"    1. Conflict with official 'torch' or 'torchcodec' packages.\n"
                        f"    2. Missing FFmpeg libraries or System library mismatch (CXXABI).\n"
                        f"  - Recommended Fix Steps:\n"
                        f"    1. Install dependencies: `conda install ffmpeg -c conda-forge` or `apt-get update && apt-get install ffmpeg` \n"
                        f"    2. Uninstall conflicts: `pip uninstall torchcodec paddlecodec -y`\n"
                        f"    3. Reinstall packages: `pip install paddlecodec --force-reinstall`\n"
                        f"  - If you encounter 'CXXABI' or 'libstdc++' errors, your system libraries might be outdated.\n"
                        f"    Try prioritizing Conda libraries by running: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python your_script.py`\n"
                        f"  - Original Error: {e}"
                    )
                    raise
                PADDLECODEC_NUM_THREADS = int(os.environ.get("PADDLECODEC_NUM_THREADS", 0))
                self._decoder = VideoDecoder(
                    video_path,
                    seek_mode="exact",
                    num_ffmpeg_threads=PADDLECODEC_NUM_THREADS,
                    device=kwargs.get("device", "cpu"),
                    dimension_order="NHWC",
                )

    def __len__(self):
        return self._decoder.metadata.num_frames

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            frame = self._decoder.get_frames_at(indices=[int(key)]).data[0]
            return _NumpyFrame(frame.numpy())
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
        else:
            indices = list(key) if not isinstance(key, list) else key
        frames = self._decoder.get_frames_at(indices=indices).data
        return _NumpyFrame(frames.numpy())

    def get_avg_fps(self):
        return self._decoder.metadata.average_fps

    def __del__(self):
        original_file = getattr(self, "original_file", None)
        if original_file and os.path.exists(original_file):
            try:
                os.remove(original_file)
            except OSError:
                pass
