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

"""Utility package for fastdeploy.input — re-exports from sub-modules."""

from fastdeploy.input.utils.common import (
    IDS_TYPE_FLAG,
    MAX_IMAGE_DIMENSION,
    process_stop_token_ids,
    validate_model_path,
)
from fastdeploy.input.utils.video import (
    VideoReaderWrapper,
    read_video_decord,
    sample_frames,
    sample_frames_paddleocr,
    sample_frames_qwen,
)

__all__ = [
    "IDS_TYPE_FLAG",
    "MAX_IMAGE_DIMENSION",
    "process_stop_token_ids",
    "validate_model_path",
    "VideoReaderWrapper",
    "read_video_decord",
    "sample_frames",
    "sample_frames_paddleocr",
    "sample_frames_qwen",
]
