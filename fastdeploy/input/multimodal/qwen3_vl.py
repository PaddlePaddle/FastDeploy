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

"""Qwen3VLProcessor — multimodal processor for Qwen3-VL."""

from fastdeploy.input.multimodal.image_processors import Qwen3ImageProcessor
from fastdeploy.input.multimodal.qwen_vl import QwenVLProcessor


class Qwen3VLProcessor(QwenVLProcessor):
    """Multimodal processor for Qwen3-VL.

    Inherits QwenVLProcessor with:
    - Qwen3ImageProcessor (patch_size=16, mean/std=[0.5,0.5,0.5])
    - Video pixel bounds for Qwen3-VL
    """

    # Qwen3-VL video pixel bounds
    video_min_pixels = 128 * 28 * 28
    video_max_pixels = 768 * 28 * 28

    def _init_extra(self, processor_kwargs):
        """Initialize Qwen3VL-specific attributes."""
        processor_kwargs = processor_kwargs or {}

        # Use Qwen3ImageProcessor instead of QwenImageProcessor
        self.image_processor = Qwen3ImageProcessor.from_pretrained(self.model_name_or_path)

        # Conv params from image_processor
        self.spatial_conv_size = self.image_processor.merge_size
        self.temporal_conv_size = self.image_processor.temporal_patch_size

        # Special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token_str)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token_str)

        # tokens_per_second from vision_config
        vision_config = getattr(self.config, "vision_config", None)
        self.tokens_per_second = getattr(vision_config, "tokens_per_second", 2)
