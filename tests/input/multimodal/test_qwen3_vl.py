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

"""Unit tests for Qwen3VLProcessor."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.qwen3_vl import Qwen3VLProcessor


def _make_qwen3_processor(**overrides):
    """Create a Qwen3VLProcessor with mocked dependencies."""
    with patch.object(Qwen3VLProcessor, "__init__", return_value=None):
        proc = Qwen3VLProcessor.__new__(Qwen3VLProcessor)

    proc.tokenizer = MagicMock()
    proc.tokenizer.convert_tokens_to_ids.side_effect = lambda x: {
        "<|image_pad|>": 100,
        "<|video_pad|>": 101,
    }.get(x, 999)
    proc.tokenizer.tokenize.return_value = ["tok"]
    proc.model_name_or_path = "test-model"
    proc.config = SimpleNamespace(vision_config=SimpleNamespace(tokens_per_second=2))
    proc._cache = None
    proc.enable_processor_cache = False

    proc.image_placeholder = "<|image_pad|>"
    proc.video_placeholder = "<|video_pad|>"
    proc.image_token_str = "<|image_pad|>"
    proc.video_token_str = "<|video_pad|>"

    proc.image_token_id = 100
    proc.video_token_id = 101

    proc.spatial_conv_size = 2
    proc.temporal_conv_size = 2
    proc.tokens_per_second = 2
    proc.FRAME_FACTOR = 2

    proc.fps = 2.0
    proc.min_frames = 4
    proc.max_frames = 768
    proc.target_frames = -1

    proc.video_min_pixels = 128 * 28 * 28
    proc.video_max_pixels = 768 * 28 * 28

    proc.limit_mm_per_prompt = {"image": 10, "video": 10, "audio": 1}

    # Mock image processor
    mock_ip = MagicMock()
    mock_ip.merge_size = 2
    mock_ip.temporal_patch_size = 2
    proc.image_processor = mock_ip

    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


class TestQwen3VLProcessorClassAttributes(unittest.TestCase):
    def test_video_pixel_bounds(self):
        """Qwen3VLProcessor has specific video pixel bounds."""
        self.assertEqual(Qwen3VLProcessor.video_min_pixels, 128 * 28 * 28)
        self.assertEqual(Qwen3VLProcessor.video_max_pixels, 768 * 28 * 28)


class TestQwen3VLInitExtra(unittest.TestCase):
    @patch("fastdeploy.input.multimodal.qwen3_vl.Qwen3ImageProcessor")
    def test_init_extra(self, mock_qwen3_ip_cls):
        """_init_extra should use Qwen3ImageProcessor."""
        mock_ip_instance = MagicMock()
        mock_ip_instance.merge_size = 2
        mock_ip_instance.temporal_patch_size = 2
        mock_qwen3_ip_cls.from_pretrained.return_value = mock_ip_instance

        proc = _make_qwen3_processor()
        proc.model_name_or_path = "/fake/model"

        # Call _init_extra
        proc._init_extra({})

        mock_qwen3_ip_cls.from_pretrained.assert_called_once_with("/fake/model")
        self.assertEqual(proc.spatial_conv_size, 2)
        self.assertEqual(proc.temporal_conv_size, 2)
        self.assertEqual(proc.image_token_id, 100)
        self.assertEqual(proc.video_token_id, 101)
        self.assertEqual(proc.tokens_per_second, 2)

    @patch("fastdeploy.input.multimodal.qwen3_vl.Qwen3ImageProcessor")
    def test_init_extra_none_kwargs(self, mock_qwen3_ip_cls):
        """_init_extra with None processor_kwargs should not crash."""
        mock_ip_instance = MagicMock()
        mock_ip_instance.merge_size = 2
        mock_ip_instance.temporal_patch_size = 2
        mock_qwen3_ip_cls.from_pretrained.return_value = mock_ip_instance

        proc = _make_qwen3_processor()
        proc.model_name_or_path = "/fake/model"

        proc._init_extra(None)
        mock_qwen3_ip_cls.from_pretrained.assert_called_once()


class TestQwen3VLPreprocessVideo(unittest.TestCase):
    def setUp(self):
        self.proc = _make_qwen3_processor()

    def test_video_passes_pixel_bounds(self):
        """preprocess_video should pass video pixel bounds to image_processor."""
        self.proc.image_processor.preprocess.return_value = {
            "pixel_values": np.ones((8, 3), dtype=np.float32),
            "grid_thw": np.array([2, 2, 2]),
        }
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 2.0}

        self.proc.preprocess_video(frames, outputs, uuid="vid", meta=meta)

        call_kwargs = self.proc.image_processor.preprocess.call_args[1]
        self.assertEqual(call_kwargs["min_pixels"], 128 * 28 * 28)
        self.assertEqual(call_kwargs["max_pixels"], 768 * 28 * 28)


if __name__ == "__main__":
    unittest.main()
