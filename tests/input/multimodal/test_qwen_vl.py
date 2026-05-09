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

"""Unit tests for QwenVLProcessor."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.mm_processor import MMContext, MMItem, TokenizationPath
from fastdeploy.input.multimodal.qwen_vl import QwenVLProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG


def _make_qwen_processor(**overrides):
    """Create a QwenVLProcessor with mocked dependencies."""
    with patch.object(QwenVLProcessor, "__init__", return_value=None):
        proc = QwenVLProcessor.__new__(QwenVLProcessor)

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

    proc.video_min_pixels = None
    proc.video_max_pixels = None

    proc.limit_mm_per_prompt = {"image": 10, "video": 10, "audio": 1}

    # Mock image processor
    mock_ip = MagicMock()
    mock_ip.merge_size = 2
    mock_ip.temporal_patch_size = 2
    proc.image_processor = mock_ip

    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


def _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4):
    """Create a mock image_processor.preprocess return value."""
    pixel_values = np.ones((num_pixels, 3), dtype=np.float32)
    grid_thw = np.array([t, h, w])
    return {"pixel_values": pixel_values, "grid_thw": grid_thw}


# ==================================================================
# Test classes
# ==================================================================


class TestQwenMakeOutputs(unittest.TestCase):
    def test_has_fps_field(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        self.assertIn("fps", outputs)
        self.assertEqual(outputs["fps"], [])
        # Also has base fields
        self.assertIn("input_ids", outputs)
        self.assertIn("mm_hashes", outputs)


class TestQwenComputePositions(unittest.TestCase):
    def setUp(self):
        self.proc = _make_qwen_processor()

    def test_compute_text_positions(self):
        pos = self.proc._compute_text_positions(start_pos=5, num_tokens=3)
        self.assertEqual(pos.shape, (3, 3))
        np.testing.assert_array_equal(pos[0], [5, 6, 7])
        np.testing.assert_array_equal(pos[1], [5, 6, 7])
        np.testing.assert_array_equal(pos[2], [5, 6, 7])

    def test_compute_text_positions_zero(self):
        pos = self.proc._compute_text_positions(start_pos=0, num_tokens=1)
        self.assertEqual(pos.shape, (3, 1))
        np.testing.assert_array_equal(pos, [[0], [0], [0]])

    def test_compute_vision_positions_image(self):
        """Single image (t=1), no temporal offset."""
        pos = self.proc._compute_vision_positions(start_pos=0, t=1, h=2, w=2, second_per_grid_t=0)
        # After spatial_conv_size division: h=1, w=1, total tokens = 1*1*1 = 1
        self.assertEqual(pos.shape, (3, 1))

    def test_compute_vision_positions_video(self):
        """Video (t=2) with temporal offset."""
        pos = self.proc._compute_vision_positions(start_pos=0, t=2, h=4, w=4, second_per_grid_t=1.0)
        # After spatial_conv_size: h=2, w=2, total = 2*2*2 = 8
        self.assertEqual(pos.shape, (3, 8))
        # First frame t_index should be 0, second frame should be 1*tokens_per_second=2
        self.assertEqual(pos[0, 0], 0)
        self.assertEqual(pos[0, 4], 2)  # second frame: 1 * int(1.0) * 2 = 2


class TestQwenPreprocessImage(unittest.TestCase):
    def setUp(self):
        self.proc = _make_qwen_processor()

    def test_raw_image(self):
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4)
        outputs = self.proc._make_outputs()

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        self.proc.preprocess_image(mock_img, outputs, uuid="img_uuid")

        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(outputs["input_ids"], [100])  # 1*2*2//4 = 1 token
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["image"]])
        self.assertEqual(outputs["num_input_image_tokens"], 1)
        self.assertEqual(outputs["fps"], [0])
        self.proc.image_processor.preprocess.assert_called_once()

    def test_cached_image(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)  # 4 pixels, merge_size=2 -> 4//4=1 token
        meta = {"thw": (1, 2, 2)}
        img_cache = (cached_pixels, meta)

        self.proc.preprocess_cached_image(img_cache, outputs, uuid="cached_uuid")

        self.assertEqual(len(outputs["images"]), 1)
        np.testing.assert_array_equal(outputs["images"][0], cached_pixels)
        self.assertEqual(outputs["input_ids"], [100])
        self.assertEqual(outputs["fps"], [0])

    def test_cached_image_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)  # 4//4=1 token
        meta = {"thw": (1, 2, 2)}
        img_cache = (cached_pixels, meta)

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_image(img_cache, outputs, uuid="u", token_len=999)


class TestQwenPreprocessVideo(unittest.TestCase):
    def setUp(self):
        self.proc = _make_qwen_processor()

    def test_raw_video(self):
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=2, h=2, w=2, num_pixels=8)
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 2.0}

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid", meta=meta)

        self.assertEqual(len(outputs["images"]), 1)
        # 2*2*2//4 = 2 tokens
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["video"]] * 2)
        self.assertEqual(outputs["fps"], [2.0])

    def test_cached_video(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((16, 3), dtype=np.float32)  # 16//4=4 tokens
        meta = {"thw": (2, 2, 2), "fps": 4.0}
        frames_cache = (cached_pixels, meta)

        self.proc.preprocess_cached_video(frames_cache, outputs, uuid="vid_uuid")

        self.assertEqual(len(outputs["images"]), 1)
        # 2*2*2//4=2 tokens (t=2, h/merge=1, w/merge=1 -> 2*1*1=2? No, num_tokens=frames.shape[0]//merge^2=16//4=4)
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertEqual(outputs["fps"], [4.0])

    def test_cached_video_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((16, 3), dtype=np.float32)
        meta = {"thw": (2, 2, 2), "fps": 4.0}
        frames_cache = (cached_pixels, meta)

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_video(frames_cache, outputs, uuid="u", token_len=999)


class TestQwenMmNumTokens(unittest.TestCase):
    def test_single_grid(self):
        result = QwenVLProcessor.mm_num_tokens([2, 4, 4])
        self.assertEqual(result, 2 * 4 * 4 // 4)

    def test_list_of_grids(self):
        result = QwenVLProcessor.mm_num_tokens([[1, 2, 2], [2, 4, 4]])
        self.assertEqual(result, [1, 8])

    def test_empty(self):
        result = QwenVLProcessor.mm_num_tokens([])
        self.assertEqual(result, 0)


class TestQwenPackPositionIds(unittest.TestCase):
    def test_pack_position_ids(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        # Simulate two text segments with 3xN position arrays
        outputs["position_ids"] = [
            np.array([[0, 1], [0, 1], [0, 1]]),
            np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4]]),
        ]
        proc.pack_position_ids(outputs)

        # Concatenated: 3x5, then transposed: 5x3
        self.assertEqual(outputs["position_ids"].shape, (5, 3))
        self.assertEqual(outputs["position_ids"].dtype, np.int64)
        self.assertEqual(outputs["image_patch_id"], 100)
        self.assertEqual(outputs["video_patch_id"], 101)


class TestQwenAppendCompletionTokens(unittest.TestCase):
    def test_appends_tokens_and_positions(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        outputs["position_ids"] = []

        proc.append_completion_tokens(outputs, [50, 60, 70])

        self.assertEqual(outputs["input_ids"], [50, 60, 70])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)
        self.assertEqual(len(outputs["position_ids"]), 1)
        self.assertEqual(outputs["position_ids"][0].shape, (3, 3))


class TestQwenAddTextPositions(unittest.TestCase):
    def test_add_text_positions(self):
        proc = _make_qwen_processor()
        outputs = proc._make_outputs()
        outputs["position_ids"] = []

        proc.add_text_positions(outputs, 3)

        self.assertEqual(len(outputs["position_ids"]), 1)
        self.assertEqual(outputs["position_ids"][0].shape, (3, 3))
        self.assertEqual(outputs["cur_position"], 3)


class TestQwenPromptTokenIds2Outputs(unittest.TestCase):
    def setUp(self):
        self.proc = _make_qwen_processor()

    def test_text_only(self):
        """No multimodal items -> all text."""
        ctx = MMContext(
            images=[],
            videos=[],
            mm_order=[],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[1, 2, 3],
        )
        outputs = self.proc.prompt_token_ids2outputs(ctx)
        self.assertEqual(outputs["input_ids"], [1, 2, 3])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)

    def test_with_image(self):
        """Image token run triggers preprocess_image."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4)

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        img_item = MMItem(type="image", data=mock_img, uuid="u1")

        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            # token_id 100 is image_token_id, 1 token of image
            prompt_token_ids=[1, 2, 100, 3],
        )
        outputs = self.proc.prompt_token_ids2outputs(ctx)

        # Text tokens (1,2) + image token(s) + text token (3)
        self.assertIn(IDS_TYPE_FLAG["image"], outputs["token_type_ids"])
        self.assertEqual(len(outputs["images"]), 1)

    def test_mm_count_mismatch(self):
        """More placeholder tokens than mm_items raises ValueError."""
        img_item = MMItem(type="image", data=MagicMock(), uuid="u1")
        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            # Two separate runs of image token -> expects 2 items but only 1 available
            prompt_token_ids=[100, 5, 100],
        )
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4)

        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)


class TestQwenLoadVideo(unittest.TestCase):
    @patch("fastdeploy.input.multimodal.qwen_vl._sample_qwen")
    @patch("fastdeploy.input.multimodal.qwen_vl.read_video_decord")
    def test_basic_load(self, mock_read, mock_sample):
        proc = _make_qwen_processor()

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value=mock_frame)
        mock_read.return_value = (mock_reader, {"num_of_frame": 4, "fps": 30.0, "duration": 2.0}, None)
        mock_sample.return_value = [0, 1]

        frames, meta = proc.load_video("http://video.mp4", {})

        self.assertEqual(frames.shape[0], 2)
        mock_read.assert_called_once()


if __name__ == "__main__":
    unittest.main()
