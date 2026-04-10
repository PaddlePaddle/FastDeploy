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

import unittest
from unittest.mock import MagicMock

import numpy as np

from fastdeploy.input.encodings import ErnieEncoding, QwenEncoding
from fastdeploy.input.mm_model_config import (
    ERNIE4_5_VL,
    PADDLEOCR_VL,
    QWEN3_VL,
    QWEN_VL,
)
from fastdeploy.input.utils import IDS_TYPE_FLAG


# ===================================================================
# Encoding-level helpers
# ===================================================================
def _make_encoding(model_type, processor_kwargs=None):
    """Instantiate a real encoding class with mocked processor dependencies.

    Returns (encoding, mock_processor) so tests can inspect mock calls.
    """
    from fastdeploy.input.mm_model_config import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_type]

    mock_processor = MagicMock()
    mock_processor.cfg = cfg
    mock_processor.enable_processor_cache = False

    # image_processor mock
    ip = MagicMock()
    ip.merge_size = 2
    ip.temporal_patch_size = 2
    mock_processor.image_processor = ip

    # tokenizer mock — convert_tokens_to_ids returns deterministic ids
    tok = MagicMock()
    _token_map = {
        "<|image_pad|>": 100,
        "<|video_pad|>": 101,
        "<|IMAGE_PLACEHOLDER|>": 102,
        "<|IMAGE_START|>": 200,
        "<|IMAGE_END|>": 201,
        "<|VIDEO_START|>": 202,
        "<|VIDEO_END|>": 203,
    }
    tok.convert_tokens_to_ids.side_effect = lambda s: _token_map.get(s, 999)
    mock_processor.tokenizer = tok
    mock_processor.config = MagicMock()
    mock_processor.config.vision_config = MagicMock()
    mock_processor.config.vision_config.tokens_per_second = 2

    from fastdeploy.input.encodings import EncodingRegistry

    cls = EncodingRegistry.get(model_type)
    enc = cls(mock_processor, processor_kwargs or {})
    return enc, mock_processor


# ===================================================================
# QwenEncoding tests
# ===================================================================
class TestQwenEncoding(unittest.TestCase):
    """Tests for QwenEncoding methods."""

    def _make_enc(self, model_type=QWEN_VL):
        return _make_encoding(model_type)

    def test_make_outputs_has_fps(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        self.assertIn("fps", outputs)
        self.assertEqual(outputs["fps"], [])
        self.assertIn("input_ids", outputs)
        self.assertEqual(outputs["cur_position"], 0)

    def test_compute_text_positions(self):
        enc, _ = self._make_enc()
        pos = enc._compute_text_positions(start_pos=5, num_tokens=3)
        # Should be 3x3 array: [[5,6,7],[5,6,7],[5,6,7]]
        self.assertEqual(pos.shape, (3, 3))
        np.testing.assert_array_equal(pos[0], [5, 6, 7])
        np.testing.assert_array_equal(pos[1], [5, 6, 7])

    def test_compute_text_positions_zero(self):
        enc, _ = self._make_enc()
        pos = enc._compute_text_positions(start_pos=0, num_tokens=1)
        self.assertEqual(pos.shape, (3, 1))
        np.testing.assert_array_equal(pos[:, 0], [0, 0, 0])

    def test_compute_vision_positions_image(self):
        """Single image (t=1, no temporal offset)."""
        enc, _ = self._make_enc()
        # t=1, h=4, w=4, spatial_conv_size=2 → gh=2, gw=2 → 4 tokens
        pos = enc._compute_vision_positions(start_pos=0, t=1, h=4, w=4, second_per_grid_t=0)
        self.assertEqual(pos.shape[0], 3)  # 3 rows
        self.assertEqual(pos.shape[1], 4)  # 4 tokens

    def test_compute_vision_positions_video(self):
        """Video with temporal offset."""
        enc, _ = self._make_enc()
        # t=2, h=4, w=4, spatial_conv_size=2 → gh=2, gw=2 → 2*4=8 tokens
        pos = enc._compute_vision_positions(start_pos=0, t=2, h=4, w=4, second_per_grid_t=1)
        self.assertEqual(pos.shape, (3, 8))

    def test_add_text_positions(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        enc.add_text_positions(outputs, 3)
        self.assertEqual(len(outputs["position_ids"]), 1)  # one 3xN array
        self.assertEqual(outputs["position_ids"][0].shape, (3, 3))
        self.assertEqual(outputs["cur_position"], 3)

    def test_append_completion_tokens(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        enc.append_completion_tokens(outputs, [10, 11, 12])
        self.assertEqual(outputs["input_ids"], [10, 11, 12])
        self.assertEqual(outputs["token_type_ids"], [0, 0, 0])
        self.assertEqual(outputs["cur_position"], 3)
        self.assertEqual(len(outputs["position_ids"]), 1)

    def test_add_image(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        # Simulate preprocess return
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((4, 3, 28, 28)),
            "grid_thw": np.array([1, 4, 4]),
        }
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        outputs = enc._make_outputs()
        enc.add_image(mock_img, outputs, uuid="img_uuid_1")

        # 1*4*4 // 4 = 4 tokens
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertEqual(outputs["num_input_image_tokens"], 4)
        self.assertEqual(outputs["mm_hashes"], ["img_uuid_1"])
        self.assertEqual(outputs["image_type_ids"], [0])
        self.assertEqual(len(outputs["fps"]), 1)
        self.assertEqual(outputs["fps"][0], 0)

    def test_add_processed_image(self):
        enc, _ = self._make_enc()
        # img shape[0] = 16 pixels, merge_size=2 → 16//4 = 4 tokens
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        outputs = enc._make_outputs()
        enc.add_processed_image((img, meta), outputs, uuid="cached_img")

        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertEqual(outputs["mm_hashes"], ["cached_img"])
        np.testing.assert_array_equal(outputs["grid_thw"][0], np.array([[1, 4, 4]]))
        self.assertEqual(outputs["fps"][0], 0)

    def test_add_processed_image_token_mismatch(self):
        enc, _ = self._make_enc()
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError):
            enc.add_processed_image((img, meta), outputs, uuid="x", token_len=999)

    def test_add_video(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((8, 3, 28, 28)),
            "grid_thw": np.array([2, 4, 4]),
        }
        frames = [MagicMock() for _ in range(2)]
        outputs = enc._make_outputs()
        meta = {"fps": 2}
        enc.add_video(frames, outputs, uuid="vid_uuid", meta=meta)

        # 2*4*4 // 4 = 8 tokens
        self.assertEqual(len(outputs["input_ids"]), 8)
        self.assertEqual(outputs["num_input_video_tokens"], 8)
        self.assertEqual(outputs["fps"][0], 2)
        self.assertEqual(outputs["image_type_ids"], [1, 1])

    def test_add_processed_video(self):
        enc, _ = self._make_enc()
        frames = np.zeros((8, 3, 28, 28))  # 8//4=2 tokens
        meta = {"thw": (2, 4, 4), "fps": 4}
        outputs = enc._make_outputs()
        enc.add_processed_video((frames, meta), outputs, uuid="cached_vid")

        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["fps"][0], 4)
        self.assertEqual(outputs["image_type_ids"], [1, 1])

    def test_add_processed_video_token_mismatch(self):
        enc, _ = self._make_enc()
        frames = np.zeros((8, 3, 28, 28))
        meta = {"thw": (2, 4, 4), "fps": 4}
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError):
            enc.add_processed_video((frames, meta), outputs, uuid="x", token_len=999)

    def test_mm_num_tokens_single(self):
        """Single grid: t*h*w//4."""
        result = QwenEncoding.mm_num_tokens([1, 4, 4])
        self.assertEqual(result, 4)  # 1*4*4//4

    def test_mm_num_tokens_list(self):
        """List of grids."""
        result = QwenEncoding.mm_num_tokens([[1, 4, 4], [2, 4, 4]])
        self.assertEqual(result, [4, 8])  # [16//4, 32//4]

    def test_mm_num_tokens_empty(self):
        self.assertEqual(QwenEncoding.mm_num_tokens([]), 0)

    def test_pack_position_ids(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        enc.add_text_positions(outputs, 3)
        enc.pack_position_ids(outputs)
        self.assertEqual(outputs["position_ids"].shape, (3, 3))
        self.assertEqual(outputs["position_ids"].dtype, np.int64)
        self.assertEqual(outputs["image_patch_id"], enc.image_token_id)
        self.assertEqual(outputs["video_patch_id"], enc.video_token_id)

    def test_prompt_token_ids2outputs_text_only(self):
        """prompt_token_ids with no mm_items — text-only path."""
        enc, _ = self._make_enc(QWEN3_VL)
        outputs = enc.prompt_token_ids2outputs([1, 2, 3])
        self.assertEqual(outputs["input_ids"], [1, 2, 3])
        self.assertEqual(len(outputs["token_type_ids"]), 3)
        self.assertEqual(outputs["cur_position"], 3)

    def test_prompt_token_ids2outputs_with_image(self):
        """prompt_token_ids with image placeholder tokens."""
        enc, mock_proc = self._make_enc(QWEN3_VL)
        ip = mock_proc.image_processor
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((4, 3, 28, 28)),
            "grid_thw": np.array([1, 4, 4]),
        }
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        # image_token_id = 100 for qwen
        # [text, img, img, img, img, text]
        mm_items = [{"type": "image", "data": mock_img, "uuid": "img_uuid"}]
        outputs = enc.prompt_token_ids2outputs([1, 100, 100, 100, 100, 2], mm_items)
        # 1 text + 4 image + 1 text = 6
        self.assertEqual(len(outputs["input_ids"]), 6)

    def test_prompt_token_ids2outputs_mm_count_mismatch(self):
        """More placeholders than mm_items raises."""
        enc, mock_proc = self._make_enc(QWEN3_VL)
        with self.assertRaises(ValueError):
            enc.prompt_token_ids2outputs([100, 100], [])


# ===================================================================
# PaddleOCREncoding tests
# ===================================================================
class TestPaddleOCREncoding(unittest.TestCase):
    """Tests for PaddleOCREncoding overrides."""

    def _make_enc(self):
        return _make_encoding(PADDLEOCR_VL)

    def test_make_outputs_has_vit_fields(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        self.assertIn("vit_seqlen", outputs)
        self.assertIn("vit_position_ids", outputs)
        self.assertIn("fps", outputs)  # inherited from QwenEncoding
        self.assertEqual(outputs["vit_seqlen"], [])
        self.assertEqual(outputs["vit_position_ids"], [])

    def test_add_image_appends_vit_fields(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((4, 3, 28, 28)),
            "grid_thw": np.array([1, 4, 4]),
        }
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        outputs = enc._make_outputs()
        enc.add_image(mock_img, outputs, uuid="img1")

        self.assertEqual(len(outputs["vit_seqlen"]), 1)
        # h=4, w=4 → numel=16
        self.assertEqual(outputs["vit_seqlen"][0], 16)
        self.assertEqual(len(outputs["vit_position_ids"]), 1)
        np.testing.assert_array_equal(outputs["vit_position_ids"][0], np.arange(16) % 16)

    def test_add_video_uses_video_token_id(self):
        """PaddleOCR uses video_token_id (not image_token_id) for video."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((8, 3, 28, 28)),
            "grid_thw": np.array([2, 4, 4]),
        }
        frames = [MagicMock() for _ in range(2)]
        outputs = enc._make_outputs()
        enc.add_video(frames, outputs, uuid="vid1", meta={"fps": 2})

        # All tokens should use video_token_id (102 for paddleocr)
        for tid in outputs["input_ids"]:
            self.assertEqual(tid, enc.video_token_id)

    def test_add_video_appends_vit_fields(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((8, 3, 28, 28)),
            "grid_thw": np.array([2, 4, 4]),
        }
        frames = [MagicMock() for _ in range(2)]
        outputs = enc._make_outputs()
        enc.add_video(frames, outputs, uuid="vid1", meta={"fps": 2})

        self.assertEqual(len(outputs["vit_seqlen"]), 1)
        self.assertEqual(outputs["vit_seqlen"][0], 16)  # h=4, w=4

    def test_add_processed_video_uses_video_token_id(self):
        enc, _ = self._make_enc()
        frames = np.zeros((8, 3, 28, 28))
        meta = {"thw": (2, 4, 4), "fps": 4}
        outputs = enc._make_outputs()
        enc.add_processed_video((frames, meta), outputs, uuid="cached_vid")

        for tid in outputs["input_ids"]:
            self.assertEqual(tid, enc.video_token_id)


# ===================================================================
# ErnieEncoding tests
# ===================================================================
class TestErnieEncoding(unittest.TestCase):
    """Tests for ErnieEncoding methods."""

    def _make_enc(self, processor_kwargs=None):
        return _make_encoding(ERNIE4_5_VL, processor_kwargs)

    def test_init_extra_defaults(self):
        enc, _ = self._make_enc()
        self.assertEqual(enc.image_min_pixels, 4 * 28 * 28)
        self.assertEqual(enc.image_max_pixels, 6177 * 28 * 28)
        self.assertEqual(enc.video_min_pixels, 299 * 28 * 28)
        self.assertEqual(enc.video_max_pixels, 1196 * 28 * 28)
        self.assertEqual(enc.frames_sample, "leading")

    def test_init_extra_custom(self):
        enc, _ = self._make_enc({"image_min_pixels": 100, "video_fps": 5})
        self.assertEqual(enc.image_min_pixels, 100)
        self.assertEqual(enc.fps, 5)

    def test_make_outputs(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        self.assertIn("input_ids", outputs)
        self.assertIn("position_ids", outputs)
        self.assertNotIn("fps", outputs)  # Ernie doesn't have fps field
        self.assertNotIn("vit_seqlen", outputs)

    def test_build_token_type_mapping(self):
        enc, _ = self._make_enc()
        mapping = enc.token_type_mapping
        self.assertEqual(mapping["<|IMAGE_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|IMAGE_END|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|VIDEO_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping["<|VIDEO_END|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping[enc.image_token_id], IDS_TYPE_FLAG["image"])
        # Default for unknown keys
        self.assertEqual(mapping["unknown"], IDS_TYPE_FLAG["text"])

    def test_compute_3d_positions_single_image(self):
        """t=1, h=4, w=4 with spatial_conv=2 → gh=2, gw=2 → 4 positions."""
        enc, _ = self._make_enc()
        pos = enc._compute_3d_positions(t=1, h=4, w=4, start_idx=0)
        self.assertEqual(len(pos), 4)
        # For t=1: t_eff=1, so all time indices are 0
        for p in pos:
            self.assertEqual(len(p), 3)
            self.assertEqual(p[0], 0)  # time dim

    def test_compute_3d_positions_video(self):
        """t=4, h=4, w=4 with temporal_conv=2, spatial_conv=2.
        t_eff=4//2=2, gh=2, gw=2 → 2*4=8 positions."""
        enc, _ = self._make_enc()
        pos = enc._compute_3d_positions(t=4, h=4, w=4, start_idx=10)
        self.assertEqual(len(pos), 8)
        # First 4 have time_idx=0, next 4 have time_idx=1
        for p in pos[:4]:
            self.assertEqual(p[0], 10)  # start_idx + 0
        for p in pos[4:]:
            self.assertEqual(p[0], 11)  # start_idx + 1

    def test_add_text_positions(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        enc.add_text_positions(outputs, 3)
        self.assertEqual(len(outputs["position_ids"]), 3)
        self.assertEqual(outputs["position_ids"][0], [0, 0, 0])
        self.assertEqual(outputs["position_ids"][1], [1, 1, 1])
        self.assertEqual(outputs["position_ids"][2], [2, 2, 2])
        self.assertEqual(outputs["cur_position"], 3)

    def test_append_completion_tokens(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        outputs["cur_position"] = 5
        enc.append_completion_tokens(outputs, [10, 11])
        self.assertEqual(outputs["input_ids"], [10, 11])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 2)
        self.assertEqual(outputs["position_ids"][0], [5, 5, 5])
        self.assertEqual(outputs["position_ids"][1], [6, 6, 6])
        self.assertEqual(outputs["cur_position"], 7)

    def test_add_processed_image(self):
        enc, _ = self._make_enc()
        # spatial_conv_size=2, so 16 // 4 = 4 tokens
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        outputs = enc._make_outputs()
        enc.add_processed_image((img, meta), outputs, uuid="ernie_img")

        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertEqual(outputs["mm_hashes"], ["ernie_img"])
        self.assertEqual(outputs["image_type_ids"], [0])
        self.assertEqual(len(outputs["position_ids"]), 4)  # list-of-lists

    def test_add_processed_image_token_mismatch(self):
        enc, _ = self._make_enc()
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError):
            enc.add_processed_image((img, meta), outputs, uuid="x", token_len=999)

    def test_add_processed_video(self):
        enc, _ = self._make_enc()
        # spatial_conv=2, temporal_conv=2: 32 // (4*2) = 4 tokens
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        outputs = enc._make_outputs()
        enc.add_processed_video((frames, meta), outputs, uuid="ernie_vid")

        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["video"]] * 4)
        self.assertEqual(outputs["image_type_ids"], [1, 1, 1, 1])
        self.assertEqual(outputs["mm_hashes"], ["ernie_vid"])

    def test_add_processed_video_token_mismatch(self):
        enc, _ = self._make_enc()
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError):
            enc.add_processed_video((frames, meta), outputs, uuid="x", token_len=999)

    def test_mm_num_tokens_image(self):
        """t=1: t*h*w//4 (no extra //2)."""
        result = ErnieEncoding.mm_num_tokens([1, 4, 4])
        self.assertEqual(result, 4)

    def test_mm_num_tokens_video(self):
        """t>1: t*h*w//4//2."""
        result = ErnieEncoding.mm_num_tokens([2, 4, 4])
        self.assertEqual(result, 4)  # 2*4*4//4//2 = 4

    def test_mm_num_tokens_list(self):
        result = ErnieEncoding.mm_num_tokens([[1, 4, 4], [4, 4, 4]])
        self.assertEqual(result, [4, 8])  # [16//4, 64//4//2]

    def test_mm_num_tokens_empty(self):
        self.assertEqual(ErnieEncoding.mm_num_tokens([]), 0)

    def test_pack_position_ids(self):
        enc, _ = self._make_enc()
        outputs = enc._make_outputs()
        enc.add_text_positions(outputs, 2)
        enc.pack_position_ids(outputs)
        self.assertIsInstance(outputs["position_ids"], np.ndarray)
        self.assertEqual(outputs["position_ids"].dtype, np.int64)
        self.assertEqual(outputs["position_ids"].shape, (2, 3))
        self.assertEqual(outputs["image_patch_id"], enc.image_token_id)

    def test_get_mm_max_tokens_per_item(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        # get_smarted_resize returns ((resized_h, resized_w), (patches_h, patches_w))
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        result = enc.get_mm_max_tokens_per_item(seq_len=1000)
        self.assertIn("image", result)
        self.assertIn("video", result)
        # patches 4*4 // (2*2) = 4 for image
        self.assertEqual(result["image"], 4)
        # patches 4*4 // (2*2*2) = 2 for video
        self.assertEqual(result["video"], 2)

    def test_get_mm_max_tokens_capped_by_seq_len(self):
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (100, 100))
        result = enc.get_mm_max_tokens_per_item(seq_len=10)
        # Should be capped at seq_len
        self.assertLessEqual(result["image"], 10)
        self.assertLessEqual(result["video"], 10)

    def test_set_video_frame_args_target_frames(self):
        enc, _ = self._make_enc()
        args = {
            "target_frames": 30,
            "fps": -1,
            "min_frames": 10,
            "max_frames": 100,
            "frames_sample": "leading",
        }
        result = enc.set_video_frame_args(args, {"duration": 10})
        self.assertEqual(result["target_frames"], 30)

    def test_set_video_frame_args_target_frames_fps_positive_raises(self):
        enc, _ = self._make_enc()
        args = {"target_frames": 30, "fps": 2, "min_frames": 0, "max_frames": 0, "frames_sample": "leading"}
        with self.assertRaises(ValueError, msg="fps must be negative"):
            enc.set_video_frame_args(args, {"duration": 10})

    def test_set_video_frame_args_target_frames_below_min_raises(self):
        enc, _ = self._make_enc()
        args = {"target_frames": 5, "fps": -1, "min_frames": 10, "max_frames": 100, "frames_sample": "leading"}
        with self.assertRaises(ValueError, msg="target_frames must be larger"):
            enc.set_video_frame_args(args, {"duration": 10})

    def test_set_video_frame_args_target_frames_above_max_raises(self):
        enc, _ = self._make_enc()
        args = {"target_frames": 200, "fps": -1, "min_frames": 10, "max_frames": 100, "frames_sample": "leading"}
        with self.assertRaises(ValueError, msg="target_frames must be smaller"):
            enc.set_video_frame_args(args, {"duration": 10})

    def test_set_video_frame_args_fps_negative_no_target_raises(self):
        enc, _ = self._make_enc()
        args = {"target_frames": -1, "fps": -1, "min_frames": 0, "max_frames": 0, "frames_sample": "leading"}
        with self.assertRaises(ValueError, msg="Must provide either"):
            enc.set_video_frame_args(args, {"duration": 10})

    def test_set_video_frame_args_min_greater_than_max_raises(self):
        enc, _ = self._make_enc()
        args = {"target_frames": -1, "fps": 2, "min_frames": 100, "max_frames": 10, "frames_sample": "leading"}
        with self.assertRaises(ValueError, msg="min_frames must be smaller"):
            enc.set_video_frame_args(args, {"duration": 10})

    def test_set_video_frame_args_fps_clamp_to_min(self):
        """When fps * duration < min_frames, switch to target_frames."""
        enc, _ = self._make_enc()
        args = {"target_frames": -1, "fps": 1, "min_frames": 30, "max_frames": 100, "frames_sample": "leading"}
        result = enc.set_video_frame_args(args, {"duration": 10})
        # 1 * 10 = 10 < 30 → target_frames = 30, fps = -1
        self.assertEqual(result["target_frames"], 30)
        self.assertEqual(result["fps"], -1)

    def test_set_video_frame_args_fps_clamp_to_max(self):
        """When fps * duration > max_frames, switch to target_frames."""
        enc, _ = self._make_enc()
        args = {"target_frames": -1, "fps": 10, "min_frames": 1, "max_frames": 50, "frames_sample": "leading"}
        result = enc.set_video_frame_args(args, {"duration": 10})
        # 10 * 10 = 100 > 50 → target_frames = 50, fps = -1
        self.assertEqual(result["target_frames"], 50)
        self.assertEqual(result["fps"], -1)

    def test_prompt_token_ids2outputs_text_only(self):
        """prompt_token_ids without mm_items — text-only path."""
        enc, _ = self._make_enc()
        outputs = enc.prompt_token_ids2outputs([10, 20, 30])
        self.assertEqual(outputs["input_ids"], [10, 20, 30])
        self.assertEqual(len(outputs["position_ids"]), 3)
        self.assertEqual(outputs["position_ids"][0], [0, 0, 0])
        self.assertEqual(outputs["position_ids"][2], [2, 2, 2])
        self.assertEqual(outputs["cur_position"], 3)

    def test_prompt_token_ids2outputs_with_processed_image(self):
        """prompt_token_ids with image boundary tokens and processed image."""
        enc, mock_proc = self._make_enc()
        # image_start=200, image_end=201, image_token=102
        # Build: [text(1), IMG_START(200), placeholder(102,102,102,102), IMG_END(201), text(2)]
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        mm_items = [{"type": "image", "data": (img, meta), "uuid": "img_uuid"}]
        outputs = enc.prompt_token_ids2outputs([1, 200, 102, 102, 102, 102, 201, 2], mm_items)
        # 1 text + 1 img_start + 4 image + 1 img_end + 1 text = 8
        self.assertEqual(len(outputs["input_ids"]), 8)
        # Boundary tokens (IMG_START, IMG_END) must be typed as "image", not "text"
        tt = outputs["token_type_ids"]
        self.assertEqual(tt[0], IDS_TYPE_FLAG["text"])  # text
        self.assertEqual(tt[1], IDS_TYPE_FLAG["image"])  # IMG_START
        for i in range(2, 6):
            self.assertEqual(tt[i], IDS_TYPE_FLAG["image"])  # image tokens
        self.assertEqual(tt[6], IDS_TYPE_FLAG["image"])  # IMG_END
        self.assertEqual(tt[7], IDS_TYPE_FLAG["text"])  # text

    # ------------------------------------------------------------------
    # add_image (raw image path)
    # ------------------------------------------------------------------
    def test_add_image(self):
        """Raw image: get_smarted_resize → preprocess → outputs populated."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        # get_smarted_resize returns ((resized_h, resized_w), (patches_h, patches_w))
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        ip.preprocess.return_value = {
            "pixel_values": np.zeros((4, 3, 28, 28)),
            "image_grid_thw": np.array([[1, 4, 4]]),
        }
        mock_img = MagicMock()
        mock_img.height = 100
        mock_img.width = 100
        mock_img.convert.return_value = mock_img
        outputs = enc._make_outputs()
        enc.add_image(mock_img, outputs, uuid="img_hash_1")

        # 4*4 // (2**2) = 4 tokens
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertTrue(all(t == enc.image_token_id for t in outputs["input_ids"]))
        self.assertEqual(outputs["num_input_image_tokens"], 4)
        self.assertEqual(outputs["mm_hashes"], ["img_hash_1"])
        self.assertEqual(outputs["image_type_ids"], [0])
        self.assertEqual(len(outputs["position_ids"]), 4)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(len(outputs["grid_thw"]), 1)
        # Verify preprocess was called
        ip.preprocess.assert_called_once()

    def test_add_image_without_uuid_hashes(self):
        """When uuid is None, mm_hashes should be computed via MultimodalHasher."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        pixel_values = np.zeros((4, 3, 28, 28))
        ip.preprocess.return_value = {
            "pixel_values": pixel_values,
            "image_grid_thw": np.array([[1, 4, 4]]),
        }
        mock_img = MagicMock()
        mock_img.height = 100
        mock_img.width = 100
        mock_img.convert.return_value = mock_img
        outputs = enc._make_outputs()

        from unittest.mock import patch

        with patch("fastdeploy.input.encodings.ernie_encoding.MultimodalHasher") as mock_hasher:
            mock_hasher.hash_features.return_value = "computed_hash"
            enc.add_image(mock_img, outputs, uuid=None)

        self.assertEqual(outputs["mm_hashes"], ["computed_hash"])
        mock_hasher.hash_features.assert_called_once_with(pixel_values)

    def test_add_image_token_len_mismatch(self):
        """token_len mismatch raises ValueError."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        mock_img = MagicMock()
        mock_img.height = 100
        mock_img.width = 100
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError, msg="image tokens num not match"):
            enc.add_image(mock_img, outputs, uuid="x", token_len=999)

    # ------------------------------------------------------------------
    # add_video (raw video frames path)
    # ------------------------------------------------------------------
    def test_add_video(self):
        """Raw video frames: get_smarted_resize → preprocess → outputs populated."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        ip.preprocess.return_value = {
            "pixel_values_videos": np.zeros((8, 3, 28, 28)),
            "video_grid_thw": np.array([[2, 4, 4]]),
        }
        # Create 2 mock PIL-like frames
        frames = []
        for _ in range(2):
            f = MagicMock()
            f.height = 100
            f.width = 100
            f.convert.return_value = MagicMock(__array__=lambda self: np.zeros((100, 100, 3)))
            # np.array(f.convert("RGB")) needs to work
            frames.append(f)

        # Patch np.array for the frame conversion inside add_video
        outputs = enc._make_outputs()

        from unittest.mock import patch

        original_np_array = np.array
        original_np_stack = np.stack

        def mock_np_array(obj, *args, **kwargs):
            if hasattr(obj, "convert"):
                return np.zeros((100, 100, 3), dtype=np.uint8)
            return original_np_array(obj, *args, **kwargs)

        with patch("fastdeploy.input.encodings.ernie_encoding.np.array", side_effect=mock_np_array):
            with patch("fastdeploy.input.encodings.ernie_encoding.np.stack", side_effect=original_np_stack):
                enc.add_video(frames, outputs, uuid="vid_hash_1")

        # 2 frames * 4*4 // (2**2 * 2) = 32 // 8 = 4 tokens
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertTrue(all(t == enc.image_token_id for t in outputs["input_ids"]))
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["video"]] * 4)
        self.assertEqual(outputs["num_input_video_tokens"], 4)
        self.assertEqual(outputs["mm_hashes"], ["vid_hash_1"])
        self.assertEqual(outputs["image_type_ids"], [1, 1])
        self.assertEqual(len(outputs["position_ids"]), 4)

    def test_add_video_without_uuid_hashes(self):
        """When uuid is None, mm_hashes should be computed via MultimodalHasher."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        pixel_values_videos = np.zeros((8, 3, 28, 28))
        ip.preprocess.return_value = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": np.array([[2, 4, 4]]),
        }
        frames = []
        for _ in range(2):
            f = MagicMock()
            f.height = 100
            f.width = 100
            frames.append(f)

        outputs = enc._make_outputs()

        from unittest.mock import patch

        original_np_array = np.array
        original_np_stack = np.stack

        def mock_np_array(obj, *args, **kwargs):
            if hasattr(obj, "convert"):
                return np.zeros((100, 100, 3), dtype=np.uint8)
            return original_np_array(obj, *args, **kwargs)

        with patch("fastdeploy.input.encodings.ernie_encoding.np.array", side_effect=mock_np_array):
            with patch("fastdeploy.input.encodings.ernie_encoding.np.stack", side_effect=original_np_stack):
                with patch("fastdeploy.input.encodings.ernie_encoding.MultimodalHasher") as mock_hasher:
                    mock_hasher.hash_features.return_value = "computed_vid_hash"
                    enc.add_video(frames, outputs, uuid=None)

        self.assertEqual(outputs["mm_hashes"], ["computed_vid_hash"])
        mock_hasher.hash_features.assert_called_once_with(pixel_values_videos)

    def test_add_video_token_len_mismatch(self):
        """token_len mismatch raises ValueError."""
        enc, mock_proc = self._make_enc()
        ip = mock_proc.image_processor
        ip.get_smarted_resize.return_value = ((56, 56), (4, 4))
        frame = MagicMock()
        frame.height = 100
        frame.width = 100
        outputs = enc._make_outputs()
        with self.assertRaises(ValueError, msg="video tokens num not match"):
            enc.add_video([frame, frame], outputs, uuid="x", token_len=999)

    # ------------------------------------------------------------------
    # load_video (mocked decord imports)
    # ------------------------------------------------------------------
    def test_load_video(self):
        """load_video calls decord helpers and returns (frames, {})."""
        enc, _ = self._make_enc()

        from unittest.mock import patch

        mock_reader = MagicMock()
        mock_meta = {"duration": 10, "fps": 30}
        mock_path = "/tmp/test_video.mp4"

        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        rendered_frame1 = MagicMock()
        rendered_frame2 = MagicMock()

        with (
            patch(
                "fastdeploy.input.utils.video.read_video_decord",
                return_value=(mock_reader, mock_meta, mock_path),
            ) as mock_read_video,
            patch(
                "fastdeploy.input.utils.video.read_frames_decord",
                return_value=([mock_frame1, mock_frame2], None, [0.0, 0.5]),
            ) as mock_read_frames,
            patch(
                "fastdeploy.input.utils.render_timestamp.render_frame_timestamp",
                side_effect=[rendered_frame1, rendered_frame2],
            ),
        ):
            frames, meta = enc.load_video("http://example.com/video.mp4", {})

        self.assertEqual(len(frames), 2)
        self.assertEqual(meta, {})
        mock_read_video.assert_called_once()
        mock_read_frames.assert_called_once()

    def test_load_video_odd_frames_padded(self):
        """When decord returns odd number of frames, load_video pads to even."""
        enc, _ = self._make_enc()

        from unittest.mock import patch

        mock_reader = MagicMock()
        mock_meta = {"duration": 10, "fps": 30}
        mock_path = "/tmp/test_video.mp4"

        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()
        mock_frame3 = MagicMock()
        rendered1 = MagicMock()
        rendered2 = MagicMock()
        rendered3 = MagicMock()

        with (
            patch(
                "fastdeploy.input.utils.video.read_video_decord",
                return_value=(mock_reader, mock_meta, mock_path),
            ),
            patch(
                "fastdeploy.input.utils.video.read_frames_decord",
                return_value=([mock_frame1, mock_frame2, mock_frame3], None, [0.0, 0.5, 1.0]),
            ),
            patch(
                "fastdeploy.input.utils.render_timestamp.render_frame_timestamp",
                side_effect=[rendered1, rendered2, rendered3],
            ),
        ):
            frames, meta = enc.load_video("http://example.com/video.mp4", {})

        # 3 frames → padded to 4
        self.assertEqual(len(frames), 4)
        self.assertEqual(meta, {})

    def test_load_video_with_item_overrides(self):
        """load_video uses per-item fps/min_frames/max_frames overrides."""
        enc, _ = self._make_enc()

        from unittest.mock import patch

        mock_reader = MagicMock()
        mock_meta = {"duration": 10, "fps": 30}
        mock_path = "/tmp/test_video.mp4"

        with (
            patch(
                "fastdeploy.input.utils.video.read_video_decord",
                return_value=(mock_reader, mock_meta, mock_path),
            ),
            patch(
                "fastdeploy.input.utils.video.read_frames_decord",
                return_value=([MagicMock(), MagicMock()], None, [0.0, 0.5]),
            ) as mock_read_frames,
            patch(
                "fastdeploy.input.utils.render_timestamp.render_frame_timestamp",
                side_effect=[MagicMock(), MagicMock()],
            ),
        ):
            item = {"fps": -1, "target_frames": 20, "min_frames": 5, "max_frames": 50}
            frames, meta = enc.load_video("http://example.com/video.mp4", item)

        self.assertEqual(len(frames), 2)
        # Verify read_frames_decord got the overridden target_frames
        call_kwargs = mock_read_frames.call_args
        self.assertEqual(
            call_kwargs[1].get("target_frames", call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None), 20
        )

    # ------------------------------------------------------------------
    # prompt_token_ids2outputs — video branch
    # ------------------------------------------------------------------
    def test_prompt_token_ids2outputs_with_processed_video(self):
        """prompt_token_ids with video boundary tokens and processed video."""
        enc, mock_proc = self._make_enc()
        # video_start=202, video_end=203, image_token=102
        # Build: [text(1), VID_START(202), placeholder(102)*4, VID_END(203), text(2)]
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        mm_items = [{"type": "video", "data": (frames, meta), "uuid": "vid_uuid"}]
        outputs = enc.prompt_token_ids2outputs([1, 202, 102, 102, 102, 102, 203, 2], mm_items)
        # 1 text + 1 vid_start + 4 video + 1 vid_end + 1 text = 8
        self.assertEqual(len(outputs["input_ids"]), 8)
        self.assertEqual(outputs["input_ids"][0], 1)
        self.assertEqual(outputs["input_ids"][1], 202)  # vid_start
        self.assertEqual(outputs["input_ids"][-1], 2)
        # Boundary tokens (VID_START, VID_END) must be typed as "image", not "text"
        tt = outputs["token_type_ids"]
        self.assertEqual(tt[0], IDS_TYPE_FLAG["text"])  # text
        self.assertEqual(tt[1], IDS_TYPE_FLAG["image"])  # VID_START
        for i in range(2, 6):
            self.assertEqual(tt[i], IDS_TYPE_FLAG["video"])  # video tokens
        self.assertEqual(tt[6], IDS_TYPE_FLAG["image"])  # VID_END
        self.assertEqual(tt[7], IDS_TYPE_FLAG["text"])  # text

    def test_prompt_token_ids2outputs_with_raw_video_url(self):
        """prompt_token_ids with raw video (string url) — triggers load_video."""
        enc, mock_proc = self._make_enc()

        from unittest.mock import patch

        mock_frames = [MagicMock() for _ in range(2)]

        mm_items = [{"type": "video", "data": "http://example.com/video.mp4", "uuid": "vid_uuid"}]

        # 2 frames, 4x4 patches → 2*4*4 // (4*2) = 4 tokens
        with (
            patch.object(enc, "load_video", return_value=(mock_frames, {})) as mock_load,
            patch.object(enc, "add_video") as mock_add_video,
        ):
            enc.prompt_token_ids2outputs([1, 202, 102, 102, 102, 102, 203, 2], mm_items)

        mock_load.assert_called_once_with("http://example.com/video.mp4", {})
        mock_add_video.assert_called_once()

    def test_prompt_token_ids2outputs_with_raw_video_dict(self):
        """prompt_token_ids with raw video (dict form) — triggers load_video."""
        enc, mock_proc = self._make_enc()

        from unittest.mock import patch

        mock_frames = [MagicMock() for _ in range(2)]
        video_dict = {"video": "http://example.com/video.mp4", "fps": 5}

        mm_items = [{"type": "video", "data": video_dict, "uuid": "vid_uuid"}]

        with (
            patch.object(enc, "load_video", return_value=(mock_frames, {})) as mock_load,
            patch.object(enc, "add_video"),
        ):
            enc.prompt_token_ids2outputs([1, 202, 102, 102, 102, 102, 203, 2], mm_items)

        mock_load.assert_called_once_with("http://example.com/video.mp4", video_dict)

    # ------------------------------------------------------------------
    # prompt_token_ids2outputs — error paths
    # ------------------------------------------------------------------
    def test_prompt_token_ids2outputs_image_placeholder_overflow(self):
        """More image start tokens than images provided raises ValueError."""
        enc, mock_proc = self._make_enc()
        mm_items = []  # no images
        with self.assertRaises(ValueError, msg="more image placeholder"):
            enc.prompt_token_ids2outputs([200, 102, 201], mm_items)  # IMG_START but no images

    def test_prompt_token_ids2outputs_image_tokens_incomplete(self):
        """Image start without matching end raises ValueError."""
        enc, mock_proc = self._make_enc()
        img = np.zeros((16, 3, 28, 28))
        meta = {"thw": (1, 4, 4)}
        mm_items = [{"type": "image", "data": (img, meta), "uuid": "uuid"}]
        # IMG_START(200) followed by placeholders but NO IMG_END(201)
        with self.assertRaises(ValueError, msg="image token ids not complete"):
            enc.prompt_token_ids2outputs([200, 102, 102, 102], mm_items)

    def test_prompt_token_ids2outputs_video_placeholder_overflow(self):
        """More video start tokens than videos provided raises ValueError."""
        enc, mock_proc = self._make_enc()
        mm_items = []  # no videos
        with self.assertRaises(ValueError, msg="more video placeholder"):
            enc.prompt_token_ids2outputs([202, 102, 203], mm_items)  # VID_START but no videos

    def test_prompt_token_ids2outputs_video_tokens_incomplete(self):
        """Video start without matching end raises ValueError."""
        enc, mock_proc = self._make_enc()
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        mm_items = [{"type": "video", "data": (frames, meta), "uuid": "uuid"}]
        # VID_START(202) followed by placeholders but NO VID_END(203)
        with self.assertRaises(ValueError, msg="video token ids not complete"):
            enc.prompt_token_ids2outputs([202, 102, 102, 102], mm_items)

    def test_prompt_token_ids2outputs_image_count_mismatch(self):
        """Fewer image placeholders than images raises ValueError."""
        enc, mock_proc = self._make_enc()
        img1 = np.zeros((16, 3, 28, 28))
        meta1 = {"thw": (1, 4, 4)}
        mm_items = [
            {"type": "image", "data": (img1, meta1), "uuid": "uuid1"},
            {"type": "image", "data": (img1, meta1), "uuid": "uuid2"},
        ]
        # Only 1 image placeholder in token ids
        with self.assertRaises(ValueError, msg="number of images does not match"):
            enc.prompt_token_ids2outputs([1, 200, 102, 102, 102, 102, 201, 2], mm_items)

    def test_prompt_token_ids2outputs_video_count_mismatch(self):
        """Fewer video placeholders than videos raises ValueError."""
        enc, mock_proc = self._make_enc()
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        mm_items = [
            {"type": "video", "data": (frames, meta), "uuid": "uuid1"},
            {"type": "video", "data": (frames, meta), "uuid": "uuid2"},
        ]
        # Only 1 video placeholder in token ids
        with self.assertRaises(ValueError, msg="number of videos does not match"):
            enc.prompt_token_ids2outputs([1, 202, 102, 102, 102, 102, 203, 2], mm_items)

    # ------------------------------------------------------------------
    # prompt_token_ids2outputs — with raw image (non-tuple)
    # ------------------------------------------------------------------
    def test_prompt_token_ids2outputs_with_raw_image(self):
        """prompt_token_ids with raw image (non-tuple) triggers add_image."""
        enc, mock_proc = self._make_enc()

        from unittest.mock import patch

        mock_img = MagicMock()  # raw image, not a tuple

        mm_items = [{"type": "image", "data": mock_img, "uuid": "img_uuid"}]

        with patch.object(enc, "add_image") as mock_add_image:
            enc.prompt_token_ids2outputs([1, 200, 102, 102, 102, 102, 201, 2], mm_items)

        mock_add_image.assert_called_once()
        call_args = mock_add_image.call_args
        self.assertIs(call_args[0][0], mock_img)
        self.assertEqual(call_args[0][2], "img_uuid")
        self.assertEqual(call_args[0][3], 4)  # token_len = 4 placeholders

    # ------------------------------------------------------------------
    # prompt_token_ids2outputs — video uuid edge case
    # ------------------------------------------------------------------
    def test_prompt_token_ids2outputs_video_uuid_none(self):
        """When video item has no uuid, uuid should be None."""
        enc, mock_proc = self._make_enc()
        frames = np.zeros((32, 3, 28, 28))
        meta = {"thw": (4, 4, 4)}
        mm_items = [{"type": "video", "data": (frames, meta)}]  # no "uuid" key
        outputs = enc.prompt_token_ids2outputs([1, 202, 102, 102, 102, 102, 203, 2], mm_items)
        self.assertEqual(len(outputs["input_ids"]), 8)


if __name__ == "__main__":
    unittest.main()
