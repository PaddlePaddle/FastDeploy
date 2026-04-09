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

    mock_processor._extract_mm_items = MagicMock()
    mock_processor.update_processor_cache = MagicMock()

    import importlib

    mod = importlib.import_module(cfg.encoding_module)
    cls = getattr(mod, cfg.encoding_class)
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
        """prompt_token_ids with no messages — text-only path."""
        enc, _ = self._make_enc(QWEN3_VL)
        request = {"prompt_token_ids": [1, 2, 3]}
        outputs = enc.prompt_token_ids2outputs(request)
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
        mock_proc._extract_mm_items.return_value = (
            [mock_img],  # images
            [],  # videos
            ["img_uuid"],  # image_uuid
            [],  # video_uuid
            None,  # dealer
            [],  # missing_idx
            [{"type": "image", "data": mock_img, "uuid": "img_uuid"}],  # mm_items
        )
        request = {
            "prompt_token_ids": [1, 100, 100, 100, 100, 2],
            "messages": [{"role": "user", "content": "hi"}],
        }
        outputs = enc.prompt_token_ids2outputs(request)
        # 1 text + 4 image + 1 text = 6
        self.assertEqual(len(outputs["input_ids"]), 6)

    def test_prompt_token_ids2outputs_mm_count_mismatch(self):
        """More placeholders than mm_items raises."""
        enc, mock_proc = self._make_enc(QWEN3_VL)
        mock_proc._extract_mm_items.return_value = (
            [],
            [],
            [],
            [],
            None,
            [],
            [],  # no mm_items
        )
        request = {
            "prompt_token_ids": [100, 100],  # image tokens but no images
            "messages": [{"role": "user", "content": "hi"}],
        }
        with self.assertRaises(ValueError):
            enc.prompt_token_ids2outputs(request)


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
        """prompt_token_ids without messages — text-only path."""
        enc, _ = self._make_enc()
        request = {"prompt_token_ids": [10, 20, 30]}
        outputs = enc.prompt_token_ids2outputs(request)
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
        mock_proc._extract_mm_items.return_value = (
            [(img, meta)],  # images (tuple = processed)
            [],
            ["img_uuid"],
            [],
            None,
            [],
            [{"type": "image", "data": (img, meta), "uuid": "img_uuid"}],
        )
        request = {
            "prompt_token_ids": [1, 200, 102, 102, 102, 102, 201, 2],
            "messages": [{"role": "user", "content": "hi"}],
        }
        outputs = enc.prompt_token_ids2outputs(request)
        # 1 text + 1 img_start + 4 image + 1 img_end + 1 text = 8
        self.assertEqual(len(outputs["input_ids"]), 8)


if __name__ == "__main__":
    unittest.main()
