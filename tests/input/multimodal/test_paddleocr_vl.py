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

"""Unit tests for PaddleOCRVLProcessor."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.mm_processor import MMContext, MMItem, TokenizationPath
from fastdeploy.input.multimodal.paddleocr_vl import PaddleOCRVLProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG


def _make_paddleocr_processor(**overrides):
    """Create a PaddleOCRVLProcessor with mocked dependencies."""
    with patch.object(PaddleOCRVLProcessor, "__init__", return_value=None):
        proc = PaddleOCRVLProcessor.__new__(PaddleOCRVLProcessor)

    proc.tokenizer = MagicMock()
    proc.tokenizer.convert_tokens_to_ids.side_effect = lambda x: {
        "<|IMAGE_PLACEHOLDER|>": 200,
        "<|video_pad|>": 201,
    }.get(x, 999)
    proc.tokenizer.tokenize.return_value = ["tok"]
    proc.model_name_or_path = "test-model"
    proc.config = SimpleNamespace(vision_config=SimpleNamespace(tokens_per_second=2))
    proc._cache = None
    proc.enable_processor_cache = False

    proc.image_placeholder = "<|IMAGE_PLACEHOLDER|>"
    proc.video_placeholder = "<|video_pad|>"
    proc.image_token_str = "<|IMAGE_PLACEHOLDER|>"
    proc.video_token_str = "<|video_pad|>"

    proc.image_token_id = 200
    proc.video_token_id = 201

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


class TestPaddleOCRMakeOutputs(unittest.TestCase):
    def test_has_vit_fields(self):
        proc = _make_paddleocr_processor()
        outputs = proc._make_outputs()
        self.assertIn("vit_seqlen", outputs)
        self.assertIn("vit_position_ids", outputs)
        self.assertEqual(outputs["vit_seqlen"], [])
        self.assertEqual(outputs["vit_position_ids"], [])

    def test_has_fps_field(self):
        """Should also have fps field inherited from QwenVLProcessor."""
        proc = _make_paddleocr_processor()
        outputs = proc._make_outputs()
        self.assertIn("fps", outputs)
        self.assertEqual(outputs["fps"], [])

    def test_has_base_fields(self):
        """Base fields from MMProcessor should be present."""
        proc = _make_paddleocr_processor()
        outputs = proc._make_outputs()
        self.assertIn("input_ids", outputs)
        self.assertIn("images", outputs)
        self.assertIn("mm_hashes", outputs)


class TestPaddleOCRPreprocessImage(unittest.TestCase):
    def setUp(self):
        self.proc = _make_paddleocr_processor()

    def test_raw_image(self):
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4)
        outputs = self.proc._make_outputs()

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        self.proc.preprocess_image(mock_img, outputs, uuid="img_uuid")

        self.assertEqual(len(outputs["images"]), 1)
        # 1*2*2//4 = 1 token
        self.assertEqual(outputs["input_ids"], [200])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["image"]])
        self.assertEqual(outputs["num_input_image_tokens"], 1)
        self.assertEqual(outputs["fps"], [0])
        self.proc.image_processor.preprocess.assert_called_once()

    def test_appends_vit_fields(self):
        """Preprocess image should append vit_seqlen and vit_position_ids."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=4, w=4, num_pixels=16)
        outputs = self.proc._make_outputs()

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        self.proc.preprocess_image(mock_img, outputs, uuid="img_uuid")

        # vit_seqlen = h * w = 4 * 4 = 16
        self.assertEqual(outputs["vit_seqlen"], [16])
        self.assertEqual(len(outputs["vit_position_ids"]), 1)
        np.testing.assert_array_equal(outputs["vit_position_ids"][0], np.arange(16) % 16)

    def test_cached_image_appends_vit_fields(self):
        """Preprocess cached image should also append vit fields."""
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)  # 4 pixels, merge_size=2 -> 4//4=1 token
        meta = {"thw": (1, 2, 2)}
        img_cache = (cached_pixels, meta)

        self.proc.preprocess_cached_image(img_cache, outputs, uuid="cached_uuid")

        # vit_seqlen = h * w = 2 * 2 = 4
        self.assertEqual(outputs["vit_seqlen"], [4])
        self.assertEqual(len(outputs["vit_position_ids"]), 1)
        np.testing.assert_array_equal(outputs["vit_position_ids"][0], np.arange(4) % 4)

    def test_cached_image_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)
        meta = {"thw": (1, 2, 2)}
        img_cache = (cached_pixels, meta)

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_image(img_cache, outputs, uuid="u", token_len=999)


class TestPaddleOCRPreprocessVideo(unittest.TestCase):
    def setUp(self):
        self.proc = _make_paddleocr_processor()

    def test_uses_video_token_id(self):
        """Video preprocessing should use video_token_id (201), not image_token_id (200)."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=2, h=2, w=2, num_pixels=8)
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 2.0}

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid", meta=meta)

        # 2*2*2//4 = 2 tokens, all should be video_token_id=201
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertTrue(all(tid == 201 for tid in outputs["input_ids"]))

    def test_appends_vit_fields(self):
        """Video preprocessing should append vit_seqlen and vit_position_ids."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=2, h=4, w=4, num_pixels=32)
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 2.0}

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid", meta=meta)

        # vit_seqlen = h * w = 4 * 4 = 16
        self.assertEqual(outputs["vit_seqlen"], [16])
        self.assertEqual(len(outputs["vit_position_ids"]), 1)
        np.testing.assert_array_equal(outputs["vit_position_ids"][0], np.arange(16) % 16)

    def test_token_type_is_video(self):
        """Token type IDs should be video type."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=2, h=2, w=2, num_pixels=8)
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 2.0}

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid", meta=meta)

        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["video"]] * 2)

    def test_cached_video_uses_video_token_id(self):
        """Cached video should also use video_token_id (201)."""
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((16, 3), dtype=np.float32)  # 16//4=4 tokens
        meta = {"thw": (2, 2, 2), "fps": 4.0}
        frames_cache = (cached_pixels, meta)

        self.proc.preprocess_cached_video(frames_cache, outputs, uuid="vid_uuid")

        # All tokens should be video_token_id=201
        self.assertEqual(len(outputs["input_ids"]), 4)
        self.assertTrue(all(tid == 201 for tid in outputs["input_ids"]))

    def test_cached_video_appends_vit_fields(self):
        """Cached video should append vit fields."""
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((16, 3), dtype=np.float32)
        meta = {"thw": (2, 4, 4), "fps": 4.0}
        frames_cache = (cached_pixels, meta)

        self.proc.preprocess_cached_video(frames_cache, outputs, uuid="vid_uuid")

        # vit_seqlen = h * w = 4 * 4 = 16
        self.assertEqual(outputs["vit_seqlen"], [16])
        self.assertEqual(len(outputs["vit_position_ids"]), 1)
        np.testing.assert_array_equal(outputs["vit_position_ids"][0], np.arange(16) % 16)

    def test_cached_video_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((16, 3), dtype=np.float32)
        meta = {"thw": (2, 2, 2), "fps": 4.0}
        frames_cache = (cached_pixels, meta)

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_video(frames_cache, outputs, uuid="u", token_len=999)

    def test_fps_recorded(self):
        """FPS from meta should be recorded in outputs."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=2, h=2, w=2, num_pixels=8)
        outputs = self.proc._make_outputs()
        frames = np.zeros((4, 224, 224, 3))
        meta = {"fps": 3.5}

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid", meta=meta)

        self.assertEqual(outputs["fps"], [3.5])


class TestPaddleOCRLoadVideo(unittest.TestCase):
    @patch("fastdeploy.input.multimodal.paddleocr_vl._sample_paddleocr")
    @patch("fastdeploy.input.multimodal.paddleocr_vl.read_video_decord")
    def test_basic_load(self, mock_read, mock_sample):
        proc = _make_paddleocr_processor()

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value=mock_frame)
        mock_read.return_value = (mock_reader, {"num_of_frame": 4, "fps": 30.0, "duration": 2.0}, None)
        mock_sample.return_value = [0, 1]

        frames, meta = proc.load_video("http://video.mp4", {})

        self.assertEqual(frames.shape[0], 2)
        mock_read.assert_called_once()

    @patch("fastdeploy.input.multimodal.paddleocr_vl._sample_paddleocr")
    @patch("fastdeploy.input.multimodal.paddleocr_vl.read_video_decord")
    def test_uses_paddleocr_sampler(self, mock_read, mock_sample):
        """Should use sample_frames_paddleocr, not _sample_qwen."""
        proc = _make_paddleocr_processor()

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value=mock_frame)
        mock_read.return_value = (mock_reader, {"num_of_frame": 8, "fps": 30.0, "duration": 4.0}, None)
        mock_sample.return_value = [0, 2, 4, 6]

        frames, meta = proc.load_video("http://video.mp4", {"fps": 2.0})

        mock_sample.assert_called_once()
        # Should pass temporal_conv_size as frame_factor (not FRAME_FACTOR)
        call_kwargs = mock_sample.call_args[1]
        self.assertEqual(call_kwargs["frame_factor"], proc.temporal_conv_size)

    @patch("fastdeploy.input.multimodal.paddleocr_vl._sample_paddleocr")
    @patch("fastdeploy.input.multimodal.paddleocr_vl.read_video_decord")
    def test_no_sampling_when_fps_negative(self, mock_read, mock_sample):
        """When fps <= 0 and target_frames <= 0, no sampling is performed."""
        proc = _make_paddleocr_processor(fps=-1.0, target_frames=-1)

        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.__getitem__ = MagicMock(return_value=mock_frame)
        mock_read.return_value = (mock_reader, {"num_of_frame": 3, "fps": 30.0, "duration": 1.0}, None)

        frames, meta = proc.load_video("http://video.mp4", {"fps": -1.0, "target_frames": -1})

        mock_sample.assert_not_called()
        self.assertEqual(frames.shape[0], 3)


class TestPaddleOCRPromptTokenIds2Outputs(unittest.TestCase):
    def setUp(self):
        self.proc = _make_paddleocr_processor()

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
        """Image token run triggers preprocess_image with correct token ID."""
        self.proc.image_processor.preprocess.return_value = _mock_preprocess_return(t=1, h=2, w=2, num_pixels=4)

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        img_item = MMItem(type="image", data=mock_img, uuid="u1")

        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            # token_id 200 is image_token_id for PaddleOCR
            prompt_token_ids=[1, 2, 200, 3],
        )
        outputs = self.proc.prompt_token_ids2outputs(ctx)

        self.assertIn(IDS_TYPE_FLAG["image"], outputs["token_type_ids"])
        self.assertEqual(len(outputs["images"]), 1)
        # Should have vit fields
        self.assertEqual(len(outputs["vit_seqlen"]), 1)
        self.assertEqual(len(outputs["vit_position_ids"]), 1)


class TestPaddleOCRClassAttributes(unittest.TestCase):
    def test_image_placeholder_differs_from_qwen(self):
        """PaddleOCR uses <|IMAGE_PLACEHOLDER|> not <|image_pad|>."""
        self.assertEqual(PaddleOCRVLProcessor.image_placeholder, "<|IMAGE_PLACEHOLDER|>")
        self.assertEqual(PaddleOCRVLProcessor.image_token_str, "<|IMAGE_PLACEHOLDER|>")

    def test_video_placeholder_same_as_qwen(self):
        """PaddleOCR video placeholder matches Qwen."""
        self.assertEqual(PaddleOCRVLProcessor.video_placeholder, "<|video_pad|>")


if __name__ == "__main__":
    unittest.main()
