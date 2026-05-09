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

"""Unit tests for Ernie4_5VLProcessor."""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.ernie4_5_vl import Ernie4_5VLProcessor
from fastdeploy.input.multimodal.mm_processor import MMContext, MMItem, TokenizationPath
from fastdeploy.input.utils import IDS_TYPE_FLAG


def _make_ernie_processor(**overrides):
    """Create an Ernie4_5VLProcessor with mocked dependencies."""
    with patch.object(Ernie4_5VLProcessor, "__init__", return_value=None):
        proc = Ernie4_5VLProcessor.__new__(Ernie4_5VLProcessor)

    proc.tokenizer = MagicMock()
    token_map = {
        "<|IMAGE_PLACEHOLDER|>": 204,
        "<|IMAGE_START|>": 200,
        "<|IMAGE_END|>": 201,
        "<|VIDEO_START|>": 202,
        "<|VIDEO_END|>": 203,
    }
    proc.tokenizer.convert_tokens_to_ids.side_effect = lambda x: token_map.get(x, 999)
    proc.tokenizer.tokenize.return_value = ["tok"]

    proc.model_name_or_path = "test-model"
    proc.config = None
    proc._cache = None
    proc.enable_processor_cache = False

    proc.image_placeholder = "<|image@placeholder|>"
    proc.video_placeholder = "<|video@placeholder|>"
    proc.image_token_str = "<|IMAGE_PLACEHOLDER|>"
    proc.video_token_str = "<|IMAGE_PLACEHOLDER|>"
    proc.tokenizer_type = "ernie4_5"

    proc.IMG_START = "<|IMAGE_START|>"
    proc.IMG_END = "<|IMAGE_END|>"
    proc.VID_START = "<|VIDEO_START|>"
    proc.VID_END = "<|VIDEO_END|>"

    proc.image_token_id = 204
    proc.video_token_id = 204

    proc.spatial_conv_size = 2
    proc.temporal_conv_size = 2

    proc.image_min_pixels = 4 * 28 * 28
    proc.image_max_pixels = 6177 * 28 * 28
    proc.video_min_pixels = 299 * 28 * 28
    proc.video_max_pixels = 1196 * 28 * 28
    proc.frames_sample = "leading"

    proc.fps = 2.0
    proc.min_frames = 4
    proc.max_frames = 768
    proc.target_frames = -1

    proc.limit_mm_per_prompt = {"image": 10, "video": 10, "audio": 1}

    # Build token_type_mapping
    mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
    for token in ("<|IMAGE_START|>", "<|IMAGE_END|>", "<|VIDEO_START|>", "<|VIDEO_END|>"):
        mapping[token] = IDS_TYPE_FLAG["image"]
    mapping[204] = IDS_TYPE_FLAG["image"]
    proc.token_type_mapping = mapping

    # Mock image processor
    mock_ip = MagicMock()
    mock_ip.merge_size = 2
    mock_ip.temporal_conv_size = 2
    proc.image_processor = mock_ip

    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ==================================================================
# Test classes
# ==================================================================


class TestErnieInitExtra(unittest.TestCase):
    def test_init_extra_defaults(self):
        proc = _make_ernie_processor()
        self.assertEqual(proc.image_min_pixels, 4 * 28 * 28)
        self.assertEqual(proc.image_max_pixels, 6177 * 28 * 28)
        self.assertEqual(proc.video_min_pixels, 299 * 28 * 28)
        self.assertEqual(proc.video_max_pixels, 1196 * 28 * 28)

    def test_init_extra_custom(self):
        proc = _make_ernie_processor(image_min_pixels=100, image_max_pixels=200)
        self.assertEqual(proc.image_min_pixels, 100)
        self.assertEqual(proc.image_max_pixels, 200)


class TestErnieMakeOutputs(unittest.TestCase):
    def test_no_fps_no_vit(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        self.assertNotIn("fps", outputs)
        self.assertNotIn("vit_seqlen", outputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("mm_hashes", outputs)


class TestErnieComputePositions(unittest.TestCase):
    def setUp(self):
        self.proc = _make_ernie_processor()

    def test_compute_3d_positions_image(self):
        """t=1 image: 1*1*1=1 token (h/2=1, w/2=1, t/2=1 but t==1 so t_eff=1)."""
        pos = self.proc._compute_3d_positions(t=1, h=2, w=2, start_idx=0)
        # t_eff=1 (since t==1), gh=1, gw=1 -> 1 token
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0], [0, 0, 0])

    def test_compute_3d_positions_video(self):
        """t=4 video: t_eff=4//2=2, gh=2, gw=2 -> 2*2*2=8 tokens."""
        pos = self.proc._compute_3d_positions(t=4, h=4, w=4, start_idx=5)
        self.assertEqual(len(pos), 8)
        # First frame tokens
        self.assertEqual(pos[0], [5, 5, 5])  # time_idx=0
        # Second frame tokens (time_idx=1)
        self.assertEqual(pos[4], [6, 5, 5])

    def test_add_text_positions(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        proc.add_text_positions(outputs, 3)
        self.assertEqual(len(outputs["position_ids"]), 3)
        self.assertEqual(outputs["position_ids"][0], [0, 0, 0])
        self.assertEqual(outputs["position_ids"][1], [1, 1, 1])
        self.assertEqual(outputs["position_ids"][2], [2, 2, 2])
        self.assertEqual(outputs["cur_position"], 3)


class TestErniePreprocessImage(unittest.TestCase):
    def setUp(self):
        self.proc = _make_ernie_processor()
        self.proc.image_processor.get_smarted_resize.return_value = ((56, 56), (2, 2))
        self.proc.image_processor.preprocess.return_value = {
            "pixel_values": np.ones((1, 3), dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 2]]),
        }

    def test_raw_image(self):
        outputs = self.proc._make_outputs()
        mock_img = MagicMock()
        mock_img.height = 224
        mock_img.width = 224
        mock_img.convert.return_value = mock_img

        self.proc.preprocess_image(mock_img, outputs, uuid="img_uuid")

        # patches_h=2, patches_w=2, num_tokens = 4 // 4 = 1
        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(outputs["input_ids"], [204])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["image"]])
        self.assertEqual(outputs["num_input_image_tokens"], 1)

    def test_cached_image(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)  # 4 // 4 = 1 token
        meta = {"thw": (1, 2, 2)}

        self.proc.preprocess_cached_image((cached_pixels, meta), outputs, uuid="u1")

        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(outputs["input_ids"], [204])

    def test_cached_image_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3), dtype=np.float32)
        meta = {"thw": (1, 2, 2)}

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_image((cached_pixels, meta), outputs, uuid="u", token_len=999)

    def test_token_len_mismatch(self):
        outputs = self.proc._make_outputs()
        mock_img = MagicMock()
        mock_img.height = 224
        mock_img.width = 224

        with self.assertRaises(ValueError):
            self.proc.preprocess_image(mock_img, outputs, uuid="u", token_len=999)


class TestErniePreprocessVideo(unittest.TestCase):
    def setUp(self):
        self.proc = _make_ernie_processor()
        self.proc.image_processor.get_smarted_resize.return_value = ((56, 56), (2, 2))
        self.proc.image_processor.preprocess.return_value = {
            "pixel_values_videos": np.ones((4, 3), dtype=np.float32),
            "video_grid_thw": np.array([[4, 2, 2]]),
        }

    def test_raw_video(self):
        outputs = self.proc._make_outputs()
        # 4 frames, each is PIL-like
        frames = [MagicMock() for _ in range(4)]
        for f in frames:
            f.height = 224
            f.width = 224
            f.convert.return_value = f

        self.proc.preprocess_video(frames, outputs, uuid="vid_uuid")

        # patches_h=2, patches_w=2, num_frames=4
        # num_tokens = (4*2*2) / (2*2*2) = 16/8 = 2
        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(len(outputs["input_ids"]), 2)
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["video"]] * 2)

    def test_cached_video(self):
        outputs = self.proc._make_outputs()
        # 8 pixels, spatial^2 * temporal = 4*2 = 8, num_tokens = 8//8 = 1
        cached_pixels = np.ones((8, 3), dtype=np.float32)
        meta = {"thw": (2, 2, 2)}

        self.proc.preprocess_cached_video((cached_pixels, meta), outputs, uuid="v1")

        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(len(outputs["input_ids"]), 1)

    def test_cached_video_token_mismatch(self):
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((8, 3), dtype=np.float32)
        meta = {"thw": (2, 2, 2)}

        with self.assertRaises(ValueError):
            self.proc.preprocess_cached_video((cached_pixels, meta), outputs, uuid="v", token_len=999)


class TestErnieMmNumTokens(unittest.TestCase):
    def test_image(self):
        # t=1: t*h*w//4
        result = Ernie4_5VLProcessor.mm_num_tokens([1, 4, 4])
        self.assertEqual(result, 1 * 4 * 4 // 4)

    def test_video(self):
        # t>1: t*h*w//4//2
        result = Ernie4_5VLProcessor.mm_num_tokens([4, 4, 4])
        self.assertEqual(result, 4 * 4 * 4 // 4 // 2)

    def test_list(self):
        result = Ernie4_5VLProcessor.mm_num_tokens([[1, 4, 4], [4, 4, 4]])
        self.assertEqual(result, [4, 8])

    def test_empty(self):
        result = Ernie4_5VLProcessor.mm_num_tokens([])
        self.assertEqual(result, 0)


class TestErniePromptTokenIds2Outputs(unittest.TestCase):
    def setUp(self):
        self.proc = _make_ernie_processor()

    def test_text_only(self):
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
        self.assertEqual(len(outputs["position_ids"]), 3)

    def test_with_processed_image(self):
        """Scans for IMG_START (200) ... IMG_END (201) boundary tokens."""
        cached_pixels = np.ones((4, 3), dtype=np.float32)  # 4//4=1 token
        meta = {"thw": (1, 2, 2)}
        img_item = MMItem(type="image", data=(cached_pixels, meta), uuid="u1")

        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            # text(1) IMG_START(200) placeholder(204) IMG_END(201) text(2)
            prompt_token_ids=[1, 200, 204, 201, 2],
        )
        outputs = self.proc.prompt_token_ids2outputs(ctx)

        self.assertIn(200, outputs["input_ids"])  # IMG_START
        self.assertIn(201, outputs["input_ids"])  # IMG_END
        self.assertEqual(len(outputs["images"]), 1)

    def test_image_placeholder_overflow(self):
        """More IMG_START tokens than images raises ValueError."""
        img_item = MMItem(type="image", data=(np.ones((4, 3)), {"thw": (1, 2, 2)}), uuid="u1")
        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            # Two IMG_START tokens but only 1 image
            prompt_token_ids=[200, 204, 201, 200, 204, 201],
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)

    def test_image_tokens_incomplete(self):
        """Missing IMG_END token raises ValueError."""
        img_item = MMItem(type="image", data=(np.ones((4, 3)), {"thw": (1, 2, 2)}), uuid="u1")
        ctx = MMContext(
            images=[img_item],
            videos=[],
            mm_order=["image"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[200, 204, 204],  # no IMG_END (201)
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)

    def test_video_placeholder_overflow(self):
        vid_item = MMItem(type="video", data=(np.ones((8, 3)), {"thw": (2, 2, 2)}), uuid="v1")
        ctx = MMContext(
            images=[],
            videos=[vid_item],
            mm_order=["video"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[202, 204, 203, 202, 204, 203],  # 2 VID_START but only 1 video
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)

    def test_video_tokens_incomplete(self):
        vid_item = MMItem(type="video", data=(np.ones((8, 3)), {"thw": (2, 2, 2)}), uuid="v1")
        ctx = MMContext(
            images=[],
            videos=[vid_item],
            mm_order=["video"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[202, 204, 204],  # no VID_END (203)
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)

    def test_image_count_mismatch(self):
        """Fewer placeholders than images raises ValueError."""
        img_item = MMItem(type="image", data=(np.ones((4, 3)), {"thw": (1, 2, 2)}), uuid="u1")
        img_item2 = MMItem(type="image", data=(np.ones((4, 3)), {"thw": (1, 2, 2)}), uuid="u2")
        ctx = MMContext(
            images=[img_item, img_item2],
            videos=[],
            mm_order=["image", "image"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[200, 204, 201],  # only 1 placeholder
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)

    def test_video_count_mismatch(self):
        vid_item = MMItem(type="video", data=(np.ones((8, 3)), {"thw": (2, 2, 2)}), uuid="v1")
        vid_item2 = MMItem(type="video", data=(np.ones((8, 3)), {"thw": (2, 2, 2)}), uuid="v2")
        ctx = MMContext(
            images=[],
            videos=[vid_item, vid_item2],
            mm_order=["video", "video"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[202, 204, 203],  # only 1 placeholder
        )
        with self.assertRaises(ValueError):
            self.proc.prompt_token_ids2outputs(ctx)


class TestErnieSetVideoFrameArgs(unittest.TestCase):
    def setUp(self):
        self.proc = _make_ernie_processor()
        self.meta = {"duration": 10.0, "num_of_frame": 300}

    def test_target_frames(self):
        args = {"target_frames": 16, "fps": -1, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        result = self.proc._set_video_frame_args(args, self.meta)
        self.assertEqual(result["target_frames"], 16)

    def test_fps_positive_with_target_raises(self):
        args = {"target_frames": 16, "fps": 2.0, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        with self.assertRaises(ValueError):
            self.proc._set_video_frame_args(args, self.meta)

    def test_below_min_raises(self):
        args = {"target_frames": 2, "fps": -1, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        with self.assertRaises(ValueError):
            self.proc._set_video_frame_args(args, self.meta)

    def test_above_max_raises(self):
        args = {"target_frames": 1000, "fps": -1, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        with self.assertRaises(ValueError):
            self.proc._set_video_frame_args(args, self.meta)

    def test_fps_negative_no_target_raises(self):
        args = {"target_frames": -1, "fps": -1, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        with self.assertRaises(ValueError):
            self.proc._set_video_frame_args(args, self.meta)

    def test_min_greater_than_max_raises(self):
        args = {"target_frames": -1, "fps": 2.0, "min_frames": 100, "max_frames": 10, "frames_sample": "leading"}
        with self.assertRaises(ValueError):
            self.proc._set_video_frame_args(args, self.meta)

    def test_fps_clamp_to_min(self):
        """fps*duration < min_frames -> target_frames set to min_frames."""
        args = {"target_frames": -1, "fps": 0.1, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        result = self.proc._set_video_frame_args(args, self.meta)
        self.assertEqual(result["target_frames"], 4)
        self.assertEqual(result["fps"], -1)

    def test_fps_clamp_to_max(self):
        """fps*duration > max_frames -> target_frames set to max_frames."""
        args = {"target_frames": -1, "fps": 100.0, "min_frames": 4, "max_frames": 768, "frames_sample": "leading"}
        result = self.proc._set_video_frame_args(args, self.meta)
        self.assertEqual(result["target_frames"], 768)
        self.assertEqual(result["fps"], -1)


class TestErnieGetMmMaxTokens(unittest.TestCase):
    def test_returns_image_and_video(self):
        proc = _make_ernie_processor()
        proc.image_processor.get_smarted_resize.return_value = ((56, 56), (14, 14))
        result = proc.get_mm_max_tokens_per_item(seq_len=99999)
        self.assertIn("image", result)
        self.assertIn("video", result)

    def test_capped_by_seq_len(self):
        proc = _make_ernie_processor()
        proc.image_processor.get_smarted_resize.return_value = ((56, 56), (14, 14))
        result = proc.get_mm_max_tokens_per_item(seq_len=10)
        self.assertLessEqual(result["image"], 10)
        self.assertLessEqual(result["video"], 10)


class TestErnieAppendCompletionTokens(unittest.TestCase):
    def test_appends_tokens_and_positions(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()

        proc.append_completion_tokens(outputs, [50, 60, 70])

        self.assertEqual(outputs["input_ids"], [50, 60, 70])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)
        self.assertEqual(len(outputs["position_ids"]), 3)
        self.assertEqual(outputs["position_ids"][0], [0, 0, 0])
        self.assertEqual(outputs["position_ids"][2], [2, 2, 2])


class TestErnieTokenTypeMapping(unittest.TestCase):
    def test_boundary_tokens_mapped(self):
        proc = _make_ernie_processor()
        self.assertEqual(proc.token_type_mapping["<|IMAGE_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(proc.token_type_mapping["<|IMAGE_END|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(proc.token_type_mapping["<|VIDEO_START|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(proc.token_type_mapping["<|VIDEO_END|>"], IDS_TYPE_FLAG["image"])
        self.assertEqual(proc.token_type_mapping[204], IDS_TYPE_FLAG["image"])


class TestErniePackPositionIds(unittest.TestCase):
    def test_pack_position_ids(self):
        proc = _make_ernie_processor()
        outputs = proc._make_outputs()
        outputs["position_ids"] = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        proc.pack_position_ids(outputs)

        self.assertEqual(outputs["position_ids"].dtype, np.int64)
        self.assertEqual(outputs["position_ids"].shape, (3, 3))
        self.assertEqual(outputs["image_patch_id"], 204)


if __name__ == "__main__":
    unittest.main()
