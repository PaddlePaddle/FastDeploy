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

Unit tests for get_max_image_tokens / get_max_video_tokens /
get_mm_max_tokens_per_item across all multimodal processor DataProcessors and
their outer VLProcessor wrappers.

Tests run without real model weights: the ImageProcessor and tokenizer are
mocked to avoid loading any files.
"""

import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers to build a minimal mock ImageProcessor matching the real interface
# ---------------------------------------------------------------------------


def _make_mock_image_processor(
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
):
    """Return a MagicMock that behaves like
    fastdeploy.input.image_processors.qwen_processor.ImageProcessor.
    """
    ip = MagicMock()
    ip.min_pixels = min_pixels
    ip.max_pixels = max_pixels
    ip.patch_size = patch_size
    ip.merge_size = merge_size
    ip.temporal_patch_size = temporal_patch_size
    return ip


# ---------------------------------------------------------------------------
# Reference formula (mirrors vllm's get_image_size_with_most_features)
# ---------------------------------------------------------------------------

import math


def _closest_factor_pair(n):
    for d in range(math.isqrt(n), 0, -1):
        if n % d == 0:
            return d, n // d
    return 1, n


def _max_tokens_for_pixels(max_pixels, patch_size, merge_size):
    """Mirrors DataProcessor._max_tokens_for_pixels (vllm-aligned)."""
    unit = patch_size * merge_size
    max_seq_len = max_pixels // (unit * unit)
    for n in range(max_seq_len, 0, -1):
        h, w = _closest_factor_pair(n)
        if w / h <= 200:
            return n
    return 1


def _expected_image_tokens(max_pixels, patch_size, merge_size, seq_len=None):
    n = _max_tokens_for_pixels(max_pixels, patch_size, merge_size)
    return min(n, seq_len) if seq_len is not None else n


def _expected_video_tokens(max_pixels, patch_size, merge_size, temporal_patch_size, max_frames, seq_len=None):
    """Mirrors get_max_video_tokens: frames padded UP to temporal_patch_size."""
    spatial = _max_tokens_for_pixels(max_pixels, patch_size, merge_size)
    padded_frames = max_frames + max_frames % temporal_patch_size
    grid_t = max(padded_frames // temporal_patch_size, 1)
    n = grid_t * spatial
    return min(n, seq_len) if seq_len is not None else n


# ===========================================================================
# Tests for qwen_vl_processor DataProcessor
# ===========================================================================


class TestQwenVLDataProcessor(unittest.TestCase):
    """Tests for fastdeploy.input.qwen_vl_processor.process.DataProcessor"""

    def _make_processor(self, max_pixels=16384 * 28 * 28, max_frames=10):
        """Build a DataProcessor with mocked heavy dependencies."""
        from fastdeploy.input.qwen_vl_processor.process import DataProcessor

        ip = _make_mock_image_processor(max_pixels=max_pixels)

        with (
            patch("fastdeploy.input.qwen_vl_processor.process.ImageProcessor.from_pretrained", return_value=ip),
            patch("paddleformers.transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        ):
            proc = DataProcessor.__new__(DataProcessor)
            proc.image_processor = ip
            proc.spatial_conv_size = ip.merge_size
            proc.temporal_conv_size = ip.temporal_patch_size
            proc.max_frames = max_frames
            proc.min_frames = 1
            return proc

    def test_get_max_image_tokens_default(self):
        """get_max_image_tokens returns a positive integer."""
        proc = self._make_processor()
        result = proc.get_max_image_tokens()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_get_max_image_tokens_matches_formula(self):
        """get_max_image_tokens matches the reference formula."""
        max_pixels = 16384 * 28 * 28
        proc = self._make_processor(max_pixels=max_pixels)
        expected = _expected_image_tokens(
            max_pixels=max_pixels,
            patch_size=proc.image_processor.patch_size,
            merge_size=proc.image_processor.merge_size,
        )
        self.assertEqual(proc.get_max_image_tokens(), expected)

    def test_get_max_image_tokens_seq_len_cap(self):
        """seq_len caps the result when smaller than the formula value."""
        proc = self._make_processor()
        uncapped = proc.get_max_image_tokens()
        cap = uncapped // 2
        self.assertEqual(proc.get_max_image_tokens(seq_len=cap), cap)

    def test_get_max_image_tokens_seq_len_larger(self):
        """seq_len larger than formula value has no effect."""
        proc = self._make_processor()
        uncapped = proc.get_max_image_tokens()
        self.assertEqual(proc.get_max_image_tokens(seq_len=uncapped * 2), uncapped)

    def test_get_max_video_tokens_default(self):
        """get_max_video_tokens returns a positive integer."""
        proc = self._make_processor()
        result = proc.get_max_video_tokens()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_get_max_video_tokens_matches_formula(self):
        """get_max_video_tokens matches the reference formula."""
        max_pixels = 16384 * 28 * 28
        max_frames = 10
        proc = self._make_processor(max_pixels=max_pixels, max_frames=max_frames)
        expected = _expected_video_tokens(
            max_pixels=max_pixels,
            patch_size=proc.image_processor.patch_size,
            merge_size=proc.image_processor.merge_size,
            temporal_patch_size=proc.image_processor.temporal_patch_size,
            max_frames=max_frames,
        )
        self.assertEqual(proc.get_max_video_tokens(), expected)

    def test_get_max_video_tokens_seq_len_cap(self):
        """seq_len caps video tokens correctly."""
        proc = self._make_processor()
        uncapped = proc.get_max_video_tokens()
        cap = uncapped // 3
        self.assertEqual(proc.get_max_video_tokens(seq_len=cap), cap)

    def test_video_tokens_ge_image_tokens(self):
        """A multi-frame video produces at least as many tokens as a single image."""
        proc = self._make_processor(max_frames=4)
        self.assertGreaterEqual(proc.get_max_video_tokens(), proc.get_max_image_tokens())

    def test_get_mm_max_tokens_per_item_returns_dict(self):
        """get_mm_max_tokens_per_item returns a dict with 'image' and 'video'."""
        proc = self._make_processor()
        result = proc.get_mm_max_tokens_per_item()
        self.assertIn("image", result)
        self.assertIn("video", result)

    def test_get_mm_max_tokens_per_item_values_consistent(self):
        """Dict values equal the individual get_max_*_tokens results."""
        proc = self._make_processor()
        result = proc.get_mm_max_tokens_per_item()
        self.assertEqual(result["image"], proc.get_max_image_tokens())
        self.assertEqual(result["video"], proc.get_max_video_tokens())

    def test_get_mm_max_tokens_per_item_seq_len_propagated(self):
        """seq_len is propagated to both sub-calls."""
        proc = self._make_processor()
        seq_len = 512
        result = proc.get_mm_max_tokens_per_item(seq_len=seq_len)
        self.assertLessEqual(result["image"], seq_len)
        self.assertLessEqual(result["video"], seq_len)


# ===========================================================================
# Tests for qwen3_vl_processor DataProcessor
# ===========================================================================


class TestQwen3VLDataProcessor(unittest.TestCase):
    """Tests for fastdeploy.input.qwen3_vl_processor.process.DataProcessor"""

    # VIDEO_MAX_PIXELS from qwen3_vl_processor/process.py
    VIDEO_MAX_PIXELS = 768 * 28 * 28

    def _make_processor(self, max_pixels=16384 * 28 * 28, max_frames=10):
        from fastdeploy.input.qwen3_vl_processor.process import DataProcessor

        ip = _make_mock_image_processor(max_pixels=max_pixels)

        proc = DataProcessor.__new__(DataProcessor)
        proc.image_processor = ip
        proc.spatial_conv_size = ip.merge_size
        proc.temporal_conv_size = ip.temporal_patch_size
        proc.max_frames = max_frames
        proc.min_frames = 1
        return proc

    def test_get_max_image_tokens_matches_formula(self):
        max_pixels = 16384 * 28 * 28
        proc = self._make_processor(max_pixels=max_pixels)
        expected = _expected_image_tokens(
            max_pixels=max_pixels,
            patch_size=proc.image_processor.patch_size,
            merge_size=proc.image_processor.merge_size,
        )
        self.assertEqual(proc.get_max_image_tokens(), expected)

    def test_get_max_video_tokens_uses_video_max_pixels(self):
        """Qwen3-VL video uses VIDEO_MAX_PIXELS, not image max_pixels."""
        proc = self._make_processor(max_pixels=16384 * 28 * 28, max_frames=10)
        expected = _expected_video_tokens(
            max_pixels=self.VIDEO_MAX_PIXELS,
            patch_size=proc.image_processor.patch_size,
            merge_size=proc.image_processor.merge_size,
            temporal_patch_size=proc.image_processor.temporal_patch_size,
            max_frames=proc.max_frames,
        )
        self.assertEqual(proc.get_max_video_tokens(), expected)

    def test_get_max_video_tokens_lt_image_with_large_image_pixels(self):
        """When image max_pixels >> VIDEO_MAX_PIXELS, video tokens < image tokens."""
        proc = self._make_processor(max_pixels=16384 * 28 * 28, max_frames=1)
        # single frame video — still uses VIDEO_MAX_PIXELS so should be less
        self.assertLessEqual(proc.get_max_video_tokens(), proc.get_max_image_tokens())

    def test_seq_len_cap_image(self):
        proc = self._make_processor()
        cap = 100
        self.assertLessEqual(proc.get_max_image_tokens(seq_len=cap), cap)

    def test_seq_len_cap_video(self):
        proc = self._make_processor()
        cap = 100
        self.assertLessEqual(proc.get_max_video_tokens(seq_len=cap), cap)

    def test_get_mm_max_tokens_per_item_structure(self):
        proc = self._make_processor()
        result = proc.get_mm_max_tokens_per_item()
        self.assertIn("image", result)
        self.assertIn("video", result)
        self.assertIsInstance(result["image"], int)
        self.assertIsInstance(result["video"], int)

    def test_get_mm_max_tokens_per_item_consistency(self):
        proc = self._make_processor()
        result = proc.get_mm_max_tokens_per_item(seq_len=4096)
        self.assertEqual(result["image"], proc.get_max_image_tokens(seq_len=4096))
        self.assertEqual(result["video"], proc.get_max_video_tokens(seq_len=4096))


# ===========================================================================
# Tests for QwenVLProcessor wrapper (outer processor)
# ===========================================================================


class TestQwenVLProcessorWrapper(unittest.TestCase):
    """Tests that QwenVLProcessor.get_mm_max_tokens_per_item delegates correctly."""

    def _make_outer_processor(self, img_tokens=1280, vid_tokens=8192):
        """Build a QwenVLProcessor whose inner DataProcessor is mocked."""
        from fastdeploy.input.qwen_vl_processor.qwen_vl_processor import QwenVLProcessor

        inner_proc = MagicMock()
        inner_proc.get_mm_max_tokens_per_item.return_value = {
            "image": img_tokens,
            "video": vid_tokens,
        }
        outer = QwenVLProcessor.__new__(QwenVLProcessor)
        outer.processor = inner_proc
        return outer

    def test_delegates_to_inner_processor(self):
        outer = self._make_outer_processor(img_tokens=1280, vid_tokens=8192)
        result = outer.get_mm_max_tokens_per_item()
        self.assertEqual(result["image"], 1280)
        self.assertEqual(result["video"], 8192)

    def test_seq_len_forwarded(self):
        outer = self._make_outer_processor()
        outer.get_mm_max_tokens_per_item(seq_len=4096)
        outer.processor.get_mm_max_tokens_per_item.assert_called_once_with(4096)

    def test_returns_dict(self):
        outer = self._make_outer_processor()
        result = outer.get_mm_max_tokens_per_item()
        self.assertIsInstance(result, dict)


# ===========================================================================
# Tests for Qwen3VLProcessor wrapper (outer processor)
# ===========================================================================


class TestQwen3VLProcessorWrapper(unittest.TestCase):
    """Tests that Qwen3VLProcessor.get_mm_max_tokens_per_item delegates correctly."""

    def _make_outer_processor(self, img_tokens=2048, vid_tokens=4096):
        from fastdeploy.input.qwen3_vl_processor.qwen3_vl_processor import (
            Qwen3VLProcessor,
        )

        inner_proc = MagicMock()
        inner_proc.get_mm_max_tokens_per_item.return_value = {
            "image": img_tokens,
            "video": vid_tokens,
        }
        outer = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
        outer.processor = inner_proc
        return outer

    def test_delegates_to_inner_processor(self):
        outer = self._make_outer_processor(img_tokens=2048, vid_tokens=4096)
        result = outer.get_mm_max_tokens_per_item()
        self.assertEqual(result["image"], 2048)
        self.assertEqual(result["video"], 4096)

    def test_seq_len_forwarded(self):
        outer = self._make_outer_processor()
        outer.get_mm_max_tokens_per_item(seq_len=8192)
        outer.processor.get_mm_max_tokens_per_item.assert_called_once_with(8192)

    def test_returns_dict(self):
        outer = self._make_outer_processor()
        result = outer.get_mm_max_tokens_per_item()
        self.assertIsInstance(result, dict)


# ===========================================================================
# Cross-processor sanity checks
# ===========================================================================


class TestCrossProcessorSanity(unittest.TestCase):
    """Verify that QwenVL and Qwen3VL DataProcessors agree on the formula
    when given identical ImageProcessor configs."""

    def _make_qwen_vl(self, **kw):
        from fastdeploy.input.qwen_vl_processor.process import DataProcessor as QDP

        ip = _make_mock_image_processor(**kw)
        proc = QDP.__new__(QDP)
        proc.image_processor = ip
        proc.spatial_conv_size = ip.merge_size
        proc.temporal_conv_size = ip.temporal_patch_size
        proc.max_frames = 8
        proc.min_frames = 1
        return proc

    def _make_qwen3_vl(self, **kw):
        from fastdeploy.input.qwen3_vl_processor.process import DataProcessor as Q3DP

        ip = _make_mock_image_processor(**kw)
        proc = Q3DP.__new__(Q3DP)
        proc.image_processor = ip
        proc.spatial_conv_size = ip.merge_size
        proc.temporal_conv_size = ip.temporal_patch_size
        proc.max_frames = 8
        proc.min_frames = 1
        return proc

    def test_image_tokens_same_config(self):
        """With identical image_processor config, image token counts match."""
        cfg = dict(max_pixels=4096 * 28 * 28, patch_size=14, merge_size=2)
        qwen = self._make_qwen_vl(**cfg)
        qwen3 = self._make_qwen3_vl(**cfg)
        self.assertEqual(qwen.get_max_image_tokens(), qwen3.get_max_image_tokens())

    def test_qwen3_video_uses_tighter_pixel_bound(self):
        """Qwen3VL video tokens <= QwenVL video tokens (tighter pixel bound)."""
        cfg = dict(max_pixels=16384 * 28 * 28, patch_size=14, merge_size=2, temporal_patch_size=2)
        qwen = self._make_qwen_vl(**cfg)
        qwen3 = self._make_qwen3_vl(**cfg)
        # Qwen3 uses VIDEO_MAX_PIXELS=768*28*28 vs QwenVL's image max_pixels
        self.assertLessEqual(qwen3.get_max_video_tokens(), qwen.get_max_video_tokens())


# ===========================================================================
# Alignment tests: verify our formula matches vllm exactly
# ===========================================================================


class TestVllmAlignment(unittest.TestCase):
    """Verify that our implementation matches vllm's formula on known cases."""

    def _make_qwen_vl(self, max_pixels, max_frames):
        from fastdeploy.input.qwen_vl_processor.process import DataProcessor as QDP

        ip = _make_mock_image_processor(max_pixels=max_pixels)
        proc = QDP.__new__(QDP)
        proc.image_processor = ip
        proc.max_frames = max_frames
        proc.min_frames = 1
        return proc

    def _make_qwen3_vl(self, max_pixels, max_frames):
        from fastdeploy.input.qwen3_vl_processor.process import DataProcessor as Q3DP

        ip = _make_mock_image_processor(max_pixels=max_pixels)
        proc = Q3DP.__new__(Q3DP)
        proc.image_processor = ip
        proc.max_frames = max_frames
        proc.min_frames = 1
        return proc

    # --- image token tests for several non-square max_pixels values ---

    def test_image_tokens_non_square_1280(self):
        """1280*28*28 is not a perfect square of unit — old formula underestimated."""
        max_pixels = 1280 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=1)
        expected = _expected_image_tokens(max_pixels, patch_size=14, merge_size=2)
        self.assertEqual(proc.get_max_image_tokens(), expected)
        self.assertEqual(expected, 1280)  # vllm ground truth

    def test_image_tokens_non_square_2048(self):
        max_pixels = 2048 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=1)
        expected = _expected_image_tokens(max_pixels, patch_size=14, merge_size=2)
        self.assertEqual(proc.get_max_image_tokens(), expected)
        self.assertEqual(expected, 2048)

    def test_image_tokens_non_square_1000(self):
        max_pixels = 1000 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=1)
        expected = _expected_image_tokens(max_pixels, patch_size=14, merge_size=2)
        self.assertEqual(proc.get_max_image_tokens(), expected)
        self.assertEqual(expected, 1000)

    def test_image_tokens_default_16384(self):
        """Default max_pixels=16384*28*28 is a perfect square — both formulas agree."""
        max_pixels = 16384 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=1)
        self.assertEqual(proc.get_max_image_tokens(), 16384)

    # --- video temporal padding tests ---

    def test_video_odd_frames_padded_up(self):
        """Odd frame counts are padded UP (not truncated down) to temporal_patch_size."""
        # max_frames=9, temporal_patch_size=2 → padded=10, grid_t=5  (not 4)
        max_pixels = 1280 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=9)
        expected = _expected_video_tokens(
            max_pixels,
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            max_frames=9,
        )
        self.assertEqual(proc.get_max_video_tokens(), expected)
        # Verify padding direction: padded_frames=10, grid_t=5, not 4
        spatial = _max_tokens_for_pixels(max_pixels, 14, 2)
        self.assertEqual(expected, 5 * spatial)

    def test_video_exact_multiple_frames(self):
        """Even frame count (exact multiple) should give grid_t = max_frames / temporal."""
        max_pixels = 1280 * 28 * 28
        proc = self._make_qwen_vl(max_pixels=max_pixels, max_frames=10)
        expected = _expected_video_tokens(
            max_pixels,
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            max_frames=10,
        )
        self.assertEqual(proc.get_max_video_tokens(), expected)
        spatial = _max_tokens_for_pixels(max_pixels, 14, 2)
        self.assertEqual(expected, 5 * spatial)

    def test_qwen3_video_non_square_pixels(self):
        """Qwen3-VL video uses VIDEO_MAX_PIXELS=768*28*28, also non-square."""
        from fastdeploy.input.qwen3_vl_processor.process import VIDEO_MAX_PIXELS

        proc = self._make_qwen3_vl(max_pixels=16384 * 28 * 28, max_frames=10)
        expected = _expected_video_tokens(
            VIDEO_MAX_PIXELS,
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            max_frames=10,
        )
        self.assertEqual(proc.get_max_video_tokens(), expected)
        self.assertEqual(expected, 5 * _max_tokens_for_pixels(VIDEO_MAX_PIXELS, 14, 2))


# ===========================================================================
# Tests for ernie4_5_vl_processor DataProcessor
# ===========================================================================


def _make_mock_adaptive_image_preprocessor(
    patch_size: int = 28,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    image_max_pixels: int = 6177 * 28 * 28,
    video_max_pixels: int = 1196 * 28 * 28,
):
    """Return a mock AdaptiveImageProcessor matching ERNIE4.5-VL's interface."""
    ip = MagicMock()
    ip.patch_size = patch_size
    ip.merge_size = merge_size
    ip.temporal_patch_size = temporal_patch_size

    def _get_smarted_resize(height, width, min_pixels, max_pixels):
        """Minimal resize that preserves aspect ratio within pixel budget."""

        area = height * width
        if area > max_pixels:
            scale = (max_pixels / area) ** 0.5
            height = int(height * scale // patch_size) * patch_size
            width = int(width * scale // patch_size) * patch_size
        elif area < min_pixels:
            scale = (min_pixels / area) ** 0.5
            height = max(int(height * scale // patch_size) * patch_size, patch_size)
            width = max(int(width * scale // patch_size) * patch_size, patch_size)
        patches_h = height // patch_size
        patches_w = width // patch_size
        return (height, width), (patches_h, patches_w)

    ip.get_smarted_resize.side_effect = _get_smarted_resize
    return ip


def _expected_ernie45_image_tokens(image_preprocessor, spatial_conv_size, image_min_pixels, image_max_pixels, seq_len):
    """Reference: mirrors ERNIE4.5-VL DataProcessor.get_max_image_tokens."""
    from fastdeploy.input.utils import MAX_IMAGE_DIMENSION

    _, (patches_h, patches_w) = image_preprocessor.get_smarted_resize(
        height=MAX_IMAGE_DIMENSION,
        width=MAX_IMAGE_DIMENSION,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    # Second call with same target
    _, (patches_h, patches_w) = image_preprocessor.get_smarted_resize(
        height=patches_h * image_preprocessor.patch_size,
        width=patches_w * image_preprocessor.patch_size,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    num_tokens = (patches_h * patches_w) // (spatial_conv_size**2)
    return min(num_tokens, seq_len)


def _expected_ernie45_video_tokens(
    image_preprocessor,
    spatial_conv_size,
    temporal_conv_size,
    image_min_pixels,
    image_max_pixels,
    video_min_pixels,
    video_max_pixels,
    max_frames,
    seq_len,
):
    """Reference: mirrors ERNIE4.5-VL DataProcessor.get_max_video_tokens (fixed)."""
    from fastdeploy.input.utils import MAX_IMAGE_DIMENSION

    # get_image_size_with_most_features uses image pixels
    resized_h, resized_w = image_preprocessor.get_smarted_resize(
        height=MAX_IMAGE_DIMENSION,
        width=MAX_IMAGE_DIMENSION,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )[0]
    _, (patches_h, patches_w) = image_preprocessor.get_smarted_resize(
        height=resized_h,
        width=resized_w,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
    )
    # max_frames is the temporal dimension (whole video)
    num_tokens = (max_frames * patches_h * patches_w) // (spatial_conv_size**2 * temporal_conv_size)
    return min(num_tokens, seq_len)


class TestErnie45VLDataProcessor(unittest.TestCase):
    """Tests for fastdeploy.input.ernie4_5_vl_processor.process.DataProcessor"""

    DEFAULT_IMAGE_MAX_PIXELS = 6177 * 28 * 28
    DEFAULT_VIDEO_MAX_PIXELS = 1196 * 28 * 28
    DEFAULT_IMAGE_MIN_PIXELS = 4 * 28 * 28
    DEFAULT_VIDEO_MIN_PIXELS = 299 * 28 * 28

    def _make_processor(
        self, max_frames=180, spatial_conv_size=2, temporal_conv_size=2, image_max_pixels=None, video_max_pixels=None
    ):
        from fastdeploy.input.ernie4_5_vl_processor.process import DataProcessor

        ip = _make_mock_adaptive_image_preprocessor(
            image_max_pixels=image_max_pixels or self.DEFAULT_IMAGE_MAX_PIXELS,
            video_max_pixels=video_max_pixels or self.DEFAULT_VIDEO_MAX_PIXELS,
            merge_size=spatial_conv_size,
            temporal_patch_size=temporal_conv_size,
        )

        proc = DataProcessor.__new__(DataProcessor)
        proc.image_preprocessor = ip
        proc.spatial_conv_size = spatial_conv_size
        proc.temporal_conv_size = temporal_conv_size
        proc.image_min_pixels = self.DEFAULT_IMAGE_MIN_PIXELS
        proc.image_max_pixels = image_max_pixels or self.DEFAULT_IMAGE_MAX_PIXELS
        proc.video_min_pixels = self.DEFAULT_VIDEO_MIN_PIXELS
        proc.video_max_pixels = video_max_pixels or self.DEFAULT_VIDEO_MAX_PIXELS
        proc.max_frames = max_frames
        return proc

    def test_get_max_image_tokens_positive(self):
        """get_max_image_tokens returns a positive integer."""
        proc = self._make_processor()
        result = proc.get_max_image_tokens(seq_len=32768)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_get_max_image_tokens_seq_len_cap(self):
        """seq_len caps the result."""
        proc = self._make_processor()
        result = proc.get_max_image_tokens(seq_len=10)
        self.assertLessEqual(result, 10)

    def test_get_max_video_tokens_includes_frames(self):
        """Video tokens must scale with max_frames (bug check: frames must be included)."""
        proc_few = self._make_processor(max_frames=1)
        proc_many = self._make_processor(max_frames=10)
        few = proc_few.get_max_video_tokens(seq_len=100000)
        many = proc_many.get_max_video_tokens(seq_len=100000)
        # With 10x more frames, token count must be larger
        self.assertGreater(many, few)

    def test_get_max_video_tokens_proportional_to_frames(self):
        """With same spatial budget, doubling frames doubles video tokens."""
        proc_10 = self._make_processor(max_frames=10)
        proc_20 = self._make_processor(max_frames=20)
        t10 = proc_10.get_max_video_tokens(seq_len=1000000)
        t20 = proc_20.get_max_video_tokens(seq_len=1000000)
        self.assertEqual(t20, t10 * 2)

    def test_get_max_video_tokens_seq_len_cap(self):
        """seq_len caps video tokens correctly."""
        proc = self._make_processor()
        result = proc.get_max_video_tokens(seq_len=100)
        self.assertLessEqual(result, 100)

    def test_get_mm_max_tokens_per_item_structure(self):
        """Returns dict with 'image' and 'video' keys."""
        proc = self._make_processor()
        result = proc.get_mm_max_tokens_per_item(seq_len=32768)
        self.assertIn("image", result)
        self.assertIn("video", result)

    def test_get_mm_max_tokens_per_item_consistency(self):
        """Dict values match individual get_max_*_tokens results."""
        proc = self._make_processor()
        seq_len = 32768
        result = proc.get_mm_max_tokens_per_item(seq_len=seq_len)
        self.assertEqual(result["image"], proc.get_max_image_tokens(seq_len=seq_len))
        self.assertEqual(result["video"], proc.get_max_video_tokens(seq_len=seq_len))

    def test_video_tokens_gt_image_tokens_with_many_frames(self):
        """Multi-frame video should produce more tokens than a single image
        when video and image share the same pixel budget."""
        # Use identical pixel budget for image and video so spatial resolution is equal,
        # then more frames means more tokens.
        shared_pixels = self.DEFAULT_VIDEO_MAX_PIXELS
        proc = self._make_processor(max_frames=10, image_max_pixels=shared_pixels, video_max_pixels=shared_pixels)
        img_tokens = proc.get_max_image_tokens(seq_len=1000000)
        vid_tokens = proc.get_max_video_tokens(seq_len=1000000)
        self.assertGreater(vid_tokens, img_tokens)


if __name__ == "__main__":
    unittest.main()
