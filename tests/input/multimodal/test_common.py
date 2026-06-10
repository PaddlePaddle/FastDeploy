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

"""Unit tests for fastdeploy.input.multimodal.common."""

import unittest

import numpy as np

from fastdeploy.input.multimodal.common import (
    ceil_by_factor,
    floor_by_factor,
    is_scaled_image,
    round_by_factor,
    smart_resize,
    smart_resize_paddleocr,
    smart_resize_qwen,
)


class TestRoundByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(round_by_factor(28, 28), 28)

    def test_round_up(self):
        self.assertEqual(round_by_factor(15, 14), 14)

    def test_round_down(self):
        self.assertEqual(round_by_factor(20, 14), 14)

    def test_zero(self):
        self.assertEqual(round_by_factor(0, 28), 0)


class TestCeilByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(ceil_by_factor(28, 28), 28)

    def test_round_up(self):
        self.assertEqual(ceil_by_factor(29, 28), 56)

    def test_zero(self):
        self.assertEqual(ceil_by_factor(0, 14), 0)


class TestFloorByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(floor_by_factor(56, 28), 56)

    def test_floor_down(self):
        self.assertEqual(floor_by_factor(55, 28), 28)

    def test_zero(self):
        self.assertEqual(floor_by_factor(0, 14), 0)


class TestIsScaledImage(unittest.TestCase):
    def test_uint8_not_scaled(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertFalse(is_scaled_image(img))

    def test_float_in_range_scaled(self):
        img = np.random.rand(10, 10, 3).astype(np.float32)
        self.assertTrue(is_scaled_image(img))

    def test_float_out_of_range_not_scaled(self):
        img = np.array([[0.0, 2.0]], dtype=np.float32)
        self.assertFalse(is_scaled_image(img))

    def test_float_negative_not_scaled(self):
        img = np.array([[-0.1, 0.5]], dtype=np.float32)
        self.assertFalse(is_scaled_image(img))


class TestSmartResizeQwen(unittest.TestCase):
    def test_normal_image(self):
        h, w = smart_resize_qwen(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_high_aspect_ratio_height(self):
        """Height >> width triggers aspect ratio clamping."""
        h, w = smart_resize_qwen(10000, 10, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(max(h, w) / min(h, w), 200)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_high_aspect_ratio_width(self):
        """Width >> height triggers aspect ratio clamping."""
        h, w = smart_resize_qwen(10, 10000, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(max(h, w) / min(h, w), 200)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_too_large_scales_down(self):
        h, w = smart_resize_qwen(10000, 10000, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(h * w, 28 * 28 * 1280)

    def test_too_small_scales_up(self):
        h, w = smart_resize_qwen(10, 10, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertGreaterEqual(h * w, 56 * 56)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            smart_resize_qwen(1, 1, factor=100000, min_pixels=100, max_pixels=1000)


class TestSmartResizePaddleocr(unittest.TestCase):
    def test_normal_image(self):
        h, w = smart_resize_paddleocr(224, 224)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_height_below_factor(self):
        """Height < factor triggers rescale."""
        h, w = smart_resize_paddleocr(10, 100, factor=28)
        self.assertGreaterEqual(h, 28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_width_below_factor(self):
        """Width < factor triggers rescale."""
        h, w = smart_resize_paddleocr(100, 10, factor=28)
        self.assertGreaterEqual(w, 28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_extreme_aspect_ratio_raises(self):
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            smart_resize_paddleocr(6000, 28, factor=28)

    def test_above_max_pixels_scales_down(self):
        h, w = smart_resize_paddleocr(2000, 2000, factor=28, max_pixels=28 * 28 * 100)
        self.assertLessEqual(h * w, 28 * 28 * 100)

    def test_below_min_pixels_scales_up(self):
        h, w = smart_resize_paddleocr(56, 56, factor=28, min_pixels=28 * 28 * 130)
        self.assertGreaterEqual(h * w, 28 * 28 * 130)


class TestSmartResizeDispatcher(unittest.TestCase):
    def test_qwen_variant(self):
        h, w = smart_resize(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280, variant="qwen")
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_paddleocr_variant(self):
        h, w = smart_resize(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280, variant="paddleocr")
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    def test_default_is_qwen(self):
        h1, w1 = smart_resize(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        h2, w2 = smart_resize_qwen(224, 224, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertEqual((h1, w1), (h2, w2))


if __name__ == "__main__":
    unittest.main()
