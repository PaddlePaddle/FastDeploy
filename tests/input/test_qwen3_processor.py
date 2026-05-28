"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from unittest.mock import patch

import numpy as np

from fastdeploy.input.image_processors.qwen3_processor import ImageProcessor


class TestImageProcessorInit(unittest.TestCase):
    """Test ImageProcessor.__init__."""

    def test_default_params(self):
        """Default init sets expected attributes."""
        proc = ImageProcessor()
        self.assertEqual(proc.patch_size, 16)
        self.assertEqual(proc.merge_size, 2)
        self.assertEqual(proc.temporal_patch_size, 2)
        self.assertEqual(proc.min_pixels, 65536)
        self.assertEqual(proc.max_pixels, 16777216)
        self.assertEqual(proc.image_mean, [0.5, 0.5, 0.5])
        self.assertEqual(proc.image_std, [0.5, 0.5, 0.5])
        self.assertAlmostEqual(proc.rescale_factor, 1 / 255)
        self.assertTrue(proc.do_rescale)
        self.assertTrue(proc.do_normalize)

    def test_custom_params(self):
        """Custom params are stored correctly."""
        proc = ImageProcessor(
            patch_size=14,
            merge_size=4,
            temporal_patch_size=4,
            min_pixels=1024,
            max_pixels=4096,
            image_mean=[0.48, 0.46, 0.40],
            image_std=[0.27, 0.26, 0.28],
            rescale_factor=1 / 128,
            do_rescale=False,
            do_normalize=False,
        )
        self.assertEqual(proc.patch_size, 14)
        self.assertEqual(proc.merge_size, 4)
        self.assertEqual(proc.temporal_patch_size, 4)
        self.assertEqual(proc.min_pixels, 1024)
        self.assertEqual(proc.max_pixels, 4096)
        self.assertEqual(proc.image_mean, [0.48, 0.46, 0.40])
        self.assertEqual(proc.image_std, [0.27, 0.26, 0.28])
        self.assertAlmostEqual(proc.rescale_factor, 1 / 128)
        self.assertFalse(proc.do_rescale)
        self.assertFalse(proc.do_normalize)


class TestImageProcessorPreprocess(unittest.TestCase):
    """Test ImageProcessor._preprocess and preprocess."""

    def _make_rgb_image(self, h=64, w=64):
        """Create a random HWC uint8 RGB image."""
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_preprocess_single_image(self):
        """preprocess() handles a single image and returns BatchFeature."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        img = self._make_rgb_image(64, 64)

        result = proc.preprocess(img)

        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)
        grid_thw = result["grid_thw"]
        # grid_t should be 1 for single image, grid_h and grid_w based on resize
        self.assertEqual(grid_thw[0], 1)
        self.assertTrue(grid_thw[1] > 0)
        self.assertTrue(grid_thw[2] > 0)

    def test_preprocess_pixel_values_shape(self):
        """pixel_values shape matches grid dimensions."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        img = self._make_rgb_image(64, 64)

        result = proc.preprocess(img)

        grid_thw = result["grid_thw"]
        pixel_values = result["pixel_values"]
        expected_tokens = int(grid_thw[0] * grid_thw[1] * grid_thw[2])
        self.assertEqual(pixel_values.shape[0], expected_tokens)
        # Each token has C * temporal_patch_size * patch_size * patch_size features
        expected_features = 3 * proc.temporal_patch_size * proc.patch_size * proc.patch_size
        self.assertEqual(pixel_values.shape[1], expected_features)

    def test_preprocess_with_override_params(self):
        """preprocess() respects parameter overrides."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        img = self._make_rgb_image(128, 128)

        result = proc.preprocess(
            img,
            min_pixels=1024,
            max_pixels=4096,
            image_mean=[0.48, 0.46, 0.40],
            image_std=[0.27, 0.26, 0.28],
            rescale_factor=1 / 255,
            do_rescale=True,
            do_normalize=True,
        )

        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)

    def test_preprocess_do_rescale_only(self):
        """preprocess() with do_rescale=True, do_normalize=False."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
            do_rescale=True,
            do_normalize=False,
        )
        img = self._make_rgb_image(32, 32)

        result = proc.preprocess(img)

        pixel_values = result["pixel_values"]
        # Rescaled values should be in [0, 1] range
        self.assertTrue(pixel_values.max() <= 1.0 + 1e-6)
        self.assertTrue(pixel_values.min() >= 0.0 - 1e-6)

    def test_preprocess_no_rescale_no_normalize(self):
        """preprocess() with do_rescale=False, do_normalize=False."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
            do_rescale=False,
            do_normalize=False,
        )
        img = self._make_rgb_image(32, 32)

        result = proc.preprocess(img)

        self.assertIn("pixel_values", result)

    def test_preprocess_resize_needed(self):
        """preprocess() resizes image when dimensions don't match target."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        # Small image that needs resizing (not a multiple of patch_size * merge_size = 32)
        img = self._make_rgb_image(50, 70)

        result = proc.preprocess(img)

        self.assertIn("pixel_values", result)
        grid_thw = result["grid_thw"]
        self.assertEqual(grid_thw[0], 1)

    def test_preprocess_multiple_frames(self):
        """preprocess() handles video frames (multiple images)."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        # 4 frames - evenly divisible by temporal_patch_size=2
        frames = [self._make_rgb_image(32, 32) for _ in range(4)]

        result = proc.preprocess(frames)

        grid_thw = result["grid_thw"]
        # grid_t = 4 / temporal_patch_size = 2
        self.assertEqual(grid_thw[0], 2)

    def test_preprocess_temporal_padding(self):
        """preprocess() pads temporal dimension when not divisible by temporal_patch_size."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        # 3 frames - not divisible by temporal_patch_size=2, should pad to 4
        frames = [self._make_rgb_image(32, 32) for _ in range(3)]

        result = proc.preprocess(frames)

        grid_thw = result["grid_thw"]
        # After padding: 4 frames / temporal_patch_size=2 = 2
        self.assertEqual(grid_thw[0], 2)

    def test_preprocess_invalid_image_raises(self):
        """preprocess() raises ValueError for invalid image type."""
        proc = ImageProcessor()

        with self.assertRaises(ValueError) as ctx:
            proc.preprocess("not_an_image")
        self.assertIn("Invalid image type", str(ctx.exception))

    def test_preprocess_already_scaled_warning(self):
        """preprocess() warns when image appears already scaled."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
            do_rescale=True,
            do_normalize=True,
        )
        # Image with values in [0, 1] (already scaled)
        img = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("fastdeploy.input.image_processors.qwen3_processor.data_processor_logger") as mock_logger:
            proc.preprocess(img)
            mock_logger.warning.assert_called_once()
            self.assertIn("already rescaled", mock_logger.warning.call_args[0][0])


class TestImageProcessorEdgeCases(unittest.TestCase):
    """Test edge cases for _preprocess."""

    def _make_rgb_image(self, h=64, w=64):
        """Create a random HWC uint8 RGB image."""
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_preprocess_infer_input_data_format(self):
        """preprocess() infers input_data_format when set to None."""
        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        img = self._make_rgb_image(32, 32)

        # Pass input_data_format=None to trigger inference
        result = proc.preprocess(img, input_data_format=None)

        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)

    def test_preprocess_channel_last_output(self):
        """preprocess() handles ChannelDimension.LAST output format."""
        from paddleformers.transformers.image_utils import ChannelDimension

        proc = ImageProcessor(
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=65536,
        )
        img = self._make_rgb_image(32, 32)

        result = proc.preprocess(img, data_format=ChannelDimension.LAST)

        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)


class TestImageProcessorRegistration(unittest.TestCase):
    """Test ImageProcessor is registered correctly."""

    def test_registered_in_registry(self):
        """ImageProcessor is registered under QWEN3_VL key."""
        from fastdeploy.input.image_processors.registry import ImageProcessorRegistry
        from fastdeploy.input.mm_model_config import QWEN3_VL

        processor_cls = ImageProcessorRegistry.get(QWEN3_VL)
        self.assertIs(processor_cls, ImageProcessor)


if __name__ == "__main__":
    unittest.main()
