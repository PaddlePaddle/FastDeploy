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
from unittest.mock import patch

import numpy as np
from PIL import Image

from fastdeploy.input.v1.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive import (
    AdaptiveImageProcessor,
    ceil_by_factor,
    floor_by_factor,
    is_scaled_image,
    make_batched_images,
    make_batched_videos,
    round_by_factor,
    smart_resize,
)


class TestImagePreprocessorAdaptive(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.processor = AdaptiveImageProcessor(
            min_pixels=56 * 56,
            max_pixels=28 * 28 * 1280,
            patch_size=14,
            temporal_conv_size=2,
            merge_size=2,
        )

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.processor.min_pixels, 56 * 56)
        self.assertEqual(self.processor.max_pixels, 28 * 28 * 1280)
        self.assertEqual(self.processor.patch_size, 14)
        self.assertEqual(self.processor.temporal_conv_size, 2)
        self.assertEqual(self.processor.merge_size, 2)

    def test_set_pixels(self):
        """Test setting pixels with valid and invalid values (lines 205-214)"""
        # Test setting only min_pixels
        self.processor.set_pixels(min_pixels=100, msg="test")
        self.assertEqual(self.processor.min_pixels, 100)
        self.assertEqual(self.processor.size["min_pixels"], 100)

        # Test setting only max_pixels
        self.processor.set_pixels(max_pixels=200, msg="test")
        self.assertEqual(self.processor.max_pixels, 200)
        self.assertEqual(self.processor.size["max_pixels"], 200)

        # Test setting both
        self.processor.set_pixels(min_pixels=150, max_pixels=250, msg="test")
        self.assertEqual(self.processor.min_pixels, 150)
        self.assertEqual(self.processor.max_pixels, 250)
        self.assertEqual(self.processor.size["min_pixels"], 150)
        self.assertEqual(self.processor.size["max_pixels"], 250)

        # Invalid cases
        with self.assertRaises(AssertionError):
            self.processor.set_pixels(min_pixels=-1)
        with self.assertRaises(AssertionError):
            self.processor.set_pixels(max_pixels=0)

    def test_get_smarted_resize(self):
        """Test get_smarted_resize with default and custom pixels"""
        height, width = 224, 224
        # Test with default pixels
        (resized_h, resized_w), (patches_h, patches_w) = self.processor.get_smarted_resize(height, width)
        self.assertIsInstance(resized_h, int)
        self.assertIsInstance(resized_w, int)
        self.assertIsInstance(patches_h, int)
        self.assertIsInstance(patches_w, int)
        # Test with custom pixels
        (resized_h, resized_w), (_, _) = self.processor.get_smarted_resize(
            height, width, min_pixels=100, max_pixels=10000
        )
        self.assertIsInstance(resized_h, int)
        self.assertIsInstance(resized_w, int)

    def test_round_by_factor(self):
        """Test round_by_factor with various cases"""
        self.assertEqual(round_by_factor(100, 28), 112)  # 100/28 ≈ 3.57, round(3.57) = 4, 4*28 = 112
        self.assertEqual(round_by_factor(50, 10), 50)
        self.assertEqual(round_by_factor(55, 10), 60)
        # Edge cases
        self.assertEqual(round_by_factor(0, 14), 0)
        self.assertEqual(round_by_factor(14, 14), 14)
        self.assertEqual(round_by_factor(13, 14), 14)  # Round up
        self.assertEqual(round_by_factor(15, 14), 14)  # Round down

    def test_ceil_by_factor(self):
        """Test ceil_by_factor with various cases"""
        self.assertEqual(ceil_by_factor(100, 28), 112)  # ceil(100/28)*28 = ceil(3.57)*28 = 4*28 = 112
        self.assertEqual(ceil_by_factor(50, 10), 50)
        self.assertEqual(ceil_by_factor(55, 10), 60)
        # Edge cases
        self.assertEqual(ceil_by_factor(0, 14), 0)
        self.assertEqual(ceil_by_factor(14, 14), 14)
        self.assertEqual(ceil_by_factor(13, 14), 14)  # Ceil up
        self.assertEqual(ceil_by_factor(15, 14), 28)  # Ceil up to next multiple

    def test_floor_by_factor(self):
        """Test floor_by_factor with various cases"""
        self.assertEqual(floor_by_factor(100, 28), 84)  # floor(100/28)*28 = floor(3.57)*28 = 3*28 = 84
        self.assertEqual(floor_by_factor(50, 10), 50)
        self.assertEqual(floor_by_factor(55, 10), 50)
        # Edge cases
        self.assertEqual(floor_by_factor(0, 14), 0)
        self.assertEqual(floor_by_factor(14, 14), 14)
        self.assertEqual(floor_by_factor(13, 14), 0)  # Floor down
        self.assertEqual(floor_by_factor(15, 14), 14)  # Floor down to multiple
        self.assertEqual(floor_by_factor(28, 14), 28)  # Exact multiple

    def test_smart_resize(self):
        """Test smart_resize with various scenarios (lines 557-587)"""
        # Basic functionality
        height, width = 224, 224
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertEqual(new_h % 28, 0)
        self.assertEqual(new_w % 28, 0)

        # High aspect ratio (height > width) - tests lines 557-563
        height, width = 10000, 10  # aspect ratio > 200
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertLessEqual(max(new_h, new_w) / min(new_h, new_w), 200)

        # High aspect ratio (width > height) - tests lines 562-563
        height, width = 10, 10000
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertLessEqual(max(new_h, new_w) / min(new_h, new_w), 200)

        # Too large - tests lines 575-578
        height, width = 10000, 10000
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(new_h * new_w, 28 * 28 * 1280)

        # Too small - tests lines 579-582
        height, width = 10, 10
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertGreaterEqual(new_h * new_w, 56 * 56)

        # Exceeds max_pixels with custom parameters
        height, width = 10000, 10000
        max_pixels = 10000
        min_pixels = 1000
        new_h, new_w = smart_resize(height, width, factor=14, min_pixels=min_pixels, max_pixels=max_pixels)
        self.assertLessEqual(new_h * new_w, max_pixels)
        self.assertGreaterEqual(new_h * new_w, min_pixels)

        # Below min_pixels with custom parameters
        height, width = 10, 10
        min_pixels = 10000
        max_pixels = 100000
        new_h, new_w = smart_resize(height, width, factor=14, min_pixels=min_pixels, max_pixels=max_pixels)
        self.assertGreaterEqual(new_h * new_w, min_pixels)
        self.assertLessEqual(new_h * new_w, max_pixels)

        # Invalid result (extreme parameters) - tests lines 584-585
        with self.assertRaises(ValueError):
            smart_resize(1, 1, factor=100000, min_pixels=100, max_pixels=1000)

    def test_is_scaled_image(self):
        """Test is_scaled_image with various image types"""
        # uint8 image
        image = np.array([[0, 255], [128, 200]], dtype=np.uint8)
        self.assertFalse(is_scaled_image(image))
        image = np.random.rand(224, 224, 3).astype(np.uint8) * 255
        self.assertFalse(is_scaled_image(image))

        # Scaled float image (values in [0, 1])
        image = np.array([[0.0, 0.5], [0.3, 1.0]], dtype=np.float32)
        self.assertTrue(is_scaled_image(image))
        image = np.random.rand(224, 224, 3).astype(np.float32) * 0.5
        self.assertTrue(is_scaled_image(image))

        # Unscaled float image (values > 1)
        image = np.array([[0.0, 255.0], [128.0, 300.0]], dtype=np.float32)
        self.assertFalse(is_scaled_image(image))
        image = np.random.rand(224, 224, 3).astype(np.float32) * 255
        self.assertFalse(is_scaled_image(image))

        # Edge cases
        image = np.array([[0.0, 1.0]], dtype=np.float32)
        self.assertTrue(is_scaled_image(image))
        image = np.array([[0.0, 1.1]], dtype=np.float32)
        self.assertFalse(is_scaled_image(image))
        image = np.array([[-0.1, 1.0]], dtype=np.float32)
        self.assertFalse(is_scaled_image(image))

    def test_make_batched_images(self):
        """Test make_batched_images with various input types"""
        # Single image
        img = Image.new("RGB", (224, 224))
        result = make_batched_images(img)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], img)

        # List of images
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = make_batched_images(imgs)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, imgs)

        # Nested list
        imgs = [[Image.new("RGB", (224, 224)) for _ in range(2)] for _ in range(2)]
        result = make_batched_images(imgs)
        self.assertEqual(len(result), 4)  # 2*2 = 4

        # Invalid inputs
        with self.assertRaises(ValueError) as context:
            make_batched_images("invalid")
        self.assertIn("Could not make batched images", str(context.exception))
        with self.assertRaises(ValueError) as context:
            make_batched_images([[1, 2, 3], [4, 5, 6]])
        self.assertIn("Could not make batched images", str(context.exception))

    def test_make_batched_videos(self):
        """Test make_batched_videos with various input types"""
        # List of images
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = make_batched_videos(imgs)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)

        # Single image in list
        img = Image.new("RGB", (224, 224))
        result = make_batched_videos([img])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)

        # Nested list
        imgs = [[Image.new("RGB", (224, 224)) for _ in range(2)] for _ in range(2)]
        result = make_batched_videos(imgs)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)

        # 4D array (single)
        video = np.random.rand(3, 224, 224, 3).astype(np.uint8)
        result = make_batched_videos(video)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)

        # 4D array in list (lines 119-120)
        videos = [np.random.rand(3, 224, 224, 3).astype(np.uint8)]
        result = make_batched_videos(videos)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)

        # Invalid input
        with self.assertRaises(ValueError) as context:
            make_batched_videos("invalid")
        self.assertIn("Could not make batched video", str(context.exception))

    def test_preprocess_images(self):
        """Test preprocess handling images"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        # Verify pixel_values shape
        pixel_values = result["pixel_values"]
        self.assertIsInstance(pixel_values, np.ndarray)

    def test_preprocess_videos(self):
        """Test preprocess handling videos"""
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = self.processor.preprocess(images=None, videos=frames)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_preprocess_invalid_images(self):
        """Test preprocess handling invalid image"""
        with self.assertRaises(ValueError):
            self.processor.preprocess(images="invalid")

    def test_preprocess_with_predetermined_grid_thw(self):
        """Test preprocess using predetermined_grid_thw"""
        img = Image.new("RGB", (224, 224))
        # predetermined_grid_thw should be (h, w) format, not [1, h, w]
        predetermined_grid_thw = [(16, 16)]  # For single image, should be (h, w) tuple
        result = self.processor.preprocess(images=img, predetermined_grid_thw=predetermined_grid_thw)
        self.assertIn("pixel_values", result)

    def test_preprocess_flags(self):
        """Test preprocess with various flags disabled"""
        img = Image.new("RGB", (224, 224))
        # Test without resize
        result = self.processor.preprocess(images=img, do_resize=False)
        self.assertIn("pixel_values", result)
        # Test without rescale
        result = self.processor.preprocess(images=img, do_rescale=False)
        self.assertIn("pixel_values", result)
        # Test without normalize
        result = self.processor.preprocess(images=img, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_custom_mean_std(self):
        """Test preprocess using custom mean and std"""
        img = Image.new("RGB", (224, 224))
        # Test with simple custom mean/std
        result = self.processor.preprocess(images=img, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        self.assertIn("pixel_values", result)
        # Test with ImageNet-style mean/std
        result = self.processor.preprocess(
            images=img, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )
        self.assertIn("pixel_values", result)

    def test_preprocess_do_convert_rgb(self):
        """Test preprocess with do_convert_rgb=True (line 289)"""
        img = Image.new("L", (224, 224))  # Grayscale image
        result = self.processor.preprocess(images=img, do_convert_rgb=True)
        self.assertIn("pixel_values", result)

    def test_preprocess_scaled_image_warning(self):
        """Test warning for scaled image in preprocess (lines 294-298)"""
        # Create a scaled image (values between 0-1)
        img_array = np.random.rand(224, 224, 3).astype(np.float32) * 0.5
        # Use patch to capture warning
        with patch(
            "fastdeploy.input.v1.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive.data_processor_logger"
        ) as mock_logger:
            # Directly call _preprocess, pass scaled image
            self.processor._preprocess(
                [img_array],  # Pass scaled numpy array
                do_rescale=True,
                do_convert_rgb=False,
            )
            # Verify warning is called when is_scaled_image returns True and do_rescale is True
            mock_logger.warning.assert_called()

    def test_preprocess_invalid_images_check(self):
        """Test invalid image check in preprocess (line 464)"""
        # Test invalid image type - need to ensure valid_images returns False
        # Use patch to make valid_images return False, but make_batched_images succeeds
        with patch(
            "fastdeploy.input.v1.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive.valid_images"
        ) as mock_valid:
            mock_valid.return_value = False
            valid_images_list = [Image.new("RGB", (224, 224))]  # Valid image, but valid_images returns False
            with self.assertRaises(ValueError) as context:
                self.processor.preprocess(images=valid_images_list)
            self.assertIn("Invalid image type", str(context.exception))

    def test_preprocess_predetermined_grid_thw_multiple_images(self):
        """Test preprocess with predetermined_grid_thw for multiple images (lines 307-310)"""
        imgs = [Image.new("RGB", (224, 224)) for _ in range(2)]
        predetermined_grid_thw = [(16, 16), (20, 20)]
        result = self.processor.preprocess(images=imgs, predetermined_grid_thw=predetermined_grid_thw)
        self.assertIn("pixel_values", result)

    def test_preprocess_predetermined_grid_thw_length_mismatch(self):
        """Test preprocess with predetermined_grid_thw length mismatch (lines 307-310, 470)"""
        imgs = [Image.new("RGB", (224, 224)) for _ in range(2)]
        predetermined_grid_thw = [(16, 16)]  # Length mismatch - only 1 element for 2 images
        # The function raises IndexError when accessing predetermined_grid_thw[img_idx] with img_idx=1
        with self.assertRaises(IndexError):
            self.processor.preprocess(images=imgs, predetermined_grid_thw=predetermined_grid_thw)

    def test_preprocess_with_input_data_format(self):
        """Test preprocess with input_data_format parameter (lines 299-301)"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        # Test with FIRST
        result = self.processor.preprocess(images=img, input_data_format=ChannelDimension.FIRST)
        self.assertIn("pixel_values", result)
        # Test with None
        result = self.processor.preprocess(images=img, input_data_format=None)
        self.assertIn("pixel_values", result)

    def test_preprocess_do_resize_with_predetermined_grid_thw(self):
        """Test preprocess with do_resize=True and predetermined_grid_thw (lines 314-317)"""
        img = Image.new("RGB", (224, 224))
        predetermined_grid_thw = [(16, 16)]
        result = self.processor.preprocess(images=img, predetermined_grid_thw=predetermined_grid_thw, do_resize=True)
        self.assertIn("pixel_values", result)

    def test_preprocess_videos_with_predetermined_grid_thw(self):
        """Test preprocess videos with predetermined_grid_thw (lines 511)"""
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        predetermined_grid_thw = [(16, 16)] * 4
        result = self.processor.preprocess(images=None, videos=frames, predetermined_grid_thw=predetermined_grid_thw)
        self.assertIn("pixel_values_videos", result)

    def test_preprocess_return_tensors(self):
        """Test preprocess with return_tensors parameter (lines 396, 523)"""
        img = Image.new("RGB", (224, 224))
        # Use string instead of TensorType enum which may not be available
        result = self.processor.preprocess(images=img, return_tensors="np")
        self.assertIn("pixel_values", result)

    def test_preprocess_do_rescale_false_with_scaled_image(self):
        """Test preprocess with do_rescale=False and scaled image (line 335)"""
        # Create a scaled image
        img_array = np.random.rand(224, 224, 3).astype(np.float32) * 0.5  # Values in [0, 0.5]
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        result = self.processor.preprocess(images=img, do_rescale=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_custom_resample(self):
        """Test preprocess with custom resample parameter (line 332)"""
        img = Image.new("RGB", (224, 224))
        from PIL import Image as PILImage

        result = self.processor.preprocess(images=img, resample=PILImage.BILINEAR)
        self.assertIn("pixel_values", result)

    def test_preprocess_custom_rescale_factor(self):
        """Test preprocess with custom rescale_factor (line 336)"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, rescale_factor=1.0 / 128.0)
        self.assertIn("pixel_values", result)

    def test_preprocess_data_format(self):
        """Test preprocess with different data_format values"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        # Test with FIRST
        result = self.processor.preprocess(images=img, data_format=ChannelDimension.FIRST)
        self.assertIn("pixel_values", result)
        # Test with LAST
        result = self.processor.preprocess(images=img, data_format=ChannelDimension.LAST)
        self.assertIn("pixel_values", result)

    def test_preprocess_multiple_images_loop(self):
        """Test preprocess loop with multiple images (lines 312-348, 468-488)"""
        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = self.processor.preprocess(images=images)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        pixel_values = result["pixel_values"]
        self.assertIsInstance(pixel_values, np.ndarray)
        self.assertEqual(len(pixel_values.shape), 2)  # Should be [grid_t * grid_h * grid_w, C * psz * psz]

    def test_preprocess_videos_loop(self):
        """Test preprocess with videos in loop (lines 496-521)"""
        # Test with multiple videos
        videos = [
            [Image.new("RGB", (224, 224)) for _ in range(4)],
            [Image.new("RGB", (224, 224)) for _ in range(4)],
        ]
        result = self.processor.preprocess(images=None, videos=videos)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)
        self.assertIsInstance(result["pixel_values_videos"], np.ndarray)
        # Test with nested list format
        videos = [[Image.new("RGB", (224, 224)) for _ in range(4)] for _ in range(2)]
        result = self.processor.preprocess(images=None, videos=videos)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)
        self.assertIsInstance(result["pixel_values_videos"], np.ndarray)

    def test_preprocess_both_images_and_videos(self):
        """Test preprocess with both images and videos (lines 458-523)"""
        images = [Image.new("RGB", (224, 224))]
        videos = [[Image.new("RGB", (224, 224)) for _ in range(4)]]
        result = self.processor.preprocess(images=images, videos=videos)
        # Due to implementation, only video results are returned when both are provided
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_preprocess_invalid_images_check_list_input(self):
        """Test preprocess with invalid images check (line 464)

        Note: The error is raised by make_batched_images before valid_images check,
        so the error message is different.
        """
        invalid_images = ["not an image", "also not an image"]

        with self.assertRaises(ValueError) as context:
            self.processor.preprocess(images=invalid_images)
        self.assertIn("Could not make batched images", str(context.exception))


if __name__ == "__main__":
    unittest.main()
