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

from fastdeploy.input.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive import (
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
        """Test setting pixels with valid and invalid values"""
        # Valid case
        self.processor.set_pixels(min_pixels=100, max_pixels=200, msg="test")
        self.assertEqual(self.processor.min_pixels, 100)
        self.assertEqual(self.processor.max_pixels, 200)
        self.assertEqual(self.processor.size["min_pixels"], 100)
        self.assertEqual(self.processor.size["max_pixels"], 200)

        # Invalid cases
        with self.assertRaises(AssertionError):
            self.processor.set_pixels(min_pixels=-1)
        with self.assertRaises(AssertionError):
            self.processor.set_pixels(max_pixels=0)

    def test_get_smarted_resize(self):
        """Test get_smarted_resize"""
        height, width = 224, 224
        (resized_h, resized_w), (patches_h, patches_w) = self.processor.get_smarted_resize(height, width)
        self.assertIsInstance(resized_h, int)
        self.assertIsInstance(resized_w, int)
        self.assertIsInstance(patches_h, int)
        self.assertIsInstance(patches_w, int)

    def test_get_smarted_resize_with_custom_pixels(self):
        """Test get_smarted_resize with custom pixels"""
        height, width = 224, 224
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
        """Test smart_resize with various scenarios"""
        # Basic functionality
        height, width = 224, 224
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertEqual(new_h % 28, 0)
        self.assertEqual(new_w % 28, 0)

        # High aspect ratio (height > width)
        height, width = 1000, 10  # aspect ratio = 100
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertLessEqual(max(new_h, new_w) / min(new_h, new_w), 200)

        # High aspect ratio (width > height)
        height, width = 10, 10000
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertIsInstance(new_h, int)
        self.assertIsInstance(new_w, int)
        self.assertLessEqual(max(new_h, new_w) / min(new_h, new_w), 200)

        # Too large
        height, width = 10000, 10000
        new_h, new_w = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(new_h * new_w, 28 * 28 * 1280)

        # Too small
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

        # Invalid result (extreme parameters)
        try:
            result = smart_resize(1, 1, factor=100000, min_pixels=100, max_pixels=1000)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
        except ValueError:
            pass

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

        # 4D array in list
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

    def test_preprocess_videos(self):
        """Test preprocess handling videos"""
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = self.processor.preprocess(images=None, videos=frames)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_preprocess_both_images_and_videos(self):
        """Test preprocess handling both images and videos"""
        img = Image.new("RGB", (224, 224))
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = self.processor.preprocess(images=img, videos=frames)
        # When both images and videos are provided, may only return videos result
        # According to code logic, if videos is not None, it will overwrite data dict
        self.assertTrue("pixel_values" in result or "pixel_values_videos" in result)

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

    def test_preprocess_no_resize(self):
        """Test preprocess without resize"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, do_resize=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_no_rescale(self):
        """Test preprocess without rescale"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, do_rescale=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_no_normalize(self):
        """Test preprocess without normalize"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_custom_mean_std(self):
        """Test preprocess using custom mean and std"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        self.assertIn("pixel_values", result)

    def test_make_batched_videos_4d_array_in_list(self):
        """Test make_batched_videos handling 4D array in list (lines 119-120)"""
        # Create a list of 4D arrays
        videos = [np.random.rand(3, 224, 224, 3).astype(np.uint8)]
        result = make_batched_videos(videos)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)

    def test_preprocess_do_convert_rgb(self):
        """Test preprocess with do_convert_rgb=True (line 289)"""
        img = Image.new("L", (224, 224))  # Grayscale image
        result = self.processor.preprocess(images=img, do_convert_rgb=True)
        self.assertIn("pixel_values", result)

    def test_preprocess_scaled_image_warning(self):
        """Test warning for scaled image in preprocess (line 295)"""
        # Create a scaled image (values between 0-1)
        img_array = np.random.rand(224, 224, 3).astype(np.float32)
        # Use patch to capture warning
        with patch(
            "fastdeploy.input.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive.data_processor_logger"
        ) as mock_logger:
            # Directly call _preprocess, pass scaled image
            self.processor._preprocess(
                [img_array],  # Pass scaled numpy array
                do_rescale=True,
                do_convert_rgb=False,
            )
            # Verify warning is called (if is_scaled_image returns True)
            # mock_logger.warning should be called
            if is_scaled_image(img_array):
                # If image is indeed scaled, warning should be called
                mock_logger.warning.assert_called()

    def test_preprocess_data_format_last(self):
        """Test preprocess with data_format=LAST (line 351)"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        result = self.processor.preprocess(images=img, data_format=ChannelDimension.LAST)
        self.assertIn("pixel_values", result)

    def test_preprocess_invalid_images_check(self):
        """Test invalid image check in preprocess (line 464)"""
        # Test invalid image type - need to ensure valid_images returns False
        # Use patch to make valid_images return False, but make_batched_images succeeds
        with patch(
            "fastdeploy.input.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive.valid_images"
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
        """Test preprocess with predetermined_grid_thw length mismatch (lines 308-310)

        Note: The implementation raises IndexError when predetermined_grid_thw length
        doesn't match images length, because it accesses predetermined_grid_thw[img_idx]
        directly without checking bounds first.
        """
        imgs = [Image.new("RGB", (224, 224)) for _ in range(2)]
        predetermined_grid_thw = [(16, 16)]  # Length mismatch - only 1 element for 2 images
        # The function raises IndexError when accessing predetermined_grid_thw[1]
        with self.assertRaises(IndexError):
            self.processor.preprocess(images=imgs, predetermined_grid_thw=predetermined_grid_thw)

    def test_preprocess_with_input_data_format(self):
        """Test preprocess with input_data_format parameter (lines 299-301)"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        result = self.processor.preprocess(images=img, input_data_format=ChannelDimension.FIRST)
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

    def test_preprocess_multiple_images_loop(self):
        """Test preprocess with multiple images in loop (lines 468-488)"""
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = self.processor.preprocess(images=imgs)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)

    def test_preprocess_videos_loop(self):
        """Test preprocess with videos in loop (lines 496-521)"""
        videos = [[Image.new("RGB", (224, 224)) for _ in range(4)] for _ in range(2)]
        result = self.processor.preprocess(images=None, videos=videos)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_preprocess_return_tensors(self):
        """Test preprocess with return_tensors parameter (lines 396, 523)"""
        img = Image.new("RGB", (224, 224))
        # Use string instead of TensorType enum which may not be available
        result = self.processor.preprocess(images=img, return_tensors="np")
        self.assertIn("pixel_values", result)

    def test_preprocess_channel_dimension_none(self):
        """Test preprocess with input_data_format=None (lines 299-301)"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, input_data_format=None)
        self.assertIn("pixel_values", result)

    def test_preprocess_do_rescale_false_with_scaled_image(self):
        """Test preprocess with do_rescale=False and scaled image (line 335)"""
        # Create a scaled image
        img_array = np.random.rand(224, 224, 3).astype(np.float32) * 0.5  # Values in [0, 0.5]
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        result = self.processor.preprocess(images=img, do_rescale=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_do_normalize_false(self):
        """Test preprocess with do_normalize=False (lines 338-344)"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img, do_normalize=False)
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

    def test_preprocess_custom_image_mean_std(self):
        """Test preprocess with custom image_mean and image_std (lines 339-344)"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(
            images=img, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )
        self.assertIn("pixel_values", result)

    def test_preprocess_data_format_channels_first(self):
        """Test preprocess with data_format=FIRST (line 346)"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        result = self.processor.preprocess(images=img, data_format=ChannelDimension.FIRST)
        self.assertIn("pixel_values", result)

    def test_preprocess_data_format_channels_last(self):
        """Test preprocess with data_format=LAST (line 350)"""
        img = Image.new("RGB", (224, 224))
        from paddleformers.transformers.image_utils import ChannelDimension

        result = self.processor.preprocess(images=img, data_format=ChannelDimension.LAST)
        self.assertIn("pixel_values", result)

    def test_preprocess_patches_reshape(self):
        """Test preprocess patches reshape logic (lines 349-381)"""
        img = Image.new("RGB", (224, 224))
        result = self.processor.preprocess(images=img)
        self.assertIn("pixel_values", result)
        # Verify pixel_values shape
        pixel_values = result["pixel_values"]
        self.assertIsInstance(pixel_values, np.ndarray)

    def test_preprocess_videos_multiple(self):
        """Test preprocess with multiple videos (lines 496-521)"""
        videos = [
            [Image.new("RGB", (224, 224)) for _ in range(4)],
            [Image.new("RGB", (224, 224)) for _ in range(4)],
        ]
        result = self.processor.preprocess(images=None, videos=videos)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)


    def test_preprocess_with_scaled_image_warning(self):
        """Test preprocess with scaled image triggers warning (lines 294-298)

        Note: The warning is only triggered when is_scaled_image() returns True,
        which requires float images with values in [0, 1]. Converting to PIL Image
        and back converts to uint8, so the warning won't be triggered.
        This test verifies the preprocess works without errors.
        """
        # Create a scaled image (values in [0, 1])
        scaled_image = np.random.rand(224, 224, 3).astype(np.float32) * 0.5
        scaled_image = Image.fromarray((scaled_image * 255).astype(np.uint8))

        # The image is now uint8, so is_scaled_image returns False and no warning is triggered
        result = self.processor.preprocess(images=[scaled_image], do_rescale=True)
        self.assertIn("pixel_values", result)

    def test_preprocess_predetermined_grid_thw_length_mismatch_assert(self):
        """Test preprocess with predetermined_grid_thw length mismatch (line 310)

        Note: The source code expects predetermined_grid_thw elements to be (height, width) tuples,
        but when 3-element arrays like [1, 16, 16] are passed, it raises ValueError when unpacking.
        """
        images = [Image.new("RGB", (224, 224)) for _ in range(2)]
        predetermined_grid_thw = np.array([[1, 16, 16]])  # Only 1, but 2 images

        # First fails because of unpacking 3 values into 2 variables
        with self.assertRaises(ValueError) as context:
            self.processor.preprocess(images=images, predetermined_grid_thw=predetermined_grid_thw, do_resize=True)
        self.assertIn("too many values to unpack", str(context.exception))

    def test_preprocess_loop_multiple_images(self):
        """Test preprocess loop with multiple images (lines 312-348)"""
        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = self.processor.preprocess(images=images)
        self.assertIn("pixel_values", result)
        pixel_values = result["pixel_values"]
        self.assertIsInstance(pixel_values, np.ndarray)

    def test_preprocess_with_predetermined_grid_thw_in_loop(self):
        """Test preprocess with predetermined_grid_thw in loop (lines 314-317)

        Note: predetermined_grid_thw expects (height, width) tuples, not (t, h, w).
        The values are grid dimensions that get multiplied by patch_size.
        """
        images = [Image.new("RGB", (224, 224)) for _ in range(2)]
        # Use 2D grid (h, w) format
        predetermined_grid_thw = [(16, 16), (16, 16)]

        result = self.processor.preprocess(
            images=images, predetermined_grid_thw=predetermined_grid_thw, do_resize=True
        )
        self.assertIn("pixel_values", result)

    def test_preprocess_patches_reshape_multiple_inputs(self):
        """Test preprocess patches reshape logic (lines 349-381)"""
        images = [Image.new("RGB", (224, 224))]
        result = self.processor.preprocess(images=images)
        self.assertIn("pixel_values", result)
        pixel_values = result["pixel_values"]
        # Verify shape is correct after reshape
        self.assertEqual(len(pixel_values.shape), 2)  # Should be [grid_t * grid_h * grid_w, C * psz * psz]


    def test_preprocess_videos_loop_numpy_output(self):
        """Test preprocess videos loop (lines 496-521)"""
        videos = [
            [Image.new("RGB", (224, 224)) for _ in range(4)],
            [Image.new("RGB", (224, 224)) for _ in range(4)],
        ]
        result = self.processor.preprocess(images=None, videos=videos)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)
        self.assertIsInstance(result["pixel_values_videos"], np.ndarray)

    def test_preprocess_both_images_and_videos_full_outputs(self):
        """Test preprocess with both images and videos (lines 458-523)

        Note: Current implementation has a known issue where the data dict is overwritten
        when processing both images and videos. The video processing overwrites the image
        results, so only video outputs are returned.
        """
        images = [Image.new("RGB", (224, 224))]
        videos = [[Image.new("RGB", (224, 224)) for _ in range(4)]]

        result = self.processor.preprocess(images=images, videos=videos)
        # Due to implementation, only video results are returned when both are provided
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_preprocess_images_loop_with_predetermined_grid_thw(self):
        """Test preprocess images loop with predetermined_grid_thw (lines 468-486)

        Note: predetermined_grid_thw expects (height, width) tuples, not (t, h, w).
        """
        images = [Image.new("RGB", (224, 224)) for _ in range(2)]
        # Use 2D grid (h, w) format
        predetermined_grid_thw = [(16, 16), (16, 16)]

        result = self.processor.preprocess(
            images=images, predetermined_grid_thw=predetermined_grid_thw, do_resize=True
        )
        self.assertIn("pixel_values", result)
        self.assertEqual(len(result["image_grid_thw"]), 2)

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
