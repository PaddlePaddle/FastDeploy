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

"""Unit tests for multimodal image processors (paddleocr, qwen, qwen3, ernie)."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fastdeploy.input.multimodal.image_processors.paddleocr import (
    PaddleOCRImageProcessor,
    adjust_size,
)
from fastdeploy.input.multimodal.image_processors.paddleocr import (
    make_batched_images as paddleocr_make_batched,
)

# ==================================================================
# PaddleOCR ImageProcessor
# ==================================================================


class TestPaddleOCRAdjustSize(unittest.TestCase):
    def test_even_patches(self):
        # 224 // 14 = 16 (even) -> stays 224
        self.assertEqual(adjust_size(224, 14), 224)

    def test_odd_patches(self):
        # 210 // 14 = 15 (odd) -> (15-1)*14 = 196
        self.assertEqual(adjust_size(210, 14), 196)


class TestPaddleOCRMakeBatchedImages(unittest.TestCase):
    def test_single_image(self):
        img = Image.new("RGB", (100, 100))
        result = paddleocr_make_batched(img)
        self.assertEqual(len(result), 1)

    def test_list_of_images(self):
        imgs = [Image.new("RGB", (100, 100)) for _ in range(3)]
        result = paddleocr_make_batched(imgs)
        self.assertEqual(len(result), 3)

    def test_nested_list(self):
        imgs = [[Image.new("RGB", (100, 100)) for _ in range(2)] for _ in range(2)]
        result = paddleocr_make_batched(imgs)
        self.assertEqual(len(result), 4)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            paddleocr_make_batched("not_an_image")


class TestPaddleOCRImageProcessorInit(unittest.TestCase):
    def test_default_init(self):
        proc = PaddleOCRImageProcessor()
        self.assertEqual(proc.patch_size, 14)
        self.assertEqual(proc.temporal_patch_size, 1)
        self.assertEqual(proc.merge_size, 2)
        self.assertTrue(proc.do_resize)
        self.assertTrue(proc.do_rescale)
        self.assertTrue(proc.do_normalize)

    def test_custom_init(self):
        proc = PaddleOCRImageProcessor(patch_size=16, merge_size=4, temporal_patch_size=2)
        self.assertEqual(proc.patch_size, 16)
        self.assertEqual(proc.merge_size, 4)
        self.assertEqual(proc.temporal_patch_size, 2)


class TestPaddleOCRImageProcessorFromPretrained(unittest.TestCase):
    @patch("builtins.open", unittest.mock.mock_open(read_data='{"do_resize": false, "patch_size": 16}'))
    def test_from_pretrained(self):
        proc = PaddleOCRImageProcessor.from_pretrained("/fake/path")
        self.assertFalse(proc.do_resize)
        self.assertEqual(proc.patch_size, 16)


class TestPaddleOCRImageProcessorPreprocess(unittest.TestCase):
    def setUp(self):
        self.proc = PaddleOCRImageProcessor(
            min_pixels=28 * 28 * 4,
            max_pixels=28 * 28 * 100,
        )

    def test_single_image(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img])
        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)
        # pixel_values: [N, C, patch_size, patch_size]
        self.assertEqual(result["pixel_values"].ndim, 4)
        self.assertEqual(result["pixel_values"].shape[1], 3)
        self.assertEqual(result["pixel_values"].shape[2], 14)
        self.assertEqual(result["pixel_values"].shape[3], 14)

    def test_batch_images(self):
        imgs = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]
        result = self.proc.preprocess(images=imgs)
        single = self.proc.preprocess(images=[imgs[0]])
        # batch should have double the patches
        self.assertEqual(result["pixel_values"].shape[0], single["pixel_values"].shape[0] * 2)

    def test_grid_thw_shape(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img])
        grid_thw = result["grid_thw"]
        self.assertEqual(len(grid_thw), 3)
        # t should be 1 for single image with temporal_patch_size=1
        self.assertEqual(grid_thw[0], 1)

    def test_no_resize(self):
        img = Image.new("RGB", (56, 56))
        result = self.proc.preprocess(images=[img], do_resize=False)
        self.assertIn("pixel_values", result)

    def test_no_rescale(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], do_rescale=False)
        self.assertIn("pixel_values", result)

    def test_no_normalize(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_custom_mean_std(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        self.assertIn("pixel_values", result)

    def test_do_convert_rgb_false(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], do_convert_rgb=False)
        self.assertIn("pixel_values", result)

    def test_videos_not_implemented(self):
        img = Image.new("RGB", (224, 224))
        with self.assertRaises(NotImplementedError):
            self.proc.preprocess(images=[img], videos=["video"])


# ==================================================================
# Qwen ImageProcessor
# ==================================================================


class TestQwenImageProcessorInit(unittest.TestCase):
    def test_default_init(self):
        from fastdeploy.input.multimodal.image_processors.qwen import QwenImageProcessor

        proc = QwenImageProcessor()
        self.assertEqual(proc.patch_size, 14)
        self.assertEqual(proc.merge_size, 2)
        self.assertEqual(proc.temporal_patch_size, 2)
        self.assertTrue(proc.do_rescale)
        self.assertTrue(proc.do_normalize)


class TestQwenImageProcessorPreprocess(unittest.TestCase):
    def setUp(self):
        from fastdeploy.input.multimodal.image_processors.qwen import QwenImageProcessor

        self.proc = QwenImageProcessor(min_pixels=4 * 28 * 28, max_pixels=100 * 28 * 28)

    def test_single_image(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img])
        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)
        # pixel_values: [grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size]
        self.assertEqual(result["pixel_values"].ndim, 2)

    def test_grid_thw_values(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img])
        grid_thw = result["grid_thw"]
        self.assertEqual(len(grid_thw), 3)
        t, h, w = grid_thw
        # For single image with temporal_patch_size=2, t should be 1
        self.assertEqual(t, 1)
        self.assertGreater(h, 0)
        self.assertGreater(w, 0)

    def test_video_frames(self):
        """Multiple frames → temporal dimension > 1."""
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = self.proc.preprocess(images=frames)
        grid_thw = result["grid_thw"]
        t = grid_thw[0]
        # 4 frames // temporal_patch_size=2 = 2
        self.assertEqual(t, 2)

    def test_odd_frames_padded(self):
        """Odd number of frames gets padded to next multiple of temporal_patch_size."""
        frames = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = self.proc.preprocess(images=frames)
        grid_thw = result["grid_thw"]
        t = grid_thw[0]
        # 3 frames -> padded to 4 -> t=4//2=2
        self.assertEqual(t, 2)

    def test_no_rescale_no_normalize(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], do_rescale=False, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_rescale_only(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], do_rescale=True, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_invalid_images_raises(self):
        with self.assertRaises(ValueError):
            self.proc.preprocess(images="invalid")

    def test_custom_pixels(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=[img], min_pixels=28 * 28, max_pixels=50 * 28 * 28)
        self.assertIn("pixel_values", result)


# ==================================================================
# Qwen3 ImageProcessor
# ==================================================================


class TestQwen3ImageProcessorInit(unittest.TestCase):
    def test_default_init(self):
        from fastdeploy.input.multimodal.image_processors.qwen3 import (
            Qwen3ImageProcessor,
        )

        proc = Qwen3ImageProcessor()
        self.assertEqual(proc.patch_size, 16)
        self.assertEqual(proc.merge_size, 2)
        self.assertEqual(proc.temporal_patch_size, 2)
        self.assertEqual(proc.image_mean, [0.5, 0.5, 0.5])
        self.assertEqual(proc.image_std, [0.5, 0.5, 0.5])

    def test_preprocess(self):
        from fastdeploy.input.multimodal.image_processors.qwen3 import (
            Qwen3ImageProcessor,
        )

        proc = Qwen3ImageProcessor(min_pixels=32 * 32, max_pixels=100 * 32 * 32)
        img = Image.new("RGB", (224, 224))
        result = proc.preprocess(images=[img])
        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)


# ==================================================================
# Ernie AdaptiveImageProcessor
# ==================================================================


class TestErnieAdaptiveImageProcessorInit(unittest.TestCase):
    def test_default_init(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            AdaptiveImageProcessor,
        )

        proc = AdaptiveImageProcessor()
        self.assertEqual(proc.patch_size, 14)
        self.assertEqual(proc.merge_size, 2)
        self.assertEqual(proc.temporal_conv_size, 2)
        self.assertTrue(proc.do_resize)

    def test_custom_init(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            AdaptiveImageProcessor,
        )

        proc = AdaptiveImageProcessor(min_pixels=100, max_pixels=50000, patch_size=16)
        self.assertEqual(proc.min_pixels, 100)
        self.assertEqual(proc.max_pixels, 50000)
        self.assertEqual(proc.patch_size, 16)


class TestErnieAdaptiveImageProcessorSetPixels(unittest.TestCase):
    def setUp(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            AdaptiveImageProcessor,
        )

        self.proc = AdaptiveImageProcessor()

    def test_set_min_pixels(self):
        self.proc.set_pixels(min_pixels=1000, msg="test")
        self.assertEqual(self.proc.min_pixels, 1000)
        self.assertEqual(self.proc.size["min_pixels"], 1000)

    def test_set_max_pixels(self):
        self.proc.set_pixels(max_pixels=100000, msg="test")
        self.assertEqual(self.proc.max_pixels, 100000)
        self.assertEqual(self.proc.size["max_pixels"], 100000)

    def test_invalid_min_pixels_raises(self):
        with self.assertRaises(AssertionError):
            self.proc.set_pixels(min_pixels=-1, msg="test")

    def test_invalid_max_pixels_raises(self):
        with self.assertRaises(AssertionError):
            self.proc.set_pixels(max_pixels=0, msg="test")


class TestErnieAdaptiveGetSmartedResize(unittest.TestCase):
    def setUp(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            AdaptiveImageProcessor,
        )

        self.proc = AdaptiveImageProcessor(min_pixels=56 * 56, max_pixels=28 * 28 * 1280)

    def test_default_pixels(self):
        (rh, rw), (ph, pw) = self.proc.get_smarted_resize(224, 224)
        self.assertEqual(rh % 28, 0)
        self.assertEqual(rw % 28, 0)
        self.assertEqual(ph, rh // 14)
        self.assertEqual(pw, rw // 14)

    def test_custom_pixels(self):
        (rh, rw), (ph, pw) = self.proc.get_smarted_resize(224, 224, min_pixels=100, max_pixels=10000)
        self.assertEqual(rh % 28, 0)
        self.assertEqual(rw % 28, 0)


class TestErnieMakeBatchedImages(unittest.TestCase):
    def test_single_image(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_images,
        )

        img = Image.new("RGB", (100, 100))
        result = make_batched_images(img)
        self.assertEqual(len(result), 1)

    def test_list_of_images(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_images,
        )

        imgs = [Image.new("RGB", (100, 100)) for _ in range(3)]
        result = make_batched_images(imgs)
        self.assertEqual(len(result), 3)

    def test_nested_list(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_images,
        )

        imgs = [[Image.new("RGB", (100, 100)) for _ in range(2)] for _ in range(2)]
        result = make_batched_images(imgs)
        self.assertEqual(len(result), 4)

    def test_invalid_raises(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_images,
        )

        with self.assertRaises(ValueError):
            make_batched_images("invalid")


class TestErnieMakeBatchedVideos(unittest.TestCase):
    def test_list_of_pil_images(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_videos,
        )

        imgs = [Image.new("RGB", (100, 100)) for _ in range(4)]
        result = make_batched_videos(imgs)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 4)

    def test_nested_list(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_videos,
        )

        videos = [[Image.new("RGB", (100, 100)) for _ in range(2)] for _ in range(3)]
        result = make_batched_videos(videos)
        self.assertEqual(len(result), 3)

    def test_4d_ndarray(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_videos,
        )

        video = np.random.randint(0, 255, (4, 100, 100, 3), dtype=np.uint8)
        result = make_batched_videos(video)
        self.assertEqual(len(result), 1)

    def test_list_of_4d_ndarrays(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_videos,
        )

        videos = [np.random.randint(0, 255, (4, 100, 100, 3), dtype=np.uint8)]
        result = make_batched_videos(videos)
        self.assertEqual(len(result), 1)

    def test_invalid_raises(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            make_batched_videos,
        )

        with self.assertRaises(ValueError):
            make_batched_videos("invalid")


class TestErnieAdaptivePreprocess(unittest.TestCase):
    def setUp(self):
        from fastdeploy.input.multimodal.image_processors.ernie import (
            AdaptiveImageProcessor,
        )

        self.proc = AdaptiveImageProcessor(min_pixels=56 * 56, max_pixels=28 * 28 * 100)

    def test_single_image(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=img)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)

    def test_multiple_images(self):
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        result = self.proc.preprocess(images=imgs)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)

    def test_video_input(self):
        frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        result = self.proc.preprocess(images=None, videos=frames)
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)

    def test_invalid_images_raises(self):
        with self.assertRaises(ValueError):
            self.proc.preprocess(images="invalid_string")

    def test_do_convert_rgb(self):
        img = Image.new("L", (224, 224))
        result = self.proc.preprocess(images=img, do_convert_rgb=True)
        self.assertIn("pixel_values", result)

    def test_predetermined_grid_thw(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=img, predetermined_grid_thw=[(16, 16)])
        self.assertIn("pixel_values", result)

    def test_no_resize(self):
        img = Image.new("RGB", (56, 56))
        result = self.proc.preprocess(images=img, do_resize=False)
        self.assertIn("pixel_values", result)

    def test_no_rescale(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=img, do_rescale=False)
        self.assertIn("pixel_values", result)

    def test_no_normalize(self):
        img = Image.new("RGB", (224, 224))
        result = self.proc.preprocess(images=img, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_both_images_and_videos(self):
        imgs = [Image.new("RGB", (224, 224))]
        videos = [[Image.new("RGB", (224, 224)) for _ in range(4)]]
        result = self.proc.preprocess(images=imgs, videos=videos)
        self.assertIn("pixel_values", result)
        self.assertIn("pixel_values_videos", result)


if __name__ == "__main__":
    unittest.main()
