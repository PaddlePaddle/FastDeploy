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

from PIL import Image

from fastdeploy.input.utils.render_timestamp import (
    get_timestamp_for_uniform_frame_extraction,
    render_frame_timestamp,
    render_single_image_with_timestamp,
    timestamp_converting,
)


class TestRenderSingleImageWithTimestamp(unittest.TestCase):
    """Test render_single_image_with_timestamp function."""

    def test_renders_text_on_image(self):
        """Renders text on an image and returns it."""
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        result = render_single_image_with_timestamp(img, "00:01:30.00", 0.1)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (200, 200))
        # The returned image is the same object (modified in place)
        self.assertIs(result, img)

    def test_font_size_scales_with_image(self):
        """Font size is based on min(width, height) * rate."""
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        # Should not raise - font_size = min(100, 50) * 0.2 = 10
        result = render_single_image_with_timestamp(img, "test", 0.2)
        self.assertIsInstance(result, Image.Image)

    def test_large_image(self):
        """Works with larger images."""
        img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        result = render_single_image_with_timestamp(img, "time: 01:23:45.67", 0.05)
        self.assertEqual(result.size, (1920, 1080))

    def test_square_image(self):
        """Works with square images."""
        img = Image.new("RGB", (300, 300), color=(64, 64, 64))
        result = render_single_image_with_timestamp(img, "0", 0.15)
        self.assertEqual(result.size, (300, 300))


class TestTimestampConverting(unittest.TestCase):
    """Test timestamp_converting function."""

    def test_zero_seconds(self):
        """0 seconds converts to 00:00:00.00."""
        self.assertEqual(timestamp_converting(0), "00:00:00.00")

    def test_seconds_only(self):
        """Fractional seconds are formatted correctly."""
        self.assertEqual(timestamp_converting(45.5), "00:00:45.50")

    def test_minutes_and_seconds(self):
        """Minutes and seconds are formatted correctly."""
        self.assertEqual(timestamp_converting(125.25), "00:02:05.25")

    def test_hours_minutes_seconds(self):
        """Hours, minutes and seconds are formatted correctly."""
        self.assertEqual(timestamp_converting(3661.5), "01:01:01.50")

    def test_exact_hour(self):
        """Exact hour boundary."""
        self.assertEqual(timestamp_converting(3600), "01:00:00.00")

    def test_exact_minute(self):
        """Exact minute boundary."""
        self.assertEqual(timestamp_converting(60), "00:01:00.00")

    def test_multiple_hours(self):
        """Multiple hours are formatted correctly."""
        self.assertEqual(timestamp_converting(7200 + 1800 + 30.99), "02:30:30.99")

    def test_small_fraction(self):
        """Small fractional seconds."""
        self.assertEqual(timestamp_converting(0.01), "00:00:00.01")


class TestGetTimestampForUniformFrameExtraction(unittest.TestCase):
    """Test get_timestamp_for_uniform_frame_extraction function."""

    def test_first_frame(self):
        """First frame (frame_id=0) has timestamp 0."""
        result = get_timestamp_for_uniform_frame_extraction(10, 0, 100.0)
        self.assertAlmostEqual(result, 0.0)

    def test_middle_frame(self):
        """Middle frame has proportional timestamp."""
        result = get_timestamp_for_uniform_frame_extraction(10, 5, 100.0)
        self.assertAlmostEqual(result, 50.0)

    def test_last_frame(self):
        """Last frame has proportional timestamp (not quite duration)."""
        result = get_timestamp_for_uniform_frame_extraction(10, 9, 100.0)
        self.assertAlmostEqual(result, 90.0)

    def test_single_frame(self):
        """Single frame extraction with frame_id=0."""
        result = get_timestamp_for_uniform_frame_extraction(1, 0, 60.0)
        self.assertAlmostEqual(result, 0.0)

    def test_float_duration(self):
        """Works with float duration."""
        result = get_timestamp_for_uniform_frame_extraction(4, 2, 10.5)
        self.assertAlmostEqual(result, 5.25)


class TestRenderFrameTimestamp(unittest.TestCase):
    """Test render_frame_timestamp function."""

    def test_renders_formatted_timestamp(self):
        """Renders 'time: HH:MM:SS.ss' on frame."""
        frame = Image.new("RGB", (200, 200), color=(100, 100, 100))
        result = render_frame_timestamp(frame, 90.5, font_rate=0.1)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (200, 200))

    def test_zero_timestamp(self):
        """Renders zero timestamp."""
        frame = Image.new("RGB", (150, 150), color=(0, 0, 0))
        result = render_frame_timestamp(frame, 0.0)
        self.assertIsInstance(result, Image.Image)

    def test_large_timestamp(self):
        """Renders large timestamp (hours)."""
        frame = Image.new("RGB", (300, 200), color=(50, 50, 50))
        result = render_frame_timestamp(frame, 7325.75, font_rate=0.05)
        self.assertIsInstance(result, Image.Image)


if __name__ == "__main__":
    unittest.main()
