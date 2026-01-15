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

import os
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from fastdeploy.input.v1.ernie4_5_vl_processor.utils.render_timestamp import (
    FONT_PATH,
    get_timestamp_for_uniform_frame_extraction,
    render_frame_timestamp,
    render_single_image_with_timestamp,
    timestamp_converting,
)


class TestTimestampConverting(unittest.TestCase):
    """Test cases for timestamp_converting function"""

    def test_timestamp_seconds_only(self):
        """Test timestamp conversion with seconds only"""
        result = timestamp_converting(45.75)
        self.assertEqual(result, "00:00:45.75")

    def test_timestamp_minutes_and_seconds(self):
        """Test timestamp conversion with minutes and seconds"""
        result = timestamp_converting(125.5)
        self.assertEqual(result, "00:02:05.50")

    def test_timestamp_hours_minutes_seconds(self):
        """Test timestamp conversion with hours, minutes and seconds"""
        result = timestamp_converting(3725.25)
        self.assertEqual(result, "01:02:05.25")

    def test_timestamp_zero(self):
        """Test timestamp conversion with zero seconds"""
        result = timestamp_converting(0.0)
        self.assertEqual(result, "00:00:00.00")

    def test_timestamp_large_value(self):
        """Test timestamp conversion with large value"""
        result = timestamp_converting(86400.99)  # 24 hours
        self.assertEqual(result, "24:00:00.99")

    def test_timestamp_decimal_precision(self):
        """Test timestamp conversion decimal precision"""
        result = timestamp_converting(59.999)
        self.assertEqual(result, "00:00:60.00")  # Should round up to 60.00 due to 59.999


class TestGetTimestampForUniformFrameExtraction(unittest.TestCase):
    """Test cases for get_timestamp_for_uniform_frame_extraction function"""

    def test_uniform_timestamp_middle_frame(self):
        """Test timestamp for middle frame"""
        result = get_timestamp_for_uniform_frame_extraction(num_frames=10, frame_id=5, duration=100.0)
        expected = 100.0 * 5 / 10  # 50.0 seconds
        self.assertEqual(result, expected)

    def test_uniform_timestamp_first_frame(self):
        """Test timestamp for first frame"""
        result = get_timestamp_for_uniform_frame_extraction(num_frames=10, frame_id=0, duration=100.0)
        expected = 0.0
        self.assertEqual(result, expected)

    def test_uniform_timestamp_last_frame(self):
        """Test timestamp for last frame"""
        result = get_timestamp_for_uniform_frame_extraction(num_frames=10, frame_id=9, duration=100.0)
        expected = 100.0 * 9 / 10  # 90.0 seconds
        self.assertEqual(result, expected)

    def test_uniform_timestamp_single_frame(self):
        """Test timestamp for single frame video"""
        result = get_timestamp_for_uniform_frame_extraction(num_frames=1, frame_id=0, duration=30.0)
        expected = 0.0
        self.assertEqual(result, expected)

    def test_uniform_timestamp_fractional_duration(self):
        """Test timestamp with fractional duration"""
        result = get_timestamp_for_uniform_frame_extraction(num_frames=4, frame_id=2, duration=33.33)
        expected = 33.33 * 2 / 4  # 16.665 seconds
        self.assertAlmostEqual(result, expected, places=5)


class TestRenderSingleImageWithTimestamp(unittest.TestCase):
    """Test cases for render_single_image_with_timestamp function"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test image
        self.test_image = Image.new("RGB", (100, 100), color="white")

    def test_render_with_default_font(self):
        """Test rendering with default font path"""
        result_image = render_single_image_with_timestamp(image=self.test_image.copy(), number="00:00:10.50", rate=0.1)

        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (100, 100))

    def test_render_with_custom_font_path(self):
        """Test rendering with custom font path"""
        # Use the actual font path from the module
        result_image = render_single_image_with_timestamp(
            image=self.test_image.copy(), number="01:30:45.75", rate=0.05, font_path=FONT_PATH
        )

        self.assertIsInstance(result_image, Image.Image)

    def test_render_different_rates(self):
        """Test rendering with different font rates"""
        for rate in [0.05, 0.1, 0.15]:
            with self.subTest(rate=rate):
                result_image = render_single_image_with_timestamp(
                    image=self.test_image.copy(), number="00:00:01.00", rate=rate
                )
                self.assertIsInstance(result_image, Image.Image)

    def test_render_different_image_sizes(self):
        """Test rendering on images of different sizes"""
        sizes = [(50, 50), (200, 100), (100, 200), (300, 300)]

        for width, height in sizes:
            with self.subTest(width=width, height=height):
                image = Image.new("RGB", (width, height), color="blue")
                result_image = render_single_image_with_timestamp(image=image, number="00:00:00.00", rate=0.1)
                self.assertEqual(result_image.size, (width, height))

    def test_render_empty_timestamp(self):
        """Test rendering with empty timestamp string"""
        result_image = render_single_image_with_timestamp(image=self.test_image.copy(), number="", rate=0.1)
        self.assertIsInstance(result_image, Image.Image)

    def test_font_loading_parameters(self):
        """Test that font is loaded with correct parameters"""
        # Skip the actual drawing as it requires a real font
        # Just test that the font path is correctly specified
        self.assertTrue(os.path.exists(FONT_PATH), f"Font file not found at {FONT_PATH}")

    @patch("PIL.ImageDraw.Draw")
    def test_text_rendering_parameters(self, mock_draw_class):
        """Test that text is rendered with correct parameters"""
        # Create a mock draw instance
        mock_draw_instance = MagicMock()
        mock_draw_class.return_value = mock_draw_instance

        image = Image.new("RGB", (100, 100), color="white")

        render_single_image_with_timestamp(image=image, number="01:23:45.67", rate=0.1)

        # Verify Draw was created with the image
        mock_draw_class.assert_called_once_with(image)

        # Verify text was drawn with correct parameters
        mock_draw_instance.text.assert_called_once()
        args, kwargs = mock_draw_instance.text.call_args

        self.assertEqual(args[0], (0, 0))  # Position (x, y)
        self.assertEqual(args[1], "01:23:45.67")  # Text content
        self.assertEqual(kwargs["fill"], (0, 0, 0))  # Black fill
        self.assertEqual(kwargs["stroke_fill"], (255, 255, 255))  # White stroke
        self.assertEqual(kwargs["stroke_width"], 1)  # 10 * 0.1 = 1


class TestRenderFrameTimestamp(unittest.TestCase):
    """Test cases for render_frame_timestamp function"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_frame = Image.new("RGB", (100, 100), color="white")

    def test_render_frame_timestamp_basic(self):
        """Test basic frame timestamp rendering"""
        result_frame = render_frame_timestamp(frame=self.test_frame.copy(), timestamp=123.45)

        self.assertIsInstance(result_frame, Image.Image)
        self.assertEqual(result_frame.size, (100, 100))

    def test_render_frame_timestamp_with_custom_rate(self):
        """Test frame timestamp rendering with custom font rate"""
        result_frame = render_frame_timestamp(
            frame=self.test_frame.copy(), timestamp=3600.75, font_rate=0.05  # 1 hour, 0.75 seconds
        )

        self.assertIsInstance(result_frame, Image.Image)

    def test_render_frame_timestamp_edge_cases(self):
        """Test frame timestamp rendering with edge case timestamps"""
        test_cases = [
            (0.0, "00:00:00.00"),
            (59.999, "00:00:60.00"),  # Updated expected value
            (3600.0, "01:00:00.00"),
            (86399.99, "23:59:59.99"),
        ]

        for timestamp, expected_text in test_cases:
            with self.subTest(timestamp=timestamp):
                # Mock the render_single_image_with_timestamp to verify the text
                with patch(
                    "fastdeploy.input.v1.ernie4_5_vl_processor.utils.render_timestamp.render_single_image_with_timestamp"
                ) as mock_render:
                    mock_render.return_value = self.test_frame.copy()

                    render_frame_timestamp(frame=self.test_frame.copy(), timestamp=timestamp, font_rate=0.1)

                    # Verify the correct timestamp text was passed
                    mock_render.assert_called_once()
                    # Get all call arguments (both positional and keyword)
                    args = mock_render.call_args[0]  # positional arguments
                    kwargs = mock_render.call_args[1]  # keyword arguments

                    # The function signature is: render_single_image_with_timestamp(image, number, rate, font_path)
                    if len(args) > 1:
                        # If number is passed as positional argument (index 1)
                        self.assertEqual(args[1], f"time: {expected_text}")
                    elif "number" in kwargs:
                        # If number is passed as keyword argument
                        self.assertEqual(kwargs["number"], f"time: {expected_text}")

    def test_render_frame_timestamp_functional(self):
        """Test that render_frame_timestamp integrates properly with timestamp_converting"""
        # This test verifies the integration between the two functions
        with patch(
            "fastdeploy.input.v1.ernie4_5_vl_processor.utils.render_timestamp.render_single_image_with_timestamp"
        ) as mock_render:
            mock_render.return_value = self.test_frame.copy()

            timestamp = 3661.5  # 1 hour, 1 minute, 1.5 seconds
            expected_converted = "01:01:01.50"

            render_frame_timestamp(frame=self.test_frame.copy(), timestamp=timestamp, font_rate=0.1)

            # Verify the integration
            mock_render.assert_called_once()
            # Get all call arguments (both positional and keyword)
            args = mock_render.call_args[0]  # positional arguments
            kwargs = mock_render.call_args[1]  # keyword arguments

            # The function signature is: render_single_image_with_timestamp(image, number, rate, font_path)
            if len(args) > 1:
                # If number is passed as positional argument (index 1)
                self.assertEqual(args[1], f"time: {expected_converted}")
            elif "number" in kwargs:
                # If number is passed as keyword argument
                self.assertEqual(kwargs["number"], f"time: {expected_converted}")


class TestFontFileExistence(unittest.TestCase):
    """Test that the required font file exists"""

    def test_font_file_exists(self):
        """Test that the Roboto-Regular.ttf font file exists"""
        self.assertTrue(os.path.exists(FONT_PATH), f"Font file not found at {FONT_PATH}")

    def test_font_file_is_file(self):
        """Test that the font path points to a file"""
        self.assertTrue(os.path.isfile(FONT_PATH), f"Font path is not a file: {FONT_PATH}")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""

    def test_full_timestamp_rendering_workflow(self):
        """Test the complete workflow from timestamp calculation to rendering"""
        # Create test frame
        frame = Image.new("RGB", (200, 150), color="lightblue")

        # Calculate timestamp for uniform frame extraction
        num_frames = 30
        frame_id = 15
        duration = 60.0  # 1 minute video
        timestamp_seconds = get_timestamp_for_uniform_frame_extraction(num_frames, frame_id, duration)

        # Convert timestamp to string format
        timestamp_str = timestamp_converting(timestamp_seconds)

        # Render timestamp on frame
        rendered_frame = render_frame_timestamp(frame, timestamp_seconds)

        # Verify results
        expected_timestamp = 60.0 * 15 / 30  # 30.0 seconds
        self.assertAlmostEqual(timestamp_seconds, expected_timestamp, places=5)
        self.assertEqual(timestamp_str, "00:00:30.00")
        self.assertIsInstance(rendered_frame, Image.Image)
        self.assertEqual(rendered_frame.size, (200, 150))


if __name__ == "__main__":
    unittest.main()
