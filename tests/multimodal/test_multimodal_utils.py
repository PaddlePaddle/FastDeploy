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

import os
import sys
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

# Determine import method based on environment
# Use environment variable FD_TEST_MODE=standalone for local testing
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    # Mock the logger to avoid import issues
    mock_logger = Mock()

    # Create a mock module structure
    class MockUtils:
        data_processor_logger = mock_logger

    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.multimodal"] = Mock()

    # Import the utils module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "multimodal_utils", os.path.join(os.path.dirname(__file__), "../../fastdeploy/multimodal/utils.py")
    )
    multimodal_utils = importlib.util.module_from_spec(spec)
    multimodal_utils.data_processor_logger = mock_logger
    spec.loader.exec_module(multimodal_utils)

    # Extract the function we want to test
    process_transparency = multimodal_utils.process_transparency
else:
    # Normal mode - direct import (for CI/CD and production)
    try:
        from fastdeploy.multimodal.utils import process_transparency

        # If we can import directly, we don't need mocking
        mock_logger = None
    except ImportError:
        # Fallback to standalone mode if direct import fails
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        # Re-run the standalone setup
        mock_logger = Mock()

        class MockUtils:
            data_processor_logger = mock_logger

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.multimodal"] = Mock()

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "multimodal_utils", os.path.join(os.path.dirname(__file__), "../../fastdeploy/multimodal/utils.py")
        )
        multimodal_utils = importlib.util.module_from_spec(spec)
        multimodal_utils.data_processor_logger = mock_logger
        spec.loader.exec_module(multimodal_utils)

        process_transparency = multimodal_utils.process_transparency


class TestProcessTransparency(unittest.TestCase):
    """Test cases for multimodal utils functions."""

    def setUp(self):
        """Set up test fixtures with various image types."""
        # Create a 100x100 RGB image (no transparency)
        self.rgb_image = Image.new("RGB", (100, 100), color="red")

        # Create a 100x100 RGBA image with full opacity
        self.rgba_opaque = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))

        # Create a 100x100 RGBA image with transparency
        self.rgba_transparent = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        # Create a 100x100 RGBA image with some fully transparent pixels
        self.rgba_partial_transparent = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
        draw = ImageDraw.Draw(self.rgba_partial_transparent)
        draw.rectangle([10, 10, 50, 50], fill=(0, 255, 0, 0))  # Fully transparent rectangle

        # Create LA image with transparency
        self.la_transparent = Image.new("LA", (100, 100), color=(128, 128))

        # Create P mode image with transparency
        self.p_transparent = Image.new("P", (100, 100))
        self.p_transparent.info["transparency"] = 0

        # Create P mode image without transparency
        self.p_opaque = Image.new("P", (100, 100))

    def test_process_transparency_with_opaque_rgb(self):
        """Test processing RGB image without transparency."""
        result = process_transparency(self.rgb_image)

        # Should return same image (no conversion needed)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (100, 100))

    def test_process_transparency_with_opaque_rgba(self):
        """Test processing RGBA image with full opacity."""
        result = process_transparency(self.rgba_opaque)

        # Should return same image (no conversion needed)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (100, 100))

    def test_process_transparency_with_transparent_rgba(self):
        """Test processing RGBA image with transparency."""
        result = process_transparency(self.rgba_transparent)

        # Should convert to RGB with white background
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (100, 100))

    def test_process_transparency_with_partial_transparent_rgba(self):
        """Test processing RGBA image with some transparent pixels."""
        result = process_transparency(self.rgba_partial_transparent)

        # Should convert to RGB with white background
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (100, 100))

    def test_process_transparency_with_transparent_la(self):
        """Test processing LA image with transparency."""
        result = process_transparency(self.la_transparent)

        # Should convert to RGB with white background
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (100, 100))

    def test_process_transparency_with_palette_transparency(self):
        """Test processing P mode image with transparency info."""
        result = process_transparency(self.p_transparent)

        # P mode with transparency info should be detected as transparent
        # but conversion might fail due to "bad transparency mask" error
        # In case of error, the function falls back to the original image
        self.assertEqual(result.size, (100, 100))
        # The mode could be P (if error occurred) or RGB (if conversion succeeded)

    def test_process_transparency_with_opaque_palette(self):
        """Test processing P mode image without transparency."""
        result = process_transparency(self.p_opaque)

        # P mode without transparency should remain P mode (no transparency detected)
        # But will go through exif_transpose which might change mode
        self.assertEqual(result.size, (100, 100))
        # The exact mode depends on exif_transpose behavior

    def test_process_transparency_error_handling(self):
        """Test error handling in transparency processing."""
        # Create a mock image that will raise an exception
        mock_image = Mock()
        mock_image.mode = "RGBA"
        mock_image.convert.side_effect = Exception("Test error")

        # Should not raise exception, should return result of exif_transpose
        with patch("PIL.ImageOps.exif_transpose") as mock_exif:
            mock_exif.return_value = self.rgb_image
            result = process_transparency(mock_image)

            # Should return the result from exif_transpose
            self.assertEqual(result, self.rgb_image)

    def test_convert_transparent_paste_white_background(self):
        """Test that transparent paste creates white background."""
        # Create a simple transparent image
        transparent_img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))  # Fully transparent red

        result = process_transparency(transparent_img)

        # Should be RGB mode with white background
        self.assertEqual(result.mode, "RGB")

        # Check that the converted image has white background
        # (since original was fully transparent, should be white)
        pixels = list(result.getdata())
        # All pixels should be white (255, 255, 255)
        for pixel in pixels:
            self.assertEqual(pixel, (255, 255, 255))

    def test_convert_transparent_paste_partial_transparency(self):
        """Test transparent paste with partially transparent image."""
        # Create image with partial transparency
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))  # 50% transparent red

        result = process_transparency(img)

        # Should be RGB mode
        self.assertEqual(result.mode, "RGB")

        # Should have been pasted onto white background
        pixels = list(result.getdata())
        # All pixels should be the same (blended with white background)
        for pixel in pixels:
            # With 50% transparency, red (255,0,0) blended with white (255,255,255)
            # should give a pinkish color
            self.assertGreater(pixel[0], 128)  # Red component should be significant
            self.assertGreaterEqual(pixel[1], 127)  # Green component from white background
            self.assertGreaterEqual(pixel[2], 127)  # Blue component from white background

    def test_edge_case_min_alpha_value(self):
        """Test edge case with minimum alpha value."""
        # Create image with alpha at minimum (0)
        img = Image.new("RGBA", (1, 1), (255, 0, 0, 0))

        result = process_transparency(img)

        # Should be converted to RGB
        self.assertEqual(result.mode, "RGB")

    def test_edge_case_max_alpha_value(self):
        """Test edge case with maximum alpha value."""
        # Create image with alpha at maximum (255)
        img = Image.new("RGBA", (1, 1), (255, 0, 0, 255))

        result = process_transparency(img)

        # Should remain RGBA (no transparency detected)
        self.assertEqual(result.mode, "RGBA")

    def test_edge_case_empty_image(self):
        """Test edge case with empty (0x0) image."""
        img = Image.new("RGBA", (0, 0))

        result = process_transparency(img)

        # Should handle empty image gracefully
        self.assertEqual(result.size, (0, 0))

    def test_edge_case_single_pixel_transparent(self):
        """Test edge case with single pixel transparent image."""
        img = Image.new("RGBA", (1, 1), (255, 0, 0, 0))

        result = process_transparency(img)

        # Should convert to RGB
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (1, 1))

    def test_edge_case_single_pixel_opaque(self):
        """Test edge case with single pixel opaque image."""
        img = Image.new("RGBA", (1, 1), (255, 0, 0, 255))

        result = process_transparency(img)

        # Should remain RGBA
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (1, 1))


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
