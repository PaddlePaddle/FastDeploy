"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import base64
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from fastdeploy.multimodal.image import ImageMediaIO


class TestImageMediaIOInit(unittest.TestCase):
    """Test ImageMediaIO initialization."""

    def test_default_mode(self):
        """Default image_mode is RGB."""
        io = ImageMediaIO()
        self.assertEqual(io.image_mode, "RGB")

    def test_custom_mode(self):
        """Custom image_mode is stored."""
        io = ImageMediaIO(image_mode="L")
        self.assertEqual(io.image_mode, "L")


class TestImageMediaIOLoadBytes(unittest.TestCase):
    """Test ImageMediaIO.load_bytes method."""

    def _make_png_bytes(self, mode="RGB", size=(4, 4)):
        """Create a small PNG image as bytes."""
        img = Image.new(mode, size, color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_load_bytes_returns_image(self):
        """load_bytes returns a PIL Image in RGB mode."""
        io = ImageMediaIO()
        data = self._make_png_bytes()
        result = io.load_bytes(data)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (4, 4))

    def test_load_bytes_converts_to_custom_mode(self):
        """load_bytes converts to specified image_mode."""
        io = ImageMediaIO(image_mode="L")
        data = self._make_png_bytes()
        result = io.load_bytes(data)

        self.assertEqual(result.mode, "L")

    def test_load_bytes_rgba_image(self):
        """load_bytes handles RGBA images (transparency processing)."""
        io = ImageMediaIO()
        # Create RGBA image with semi-transparent pixels
        img = Image.new("RGBA", (4, 4), color=(255, 0, 0, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        result = io.load_bytes(data)
        self.assertEqual(result.mode, "RGB")


class TestImageMediaIOLoadBase64(unittest.TestCase):
    """Test ImageMediaIO.load_base64 method."""

    def test_load_base64_decodes_and_returns_image(self):
        """load_base64 decodes base64 string and returns PIL Image."""
        io = ImageMediaIO()
        # Create a small image and encode to base64
        img = Image.new("RGB", (2, 2), color=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = io.load_base64("image/png", b64_str)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (2, 2))


class TestImageMediaIOLoadFile(unittest.TestCase):
    """Test ImageMediaIO.load_file method."""

    def test_load_file_returns_image(self):
        """load_file opens file from path and returns PIL Image."""
        io = ImageMediaIO()
        # Create a temp image file
        img = Image.new("RGB", (8, 8), color=(0, 0, 255))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            filepath = f.name

        try:
            result = io.load_file(filepath)
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.size, (8, 8))
        finally:
            os.unlink(filepath)

    def test_load_file_converts_mode(self):
        """load_file converts to custom image_mode."""
        io = ImageMediaIO(image_mode="L")
        img = Image.new("RGB", (4, 4), color=(128, 128, 128))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            filepath = f.name

        try:
            result = io.load_file(filepath)
            self.assertEqual(result.mode, "L")
        finally:
            os.unlink(filepath)


class TestImageMediaIOLoadFileRequest(unittest.TestCase):
    """Test ImageMediaIO.load_file_request method."""

    @patch("fastdeploy.multimodal.image.requests.get")
    def test_load_file_request_fetches_url(self, mock_get):
        """load_file_request fetches image from URL and returns PIL Image."""
        io = ImageMediaIO()

        # Create a fake response with image data
        img = Image.new("RGB", (3, 3), color=(255, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.raw = buf
        mock_get.return_value = mock_response

        result = io.load_file_request("http://example.com/img.png")

        mock_get.assert_called_once_with("http://example.com/img.png", stream=True)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (3, 3))


class TestImageMediaIOEncodeBase64(unittest.TestCase):
    """Test ImageMediaIO.encode_base64 method."""

    def test_encode_base64_returns_valid_string(self):
        """encode_base64 returns a valid base64-encoded string."""
        io = ImageMediaIO()
        img = Image.new("RGB", (4, 4), color=(100, 150, 200))

        result = io.encode_base64(img)

        # Should be a valid base64 string
        decoded_bytes = base64.b64decode(result)
        # Should be a valid JPEG image
        restored = Image.open(BytesIO(decoded_bytes))
        self.assertEqual(restored.format, "JPEG")
        self.assertEqual(restored.size, (4, 4))

    def test_encode_base64_png_format(self):
        """encode_base64 supports custom image_format."""
        io = ImageMediaIO()
        img = Image.new("RGB", (4, 4), color=(50, 50, 50))

        result = io.encode_base64(img, image_format="PNG")

        decoded_bytes = base64.b64decode(result)
        restored = Image.open(BytesIO(decoded_bytes))
        self.assertEqual(restored.format, "PNG")

    def test_encode_base64_converts_mode(self):
        """encode_base64 converts image to image_mode before saving."""
        io = ImageMediaIO(image_mode="L")
        img = Image.new("RGB", (4, 4), color=(128, 128, 128))

        result = io.encode_base64(img)

        decoded_bytes = base64.b64decode(result)
        restored = Image.open(BytesIO(decoded_bytes))
        self.assertEqual(restored.mode, "L")


if __name__ == "__main__":
    unittest.main()
