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
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

# Mock cv2 before importing module under test
mock_cv2 = MagicMock()
mock_cv2.__spec__ = MagicMock()
sys.modules.setdefault("cv2", mock_cv2)

from fastdeploy.multimodal.video import (  # noqa: E402
    VideoMediaIO,
    rescale_video_size,
    resize_video,
    sample_frames_from_video,
)


class TestResizeVideo(unittest.TestCase):
    """Test resize_video function."""

    def setUp(self):
        mock_cv2.reset_mock()

    def test_resize_calls_cv2_for_each_frame(self):
        """resize_video calls cv2.resize for each frame."""
        frames = np.random.randint(0, 255, (3, 10, 20, 3), dtype=np.uint8)
        target_size = (5, 8)  # (height, width)

        # Mock cv2.resize to return array of target size
        def fake_resize(frame, dsize):
            w, h = dsize
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)

        mock_cv2.resize.side_effect = fake_resize

        result = resize_video(frames, target_size)

        self.assertEqual(mock_cv2.resize.call_count, 3)
        self.assertEqual(result.shape, (3, 5, 8, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_resize_single_frame(self):
        """resize_video works with a single frame."""
        frames = np.ones((1, 4, 6, 3), dtype=np.float32)
        target_size = (2, 3)

        def fake_resize(frame, dsize):
            w, h = dsize
            return np.ones((h, w, frame.shape[2]), dtype=frame.dtype)

        mock_cv2.resize.side_effect = fake_resize

        result = resize_video(frames, target_size)

        self.assertEqual(result.shape, (1, 2, 3, 3))
        mock_cv2.resize.assert_called_once()
        # Verify dsize is (width, height) for cv2
        call_args = mock_cv2.resize.call_args[0]
        self.assertEqual(call_args[1], (3, 2))


class TestRescaleVideoSize(unittest.TestCase):
    """Test rescale_video_size function."""

    def setUp(self):
        mock_cv2.reset_mock()

    def test_rescale_doubles_size(self):
        """rescale_video_size with factor 2.0 doubles dimensions."""
        frames = np.zeros((2, 10, 20, 3), dtype=np.uint8)

        def fake_resize(frame, dsize):
            w, h = dsize
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)

        mock_cv2.resize.side_effect = fake_resize

        result = rescale_video_size(frames, 2.0)

        self.assertEqual(result.shape, (2, 20, 40, 3))

    def test_rescale_halves_size(self):
        """rescale_video_size with factor 0.5 halves dimensions."""
        frames = np.zeros((4, 16, 32, 3), dtype=np.uint8)

        def fake_resize(frame, dsize):
            w, h = dsize
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)

        mock_cv2.resize.side_effect = fake_resize

        result = rescale_video_size(frames, 0.5)

        self.assertEqual(result.shape, (4, 8, 16, 3))


class TestSampleFramesFromVideo(unittest.TestCase):
    """Test sample_frames_from_video function."""

    def test_sample_all_frames_with_minus_one(self):
        """num_frames=-1 returns all frames unchanged."""
        frames = np.arange(24).reshape(4, 2, 3, 1)
        result = sample_frames_from_video(frames, -1)
        np.testing.assert_array_equal(result, frames)

    def test_sample_exact_count(self):
        """Sampling exact number of frames returns correct count."""
        frames = np.arange(60).reshape(10, 2, 3, 1)
        result = sample_frames_from_video(frames, 5)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1:], (2, 3, 1))

    def test_sample_one_frame(self):
        """Sampling 1 frame returns single frame."""
        frames = np.random.rand(20, 4, 4, 3)
        result = sample_frames_from_video(frames, 1)
        self.assertEqual(result.shape, (1, 4, 4, 3))

    def test_sample_all_frames_explicit(self):
        """Sampling total_frames returns all frames."""
        frames = np.random.rand(5, 2, 2, 3)
        result = sample_frames_from_video(frames, 5)
        self.assertEqual(result.shape[0], 5)
        # First and last should match
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])

    def test_sample_evenly_spaced(self):
        """Sampled frames are evenly spaced using linspace indices."""
        frames = np.arange(100).reshape(10, 2, 5, 1)
        result = sample_frames_from_video(frames, 3)
        # linspace(0, 9, 3) = [0, 4, 9] (rounded to int)
        expected_indices = np.linspace(0, 9, 3, dtype=int)
        np.testing.assert_array_equal(result, frames[expected_indices])


class TestVideoMediaIOInit(unittest.TestCase):
    """Test VideoMediaIO initialization."""

    def test_init(self):
        """VideoMediaIO can be instantiated."""
        io = VideoMediaIO()
        self.assertIsNotNone(io)


class TestVideoMediaIOLoadBytes(unittest.TestCase):
    """Test VideoMediaIO.load_bytes method."""

    def test_load_bytes_returns_same_data(self):
        """load_bytes returns the input bytes unchanged."""
        io = VideoMediaIO()
        data = b"\x00\x01\x02\x03video_data"
        result = io.load_bytes(data)
        self.assertEqual(result, data)

    def test_load_bytes_empty(self):
        """load_bytes handles empty bytes."""
        io = VideoMediaIO()
        result = io.load_bytes(b"")
        self.assertEqual(result, b"")


class TestVideoMediaIOLoadBase64(unittest.TestCase):
    """Test VideoMediaIO.load_base64 method."""

    def test_load_base64_decodes_data(self):
        """load_base64 decodes base64 and returns bytes."""
        io = VideoMediaIO()
        raw = b"fake_video_content"
        b64_str = base64.b64encode(raw).decode("utf-8")

        result = io.load_base64("video/mp4", b64_str)
        self.assertEqual(result, raw)

    def test_load_base64_video_jpeg_raises(self):
        """load_base64 raises ValueError for video/jpeg media type."""
        io = VideoMediaIO()
        with self.assertRaises(ValueError) as ctx:
            io.load_base64("video/jpeg", "dGVzdA==")
        self.assertIn("not supported", str(ctx.exception))

    def test_load_base64_case_insensitive(self):
        """load_base64 rejects video/jpeg case-insensitively."""
        io = VideoMediaIO()
        with self.assertRaises(ValueError):
            io.load_base64("Video/JPEG", "dGVzdA==")


class TestVideoMediaIOLoadFile(unittest.TestCase):
    """Test VideoMediaIO.load_file method."""

    def test_load_file_reads_content(self):
        """load_file reads file and returns bytes."""
        io = VideoMediaIO()
        content = b"fake_video_binary_data_12345"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(content)
            filepath = f.name

        try:
            result = io.load_file(filepath)
            self.assertEqual(result, content)
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    unittest.main()
