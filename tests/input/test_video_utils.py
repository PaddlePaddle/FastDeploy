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

import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.utils.video import (
    _is_gif,
    read_video_decord,
    sample_frames,
    sample_frames_paddleocr,
    sample_frames_qwen,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GIF87_HEADER = b"GIF87a" + b"\x00" * 10
GIF89_HEADER = b"GIF89a" + b"\x00" * 10
NOT_GIF = b"NOTGIF" + b"\x00" * 10


def _make_mock_reader(num_frames=100, fps=25.0):
    """Return a mock that mimics decord.VideoReader."""
    reader = MagicMock()
    reader.__len__ = MagicMock(return_value=num_frames)
    reader.get_avg_fps = MagicMock(return_value=fps)
    reader.seek = MagicMock(return_value=None)
    frame = MagicMock()
    frame.asnumpy = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    reader.__getitem__ = MagicMock(return_value=frame)
    return reader


# ---------------------------------------------------------------------------
# _is_gif
# ---------------------------------------------------------------------------


class TestIsGif(unittest.TestCase):
    def test_gif87a(self):
        self.assertTrue(_is_gif(GIF87_HEADER))

    def test_gif89a(self):
        self.assertTrue(_is_gif(GIF89_HEADER))

    def test_not_gif(self):
        self.assertFalse(_is_gif(NOT_GIF))

    def test_short_bytes(self):
        self.assertFalse(_is_gif(b"GIF"))


# ---------------------------------------------------------------------------
# VideoReaderWrapper (mock decord + moviepy)
# ---------------------------------------------------------------------------


class TestVideoReaderWrapper(unittest.TestCase):
    def _make_wrapper(self, video_path, mock_reader=None):
        """Construct a VideoReaderWrapper with decord mocked out."""
        from fastdeploy.input.utils.video import VideoReaderWrapper

        if mock_reader is None:
            mock_reader = _make_mock_reader()

        mock_decord = MagicMock()
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict("sys.modules", {"decord": mock_decord, "moviepy": MagicMock(), "moviepy.editor": MagicMock()}):
            wrapper = VideoReaderWrapper(video_path)

        wrapper._reader = mock_reader
        return wrapper

    def test_len(self):
        reader = _make_mock_reader(num_frames=42)
        wrapper = self._make_wrapper("/fake/video.mp4", reader)
        self.assertEqual(len(wrapper), 42)

    def test_getitem_resets_seek(self):
        reader = _make_mock_reader()
        wrapper = self._make_wrapper("/fake/video.mp4", reader)
        _ = wrapper[0]
        reader.seek.assert_called_with(0)

    def test_get_avg_fps(self):
        reader = _make_mock_reader(fps=30.0)
        wrapper = self._make_wrapper("/fake/video.mp4", reader)
        self.assertEqual(wrapper.get_avg_fps(), 30.0)

    def test_seek(self):
        reader = _make_mock_reader()
        wrapper = self._make_wrapper("/fake/video.mp4", reader)
        wrapper.seek(5)
        reader.seek.assert_called_with(5)

    def test_del_no_original_file(self):
        """__del__ should be a no-op when original_file is None."""
        from fastdeploy.input.utils.video import VideoReaderWrapper

        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = None
        wrapper._reader = _make_mock_reader()
        # Should not raise
        wrapper.__del__()

    def test_del_removes_temp_file(self):
        """__del__ removes the file only when original_file is set."""
        import os
        import tempfile

        from fastdeploy.input.utils.video import VideoReaderWrapper

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = tmp_path
        wrapper._reader = _make_mock_reader()
        wrapper.__del__()
        self.assertFalse(os.path.exists(tmp_path))

    def test_non_gif_string_path_does_not_set_original_file(self):
        """Passing a non-GIF string path must NOT set original_file (bug fix)."""
        from fastdeploy.input.utils.video import VideoReaderWrapper

        mock_reader = _make_mock_reader()
        mock_decord = MagicMock()
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict("sys.modules", {"decord": mock_decord, "moviepy": MagicMock(), "moviepy.editor": MagicMock()}):
            wrapper = VideoReaderWrapper("/fake/video.mp4")

        self.assertIsNone(wrapper.original_file)

    def test_bytesio_non_gif_path_does_not_set_original_file(self):
        """Passing a BytesIO that is NOT a GIF must not set original_file."""
        from fastdeploy.input.utils.video import VideoReaderWrapper

        mock_reader = _make_mock_reader()
        mock_decord = MagicMock()
        mock_decord.VideoReader.return_value = mock_reader

        bio = io.BytesIO(NOT_GIF)
        with patch.dict("sys.modules", {"decord": mock_decord, "moviepy": MagicMock(), "moviepy.editor": MagicMock()}):
            wrapper = VideoReaderWrapper(bio)

        self.assertIsNone(wrapper.original_file)


# ---------------------------------------------------------------------------
# read_video_decord
# ---------------------------------------------------------------------------


class TestReadVideoDecord(unittest.TestCase):
    def _patch_wrapper(self, num_frames=100, fps=25.0):
        """Return a context manager that replaces VideoReaderWrapper with a mock."""
        from fastdeploy.input.utils import video

        mock_wrapper = MagicMock()
        mock_wrapper.__len__ = MagicMock(return_value=num_frames)
        mock_wrapper.get_avg_fps = MagicMock(return_value=fps)
        return patch.object(video, "VideoReaderWrapper", return_value=mock_wrapper), mock_wrapper

    def test_existing_wrapper_passthrough(self):
        """Already-wrapped reader is returned as-is."""
        from fastdeploy.input.utils.video import VideoReaderWrapper

        mock_wrapper = MagicMock(spec=VideoReaderWrapper)
        mock_wrapper.__len__ = MagicMock(return_value=50)
        mock_wrapper.get_avg_fps = MagicMock(return_value=10.0)

        reader, meta, path = read_video_decord(mock_wrapper)

        self.assertIs(reader, mock_wrapper)
        self.assertEqual(meta["num_of_frame"], 50)
        self.assertAlmostEqual(meta["fps"], 10.0)
        self.assertAlmostEqual(meta["duration"], 5.0)

    def test_bytes_input_converted_to_bytesio(self):
        """bytes input is converted to BytesIO before creating VideoReaderWrapper."""
        from fastdeploy.input.utils import video

        captured = []

        class FakeWrapper:
            def __init__(self, path, *args, **kwargs):
                captured.append(path)

            def __len__(self):
                return 30

            def get_avg_fps(self):
                return 10.0

        with patch.object(video, "VideoReaderWrapper", FakeWrapper):
            reader, meta, path = read_video_decord(b"fake_video_bytes")

        self.assertIsInstance(captured[0], io.BytesIO)

    def test_string_path_input(self):
        """String path is passed through to VideoReaderWrapper."""
        from fastdeploy.input.utils import video

        class FakeWrapper:
            def __init__(self, path, *args, **kwargs):
                pass

            def __len__(self):
                return 60

            def get_avg_fps(self):
                return 30.0

        with patch.object(video, "VideoReaderWrapper", FakeWrapper):
            reader, meta, path = read_video_decord("/fake/path.mp4")

        self.assertEqual(meta["num_of_frame"], 60)
        self.assertAlmostEqual(meta["duration"], 2.0)
        self.assertEqual(path, "/fake/path.mp4")


# ---------------------------------------------------------------------------
# sample_frames_qwen
# ---------------------------------------------------------------------------


class TestSampleFramesQwen(unittest.TestCase):
    META = {"num_of_frame": 100, "fps": 25.0}

    def test_num_frames_basic(self):
        indices = sample_frames_qwen(2, 4, 100, self.META, num_frames=8)
        self.assertEqual(len(indices), 8)

    def test_fps_basic(self):
        indices = sample_frames_qwen(2, 4, 100, self.META, fps=2.0)
        self.assertGreater(len(indices), 0)
        self.assertEqual(len(indices) % 2, 0)

    def test_fps_and_num_frames_raises(self):
        with self.assertRaises(ValueError):
            sample_frames_qwen(2, 4, 100, self.META, fps=2.0, num_frames=10)

    def test_num_frames_exceeds_total_raises(self):
        with self.assertRaises(ValueError):
            sample_frames_qwen(2, 4, 100, self.META, num_frames=200)

    def test_fps_warning_when_nframes_exceeds_total(self):
        """fps so high that computed num_frames > total → warning logged."""
        with self.assertLogs(logger="fastdeploy.main", level="WARNING"):
            sample_frames_qwen(2, 4, 100, {"num_of_frame": 10, "fps": 1.0}, fps=100.0)

    def test_divisible_by_4_correction(self):
        """Result must be divisible by 4 when num_frames > 2."""
        indices = sample_frames_qwen(2, 4, 100, self.META, fps=1.5)
        if len(indices) > 2:
            self.assertEqual(len(indices) % 4, 0)

    def test_no_sampling_returns_all_frames(self):
        """Both fps and num_frames at sentinel → return all frames."""
        indices = sample_frames_qwen(2, 4, 100, self.META)
        self.assertEqual(len(indices), 100)

    def test_indices_dtype(self):
        indices = sample_frames_qwen(2, 4, 100, self.META, num_frames=8)
        self.assertEqual(indices.dtype, np.int32)


# ---------------------------------------------------------------------------
# sample_frames_paddleocr
# ---------------------------------------------------------------------------


class TestSampleFramesPaddleocr(unittest.TestCase):
    META = {"num_of_frame": 100, "fps": 25.0}

    def test_num_frames_basic(self):
        indices = sample_frames_paddleocr(1, 4, 100, self.META, num_frames=10)
        self.assertEqual(len(indices), 10)

    def test_fps_basic(self):
        indices = sample_frames_paddleocr(1, 4, 100, self.META, fps=2.0)
        self.assertGreater(len(indices), 0)

    def test_fps_and_num_frames_raises(self):
        with self.assertRaises(ValueError):
            sample_frames_paddleocr(1, 4, 100, self.META, fps=2.0, num_frames=10)

    def test_num_frames_exceeds_total_raises(self):
        with self.assertRaises(ValueError):
            sample_frames_paddleocr(1, 4, 100, self.META, num_frames=200)

    def test_none_sentinels_no_sampling(self):
        """fps=None, num_frames=None → return all frames."""
        indices = sample_frames_paddleocr(1, 4, 100, self.META)
        self.assertEqual(len(indices), 100)

    def test_no_4_correction(self):
        """paddleocr variant does NOT apply %4 correction."""
        # 6 frames is not divisible by 4; paddleocr should keep it
        meta = {"num_of_frame": 100, "fps": 25.0}
        indices = sample_frames_paddleocr(1, 1, 100, meta, num_frames=6)
        self.assertEqual(len(indices), 6)

    def test_indices_dtype(self):
        indices = sample_frames_paddleocr(1, 4, 100, self.META, num_frames=8)
        self.assertEqual(indices.dtype, np.int32)


# ---------------------------------------------------------------------------
# sample_frames dispatcher
# ---------------------------------------------------------------------------


class TestSampleFramesDispatcher(unittest.TestCase):
    META = {"num_of_frame": 100, "fps": 25.0}

    def test_default_variant_is_paddleocr(self):
        with patch("fastdeploy.input.utils.video.sample_frames_paddleocr", wraps=sample_frames_paddleocr) as mock_fn:
            sample_frames(1, 4, 100, self.META, num_frames=8)
            mock_fn.assert_called_once()

    def test_qwen_variant_dispatched(self):
        with patch("fastdeploy.input.utils.video.sample_frames_qwen", wraps=sample_frames_qwen) as mock_fn:
            sample_frames(2, 4, 100, self.META, num_frames=8, variant="qwen")
            mock_fn.assert_called_once()

    def test_qwen_none_fps_converted_to_sentinel(self):
        """None fps/num_frames → converted to -1 before calling sample_frames_qwen."""
        with patch("fastdeploy.input.utils.video.sample_frames_qwen", return_value=np.array([])) as mock_fn:
            sample_frames(2, 4, 100, self.META, fps=None, num_frames=None, variant="qwen")
            args = mock_fn.call_args[0]
            self.assertEqual(args[4], -1)  # fps sentinel
            self.assertEqual(args[5], -1)  # num_frames sentinel

    def test_paddleocr_variant_result_consistent(self):
        direct = sample_frames_paddleocr(1, 4, 100, self.META, num_frames=8)
        via_dispatcher = sample_frames(1, 4, 100, self.META, num_frames=8, variant="paddleocr")
        np.testing.assert_array_equal(direct, via_dispatcher)

    def test_qwen_variant_result_consistent(self):
        direct = sample_frames_qwen(2, 4, 100, self.META, num_frames=8)
        via_dispatcher = sample_frames(2, 4, 100, self.META, num_frames=8, variant="qwen")
        np.testing.assert_array_equal(direct, via_dispatcher)


if __name__ == "__main__":
    unittest.main()
