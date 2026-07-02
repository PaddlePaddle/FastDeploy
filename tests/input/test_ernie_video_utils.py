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

"""Unit tests for ernie4_5_vl_processor.utils.video_utils (paddlecodec backend)."""

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.ernie4_5_vl_processor.utils.video_utils import (
    VideoReaderWrapper,
    _NumpyFrame,
    is_gif,
)

MODULE = "fastdeploy.input.ernie4_5_vl_processor.utils.video_utils"

GIF87_HEADER = b"GIF87a" + b"\x00" * 10
GIF89_HEADER = b"GIF89a" + b"\x00" * 10
NOT_GIF = b"NOTGIF" + b"\x00" * 10


def _make_mock_decoder(num_frames=100, fps=25.0):
    """Return a mock that mimics torchcodec VideoDecoder."""
    decoder = MagicMock()
    decoder.metadata.num_frames = num_frames
    decoder.metadata.average_fps = fps

    def _get_frames_at(indices):
        batch = MagicMock()
        tensor = MagicMock()
        tensor.numpy.return_value = np.zeros((len(indices), 480, 640, 3), dtype=np.uint8)
        first = MagicMock()
        first.numpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor.__getitem__ = MagicMock(return_value=first)
        batch.data = tensor
        return batch

    decoder.get_frames_at = MagicMock(side_effect=_get_frames_at)
    return decoder


class _Guard:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class TestIsGif(unittest.TestCase):
    def test_gif87a(self):
        self.assertTrue(is_gif(GIF87_HEADER))

    def test_gif89a(self):
        self.assertTrue(is_gif(GIF89_HEADER))

    def test_not_gif(self):
        self.assertFalse(is_gif(NOT_GIF))


class TestNumpyFrame(unittest.TestCase):
    def test_asnumpy_roundtrip(self):
        arr = np.arange(6).reshape(2, 3)
        self.assertIs(_NumpyFrame(arr).asnumpy(), arr)


class TestVideoReaderWrapper(unittest.TestCase):
    def _make_wrapper(self, video_path, mock_decoder=None, decoder_factory=None):
        if mock_decoder is None:
            mock_decoder = _make_mock_decoder()

        decoders_module = MagicMock()
        if decoder_factory is not None:
            decoders_module.VideoDecoder = decoder_factory
        else:
            decoders_module.VideoDecoder.return_value = mock_decoder

        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = _Guard()

        with (
            patch.dict(
                "sys.modules",
                {"torchcodec": MagicMock(), "torchcodec.decoders": decoders_module},
            ),
            patch(f"{MODULE}.paddle", mock_paddle),
            patch(f"{MODULE}.mp", MagicMock()),
        ):
            return VideoReaderWrapper(video_path)

    def test_len(self):
        wrapper = self._make_wrapper("/fake/video.mp4", _make_mock_decoder(num_frames=42))
        self.assertEqual(len(wrapper), 42)

    def test_getitem_int(self):
        decoder = _make_mock_decoder()
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        frame = wrapper[0]
        self.assertIsInstance(frame.asnumpy(), np.ndarray)
        decoder.get_frames_at.assert_called_with(indices=[0])

    def test_getitem_slice(self):
        decoder = _make_mock_decoder(num_frames=10)
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        wrapper[1:4]
        decoder.get_frames_at.assert_called_with(indices=[1, 2, 3])

    def test_getitem_list(self):
        decoder = _make_mock_decoder()
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        wrapper[[2, 5]]
        decoder.get_frames_at.assert_called_with(indices=[2, 5])

    def test_get_avg_fps(self):
        wrapper = self._make_wrapper("/fake/video.mp4", _make_mock_decoder(fps=12.0))
        self.assertEqual(wrapper.get_avg_fps(), 12.0)

    def test_decoder_args(self):
        captured = {}

        def factory(path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return _make_mock_decoder()

        self._make_wrapper("/fake/video.mp4", decoder_factory=factory)
        self.assertEqual(captured["path"], "/fake/video.mp4")
        self.assertEqual(captured["seek_mode"], "exact")
        self.assertEqual(captured["dimension_order"], "NHWC")
        self.assertEqual(captured["device"], "cpu")

    def test_num_ffmpeg_threads_env(self):
        captured = {}

        def factory(path, **kwargs):
            captured.update(kwargs)
            return _make_mock_decoder()

        with patch.dict("os.environ", {"PADDLECODEC_NUM_THREADS": "8"}):
            self._make_wrapper("/fake/video.mp4", decoder_factory=factory)
        self.assertEqual(captured["num_ffmpeg_threads"], 8)

    def test_non_gif_string_does_not_set_original_file(self):
        wrapper = self._make_wrapper("/fake/video.mp4")
        self.assertIsNone(wrapper.original_file)

    def test_bytesio_non_gif_does_not_set_original_file(self):
        wrapper = self._make_wrapper(io.BytesIO(NOT_GIF))
        self.assertIsNone(wrapper.original_file)

    def test_gif_string_sets_original_file(self):
        mp_mock = MagicMock()
        decoders_module = MagicMock()
        decoders_module.VideoDecoder.return_value = _make_mock_decoder()
        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = _Guard()

        with (
            patch.dict(
                "sys.modules",
                {"torchcodec": MagicMock(), "torchcodec.decoders": decoders_module},
            ),
            patch(f"{MODULE}.paddle", mock_paddle),
            patch(f"{MODULE}.mp", mp_mock),
        ):
            wrapper = VideoReaderWrapper("/fake/anim.gif")

        mp_mock.VideoFileClip.assert_called_once_with("/fake/anim.gif")
        self.assertIsNotNone(wrapper.original_file)
        self.assertTrue(wrapper.original_file.endswith(".mp4"))

    def test_gif_bytes_sets_original_file(self):
        mp_mock = MagicMock()
        decoders_module = MagicMock()
        decoders_module.VideoDecoder.return_value = _make_mock_decoder()
        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = _Guard()

        with (
            patch.dict(
                "sys.modules",
                {"torchcodec": MagicMock(), "torchcodec.decoders": decoders_module},
            ),
            patch(f"{MODULE}.paddle", mock_paddle),
            patch(f"{MODULE}.mp", mp_mock),
        ):
            wrapper = VideoReaderWrapper(GIF89_HEADER)

        mp_mock.VideoFileClip.assert_called_once()
        self.assertIsNotNone(wrapper.original_file)

    def test_import_failure_reraises_and_logs(self):
        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = _Guard()

        broken = MagicMock()
        type(broken).VideoDecoder = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        with (
            patch.dict(
                "sys.modules",
                {"torchcodec": MagicMock(), "torchcodec.decoders": broken},
            ),
            patch(f"{MODULE}.paddle", mock_paddle),
            patch(f"{MODULE}.mp", MagicMock()),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            with self.assertRaises(RuntimeError):
                VideoReaderWrapper("/fake/video.mp4")

        mock_logger.error.assert_called_once()

    def test_del_no_original_file(self):
        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = None
        wrapper.__del__()  # should not raise

    def test_del_removes_temp_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name
        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = tmp_path
        wrapper.__del__()
        self.assertFalse(os.path.exists(tmp_path))


if __name__ == "__main__":
    unittest.main()
