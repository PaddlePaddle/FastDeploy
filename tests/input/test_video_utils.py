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

from fastdeploy.input.video_utils import (
    _is_gif,
    read_video_paddlecodec,
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


def _make_mock_decoder(num_frames=100, fps=25.0):
    """Return a mock that mimics torchcodec VideoDecoder."""
    decoder = MagicMock()
    decoder.metadata.num_frames = num_frames
    decoder.metadata.average_fps = fps

    def _get_frames_at(indices):
        batch = MagicMock()
        tensor = MagicMock()
        tensor.numpy.return_value = np.zeros((len(indices), 480, 640, 3), dtype=np.uint8)
        # data[0] should also expose .numpy()
        first = MagicMock()
        first.numpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor.__getitem__ = MagicMock(return_value=first)
        batch.data = tensor
        return batch

    decoder.get_frames_at = MagicMock(side_effect=_get_frames_at)
    return decoder


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
# VideoReaderWrapper (mock paddlecodec/torchcodec + moviepy)
# ---------------------------------------------------------------------------


class TestVideoReaderWrapper(unittest.TestCase):
    @staticmethod
    def _guard():
        """A no-op context manager standing in for paddle.use_compat_guard."""

        class _Guard:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        return _Guard()

    def _make_wrapper(self, video_path, mock_decoder=None, decoder_factory=None, moviepy_mock=None):
        """Construct a VideoReaderWrapper with torchcodec/paddle mocked out.

        - mock_decoder: decoder instance returned by VideoDecoder(...)
        - decoder_factory: optional callable used as VideoDecoder (captures args /
          raises). Takes precedence over mock_decoder.
        - moviepy_mock: optional mock for the moviepy module (for GIF path).
        """
        from fastdeploy.input.video_utils import VideoReaderWrapper

        if mock_decoder is None:
            mock_decoder = _make_mock_decoder()

        decoders_module = MagicMock()
        if decoder_factory is not None:
            decoders_module.VideoDecoder = decoder_factory
        else:
            decoders_module.VideoDecoder.return_value = mock_decoder

        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = self._guard()

        moviepy = moviepy_mock or MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "torchcodec": MagicMock(),
                    "torchcodec.decoders": decoders_module,
                    "moviepy": moviepy,
                    "moviepy.editor": moviepy,
                },
            ),
            patch("fastdeploy.input.video_utils.paddle", mock_paddle),
        ):
            wrapper = VideoReaderWrapper(video_path)

        return wrapper

    def test_len(self):
        decoder = _make_mock_decoder(num_frames=42)
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        self.assertEqual(len(wrapper), 42)

    def test_getitem_int_returns_numpy_frame(self):
        decoder = _make_mock_decoder()
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        frame = wrapper[0]
        self.assertIsInstance(frame.asnumpy(), np.ndarray)
        # int access uses single-element indices list
        decoder.get_frames_at.assert_called_with(indices=[0])

    def test_getitem_numpy_integer(self):
        decoder = _make_mock_decoder()
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        frame = wrapper[np.int64(3)]
        self.assertIsInstance(frame.asnumpy(), np.ndarray)
        decoder.get_frames_at.assert_called_with(indices=[3])

    def test_getitem_slice(self):
        decoder = _make_mock_decoder(num_frames=10)
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        frames = wrapper[2:5]
        self.assertIsInstance(frames.asnumpy(), np.ndarray)
        decoder.get_frames_at.assert_called_with(indices=[2, 3, 4])

    def test_getitem_list(self):
        decoder = _make_mock_decoder()
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        frames = wrapper[[1, 4, 7]]
        self.assertIsInstance(frames.asnumpy(), np.ndarray)
        decoder.get_frames_at.assert_called_with(indices=[1, 4, 7])

    def test_get_avg_fps(self):
        decoder = _make_mock_decoder(fps=30.0)
        wrapper = self._make_wrapper("/fake/video.mp4", decoder)
        self.assertEqual(wrapper.get_avg_fps(), 30.0)

    def test_decoder_constructed_with_expected_args(self):
        """VideoDecoder must receive the expected keyword arguments."""
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
        self.assertIn("num_ffmpeg_threads", captured)

    def test_num_ffmpeg_threads_from_env(self):
        """PADDLECODEC_NUM_THREADS env var controls num_ffmpeg_threads."""
        captured = {}

        def factory(path, **kwargs):
            captured.update(kwargs)
            return _make_mock_decoder()

        with patch.dict("os.environ", {"PADDLECODEC_NUM_THREADS": "4"}):
            self._make_wrapper("/fake/video.mp4", decoder_factory=factory)

        self.assertEqual(captured["num_ffmpeg_threads"], 4)

    def test_torchcodec_import_failure_reraises_and_logs(self):
        """When torchcodec import fails, the error is logged and re-raised."""
        from fastdeploy.input.video_utils import VideoReaderWrapper

        mock_paddle = MagicMock()
        mock_paddle.use_compat_guard.return_value = self._guard()

        # A module whose attribute access raises ImportError mimics a broken backend
        broken = MagicMock()
        type(broken).VideoDecoder = property(lambda self: (_ for _ in ()).throw(ImportError("boom")))

        with (
            patch.dict(
                "sys.modules",
                {
                    "torchcodec": MagicMock(),
                    "torchcodec.decoders": broken,
                    "moviepy": MagicMock(),
                    "moviepy.editor": MagicMock(),
                },
            ),
            patch("fastdeploy.input.video_utils.paddle", mock_paddle),
            patch("fastdeploy.input.video_utils.logger") as mock_logger,
        ):
            with self.assertRaises(ImportError):
                VideoReaderWrapper("/fake/video.mp4")

        mock_logger.error.assert_called_once()

    def test_del_no_original_file(self):
        """__del__ should be a no-op when original_file is None."""
        from fastdeploy.input.video_utils import VideoReaderWrapper

        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = None
        wrapper._decoder = _make_mock_decoder()
        # Should not raise
        wrapper.__del__()

    def test_del_removes_temp_file(self):
        """__del__ removes the file only when original_file is set."""
        import os
        import tempfile

        from fastdeploy.input.video_utils import VideoReaderWrapper

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        wrapper = object.__new__(VideoReaderWrapper)
        wrapper.original_file = tmp_path
        wrapper._decoder = _make_mock_decoder()
        wrapper.__del__()
        self.assertFalse(os.path.exists(tmp_path))

    def test_non_gif_string_path_does_not_set_original_file(self):
        """Passing a non-GIF string path must NOT set original_file (bug fix)."""
        wrapper = self._make_wrapper("/fake/video.mp4")
        self.assertIsNone(wrapper.original_file)

    def test_bytesio_non_gif_path_does_not_set_original_file(self):
        """Passing a BytesIO that is NOT a GIF must not set original_file."""
        bio = io.BytesIO(NOT_GIF)
        wrapper = self._make_wrapper(bio)
        self.assertIsNone(wrapper.original_file)

    def test_gif_string_path_converts_to_mp4_and_sets_original_file(self):
        """A .gif string path is transcoded to mp4 and tracked for cleanup."""
        moviepy = MagicMock()
        clip = moviepy.editor.VideoFileClip.return_value

        wrapper = self._make_wrapper("/fake/anim.gif", moviepy_mock=moviepy)

        moviepy.editor.VideoFileClip.assert_called_once_with("/fake/anim.gif")
        clip.write_videofile.assert_called_once()
        clip.close.assert_called_once()
        # original_file points at the generated temp mp4
        self.assertIsNotNone(wrapper.original_file)
        self.assertTrue(wrapper.original_file.endswith(".mp4"))

    def test_gif_bytes_converts_to_mp4(self):
        """GIF bytes are written to a temp gif then transcoded to mp4."""
        moviepy = MagicMock()

        wrapper = self._make_wrapper(GIF89_HEADER, moviepy_mock=moviepy)

        moviepy.editor.VideoFileClip.assert_called_once()
        self.assertIsNotNone(wrapper.original_file)
        self.assertTrue(wrapper.original_file.endswith(".mp4"))

    def test_gif_bytesio_converts_to_mp4(self):
        """GIF content in a BytesIO is transcoded to mp4."""
        moviepy = MagicMock()

        wrapper = self._make_wrapper(io.BytesIO(GIF87_HEADER), moviepy_mock=moviepy)

        moviepy.editor.VideoFileClip.assert_called_once()
        self.assertIsNotNone(wrapper.original_file)


# ---------------------------------------------------------------------------
# read_video_paddlecodec
# ---------------------------------------------------------------------------


class TestReadVideoPaddlecodec(unittest.TestCase):
    def test_existing_wrapper_passthrough(self):
        """Already-wrapped reader is returned as-is."""
        from fastdeploy.input.video_utils import VideoReaderWrapper

        mock_wrapper = MagicMock(spec=VideoReaderWrapper)
        mock_wrapper.__len__ = MagicMock(return_value=50)
        mock_wrapper.get_avg_fps = MagicMock(return_value=10.0)

        reader, meta, path = read_video_paddlecodec(mock_wrapper)

        self.assertIs(reader, mock_wrapper)
        self.assertEqual(meta["num_of_frame"], 50)
        self.assertAlmostEqual(meta["fps"], 10.0)
        self.assertAlmostEqual(meta["duration"], 5.0)

    def test_bytes_input_converted_to_bytesio(self):
        """bytes input is converted to BytesIO before creating VideoReaderWrapper."""
        from fastdeploy.input import video_utils

        captured = []

        class FakeWrapper:
            def __init__(self, path, *args, **kwargs):
                captured.append(path)

            def __len__(self):
                return 30

            def get_avg_fps(self):
                return 10.0

        with patch.object(video_utils, "VideoReaderWrapper", FakeWrapper):
            reader, meta, path = read_video_paddlecodec(b"fake_video_bytes")

        self.assertIsInstance(captured[0], io.BytesIO)

    def test_string_path_input(self):
        """String path is passed through to VideoReaderWrapper."""
        from fastdeploy.input import video_utils

        class FakeWrapper:
            def __init__(self, path, *args, **kwargs):
                pass

            def __len__(self):
                return 60

            def get_avg_fps(self):
                return 30.0

        with patch.object(video_utils, "VideoReaderWrapper", FakeWrapper):
            reader, meta, path = read_video_paddlecodec("/fake/path.mp4")

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
        with self.assertLogs(level="WARNING"):
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
        with patch("fastdeploy.input.video_utils.sample_frames_paddleocr", wraps=sample_frames_paddleocr) as mock_fn:
            sample_frames(1, 4, 100, self.META, num_frames=8)
            mock_fn.assert_called_once()

    def test_qwen_variant_dispatched(self):
        with patch("fastdeploy.input.video_utils.sample_frames_qwen", wraps=sample_frames_qwen) as mock_fn:
            sample_frames(2, 4, 100, self.META, num_frames=8, variant="qwen")
            mock_fn.assert_called_once()

    def test_qwen_none_fps_converted_to_sentinel(self):
        """None fps/num_frames → converted to -1 before calling sample_frames_qwen."""
        with patch("fastdeploy.input.video_utils.sample_frames_qwen", return_value=np.array([])) as mock_fn:
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
