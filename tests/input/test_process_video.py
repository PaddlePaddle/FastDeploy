import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image as PILImage

import fastdeploy.input.ernie4_5_vl_processor.process_video as process_video_module
from fastdeploy.input.ernie4_5_vl_processor.process_video import (
    get_frame_indices,
    read_frames_decord,
    read_video_decord,
)


class _MockFrame:
    """Lightweight frame wrapper that mimics the real frame object."""

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        """Return the underlying numpy array."""
        return self._arr


class MockVideoReaderWrapper:
    """
    Simple mock implementation of a video reader:

    - __len__ returns the total number of frames
    - __getitem__ returns a _MockFrame(arr)
    - get_avg_fps() returns fps
    - Specific indices can be configured to raise errors in __getitem__
    """

    def __init__(
        self,
        src,
        num_threads=1,
        vlen=12,
        fps=6,
        fail_indices=None,
        h=4,
        w=5,
        c=3,
    ):
        self.src = src
        self._vlen = vlen
        self._fps = fps
        self._fail = set(fail_indices or [])
        self._h, self._w, self._c = h, w, c

    def __len__(self):
        return self._vlen

    def get_avg_fps(self):
        return self._fps

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._vlen:
            raise IndexError("index out of range")
        if idx in self._fail:
            raise ValueError(f"forced fail at {idx}")
        # Create a frame whose pixel value encodes the index (for easy debugging)
        arr = np.zeros((self._h, self._w, self._c), dtype=np.uint8)
        arr[:] = idx % 255
        return _MockFrame(arr)


class TestReadVideoDecord(unittest.TestCase):
    def test_read_video_decord_with_wrapper(self):
        """Test passing an existing VideoReaderWrapper instance directly."""
        # Patch VideoReaderWrapper in the target module so isinstance checks use our mock class
        with patch.object(process_video_module, "VideoReaderWrapper", MockVideoReaderWrapper):
            mock_reader = MockVideoReaderWrapper("dummy", vlen=10, fps=5)
            reader, meta, path = read_video_decord(mock_reader, save_to_disk=False)

        self.assertIs(reader, mock_reader)
        self.assertEqual(meta["fps"], 5)
        self.assertEqual(meta["num_of_frame"], 10)
        self.assertTrue(math.isclose(meta["duration"], 10 / 5, rel_tol=1e-6))
        # The original reader object should be returned unchanged
        self.assertIs(path, mock_reader)

    def test_read_video_decord_with_bytes(self):
        """Test that bytes input is wrapped into BytesIO and passed to VideoReaderWrapper."""
        with patch.object(process_video_module, "VideoReaderWrapper", MockVideoReaderWrapper):
            data = b"\x00\x01\x02\x03"
            reader, meta, path = read_video_decord(data, save_to_disk=False)

        self.assertIsInstance(reader, MockVideoReaderWrapper)
        self.assertEqual(meta["fps"], 6)
        self.assertEqual(meta["num_of_frame"], 12)
        self.assertTrue(math.isclose(meta["duration"], 12 / 6, rel_tol=1e-6))
        self.assertIsInstance(path, io.BytesIO)


class TestGetFrameIndices(unittest.TestCase):
    def test_by_target_frames_middle(self):
        """Test target_frames mode with 'middle' sampling strategy."""
        vlen = 12
        out = get_frame_indices(
            vlen=vlen,
            target_frames=4,
            target_fps=-1,
            frames_sample="middle",
            input_fps=-1,
        )
        # 12 frames split into 4 segments -> midpoints [1, 4, 7, 10]
        self.assertEqual(out, [1, 4, 7, 10])

    def test_by_target_frames_leading(self):
        """Test target_frames mode with 'leading' sampling strategy."""
        vlen = 10
        out = get_frame_indices(
            vlen=vlen,
            target_frames=5,
            target_fps=-1,
            frames_sample="leading",
            input_fps=-1,
        )
        # 10 frames split into 5 segments -> segment starts [0, 2, 4, 6, 8]
        self.assertEqual(out, [0, 2, 4, 6, 8])

    def test_by_target_frames_rand(self):
        """Test target_frames mode with 'rand' sampling strategy."""
        vlen = 10
        out = get_frame_indices(
            vlen=vlen,
            target_frames=4,
            target_fps=-1,
            frames_sample="rand",
            input_fps=-1,
        )
        self.assertEqual(len(out), 4)
        self.assertTrue(all(0 <= i < vlen for i in out))

    def test_by_target_frames_fix_start(self):
        """Test target_frames mode with a fixed start offset."""
        vlen = 10
        out = get_frame_indices(
            vlen=vlen,
            target_frames=5,
            target_fps=-1,
            frames_sample="middle",  # overridden by fix_start
            fix_start=1,
            input_fps=-1,
        )
        # Segment starts [0, 2, 4, 6, 8] -> +1 => [1, 3, 5, 7, 9]
        self.assertEqual(out, [1, 3, 5, 7, 9])

    def test_target_frames_greater_than_vlen(self):
        """Test that target_frames > vlen falls back to using vlen samples."""
        vlen = 5
        out = get_frame_indices(
            vlen=vlen,
            target_frames=10,
            target_fps=-1,
            frames_sample="middle",
            input_fps=-1,
        )
        self.assertEqual(len(out), vlen)
        self.assertTrue(all(0 <= i < vlen for i in out))

    def test_by_target_fps_middle(self):
        """Test target_fps mode with 'middle' sampling strategy."""
        vlen, in_fps = 12, 6
        out = get_frame_indices(
            vlen=vlen,
            target_frames=-1,
            target_fps=2,
            frames_sample="middle",
            input_fps=in_fps,
        )
        # Roughly 4 frames expected
        self.assertTrue(3 <= len(out) <= 5)
        self.assertTrue(all(0 <= i < vlen for i in out))

    def test_by_target_fps_leading(self):
        """Test target_fps mode with 'leading' sampling strategy."""
        vlen, in_fps = 12, 6
        out = get_frame_indices(
            vlen=vlen,
            target_frames=-1,
            target_fps=2,
            frames_sample="leading",
            input_fps=in_fps,
        )
        self.assertTrue(3 <= len(out) <= 5)
        self.assertTrue(all(0 <= i < vlen for i in out))

    def test_by_target_fps_rand(self):
        """Test target_fps mode with 'rand' sampling strategy."""
        vlen, in_fps = 12, 6
        out = get_frame_indices(
            vlen=vlen,
            target_frames=-1,
            target_fps=2,
            frames_sample="rand",
            input_fps=in_fps,
        )
        self.assertTrue(3 <= len(out) <= 5)
        self.assertTrue(all(0 <= i < vlen for i in out))

    def test_invalid_both_negative(self):
        """Test that both target_frames and target_fps being negative raises ValueError."""
        with self.assertRaises(ValueError):
            get_frame_indices(
                vlen=10,
                target_frames=-1,
                target_fps=-1,
                frames_sample="middle",
            )

    def test_invalid_both_specified(self):
        """Test that specifying both target_frames and target_fps raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_frame_indices(
                vlen=10,
                target_frames=4,
                target_fps=2,
                frames_sample="middle",
                input_fps=6,
            )

    def test_invalid_target_fps_missing_input(self):
        """Test that target_fps > 0 with invalid input_fps raises AssertionError."""
        with self.assertRaises(AssertionError):
            get_frame_indices(
                vlen=10,
                target_frames=-1,
                target_fps=2,
                frames_sample="middle",
                input_fps=-1,
            )


class TestReadFramesDecord(unittest.TestCase):
    def test_basic_read_no_save(self):
        """Test normal frame reading without saving to disk."""
        reader = MockVideoReaderWrapper("dummy", vlen=8, fps=4)
        meta = {"fps": 4, "duration": 8 / 4, "num_of_frame": 8}

        ret, idxs, ts = read_frames_decord(
            video_path="dummy",
            video_reader=reader,
            video_meta=meta,
            target_frames=4,
            frames_sample="middle",
            save_to_disk=False,
        )

        # Should return 4 PIL.Image instances
        self.assertEqual(len(ret), 4)
        for img in ret:
            self.assertIsInstance(img, PILImage.Image)

        self.assertEqual(idxs, [0, 2, 4, 6])
        dur = meta["duration"]
        n = meta["num_of_frame"]
        for i, t in zip(idxs, ts):
            self.assertTrue(math.isclose(t, i * dur / n, rel_tol=1e-6))

    def test_read_and_save_to_disk(self):
        """Test reading frames and saving them as PNG files on disk."""
        reader = MockVideoReaderWrapper("dummy", vlen=4, fps=2)
        meta = {"fps": 2, "duration": 4 / 2, "num_of_frame": 4}

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                process_video_module,
                "get_filename",
                return_value="det_id",
            ),
        ):
            ret, idxs, ts = read_frames_decord(
                video_path="dummy",
                video_reader=reader,
                video_meta=meta,
                target_frames=2,
                frames_sample="leading",
                save_to_disk=True,
                cache_dir=tmpdir,
            )

            self.assertEqual(len(ret), 2)
            for i, pth in enumerate(ret):
                self.assertIsInstance(pth, str)
                self.assertTrue(os.path.exists(pth))
                self.assertEqual(os.path.basename(pth), f"{i}.png")

    def test_fallback_previous_success(self):
        """Test that a failed frame read falls back to a previous valid frame when possible."""
        reader = MockVideoReaderWrapper("dummy", vlen=10, fps=5, fail_indices={3})
        meta = {"fps": 5, "duration": 10 / 5, "num_of_frame": 10}
        idxs = [1, 2, 3, 6]

        ret, new_idxs, ts = read_frames_decord(
            video_path="dummy",
            video_reader=reader,
            video_meta=meta,
            frame_indices=idxs.copy(),
            save_to_disk=False,
            tol=5,
        )

        # Index 3 fails and should be replaced by 2 or 4 (previous/next search)
        self.assertIn(new_idxs[2], (2, 4))
        self.assertEqual(len(ret), 4)

    def test_fallback_next_when_prev_fails(self):
        """Test that when current and previous frames fail, a later frame is used as fallback."""
        reader = MockVideoReaderWrapper("dummy", vlen=10, fps=5, fail_indices={2, 3})
        meta = {"fps": 5, "duration": 10 / 5, "num_of_frame": 10}
        idxs = [1, 2, 3, 6]

        ret, new_idxs, ts = read_frames_decord(
            video_path="dummy",
            video_reader=reader,
            video_meta=meta,
            frame_indices=idxs.copy(),
            save_to_disk=False,
            tol=5,
        )

        # Frame 3 should eventually be replaced by 4
        self.assertEqual(new_idxs[2], 4)
        self.assertEqual(len(ret), 4)

    def test_len_assert_when_no_fallback(self):
        """Test that assertion is triggered when no valid fallback frame can be found."""

        class FailAllAroundReader(MockVideoReaderWrapper):
            """Reader that fails on index 1 and has too small length to find fallback."""

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._vlen = 2
                self._fps = 2
                self._fail = {1}

            def __getitem__(self, idx):
                if idx in self._fail:
                    raise ValueError("fail hard")
                return super().__getitem__(idx)

        reader = FailAllAroundReader("dummy")
        meta = {"fps": 2, "duration": 2 / 2, "num_of_frame": 2}

        # Request 2 frames: index 0 succeeds, index 1 always fails,
        # and tol=0 disallows searching neighbors -> stack and length assertion should fail
        with self.assertRaises(AssertionError):
            read_frames_decord(
                video_path="dummy",
                video_reader=reader,
                video_meta=meta,
                target_frames=2,
                frames_sample="leading",
                save_to_disk=False,
                tol=0,
            )


if __name__ == "__main__":
    unittest.main()
