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
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Mock librosa and soundfile before importing the module under test
mock_librosa = MagicMock()
mock_librosa.__spec__ = MagicMock()
mock_soundfile = MagicMock()
mock_soundfile.__spec__ = MagicMock()
sys.modules.setdefault("librosa", mock_librosa)
sys.modules.setdefault("soundfile", mock_soundfile)

from fastdeploy.multimodal.audio import AudioMediaIO, resample_audio  # noqa: E402


class TestResampleAudio(unittest.TestCase):
    """Test resample_audio function."""

    def setUp(self):
        mock_librosa.reset_mock()

    def test_resample_calls_librosa(self):
        """resample_audio delegates to librosa.resample with correct args."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        expected = np.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)
        mock_librosa.resample.return_value = expected

        result = resample_audio(audio, orig_sr=16000, target_sr=32000)

        mock_librosa.resample.assert_called_once_with(audio, orig_sr=16000, target_sr=32000)
        np.testing.assert_array_equal(result, expected)

    def test_resample_same_sr(self):
        """resample_audio works when orig_sr equals target_sr."""
        audio = np.zeros(100, dtype=np.float32)
        mock_librosa.resample.return_value = audio

        result = resample_audio(audio, orig_sr=16000, target_sr=16000)
        mock_librosa.resample.assert_called_once()
        np.testing.assert_array_equal(result, audio)


class TestAudioMediaIOLoadBytes(unittest.TestCase):
    """Test AudioMediaIO.load_bytes method."""

    def setUp(self):
        mock_librosa.reset_mock()

    def test_load_bytes_returns_audio_and_sr(self):
        """load_bytes returns (ndarray, sample_rate) tuple."""
        audio_data = np.array([0.5, -0.5], dtype=np.float32)
        mock_librosa.load.return_value = (audio_data, 22050.0)

        io = AudioMediaIO()
        result = io.load_bytes(b"fake_wav_data")

        self.assertEqual(result[1], 22050.0)
        np.testing.assert_array_equal(result[0], audio_data)
        # Verify sr=None was passed
        call_args = mock_librosa.load.call_args
        self.assertIsNone(call_args[1]["sr"])


class TestAudioMediaIOLoadBase64(unittest.TestCase):
    """Test AudioMediaIO.load_base64 method."""

    def setUp(self):
        mock_librosa.reset_mock()

    def test_load_base64_decodes_and_loads(self):
        """load_base64 decodes base64 then calls load_bytes."""
        raw_bytes = b"fake_audio_content"
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")

        audio_data = np.array([1.0, 0.0], dtype=np.float32)
        mock_librosa.load.return_value = (audio_data, 44100.0)

        io = AudioMediaIO()
        result = io.load_base64("audio/wav", b64_str)

        self.assertEqual(result[1], 44100.0)
        np.testing.assert_array_equal(result[0], audio_data)
        mock_librosa.load.assert_called_once()


class TestAudioMediaIOLoadFile(unittest.TestCase):
    """Test AudioMediaIO.load_file method."""

    def setUp(self):
        mock_librosa.reset_mock()

    def test_load_file_calls_librosa_with_path(self):
        """load_file passes filepath to librosa.load with sr=None."""
        audio_data = np.array([0.1], dtype=np.float32)
        mock_librosa.load.return_value = (audio_data, 16000.0)

        io = AudioMediaIO()
        filepath = Path("/tmp/test.wav")
        result = io.load_file(filepath)

        mock_librosa.load.assert_called_once_with(filepath, sr=None)
        self.assertEqual(result[1], 16000.0)
        np.testing.assert_array_equal(result[0], audio_data)


class TestAudioMediaIOEncodeBase64(unittest.TestCase):
    """Test AudioMediaIO.encode_base64 method."""

    def setUp(self):
        mock_soundfile.reset_mock()

    def test_encode_base64_produces_valid_base64(self):
        """encode_base64 writes WAV and returns base64 string."""
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        sr = 16000.0

        # Mock soundfile.write to write known bytes into the buffer
        def fake_write(buffer, audio, sample_rate, format):
            buffer.write(b"RIFF_fake_wav_data")

        mock_soundfile.write.side_effect = fake_write

        io = AudioMediaIO()
        result = io.encode_base64((audio_data, sr))

        # Verify it's a valid base64 string
        decoded = base64.b64decode(result)
        self.assertEqual(decoded, b"RIFF_fake_wav_data")

        # Verify soundfile.write was called with correct args
        call_args = mock_soundfile.write.call_args
        np.testing.assert_array_equal(call_args[0][1], audio_data)
        self.assertEqual(call_args[0][2], sr)
        self.assertEqual(call_args[1]["format"], "WAV")


if __name__ == "__main__":
    unittest.main()
