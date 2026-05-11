"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import numpy as np
import pytest

from fastdeploy.inter_communicator.ipc_signal import IPCSignal, shared_memory_exists


class TestSharedMemoryExists(unittest.TestCase):
    """Test cases for shared_memory_exists function."""

    def test_returns_false_for_nonexistent_memory(self):
        """Test that shared_memory_exists returns False for non-existent shared memory."""
        result = shared_memory_exists(f"nonexistent_shm_{time.time()}")
        self.assertFalse(result)

    def test_returns_true_for_existing_memory(self):
        """Test that shared_memory_exists returns True for existing shared memory."""
        name = f"test_shm_{time.time()}"
        shm = SharedMemory(name=name, create=True, size=1024)
        try:
            result = shared_memory_exists(name)
            self.assertTrue(result)
        finally:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def test_shared_memory_exists_logs_error_on_unexpected_exception(self):
        """Test shared_memory_exists logs error with traceback on unexpected exception."""
        import fastdeploy.inter_communicator.ipc_signal as ipc_module

        with patch("fastdeploy.inter_communicator.ipc_signal.SharedMemory", side_effect=OSError("unexpected")):
            with patch.object(ipc_module.llm_logger, "error") as mock_error:
                result = shared_memory_exists("any_name")

        self.assertFalse(result)
        mock_error.assert_called_once()
        error_msg = mock_error.call_args[0][0]
        self.assertIn("Unexpected error", error_msg)
        self.assertIn("unexpected", error_msg)


@pytest.mark.parametrize(
    "dtype,shape,initial_value",
    [
        (np.int32, (10,), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (np.float32, (5,), [0.0, 1.5, 2.5, 3.5, 4.5]),
        (np.int64, (3, 3), [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        (np.uint8, (4,), [0, 127, 200, 255]),
    ],
)
def test_ipc_signal_create_with_array(dtype, shape, initial_value):
    """Test IPCSignal creation with numpy array."""
    name = f"test_ipc_signal_{time.time()}"
    array = np.array(initial_value, dtype=dtype)

    signal = IPCSignal(name=name, array=array, dtype=dtype, create=True)
    try:
        # Verify value is initialized correctly
        np.testing.assert_array_equal(signal.value, array)
        np.testing.assert_equal(signal.value.dtype, dtype)

        # Verify shared memory exists
        assert shared_memory_exists(name)
    finally:
        try:
            signal.clear()
        except Exception:
            pass


class TestIPCSignal(unittest.TestCase):
    """Test cases for IPCSignal class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_name_base = f"test_ipc_signal_{time.time()}"
        self._signals_to_clean = []

    def tearDown(self):
        """Clean up all tracked signals."""
        for signal in self._signals_to_clean:
            try:
                signal.clear()
            except Exception:
                pass

    def _track(self, signal):
        """Register a signal for automatic cleanup in tearDown."""
        self._signals_to_clean.append(signal)
        return signal

    def test_create_with_suffix(self):
        """Test IPCSignal creation with suffix."""
        name = self.test_name_base
        suffix = 123

        array = np.array([1, 2, 3], dtype=np.int32)
        signal = self._track(IPCSignal(name=name, array=array, dtype=np.int32, suffix=suffix, create=True))

        expected_name = f"{name}.{suffix}"
        self.assertTrue(shared_memory_exists(expected_name))
        np.testing.assert_array_equal(signal.value, array)

    def test_attach_to_existing(self):
        """Test IPCSignal attaching to existing shared memory."""
        name = f"{self.test_name_base}_attach"
        array = np.array([10, 20, 30], dtype=np.int64)

        # Create shared memory
        signal1 = self._track(IPCSignal(name=name, array=array, dtype=np.int64, create=True))
        signal1.value[0] = 99  # Modify value

        # Attach to existing
        signal2 = IPCSignal(name=name, array=array, dtype=np.int64, create=False)

        # Verify value is shared
        self.assertEqual(signal2.value[0], 99)
        np.testing.assert_array_equal(signal2.value, signal1.value)

    def test_dtype_mismatch_raises_assertion(self):
        """Test that dtype mismatch raises AssertionError."""
        name = f"{self.test_name_base}_mismatch"
        array = np.array([1, 2, 3], dtype=np.int32)

        with self.assertRaises(AssertionError):
            IPCSignal(name=name, array=array, dtype=np.float32, create=True)

    def test_non_numpy_array_raises_assertion(self):
        """Test that non-numpy array raises AssertionError."""
        name = f"{self.test_name_base}_non_array"

        with self.assertRaises(AssertionError):
            IPCSignal(name=name, array=[1, 2, 3], dtype=np.int32, create=True)

    def test_create_with_shm_size(self):
        """Test IPCSignal creation with shm_size (no array)."""
        name = f"{self.test_name_base}_size"

        signal = self._track(IPCSignal(name=name, shm_size=1024, create=True))

        # Verify signal is created but value is None (no array template)
        self.assertTrue(shared_memory_exists(name))
        self.assertIsNone(signal.value)

    def test_attach_with_shm_size(self):
        """Test IPCSignal attach with shm_size (no array)."""
        name = f"{self.test_name_base}_attach_size"

        # Create
        self._track(IPCSignal(name=name, shm_size=512, create=True))

        # Attach
        signal2 = IPCSignal(name=name, shm_size=512, create=False)

        self.assertTrue(shared_memory_exists(name))
        self.assertIsNone(signal2.value)

    def test_shm_size_required_without_array_and_dtype(self):
        """Test that shm_size is required when array and dtype are None."""
        name = f"{self.test_name_base}_no_size"

        with self.assertRaises(AssertionError):
            IPCSignal(name=name, create=True)

    def test_clear_removes_shared_memory(self):
        """Test that clear() properly removes shared memory."""
        name = f"{self.test_name_base}_clear"
        array = np.array([1, 2, 3], dtype=np.int32)

        signal = IPCSignal(name=name, array=array, dtype=np.int32, create=True)
        self.assertTrue(shared_memory_exists(name))

        signal.clear()
        self.assertFalse(shared_memory_exists(name))

    def test_clear_idempotent(self):
        """Test that clear() can be called multiple times safely."""
        name = f"{self.test_name_base}_idempotent"
        array = np.array([1, 2, 3], dtype=np.int32)

        signal = IPCSignal(name=name, array=array, dtype=np.int32, create=True)

        # Should not raise exception
        signal.clear()
        signal.clear()  # Call again

    def test_value_sharing_between_processes_mock(self):
        """Test that value is shared (mocked for unit test)."""
        name = f"{self.test_name_base}_shared"
        array = np.array([100, 200, 300], dtype=np.int64)

        signal1 = self._track(IPCSignal(name=name, array=array, dtype=np.int64, create=True))
        signal2 = IPCSignal(name=name, array=array, dtype=np.int64, create=False)

        # Modify through signal1
        signal1.value[0] = 999
        signal1.value[1] = 888
        signal1.value[2] = 777

        # Verify signal2 sees changes
        self.assertEqual(signal2.value[0], 999)
        self.assertEqual(signal2.value[1], 888)
        self.assertEqual(signal2.value[2], 777)

    def test_multiple_array_creation_replaces_existing(self):
        """Test that creating with same name replaces existing shared memory."""
        name = f"{self.test_name_base}_replace"
        array1 = np.array([1, 2, 3], dtype=np.int32)
        array2 = np.array([4, 5, 6], dtype=np.int32)

        signal1 = IPCSignal(name=name, array=array1, dtype=np.int32, create=True)
        signal1.clear()

        signal2 = self._track(IPCSignal(name=name, array=array2, dtype=np.int32, create=True))

        np.testing.assert_array_equal(signal2.value, array2)

    def test_clear_closes_and_unlinks(self):
        """Test that clear() both closes and unlinks the shared memory."""
        name = f"{self.test_name_base}_unlink"
        array = np.array([1, 2, 3], dtype=np.int32)

        signal = IPCSignal(name=name, array=array, dtype=np.int32, create=True)

        # After clear, the shared memory should be removed
        signal.clear()
        self.assertFalse(shared_memory_exists(name))

        # Attempting to attach should fail
        try:
            _ = SharedMemory(name=name, create=False)
            self.fail("Should have raised FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_raw_buffer_read_write_with_shm_size(self):
        """Test raw buffer read/write in shm_size mode."""
        name = f"{self.test_name_base}_raw_buf"
        data = b"hello ipc signal"

        signal1 = self._track(IPCSignal(name=name, shm_size=1024, create=True))
        signal1.shm.buf[: len(data)] = data

        signal2 = IPCSignal(name=name, shm_size=1024, create=False)
        self.assertEqual(bytes(signal2.shm.buf[: len(data)]), data)

    def test_create_overwrites_existing_without_clear(self):
        """Test that create=True on existing name auto-unlinks and recreates."""
        name = f"{self.test_name_base}_overwrite"
        array1 = np.array([1, 2, 3], dtype=np.int32)
        array2 = np.array([7, 8, 9], dtype=np.int32)

        # Create first signal, do NOT clear it
        IPCSignal(name=name, array=array1, dtype=np.int32, create=True)

        # Create again with same name — should auto-unlink old and recreate
        signal2 = self._track(IPCSignal(name=name, array=array2, dtype=np.int32, create=True))
        np.testing.assert_array_equal(signal2.value, array2)

    def test_attach_nonexistent_raises_error(self):
        """Test that create=False on non-existent shm raises FileNotFoundError."""
        name = f"nonexistent_signal_{time.time()}"
        array = np.array([1, 2, 3], dtype=np.int32)

        with self.assertRaises(FileNotFoundError):
            IPCSignal(name=name, array=array, dtype=np.int32, create=False)


class TestIPCSignalEdgeCases(unittest.TestCase):
    """Test edge cases for IPCSignal."""

    def test_empty_array_raises_error(self):
        """Test IPCSignal with empty array raises ValueError due to nbytes=0."""
        name = f"test_empty_array_{time.time()}"
        array = np.array([], dtype=np.int32)

        with self.assertRaises(ValueError):
            IPCSignal(name=name, array=array, dtype=np.int32, create=True)

    def test_large_array(self):
        """Test IPCSignal with large array."""
        name = f"test_large_array_{time.time()}"
        size = 10000
        array = np.arange(size, dtype=np.int64)

        signal = IPCSignal(name=name, array=array, dtype=np.int64, create=True)
        try:
            np.testing.assert_array_equal(signal.value, array)
        finally:
            try:
                signal.clear()
            except Exception:
                pass

    def test_multidimensional_array(self):
        """Test IPCSignal with multidimensional array."""
        name = f"test_multi_array_{time.time()}"
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

        signal = IPCSignal(name=name, array=array, dtype=np.int32, create=True)
        try:
            self.assertEqual(signal.value.shape, (3, 3))
            np.testing.assert_array_equal(signal.value, array)
        finally:
            try:
                signal.clear()
            except Exception:
                pass

    def test_different_numeric_types(self):
        """Test IPCSignal with different numeric types."""
        name_base = f"test_types_{time.time()}"

        test_cases = [
            (np.int8, [1, 2, 3]),
            (np.int16, [1000, 2000, 3000]),
            (np.int32, [100000, 200000, 300000]),
            (np.int64, [1000000000, 2000000000, 3000000000]),
            (np.float32, [1.5, 2.5, 3.5]),
            (np.float64, [1.123456789, 2.987654321, 3.5]),
        ]

        for i, (dtype, values) in enumerate(test_cases):
            name = f"{name_base}_{i}"
            array = np.array(values, dtype=dtype)
            signal = IPCSignal(name=name, array=array, dtype=dtype, create=True)
            try:
                np.testing.assert_array_equal(signal.value, array)
            finally:
                try:
                    signal.clear()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
