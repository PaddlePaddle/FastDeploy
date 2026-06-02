"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import ctypes
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TestFileStoreConfig(unittest.TestCase):
    """Tests for FileStoreConfig dataclass."""

    def test_default_values(self):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStoreConfig,
        )

        config = FileStoreConfig()
        self.assertEqual(config.namespace, "")
        self.assertEqual(config.tp_rank, 0)
        self.assertEqual(config.tp_size, 1)

    def test_custom_values(self):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStoreConfig,
        )

        config = FileStoreConfig(file_path="/tmp/test_store", namespace="ns1", tp_rank=2, tp_size=4)
        self.assertEqual(config.file_path, "/tmp/test_store")
        self.assertEqual(config.namespace, "ns1")
        self.assertEqual(config.tp_rank, 2)
        self.assertEqual(config.tp_size, 4)


class TestFileStoreInit(unittest.TestCase):
    """Tests for FileStore initialization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_init_creates_directory(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        new_dir = os.path.join(self.test_dir, "new_subdir")
        store = FileStore(file_path=new_dir)
        self.assertTrue(os.path.exists(new_dir))
        self.assertEqual(store.file_path, new_dir)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_init_with_namespace(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        store = FileStore(file_path=self.test_dir, namespace="my_ns")
        expected_path = os.path.join(self.test_dir, "my_ns")
        self.assertEqual(store.file_path, expected_path)
        self.assertTrue(os.path.exists(expected_path))

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_init_existing_directory(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        store = FileStore(file_path=self.test_dir)
        self.assertEqual(store.file_path, self.test_dir)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_init_none_file_path_raises(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        with self.assertRaises(ValueError) as ctx:
            FileStore(file_path=None)
        self.assertIn("file_path must be specified", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_init_non_zero_tp_rank_skips_mkdir(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        new_dir = os.path.join(self.test_dir, "non_existent")
        store = FileStore(file_path=new_dir, tp_rank=1)
        self.assertFalse(os.path.exists(new_dir))
        self.assertEqual(store.file_path, new_dir)


class TestFileStoreOperations(unittest.TestCase):
    """Tests for FileStore set/get/exists/clear operations."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_register_buffer_returns_none(self):
        self.assertIsNone(self.store.register_buffer(0, 0))

    def test_get_tensor_path(self):
        path = self.store._get_tensor_path("my_key")
        self.assertEqual(path, os.path.join(self.test_dir, "my_key.pd"))

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_exists_returns_correct_results(self, mock_logger):
        # Create a fake file for one key
        fake_path = os.path.join(self.test_dir, "key1.pd")
        with open(fake_path, "w") as f:
            f.write("data")

        result = self.store.exists(["key1", "key2"])
        self.assertTrue(result["key1"])
        self.assertFalse(result["key2"])

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("os.close")
    @patch("os.fsync")
    @patch("os.open", return_value=99)
    @patch("paddle.save")
    def test_set_saves_tensor(self, mock_paddle_save, mock_os_open, mock_fsync, mock_os_close, mock_logger):
        # Create a buffer with known data
        data = b"\x01\x02\x03\x04"
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        result = self.store.set("test_key", target_location=ptr, target_size=len(data))
        self.assertEqual(result, 0)
        mock_paddle_save.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_set_skips_existing_key(self, mock_logger):
        # Create the file so it already "exists"
        tensor_path = os.path.join(self.test_dir, "existing_key.pd")
        with open(tensor_path, "w") as f:
            f.write("data")

        data = b"\x01\x02\x03\x04"
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        result = self.store.set("existing_key", target_location=ptr, target_size=len(data))
        self.assertEqual(result, 0)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("paddle.save", side_effect=OSError("disk full"))
    def test_set_handles_save_failure(self, mock_paddle_save, mock_logger):
        data = b"\x01\x02\x03\x04"
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        result = self.store.set("fail_key", target_location=ptr, target_size=len(data))
        self.assertEqual(result, -1)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_get_nonexistent_key(self, mock_logger):
        data = b"\x00" * 10
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        result = self.store.get("no_such_key", target_location=ptr, target_size=10)
        self.assertEqual(result, -1)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("paddle.load")
    def test_get_invalid_target_size(self, mock_paddle_load, mock_logger):
        import paddle

        # Create the file so os.path.exists passes
        tensor_path = os.path.join(self.test_dir, "key_size.pd")
        with open(tensor_path, "w") as f:
            f.write("data")

        mock_paddle_load.return_value = paddle.to_tensor([1, 2, 3], dtype="uint8")

        data = b"\x00" * 10
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        result = self.store.get("key_size", target_location=ptr, target_size=0)
        self.assertEqual(result, -1)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("paddle.load")
    def test_get_success(self, mock_paddle_load, mock_logger):
        import paddle

        # Create the file so os.path.exists passes
        tensor_path = os.path.join(self.test_dir, "good_key.pd")
        with open(tensor_path, "w") as f:
            f.write("data")

        test_data = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_tensor = paddle.to_tensor(test_data, place="cpu")
        mock_paddle_load.return_value = mock_tensor

        # Allocate target buffer
        target_size = len(test_data)
        buf = ctypes.create_string_buffer(target_size)
        ptr = ctypes.addressof(buf)

        result = self.store.get("good_key", target_location=ptr, target_size=target_size)
        self.assertEqual(result, target_size)

        # Verify data was copied
        copied = ctypes.string_at(ptr, target_size)
        self.assertEqual(copied, test_data.tobytes())

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("paddle.load", side_effect=FileNotFoundError("not found"))
    def test_get_handles_load_failure(self, mock_paddle_load, mock_logger):
        tensor_path = os.path.join(self.test_dir, "bad_key.pd")
        with open(tensor_path, "w") as f:
            f.write("data")

        buf = ctypes.create_string_buffer(10)
        ptr = ctypes.addressof(buf)

        result = self.store.get("bad_key", target_location=ptr, target_size=10)
        self.assertEqual(result, -1)


class TestFileStoreBatchOperations(unittest.TestCase):
    """Tests for FileStore batch_set and batch_get."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_batch_set_length_mismatch(self, mock_logger):
        result = self.store.batch_set(
            keys=["k1", "k2"],
            target_locations=[100],
            target_sizes=[10, 20],
        )
        self.assertEqual(result, [-1, -1])

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("os.close")
    @patch("os.fsync")
    @patch("os.open", return_value=99)
    @patch("paddle.save")
    def test_batch_set_success(self, mock_paddle_save, mock_os_open, mock_fsync, mock_os_close, mock_logger):
        data1 = b"\x01\x02"
        data2 = b"\x03\x04"
        buf1 = ctypes.create_string_buffer(data1)
        buf2 = ctypes.create_string_buffer(data2)
        ptr1 = ctypes.addressof(buf1)
        ptr2 = ctypes.addressof(buf2)

        result = self.store.batch_set(
            keys=["k1", "k2"],
            target_locations=[ptr1, ptr2],
            target_sizes=[len(data1), len(data2)],
        )
        self.assertEqual(result, [0, 0])

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_batch_get_length_mismatch(self, mock_logger):
        result = self.store.batch_get(
            keys=["k1", "k2"],
            target_locations=[100],
            target_sizes=[10, 20],
        )
        self.assertEqual(result, [-1, -1])

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_batch_get_nonexistent_keys(self, mock_logger):
        buf1 = ctypes.create_string_buffer(10)
        buf2 = ctypes.create_string_buffer(10)
        ptr1 = ctypes.addressof(buf1)
        ptr2 = ctypes.addressof(buf2)

        result = self.store.batch_get(
            keys=["no_key1", "no_key2"],
            target_locations=[ptr1, ptr2],
            target_sizes=[10, 10],
        )
        self.assertEqual(result, [-1, -1])


class TestFileStoreQuery(unittest.TestCase):
    """Tests for FileStore query method."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_query_empty_keys(self, mock_logger):
        result = self.store.query(k_cache_keys=[], v_cache_keys=[])
        self.assertEqual(result, 0)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_query_none_keys(self, mock_logger):
        result = self.store.query(k_cache_keys=None, v_cache_keys=None)
        self.assertEqual(result, 0)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_query_with_matching_pairs(self, mock_logger):
        # Create files for k1 and v1 (a complete pair)
        with open(os.path.join(self.test_dir, "k1.pd"), "w") as f:
            f.write("data")
        with open(os.path.join(self.test_dir, "v1.pd"), "w") as f:
            f.write("data")
        # Only create k2, not v2 (incomplete pair)
        with open(os.path.join(self.test_dir, "k2.pd"), "w") as f:
            f.write("data")

        result = self.store.query(k_cache_keys=["k1", "k2"], v_cache_keys=["v1", "v2"])
        self.assertEqual(result, 1)  # Only k1/v1 pair is complete

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_query_mismatched_lengths_returns_zero(self, mock_logger):
        # AssertionError is caught by the except block in query(), returns 0
        result = self.store.query(k_cache_keys=["k1"], v_cache_keys=["v1", "v2"])
        self.assertEqual(result, 0)


class TestFileStoreClear(unittest.TestCase):
    """Tests for FileStore clear method."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_clear_removes_pd_files(self, mock_logger):
        # Create some .pd files
        for name in ["a.pd", "b.pd", "c.txt"]:
            with open(os.path.join(self.test_dir, name), "w") as f:
                f.write("data")

        result = self.store.clear()
        self.assertTrue(result)
        # .pd files should be removed
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "a.pd")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "b.pd")))
        # Non-.pd files should remain
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "c.txt")))

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_clear_refuses_dangerous_paths(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
            FileStore,
        )

        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            store = FileStore(file_path=self.test_dir)
        store.file_path = "/"
        with self.assertRaises(RuntimeError) as ctx:
            store.clear()
        self.assertIn("Refuse to clear dangerous path", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    @patch("subprocess.run", side_effect=__import__("subprocess").CalledProcessError(1, "rm"))
    def test_clear_handles_subprocess_failure(self, mock_run, mock_logger):
        result = self.store.clear()
        self.assertFalse(result)


class TestFileStoreCopyTensorToPtr(unittest.TestCase):
    """Tests for _copy_tensor_to_ptr helper."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_copy_non_tensor_returns_negative(self, mock_logger):
        buf = ctypes.create_string_buffer(10)
        ptr = ctypes.addressof(buf)
        result = self.store._copy_tensor_to_ptr("not a tensor", ptr, 10)
        self.assertEqual(result, -1)

    @patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger")
    def test_copy_size_mismatch_returns_negative(self, mock_logger):
        import paddle

        tensor = paddle.to_tensor([1, 2], dtype="uint8")  # 2 bytes
        buf = ctypes.create_string_buffer(100)
        ptr = ctypes.addressof(buf)
        # Request more bytes than tensor has
        result = self.store._copy_tensor_to_ptr(tensor, ptr, 100)
        self.assertEqual(result, -1)

    def test_copy_success(self):
        import paddle

        test_data = np.array([10, 20, 30, 40], dtype=np.uint8)
        tensor = paddle.to_tensor(test_data, place="cpu")
        buf = ctypes.create_string_buffer(4)
        ptr = ctypes.addressof(buf)

        result = self.store._copy_tensor_to_ptr(tensor, ptr, 4)
        self.assertEqual(result, 4)
        copied = ctypes.string_at(ptr, 4)
        self.assertEqual(copied, test_data.tobytes())


class TestFileStoreTensorFromPtr(unittest.TestCase):
    """Tests for _tensor_from_ptr helper."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch("fastdeploy.cache_manager.transfer_factory.file_store.file_store.logger"):
            from fastdeploy.cache_manager.transfer_factory.file_store.file_store import (
                FileStore,
            )

            self.store = FileStore(file_path=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tensor_from_ptr(self):
        data = b"\x01\x02\x03\x04"
        buf = ctypes.create_string_buffer(data)
        ptr = ctypes.addressof(buf)

        tensor = self.store._tensor_from_ptr(ptr, len(data))
        result = tensor.numpy()
        expected = np.frombuffer(data, dtype="uint8")
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
