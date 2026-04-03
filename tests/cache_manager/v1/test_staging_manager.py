"""
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

Unit tests for StagingManager class.

Tests cover:
- Initialization and lifecycle (initialize / shutdown)
- Staging bytes computation (compute_staging_bytes / total_staging_bytes)
- Gather / scatter correctness (roundtrip via ctypes buffers)
- batch_set_block / batch_get_block with mocked StorageConnector
- Chunking behavior when batch exceeds staging_batch_size
"""

import ctypes
import unittest
from unittest.mock import Mock


class TestStagingManagerInit(unittest.TestCase):
    """Test StagingManager initialization and lifecycle."""

    def _make_manager(self, batch_size=4):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        connector = Mock()
        connector.register_buffer = Mock()
        return StagingManager(connector, staging_batch_size=batch_size), connector

    def test_not_initialized_by_default(self):
        mgr, _ = self._make_manager()
        self.assertFalse(mgr.initialized)

    def test_initialize_allocates_buffers(self):
        mgr, connector = self._make_manager(batch_size=2)
        strides = {"key": 64, "value": 64}

        with unittest.mock.patch(
            "fastdeploy.cache_manager.ops.cuda_host_alloc",
            side_effect=lambda size: size,  # return size as fake ptr
        ) as mock_alloc:
            mgr.initialize(num_layers=4, strides=strides)

        self.assertTrue(mgr.initialized)
        # 2 kinds x 2 directions = 4 buffers
        self.assertEqual(mock_alloc.call_count, 4)
        self.assertEqual(connector.register_buffer.call_count, 4)
        # Each buffer: batch_size(2) * num_layers(4) * stride(64) = 512
        for c in mock_alloc.call_args_list:
            self.assertEqual(c[0][0], 512)

    def test_double_initialize_is_noop(self):
        mgr, _ = self._make_manager(batch_size=2)
        with unittest.mock.patch(
            "fastdeploy.cache_manager.ops.cuda_host_alloc",
            return_value=1000,
        ) as mock_alloc:
            mgr.initialize(num_layers=2, strides={"key": 32, "value": 32})
            count1 = mock_alloc.call_count
            mgr.initialize(num_layers=2, strides={"key": 32, "value": 32})
            self.assertEqual(mock_alloc.call_count, count1)

    def test_shutdown_frees_buffers(self):
        mgr, _ = self._make_manager(batch_size=2)
        with unittest.mock.patch(
            "fastdeploy.cache_manager.ops.cuda_host_alloc",
            return_value=1000,
        ):
            mgr.initialize(num_layers=2, strides={"key": 32, "value": 32})

        with unittest.mock.patch(
            "fastdeploy.cache_manager.ops.cuda_host_free",
        ) as mock_free:
            mgr.shutdown()

        self.assertFalse(mgr.initialized)
        self.assertEqual(mock_free.call_count, 4)


class TestStagingBytesComputation(unittest.TestCase):
    """Test compute_staging_bytes and total_staging_bytes."""

    def test_compute_staging_bytes(self):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        mgr = StagingManager(Mock(), staging_batch_size=8)
        strides = {"key": 100, "value": 200}
        # 2 directions * 8 blocks * 4 layers * (100 + 200) = 2 * 8 * 4 * 300 = 19200
        result = mgr.compute_staging_bytes(num_layers=4, strides=strides)
        self.assertEqual(result, 19200)

    def test_total_staging_bytes_after_init(self):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        mgr = StagingManager(Mock(), staging_batch_size=8)
        with unittest.mock.patch(
            "fastdeploy.cache_manager.ops.cuda_host_alloc",
            return_value=1000,
        ):
            mgr.initialize(num_layers=4, strides={"key": 100, "value": 200})
        self.assertEqual(mgr.total_staging_bytes(), 19200)


class TestGatherScatterRoundtrip(unittest.TestCase):
    """Test _gather_block and _scatter_block correctness using real ctypes buffers."""

    def setUp(self):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        self.num_layers = 3
        self.stride = 16  # bytes per layer per block
        self.batch_size = 2
        self.num_blocks = 4

        connector = Mock()
        connector.register_buffer = Mock()
        self.mgr = StagingManager(connector, staging_batch_size=self.batch_size)

        # Allocate real ctypes buffers for host (per-layer) and staging
        self.host_ptrs = []
        self._host_bufs = []
        for _ in range(self.num_layers):
            buf = ctypes.create_string_buffer(self.num_blocks * self.stride)
            self._host_bufs.append(buf)
            self.host_ptrs.append(ctypes.addressof(buf))

        # Manually set up staging manager internals (bypass cuda_host_alloc)
        staging_size = self.batch_size * self.num_layers * self.stride
        self._staging_buf = ctypes.create_string_buffer(staging_size)
        staging_ptr = ctypes.addressof(self._staging_buf)

        self.mgr._num_layers = self.num_layers
        self.mgr._strides = {"key": self.stride}
        self.mgr._bufs = {
            "write_key": staging_ptr,
            "read_key": staging_ptr,
        }
        self.mgr._initialized = True

    def test_gather_then_scatter_preserves_data(self):
        """Write known data to host, gather to staging, clear host, scatter back, verify."""
        # Fill host buffers with known pattern: layer_idx * 10 + block_id
        block_id = 2
        for layer_idx in range(self.num_layers):
            offset = block_id * self.stride
            data = bytes([layer_idx * 10 + block_id] * self.stride)
            ctypes.memmove(self.host_ptrs[layer_idx] + offset, data, self.stride)

        # Gather block 2 into staging at batch_offset=0
        self.mgr._gather_block("write", "key", 0, block_id, self.host_ptrs)

        # Clear host block 2
        for layer_idx in range(self.num_layers):
            offset = block_id * self.stride
            ctypes.memset(self.host_ptrs[layer_idx] + offset, 0, self.stride)

        # Scatter from staging back to host block 2
        self.mgr._scatter_block("write", "key", 0, block_id, self.host_ptrs)

        # Verify data matches original
        for layer_idx in range(self.num_layers):
            offset = block_id * self.stride
            expected = bytes([layer_idx * 10 + block_id] * self.stride)
            actual = ctypes.string_at(self.host_ptrs[layer_idx] + offset, self.stride)
            self.assertEqual(actual, expected, f"Mismatch at layer {layer_idx}")


class TestBatchSetBlock(unittest.TestCase):
    """Test batch_set_block with mocked connector."""

    def _setup_manager(self, batch_size=4):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        connector = Mock()
        connector.register_buffer = Mock()
        connector.batch_set = Mock(return_value=[True, True])  # 2 keys per block (key + value)

        mgr = StagingManager(connector, staging_batch_size=batch_size)

        num_layers = 2
        stride = 8
        num_blocks = 10

        # Allocate real host buffers
        host_key_ptrs = []
        host_val_ptrs = []
        self._bufs = []
        for _ in range(num_layers):
            kb = ctypes.create_string_buffer(num_blocks * stride)
            vb = ctypes.create_string_buffer(num_blocks * stride)
            self._bufs.extend([kb, vb])
            host_key_ptrs.append(ctypes.addressof(kb))
            host_val_ptrs.append(ctypes.addressof(vb))

        # Manually init staging (bypass cuda_host_alloc)
        staging_size = batch_size * num_layers * stride
        self._staging_wk = ctypes.create_string_buffer(staging_size)
        self._staging_wv = ctypes.create_string_buffer(staging_size)
        mgr._num_layers = num_layers
        mgr._strides = {"key": stride, "value": stride}
        mgr._bufs = {
            "write_key": ctypes.addressof(self._staging_wk),
            "write_value": ctypes.addressof(self._staging_wv),
        }
        mgr._initialized = True

        return mgr, connector, host_key_ptrs, host_val_ptrs

    def test_batch_set_calls_connector(self):
        mgr, connector, kp, vp = self._setup_manager()

        keys_per_kind = {
            "key": ["h1_0_key"],
            "value": ["h1_0_value"],
        }
        host_ptrs_per_kind = {"key": kp, "value": vp}

        result = mgr.batch_set_block(keys_per_kind, host_ptrs_per_kind, [0])
        self.assertEqual(result, [True])
        connector.batch_set.assert_called_once()

        # Verify keys passed to connector
        call_args = connector.batch_set.call_args
        passed_keys = call_args[0][0]
        self.assertIn("h1_0_key", passed_keys)
        self.assertIn("h1_0_value", passed_keys)

    def test_batch_set_failure_propagates(self):
        mgr, connector, kp, vp = self._setup_manager()
        connector.batch_set.return_value = [False, True]  # key fails, value ok

        keys_per_kind = {
            "key": ["h1_0_key"],
            "value": ["h1_0_value"],
        }
        result = mgr.batch_set_block(keys_per_kind, {"key": kp, "value": vp}, [0])
        self.assertEqual(result, [False])


class TestBatchGetBlock(unittest.TestCase):
    """Test batch_get_block with mocked connector."""

    def _setup_manager(self, batch_size=4):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        connector = Mock()
        connector.register_buffer = Mock()
        connector.batch_get = Mock(return_value=[True, True])

        mgr = StagingManager(connector, staging_batch_size=batch_size)

        num_layers = 2
        stride = 8
        num_blocks = 10

        host_key_ptrs = []
        host_val_ptrs = []
        self._bufs = []
        for _ in range(num_layers):
            kb = ctypes.create_string_buffer(num_blocks * stride)
            vb = ctypes.create_string_buffer(num_blocks * stride)
            self._bufs.extend([kb, vb])
            host_key_ptrs.append(ctypes.addressof(kb))
            host_val_ptrs.append(ctypes.addressof(vb))

        staging_size = batch_size * num_layers * stride
        self._staging_rk = ctypes.create_string_buffer(staging_size)
        self._staging_rv = ctypes.create_string_buffer(staging_size)
        mgr._num_layers = num_layers
        mgr._strides = {"key": stride, "value": stride}
        mgr._bufs = {
            "read_key": ctypes.addressof(self._staging_rk),
            "read_value": ctypes.addressof(self._staging_rv),
        }
        mgr._initialized = True

        return mgr, connector, host_key_ptrs, host_val_ptrs

    def test_batch_get_calls_connector(self):
        mgr, connector, kp, vp = self._setup_manager()

        keys_per_kind = {
            "key": ["h1_0_key"],
            "value": ["h1_0_value"],
        }
        result = mgr.batch_get_block(keys_per_kind, {"key": kp, "value": vp}, [0])
        self.assertEqual(result, [True])
        connector.batch_get.assert_called_once()

    def test_batch_get_failure_skips_scatter(self):
        mgr, connector, kp, vp = self._setup_manager()
        connector.batch_get.return_value = [False, True]  # key fails

        keys_per_kind = {
            "key": ["h1_0_key"],
            "value": ["h1_0_value"],
        }
        result = mgr.batch_get_block(keys_per_kind, {"key": kp, "value": vp}, [0])
        self.assertEqual(result, [False])


class TestChunking(unittest.TestCase):
    """Test that batches larger than staging_batch_size are chunked correctly."""

    def test_multiple_chunks(self):
        from fastdeploy.cache_manager.v1.storage.staging_manager import StagingManager

        connector = Mock()
        connector.register_buffer = Mock()
        # Return success for all keys in each chunk
        connector.batch_set = Mock(side_effect=lambda k, p, s: [True] * len(k))

        mgr = StagingManager(connector, staging_batch_size=2)

        num_layers = 2
        stride = 8
        num_blocks = 10

        host_key_ptrs = []
        host_val_ptrs = []
        self._bufs = []
        for _ in range(num_layers):
            kb = ctypes.create_string_buffer(num_blocks * stride)
            vb = ctypes.create_string_buffer(num_blocks * stride)
            self._bufs.extend([kb, vb])
            host_key_ptrs.append(ctypes.addressof(kb))
            host_val_ptrs.append(ctypes.addressof(vb))

        staging_size = 2 * num_layers * stride
        self._wk = ctypes.create_string_buffer(staging_size)
        self._wv = ctypes.create_string_buffer(staging_size)
        mgr._num_layers = num_layers
        mgr._strides = {"key": stride, "value": stride}
        mgr._bufs = {
            "write_key": ctypes.addressof(self._wk),
            "write_value": ctypes.addressof(self._wv),
        }
        mgr._initialized = True

        # Send 5 blocks through batch_size=2 staging → expect 3 chunks
        keys_per_kind = {
            "key": [f"h{i}_0_key" for i in range(5)],
            "value": [f"h{i}_0_value" for i in range(5)],
        }
        result = mgr.batch_set_block(keys_per_kind, {"key": host_key_ptrs, "value": host_val_ptrs}, list(range(5)))

        self.assertEqual(len(result), 5)
        self.assertTrue(all(result))
        # 3 chunks: [0,1], [2,3], [4]
        self.assertEqual(connector.batch_set.call_count, 3)


if __name__ == "__main__":
    unittest.main()
