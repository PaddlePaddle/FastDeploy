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

"""
Unit tests for CacheController class with the new LayerDoneCounter design.

Tests cover:
- Initialization
- load_host_to_device returns LayerDoneCounter
- evict_device_to_host returns LayerDoneCounter
- submit_swap_tasks returns LayerDoneCounter
- LayerDoneCounter methods: wait_for_layer, wait_all, mark_layer_done, mark_all_done
- Statistics
- Edge cases (empty metadata, failed transfers)
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from utils import get_default_test_fd_config

from fastdeploy.cache_manager.v1.metadata import CacheSwapMetadata


def create_cache_controller(
    enable_prefix_caching: bool = True,
    num_host_blocks: int = 50,
    num_layers: int = 4,
):
    """Helper to create CacheController with test config."""
    from fastdeploy.cache_manager.v1.cache_controller import CacheController

    config = get_default_test_fd_config()
    config.cache_config.enable_prefix_caching = enable_prefix_caching
    config.cache_config.num_cpu_blocks = num_host_blocks
    config.cache_config.cache_dtype = "bfloat16"
    config.model_config.num_hidden_layers = num_layers
    config.model_config.dtype = "bfloat16"

    return CacheController(config, local_rank=0, device_id=0)


def create_mock_device_cache_kvs_map(
    num_layers: int = 4,
    local_rank: int = 0,
    device_id: int = 0,
    num_blocks: int = 100,
    num_heads: int = 32,
    block_size: int = 64,
    head_dim: int = 128,
    dtype: str = "bfloat16",
):
    """Helper to create mock device cache_kvs_map."""
    import paddle

    cache_kvs_map = {}

    for layer_idx in range(num_layers):
        key_name = f"key_caches_{layer_idx}_rank{local_rank}.device{device_id}"
        val_name = f"value_caches_{layer_idx}_rank{local_rank}.device{device_id}"

        key_tensor = paddle.zeros([num_blocks, num_heads, block_size, head_dim], dtype=dtype)
        val_tensor = paddle.zeros([num_blocks, num_heads, block_size, head_dim], dtype=dtype)

        cache_kvs_map[key_name] = key_tensor
        cache_kvs_map[val_name] = val_tensor

    return cache_kvs_map


def create_mock_host_cache_kvs_map(
    num_layers: int = 4,
    local_rank: int = 0,
    device_id: int = 0,
    base_ptr: int = 1000000,
):
    """Helper to create mock host cache_kvs_map (with int pointers)."""
    cache_kvs_map = {}

    for layer_idx in range(num_layers):
        key_name = f"key_caches_{layer_idx}_rank{local_rank}.device{device_id}"
        val_name = f"value_caches_{layer_idx}_rank{local_rank}.device{device_id}"

        cache_kvs_map[key_name] = base_ptr + layer_idx * 10000
        cache_kvs_map[val_name] = base_ptr + layer_idx * 10000 + 5000

    return cache_kvs_map


def setup_transfer_env(controller, num_layers=4):
    """Helper to set up device and host cache for transfer tests."""
    device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers)
    controller._transfer_manager.set_cache_kvs_map(device_cache)
    host_cache = create_mock_host_cache_kvs_map(num_layers=num_layers)
    controller._transfer_manager.set_host_cache_kvs_map(host_cache)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestCacheControllerInit(unittest.TestCase):
    """Test CacheController initialization."""

    def test_init_creates_executor(self):
        """Test that ThreadPoolExecutor is created on init."""
        controller = create_cache_controller()
        self.assertIsNotNone(controller._executor)
        self.assertEqual(controller._executor._max_workers, 1)

    def test_init_creates_transfer_manager(self):
        """Test that TransferManager is created on init."""
        controller = create_cache_controller()
        self.assertIsNotNone(controller._transfer_manager)

    def test_init_no_singleton_layer_counter(self):
        """Test that LayerDoneCounter is NOT created as singleton on init (per-transfer design)."""
        controller = create_cache_controller(num_layers=4)
        # In the new design, _layer_counter is None initially, set per transfer
        self.assertIsNone(controller._layer_done_counter)

    def test_init_empty_pending_evict_counters(self):
        """Test that pending evict counters list is empty on init."""
        controller = create_cache_controller()
        self.assertEqual(len(controller._pending_evict_counters), 0)


# ============================================================================
# load_host_to_device Tests
# ============================================================================


class TestLoadHostToDevice(unittest.TestCase):
    """Test load_host_to_device returns LayerDoneCounter."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_returns_layer_done_counter(self, mock_swap):
        """Test that load_host_to_device returns LayerDoneCounter."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(
            src_block_ids=[10, 11, 12],
            dst_block_ids=[0, 1, 2],
            src_type="host",
            dst_type="device",
        )
        counter = self.controller.load_host_to_device(meta)

        self.assertIsNotNone(counter)
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_single_metadata_completes_successfully(self, mock_swap):
        """Test that single metadata task completes with success."""
        mock_swap.return_value = True

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        counter = self.controller.load_host_to_device(meta)

        # Wait for all layers to complete
        counter.wait_all(timeout=5.0)
        self.assertTrue(counter.is_all_done())
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_wait_for_layer(self, mock_swap):
        """Test wait_for_layer returns when layer is done."""
        mock_swap.return_value = True

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        counter = self.controller.load_host_to_device(meta)

        # Wait for a specific layer
        result = counter.wait_for_layer(0, timeout=5.0)
        self.assertTrue(result)
        self.assertTrue(counter.is_layer_done(0))

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_multiple_metadata_creates_separate_counters(self, mock_swap):
        """Test that multiple CacheSwapMetadatas create separate counters."""
        mock_swap.return_value = None

        meta1 = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        meta2 = CacheSwapMetadata(src_block_ids=[11], dst_block_ids=[1])

        counter1 = self.controller.load_host_to_device(meta1)
        counter2 = self.controller.load_host_to_device(meta2)

        # Each should have its own counter
        self.assertIsNot(counter1, counter2)

    def test_empty_src_block_ids_sets_error(self):
        """Test that empty src block IDs set error."""
        meta = CacheSwapMetadata(src_block_ids=[], dst_block_ids=[0])
        self.controller.load_host_to_device(meta)

        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)

    def test_empty_dst_block_ids_sets_error(self):
        """Test that empty dst block IDs set error."""
        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[])
        self.controller.load_host_to_device(meta)

        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_returns_immediately_non_blocking(self, mock_swap):
        """Test that load_host_to_device returns without blocking."""

        def slow_swap(*args, **kwargs):
            time.sleep(0.5)
            return None

        mock_swap.side_effect = slow_swap

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        start = time.time()
        self.controller.load_host_to_device(meta)
        elapsed = time.time() - start

        # Should return immediately, not wait for 0.5s transfer
        self.assertLess(elapsed, 0.2)


# ============================================================================
# evict_device_to_host Tests
# ============================================================================


class TestEvictDeviceToHost(unittest.TestCase):
    """Test evict_device_to_host returns LayerDoneCounter."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_returns_layer_done_counter(self, mock_swap):
        """Test that evict_device_to_host returns LayerDoneCounter."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        counter = self.controller.evict_device_to_host(meta)

        self.assertIsNotNone(counter)
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_single_metadata_completes(self, mock_swap):
        """Test that eviction completes successfully."""
        mock_swap.return_value = True

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        counter = self.controller.evict_device_to_host(meta)

        counter.wait_all(timeout=5.0)
        self.assertTrue(counter.is_all_done())
        self.assertTrue(meta.success)


# ============================================================================
# submit_swap_tasks Tests
# ============================================================================


class TestSubmitSwapTasks(unittest.TestCase):
    """Test submit_swap_tasks method returns LayerDoneCounter."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_submit_swap_tasks_returns_layer_done_counter(self, mock_evict, mock_swap_in):
        """Test submit_swap_tasks returns LayerDoneCounter for swap_in."""
        mock_evict.return_value = None
        mock_swap_in.return_value = None

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        swap_in_meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        counter = self.controller.submit_swap_tasks(evict_meta, swap_in_meta)

        self.assertIsNotNone(counter)
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_submit_swap_tasks_evict_only_returns_none(self, mock_evict):
        """Test submit_swap_tasks with only evict metadata returns None."""
        mock_evict.return_value = None

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])

        counter = self.controller.submit_swap_tasks(evict_meta, None)

        # Evict-only returns None (no swap-in counter)
        self.assertIsNone(counter)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_submit_swap_tasks_sets_swap_layer_done_counter(self, mock_evict, mock_swap_in):
        """Test submit_swap_tasks sets swap_layer_done_counter property."""
        mock_evict.return_value = None
        mock_swap_in.return_value = None

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        swap_in_meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        counter = self.controller.submit_swap_tasks(evict_meta, swap_in_meta)

        # swap_layer_done_counter should be set
        self.assertIs(self.controller.swap_layer_done_counter, counter)


# ============================================================================
# LayerDoneCounter Tests
# ============================================================================


class TestLayerDoneCounter(unittest.TestCase):
    """Test LayerDoneCounter independent sync primitive."""

    def test_layer_done_counter_basic(self):
        """Test basic LayerDoneCounter functionality."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)

        # Initially not done
        self.assertFalse(counter.is_all_done())
        self.assertEqual(counter.get_completed_count(), 0)

        # Mark one layer done
        counter.mark_layer_done(0)
        self.assertTrue(counter.is_layer_done(0))
        self.assertFalse(counter.is_layer_done(1))
        self.assertEqual(counter.get_completed_count(), 1)
        self.assertFalse(counter.is_all_done())

    def test_layer_done_counter_mark_all_done(self):
        """Test mark_all_done marks all layers."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)

        counter.mark_all_done()

        self.assertTrue(counter.is_all_done())
        self.assertEqual(counter.get_completed_count(), 4)
        self.assertTrue(counter.is_layer_done(0))
        self.assertTrue(counter.is_layer_done(3))

    def test_layer_done_counter_wait_for_layer_immediate(self):
        """Test wait_for_layer returns immediately if done."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)
        counter.mark_all_done()

        result = counter.wait_for_layer(0, timeout=1.0)
        self.assertTrue(result)

    def test_layer_done_counter_wait_all(self):
        """Test wait_all waits for all layers."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)

        # Mark all done
        counter.mark_all_done()

        result = counter.wait_all(timeout=1.0)
        self.assertTrue(result)
        self.assertTrue(counter.is_all_done())

    def test_layer_done_counter_get_pending_layers(self):
        """Test get_pending_layers returns correct list."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)
        counter.mark_layer_done(1)

        pending = counter.get_pending_layers()
        self.assertEqual(pending, [0, 2, 3])

    def test_layer_done_counter_callback(self):
        """Test callback is called on layer complete."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)
        callback_layers = []

        def callback(layer_idx):
            callback_layers.append(layer_idx)

        counter.register_callback(callback)
        counter.mark_layer_done(2)

        self.assertEqual(callback_layers, [2])

    def test_layer_done_counter_stats(self):
        """Test get_stats returns correct stats."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=4)
        counter.mark_layer_done(0)
        counter.mark_layer_done(1)

        stats = counter.get_stats()
        self.assertEqual(stats["num_layers"], 4)
        self.assertEqual(stats["completed_layers"], 2)
        self.assertEqual(stats["pending_layers"], 2)


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStats(unittest.TestCase):
    """Test statistics functionality."""

    def test_get_stats_returns_expected_keys(self):
        """Test get_stats returns expected keys."""
        controller = create_cache_controller(num_layers=4)
        stats = controller.get_stats()

        self.assertIn("initialized", stats)
        self.assertIn("num_layers", stats)
        self.assertTrue(stats["initialized"])
        self.assertEqual(stats["num_layers"], 4)


# ============================================================================
# Reset Tests
# ============================================================================


class TestReset(unittest.TestCase):
    """Test reset_cache method."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.evict_to_host_async")
    def test_reset_cache_clears_pending_evict_counters(self, mock_evict):
        """Test reset_cache clears pending evict counters."""
        mock_evict.return_value = True

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        counter = self.controller.evict_device_to_host(evict_meta)

        # Manually add counter to pending evict counters (simulating what submit_swap_tasks does)
        self.controller._pending_evict_counters.append(counter)

        self.assertEqual(len(self.controller._pending_evict_counters), 1)

        result = self.controller.reset_cache()
        self.assertTrue(result)
        self.assertEqual(len(self.controller._pending_evict_counters), 0)


# ============================================================================
# KV Cache Management Tests
# ============================================================================


class TestKVCacheManagement(unittest.TestCase):
    """Test KV cache initialization and retrieval."""

    def test_get_kv_caches_without_init(self):
        """Test get_kv_caches returns empty dict when not initialized."""
        controller = create_cache_controller()
        result = controller.get_kv_caches()
        self.assertIsNotNone(result)

    def test_get_host_cache_kvs_map_without_init(self):
        """Test get_host_cache_kvs_map returns empty dict when not initialized."""
        controller = create_cache_controller()
        result = controller.get_host_cache_kvs_map()
        self.assertEqual(len(result), 0)


# ============================================================================
# Transfer Failure Tests
# ============================================================================


class TestTransferFailure(unittest.TestCase):
    """Test behavior when transfer fails."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.CacheTransferManager.load_layers_to_device_async")
    def test_layer_by_layer_transfer_failure(self, mock_swap):
        """Test that transfer failure is properly reported."""
        mock_swap.side_effect = RuntimeError("CUDA error")

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device(meta)

        # The counter's is_all_done() should return False since the transfer failed
        # (mark_all_done is not called on failure)
        # Give the executor a moment to process
        import time

        time.sleep(0.1)

        # The error should be caught and stored in meta.error_message
        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)
        self.assertIn("CUDA error", meta.error_message)


# ============================================================================
# Storage Placeholder Tests
# ============================================================================


class TestStoragePlaceholders(unittest.TestCase):
    """Test storage placeholder methods."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)

    def test_prefetch_from_storage_returns_error_handler(self):
        """Test prefetch_from_storage returns error handler (not implemented)."""
        from fastdeploy.cache_manager.v1.metadata import StorageMetadata

        mock_metadata = MagicMock(spec=StorageMetadata)
        handler = self.controller.prefetch_from_storage(mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)

    def test_backup_device_to_storage_returns_error_handler(self):
        """Test backup_device_to_storage returns error handler (not implemented)."""
        from fastdeploy.cache_manager.v1.metadata import StorageMetadata

        mock_metadata = MagicMock(spec=StorageMetadata)
        handler = self.controller.backup_device_to_storage([0, 1], mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)

    def test_backup_host_to_storage_returns_error_handler(self):
        """Test backup_host_to_storage returns error handler (not implemented)."""
        from fastdeploy.cache_manager.v1.metadata import StorageMetadata

        mock_metadata = MagicMock(spec=StorageMetadata)
        handler = self.controller.backup_host_to_storage([0, 1], mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)


class TestPDTransferPlaceholders(unittest.TestCase):
    """Test PD transfer placeholder methods."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)

    def test_send_to_node_returns_error_handler(self):
        """Test send_to_node returns error handler (not implemented)."""
        from fastdeploy.cache_manager.v1.metadata import PDTransferMetadata

        mock_metadata = MagicMock(spec=PDTransferMetadata)
        handler = self.controller.send_to_node(mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)

    def test_wait_for_transfer_from_node_returns_error_handler(self):
        """Test wait_for_transfer_from_node returns error handler (not implemented)."""
        from fastdeploy.cache_manager.v1.metadata import PDTransferMetadata

        mock_metadata = MagicMock(spec=PDTransferMetadata)
        handler = self.controller.wait_for_transfer_from_node(mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)


# ============================================================================
# CacheSwapMetadata Mapping Tests
# ============================================================================


class TestCacheSwapMetadataMapping(unittest.TestCase):
    """Test CacheSwapMetadata mapping property."""

    def test_mapping_empty_when_not_success(self):
        meta = CacheSwapMetadata(src_block_ids=[1, 2], dst_block_ids=[10, 11])
        self.assertEqual(meta.mapping, {})

    def test_mapping_returns_dict_after_success(self):
        meta = CacheSwapMetadata(src_block_ids=[1, 2], dst_block_ids=[10, 11])
        meta.success = True
        expected = {1: 10, 2: 11}
        self.assertEqual(meta.mapping, expected)


if __name__ == "__main__":
    unittest.main()
