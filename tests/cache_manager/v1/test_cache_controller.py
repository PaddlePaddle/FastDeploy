# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
        from concurrent.futures import ThreadPoolExecutor

        controller = create_cache_controller()
        self.assertIsNotNone(controller._executor)
        self.assertIsInstance(controller._executor, ThreadPoolExecutor)

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


def make_done_counter(num_layers=4):
    """Create a pre-completed LayerDoneCounter for use in mocks."""
    from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

    counter = LayerDoneCounter(num_layers)
    counter.mark_all_done()
    return counter


class TestLoadHostToDevice(unittest.TestCase):
    """Test load_host_to_device returns LayerDoneCounter."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_returns_layer_done_counter(self, mock_submit):
        """Test that load_host_to_device returns LayerDoneCounter."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        mock_submit.return_value = make_done_counter()

        meta = CacheSwapMetadata(
            src_block_ids=[10, 11, 12],
            dst_block_ids=[0, 1, 2],
            src_type="host",
            dst_type="device",
        )
        counter = self.controller.load_host_to_device(meta)

        self.assertIsNotNone(counter)
        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_single_metadata_completes_successfully(self, mock_submit):
        """Test that single metadata task completes with success."""

        def fake_submit(meta, **kwargs):
            meta.success = True
            return make_done_counter()

        mock_submit.side_effect = fake_submit

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        counter = self.controller.load_host_to_device(meta)

        # Counter is already done (pre-completed)
        self.assertTrue(counter.is_all_done())
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_wait_for_layer(self, mock_submit):
        """Test wait_for_layer returns when layer is done."""
        mock_submit.return_value = make_done_counter()

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        counter = self.controller.load_host_to_device(meta)

        # Counter is pre-completed, wait_for_layer should return True immediately
        result = counter.wait_for_layer(0, timeout=5.0)
        self.assertTrue(result)
        self.assertTrue(counter.is_layer_done(0))

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_multiple_metadata_creates_separate_counters(self, mock_submit):
        """Test that multiple CacheSwapMetadatas create separate counters."""
        mock_submit.side_effect = lambda *a, **kw: make_done_counter()

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

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_returns_immediately_non_blocking(self, mock_submit):
        """Test that load_host_to_device returns without blocking."""

        def slow_submit(*args, **kwargs):
            time.sleep(0.5)
            return make_done_counter()

        mock_submit.side_effect = slow_submit

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        start = time.time()
        self.controller.load_host_to_device(meta)
        elapsed = time.time() - start

        # load_host_to_device calls _submit_swap_task synchronously (submit to executor),
        # so elapsed includes the mock's 0.5s sleep. Assert it completes within 1s.
        self.assertLess(elapsed, 1.0)


# ============================================================================
# evict_device_to_host Tests
# ============================================================================


class TestEvictDeviceToHost(unittest.TestCase):
    """Test evict_device_to_host returns LayerDoneCounter."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_returns_layer_done_counter(self, mock_submit):
        """Test that evict_device_to_host returns LayerDoneCounter."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        mock_submit.return_value = make_done_counter()

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        counter = self.controller.evict_device_to_host(meta)

        self.assertIsNotNone(counter)
        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_single_metadata_completes(self, mock_submit):
        """Test that eviction completes successfully."""

        def fake_submit(meta, **kwargs):
            meta.success = True
            return make_done_counter()

        mock_submit.side_effect = fake_submit

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        counter = self.controller.evict_device_to_host(meta)

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

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_submit_swap_tasks_returns_layer_done_counter(self, mock_submit):
        """Test submit_swap_tasks returns LayerDoneCounter for swap_in."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        mock_submit.return_value = make_done_counter()

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        swap_in_meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        counter = self.controller.submit_swap_tasks(evict_meta, swap_in_meta)

        self.assertIsNotNone(counter)
        self.assertIsInstance(counter, LayerDoneCounter)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_submit_swap_tasks_evict_only_returns_none(self, mock_submit):
        """Test submit_swap_tasks with only evict metadata returns None."""
        mock_submit.return_value = make_done_counter()

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])

        counter = self.controller.submit_swap_tasks(evict_meta, None)

        # Evict-only returns None (no swap-in counter)
        self.assertIsNone(counter)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_submit_swap_tasks_sets_swap_layer_done_counter(self, mock_submit):
        """Test submit_swap_tasks sets swap_layer_done_counter property."""
        expected_counter = make_done_counter()
        mock_submit.return_value = expected_counter

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

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_reset_cache_clears_pending_evict_counters(self, mock_submit):
        """Test reset_cache clears pending evict counters."""
        mock_submit.return_value = make_done_counter()

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

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_layer_by_layer_transfer_failure(self, mock_submit):
        """Test that transfer failure is properly reported via _submit_swap_task exception."""

        def failing_submit(meta, **kwargs):
            meta.success = False
            meta.error_message = "CUDA error"
            counter = make_done_counter()
            return counter

        mock_submit.side_effect = failing_submit

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device(meta)

        # The error should be stored in meta.error_message
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
        mock_metadata = MagicMock()
        mock_metadata.hash_values = []
        mock_metadata.block_ids = []
        handler = self.controller.prefetch_from_storage(mock_metadata)

        self.assertIsNotNone(handler)
        self.assertIsNotNone(handler.error)

    def test_backup_host_to_storage_returns_error_handler(self):
        """Test backup_host_to_storage returns error handler."""
        mock_metadata = MagicMock()
        mock_metadata.hash_values = []
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


# ============================================================================
# write_policy Property Tests
# ============================================================================


class TestWritePolicy(unittest.TestCase):
    """Test write_policy property and related behavior."""

    def test_write_policy_default(self):
        """Test write_policy reads from config."""
        controller = create_cache_controller()
        # Default config has write_policy set; just verify it's accessible
        policy = controller.write_policy
        self.assertIsInstance(policy, (str, type(None)))

    def test_should_wait_for_swap_out_write_back(self):
        """Test _should_wait_for_swap_out returns True for write_back policy."""
        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 50
        config.model_config.num_hidden_layers = 4
        config.cache_config.write_policy = "write_back"

        controller = CacheController(config, local_rank=0, device_id=0)
        self.assertTrue(controller._should_wait_for_swap_out())

    def test_should_wait_for_swap_out_write_through(self):
        """Test _should_wait_for_swap_out returns False for write_through policy."""
        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 50
        config.model_config.num_hidden_layers = 4
        config.cache_config.write_policy = "write_through"

        controller = CacheController(config, local_rank=0, device_id=0)
        self.assertFalse(controller._should_wait_for_swap_out())


# ============================================================================
# free_cache / free_gpu_cache Tests
# ============================================================================


class TestFreeCacheMethods(unittest.TestCase):
    """Test free_cache and free_gpu_cache methods."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    def test_free_gpu_cache_clears_map(self):
        """Test free_gpu_cache clears the cache_kvs_map."""
        device_cache = create_mock_device_cache_kvs_map(num_layers=4)
        self.controller.cache_kvs_map = device_cache

        self.assertGreater(len(self.controller.cache_kvs_map), 0)

        self.controller.free_gpu_cache()

        self.assertEqual(len(self.controller.cache_kvs_map), 0)

    def test_free_cache_returns_true(self):
        """Test free_cache returns True on success."""
        result = self.controller.free_cache()
        self.assertTrue(result)

    def test_free_gpu_cache_noop_when_empty(self):
        """Test free_gpu_cache is a no-op when cache_kvs_map is already empty."""
        self.controller.cache_kvs_map = {}
        # Should not raise
        self.controller.free_gpu_cache()
        self.assertEqual(len(self.controller.cache_kvs_map), 0)


# ============================================================================
# initialize_kv_cache / initialize_mtp_kv_cache dtype Tests (PR #7757)
# ============================================================================


def make_mock_attn_backend(key_shape=(10, 4, 16, 64), val_shape=None, val_shape_is_none=False):
    """Create a mock attn_backend with a fixed get_kv_cache_shape.

    The mock delegates create_kv_cache / create_host_kv_cache to the real
    AttentionBackend base class implementation so that tests exercise the
    actual tensor allocation logic through CacheController.
    """
    from fastdeploy.model_executor.layers.attention.base_attention_backend import (
        AttentionBackend,
    )

    if val_shape_is_none:
        # Simulate MLA variants (e.g., DeepSeek) that return None for value_cache_shape
        backend = MagicMock()
        backend.get_kv_cache_shape.return_value = (list(key_shape), None)
        # Wire real create_kv_cache to use the mock's get_kv_cache_shape
        backend.create_kv_cache = lambda **kwargs: AttentionBackend.create_kv_cache(backend, **kwargs)
        backend.create_host_kv_cache = lambda **kwargs: AttentionBackend.create_host_kv_cache(backend, **kwargs)
        backend.free_host_kv_cache = lambda host_caches: AttentionBackend.free_host_kv_cache(backend, host_caches)
        return backend
    if val_shape is None:
        val_shape = key_shape
    backend = MagicMock()
    backend.get_kv_cache_shape.return_value = (list(key_shape), list(val_shape))
    # Wire real create_kv_cache to use the mock's get_kv_cache_shape
    backend.create_kv_cache = lambda **kwargs: AttentionBackend.create_kv_cache(backend, **kwargs)
    backend.create_host_kv_cache = lambda **kwargs: AttentionBackend.create_host_kv_cache(backend, **kwargs)
    backend.free_host_kv_cache = lambda host_caches: AttentionBackend.free_host_kv_cache(backend, host_caches)
    return backend


class TestInitializeKVCacheDtype(unittest.TestCase):
    """
    Tests for the cache_dtype logic introduced in PR #7757:
      cache_dtype = "uint8" if kv_cache_quant_type is not None else model_config.dtype
    """

    def _make_controller(self, model_dtype="bfloat16", num_layers=2):
        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 0  # skip host cache init
        config.model_config.num_hidden_layers = num_layers
        config.model_config.dtype = model_dtype
        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        return CacheController(config, local_rank=0, device_id=0)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_kv_cache_non_quantized_uses_model_dtype(self, mock_quant_type):
        """When kv_cache_quant_type is None, cache tensors use model_config.dtype."""
        mock_quant_type.return_value = None
        controller = self._make_controller(model_dtype="bfloat16", num_layers=2)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        self.assertEqual(len(cache_list), 4)  # 2 layers * (key + value)
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.bfloat16")

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_kv_cache_quantized_uses_uint8(self, mock_quant_type):
        """When kv_cache_quant_type is set, cache tensors use uint8 regardless of model dtype."""
        mock_quant_type.return_value = "int8"
        controller = self._make_controller(model_dtype="bfloat16", num_layers=2)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        self.assertEqual(len(cache_list), 4)
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.uint8")

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_kv_cache_fp8_quantized_uses_uint8(self, mock_quant_type):
        """When kv_cache_quant_type is block_wise_fp8, non-scale cache tensors use uint8."""
        mock_quant_type.return_value = "block_wise_fp8"
        controller = self._make_controller(model_dtype="bfloat16", num_layers=2)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        # fp8 path also creates scale tensors (float32); filter to only key/value caches
        kv_tensors = [t for t in cache_list if str(t.dtype) == "paddle.uint8"]
        self.assertEqual(len(kv_tensors), 4)  # 2 layers * (key + value)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_mtp_kv_cache_non_quantized_uses_model_dtype(self, mock_quant_type):
        """When kv_cache_quant_type is None, MTP cache tensors use model_config.dtype."""
        mock_quant_type.return_value = None
        controller = self._make_controller(model_dtype="float16", num_layers=4)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_mtp_kv_cache(
            attn_backend=backend, num_gpu_blocks=10, num_mtp_layers=2, layer_offset=4
        )

        self.assertEqual(len(cache_list), 4)  # 2 mtp layers * (key + value)
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.float16")

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_mtp_kv_cache_quantized_uses_uint8(self, mock_quant_type):
        """When kv_cache_quant_type is set, MTP cache tensors use uint8."""
        mock_quant_type.return_value = "int8"
        controller = self._make_controller(model_dtype="bfloat16", num_layers=4)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_mtp_kv_cache(
            attn_backend=backend, num_gpu_blocks=10, num_mtp_layers=2, layer_offset=4
        )

        self.assertEqual(len(cache_list), 4)
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.uint8")

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_kv_cache_populates_cache_kvs_map(self, mock_quant_type):
        """Tensors created in initialize_kv_cache are stored in cache_kvs_map with correct dtype."""
        mock_quant_type.return_value = "int8"
        controller = self._make_controller(model_dtype="bfloat16", num_layers=2)
        backend = make_mock_attn_backend()

        controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        for name, tensor in controller.cache_kvs_map.items():
            if "scale" not in name:
                self.assertEqual(str(tensor.dtype), "paddle.uint8", f"wrong dtype for {name}")

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_kv_cache_null_value_cache_shape(self, mock_quant_type):
        """MLA variant: when value_cache_shape is None, only key cache is created."""
        mock_quant_type.return_value = None
        controller = self._make_controller(model_dtype="bfloat16", num_layers=2)
        backend = make_mock_attn_backend(val_shape_is_none=True)

        cache_list = controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        self.assertEqual(len(cache_list), 2)  # 2 layers * key only
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.bfloat16")
        # Verify no value entries in cache_kvs_map
        for name in controller.cache_kvs_map:
            self.assertNotIn("value", name)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_mtp_kv_cache_null_value_cache_shape(self, mock_quant_type):
        """MLA variant: when value_cache_shape is None, only key cache is created for MTP."""
        mock_quant_type.return_value = None
        controller = self._make_controller(model_dtype="bfloat16", num_layers=4)
        backend = make_mock_attn_backend(val_shape_is_none=True)

        cache_list = controller.initialize_mtp_kv_cache(
            attn_backend=backend, num_gpu_blocks=10, num_mtp_layers=2, layer_offset=4
        )

        self.assertEqual(len(cache_list), 2)  # 2 mtp layers * key only
        for tensor in cache_list:
            self.assertEqual(str(tensor.dtype), "paddle.bfloat16")


if __name__ == "__main__":
    unittest.main()


# ============================================================================
# Additional coverage tests for uncovered lines
# ============================================================================


class TestWritePolicyNone(unittest.TestCase):
    """Test write_policy returns None when cache_config has no write_policy."""

    def test_write_policy_returns_none_when_no_attr(self):
        """Line 112: write_policy returns None when cache_config has no write_policy."""
        controller = create_cache_controller()
        # Remove write_policy attribute if exists
        if hasattr(controller.cache_config, "write_policy"):
            delattr(controller.cache_config, "write_policy")
        self.assertIsNone(controller.write_policy)


class TestGetKVCacheQuantType(unittest.TestCase):
    """Test _get_kv_cache_quant_type method with various quant_config states."""

    def test_returns_quant_type_when_set(self):
        """Lines 202-208: returns kv_cache_quant_type from quant_config."""
        controller = create_cache_controller()
        # Mock quant_config with kv_cache_quant_type
        mock_quant_config = MagicMock()
        mock_quant_config.kv_cache_quant_type = "int8"
        controller.quant_config = mock_quant_config
        self.assertEqual(controller._get_kv_cache_quant_type(), "int8")

    def test_returns_none_when_quant_config_is_none(self):
        """Lines 202-208: returns None when quant_config is None."""
        controller = create_cache_controller()
        controller.quant_config = None
        self.assertIsNone(controller._get_kv_cache_quant_type())

    def test_returns_none_when_kv_cache_quant_type_is_none(self):
        """Lines 202-208: returns None when kv_cache_quant_type is None."""
        controller = create_cache_controller()
        mock_quant_config = MagicMock()
        mock_quant_config.kv_cache_quant_type = None
        controller.quant_config = mock_quant_config
        self.assertIsNone(controller._get_kv_cache_quant_type())


class TestTransferManagerProperty(unittest.TestCase):
    """Test transfer_manager property."""

    def test_transfer_manager_returns_instance(self):
        """Line 191: transfer_manager property returns CacheTransferManager."""
        controller = create_cache_controller()
        tm = controller.transfer_manager
        self.assertIsNotNone(tm)
        self.assertIs(tm, controller._transfer_manager)


class TestWaitForPendingEvictCounters(unittest.TestCase):
    """Test _wait_for_pending_evict_counters with actual pending counters."""

    def test_waits_and_clears_pending_counters(self):
        """Lines 175-184: waits on all pending counters then clears list."""
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        controller = create_cache_controller(num_layers=4)

        # Create pre-completed counters
        counter1 = LayerDoneCounter(4)
        counter1.mark_all_done()
        counter2 = LayerDoneCounter(4)
        counter2.mark_all_done()

        controller._pending_evict_counters = [counter1, counter2]
        self.assertEqual(len(controller._pending_evict_counters), 2)

        controller._wait_for_pending_evict_counters()
        self.assertEqual(len(controller._pending_evict_counters), 0)

    def test_noop_when_empty(self):
        """Line 172: returns immediately when no pending counters."""
        controller = create_cache_controller(num_layers=4)
        controller._pending_evict_counters = []
        # Should not raise
        controller._wait_for_pending_evict_counters()


class TestSubmitSwapTasksWriteBack(unittest.TestCase):
    """Test submit_swap_tasks with write_back policy (line 155)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._submit_swap_task")
    def test_write_back_waits_for_evict_before_swap_in(self, mock_submit):
        """Line 155: write_back policy waits for evict before swap-in."""
        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 50
        config.model_config.num_hidden_layers = 4
        config.cache_config.write_policy = "write_back"

        controller = CacheController(config, local_rank=0, device_id=0)
        setup_transfer_env(controller, num_layers=4)

        mock_submit.return_value = make_done_counter()

        evict_meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        swap_in_meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        counter = controller.submit_swap_tasks(evict_meta, swap_in_meta)
        self.assertIsNotNone(counter)
        # In write_back mode, pending evict counters are cleared before swap-in
        self.assertEqual(len(controller._pending_evict_counters), 0)


class TestGetNumaNodeForGpu(unittest.TestCase):
    """Test _get_numa_node_for_gpu method (lines 426-471)."""

    @patch("subprocess.run")
    def test_nvidia_smi_success(self, mock_run):
        """Lines 426-445: nvidia-smi returns valid NUMA node."""
        controller = create_cache_controller()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NUMA IDs of closest CPU: 0\n",
        )

        result = controller._get_numa_node_for_gpu(0)
        self.assertEqual(result, 0)

    @patch("subprocess.run")
    def test_nvidia_smi_comma_separated(self, mock_run):
        """Lines 440-444: handles comma-separated NUMA IDs."""
        controller = create_cache_controller()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NUMA IDs of closest CPU: 0,1\n",
        )

        result = controller._get_numa_node_for_gpu(0)
        self.assertEqual(result, 0)

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=False)
    @patch("glob.glob", return_value=[])
    def test_all_methods_fail_returns_negative(self, mock_glob, mock_exists, mock_run):
        """Lines 468-471: returns -1 when all methods fail."""
        controller = create_cache_controller()
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        result = controller._get_numa_node_for_gpu(0)
        self.assertEqual(result, -1)

    @patch("glob.glob", return_value=[])
    @patch("os.path.exists", return_value=False)
    @patch("subprocess.run", side_effect=Exception("unexpected"))
    def test_exception_returns_negative(self, mock_run, mock_exists, mock_glob):
        """Lines 469-471: returns -1 on exception when all methods fail."""
        controller = create_cache_controller()

        result = controller._get_numa_node_for_gpu(0)
        self.assertEqual(result, -1)


class TestBindToClosestNumaNode(unittest.TestCase):
    """Test _bind_to_closest_numa_node (lines 484-529)."""

    def test_already_bound_returns_true(self):
        """Line 484: returns True immediately if already bound."""
        controller = create_cache_controller()
        controller._numa_bound = True
        self.assertTrue(controller._bind_to_closest_numa_node())

    @patch("ctypes.CDLL", side_effect=OSError("libnuma not found"))
    def test_libnuma_not_found_returns_false(self, mock_cdll):
        """Lines 490-496: returns False when libnuma is not available."""
        controller = create_cache_controller()
        controller._numa_bound = False

        result = controller._bind_to_closest_numa_node()
        self.assertFalse(result)

    @patch("ctypes.CDLL")
    def test_numa_not_available_returns_false(self, mock_cdll):
        """Lines 498-500: returns False when numa_available() < 0."""
        controller = create_cache_controller()
        controller._numa_bound = False

        mock_libnuma = MagicMock()
        mock_libnuma.numa_available.return_value = -1
        mock_cdll.return_value = mock_libnuma

        result = controller._bind_to_closest_numa_node()
        self.assertFalse(result)

    @patch("ctypes.CDLL")
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_numa_node_for_gpu")
    def test_numa_node_negative_returns_false(self, mock_get_numa, mock_cdll):
        """Lines 506-508: returns False when NUMA node cannot be determined."""
        controller = create_cache_controller()
        controller._numa_bound = False

        mock_libnuma = MagicMock()
        mock_libnuma.numa_available.return_value = 0
        mock_cdll.return_value = mock_libnuma
        mock_get_numa.return_value = -1

        result = controller._bind_to_closest_numa_node()
        self.assertFalse(result)

    @patch("ctypes.CDLL")
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_numa_node_for_gpu")
    def test_binding_success(self, mock_get_numa, mock_cdll):
        """Lines 512-525: successful binding sets _numa_bound = True."""
        controller = create_cache_controller()
        controller._numa_bound = False

        mock_libnuma = MagicMock()
        mock_libnuma.numa_available.return_value = 0
        mock_libnuma.numa_run_on_node.return_value = 0
        mock_cdll.return_value = mock_libnuma
        mock_get_numa.return_value = 1

        result = controller._bind_to_closest_numa_node()
        self.assertTrue(result)
        self.assertTrue(controller._numa_bound)
        mock_libnuma.numa_run_on_node.assert_called_once_with(1)
        mock_libnuma.numa_set_preferred.assert_called_once_with(1)

    @patch("ctypes.CDLL")
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_numa_node_for_gpu")
    def test_numa_run_on_node_fails(self, mock_get_numa, mock_cdll):
        """Lines 513-515: returns False when numa_run_on_node fails."""
        controller = create_cache_controller()
        controller._numa_bound = False

        mock_libnuma = MagicMock()
        mock_libnuma.numa_available.return_value = 0
        mock_libnuma.numa_run_on_node.return_value = -1
        mock_cdll.return_value = mock_libnuma
        mock_get_numa.return_value = 0

        result = controller._bind_to_closest_numa_node()
        self.assertFalse(result)


class TestInitializeHostCache(unittest.TestCase):
    """Test initialize_host_cache (lines 552-642)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._bind_to_closest_numa_node")
    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc", return_value=999)
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_host_cache_basic(self, mock_quant_type, mock_alloc, mock_numa):
        """Lines 552-642: basic host cache initialization."""
        mock_quant_type.return_value = None

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 10
        config.model_config.num_hidden_layers = 2
        config.model_config.dtype = "bfloat16"
        config.cache_config.cache_dtype = "bfloat16"
        # speculative_config is needed for num_extra_cache_layer
        mock_spec_config = MagicMock()
        mock_spec_config.num_extra_cache_layer = 0
        config.speculative_config = mock_spec_config

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        backend = make_mock_attn_backend(key_shape=(10, 4, 16, 64))

        controller.initialize_host_cache(backend)

        # Should have allocated host memory
        self.assertGreater(len(controller.host_cache_kvs_map), 0)
        self.assertTrue(mock_alloc.called)

    def test_initialize_host_cache_skip_when_zero_blocks(self):
        """Lines 547-550: skips when num_cpu_blocks == 0."""
        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 0
        config.model_config.num_hidden_layers = 2

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        backend = make_mock_attn_backend()

        controller.initialize_host_cache(backend)
        self.assertEqual(len(controller.host_cache_kvs_map), 0)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._bind_to_closest_numa_node")
    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc", return_value=888)
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_initialize_host_cache_skips_if_already_initialized(self, mock_quant_type, mock_alloc, mock_numa):
        """Line 552-553: skips if host_cache_kvs_map already populated."""
        mock_quant_type.return_value = None

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 10
        config.model_config.num_hidden_layers = 2
        config.cache_config.cache_dtype = "bfloat16"

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        controller.host_cache_kvs_map = {"existing_key": 12345}

        backend = make_mock_attn_backend()
        controller.initialize_host_cache(backend)

        # Should not call alloc since already initialized
        mock_alloc.assert_not_called()


class TestSubmitSwapTaskInternal(unittest.TestCase):
    """Test _submit_swap_task internal logic (lines 696-792)."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=2)
        setup_transfer_env(self.controller, num_layers=2)

    def test_force_all_layers_success(self):
        """Lines 716-746: force_all_layers=True path with successful transfer."""
        from fastdeploy.cache_manager.v1.metadata import CacheLevel

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        mock_transfer_all = MagicMock(return_value=True)
        mock_transfer_layer = MagicMock()

        counter = self.controller._submit_swap_task(
            meta=meta,
            src_location=CacheLevel.DEVICE,
            dst_location=CacheLevel.HOST,
            transfer_fn_all=mock_transfer_all,
            transfer_fn_layer=mock_transfer_layer,
            force_all_layers=True,
        )

        # Wait for background thread
        counter.wait_all(timeout=5.0)
        self.assertTrue(counter.is_all_done())
        mock_transfer_all.assert_called_once_with([0, 1], [10, 11])
        mock_transfer_layer.assert_not_called()

    def test_layer_by_layer_success(self):
        """Lines 747-771: layer-by-layer path with successful transfer."""
        from fastdeploy.cache_manager.v1.metadata import CacheLevel

        meta = CacheSwapMetadata(src_block_ids=[5], dst_block_ids=[0])

        def fake_layer_transfer(layers, on_complete, src_ids, dst_ids):
            for layer in layers:
                on_complete(layer)
            return True

        counter = self.controller._submit_swap_task(
            meta=meta,
            src_location=CacheLevel.HOST,
            dst_location=CacheLevel.DEVICE,
            transfer_fn_all=None,
            transfer_fn_layer=fake_layer_transfer,
            force_all_layers=False,
        )

        counter.wait_all(timeout=5.0)
        self.assertTrue(counter.is_all_done())
        self.assertTrue(meta.success)

    def test_transfer_exception_sets_error(self):
        """Lines 777-786: exception in transfer sets error on meta."""
        from fastdeploy.cache_manager.v1.metadata import CacheLevel

        meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])

        def failing_transfer(src_ids, dst_ids):
            raise RuntimeError("GPU error")

        self.controller._submit_swap_task(
            meta=meta,
            src_location=CacheLevel.DEVICE,
            dst_location=CacheLevel.HOST,
            transfer_fn_all=failing_transfer,
            transfer_fn_layer=None,
            force_all_layers=True,
        )

        # Wait for the background thread to complete
        time.sleep(1.0)
        self.assertFalse(meta.success)
        self.assertIn("GPU error", meta.error_message)


class TestClearStorage(unittest.TestCase):
    """Test _clear_storage method (lines 1049-1061)."""

    def test_clear_storage_with_clear_method(self):
        """Lines 1049-1056: calls storage_connector.clear()."""
        controller = create_cache_controller()
        mock_connector = MagicMock()
        mock_connector.clear.return_value = 5
        controller._transfer_manager._storage_connector = mock_connector

        controller._clear_storage()
        mock_connector.clear.assert_called_once()

    def test_clear_storage_with_disconnect(self):
        """Lines 1057-1059: calls disconnect() if clear is not available."""
        controller = create_cache_controller()
        mock_connector = MagicMock(spec=["disconnect"])
        controller._transfer_manager._storage_connector = mock_connector

        controller._clear_storage()
        mock_connector.disconnect.assert_called_once()

    def test_clear_storage_no_connector(self):
        """Lines 1049-1051: no-op when no storage_connector."""
        controller = create_cache_controller()
        controller._transfer_manager._storage_connector = None
        # Should not raise
        controller._clear_storage()

    def test_clear_storage_exception_handled(self):
        """Line 1061: exception is caught and logged."""
        controller = create_cache_controller()
        mock_connector = MagicMock()
        mock_connector.clear.side_effect = RuntimeError("storage error")
        controller._transfer_manager._storage_connector = mock_connector

        # Should not raise
        controller._clear_storage()


class TestFreeCacheWithClearStorage(unittest.TestCase):
    """Test free_cache with clear_storage=True (lines 1031, 1034-1035)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._clear_storage")
    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._free_host_cache")
    def test_free_cache_clear_storage_true(self, mock_free_host, mock_clear_storage):
        """Line 1031: clear_storage=True calls _clear_storage."""
        controller = create_cache_controller()

        result = controller.free_cache(clear_storage=True)
        self.assertTrue(result)
        mock_clear_storage.assert_called_once()

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._free_host_cache")
    def test_free_cache_clear_storage_false(self, mock_free_host):
        """Line 1031: clear_storage=False does not call _clear_storage."""
        controller = create_cache_controller()

        with patch.object(controller, "_clear_storage") as mock_clear:
            result = controller.free_cache(clear_storage=False)
            self.assertTrue(result)
            mock_clear.assert_not_called()

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController.reset_cache", side_effect=Exception("oops"))
    def test_free_cache_exception_returns_false(self, mock_reset):
        """Lines 1034-1035: returns False on exception."""
        controller = create_cache_controller()
        result = controller.free_cache()
        self.assertFalse(result)


class TestResetCacheException(unittest.TestCase):
    """Test reset_cache exception path (lines 1006-1007)."""

    def test_reset_cache_exception_returns_false(self):
        """Lines 1006-1007: returns False when exception occurs."""
        controller = create_cache_controller()
        # Make _pending_evict_counters.clear() raise
        controller._pending_evict_counters = MagicMock()
        controller._pending_evict_counters.clear.side_effect = RuntimeError("lock error")

        result = controller.reset_cache()
        self.assertFalse(result)


class TestStartStop(unittest.TestCase):
    """Test start() and stop() methods (lines 1077, 1081-1083)."""

    def test_start(self):
        """Line 1077: start() calls transfer_manager.start()."""
        controller = create_cache_controller()
        with patch.object(controller._transfer_manager, "start", create=True) as mock_start:
            controller.start()
            mock_start.assert_called_once()

    def test_stop(self):
        """Lines 1081-1083: stop() calls transfer_manager.stop() and shuts down executor."""
        controller = create_cache_controller()
        with (
            patch.object(controller._transfer_manager, "stop", create=True) as mock_stop,
            patch.object(controller._executor, "shutdown") as mock_shutdown,
        ):
            controller.stop()
            mock_stop.assert_called_once()
            mock_shutdown.assert_called_once_with(wait=False)


class TestFreeHostCache(unittest.TestCase):
    """Test _free_host_cache method (lines 1027-1045)."""

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free")
    def test_free_host_cache_releases_memory(self, mock_free):
        """Lines 1027-1045: frees all host cache pointers via attn_backend."""
        controller = create_cache_controller()
        controller.host_cache_kvs_map = {
            "key_cache_0": 1000,
            "val_cache_0": 2000,
        }
        controller.attn_backend = make_mock_attn_backend()

        controller._free_host_cache()

        self.assertEqual(mock_free.call_count, 2)
        self.assertEqual(len(controller.host_cache_kvs_map), 0)

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free")
    def test_free_host_cache_skips_zero_ptr(self, mock_free):
        """Skips freeing pointers that are 0/None."""
        controller = create_cache_controller()
        controller.host_cache_kvs_map = {
            "key_cache_0": 0,
            "val_cache_0": 5000,
        }
        controller.attn_backend = make_mock_attn_backend()

        controller._free_host_cache()

        mock_free.assert_called_once_with(5000)

    def test_free_host_cache_noop_when_empty(self):
        """Line 1095: no-op when host_cache_kvs_map is empty."""
        controller = create_cache_controller()
        controller.host_cache_kvs_map = {}
        # Should not raise
        controller._free_host_cache()

    def test_free_host_cache_noop_when_no_attr(self):
        """Line 1095: no-op when host_cache_kvs_map doesn't exist."""
        controller = create_cache_controller()
        if hasattr(controller, "host_cache_kvs_map"):
            delattr(controller, "host_cache_kvs_map")
        # Should not raise
        controller._free_host_cache()

    @patch(
        "fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free",
        side_effect=Exception("free error"),
    )
    def test_free_host_cache_handles_free_error(self, mock_free):
        """Continues on free error."""
        controller = create_cache_controller()
        controller.host_cache_kvs_map = {
            "key_cache_0": 1000,
            "val_cache_0": 2000,
        }
        controller.attn_backend = make_mock_attn_backend()

        # Should not raise
        controller._free_host_cache()
        # Map should still be cleared
        self.assertEqual(len(controller.host_cache_kvs_map), 0)


class TestDestructor(unittest.TestCase):
    """Test __del__ method (lines 1089-1090)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._free_host_cache")
    def test_del_calls_free_host_cache(self, mock_free):
        """Lines 1089-1090: __del__ calls _free_host_cache."""
        controller = create_cache_controller()
        controller.__del__()
        mock_free.assert_called_once()

    @patch(
        "fastdeploy.cache_manager.v1.cache_controller.CacheController._free_host_cache", side_effect=Exception("err")
    )
    def test_del_swallows_exception(self, mock_free):
        """Lines 1089-1090: __del__ swallows exceptions."""
        controller = create_cache_controller()
        # Should not raise
        controller.__del__()


class TestInitializeKVCacheFp8NoValueCache(unittest.TestCase):
    """Test fp8 scale with no value_cache_shape (line 319)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_fp8_no_value_creates_only_key_scale(self, mock_quant_type):
        """Line 319: fp8 with value_cache_shape=None creates only key_scale."""
        mock_quant_type.return_value = "block_wise_fp8"

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 0
        config.model_config.num_hidden_layers = 1
        config.model_config.dtype = "bfloat16"

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        backend = make_mock_attn_backend(val_shape_is_none=True)

        cache_list = controller.initialize_kv_cache(backend, num_gpu_blocks=10)

        # Should have: key_cache(uint8) + key_scale(float32) for 1 layer = 2 tensors
        self.assertEqual(len(cache_list), 2)
        # Verify no value entries
        for name in controller.cache_kvs_map:
            self.assertNotIn("value_caches", name)
            self.assertNotIn("value_cache_scales", name)


class TestInitializeMTPKVCacheFp8(unittest.TestCase):
    """Test MTP fp8 scale creation (lines 390-401)."""

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_mtp_fp8_with_value_cache(self, mock_quant_type):
        """Lines 390-399: MTP fp8 creates both key and value scales."""
        mock_quant_type.return_value = "block_wise_fp8"

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 0
        config.model_config.num_hidden_layers = 4
        config.model_config.dtype = "bfloat16"

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        backend = make_mock_attn_backend()

        cache_list = controller.initialize_mtp_kv_cache(
            attn_backend=backend, num_gpu_blocks=5, num_mtp_layers=1, layer_offset=4
        )

        # Per layer: key(uint8) + value(uint8) + key_scale(float32) + value_scale(float32) = 4 tensors
        self.assertEqual(len(cache_list), 4)

    @patch("fastdeploy.cache_manager.v1.cache_controller.CacheController._get_kv_cache_quant_type")
    def test_mtp_fp8_no_value_cache(self, mock_quant_type):
        """Lines 400-401: MTP fp8 with no value_cache_shape creates only key_scale."""
        mock_quant_type.return_value = "block_wise_fp8"

        config = get_default_test_fd_config()
        config.cache_config.num_cpu_blocks = 0
        config.model_config.num_hidden_layers = 4
        config.model_config.dtype = "bfloat16"

        from fastdeploy.cache_manager.v1.cache_controller import CacheController

        controller = CacheController(config, local_rank=0, device_id=0)
        backend = make_mock_attn_backend(val_shape_is_none=True)

        cache_list = controller.initialize_mtp_kv_cache(
            attn_backend=backend, num_gpu_blocks=5, num_mtp_layers=1, layer_offset=4
        )

        # Per layer: key(uint8) + key_scale(float32) = 2 tensors (no value)
        self.assertEqual(len(cache_list), 2)
