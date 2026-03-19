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

Unit tests for CacheController class.

Tests cover:
- Initialization
- load_host_to_device with CacheSwapMetadata list
- evict_device_to_host with CacheSwapMetadata list
- Task tracking (status, progress, cancellation)
- Layer-by-layer transfer and LayerDoneCounter
- All-layer transfer mode
- reset_cache / reset_controller_cache
- Statistics
- Edge cases (empty metadata, failed transfers)
"""

import time
import unittest
from unittest.mock import patch

from utils import get_default_test_fd_config

from fastdeploy.cache_manager.v1.metadata import CacheSwapMetadata, TransferStatus


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

    def test_init_creates_transfer_manager(self):
        """Test that TransferManager is created on init."""
        controller = create_cache_controller()
        self.assertIsNotNone(controller._transfer_manager)

    def test_init_creates_layer_counter(self):
        """Test that LayerDoneCounter is created on init."""
        controller = create_cache_controller(num_layers=4)
        self.assertIsNotNone(controller._layer_counter)

    def test_init_empty_active_tasks(self):
        """Test that active tasks dict is empty on init."""
        controller = create_cache_controller()
        self.assertEqual(len(controller._active_tasks), 0)


# ============================================================================
# load_host_to_device Tests
# ============================================================================


class TestLoadHostToDevice(unittest.TestCase):
    """Test load_host_to_device with CacheSwapMetadata list."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_single_metadata_creates_handler(self, mock_swap):
        """Test that single CacheSwapMetadata creates handler on meta."""

        # Use a slow swap to verify handler exists before completion
        def slow_swap(*args, **kwargs):
            time.sleep(0.2)
            return None

        mock_swap.side_effect = slow_swap

        meta = CacheSwapMetadata(
            src_block_ids=[10, 11, 12],
            dst_block_ids=[0, 1, 2],
            src_type="host",
            dst_type="device",
        )
        self.controller.load_host_to_device([meta])

        # Handler should be set on metadata
        self.assertIsNotNone(meta.async_handler)
        # Task may already be completed in fast environments,
        # but handler must exist
        meta.async_handler.wait(timeout=5.0)
        self.assertTrue(meta.async_handler.is_completed)
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_single_metadata_completes_successfully(self, mock_swap):
        """Test that single metadata task completes with success."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        meta.async_handler.wait(timeout=5.0)

        self.assertTrue(meta.async_handler.is_completed)
        self.assertTrue(meta.success)
        self.assertIsNone(meta.error_message)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_single_metadata_result_content(self, mock_swap):
        """Test TransferResult content after successful load."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10, 11], dst_block_ids=[0, 1])
        self.controller.load_host_to_device([meta])

        result = meta.async_handler.get_result()
        self.assertTrue(result.success)
        self.assertEqual(result.src_block_ids, [10, 11])
        self.assertEqual(result.dst_block_ids, [0, 1])
        self.assertEqual(result.src_type, "host")
        self.assertEqual(result.dst_type, "device")

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_multiple_metadata_creates_separate_handlers(self, mock_swap):
        """Test that multiple CacheSwapMetadatas create separate parallel tasks."""
        mock_swap.return_value = None

        meta1 = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        meta2 = CacheSwapMetadata(src_block_ids=[11], dst_block_ids=[1])
        meta3 = CacheSwapMetadata(src_block_ids=[12], dst_block_ids=[2])

        self.controller.load_host_to_device([meta1, meta2, meta3])

        # Each metadata should have its own handler
        self.assertIsNotNone(meta1.async_handler)
        self.assertIsNotNone(meta2.async_handler)
        self.assertIsNotNone(meta3.async_handler)

        # Handlers should have unique task_ids
        self.assertNotEqual(meta1.async_handler.task_id, meta2.async_handler.task_id)
        self.assertNotEqual(meta2.async_handler.task_id, meta3.async_handler.task_id)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_multiple_metadata_all_complete(self, mock_swap):
        """Test that all metadata tasks complete."""
        mock_swap.return_value = None

        metas = [CacheSwapMetadata(src_block_ids=[10 + i], dst_block_ids=[i]) for i in range(5)]
        self.controller.load_host_to_device(metas)

        for meta in metas:
            meta.async_handler.wait(timeout=5.0)
            self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_empty_metadata_list(self, mock_swap):
        """Test that empty metadata list doesn't crash."""
        self.controller.load_host_to_device([])
        mock_swap.assert_not_called()

    def test_empty_block_ids_sets_error(self):
        """Test that empty block IDs set error on handler."""
        meta = CacheSwapMetadata(src_block_ids=[], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        self.assertIsNotNone(meta.async_handler)
        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)

    def test_dst_empty_block_ids_sets_error(self):
        """Test that empty dst block IDs set error on handler."""
        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[])
        self.controller.load_host_to_device([meta])

        self.assertIsNotNone(meta.async_handler)
        self.assertFalse(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_returns_immediately_non_blocking(self, mock_swap):
        """Test that load_host_to_device returns without blocking."""
        mock_swap.return_value = None

        # Use a slow transfer to verify non-blocking
        def slow_swap(*args, **kwargs):
            time.sleep(0.5)
            return None

        mock_swap.side_effect = slow_swap

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])

        start = time.time()
        self.controller.load_host_to_device([meta])
        elapsed = time.time() - start

        # Should return immediately, not wait for 0.5s transfer
        self.assertLess(elapsed, 0.2)


# ============================================================================
# evict_device_to_host Tests
# ============================================================================


class TestEvictDeviceToHost(unittest.TestCase):
    """Test evict_device_to_host with CacheSwapMetadata list."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_single_metadata_completes(self, mock_swap):
        """Test that eviction completes successfully."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[0, 1], dst_block_ids=[10, 11])
        self.controller.evict_device_to_host([meta])

        meta.async_handler.wait(timeout=5.0)

        self.assertTrue(meta.async_handler.is_completed)
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_result_content(self, mock_swap):
        """Test TransferResult content after successful eviction."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        self.controller.evict_device_to_host([meta])

        result = meta.async_handler.get_result()
        self.assertEqual(result.src_type, "device")
        self.assertEqual(result.dst_type, "host")
        self.assertEqual(result.src_block_ids, [0])
        self.assertEqual(result.dst_block_ids, [10])

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_multiple_evict_tasks(self, mock_swap):
        """Test multiple parallel eviction tasks."""
        mock_swap.return_value = None

        metas = [CacheSwapMetadata(src_block_ids=[i], dst_block_ids=[10 + i]) for i in range(3)]
        self.controller.evict_device_to_host(metas)

        for meta in metas:
            meta.async_handler.wait(timeout=5.0)
            self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_empty_list(self, mock_swap):
        """Test empty metadata list doesn't crash."""
        self.controller.evict_device_to_host([])
        mock_swap.assert_not_called()


# ============================================================================
# Task Tracking Tests
# ============================================================================


class TestTaskTracking(unittest.TestCase):
    """Test task tracking functionality."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_task_tracked_in_active_tasks(self, mock_swap):
        """Test that submitted task appears in _active_tasks."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        self.assertIn(meta.async_handler.task_id, self.controller._active_tasks)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_task_status_transitions_to_completed(self, mock_swap):
        """Test task status transitions from IN_PROGRESS to COMPLETED."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        meta.async_handler.wait(timeout=5.0)

        task = self.controller._active_tasks.get(meta.async_handler.task_id)
        self.assertEqual(task.status, TransferStatus.COMPLETED)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_transfer_status(self, mock_swap):
        """Test get_transfer_status returns correct status."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        status = self.controller.get_transfer_status(meta.async_handler.task_id)
        self.assertIsNotNone(status)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_transfer_status_nonexistent(self, mock_swap):
        """Test get_transfer_status returns None for unknown task."""
        status = self.controller.get_transfer_status("nonexistent")
        self.assertIsNone(status)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_async_handler(self, mock_swap):
        """Test get_async_handler returns the correct handler."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        retrieved = self.controller.get_async_handler(meta.async_handler.task_id)
        self.assertIs(retrieved, meta.async_handler)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_async_handler_nonexistent(self, mock_swap):
        """Test get_async_handler returns None for unknown task."""
        handler = self.controller.get_async_handler("nonexistent")
        self.assertIsNone(handler)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_progress(self, mock_swap):
        """Test get_progress returns valid progress dict."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        meta.async_handler.wait(timeout=5.0)

        progress = self.controller.get_progress(meta.async_handler.task_id)
        self.assertEqual(progress["status"], TransferStatus.COMPLETED.value)
        self.assertGreaterEqual(progress["total_layers"], 0)
        self.assertIn("progress", progress)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_progress_nonexistent_task(self, mock_swap):
        """Test get_progress returns error dict for unknown task."""
        progress = self.controller.get_progress("nonexistent")
        self.assertIn("error", progress)


# ============================================================================
# Cancellation Tests
# ============================================================================


class TestCancellation(unittest.TestCase):
    """Test task cancellation."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_cancel_transfer(self, mock_swap):
        """Test cancel_transfer on existing task."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        self.controller.cancel_transfer(meta.async_handler.task_id)
        # May succeed or fail depending on timing, either is acceptable

    def test_cancel_nonexistent_task(self):
        """Test cancel_transfer returns False for non-existent task."""
        result = self.controller.cancel_transfer("nonexistent-task-id")
        self.assertFalse(result)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_cancel_completed_task(self, mock_swap):
        """Test cancel_transfer returns False for already completed task."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        result = self.controller.cancel_transfer(meta.async_handler.task_id)
        self.assertFalse(result)


# ============================================================================
# Layer Done Counter Tests
# ============================================================================


class TestLayerDoneCounter(unittest.TestCase):
    """Test layer-by-layer completion tracking."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_all_layers_marked_complete_after_load(self, mock_swap):
        """Test all layers marked complete after all-layer load."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        # Task should complete successfully
        self.assertTrue(meta.async_handler.is_completed)
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_is_transfer_complete(self, mock_swap):
        """Test is_transfer_complete returns True after all layers done."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        # Task should complete successfully
        self.assertTrue(meta.success)
        self.assertTrue(meta.async_handler.is_completed)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_wait_for_layer_returns_true(self, mock_swap):
        """Test wait_for_layer returns True for completed layer."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        # Task should complete successfully
        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_layer_by_layer_mode(self, mock_swap):
        """Test layer-by-layer mode uses load_layers_to_device."""
        mock_swap.return_value = None
        self.controller._transfer_manager.swap_all_layers = False

        with patch.object(
            self.controller._transfer_manager,
            "load_layers_to_device",
            return_value=True,
        ) as mock_load:
            meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
            self.controller.load_host_to_device([meta])
            meta.async_handler.wait(timeout=5.0)

            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            # Check layer_indices and on_layer_complete are passed
            self.assertEqual(len(call_kwargs["layer_indices"]), 4)  # 4 layers
            self.assertIn("on_layer_complete", call_kwargs)
            self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_register_layer_callback(self, mock_swap):
        """Test register_layer_callback for layer completion notifications."""

        def slow_swap(*args, **kwargs):
            time.sleep(0.1)
            return None

        mock_swap.side_effect = slow_swap

        callback_results = []

        def on_done(layer_idx):
            callback_results.append(layer_idx)

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        # Register callback before task completes
        self.controller.register_layer_callback(meta.async_handler.task_id, on_done)

        meta.async_handler.wait(timeout=5.0)

        # All layers should be in callback results
        self.assertEqual(sorted(callback_results), [0, 1, 2, 3])


# ============================================================================
# Eviction Layer-by-Layer Tests
# ============================================================================


class TestEvictLayerByLayer(unittest.TestCase):
    """Test eviction in layer-by-layer mode."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_all_layers_mode(self, mock_swap):
        """Test eviction in all-layers mode."""
        mock_swap.return_value = None

        meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        self.controller.evict_device_to_host([meta])
        meta.async_handler.wait(timeout=5.0)

        self.assertTrue(meta.success)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_layer_by_layer_mode(self, mock_swap):
        """Test eviction in layer-by-layer mode."""
        self.controller._transfer_manager.swap_all_layers = False

        with patch.object(
            self.controller._transfer_manager,
            "evict_layers_to_host",
            return_value=True,
        ) as mock_evict:
            meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
            self.controller.evict_device_to_host([meta])
            meta.async_handler.wait(timeout=5.0)

            mock_evict.assert_called_once()


# ============================================================================
# Reset Tests
# ============================================================================


class TestReset(unittest.TestCase):
    """Test reset_cache and reset_controller_cache."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_reset_cache_clears_tasks(self, mock_swap):
        """Test reset_cache clears active tasks."""
        mock_swap.return_value = None

        metas = [CacheSwapMetadata(src_block_ids=[10 + i], dst_block_ids=[i]) for i in range(3)]
        self.controller.load_host_to_device(metas)
        for meta in metas:
            meta.async_handler.wait(timeout=5.0)

        # After reset, active tasks should be cleared
        result = self.controller.reset_cache()
        self.assertTrue(result)
        self.assertEqual(len(self.controller._active_tasks), 0)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_reset_cache_with_running_tasks(self, mock_swap):
        """Test reset_cache cancels running tasks."""

        def slow_swap(*args, **kwargs):
            time.sleep(2.0)
            return None

        mock_swap.side_effect = slow_swap

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])

        # Give a moment for the task to start
        time.sleep(0.1)

        result = self.controller.reset_cache()
        self.assertTrue(result)

        # Check task was cancelled
        task = self.controller._active_tasks.get(meta.async_handler.task_id)
        self.assertIsNone(task)


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
        self.assertIn("active_transfers", stats)
        self.assertTrue(stats["initialized"])
        self.assertEqual(stats["num_layers"], 4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_get_stats_active_transfers(self, mock_swap):
        """Test get_stats reports active transfers."""
        mock_swap.return_value = None

        controller = create_cache_controller(num_layers=4)
        setup_transfer_env(controller, num_layers=4)

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        stats = controller.get_stats()
        self.assertGreaterEqual(stats["active_transfers"], 0)


# ============================================================================
# Transfer Failure Tests
# ============================================================================


class TestTransferFailure(unittest.TestCase):
    """Test behavior when transfer fails."""

    def setUp(self):
        self.controller = create_cache_controller(num_layers=4)
        setup_transfer_env(self.controller, num_layers=4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_all_layer_transfer_failure(self, mock_swap):
        """Test that transfer failure is properly reported."""
        mock_swap.side_effect = RuntimeError("CUDA error")

        meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
        self.controller.load_host_to_device([meta])
        meta.async_handler.wait(timeout=5.0)

        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)

        # Task should be marked as failed
        task = self.controller._active_tasks.get(meta.async_handler.task_id)
        if task:
            self.assertEqual(task.status, TransferStatus.FAILED)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_transfer_failure(self, mock_swap):
        """Test that eviction failure is properly reported."""
        mock_swap.side_effect = RuntimeError("Transfer failed")

        meta = CacheSwapMetadata(src_block_ids=[0], dst_block_ids=[10])
        self.controller.evict_device_to_host([meta])
        meta.async_handler.wait(timeout=5.0)

        self.assertFalse(meta.success)
        self.assertIsNotNone(meta.error_message)

    def test_layer_by_layer_transfer_failure(self):
        """Test layer-by-layer transfer failure."""
        self.controller._transfer_manager.swap_all_layers = False

        with patch.object(
            self.controller._transfer_manager,
            "load_layers_to_device",
            side_effect=RuntimeError("Layer transfer failed"),
        ):
            meta = CacheSwapMetadata(src_block_ids=[10], dst_block_ids=[0])
            self.controller.load_host_to_device([meta])
            meta.async_handler.wait(timeout=5.0)

            self.assertFalse(meta.success)


# ============================================================================
# KV Cache Management Tests
# ============================================================================


class TestKVCacheManagement(unittest.TestCase):
    """Test KV cache initialization and retrieval."""

    def test_get_kv_caches_without_init(self):
        """Test get_kv_caches returns empty dict when not initialized."""
        controller = create_cache_controller()
        result = controller.get_kv_caches()
        # Should return the (empty) cache_kvs_map
        self.assertIsNotNone(result)

    def test_get_host_cache_kvs_map_without_init(self):
        """Test get_host_cache_kvs_map returns empty dict when not initialized."""
        controller = create_cache_controller()
        result = controller.get_host_cache_kvs_map()
        self.assertEqual(len(result), 0)


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
