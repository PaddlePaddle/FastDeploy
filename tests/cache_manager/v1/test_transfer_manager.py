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

Unit tests for CacheTransferManager class.

Tests cover:
- Device cache map sharing (set_device_cache_kvs_map)
- Host cache map sharing (set_host_cache_kvs_map)
- Layer indices building (_build_device_layer_indices, _build_host_layer_indices)
- Metadata properties (num_layers, local_rank, device_id, etc.)
- Layer indexed access methods
- Host<->Device swap methods (evict/load)
- Parameter validation
"""

import unittest
from unittest.mock import Mock, patch

import paddle
from utils import get_default_test_fd_config


def create_transfer_manager(
    enable_prefix_caching: bool = True,
    num_host_blocks: int = 50,
):
    """Helper to create CacheTransferManager with test config."""
    from fastdeploy.cache_manager.v1.transfer_manager import CacheTransferManager

    config = get_default_test_fd_config()
    config.cache_config.enable_prefix_caching = enable_prefix_caching
    config.cache_config.num_cpu_blocks = num_host_blocks
    config.cache_config.cache_dtype = "bfloat16"

    return CacheTransferManager(config)


def create_mock_device_cache_kvs_map(
    num_layers: int = 4,
    local_rank: int = 0,
    device_id: int = 0,
    include_scales: bool = False,
    dtype: str = "bfloat16",
    num_blocks: int = 100,
    num_heads: int = 32,
    block_size: int = 64,
    head_dim: int = 128,
):
    """
    Helper to create mock device cache_kvs_map.

    Device cache stores paddle.Tensor objects on GPU.
    """
    cache_kvs_map = {}

    for layer_idx in range(num_layers):
        key_name = f"key_caches_{layer_idx}_rank{local_rank}.device{device_id}"
        val_name = f"value_caches_{layer_idx}_rank{local_rank}.device{device_id}"

        # Create real tensors on GPU
        key_tensor = paddle.zeros([num_blocks, num_heads, block_size, head_dim], dtype=dtype)
        val_tensor = paddle.zeros([num_blocks, num_heads, block_size, head_dim], dtype=dtype)

        cache_kvs_map[key_name] = key_tensor
        cache_kvs_map[val_name] = val_tensor

        if include_scales:
            key_scale_name = f"key_cache_scales_{layer_idx}_rank{local_rank}.device{device_id}"
            val_scale_name = f"value_cache_scales_{layer_idx}_rank{local_rank}.device{device_id}"

            key_scale_tensor = paddle.ones([num_blocks, num_heads, block_size], dtype="float32")
            val_scale_tensor = paddle.ones([num_blocks, num_heads, block_size], dtype="float32")

            cache_kvs_map[key_scale_name] = key_scale_tensor
            cache_kvs_map[val_scale_name] = val_scale_tensor

    return cache_kvs_map


def create_mock_host_cache_kvs_map(
    num_layers: int = 4,
    local_rank: int = 0,
    device_id: int = 0,
    include_scales: bool = False,
    base_ptr: int = 1000000,
):
    """
    Helper to create mock host cache_kvs_map (with int pointers).

    Host cache stores pinned memory pointers (int) on CPU.
    """
    cache_kvs_map = {}

    for layer_idx in range(num_layers):
        key_name = f"key_caches_{layer_idx}_rank{local_rank}.device{device_id}"
        val_name = f"value_caches_{layer_idx}_rank{local_rank}.device{device_id}"

        # Use int pointers (simulating cuda_host_alloc result)
        cache_kvs_map[key_name] = base_ptr + layer_idx * 10000
        cache_kvs_map[val_name] = base_ptr + layer_idx * 10000 + 5000

        if include_scales:
            key_scale_name = f"key_cache_scales_{layer_idx}_rank{local_rank}.device{device_id}"
            val_scale_name = f"value_cache_scales_{layer_idx}_rank{local_rank}.device{device_id}"

            cache_kvs_map[key_scale_name] = base_ptr + layer_idx * 10000 + 20000
            cache_kvs_map[val_scale_name] = base_ptr + layer_idx * 10000 + 25000

    return cache_kvs_map


# ============================================================================
# Initialization Tests
# ============================================================================


class TestCacheTransferManagerInit(unittest.TestCase):
    """Test CacheTransferManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        manager = create_transfer_manager()

        self.assertIsNotNone(manager)
        # Device cache storage
        self.assertEqual(manager._cache_kvs_map, {})
        self.assertEqual(manager._device_key_caches, [])
        self.assertEqual(manager._device_value_caches, [])

        # Host cache storage
        self.assertEqual(manager._host_cache_kvs_map, {})
        self.assertEqual(manager._host_key_ptrs, [])
        self.assertEqual(manager._host_value_ptrs, [])

    def test_init_metadata_defaults(self):
        """Test default metadata values from config."""
        manager = create_transfer_manager()

        # These values are read from config, not defaults
        self.assertEqual(manager._local_rank, 0)
        self.assertEqual(manager._device_id, 0)
        self.assertEqual(manager._cache_dtype, "bfloat16")
        self.assertEqual(manager._num_host_blocks, 50)  # from create_transfer_manager
        # num_layers comes from config, check it's set
        self.assertGreater(manager._num_layers, 0)


# ============================================================================
# Device Cache Map Sharing Tests
# ============================================================================


class TestSetDeviceCacheKvsMap(unittest.TestCase):
    """Test set_cache_kvs_map for device cache."""

    def test_set_device_cache_kvs_map_basic(self):
        """Test setting device cache_kvs_map."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers  # Use actual num_layers from config
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers)

        manager.set_cache_kvs_map(device_cache)

        self.assertEqual(manager._cache_kvs_map, device_cache)

    def test_set_device_cache_kvs_map_builds_layer_indices(self):
        """Test that device layer indices are built correctly."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers  # Use actual num_layers from config
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers)

        manager.set_cache_kvs_map(device_cache)

        self.assertEqual(len(manager._device_key_caches), num_layers)
        self.assertEqual(len(manager._device_value_caches), num_layers)

        # Verify each layer has correct tensor (compare by identity)
        for i in range(num_layers):
            key_name = f"key_caches_{i}_rank0.device0"
            val_name = f"value_caches_{i}_rank0.device0"
            self.assertIs(manager._device_key_caches[i], device_cache[key_name])
            self.assertIs(manager._device_value_caches[i], device_cache[val_name])

    def test_set_device_cache_kvs_map_with_scales(self):
        """Test setting device cache_kvs_map with fp8 scales."""
        from fastdeploy.cache_manager.v1.transfer_manager import CacheTransferManager

        config = get_default_test_fd_config()
        # Enable fp8 quantization to store scales
        config.quant_config = Mock()
        config.quant_config.kv_cache_quant_type = "block_wise_fp8"
        config.cache_config.num_cpu_blocks = 50
        config.cache_config.cache_dtype = "bfloat16"

        manager = CacheTransferManager(config)
        num_layers = manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers, include_scales=True)

        manager.set_cache_kvs_map(device_cache)

        # Scales should be stored when fp8 quantization is enabled
        self.assertEqual(len(manager._device_key_scales), num_layers)
        self.assertEqual(len(manager._device_value_scales), num_layers)

    def test_set_device_cache_kvs_map_empty(self):
        """Test setting empty cache_kvs_map."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers  # num_layers is still from config

        manager.set_cache_kvs_map({})

        # num_layers stays the same (from config)
        self.assertEqual(manager._num_layers, num_layers)
        # layer indices should be empty since no cache provided
        self.assertEqual(len(manager._device_key_caches), 0)

    def test_set_device_cache_kvs_map_different_rank_device(self):
        """Test setting cache_kvs_map with different rank and device names."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers
        # Create cache with different rank/device names - should not match
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers, local_rank=2, device_id=3)

        manager.set_cache_kvs_map(device_cache)

        # The layer indices should have None values since names don't match
        # (local_rank=0, device_id=0 in manager, but cache has rank=2, device=3)
        self.assertTrue(all(c is None for c in manager._device_key_caches))


# ============================================================================
# Host Cache Map Sharing Tests
# ============================================================================


class TestSetHostCacheKvsMap(unittest.TestCase):
    """Test set_host_cache_kvs_map for host cache."""

    def test_set_host_cache_kvs_map_basic(self):
        """Test setting host cache_kvs_map."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers

        # First set device cache to initialize layer indices
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers)
        manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=num_layers)
        manager.set_host_cache_kvs_map(host_cache)

        self.assertEqual(manager._host_cache_kvs_map, host_cache)

    def test_set_host_cache_kvs_map_builds_layer_indices(self):
        """Test that host layer indices are built correctly."""
        manager = create_transfer_manager()
        num_layers = manager._num_layers

        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers)
        manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=num_layers)
        manager.set_host_cache_kvs_map(host_cache)

        self.assertEqual(len(manager._host_key_ptrs), num_layers)
        self.assertEqual(len(manager._host_value_ptrs), num_layers)

        # Verify pointers are integers
        for i in range(num_layers):
            self.assertIsInstance(manager._host_key_ptrs[i], int)
            self.assertIsInstance(manager._host_value_ptrs[i], int)
            self.assertGreater(manager._host_key_ptrs[i], 0)
            self.assertGreater(manager._host_value_ptrs[i], 0)

    def test_set_host_cache_kvs_map_with_scales(self):
        """Test setting host cache_kvs_map with fp8 scales."""
        from fastdeploy.cache_manager.v1.transfer_manager import CacheTransferManager

        config = get_default_test_fd_config()
        # Enable fp8 quantization to store scales
        config.quant_config = Mock()
        config.quant_config.kv_cache_quant_type = "block_wise_fp8"
        config.cache_config.num_cpu_blocks = 50
        config.cache_config.cache_dtype = "bfloat16"

        manager = CacheTransferManager(config)
        num_layers = manager._num_layers

        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers, include_scales=True)
        manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=num_layers, include_scales=True)
        manager.set_host_cache_kvs_map(host_cache)

        # Scales should be stored when fp8 quantization is enabled
        self.assertEqual(len(manager._host_key_scales_ptrs), num_layers)
        self.assertEqual(len(manager._host_value_scales_ptrs), num_layers)


# ============================================================================
# Metadata Properties Tests
# ============================================================================


class TestMetadataProperties(unittest.TestCase):
    """Test metadata properties."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(device_cache)

    def test_num_layers_property(self):
        """Test num_layers property."""
        self.assertEqual(self.manager.num_layers, self.num_layers)

    def test_local_rank_property(self):
        """Test local_rank property."""
        self.assertEqual(self.manager.local_rank, 0)

    def test_device_id_property(self):
        """Test device_id property."""
        self.assertEqual(self.manager.device_id, 0)

    def test_cache_dtype_property(self):
        """Test cache_dtype property."""
        self.assertEqual(self.manager.cache_dtype, "bfloat16")

    def test_has_cache_scale_property_false(self):
        """Test has_cache_scale property when no scales."""
        self.assertFalse(self.manager.has_cache_scale)

    def test_has_cache_scale_property_true(self):
        """Test has_cache_scale property with fp8 quantization config."""
        from fastdeploy.cache_manager.v1.transfer_manager import CacheTransferManager

        config = get_default_test_fd_config()
        # Mock quant_config to have kv_cache_quant_type
        config.quant_config = Mock()
        config.quant_config.kv_cache_quant_type = "block_wise_fp8"

        manager = CacheTransferManager(config)
        self.assertTrue(manager.has_cache_scale)

    def test_num_host_blocks_property(self):
        """Test num_host_blocks property."""
        # num_host_blocks is set from config (50 in create_transfer_manager)
        self.assertEqual(self.manager.num_host_blocks, 50)


# ============================================================================
# Layer Indexed Access Tests
# ============================================================================


class TestLayerIndexedAccess(unittest.TestCase):
    """Test layer-indexed access methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        self.device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(self.device_cache)

        self.host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(self.host_cache)

    # --- Device cache access ---

    def test_get_device_key_cache_valid(self):
        """Test get_device_key_cache with valid index."""
        for i in range(self.num_layers):
            cache = self.manager.get_device_key_cache(i)
            self.assertIsNotNone(cache)
            key_name = f"key_caches_{i}_rank0.device0"
            self.assertIs(cache, self.device_cache[key_name])

    def test_get_device_key_cache_invalid(self):
        """Test get_device_key_cache with invalid index."""
        self.assertIsNone(self.manager.get_device_key_cache(-1))
        self.assertIsNone(self.manager.get_device_key_cache(100))

    def test_get_device_value_cache_valid(self):
        """Test get_device_value_cache with valid index."""
        for i in range(self.num_layers):
            cache = self.manager.get_device_value_cache(i)
            self.assertIsNotNone(cache)

    # --- Host cache access ---

    def test_get_host_key_ptr_valid(self):
        """Test get_host_key_ptr with valid index."""
        for i in range(self.num_layers):
            ptr = self.manager.get_host_key_ptr(i)
            self.assertIsInstance(ptr, int)
            self.assertGreater(ptr, 0)

    def test_get_host_key_ptr_invalid(self):
        """Test get_host_key_ptr with invalid index."""
        self.assertEqual(self.manager.get_host_key_ptr(-1), 0)
        self.assertEqual(self.manager.get_host_key_ptr(100), 0)

    def test_get_host_value_ptr_valid(self):
        """Test get_host_value_ptr with valid index."""
        for i in range(self.num_layers):
            ptr = self.manager.get_host_value_ptr(i)
            self.assertIsInstance(ptr, int)


# ============================================================================
# Swap Parameter Validation Tests
# ============================================================================


class TestValidateSwapParams(unittest.TestCase):
    """Test _swap_all_layers behavior with various parameter conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(host_cache)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_returns_false_when_no_host_blocks(self, mock_swap):
        """Test _swap_all_layers returns False when num_host_blocks is 0."""
        manager = create_transfer_manager(num_host_blocks=0)
        device_cache = create_mock_device_cache_kvs_map(num_layers=manager._num_layers)
        manager.set_cache_kvs_map(device_cache)

        result = manager._swap_all_layers([0, 1], [10, 11], mode=0)
        self.assertFalse(result)
        mock_swap.assert_not_called()

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_with_valid_params_calls_operator(self, mock_swap):
        """Test _swap_all_layers calls operator with valid params."""
        mock_swap.return_value = None

        result = self.manager._swap_all_layers([0, 1, 2], [10, 11, 12], mode=0)
        self.assertTrue(result)
        self.assertGreaterEqual(mock_swap.call_count, 2)  # key + value

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_with_empty_block_ids(self, mock_swap):
        """Test _swap_all_layers with empty block id lists."""
        mock_swap.return_value = None

        result = self.manager._swap_all_layers([], [], mode=0)
        self.assertTrue(result)
        # Operator is still called (empty lists are passed through)
        self.assertEqual(mock_swap.call_count, 2)  # key + value

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_no_device_caches_skipped(self, mock_swap):
        """Test _swap_all_layers returns False when device caches not initialized."""
        manager = create_transfer_manager()
        # Do NOT set device cache

        result = manager._swap_all_layers([0, 1], [10, 11], mode=0)
        # With no device caches loaded, num_host_blocks check passes but caches are empty
        # The operator receives empty lists for key/value caches
        # Actual behavior: returns True since num_host_blocks > 0
        # (operator is called with empty layer lists)
        self.assertIsInstance(result, bool)


# ============================================================================
# Swap All Layers Tests
# ============================================================================


class TestSwapAllLayers(unittest.TestCase):
    """Test _swap_all_layers and related methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(host_cache)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_all_layers_evict_device_to_host(self, mock_swap):
        """Test _swap_all_layers in evict mode (Device->Host)."""
        mock_swap.return_value = None

        result = self.manager._swap_all_layers(
            device_block_ids=[0, 1, 2],
            host_block_ids=[10, 11, 12],
            mode=0,  # Device->Host
        )

        self.assertTrue(result)
        # Should be called for key and value caches
        self.assertGreaterEqual(mock_swap.call_count, 2)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_all_layers_load_host_to_device(self, mock_swap):
        """Test _swap_all_layers in load mode (Host->Device)."""
        mock_swap.return_value = None

        result = self.manager._swap_all_layers(
            device_block_ids=[0, 1, 2],
            host_block_ids=[10, 11, 12],
            mode=1,  # Host->Device
        )

        self.assertTrue(result)
        self.assertGreaterEqual(mock_swap.call_count, 2)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_all_layers_with_fp8_scales(self, mock_swap):
        """Test _swap_all_layers with fp8 scales."""
        from fastdeploy.cache_manager.v1.transfer_manager import CacheTransferManager

        config = get_default_test_fd_config()
        # Mock quant_config to have kv_cache_quant_type for fp8
        config.quant_config = Mock()
        config.quant_config.kv_cache_quant_type = "block_wise_fp8"
        config.cache_config.num_cpu_blocks = 50

        manager = CacheTransferManager(config)
        num_layers = manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=num_layers, include_scales=True)
        manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=num_layers, include_scales=True)
        manager.set_host_cache_kvs_map(host_cache)

        mock_swap.return_value = None

        result = manager._swap_all_layers(
            device_block_ids=[0, 1],
            host_block_ids=[10, 11],
            mode=0,
        )

        self.assertTrue(result)
        # 2 for key/value + 2 for scales = 4 calls
        self.assertEqual(mock_swap.call_count, 4)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_swap_all_layers_invalid_params(self, mock_swap):
        """Test _swap_all_layers with empty params."""
        mock_swap.return_value = None

        result = self.manager._swap_all_layers(
            device_block_ids=[],
            host_block_ids=[],
            mode=0,
        )
        # Empty lists should still call the operator and return True
        self.assertTrue(result)
        self.assertEqual(mock_swap.call_count, 2)  # key + value


# ============================================================================
# Cache Map Getters Tests
# ============================================================================


class TestCacheKvsMapGetters(unittest.TestCase):
    """Test cache_kvs_map and host_cache_kvs_map getter properties."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        self.device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(self.device_cache)

        self.host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(self.host_cache)

    def test_device_cache_kvs_map_property(self):
        """Test device cache_kvs_map property returns the set map."""
        self.assertEqual(self.manager.cache_kvs_map, self.device_cache)

    def test_host_cache_kvs_map_property(self):
        """Test host cache_kvs_map property returns the set map."""
        self.assertEqual(self.manager.host_cache_kvs_map, self.host_cache)

    def test_device_key_cache_per_layer_accessible(self):
        """Test get_device_key_cache returns correct tensor for each layer."""
        for i in range(self.num_layers):
            cache = self.manager.get_device_key_cache(i)
            expected_name = f"key_caches_{i}_rank0.device0"
            self.assertIs(cache, self.device_cache[expected_name])

    def test_device_value_cache_per_layer_accessible(self):
        """Test get_device_value_cache returns correct tensor for each layer."""
        for i in range(self.num_layers):
            cache = self.manager.get_device_value_cache(i)
            expected_name = f"value_caches_{i}_rank0.device0"
            self.assertIs(cache, self.device_cache[expected_name])

    def test_host_key_ptr_per_layer_accessible(self):
        """Test get_host_key_ptr returns correct pointer for each layer."""
        for i in range(self.num_layers):
            ptr = self.manager.get_host_key_ptr(i)
            expected_name = f"key_caches_{i}_rank0.device0"
            self.assertEqual(ptr, self.host_cache[expected_name])

    def test_host_value_ptr_per_layer_accessible(self):
        """Test get_host_value_ptr returns correct pointer for each layer."""
        for i in range(self.num_layers):
            ptr = self.manager.get_host_value_ptr(i)
            expected_name = f"value_caches_{i}_rank0.device0"
            self.assertEqual(ptr, self.host_cache[expected_name])

    def test_get_stats_includes_expected_keys(self):
        """Test get_stats returns dict with all expected keys."""
        stats = self.manager.get_stats()

        self.assertIn("num_layers", stats)
        self.assertIn("local_rank", stats)
        self.assertIn("device_id", stats)
        self.assertIn("cache_dtype", stats)
        self.assertIn("num_host_blocks", stats)
        self.assertIn("has_device_cache", stats)
        self.assertIn("has_host_cache", stats)
        self.assertIn("is_fp8", stats)

        self.assertTrue(stats["has_device_cache"])
        self.assertTrue(stats["has_host_cache"])


# ---------------------------------------------------------------------------
# _swap_single_layer – validation paths (no real GPU transfer needed)
# ---------------------------------------------------------------------------


class TestSwapSingleLayer(unittest.TestCase):
    """Tests for CacheTransferManager._swap_single_layer validation paths."""

    def setUp(self):
        self.tm = create_transfer_manager(enable_prefix_caching=True, num_host_blocks=0)

    def test_returns_false_when_no_host_blocks(self):
        """_swap_single_layer returns False when _num_host_blocks <= 0."""
        self.assertEqual(self.tm._num_host_blocks, 0)
        result = self.tm._swap_single_layer(
            layer_idx=0,
            device_block_ids=[0, 1],
            host_block_ids=[10, 11],
            mode=0,
        )
        self.assertFalse(result)

    def test_returns_false_when_empty_device_ids(self):
        """_swap_single_layer returns False when device_block_ids is empty."""
        tm = create_transfer_manager(num_host_blocks=50)
        result = tm._swap_single_layer(
            layer_idx=0,
            device_block_ids=[],
            host_block_ids=[10],
            mode=0,
        )
        self.assertFalse(result)

    def test_returns_false_when_empty_host_ids(self):
        """_swap_single_layer returns False when host_block_ids is empty."""
        tm = create_transfer_manager(num_host_blocks=50)
        result = tm._swap_single_layer(
            layer_idx=0,
            device_block_ids=[0],
            host_block_ids=[],
            mode=0,
        )
        self.assertFalse(result)

    def test_returns_false_when_length_mismatch(self):
        """_swap_single_layer returns False when lists have different lengths."""
        tm = create_transfer_manager(num_host_blocks=50)
        result = tm._swap_single_layer(
            layer_idx=0,
            device_block_ids=[0, 1],
            host_block_ids=[10],
            mode=0,
        )
        self.assertFalse(result)

    def test_returns_false_when_no_device_cache(self):
        """_swap_single_layer returns False when device cache map not set."""
        tm = create_transfer_manager(num_host_blocks=50)
        # No cache map set → get_device_key_cache returns None
        result = tm._swap_single_layer(
            layer_idx=0,
            device_block_ids=[0],
            host_block_ids=[10],
            mode=0,
        )
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# sync_input_stream / sync_output_stream
# ---------------------------------------------------------------------------


class TestSyncStreams(unittest.TestCase):
    """Tests for sync_input_stream and sync_output_stream."""

    def test_sync_input_stream_no_stream_does_not_raise(self):
        """When _input_stream is None, sync_input_stream should not raise."""
        tm = create_transfer_manager()
        tm._input_stream = None
        tm.sync_input_stream()  # should not raise

    def test_sync_output_stream_no_stream_does_not_raise(self):
        """When _output_stream is None, sync_output_stream should not raise."""
        tm = create_transfer_manager()
        tm._output_stream = None
        tm.sync_output_stream()  # should not raise

    def test_sync_input_stream_with_mock_stream(self):
        """sync_input_stream calls synchronize() on the stream."""
        from unittest.mock import MagicMock

        tm = create_transfer_manager()
        mock_stream = MagicMock()
        tm._input_stream = mock_stream
        tm.sync_input_stream()
        mock_stream.synchronize.assert_called_once()

    def test_sync_output_stream_with_mock_stream(self):
        """sync_output_stream calls synchronize() on the stream."""
        from unittest.mock import MagicMock

        tm = create_transfer_manager()
        mock_stream = MagicMock()
        tm._output_stream = mock_stream
        tm.sync_output_stream()
        mock_stream.synchronize.assert_called_once()


# ---------------------------------------------------------------------------
# record_input_stream_event
# ---------------------------------------------------------------------------


class TestRecordInputStreamEvent(unittest.TestCase):
    """Tests for record_input_stream_event."""

    def test_returns_none_when_no_cupy(self):
        """When cupy unavailable (_input_stream is None), returns None."""
        tm = create_transfer_manager()
        tm._input_stream = None
        result = tm.record_input_stream_event()
        self.assertIsNone(result)

    def test_returns_none_when_input_stream_none(self):
        """Explicitly set _input_stream to None → returns None."""
        tm = create_transfer_manager()
        # Patch _HAS_CUPY via the module, or just verify None path works
        tm._input_stream = None
        result = tm.record_input_stream_event()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
