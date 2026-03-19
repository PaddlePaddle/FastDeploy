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
    """Test _validate_swap_params method."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(device_cache)

        host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(host_cache)

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        self.assertTrue(self.manager._validate_swap_params([0, 1, 2], [10, 11, 12]))

    def test_validate_empty_device_blocks(self):
        """Test validation with empty device block list."""
        self.assertFalse(self.manager._validate_swap_params([], [10, 11]))

    def test_validate_empty_host_blocks(self):
        """Test validation with empty host block list."""
        self.assertFalse(self.manager._validate_swap_params([0, 1], []))

    def test_validate_mismatched_lengths(self):
        """Test validation with mismatched block list lengths."""
        self.assertFalse(self.manager._validate_swap_params([0, 1, 2], [10, 11]))

    def test_validate_no_device_caches(self):
        """Test validation when device caches not initialized."""
        manager = create_transfer_manager()
        self.assertFalse(manager._validate_swap_params([0, 1], [10, 11]))

    def test_validate_no_host_pointers(self):
        """Test validation when host pointers not initialized."""
        manager = create_transfer_manager()
        device_cache = create_mock_device_cache_kvs_map(num_layers=manager._num_layers)
        manager.set_cache_kvs_map(device_cache)
        # Don't set host cache
        self.assertFalse(manager._validate_swap_params([0, 1], [10, 11]))

    def test_validate_zero_host_blocks(self):
        """Test validation when num_host_blocks is zero."""
        manager = create_transfer_manager(num_host_blocks=0)
        device_cache = create_mock_device_cache_kvs_map(num_layers=manager._num_layers)
        manager.set_cache_kvs_map(device_cache)
        host_cache = create_mock_host_cache_kvs_map(num_layers=manager._num_layers)
        manager.set_host_cache_kvs_map(host_cache)
        self.assertFalse(manager._validate_swap_params([0, 1], [10, 11]))


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

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_evict_to_host_all_layers(self, mock_swap):
        """Test evict_to_host_all_layers wrapper."""
        mock_swap.return_value = None

        result = self.manager.evict_to_host_all_layers(
            device_block_ids=[0, 1, 2],
            host_block_ids=[10, 11, 12],
        )

        self.assertTrue(result)
        # Verify mode=0 was passed (7th positional argument)
        first_call = mock_swap.call_args
        self.assertEqual(first_call[0][6], 0)

    @patch("fastdeploy.cache_manager.v1.transfer_manager.swap_cache_all_layers")
    def test_load_to_device_all_layers(self, mock_swap):
        """Test load_to_device_all_layers wrapper."""
        mock_swap.return_value = None

        result = self.manager.load_to_device_all_layers(
            host_block_ids=[10, 11, 12],
            device_block_ids=[0, 1, 2],
        )

        self.assertTrue(result)
        # Verify mode=1 was passed (7th positional argument)
        first_call = mock_swap.call_args
        self.assertEqual(first_call[0][6], 1)


# ============================================================================
# Cache Map Getters Tests
# ============================================================================


class TestCacheKvsMapGetters(unittest.TestCase):
    """Test cache_kvs_map getter methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = create_transfer_manager()
        self.num_layers = self.manager._num_layers
        self.device_cache = create_mock_device_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_cache_kvs_map(self.device_cache)

        self.host_cache = create_mock_host_cache_kvs_map(num_layers=self.num_layers)
        self.manager.set_host_cache_kvs_map(self.host_cache)

    def test_device_cache_kvs_map_property(self):
        """Test device cache_kvs_map property."""
        self.assertEqual(self.manager.cache_kvs_map, self.device_cache)

    def test_host_cache_kvs_map_property(self):
        """Test host cache_kvs_map property."""
        self.assertEqual(self.manager.host_cache_kvs_map, self.host_cache)

    def test_get_device_cache_tensor_found(self):
        """Test get_cache_tensor when tensor exists."""
        tensor = self.manager.get_cache_tensor("key_caches_0_rank0.device0")
        self.assertIsNotNone(tensor)

    def test_get_device_cache_tensor_not_found(self):
        """Test get_cache_tensor when tensor doesn't exist."""
        tensor = self.manager.get_cache_tensor("nonexistent")
        self.assertIsNone(tensor)

    def test_get_host_cache_pointer_found(self):
        """Test get_host_cache_tensor when pointer exists."""
        ptr = self.manager.get_host_cache_tensor("key_caches_0_rank0.device0")
        self.assertIsNotNone(ptr)
        self.assertIsInstance(ptr, int)

    def test_get_layer_device_caches(self):
        """Test get_layer_caches returns correct tensors for a layer."""
        layer_caches = self.manager.get_layer_caches(0)

        self.assertIn("key_caches_0_rank0.device0", layer_caches)
        self.assertIn("value_caches_0_rank0.device0", layer_caches)
        self.assertEqual(len(layer_caches), 2)

    def test_get_layer_host_caches(self):
        """Test get_host_layer_caches returns correct pointers for a layer."""
        layer_caches = self.manager.get_host_layer_caches(0)

        self.assertIn("key_caches_0_rank0.device0", layer_caches)
        self.assertIn("value_caches_0_rank0.device0", layer_caches)


if __name__ == "__main__":
    unittest.main()
