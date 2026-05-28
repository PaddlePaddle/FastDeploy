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

"""Unit tests for the base AttentionBackend KV cache methods (create_kv_cache,
create_host_kv_cache, free_host_kv_cache)."""

import unittest
from unittest.mock import patch

from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)


class _ConcreteBackend(AttentionBackend):
    """Minimal concrete subclass that supplies get_kv_cache_shape."""

    def __init__(self, key_shape, value_shape):
        super().__init__()
        self._key_shape = list(key_shape)
        self._value_shape = list(value_shape) if value_shape is not None else None

    def init_attention_metadata(self, forward_meta):
        pass

    def get_kv_cache_shape(self, max_num_blocks, kv_cache_quant_type=None):
        key_shape = [max_num_blocks] + self._key_shape
        val_shape = [max_num_blocks] + self._value_shape if self._value_shape else []
        return key_shape, val_shape


def _make_backend(key_shape=(4, 16, 64), value_shape=(4, 16, 64)):
    return _ConcreteBackend(key_shape, value_shape)


class TestCreateKVCache(unittest.TestCase):
    """Test base AttentionBackend.create_kv_cache."""

    def test_basic_key_value(self):
        backend = _make_backend(key_shape=(2, 8, 32), value_shape=(2, 8, 32))
        caches = backend.create_kv_cache(num_layers=2, num_blocks=5, cache_dtype="float16")
        self.assertEqual(len(caches), 4)  # 2 layers * (key + value)
        for layer_idx in range(2):
            k = caches[("key", layer_idx)]
            v = caches[("value", layer_idx)]
            self.assertEqual(k.shape, [5, 2, 8, 32])
            self.assertEqual(v.shape, [5, 2, 8, 32])
            self.assertEqual(str(k.dtype), "paddle.float16")

    def test_fp8_creates_scale_tensors(self):
        backend = _make_backend(key_shape=(2, 8, 32), value_shape=(2, 8, 32))
        caches = backend.create_kv_cache(
            num_layers=1,
            num_blocks=3,
            cache_dtype="uint8",
            kv_cache_quant_type="block_wise_fp8",
        )
        self.assertIn(("key", 0), caches)
        self.assertIn(("value", 0), caches)
        self.assertIn(("key_scale", 0), caches)
        self.assertIn(("value_scale", 0), caches)
        # scale shape: [num_blocks, k1, k2] = [3, 2, 8]
        self.assertEqual(caches[("key_scale", 0)].shape, [3, 2, 8])

    def test_no_value_shape_only_key(self):
        backend = _ConcreteBackend(key_shape=(1, 4, 128), value_shape=None)
        caches = backend.create_kv_cache(num_layers=3, num_blocks=2, cache_dtype="bfloat16")
        self.assertEqual(len(caches), 3)  # only key per layer
        for layer_idx in range(3):
            self.assertIn(("key", layer_idx), caches)
            self.assertNotIn(("value", layer_idx), caches)

    def test_layer_offset(self):
        backend = _make_backend(key_shape=(1, 1, 1), value_shape=(1, 1, 1))
        caches = backend.create_kv_cache(num_layers=2, num_blocks=1, cache_dtype="float32", layer_offset=10)
        self.assertIn(("key", 10), caches)
        self.assertIn(("key", 11), caches)
        self.assertNotIn(("key", 0), caches)


class TestCreateHostKVCache(unittest.TestCase):
    """Test base AttentionBackend.create_host_kv_cache."""

    def setUp(self):
        self.backend = _make_backend(key_shape=(2, 8, 32), value_shape=(2, 8, 32))

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc")
    def test_basic_allocation(self, mock_alloc):
        mock_alloc.return_value = 12345
        caches = self.backend.create_host_kv_cache(num_layers=2, num_blocks=5, cache_item_bytes=2)
        self.assertEqual(len(caches), 4)  # key + value per layer
        for (role, layer_idx), ptr in caches.items():
            self.assertEqual(ptr, 12345)
        self.assertTrue(mock_alloc.called)

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc")
    def test_fp8_allocation_with_scales(self, mock_alloc):
        mock_alloc.return_value = 99999
        caches = self.backend.create_host_kv_cache(
            num_layers=1,
            num_blocks=3,
            cache_item_bytes=2,
            kv_cache_quant_type="block_wise_fp8",
        )
        self.assertIn(("key", 0), caches)
        self.assertIn(("value", 0), caches)
        self.assertIn(("key_scale", 0), caches)
        self.assertIn(("value_scale", 0), caches)

    def test_unavailable_raises_runtime_error(self):
        with patch(
            "fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc",
            None,
        ):
            with self.assertRaises(RuntimeError):
                self.backend.create_host_kv_cache(num_layers=1, num_blocks=1, cache_item_bytes=1)

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_alloc")
    def test_no_value_shape_host(self, mock_alloc):
        mock_alloc.return_value = 55555
        backend = _ConcreteBackend(key_shape=(1, 4, 128), value_shape=None)
        caches = backend.create_host_kv_cache(num_layers=2, num_blocks=3, cache_item_bytes=1)
        # Only key per layer, no value
        for layer_idx in range(2):
            self.assertIn(("key", layer_idx), caches)
            self.assertNotIn(("value", layer_idx), caches)


class TestFreeHostKVCache(unittest.TestCase):
    """Test base AttentionBackend.free_host_kv_cache."""

    def setUp(self):
        self.backend = _make_backend()

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free")
    def test_frees_all_entries(self, mock_free):
        host_caches = {"key_0": 100, "value_0": 200, "key_1": 300}
        self.backend.free_host_kv_cache(host_caches)
        self.assertEqual(mock_free.call_count, 3)
        self.assertEqual(len(host_caches), 0)

    def test_empty_dict_noop(self):
        host_caches = {}
        self.backend.free_host_kv_cache(host_caches)

    @patch("fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free")
    def test_skips_zero_ptr(self, mock_free):
        host_caches = {"key_0": 100, "null_key": 0}
        self.backend.free_host_kv_cache(host_caches)
        self.assertEqual(mock_free.call_count, 1)
        mock_free.assert_called_once_with(100)

    def test_unavailable_clears_and_warns(self):
        with patch(
            "fastdeploy.model_executor.layers.attention.base_attention_backend.cuda_host_free",
            None,
        ):
            host_caches = {"key_0": 100}
            self.backend.free_host_kv_cache(host_caches)
            self.assertEqual(len(host_caches), 0)


if __name__ == "__main__":
    unittest.main()
