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

"""Unit tests for MLAAttentionBackend KV cache methods (create_kv_cache,
create_host_kv_cache)."""

import unittest
from unittest.mock import patch

from fastdeploy.model_executor.layers.attention.mla_attention_backend import (
    MLAAttentionBackend,
)


def _make_mla_backend(block_size=64, kv_lora_rank=512, qk_rope_head_dim=64):
    """Create an MLAAttentionBackend with mocked __init__ and required attrs."""
    backend = MLAAttentionBackend.__new__(MLAAttentionBackend)
    backend.block_size = block_size
    backend.kv_lora_rank = kv_lora_rank
    backend.qk_rope_head_dim = qk_rope_head_dim
    return backend


class TestMLACreateKVCache(unittest.TestCase):
    """Test MLAAttentionBackend.create_kv_cache (latent key only, no value/scales)."""

    def test_only_key_tensors(self):
        backend = _make_mla_backend()
        caches = backend.create_kv_cache(num_layers=2, num_blocks=5, cache_dtype="bfloat16")
        self.assertEqual(len(caches), 2)  # only key per layer
        for layer_idx in range(2):
            self.assertIn(("key", layer_idx), caches)
            self.assertNotIn(("value", layer_idx), caches)
            self.assertNotIn(("key_scale", layer_idx), caches)

    def test_key_shape_correct(self):
        backend = _make_mla_backend(block_size=32, kv_lora_rank=256, qk_rope_head_dim=32)
        caches = backend.create_kv_cache(num_layers=1, num_blocks=10, cache_dtype="float16")
        t = caches[("key", 0)]
        # key_shape = [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
        expected_dim = 256 + 32
        self.assertEqual(t.shape, [10, 1, 32, expected_dim])

    def test_layer_offset(self):
        backend = _make_mla_backend()
        caches = backend.create_kv_cache(num_layers=2, num_blocks=3, cache_dtype="float16", layer_offset=4)
        self.assertIn(("key", 4), caches)
        self.assertIn(("key", 5), caches)
        self.assertNotIn(("key", 0), caches)


class TestMLACreateHostKVCache(unittest.TestCase):
    """Test MLAAttentionBackend.create_host_kv_cache."""

    def test_only_key_ptrs(self):
        backend = _make_mla_backend()
        with patch("fastdeploy.cache_manager.ops.cuda_host_alloc", return_value=77777):
            caches = backend.create_host_kv_cache(num_layers=2, num_blocks=5, cache_item_bytes=2)
        self.assertEqual(len(caches), 2)
        for layer_idx in range(2):
            self.assertIn(("key", layer_idx), caches)
            self.assertEqual(caches[("key", layer_idx)], 77777)

    def test_unavailable_raises(self):
        backend = _make_mla_backend()
        with patch("fastdeploy.cache_manager.ops.cuda_host_alloc", None):
            with self.assertRaises(RuntimeError):
                backend.create_host_kv_cache(num_layers=1, num_blocks=1, cache_item_bytes=1)


if __name__ == "__main__":
    unittest.main()
