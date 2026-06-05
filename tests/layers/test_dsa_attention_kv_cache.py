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

"""Unit tests for DSAAttentionBackend KV cache methods (create_kv_cache,
create_host_kv_cache)."""

import unittest

from fastdeploy.model_executor.layers.attention.dsa_attention_backend import (
    DSAAttentionBackend,
)


def _make_dsa_backend(
    block_size=64,
    kv_lora_rank=576,
    qk_rope_head_dim=64,
    index_head_dim=128,
    index_n_heads=32,
    index_topk=128,
    quant_block_size=128,
):
    """Create a DSAAttentionBackend with mocked __init__ and required attrs."""
    backend = DSAAttentionBackend.__new__(DSAAttentionBackend)
    backend.block_size = block_size
    backend.kv_lora_rank = kv_lora_rank
    backend.qk_rope_head_dim = qk_rope_head_dim
    backend.index_head_dim = index_head_dim
    backend.index_n_heads = index_n_heads
    backend.index_topk = index_topk
    backend.quant_block_size = quant_block_size
    return backend


class TestDSACreateKVCache(unittest.TestCase):
    """Test DSAAttentionBackend.create_kv_cache (uint8 key + indexer, no value/scales)."""

    def test_key_and_indexer_tensors(self):
        backend = _make_dsa_backend()
        caches = backend.create_kv_cache(num_layers=2, num_blocks=5)
        # 2 layers * (key + indexer) = 4 entries
        self.assertEqual(len(caches), 4)
        for layer_idx in range(2):
            self.assertIn(("key", layer_idx), caches)
            self.assertIn(("indexer", layer_idx), caches)
            self.assertNotIn(("value", layer_idx), caches)
            self.assertNotIn(("key_scale", layer_idx), caches)

    def test_dtype_is_uint8(self):
        backend = _make_dsa_backend()
        caches = backend.create_kv_cache(num_layers=1, num_blocks=3, cache_dtype="bfloat16")
        # DSA ignores cache_dtype and always uses uint8
        self.assertEqual(str(caches[("key", 0)].dtype), "paddle.uint8")
        self.assertEqual(str(caches[("indexer", 0)].dtype), "paddle.uint8")

    def test_key_shape(self):
        backend = _make_dsa_backend(block_size=32, kv_lora_rank=192, qk_rope_head_dim=32)
        caches = backend.create_kv_cache(num_layers=1, num_blocks=7)
        k = caches[("key", 0)]
        # key_shape = [num_blocks, 1, block_size, fp8_key_cache_dim]
        self.assertEqual(k.shape[0], 7)
        self.assertEqual(k.shape[1], 1)
        self.assertEqual(k.shape[2], 32)

    def test_indexer_shape(self):
        backend = _make_dsa_backend(
            block_size=32,
            kv_lora_rank=192,
            qk_rope_head_dim=32,
            index_head_dim=64,
            index_n_heads=16,
            index_topk=4,
            quant_block_size=128,
        )
        caches = backend.create_kv_cache(num_layers=1, num_blocks=7)
        idx = caches[("indexer", 0)]
        # indexer_shape = [num_blocks, block_size, fp8_indexer_dim]
        self.assertEqual(idx.shape[0], 7)
        self.assertEqual(idx.shape[1], 32)

    def test_layer_offset(self):
        backend = _make_dsa_backend()
        caches = backend.create_kv_cache(num_layers=2, num_blocks=3, layer_offset=10)
        self.assertIn(("key", 10), caches)
        self.assertIn(("indexer", 10), caches)
        self.assertNotIn(("key", 0), caches)


class TestDSACreateHostKVCache(unittest.TestCase):
    """Test DSAAttentionBackend.create_host_kv_cache (raises NotImplementedError)."""

    def test_not_implemented(self):
        backend = _make_dsa_backend()
        with self.assertRaises(NotImplementedError):
            backend.create_host_kv_cache(num_layers=1, num_blocks=1, cache_item_bytes=1)


if __name__ == "__main__":
    unittest.main()
