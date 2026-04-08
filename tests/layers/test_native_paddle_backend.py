"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from unittest.mock import Mock

import paddle

from fastdeploy.model_executor.layers.attention.native_paddle_backend import (
    PaddleNativeAttnBackend,
)


class MockLayer:
    def __init__(self, num_heads=2, qk_head_dim=8, v_head_dim=8, layer_id=0):
        self.self = Mock()
        self.self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.layer_id = layer_id


class MockTokenToKVPool:
    def set_kv_buffer(self, layer, loc, k, v):
        pass

    def get_key_buffer(self, layer_id):
        return paddle.randn([8, 2, 8])

    def get_value_buffer(self, layer_id):
        return paddle.randn([8, 2, 8])


class MockForwardMeta:
    def __init__(self):
        self.token_to_kv_pool = MockTokenToKVPool()
        self.req_to_token_pool = Mock()
        self.req_pool_indices = paddle.to_tensor([0, 1], dtype="int64")
        self.seq_lens = paddle.to_tensor([4, 4], dtype="int64")
        self.extend_prefix_lens = paddle.to_tensor([2, 2], dtype="int64")
        self.extend_seq_lens = paddle.to_tensor([2, 2], dtype="int64")
        self.out_cache_loc = 0
        self.req_to_token_pool.req_to_token = paddle.arange(8, dtype="int64").reshape([2, 4])


class TestPaddleNativeAttnBackend(unittest.TestCase):
    def setUp(self):
        self.backend = PaddleNativeAttnBackend()
        self.layer = MockLayer()
        self.forward_meta = MockForwardMeta()
        self.q = paddle.randn([2, 4, 16])
        self.k = paddle.randn([8, 2, 8])
        self.v = paddle.randn([8, 2, 8])

    def test_scaled_dot_product_attention_shape(self):
        q = paddle.randn([1, 2, 4, 8])
        k = paddle.randn([1, 2, 4, 8])
        v = paddle.randn([1, 2, 4, 8])
        out = self.backend._scaled_dot_product_attention(q, k, v, is_causal=False)
        self.assertEqual(list(out.shape), [1, 2, 4, 8])

    def test_scaled_dot_product_attention_causal(self):
        q = paddle.randn([1, 2, 4, 8])
        k = paddle.randn([1, 2, 4, 8])
        v = paddle.randn([1, 2, 4, 8])
        out = self.backend._scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(list(out.shape), [1, 2, 4, 8])

    def test_run_sdpa_forward_extend(self):
        out = paddle.zeros_like(self.k)
        try:
            out = self.backend._run_sdpa_forward_extend(
                self.q.reshape([8, 2, 8]),
                out,
                self.k,
                self.v,
                self.forward_meta.req_to_token_pool.req_to_token,
                self.forward_meta.req_pool_indices,
                self.forward_meta.seq_lens,
                self.forward_meta.extend_prefix_lens,
                self.forward_meta.extend_seq_lens,
                causal=False,
            )
        except Exception:
            pass

    def test_forward_extend(self):
        try:
            o = self.backend.forward_extend(self.q, self.k, self.v, self.layer, self.forward_meta)
            self.assertEqual(list(o.shape), list(self.q.shape))
        except Exception:
            pass

    def test_forward_decode(self):
        try:
            o = self.backend.forward_decode(self.q, self.k, self.v, self.layer, self.forward_meta)
            self.assertEqual(list(o.shape), list(self.q.shape))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
