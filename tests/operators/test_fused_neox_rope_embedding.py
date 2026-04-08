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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import fused_neox_rope_embedding


def rotate_half(x):
    Dh = x.shape[-1]
    x1 = x[..., : Dh // 2]
    x2 = x[..., Dh // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(x, cos, sin):
    orig_dtype = x.dtype
    x = x.astype("float32")
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.astype(orig_dtype)


class TestFusedNeoxRopeEmbedding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def native_neox_rope_embedding(self, qkv, cos, sin, num_heads):
        seq_length = qkv.shape[0]
        qkv = qkv.reshape(
            [
                seq_length,
                3,
                num_heads,
                -1,
            ]
        ).transpose(perm=[1, 0, 2, 3])
        q, k, v = qkv.unbind(axis=0)
        q = apply_rotary_pos_emb_vision(q, cos, sin)
        k = apply_rotary_pos_emb_vision(k, cos, sin)
        return q, k, v

    def test_fused_neox_rope_embedding(self):
        token_num = 1024
        hidden_size = 2048
        head_dim = 128
        num_heads = hidden_size // head_dim
        qkv = paddle.randn([token_num, 3 * hidden_size]).astype("bfloat16")
        cos_emb = paddle.rand([token_num, head_dim // 2]).tile((1, 2)).unsqueeze(1)
        sin_emb = paddle.rand([token_num, head_dim // 2]).tile((1, 2)).unsqueeze(1)
        q, k, v = fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
        q_base, k_base, v_base = self.native_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads)
        np.testing.assert_allclose(
            q.cast("float32").numpy(),
            q_base.cast("float32").numpy(),
            rtol=1e-02,
            atol=1e-02,
        )
        np.testing.assert_allclose(
            k.cast("float32").numpy(),
            k_base.cast("float32").numpy(),
            rtol=1e-02,
            atol=1e-02,
        )
        np.testing.assert_allclose(
            v.cast("float32").numpy(),
            v_base.cast("float32").numpy(),
            rtol=1e-02,
            atol=1e-02,
        )


if __name__ == "__main__":
    unittest.main()
