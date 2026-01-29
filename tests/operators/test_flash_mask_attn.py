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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import flash_mask_attention


class TestFlashMaskAttention(unittest.TestCase):
    def setUp(self):
        self.bsz = 1
        self.num_head = 8
        self.num_kv_head = 1
        self.q_len = 888
        self.k_len = 1024
        self.head_dim = 128
        np.random.seed(self.q_len)

    def naive_attn(self, q_input, k_input, v_input, mask):

        new_q = q_input.reshape([self.q_len, self.num_head, self.head_dim])
        new_k = (
            k_input.reshape([self.k_len + self.q_len, self.num_kv_head, self.head_dim])
            .tile([1, self.num_head, 1])
            .contiguous()
        )
        new_v = (
            v_input.reshape([self.k_len + self.q_len, self.num_kv_head, self.head_dim])
            .tile([1, self.num_head, 1])
            .contiguous()
        )

        p = paddle.einsum("ilk, jlk->lij", new_q, new_k)
        p = p / (np.sqrt(self.head_dim))

        tmp_zeros = np.zeros((self.q_len, self.q_len + self.k_len)) - 1
        cpu_mask = mask.cpu().numpy()
        for i in range(self.q_len):
            tmp_zeros[i][cpu_mask[2 * i] : cpu_mask[2 * i + 1]] = 0
        mask = tmp_zeros * 1000
        mask = paddle.to_tensor(mask, dtype=q_input.dtype)
        p = p + mask[None, :]
        p = paddle.nn.functional.softmax(p, -1)

        out = paddle.einsum("lij, jlk->ilk", p, new_v).reshape([self.q_len, self.num_head * self.head_dim])
        return out

    def paddle_flash_attn_mask(self, q_input, k_input, v_input, attn_out, mask):
        bsz = self.bsz
        cu_seq_q = paddle.arange(bsz + 1) * self.q_len
        cu_seq_k = paddle.arange(bsz + 1) * (self.q_len + self.k_len)
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")
        seq_len_encoder = paddle.ones(bsz) * self.q_len
        seq_len_encoder = seq_len_encoder.astype("int32")

        flash_mask_attention(
            q_input,
            k_input,
            v_input,
            cu_seq_q,
            cu_seq_k,
            seq_len_encoder,
            attn_out,
            mask,
            self.num_head,
            self.num_kv_head,
            self.head_dim,
        )

    def test_flash_mask_attention(self):
        q_input = paddle.randn([self.q_len, self.num_head * self.head_dim], dtype="bfloat16")
        k_input = paddle.randn([self.q_len + self.k_len, self.num_kv_head, self.head_dim], dtype="bfloat16")
        v_input = paddle.randn(k_input.shape, dtype="bfloat16")

        mask_start = paddle.zeros([self.q_len], dtype="int32")
        mask_end = paddle.zeros([self.q_len], dtype="int32") + self.q_len + self.k_len
        mask = paddle.stack([mask_start, mask_end], axis=-1).reshape([-1])

        naive_attn_out = self.naive_attn(q_input, k_input, v_input, mask)

        paddle_attn_out = paddle.empty(q_input.shape, dtype="bfloat16")
        self.paddle_flash_attn_mask(q_input, k_input, v_input, paddle_attn_out, mask)

        max_diff = (paddle_attn_out - naive_attn_out).abs().max().item()
        self.assertLessEqual(max_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
