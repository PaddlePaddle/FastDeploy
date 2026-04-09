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

from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
    flash_attn_func,
)
from fastdeploy.model_executor.layers.attention.ops import (
    flash_attn_v4,
    get_attn_mask_q,
)
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
        prop = paddle.device.cuda.get_device_properties()
        self.sm_version = prop.major * 10 + prop.minor

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
        if self.sm_version < 89 or self.sm_version >= 100:
            self.skipTest("flash_mask_attention V3 requires SM89+ but less than SM100.")
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

    def causal_attention_naive(self, q_input, k_input, v_input, cu_seq_q, cu_seq_k):
        """Causal attention reference implementation for flash_attn_v4 testing."""
        bsz = cu_seq_q.shape[0] - 1
        q_token_sum, num_head, head_dim = q_input.shape
        k_token_sum, num_kv_head, _ = k_input.shape
        gqa_group_size = num_head // num_kv_head
        qk_scale = 1 / np.sqrt(head_dim)
        out = paddle.zeros([num_head, q_token_sum, head_dim], q_input.dtype)
        for bi in range(bsz):
            q = q_input[cu_seq_q[bi] : cu_seq_q[bi + 1], :, :].transpose([1, 0, 2]).astype("float32").numpy()
            k = k_input[cu_seq_k[bi] : cu_seq_k[bi + 1], :, :].transpose([1, 2, 0]).astype("float32").numpy()
            v = v_input[cu_seq_k[bi] : cu_seq_k[bi + 1], :, :].transpose([1, 0, 2]).astype("float32").numpy()
            qk = np.matmul(q, np.repeat(k, gqa_group_size, 0))
            qk *= qk_scale
            condition = np.tril(np.ones(qk.shape), q.shape[1] - k.shape[2])
            mask = np.ones(condition.shape).astype("float32") * -1000000
            qk = np.where(condition > 0, qk, mask)
            qk_max = qk.max(axis=-1, keepdims=True)
            qk -= qk_max
            qk = np.exp(qk)
            exp_sum = qk.sum(axis=-1, keepdims=True)
            exp_sum_inv = 1.0 / exp_sum
            temp_out = paddle.to_tensor(np.matmul(qk, np.repeat(v, gqa_group_size, 0)))
            out[:, cu_seq_q[bi] : cu_seq_q[bi + 1], :] = temp_out * exp_sum_inv
        return out.transpose([1, 0, 2])

    def test_flash_encoder_attn_fwd(self):
        if self.sm_version < 100:
            self.skipTest("Flash Encoder Attention V4 requires SM100+.")

        q_input = paddle.randn([self.q_len, self.num_head, self.head_dim], dtype="bfloat16")
        k_input = paddle.randn([self.q_len, self.num_kv_head, self.head_dim], dtype="bfloat16")
        v_input = paddle.randn(k_input.shape, dtype="bfloat16")

        mask = paddle.arange(self.q_len).astype("int32") + 1

        bsz = self.bsz
        cu_seq_q = paddle.arange(bsz + 1) * self.q_len
        cu_seq_k = paddle.arange(bsz + 1) * self.q_len
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")

        naive_attn_out = self.causal_attention_naive(q_input, k_input, v_input, cu_seq_q, cu_seq_k)

        paddle_attn_out = paddle.empty(q_input.shape, dtype="bfloat16")

        flash_attn_v4(
            q_input,
            k_input,
            v_input,
            cu_seq_q,
            cu_seq_k,
            paddle_attn_out,
            mask,
        )

        max_diff = (paddle_attn_out - naive_attn_out).abs().max().item()
        self.assertLessEqual(max_diff, 0.05)

    def test_fa4(
        self,
    ):
        if self.sm_version < 100:
            self.skipTest("Flash Attention V4 requires SM100+.")
        q_input = paddle.randn([self.q_len, self.num_head * self.head_dim], dtype="bfloat16")
        k_input = paddle.randn([self.q_len + self.k_len, self.num_kv_head, self.head_dim], dtype="bfloat16")
        v_input = paddle.randn(k_input.shape, dtype="bfloat16")

        mask_start = paddle.zeros([self.q_len], dtype="int32")
        mask_end = paddle.zeros([self.q_len], dtype="int32") + self.q_len + self.k_len
        mask = paddle.stack([mask_start, mask_end], axis=-1).reshape([-1])

        naive_attn_out = self.naive_attn(q_input, k_input, v_input, mask)

        bsz = self.bsz
        cu_seq_q = paddle.arange(bsz + 1) * self.q_len
        cu_seq_k = paddle.arange(bsz + 1) * (self.q_len + self.k_len)
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")

        attn_mask_q = get_attn_mask_q(
            cu_seqlens_q=cu_seq_q,
            cu_seqlens_k=cu_seq_k,
            attn_mask_kv=mask,
            kv_token_num=self.q_len + self.k_len,
        )

        paddle_attn_out = flash_attn_func(
            q_input,
            k_input,
            v_input,
            attn_mask_q=attn_mask_q,
            num_heads=self.num_head,
            kv_num_heads=self.num_kv_head,
            head_dim=self.head_dim,
            version=4,
        )[0].reshape([self.q_len, self.num_head * self.head_dim])

        max_diff = (paddle_attn_out - naive_attn_out).abs().max().item()
        self.assertLessEqual(max_diff, 0.05)

    def test_fa3_with_mask(
        self,
    ):
        if self.sm_version < 89 or self.sm_version >= 100:
            self.skipTest("Flash Attention V3 requires SM89+ but less than SM100.")
        q_input = paddle.randn([self.q_len, self.num_head * self.head_dim], dtype="bfloat16")
        k_input = paddle.randn([self.q_len + self.k_len, self.num_kv_head, self.head_dim], dtype="bfloat16")
        v_input = paddle.randn(k_input.shape, dtype="bfloat16")

        mask_start = paddle.zeros([self.q_len], dtype="int32")
        mask_end = paddle.zeros([self.q_len], dtype="int32") + self.q_len + self.k_len
        mask = paddle.stack([mask_start, mask_end], axis=-1).reshape([-1])

        naive_attn_out = self.naive_attn(q_input, k_input, v_input, mask)

        bsz = self.bsz
        cu_seq_q = paddle.arange(bsz + 1) * self.q_len
        cu_seq_k = paddle.arange(bsz + 1) * (self.q_len + self.k_len)
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")

        attn_mask_q = get_attn_mask_q(
            cu_seqlens_q=cu_seq_q,
            cu_seqlens_k=cu_seq_k,
            attn_mask_kv=mask,
            kv_token_num=self.q_len + self.k_len,
        )

        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        paddle_attn_out = flash_attn_func(
            q_input,
            k_input,
            v_input,
            attn_mask_q=attn_mask_q,
            num_heads=self.num_head,
            kv_num_heads=self.num_kv_head,
            head_dim=self.head_dim,
            version=3,
        )[0].reshape([self.q_len, self.num_head * self.head_dim])

        max_diff = (paddle_attn_out - naive_attn_out).abs().max().item()
        self.assertLessEqual(max_diff, 0.05)

    def test_fa2_with_mask(
        self,
    ):
        q_input = paddle.randn([self.q_len, self.num_head * self.head_dim], dtype="bfloat16")
        k_input = paddle.randn([self.q_len + self.k_len, self.num_kv_head, self.head_dim], dtype="bfloat16")
        v_input = paddle.randn(k_input.shape, dtype="bfloat16")

        mask_start = paddle.zeros([self.q_len], dtype="int32")
        mask_end = paddle.zeros([self.q_len], dtype="int32") + self.q_len + self.k_len
        mask = paddle.stack([mask_start, mask_end], axis=-1).reshape([-1])

        naive_attn_out = self.naive_attn(q_input, k_input, v_input, mask)

        bsz = self.bsz
        cu_seq_q = paddle.arange(bsz + 1) * self.q_len
        cu_seq_k = paddle.arange(bsz + 1) * (self.q_len + self.k_len)
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")

        attn_mask_q = get_attn_mask_q(
            cu_seqlens_q=cu_seq_q,
            cu_seqlens_k=cu_seq_k,
            attn_mask_kv=mask,
            kv_token_num=self.q_len + self.k_len,
        )

        paddle.set_flags({"FLAGS_flash_attn_version": 2})
        paddle_attn_out = flash_attn_func(
            q_input,
            k_input,
            v_input,
            attn_mask_q=attn_mask_q,
            num_heads=self.num_head,
            kv_num_heads=self.num_kv_head,
            head_dim=self.head_dim,
            version=2,
        )[0].reshape([self.q_len, self.num_head * self.head_dim])

        max_diff = (paddle_attn_out - naive_attn_out).abs().max().item()
        self.assertLessEqual(max_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
