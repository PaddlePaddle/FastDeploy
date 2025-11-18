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

from fastdeploy.model_executor.ops.gpu import flash_attention_mask


class TestFlashMaskAttention(unittest.TestCase):
    def setUp(self):
        self.bsz = 1
        self.num_head = 8
        self.num_kv_head = 1
        self.q_seq_len = 1024
        self.k_seq_len = 1024
        self.head_dim = 128
        np.random.seed(self.q_seq_len)

    def naive_attn(self, q_input, k_input, v_input, mask):
        gqa_group_size = q_input.shape[2] // k_input.shape[2]

        q_cur = q_input.transpose([0, 2, 1, 3])
        k_cur = k_input.transpose([0, 2, 1, 3])
        v_cur = v_input.transpose([0, 2, 1, 3])
        out = np.zeros(q_cur.shape, dtype=q_input.dtype)

        for bsz in range(0, q_cur.shape[0]):
            for hi in range(0, q_cur.shape[1]):
                qk = np.matmul(q_cur[bsz, hi], k_cur[bsz, hi // gqa_group_size].T) * (1.0 / np.sqrt(q_cur.shape[3]))
                for i in range(0, qk.shape[0]):
                    qk[i, mask[i] :] = -1000000

                qk_max = np.expand_dims(qk.max(axis=-1), -1)
                qk -= qk_max
                qk = np.exp(qk)

                exp_sum = np.expand_dims(qk.sum(axis=-1), -1)
                exp_sum_inv = 1.0 / exp_sum

                out[bsz, hi] = (np.matmul(qk, v_cur[bsz, hi // gqa_group_size]) * exp_sum_inv).astype(q_input.dtype)
        return out

    def paddle_flash_attn_mask(self, q_input, k_input, v_input, attn_out, mask):
        bsz = q_input.shape[0]
        cu_seq_q = paddle.arange(bsz + 1) * q_input.shape[1]
        cu_seq_k = paddle.arange(bsz + 1) * k_input.shape[1]
        cu_seq_q = cu_seq_q.astype("int32")
        cu_seq_k = cu_seq_k.astype("int32")
        seq_len_encoder = paddle.ones(bsz) * q_input.shape[1]
        seq_len_encoder = seq_len_encoder.astype("int32")
        q_input = paddle.to_tensor(q_input).astype("bfloat16").reshape([-1, q_input.shape[2], q_input.shape[3]])
        k_input = paddle.to_tensor(k_input).astype("bfloat16").reshape([-1, k_input.shape[2], k_input.shape[3]])
        v_input = paddle.to_tensor(v_input).astype("bfloat16").reshape([-1, v_input.shape[2], v_input.shape[3]])
        v_input_pad = paddle.zeros([v_input.shape[0] + 128, v_input.shape[1], v_input.shape[2]]).astype("bfloat16")
        v_input_pad[0 : v_input.shape[0]] = v_input
        mask = paddle.to_tensor(mask).astype("int32")

        flash_attention_mask(
            q_input,
            k_input,
            v_input_pad,
            cu_seq_q,
            cu_seq_k,
            seq_len_encoder,
            attn_out,
            mask,
            int(q_input.shape[1]),
            int(k_input.shape[1]),
            int(q_input.shape[2]),
            int(k_input.shape[0]),
            int(q_input.shape[0]),
            int(k_input.shape[0]),
        )

    def test_flash_attention_mask(self):
        q_input = np.random.normal(0, 0.5, size=(self.bsz, self.q_seq_len, self.num_head, self.head_dim))
        k_input = np.random.normal(
            0, 0.5, size=(self.bsz, self.q_seq_len + self.k_seq_len, self.num_kv_head, self.head_dim)
        )
        v_input = np.random.normal(
            0, 0.5, size=(self.bsz, self.q_seq_len + self.k_seq_len, self.num_kv_head, self.head_dim)
        )

        random_len = np.random.randint(self.q_seq_len // 2, size=2)
        text_len = random_len[0]
        image_len = random_len[1]

        mask = np.array([i + 1 for i in range(0, self.q_seq_len)]) + self.k_seq_len
        mask[text_len : text_len + image_len] = text_len + image_len + self.k_seq_len

        naive_attn_out = self.naive_attn(q_input, k_input, v_input, mask)
        paddle_attn_out = paddle.zeros(naive_attn_out.shape, dtype="bfloat16")
        self.paddle_flash_attn_mask(q_input, k_input, v_input, paddle_attn_out, mask)

        max_diff = float((paddle_attn_out.reshape([-1]) - paddle.to_tensor(naive_attn_out).reshape([-1])).max())
        self.assertLessEqual(max_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
