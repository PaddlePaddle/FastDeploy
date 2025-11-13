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

from fastdeploy.model_executor.ops.gpu import (
    dynamic_quant_cache_decoder_attention,
    dynamic_quant_cache_write_decoder,
    dynamic_quant_cache_write_encoder,
    dynamic_quant_get_kv_from_cache,
    get_qk_tokens_num,
    split_qkv_and_rope,
)


class TestC2Attnention(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.batch_size = 1
        self.seq_len = 64
        self.head_dim = int(128)
        self.num_heads = int(8)
        self.kv_num_heads = int(1)
        self.attn_block_m = 128
        self.max_seq_len = int(8192)
        self.block_size = 64
        self.c16_remain_seq_len = 128
        self.cache_quant_type_str = "dynamic_int2_zp"
        self.tokens = self.seq_len * self.batch_size
        self.seq_lens_encoder = paddle.to_tensor([self.seq_len] * self.batch_size).astype("int32")
        self.seq_lens_decoder = paddle.to_tensor([0] * self.batch_size).astype("int32")
        self.seq_lens_this_time = paddle.to_tensor([self.seq_len] * self.batch_size).astype("int32")
        self.prompt_lens = paddle.to_tensor([self.seq_len] * self.batch_size).astype("int64")
        self.step_idx = paddle.to_tensor([0] * self.batch_size).astype("int64")
        self.qkv_out = paddle.randn([self.tokens, self.num_heads + self.kv_num_heads * 2, self.head_dim]).astype(
            "bfloat16"
        )
        self.q_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.num_heads, self.head_dim],
            dtype="float16",
        )
        self.k_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.kv_num_heads, self.head_dim],
            dtype="float16",
        )
        self.v_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.kv_num_heads, self.head_dim],
            dtype="float16",
        )
        self.rotary_embs = paddle.ones([2, self.seq_len, self.head_dim // 2], dtype="float32")

        cu_seqlens_k, qk_token_num = get_qk_tokens_num(
            self.seq_lens_encoder, self.seq_lens_this_time, self.seq_lens_decoder
        )
        self.cu_seqlens_k = cu_seqlens_k
        self.cu_seqlens_q = paddle.arange(self.batch_size + 1).astype("int32") * self.seq_len
        self.qk_token_num = qk_token_num

        self.cachek_c2 = paddle.zeros(
            [self.max_seq_len // self.block_size, self.kv_num_heads, 24, self.block_size]
        ).astype("uint8")
        self.cachev_c2 = paddle.zeros(
            [self.max_seq_len // self.block_size, self.kv_num_heads, 24, self.block_size]
        ).astype("uint8")
        self.cachek_c16 = paddle.zeros(
            [
                self.batch_size,
                self.c16_remain_seq_len + self.block_size,
                self.kv_num_heads,
                self.head_dim,
            ],
            dtype="float16",
        )
        self.cachev_c16 = paddle.zeros(
            [
                self.batch_size,
                self.c16_remain_seq_len + self.block_size,
                self.kv_num_heads,
                self.head_dim,
            ],
            dtype="float16",
        )
        self.block_tables = (
            paddle.arange(self.max_seq_len // self.block_size).astype("int32").reshape([self.batch_size, -1])
        )

    def dynamic_quant_cache_decoder_attention_np(self, q_input):

        out = paddle.zeros([self.batch_size, self.num_heads, self.head_dim], dtype=q_input.dtype)
        gqa_group_size = self.num_heads // self.kv_num_heads

        q_input = q_input.reshape([self.batch_size, self.num_heads, self.head_dim])
        for i in range(self.num_heads):
            qk = paddle.matmul(q_input[0, i], self.cachek_c16[0, :, i // gqa_group_size].T)
            qk = qk / np.sqrt(self.head_dim)
            qk[self.seq_len + 1 :] = -1000000
            max_v = qk.max()
            qk -= max_v
            qk = paddle.exp(qk)
            qk = qk / qk.sum()
            value = paddle.matmul(qk, self.cachev_c16[0, :, i // gqa_group_size])

            out[0, i] = value
        return out

    def dynamic_quant_cache_write_decoder_np(self, qkv_output):
        return qkv_output[0 : self.batch_size, 0 : self.num_heads]

    def dynamic_quant_cache_write_encoder_np(self, k_input, v_input):
        cachek_c16 = paddle.zeros_like(self.cachek_c16)
        cachev_c16 = paddle.zeros_like(self.cachev_c16)

        for i in range(self.batch_size):
            cachek_c16[i][0 : self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i]] = k_input[
                self.cu_seqlens_k[i] : self.cu_seqlens_k[i + 1]
            ]
            cachev_c16[i][0 : self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i]] = v_input[
                self.cu_seqlens_k[i] : self.cu_seqlens_k[i + 1]
            ]

        return cachek_c16, cachev_c16

    def dynamic_quant_get_kv_from_cache_np(self, cachek_c16, cachev_c16):
        k_input = paddle.zeros_like(self.k_input)
        v_input = paddle.zeros_like(self.v_input)

        for i in range(self.batch_size):
            k_input[self.cu_seqlens_k[i] : self.cu_seqlens_k[i + 1]] = cachek_c16[i][
                0 : self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i]
            ]
            v_input[self.cu_seqlens_k[i] : self.cu_seqlens_k[i + 1]] = cachev_c16[i][
                0 : self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i]
            ]

        return k_input, v_input

    def get_qk_tokens_num_np(self):
        cu_seqlens_k = [0] * (self.batch_size + 1)
        qk_tokens = [0] * 4

        total_tokens = 0

        for i in range(self.batch_size):
            cache_len = int(self.seq_lens_decoder[i])
            q_len = int(self.seq_lens_encoder[i])
            tokens = int(self.seq_lens_this_time[i])

            qk_tokens[1] = max(qk_tokens[1], cache_len)
            if q_len == 0:
                cache_len = tokens
            total_tokens += cache_len + q_len
            cu_seqlens_k[i + 1] = total_tokens
            qk_tokens[0] = max(qk_tokens[0], q_len)

            qk_tokens[2] += tokens
            qk_tokens[3] += cache_len + q_len

        return cu_seqlens_k, qk_tokens

    def split_qkv_and_rope_np(self):
        q_input = paddle.zeros_like(self.q_input)
        k_input = paddle.zeros_like(self.k_input)
        v_input = paddle.zeros_like(self.v_input)

        q_input[0 : self.tokens] = self.qkv_out[0 : self.tokens, 0 : self.num_heads]
        k_input[0 : self.tokens] = self.qkv_out[0 : self.tokens, self.num_heads : self.num_heads + self.kv_num_heads]
        v_input[0 : self.tokens] = self.qkv_out[0 : self.tokens, self.num_heads + self.kv_num_heads :]

        return q_input, k_input, v_input

    def test_c2_attn(self):
        cu_seqlens_k, qk_token_num = get_qk_tokens_num(
            self.seq_lens_encoder, self.seq_lens_this_time, self.seq_lens_decoder
        )
        cu_seqlens_k_np, qk_token_num_np = self.get_qk_tokens_num_np()

        assert np.allclose(cu_seqlens_k, cu_seqlens_k_np)
        assert np.allclose(qk_token_num, qk_token_num_np)

        split_qkv_and_rope(
            self.qkv_out,
            self.q_input,
            self.k_input,
            self.v_input,
            self.rotary_embs,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            None,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.qk_token_num[0],
            self.max_seq_len,
            self.cache_quant_type_str,
        )

        q_input_np, k_input_np, v_input_np = self.split_qkv_and_rope_np()
        assert np.allclose(self.q_input, q_input_np)
        assert np.allclose(self.k_input, k_input_np)
        assert np.allclose(self.v_input, v_input_np)

        dynamic_quant_cache_write_encoder(
            self.k_input,
            self.v_input,
            self.cachek_c2,
            self.cachev_c2,
            self.cachek_c16,
            self.cachev_c16,
            self.cu_seqlens_k,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.block_tables,
            self.prompt_lens,
            self.c16_remain_seq_len,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            int(self.seq_len),
            self.cache_quant_type_str,
        )

        cachek_c16_np, cachev_c16_np = self.dynamic_quant_cache_write_encoder_np(self.k_input, self.v_input)

        assert np.allclose(self.cachek_c16, cachek_c16_np)
        assert np.allclose(self.cachev_c16, cachev_c16_np)

        dynamic_quant_get_kv_from_cache(
            self.k_input,
            self.v_input,
            self.cachek_c2,
            self.cachev_c2,
            self.cachek_c16,
            self.cachev_c16,
            self.cu_seqlens_k,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.block_tables,
            self.prompt_lens,
            self.c16_remain_seq_len,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            qk_token_num[0] + qk_token_num[1],
            self.cache_quant_type_str,
        )

        k_input_np, v_input_np = self.dynamic_quant_get_kv_from_cache_np(self.cachek_c16, self.cachev_c16)
        assert np.allclose(self.k_input, k_input_np)
        assert np.allclose(self.v_input, v_input_np)

        qkv_out_decoder = paddle.randn(
            [self.batch_size, self.num_heads + 2 * self.kv_num_heads, self.head_dim], dtype="bfloat16"
        )
        q_input = dynamic_quant_cache_write_decoder(
            qkv_out_decoder,
            self.rotary_embs,
            self.cachek_c2,
            self.cachev_c2,
            self.cachek_c16,
            self.cachev_c16,
            self.cu_seqlens_q,
            self.seq_lens_decoder,
            self.seq_lens_encoder,
            self.block_tables,
            self.step_idx,
            None,
            self.c16_remain_seq_len,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.max_seq_len,
            self.cache_quant_type_str,
        )[0]

        q_input_np = self.dynamic_quant_cache_write_decoder_np(qkv_out_decoder).astype("float16")

        assert np.allclose(q_input, q_input_np)

        out = paddle.zeros([qkv_out_decoder.shape[0], self.num_heads * self.head_dim], dtype=qkv_out_decoder.dtype)
        dynamic_quant_cache_decoder_attention(
            q_input,
            self.cachek_c2,
            self.cachev_c2,
            self.cachek_c16,
            self.cachev_c16,
            out,
            self.cu_seqlens_q,
            self.seq_lens_decoder,
            self.seq_lens_encoder,
            self.block_tables,
            self.c16_remain_seq_len,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            qk_token_num[0],
            self.max_seq_len,
            self.cache_quant_type_str,
        )

        out_np = self.dynamic_quant_cache_decoder_attention_np(q_input).reshape(out.shape).astype("bfloat16")
        assert np.allclose(out, out_np, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    unittest.main()
