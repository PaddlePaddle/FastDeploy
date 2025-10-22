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

import paddle

try:
    from fastdeploy.model_executor.ops.gpu import (
        fused_block_mean_and_rope,
        get_cur_cu_seq_len_k,
        moba_encoder_attn,
        moba_mlp_einsum,
        moba_qk_gemm,
        moba_qk_sort_encoder,
    )
except:
    moba_attention = None
    get_cur_cu_seq_len_k = None
import os
import unittest

import numpy as np

from fastdeploy import LLM, SamplingParams


def naive_attn(q_input, k_input, v_input, mask):
    gqa_group_size = q_input.shape[2] // k_input.shape[2]

    q_cur = q_input.transpose([0, 2, 1, 3])
    k_cur = k_input.transpose([0, 2, 1, 3])
    v_cur = v_input.transpose([0, 2, 1, 3])
    out = paddle.zeros(q_cur.shape, dtype=q_input.dtype)

    for bsz in range(0, q_cur.shape[0]):
        for hi in range(0, q_cur.shape[1]):
            qk = paddle.matmul(q_cur[bsz, hi], k_cur[bsz, hi // gqa_group_size].T) * (1.0 / np.sqrt(q_cur.shape[3]))
            qk += mask
            qk_max = qk.max(axis=-1).unsqueeze(-1)
            qk -= qk_max
            qk = qk.exp()

            exp_sum = qk.sum(axis=-1).unsqueeze(-1)
            exp_sum_inv = 1.0 / exp_sum

            out[bsz, hi] = (paddle.matmul(qk, v_cur[bsz, hi // gqa_group_size]) * exp_sum_inv).astype(q_input.dtype)
    return out


class TestPlasAttention(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.seq_len = int(8 * 1024)
        self.num_heads = int(8)
        self.num_kv_heads = int(1)
        self.head_dim = int(128)
        self.max_num_seqs = 1
        self.plas_max_seq_length = int(128 * 1024)
        self.plas_block_size = int(128)
        self.plas_encoder_top_k_left = 2
        self.plas_encoder_top_k_right = 3
        self.plas_use_encoder_seq_limit = int(4 * 1024)
        self.cache_k_block_means = paddle.zeros(
            [
                self.max_num_seqs,
                self.plas_max_seq_length // self.plas_block_size,
                self.num_kv_heads,
                self.head_dim,
            ],
            dtype="bfloat16",
        )
        self.attn_block_m = 128
        self.tokens = self.seq_len * self.max_num_seqs
        self.q_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.num_heads, self.head_dim],
            dtype="bfloat16",
        )
        self.k_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.num_kv_heads, self.head_dim],
            dtype="bfloat16",
        )
        self.v_input = paddle.zeros(
            [self.tokens + self.attn_block_m, self.num_kv_heads, self.head_dim],
            dtype="bfloat16",
        )
        self.rotary_embs = paddle.ones([2, self.seq_len, self.head_dim // 2], dtype="float32")

        self.attn_gate_weight = paddle.randn(
            [self.num_kv_heads, self.plas_block_size, self.head_dim], dtype="bfloat16"
        )

        self.gqa_group_size = self.num_heads // self.num_kv_heads

        self.num_blocks = (self.seq_len + self.plas_block_size - 1) // self.plas_block_size

        self.sparse_step = 4

    def compare_split_qkv_rope(self, qkv_out):
        assert (qkv_out[:, 0 : self.num_heads, :] - self.q_input[0 : self.tokens]).abs().max() < 1e-3
        assert (
            qkv_out[:, self.num_heads : self.num_heads + self.num_kv_heads, :] - self.k_input[0 : self.tokens]
        ).abs().max() < 1e-3
        assert (qkv_out[:, self.num_heads + self.num_kv_heads :, :] - self.v_input[0 : self.tokens]).abs().max() < 1e-3

        for i in range(self.max_num_seqs):
            k_padding = paddle.zeros(
                [
                    (self.seq_len + self.plas_block_size - 1) // self.plas_block_size * self.plas_block_size,
                    self.num_kv_heads,
                    self.head_dim,
                ],
                dtype="bfloat16",
            )
            k_padding[0 : self.seq_len] = self.k_input[i * self.seq_len : (i + 1) * self.seq_len]
            real_k_block_means = k_padding.reshape([-1, self.plas_block_size, self.num_kv_heads, self.head_dim])
            real_k_block_means = real_k_block_means.mean(axis=1)
            compute_k_block_means = self.cache_k_block_means[i, 0 : real_k_block_means.shape[0]]
            assert (compute_k_block_means - real_k_block_means).abs().max() < 0.003

        print("[consistency]plas attention: split_qkv_rope matches.")

    def compare_mlp_einsum(self, k_gate_weight):
        for i in range(self.max_num_seqs):
            k_padding = paddle.zeros(
                [
                    (self.seq_len + self.plas_block_size - 1) // self.plas_block_size * self.plas_block_size,
                    self.num_kv_heads,
                    self.head_dim,
                ],
                dtype="bfloat16",
            )
            k_padding[0 : self.seq_len] = self.k_input[i * self.seq_len : (i + 1) * self.seq_len]
            k_padding = k_padding.reshape([-1, self.plas_block_size, self.num_kv_heads, self.head_dim])
            real_result = paddle.einsum("nbhd,hbd->nhd", k_padding, self.attn_gate_weight)
            compute_result = k_gate_weight[i][0 : real_result.shape[0]]

            assert (real_result - compute_result).abs().max() < 0.5

        print("[consistency]plas attention: MLP einsum matches.")

    def compare_qk_gemm(self, qk_gate_weight):
        for i in range(self.max_num_seqs):
            q_input = self.q_input[i * self.seq_len : (i + 1) * self.seq_len]
            k_input_mean = self.cache_k_block_means[i][0 : self.num_blocks]

            qk_gemm_out = paddle.zeros(
                [
                    self.seq_len,
                    self.num_heads,
                    self.num_blocks,
                ],
                dtype="bfloat16",
            )

            for j in range(self.num_heads):
                qk_gemm_out[:, j, :] = paddle.matmul(
                    q_input[:, j, :], k_input_mean[:, j // self.gqa_group_size, :], transpose_y=True
                )

            conpute_result = qk_gate_weight[i * self.seq_len : (i + 1) * self.seq_len, :, 0 : self.num_blocks]
            assert (qk_gemm_out - conpute_result).abs().max() < 1e-4

        print("[consistency]plas attention: qk_gemm matches.")

    def compare_qk_gate_topk(self, qk_gate_topk_idx):
        limit_topk = self.plas_use_encoder_seq_limit // self.plas_block_size
        for i in range(self.max_num_seqs):
            qk_gate_topk_idx_batch = qk_gate_topk_idx[i * self.num_blocks : (i + 1) * self.num_blocks]
            qk_gate_topk_idx_batch_no_sparse = qk_gate_topk_idx_batch[0 : limit_topk - 1]

            assert (
                qk_gate_topk_idx_batch_no_sparse
                - paddle.ones(qk_gate_topk_idx_batch_no_sparse.shape, qk_gate_topk_idx_batch_no_sparse.dtype)
            ).abs().max() < 1e-6

            for j in range(limit_topk, self.num_blocks):
                qk_gate_topk_idx_batch_sparse = qk_gate_topk_idx_batch[j, :, 1 : (j + 1) // self.sparse_step]

                assert (
                    qk_gate_topk_idx_batch_sparse
                    - paddle.ones(qk_gate_topk_idx_batch_sparse.shape, qk_gate_topk_idx_batch_sparse.dtype)
                    * self.sparse_step
                ).abs().max() < 1e-6
        print("[consistency]plas attention: qk_gate_topk matches.")

    def compare_attn(self, attn_out, qk_gate_topk_idx):
        x = (
            paddle.tensor.triu(paddle.ones([self.plas_block_size, self.plas_block_size], dtype="bfloat16"), 1)
            * -1000000
        )
        limit_topk = self.plas_use_encoder_seq_limit // self.plas_block_size
        for i in range(self.max_num_seqs):
            q_input = self.q_input[i * self.seq_len : (i + 1) * self.seq_len].unsqueeze(axis=0)
            k_input = self.k_input[i * self.seq_len : (i + 1) * self.seq_len].unsqueeze(axis=0)
            v_input = self.v_input[i * self.seq_len : (i + 1) * self.seq_len].unsqueeze(axis=0)
            mask = paddle.tensor.triu(paddle.ones([self.seq_len, self.seq_len], dtype="bfloat16"), 1) * -1000000
            mask[self.plas_use_encoder_seq_limit - self.plas_block_size :] = -1000000
            for i in range(limit_topk - 1, self.num_blocks):
                n_block = i
                mask[
                    i * self.plas_block_size : i * self.plas_block_size + self.plas_block_size,
                    n_block * self.plas_block_size : n_block * self.plas_block_size + self.plas_block_size,
                ] = x
                idx = 0
                n_block -= int(qk_gate_topk_idx[i, 0, idx])
                idx += 1
                while n_block >= 0:
                    mask[
                        i * self.plas_block_size : i * self.plas_block_size + self.plas_block_size,
                        n_block * self.plas_block_size : n_block * self.plas_block_size + self.plas_block_size,
                    ] = 0
                    n_block -= int(qk_gate_topk_idx[i, 0, idx])
                    idx += 1
            naive_attn_out = naive_attn(q_input, k_input, v_input, mask).squeeze(axis=0).transpose([1, 0, 2])
            assert (attn_out - naive_attn_out).abs().max() < 0.016

    def test_plas_attention(self):
        qkv_out = paddle.randn([self.tokens, self.num_heads + 2 * self.num_kv_heads, self.head_dim], dtype="bfloat16")

        seq_len_encoder = paddle.to_tensor([self.seq_len] * self.max_num_seqs, dtype="int32")
        seq_len_decoder = paddle.to_tensor([0] * self.max_num_seqs, dtype="int32")
        cu_seq_q = paddle.arange(self.max_num_seqs + 1).astype("int32") * self.seq_len
        cu_seq_k = paddle.arange(self.max_num_seqs + 1).astype("int32") * self.seq_len
        seq_lens_this_time = paddle.to_tensor([self.seq_len] * self.max_num_seqs, dtype="int32")

        cu_seq_q_pack, cu_seqlens_k, q_pack_tokens = get_cur_cu_seq_len_k(
            seq_len_encoder,
            seq_len_decoder,
            seq_lens_this_time,
            int(self.attn_block_m),
        )

        fused_block_mean_and_rope(
            qkv_out,
            self.cache_k_block_means,
            self.q_input,
            self.k_input,
            self.v_input,
            self.rotary_embs,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            None,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.plas_max_seq_length,
            self.seq_len,
            self.seq_len,
            "none",
        )

        self.compare_split_qkv_rope(qkv_out)

        k_gate_weight = moba_mlp_einsum(
            self.k_input,
            self.attn_gate_weight,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_k,
            self.seq_len,
            self.num_kv_heads,
        )

        self.compare_mlp_einsum(k_gate_weight)

        qk_gate_weight = moba_qk_gemm(
            self.q_input,
            self.cache_k_block_means,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            self.seq_len,
            self.seq_len,
            self.num_heads,
            self.num_kv_heads,
            False,
            self.max_num_seqs,
        )

        self.compare_qk_gemm(qk_gate_weight)

        for i in range(0, self.num_blocks, self.sparse_step):
            qk_gate_weight[:, :, i] = 100

        qk_gate_topk_idx = moba_qk_sort_encoder(
            qk_gate_weight,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            cu_seq_q_pack,
            q_pack_tokens,
            self.seq_len,
            self.seq_len,
            self.num_heads,
            self.num_kv_heads,
            self.plas_encoder_top_k_left,
            self.plas_encoder_top_k_right,
            self.plas_use_encoder_seq_limit,
        )

        self.compare_qk_gate_topk(qk_gate_topk_idx)

        attn_out = paddle.zeros([self.tokens, self.num_heads, self.head_dim], dtype="bfloat16")

        moba_encoder_attn(
            self.q_input,
            self.k_input,
            self.v_input,
            qk_gate_topk_idx,
            cu_seq_q,
            cu_seq_k,
            cu_seq_q_pack,
            seq_len_encoder,
            seq_len_decoder,
            attn_out,
            self.seq_len,
            self.seq_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.plas_max_seq_length,
        )

        self.compare_attn(attn_out, qk_gate_topk_idx)

    def test_server(self):
        if get_cur_cu_seq_len_k is None:
            return
        os.environ["FD_ATTENTION_BACKEND"] = "PLAS_ATTN"
        base_path = os.getenv("MODEL_PATH")
        if base_path:
            model_path = os.path.join(base_path, "./ernie-4_5-21b-a3b-bf16-paddle")
        else:
            model_path = "./ernie-4_5-21b-a3b-bf16-paddle"

        plas_attention_config = {
            "plas_encoder_top_k_left": 50,
            "plas_encoder_top_k_right": 60,
            "plas_decoder_top_k_left": 100,
            "plas_decoder_top_k_right": 120,
        }

        graph_optimization_config = {"use_cudagraph": False}
        # 加载模型
        llm = LLM(
            model=model_path,
            tensor_parallel_size=2,
            max_model_len=131072,
            engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT")),
            cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT")),
            max_num_seqs=32,
            quantization="wint4",
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
            plas_attention_config=plas_attention_config,
            graph_optimization_config=graph_optimization_config,
        )

        prompts = ["Hello world!"]
        sampling_params = SamplingParams(temperature=1.0, top_p=0.0, max_tokens=32)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        for output in outputs:
            print(output.outputs.text)


if __name__ == "__main__":
    if paddle.is_compiled_with_cuda():
        unittest.main()
