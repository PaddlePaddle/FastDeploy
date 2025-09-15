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
    wfp8afp8_gemm_sparse_idx_convert,
    wfp8afp8_sparse_gemm,
)


def wfp8afp8_gemm_naive(input_bf16, weight_quant, tokens, weight_scale, BATCH, N):
    weight = weight_quant.astype("bfloat16") / weight_scale
    input_bf16 = input_bf16.astype("bfloat16")
    all_tokens = int(tokens.sum())
    out = paddle.zeros([all_tokens, N], dtype="bfloat16")
    pre_fix_token = 0
    for i in range(BATCH):
        input = input_bf16[pre_fix_token : pre_fix_token + tokens[i], :]
        out_i = paddle.matmul(input, weight[i], transpose_y=True)
        out[pre_fix_token : pre_fix_token + tokens[i], :] = out_i
        pre_fix_token += tokens[i]
    return out


def peruate_scale(weight_scale, N):
    BATCH = weight_scale.shape[0]
    weight_scale = weight_scale.reshape([BATCH, N])
    temp = paddle.zeros([16])
    for b in range(BATCH):
        for n in range(0, N, 16):
            temp[:] = weight_scale[b, n : n + 16]
            for j in range(0, 16, 2):
                weight_scale[b, n + j] = temp[j // 2]
                weight_scale[b, n + j + 1] = temp[j // 2 + 8]
    return weight_scale


def sparse(weight, sparse_idx):
    pack_weight = np.zeros([weight.shape[0], weight.shape[1], weight.shape[2] // 2], dtype=weight.dtype)

    idx_select = [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 3, 1, 2],
        [1, 2, 0, 3],
        [1, 3, 0, 2],
        [2, 3, 0, 1],
    ]
    for b in range(weight.shape[0]):
        for i in range(weight.shape[1]):
            for j in range(0, weight.shape[2], 4):
                idx = sparse_idx[b, i, j // 4]
                idx1 = idx_select[idx][0]
                idx2 = idx_select[idx][1]
                idx3 = idx_select[idx][2]
                idx4 = idx_select[idx][3]

                weight[b, i, j + idx1] = 0
                weight[b, i, j + idx2] = 0

                pack_weight[b, i, j // 4 * 2] = weight[b, i, j + idx3]
                pack_weight[b, i, j // 4 * 2 + 1] = weight[b, i, j + idx4]
    return weight, pack_weight


def convert(weight, sparse_idx, K):
    BATCH = weight.shape[0]
    temp = np.zeros(weight.shape, dtype=weight.dtype)

    for i in range(0, weight.shape[1], 128):
        for j in range(0, 128):
            dst_idx = j // 2 + (j % 2) * 64
            temp[:, j + i, :] = weight[:, i + dst_idx, :]

    temp_trans = np.zeros([BATCH, weight.shape[1] // 128, K // 128, 128, 64], dtype=weight.dtype)
    temp_E = np.zeros([BATCH, weight.shape[1] // 128, K // 128, 128, 32], dtype=sparse_idx.dtype)

    for b in range(BATCH):
        for i in range(weight.shape[1] // 128):
            for j in range(K // 128):
                temp_trans[b, i, j] = temp[b, i * 128 : i * 128 + 128, j * 64 : j * 64 + 64]
                temp_E[b, i, j] = sparse_idx[b, i * 128 : i * 128 + 128, j * 32 : j * 32 + 32]

    return temp_trans, temp_E


class TestWFp8Afp8SparseGemm(unittest.TestCase):
    def test_wfp8afp8_sparse_gemm(self):
        paddle.seed(0)
        tokens_per_group = 10
        N = 128
        K = 128
        BATCH = 1
        TokenPadding = 0

        tokens = [tokens_per_group] * BATCH
        tokens_prefix_sum = np.cumsum(tokens)
        tokens_prefix_sum = np.insert(tokens_prefix_sum, 0, 0)

        tokens = paddle.to_tensor(tokens, dtype="int32")
        tokens_prefix_sum = paddle.to_tensor(tokens_prefix_sum, dtype="int32")

        all_tokens = int(tokens.sum())

        input_fp8 = paddle.randn([all_tokens, K], dtype="bfloat16").astype(paddle.float8_e4m3fn)

        weight = paddle.randn([BATCH, N, K], dtype="bfloat16")

        weight_scale = 40 / weight.abs().max(axis=-1).reshape([BATCH, N, 1])

        weight_quant = (weight * weight_scale).astype(paddle.float8_e4m3fn).astype("bfloat16")

        weight_quant = weight_quant.numpy()

        sparse_idx = np.random.randint(0, high=6, size=(BATCH, N, K // 4))

        weight_quant, pack_weight = sparse(weight_quant, sparse_idx)

        weight_quant = paddle.to_tensor(weight_quant)
        out_naive = wfp8afp8_gemm_naive(input_fp8, weight_quant, tokens, weight_scale, BATCH, N)

        pack_weight, convert_sparse_idx = convert(pack_weight, sparse_idx, K)

        pack_weight = paddle.to_tensor(pack_weight).astype(paddle.float8_e4m3fn)
        convert_sparse_idx = paddle.to_tensor(convert_sparse_idx).astype("uint8").cpu()
        convert_sparse_idx = wfp8afp8_gemm_sparse_idx_convert(convert_sparse_idx, int(BATCH), int(N), int(K)).cuda()

        weight_scale = paddle.to_tensor(peruate_scale(weight_scale, N)).astype("float32")

        out_pd = paddle.zeros([all_tokens, N], dtype="bfloat16")

        wfp8afp8_sparse_gemm(
            input_fp8,
            convert_sparse_idx,
            pack_weight.reshape([BATCH, N, K // 2]),
            tokens_prefix_sum if TokenPadding == 0 else tokens,
            1 / weight_scale,
            out_pd,
            int(TokenPadding),
            int(tokens_per_group),
            True,
        )

        print((out_pd - out_naive).abs().max())


if __name__ == "__main__":
    unittest.main()
