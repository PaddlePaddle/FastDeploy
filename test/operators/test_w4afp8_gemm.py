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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import w4afp8_gemm, w4afp8_gemm_weight_convert


def w4afp8_gemm_naive(input_bf16, weight_quant, tokens, weight_dequant_scale, BATCH, N):
    all_tokens = int(tokens.sum())
    out = paddle.zeros([all_tokens, N], dtype="bfloat16")
    pre_fix_token = 0
    for i in range(BATCH):
        input = input_bf16[pre_fix_token : pre_fix_token + tokens[i], :]
        weight = (weight_quant[i] - 7.0) * weight_dequant_scale[i]
        out_i = paddle.matmul(input, weight.astype("bfloat16"), transpose_y=True)
        out[pre_fix_token : pre_fix_token + tokens[i], :] = out_i
        pre_fix_token += tokens[i]
    return out


def peruate_scale(weight_scale):
    weight_scale = weight_scale.reshape([BATCH, N])
    temp = paddle.zeros([16])
    for b in range(BATCH):
        for n in range(0, N, 16):
            temp[:] = weight_scale[b, n : n + 16]
            for j in range(0, 16, 2):
                weight_scale[b, n + j] = temp[j // 2]
                weight_scale[b, n + j + 1] = temp[j // 2 + 8]
    return weight_scale


paddle.seed(0)
tokens_per_group = 32
N = 8192
K = 3584
BATCH = 8
TokenPadding = 0

tokens = [tokens_per_group] * BATCH
tokens_perfix_sum = np.cumsum(tokens)
tokens_perfix_sum = np.insert(tokens_perfix_sum, 0, 0)

tokens = paddle.to_tensor(tokens, dtype="int32")
tokens_perfix_sum = paddle.to_tensor(tokens_perfix_sum, dtype="int32")

all_tokens = int(tokens.sum())

input_fp8 = paddle.randn([all_tokens, K], dtype="bfloat16").astype(paddle.float8_e4m3fn)
input_bf16 = input_fp8.astype("bfloat16")
weight = paddle.randn([BATCH, N, K], dtype="bfloat16") / 10

weight_scale = 7 / weight.abs().max(axis=-1).reshape([BATCH, N, 1])
weight_quant = (weight * weight_scale).astype("int") + 7
weight_quant = paddle.clip(weight_quant, 0, 14)
weight_quant = weight_quant.astype("bfloat16")
weight_dequant_scale = 1 / weight_scale.astype("float32")
input_row_sum = input_bf16.sum(axis=1) * -7 / 512
max_tokens = int(tokens.max())

out_naive = w4afp8_gemm_naive(input_bf16, weight_quant, tokens, weight_dequant_scale, BATCH, N)
weight_dequant_scale = paddle.to_tensor(peruate_scale(weight_dequant_scale) * 512)

weight_int4 = w4afp8_gemm_weight_convert(weight_quant.astype("uint8").cpu())

if TokenPadding == 0:
    out_cuda = w4afp8_gemm(
        input_fp8,
        weight_int4.cuda(),
        tokens_perfix_sum,
        input_row_sum.astype("float32"),
        weight_dequant_scale.astype("float32"),
        int(TokenPadding),
        max_tokens,
        True,
    )
else:
    out_cuda = w4afp8_gemm(
        input_fp8,
        weight_int4.cuda(),
        tokens,
        input_row_sum.astype("float32"),
        weight_dequant_scale.astype("float32"),
        int(TokenPadding),
        max_tokens,
        True,
    )

gap = (out_cuda - out_naive).abs()
assert float(gap.mean()) < 0.07
