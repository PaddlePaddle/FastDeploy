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
    w4afp8_gemm,
    w4afp8_gemm_scale_permute,
    w4afp8_gemm_weight_convert,
)


class TestW4AFP8GEMM(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.tokens_per_group = 1
        self.N = 256
        self.K = 256
        self.BATCH = 2
        self.TokenPadding = 0

        tokens = [self.tokens_per_group] * self.BATCH
        self.tokens_prefix_sum = np.cumsum(tokens)

        self.tokens = paddle.to_tensor(tokens, dtype="int64")
        self.tokens_prefix_sum = paddle.to_tensor(self.tokens_prefix_sum, dtype="int64")
        self.all_tokens = int(self.tokens.sum())

        self.input_fp8 = paddle.randn([self.all_tokens, self.K], dtype="bfloat16").astype(paddle.float8_e4m3fn)
        self.input_bf16 = self.input_fp8.astype("bfloat16")
        self.weight = paddle.randn([self.BATCH, self.N, self.K], dtype="bfloat16")

        self.weight_scale = 7 / self.weight.abs().max(axis=-1).reshape([self.BATCH, self.N, 1])
        self.weight_quant = (self.weight * self.weight_scale).astype("int")
        self.weight_quant = paddle.clip(self.weight_quant, -7, 7)
        self.weight_quant_naive = self.weight_quant.astype("float32")
        self.weight_quant = self.weight_quant.astype("bfloat16")
        self.weight_quant = paddle.where(self.weight_quant > 0, self.weight_quant, 8 - self.weight_quant)
        self.weight_dequant_scale = 1 / self.weight_scale.astype("float32")
        self.max_tokens = int(self.tokens.max())

    def w4afp8_gemm_naive(self, input_bf16, weight_quant, tokens, weight_dequant_scale):
        all_tokens = int(tokens.sum())
        out = paddle.zeros([all_tokens, self.N], dtype="bfloat16")
        pre_fix_token = 0
        for i in range(self.BATCH):
            input = input_bf16[pre_fix_token : pre_fix_token + tokens[i], :]
            weight = weight_quant[i] * weight_dequant_scale[i]
            out_i = paddle.matmul(input, weight.astype("bfloat16"), transpose_y=True)
            out[pre_fix_token : pre_fix_token + tokens[i], :] = out_i
            pre_fix_token += tokens[i]
        return out

    def permute_scale(self, weight_scale):
        weight_scale = weight_scale.reshape([self.BATCH, self.N])
        temp = paddle.zeros([16])
        for b in range(self.BATCH):
            for n in range(0, self.N, 16):
                temp[:] = weight_scale[b, n : n + 16]
                for j in range(0, 16, 2):
                    weight_scale[b, n + j] = temp[j // 2]
                    weight_scale[b, n + j + 1] = temp[j // 2 + 8]
        return weight_scale

    def get_per_group_scale(self, processed_weight_scale):
        processed_weight_scale = processed_weight_scale.repeat_interleave(self.K // 128, axis=-1)
        origin_shape = processed_weight_scale.shape
        processed_weight_scale = processed_weight_scale.transpose([0, 2, 1])
        processed_weight_scale = processed_weight_scale.reshape([-1, processed_weight_scale.shape[-1]])

        processed_weight_scale = w4afp8_gemm_scale_permute(processed_weight_scale)
        processed_weight_scale = processed_weight_scale.reshape(
            [origin_shape[0], origin_shape[2], origin_shape[1] // 128, 128]
        )
        processed_weight_scale = processed_weight_scale.transpose([0, 2, 1, 3])
        return processed_weight_scale

    def test_w4afp8_gemm(self):
        out_naive = self.w4afp8_gemm_naive(
            self.input_bf16, self.weight_quant_naive, self.tokens, self.weight_dequant_scale
        )

        # weight_dequant_scale = paddle.to_tensor(self.permute_scale(self.weight_dequant_scale) * 512)
        weight_dequant_scale = self.get_per_group_scale(self.weight_dequant_scale * 512)
        weight_int4 = w4afp8_gemm_weight_convert(self.weight_quant.astype("uint8").cpu()).cuda()

        if self.TokenPadding == 0:
            out_cuda = w4afp8_gemm(
                self.input_fp8,
                weight_int4,
                self.tokens_prefix_sum,
                weight_dequant_scale.astype("float32"),
                None,
                int(self.TokenPadding),
                self.all_tokens,
                True,
            )
        else:
            out_cuda = w4afp8_gemm(
                self.input_fp8,
                weight_int4,
                self.tokens,
                weight_dequant_scale.astype("float32"),
                None,
                int(self.TokenPadding),
                self.max_tokens,
                True,
            )

        gap = (out_cuda - out_naive).abs()
        self.assertLess(float(gap.mean()), 0.11)


if __name__ == "__main__":
    unittest.main()
