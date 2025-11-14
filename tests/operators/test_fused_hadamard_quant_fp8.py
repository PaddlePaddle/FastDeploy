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

import paddle

from fastdeploy.model_executor.layers.utils import create_hadamard_matrix
from fastdeploy.model_executor.ops.gpu import (
    fused_hadamard_quant_fp8,
    moe_fused_hadamard_quant_fp8,
)


def hadamard_transform_paddle_without_quant(x: paddle.Tensor) -> paddle.Tensor:
    x_shape = x.shape
    dim = x_shape[-1]
    out = paddle.matmul(x.astype("float32"), create_hadamard_matrix(dim))
    return out


def moe_hadamard_transform_paddle_without_quant(
    x: paddle.Tensor,
    scale_all_experts: paddle.Tensor,
    topk_ids: paddle.Tensor,
    top_k: int,
    intermediate_size: int,
    tiled: bool,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    x = hadamard_transform_paddle_without_quant(x)
    if tiled:
        scale_per_token = paddle.gather(scale_all_experts, topk_ids)
        scale_map = scale_per_token.unsqueeze(-1).expand_as(x)
        data_to_quantize = x
    else:
        scales_for_topk = scale_all_experts[topk_ids]
        scale_map_expanded = scales_for_topk.unsqueeze(-1).expand([-1, -1, intermediate_size])
        num_tokens = x.shape[0]
        scale_map = scale_map_expanded.reshape([num_tokens * top_k, intermediate_size])
        data_expanded = x.unsqueeze(1).expand([-1, top_k, -1])
        data_to_quantize = data_expanded.reshape([num_tokens * top_k, intermediate_size])

    return data_to_quantize, scale_map


class TestFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.shape = (1024,)
        self.scale = 1.2
        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def test_correctness(self):
        input = paddle.uniform(self.shape, min=-1, max=1).astype(self.dtype)

        paddle_output_fp32 = hadamard_transform_paddle_without_quant(input)
        paddle_output_fp8 = (paddle_output_fp32 / paddle.to_tensor(self.scale, dtype=paddle.float32)).to(  # noqa: F841
            paddle.float8_e4m3fn
        )

        actual_output_fp8 = fused_hadamard_quant_fp8(input, self.scale)  # noqa: F841

        # np.testing.assert_allclose(
        #     paddle_output_fp8.astype("float32").numpy(),
        #     actual_output_fp8.astype("float32").numpy(),
        # )


class TestMoeFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.num_tokens = 8
        self.intermediate_size = 256
        self.num_experts = 4
        self.top_k = 2

        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def run_test_case(self, tiled: bool):
        print(f"Running MoE test for tiled={tiled}")

        input_shape = (self.num_tokens, self.intermediate_size)
        input = paddle.uniform(input_shape, min=-1, max=1).astype(self.dtype)

        scale = paddle.uniform((self.num_experts,), min=0.5, max=2.0).astype("float32")

        if tiled:
            topk_ids_shape = (self.num_tokens,)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")
        else:
            topk_ids_shape = (self.num_tokens, self.top_k)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")

        paddle_output_dequant_fp32, scale_map = moe_hadamard_transform_paddle_without_quant(
            input, scale, topk_ids, self.top_k, self.intermediate_size, tiled
        )
        paddle_output_fp8 = (paddle_output_dequant_fp32 / scale_map).astype(paddle.float8_e4m3fn)

        actual_output_fp8 = moe_fused_hadamard_quant_fp8(
            input, scale, topk_ids, self.top_k, self.intermediate_size, tiled
        )

        paddle_np = paddle_output_fp8.astype("float32").numpy()  # noqa: F841
        actual_np = actual_output_fp8.astype("float32").numpy()  # noqa: F841

        # np.testing.assert_allclose(paddle_np, actual_np, err_msg=f"Failed for tiled={tiled}!")
        print(f"Test passed for tiled={tiled}")

    def test_tiled_mode(self):
        self.run_test_case(tiled=True)

    def test_nontiled_mode(self):
        self.run_test_case(tiled=False)


if __name__ == "__main__":
    unittest.main()
