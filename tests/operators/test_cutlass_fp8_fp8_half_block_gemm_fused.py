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

import os
import unittest
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_block_gemm_fused


class TestFp8Fp8HalfBlockGemmFused(unittest.TestCase):
    def setUp(self):
        paddle.seed(2025)
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor
        print(f"sm version: {self.sm_version}")
        self.E4M3_MAX_POS = 448.0
        os.environ["FLAGS_cuda_core_fp8_gemm"] = "1"
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def test_block_gemm_case(self):
        # if self.sm_version < 90:
        #     self.skipTest("cutlass_fp8_fp8_half_block_gemm_fused only support sm90+")

        nks = [
            [2048, 2048],
            [2048, 5504],
            [6144, 2048],
            [4096, 4096],
            [4096, 12800],
            [6144, 4096],
            [5120, 5120],
            [5120, 13824],
            [15360, 5120],
        ]
        m_values = [1, 2, 3, 4]
        activation_types = ["", "relu"]

        combinations = product(m_values, nks, activation_types)

        for m, (n, k), act_type in combinations:
            x = paddle.rand([m, k]).clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS).to(paddle.float8_e4m3fn)
            y = paddle.rand([n, k]).clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS).to(paddle.float8_e4m3fn)

            x_scale = (
                paddle.rand(
                    [(k + 127) // 128, m],
                    dtype="float32",
                )
                + 1.0
            )
            y_scale = (
                paddle.rand(
                    [(n + 127) // 128, (k + 127) // 128],
                    dtype="float32",
                )
                + 1.0
            )

            bias = paddle.rand([n]).to(paddle.bfloat16)

            x_fp32 = x.astype("float32")
            K = x_fp32.shape[-1]
            x_scale_expanded = paddle.repeat_interleave(x_scale, repeats=128, axis=-2)[:K, :]
            x_scale_expanded = x_scale_expanded.transpose([1, 0])
            x_dequant = x_fp32 * x_scale_expanded

            y_fp32 = y.astype("float32")
            K_y = y_fp32.shape[-1]
            N_y = y_fp32.shape[-2]

            y_scale_expanded_k = paddle.repeat_interleave(y_scale, repeats=128, axis=-2)[..., :K_y, :]
            y_scale_expanded_kn = paddle.repeat_interleave(y_scale_expanded_k, repeats=128, axis=-1)[..., :N_y]
            y_dequant = y_fp32 * y_scale_expanded_kn

            x_bf16 = x_dequant.astype("bfloat16")
            y_bf16 = y_dequant.astype("bfloat16")
            ref_out = paddle.matmul(x_bf16, y_bf16, transpose_y=True)

            if bias is not None:
                ref_out = ref_out + bias

            act = "noact" if (act_type == "" or act_type == "identity") else act_type
            if act == "relu":
                ref_out = paddle.nn.functional.relu(ref_out)

            result = cutlass_fp8_fp8_half_block_gemm_fused(
                x,
                y,
                x_scale,
                y_scale,
                bias,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
                act=act_type,
            )

            np.testing.assert_allclose(
                ref_out.astype("float32").numpy(),
                result.astype("float32").numpy(),
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
