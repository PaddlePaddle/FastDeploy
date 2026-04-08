# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_fp8_dual_gemm_fused


class TestFp8Fp8Fp8DualGemm(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment,
        including setting random seeds and environment variables.
        """
        paddle.seed(2024)
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor
        print(f"sm version: {self.sm_version}")
        self.E4M3_MAX_POS = 448.0
        os.environ["FLAGS_cuda_core_fp8_gemm"] = "1"
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def test_dual_gemm_case(self):
        """
        Check if the cutlass_fp8_fp8_fp8_dual_gemm_fused function works properly.
        """
        if self.sm_version < 90:
            self.skipTest("cutlass_fp8_fp8_fp8_dual_gemm_fused only support sm90+")
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
        transpose_combinations = [(False, True)]
        activation_types = [""]

        combinations = product(m_values, nks, transpose_combinations, activation_types)
        for m, (n, k), (trans_x, trans_y), act_type in combinations:
            x = (
                paddle.rand([m, k] if not trans_x else [k, m])
                .clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS)
                .to(paddle.float8_e4m3fn)
            )

            y0 = (
                paddle.rand([k, n] if not trans_y else [n, k])
                .clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS)
                .to(paddle.float8_e4m3fn)
            )

            y1 = (
                paddle.rand([k, n] if not trans_y else [n, k])
                .clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS)
                .to(paddle.float8_e4m3fn)
            )

            scale0 = 1.2
            scale1 = 0.8
            scale_out = 1.0

            x_bf16 = x.astype("bfloat16")
            y0_bf16 = y0.astype("bfloat16")
            y1_bf16 = y1.astype("bfloat16")

            gemm0 = paddle.matmul(x_bf16, y0_bf16, transpose_x=trans_x, transpose_y=trans_y)
            gemm1 = paddle.matmul(x_bf16, y1_bf16, transpose_x=trans_x, transpose_y=trans_y)

            gemm0 = gemm0 * scale0
            gemm1 = gemm1 * scale1

            if act_type == "" or act_type == "swiglu":
                ref_out = gemm0 * paddle.nn.functional.sigmoid(gemm1)

            ref_out = ref_out.clip(min=-self.E4M3_MAX_POS, max=self.E4M3_MAX_POS).to(paddle.float8_e4m3fn)

            result = cutlass_fp8_fp8_fp8_dual_gemm_fused(
                x,
                y0,
                y1,
                bias0=None,
                bias1=None,
                transpose_x=trans_x,
                transpose_y=trans_y,
                scale0=scale0,
                scale1=scale1,
                scale_out=scale_out,
                activation_type=act_type,
            )

            np.testing.assert_allclose(
                ref_out.astype("float32").numpy(), result.astype("float32").numpy(), rtol=5e-3, atol=5e-3
            )


if __name__ == "__main__":
    unittest.main()
