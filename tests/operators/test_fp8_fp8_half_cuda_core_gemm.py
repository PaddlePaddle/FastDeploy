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

"""UT for fp8_fp8_half_cuda_core_gemm kernel"""

import os
import unittest
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_gemm_fused


class Test(unittest.TestCase):
    def setUp(self):
        """
            Initialize the test environment,
        including setting random seeds and environment variables.
        """
        paddle.seed(2024)
        self.E4M3_MAX_POS = 448.0
        os.environ["FLAGS_cuda_core_fp8_gemm"] = "1"
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def testcase1(self):
        """
        Check if the cutlass_fp8_fp8_half_gemm_fused function works properly.
        """

        nks = [[2048, 2048], [2048, 5504], [6144, 2048]]
        nks = nks + [[4096, 4096], [4096, 12800], [6144, 4096]]
        nks = nks + [[5120, 5120], [5120, 13824], [15360, 5120]]

        m = [1, 2, 3, 4]

        combinations = list(product(m, nks))
        for m, (n, k) in combinations:
            act = paddle.rand([m, k]).clip(min=-1 * self.E4M3_MAX_POS, max=self.E4M3_MAX_POS).to(paddle.float8_e4m3fn)
            weight = (
                paddle.rand([n, k]).clip(min=-1 * self.E4M3_MAX_POS, max=self.E4M3_MAX_POS).to(paddle.float8_e4m3fn)
            )
            bias = (paddle.rand([n])).to(paddle.bfloat16)
            scale = 1.2

            result = paddle.matmul(
                act.astype("bfloat16"),
                weight.astype("bfloat16"),
                transpose_y=True,
            )
            result = result * scale
            result = result + bias

            result_cuda = cutlass_fp8_fp8_half_gemm_fused(
                act,
                weight,
                bias=bias,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
                scale=scale,
                activation_type="",
            )

            np.testing.assert_allclose(result.numpy(), result_cuda.numpy(), rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
