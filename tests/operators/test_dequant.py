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
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import dequant_int8, gemm_dequant


class Test(unittest.TestCase):
    def setUp(self):
        """
            Initialize the test environment,
        including setting random seeds.
        """
        paddle.seed(2024)

    def testcase1(self):
        """
        Check if the gemm_dequant function works properly.
        """

        nks = [[2048, 2048], [2048, 5504], [6144, 2048]]
        m = [2, 8]

        combinations = list(product(m, nks))
        for m, (n, k) in combinations:
            act = paddle.rand([m, k])
            weight = paddle.rand([n, k])
            act_int_tensor = (act * 128).astype("int8")
            weight_int_tensor = (weight * 128).astype("int8")
            scale = paddle.rand([n])
            linear_out = paddle.matmul(act_int_tensor, weight_int_tensor, transpose_y=True)
            result = dequant_int8(linear_out, scale, "bfloat16")

            result_gemm_dequant = gemm_dequant(
                act_int_tensor,
                weight_int_tensor,
                scale,
                out_dtype="bfloat16",
            )
            np.testing.assert_allclose(
                result.numpy(),
                result_gemm_dequant.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )


if __name__ == "__main__":
    unittest.main()
