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

""" UT for per_channel_fp8_fp8_half_gemm_fused kernel """

import os
import paddle
import numpy as np
from itertools import product
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        """
            Initialize the test environment,
        including setting random seeds and environment variables.
        """
        paddle.seed(2003)
        os.environ["FLAGS_use_cutlass_device_best_config_path"] = "default"

    def testcase1(self):
        """
        Check if the per_channel_fp8_fp8_half_gemm_fused function works properly.
        """
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        if cc < 89:
            self.skipTest("per_channel_fp8_fp8_half_gemm_fused only support sm89+")

        from fastdeploy.model_executor.ops.gpu import per_channel_fp8_fp8_half_gemm_fused

        nks = [[2048, 2048], [2048, 5504], [6144, 2048]]
        nks = nks + [[4096, 4096], [4096, 12800], [6144, 4096]]
        nks = nks + [[5120, 5120], [5120, 13824], [15360, 5120]]

        m = [1, 32, 64, 128, 256, 512, 1024, 2048]

        combinations = list(product(m, nks))
        for m, (n, k) in combinations:
            A_bf16 = paddle.rand(shape=[m, k], dtype="bfloat16")
            A_fp8 = paddle.cast(A_bf16, "float8_e4m3fn")
            B_bf16 = paddle.rand(shape=[n, k], dtype="bfloat16")
            B_fp8 = B_bf16.astype("float8_e4m3fn")

            scalar_scale = paddle.full([1], 0.5, dtype="float32")
            channel_scale = paddle.rand(shape=[n], dtype="float32")
            bias = paddle.rand(shape=[n], dtype="bfloat16")

            result_bf16 = (
                paddle.matmul(A_bf16, B_bf16, transpose_y=True)
                * scalar_scale
                * channel_scale
                + bias
            )
            result_fp8 = per_channel_fp8_fp8_half_gemm_fused(
                A_fp8,
                B_fp8,
                bias=bias,
                scalar_scale=scalar_scale,
                channel_scale=channel_scale,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
            )
            # absolute_error = paddle.abs(result_bf16 - result_fp8)
            # mean_absolute_error = paddle.mean(absolute_error)
            relative_error = paddle.abs(result_bf16 - result_fp8) / (
                paddle.abs(result_bf16)
            )
            mean_relative_error = paddle.mean(relative_error)
            np.testing.assert_allclose(
                mean_relative_error.numpy(), np.array([0.001]), rtol=0.001, atol=0.25
            )


if __name__ == "__main__":
    unittest.main()
