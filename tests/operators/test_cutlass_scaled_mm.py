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
"""UT for air_topp_sampling kernel"""

import subprocess
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.layers.quantization.ops import (
    cutlass_scaled_mm,
    scaled_fp8_quant,
)


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        np.random.seed(42)
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor
        print(self.prop)
        print(paddle.__git_commit__)
        nvcc_output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        self.nvcc_cuda_version = float(output[release_idx].split(",")[0])

    def test_cutlass_scaled_mm_fp8(self):
        """
        Check cutlass_scaled_mm output.
        """
        if self.sm_version < 89:
            self.skipTest("cutlass_scaled_mm with fp8 input only support sm89+")
        M = 32
        N = 1024
        K = 1024
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        b_q, b_scales = scaled_fp8_quant(b, use_per_token_if_dynamic=False)
        a_q, a_scales = scaled_fp8_quant(a, use_per_token_if_dynamic=True)

        # Ensure quantized tensors and scales are valid
        assert a_q.numel() > 0, "Quantized tensor 'a_q' must not be empty"
        assert b_q.numel() > 0, "Quantized tensor 'b_q' must not be empty"
        assert a_scales.numel() > 0, "Scale tensor 'a_scales' must not be empty"
        assert b_scales.numel() > 0, "Scale tensor 'b_scales' must not be empty"

        bias = paddle.rand([N], dtype=paddle.bfloat16)
        baseline = paddle.matmul(a, b, transpose_x=False, transpose_y=True)
        if bias is not None:
            baseline = paddle.add(baseline, bias)
        out_type = a.dtype
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, out_type, bias)
        equal = np.allclose(baseline.numpy(), c.numpy(), rtol=1e-2, atol=1e-2)
        print(equal)  #

    def test_cutlass_scaled_mm_int8(self):
        """
        Check cutlass_scaled_mm output.
        """
        M = 32
        N = 1024
        K = 512
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        a_scales = (a.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        a_q = paddle.clip(a / a_scales, -127, 127).cast(paddle.int8)
        b_scales = (b.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        b_q = paddle.clip(b / b_scales, -127, 127).cast(paddle.int8)

        bias = paddle.rand([N], dtype=paddle.bfloat16)
        baseline = paddle.matmul(a, b, transpose_x=False, transpose_y=True)
        if bias is not None:
            baseline = paddle.add(baseline, bias)
        out_type = a.dtype
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, out_type, bias)
        equal = np.allclose(baseline.numpy(), c.numpy(), rtol=1e-2, atol=1e-2)
        print(equal)  #


if __name__ == "__main__":
    unittest.main()
