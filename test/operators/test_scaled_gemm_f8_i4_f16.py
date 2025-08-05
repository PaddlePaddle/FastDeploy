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

"""UT for fp8_int4_gemm kernel"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import (
    scaled_gemm_f8_i4_f16,
    scaled_gemm_f8_i4_f16_weight_quantize,
)


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        if cc != 89:
            self.skipTest("scaled_gemm_f8_i4_f16 only support sm 89!")

    def quant_fp8_pertensor(self, tensor):
        """
        quant_fp8_pertensor
        """
        scale = paddle.max(paddle.abs(tensor))
        tensor = paddle.cast((tensor * 448 / scale).clip(-448, 448), "float8_e4m3fn").astype(tensor.dtype)
        return tensor, scale

    def dequant_fp8_pertensor(self, tensor, scale):
        """
        dequant_fp8_pertensor
        """
        tensor = (tensor / 448 * scale).astype(tensor.dtype)
        return tensor

    def quant_int4_fp8_matmul(self, A, B, dtype):
        """
        quant_int4_fp8_matmul
        """
        A_fp8, A_fp8_scale = self.quant_fp8_pertensor(A)
        B_fp8, B_fp8_scale = self.quant_fp8_pertensor(B)

        processed_B, w_scale = scaled_gemm_f8_i4_f16_weight_quantize(B_fp8, groupsize=-1, scale_dtype="float16")
        w_scale = paddle.view(w_scale, dtype)
        out_scale = (A_fp8_scale / 448) * (B_fp8_scale / 448)

        out = scaled_gemm_f8_i4_f16(
            x=paddle.cast(A_fp8, "float8_e4m3fn").cuda(),
            y=processed_B.cuda(),
            scale=w_scale.cuda(),
            zero_points=None,
            bias=None,
            out_scale=out_scale,
            groupsize=0,
            out_dtype=dtype,
        )
        return out

    def test_fp16(self):
        """
        Check fp16.
        """
        A_fp32 = paddle.ones((4, 128)).clip(-448, 448)
        B_fp32 = paddle.ones((128, 512)).clip(-448, 448)
        C_fp32 = paddle.matmul(A_fp32, B_fp32)

        out = self.quant_int4_fp8_matmul(A_fp32, B_fp32, "float16")
        out = paddle.cast(out, "float32")

        np.testing.assert_allclose(C_fp32.numpy(), out.numpy(), rtol=1e-04, atol=1e-04)

    def test_bf16(self):
        """
        Check bf16.
        """
        A_fp32 = paddle.ones((4, 128)).clip(-448, 448)
        B_fp32 = paddle.ones((128, 512)).clip(-448, 448)
        C_fp32 = paddle.matmul(A_fp32, B_fp32)

        out = self.quant_int4_fp8_matmul(A_fp32, B_fp32, "bfloat16")
        out = paddle.cast(out, "float32")

        np.testing.assert_allclose(C_fp32.numpy(), out.numpy(), rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
