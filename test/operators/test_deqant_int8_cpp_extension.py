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

import unittest

import numpy as np
import paddle


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def dequant_int8_test(self, dynamic_mode=False):
        """
        Check air_topp_sampling output with paddle.tensor.top_p_sampling.
        """
        if not dynamic_mode:
            paddle.enable_static()
        else:
            paddle.disable_static()
        from fastdeploy.model_executor.ops.gpu import dequant_int8

        input_tensor = paddle.cast(paddle.ones([128, 128]), "int32")
        scale_tensor = paddle.cast(paddle.ones([128]), "float32")
        out = dequant_int8(input_tensor, scale_tensor, "float16")
        return out

    def test(self):
        op_out = self.dequant_int8_test()
        exe = paddle.static.Executor()
        exe.run(paddle.static.default_startup_program())
        op_out = exe.run(fetch_list=[op_out])[0]
        func_out = self.dequant_int8_test(True)
        np.testing.assert_allclose(op_out, func_out.numpy(), rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
