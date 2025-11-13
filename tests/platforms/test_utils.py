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
from unittest.mock import patch

import numpy as np
import paddle

from fastdeploy.platforms.utils import convert_to_npu_dequant_scale


class TestConvertToNpuDequantScale(unittest.TestCase):

    def test_npu_not_available(self):
        with patch("paddle.is_compiled_with_custom_device", return_value=False):
            x = paddle.to_tensor([1.0, 2.0, 3.0], dtype=paddle.float32)
            out = convert_to_npu_dequant_scale(x)
            self.assertTrue((out.numpy() == x.numpy()).all())

    def test_npu_available(self):
        with patch("paddle.is_compiled_with_custom_device", return_value=True):
            x = paddle.to_tensor([1, 2, 3], dtype=paddle.float32)
            out = convert_to_npu_dequant_scale(x)
            self.assertEqual(out.dtype, paddle.int64)
            # Verify scaled output matches expected NPU dequantization format
            arr = x.numpy()
            new_deq_scale = np.stack([arr.reshape(-1, 1), np.zeros_like(arr).reshape(-1, 1)], axis=-1).reshape(-1)
            expected = np.frombuffer(new_deq_scale.tobytes(), dtype=np.int64)
            self.assertTrue((out.numpy() == expected).all())


if __name__ == "__main__":
    unittest.main()
