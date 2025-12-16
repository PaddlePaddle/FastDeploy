"""
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
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.model_executor.layers.quantization.tensor_wise_fp8 import (
    TensorWiseFP8Config,
    TensorWiseFP8LinearMethod,
)


# Dummy classes for test
class DummyLayer:
    """Dummy linear layer for test purposes"""

    def __init__(self):
        self.weight_shape = [4, 8]
        self.weight_key = "weight"
        self.weight_scale_key = "weight_scale"
        self.act_scale_key = "act_scale"
        self.weight_dtype = "float32"
        self.weight = MagicMock()  # Mock weight to avoid dtype copy errors

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        """Mock parameter creation"""
        return MagicMock()


class DummyFusedMoE:
    """Dummy FusedMoE class for patching"""

    pass


class TestTensorWiseFP8Config(unittest.TestCase):
    """Test suite for TensorWiseFP8Config"""

    def test_get_quant_method_linear(self):
        """Verify linear layer returns TensorWiseFP8LinearMethod"""
        cfg = TensorWiseFP8Config()
        layer = DummyLayer()
        method = cfg.get_quant_method(layer)
        self.assertIsInstance(method, TensorWiseFP8LinearMethod)

    def test_get_quant_method_moe(self):
        """Verify FusedMoE layer returns valid quant method"""
        cfg = TensorWiseFP8Config()
        layer = DummyFusedMoE()
        with patch("fastdeploy.model_executor.layers.moe.FusedMoE", DummyFusedMoE):
            method = cfg.get_quant_method(layer)
            self.assertTrue(hasattr(method, "quant_config"))


class TestTensorWiseFP8LinearMethod(unittest.TestCase):
    """Test suite for TensorWiseFP8LinearMethod"""

    def setUp(self):
        """Initialize test fixtures"""
        self.layer = DummyLayer()
        self.method = TensorWiseFP8LinearMethod(TensorWiseFP8Config())
        # Initialize scales to avoid apply errors
        self.method.act_scale = 1.0
        self.method.total_scale = 1.0

    def test_create_weights(self):
        """Verify weight dtype is set to float8_e4m3fn"""
        self.method.create_weights(self.layer)
        self.assertEqual(self.layer.weight_dtype, "float8_e4m3fn")

    def test_process_prequanted_weights(self):
        """Verify prequantized weights and scales are processed correctly"""
        self.layer.weight.copy_ = MagicMock()
        state_dict = {
            "weight": paddle.randn([8, 4]),
            "weight_scale": paddle.to_tensor([0.5], dtype="float32"),
            "act_scale": paddle.to_tensor([2.0], dtype="float32"),
        }
        self.method.process_prequanted_weights(self.layer, state_dict)
        self.assertAlmostEqual(self.method.act_scale, 2.0)
        self.assertAlmostEqual(self.method.total_scale, 1.0)
        self.layer.weight.copy_.assert_called_once()

    @patch("fastdeploy.model_executor.ops.gpu.fused_hadamard_quant_fp8", autospec=True)
    @patch("fastdeploy.model_executor.ops.gpu.cutlass_fp8_fp8_half_gemm_fused", autospec=True)
    def test_apply(self, mock_gemm, mock_quant):
        """Verify apply method executes with mocked ops"""
        mock_quant.side_effect = lambda x, scale: x
        mock_gemm.side_effect = lambda x, w, **kwargs: x
        x = paddle.randn([4, 8])
        out = self.method.apply(self.layer, x)
        self.assertTrue((out == x).all())


if __name__ == "__main__":
    unittest.main()
