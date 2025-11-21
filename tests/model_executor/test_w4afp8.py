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

from fastdeploy.model_executor.layers.quantization.w4afp8 import (
    W4AFP8Config,
    W4AFP8LinearMethod,
)


class DummyLayer:
    def __init__(self):
        self.weight_shape = [8, 16]  # Mock weight shape
        self.weight_dtype = None  # To be set by quant method
        self._dtype = "float32"  # Base dtype
        self.prefix = "dummy"  # Layer identifier prefix
        self.bias = None  # No bias by default
        self.add_bias = False  # Disable bias addition

    # Mock parameter creation for weight/bias
    def create_parameter(self, shape, dtype, is_bias, default_initializer):
        param = paddle.zeros(shape, dtype=dtype)
        return param

    # Mock method to set weight value
    def set_value(self, tensor):
        self.weight = tensor


class DummyFusedMoE:
    pass


class TestW4AFP8Config(unittest.TestCase):
    """Test suite for W4AFP8Config"""

    def test_name_and_from_config(self):
        # Test config initialization from dict and name property
        config_dict = {"weight_scale_dict": {}, "act_scale_dict": {}, "is_permuted": True, "hadamard_block_size": 128}
        cfg = W4AFP8Config.from_config(config_dict)

        # Verify config properties
        self.assertEqual(cfg.name(), "w4afp8")
        self.assertEqual(cfg.is_permuted, True)
        self.assertEqual(cfg.hadamard_block_size, 128)

    def test_get_quant_method_linear(self):
        # Test quant method retrieval for linear layer
        cfg = W4AFP8Config({}, {}, True, 128)
        layer = DummyLayer()
        method = cfg.get_quant_method(layer)

        # Verify method type and config binding
        self.assertIsInstance(method, W4AFP8LinearMethod)
        self.assertEqual(method.quant_config, cfg)

    def test_get_quant_method_moe(self):
        # Test quant method retrieval for MoE layer
        cfg = W4AFP8Config({}, {}, True, 128)
        layer = DummyFusedMoE()

        # Patch FusedMoE to dummy class for test scope
        with patch("fastdeploy.model_executor.layers.moe.FusedMoE", DummyFusedMoE):
            method = cfg.get_quant_method(layer)

        # Verify method has required config attribute
        self.assertTrue(hasattr(method, "quant_config"))


class TestW4AFP8LinearMethod(unittest.TestCase):
    """Test suite for W4AFP8LinearMethod"""

    def setUp(self):
        # Initialize test fixtures: config, method, and dummy layer
        self.cfg = W4AFP8Config({}, {}, True, 128)
        self.method = W4AFP8LinearMethod(self.cfg)
        self.layer = DummyLayer()

    def test_create_weights(self):
        # Test weight creation with correct dtype and shape
        self.method.create_weights(self.layer)

        # Verify weight properties
        self.assertEqual(self.layer.weight_dtype, "int8")  # W4A uses int8 storage
        self.assertIsInstance(self.layer.weight, paddle.Tensor)
        self.assertEqual(list(self.layer.weight.shape), [8, 8])  # Shape adjusted for quantization

    @patch(
        "fastdeploy.model_executor.layers.quantization.w4afp8.fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize"
    )
    def test_process_loaded_weights(self, mock_quant):
        # Mock quantize op output: (quantized_weight, scale)
        dummy_weights = paddle.ones([8, 16])
        mock_quant.return_value = (paddle.ones([4, 16]), paddle.ones([4], dtype="float32"))

        # Mock layer weight and scale attributes
        self.layer.weight = MagicMock()
        self.layer.weight_scale = MagicMock()

        # Execute weight processing
        self.method.process_loaded_weights(self.layer, dummy_weights)

        # Verify weight and scale are set
        self.layer.weight.set_value.assert_called_once()
        self.layer.weight_scale.set_value.assert_called_once()

    @patch(
        "fastdeploy.model_executor.layers.quantization.w4afp8.fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16"
    )
    def test_apply(self, mock_gemm):
        # Prepare input and layer attributes
        x = paddle.ones([2, 4])  # Mock input tensor
        self.layer.weight = paddle.ones([4, 4])  # Quantized weight
        self.layer.weight_scale = paddle.ones([4], dtype="float32")  # Weight scale
        self.layer.add_bias = False  # Disable bias
        self.layer.prefix = "dummy"

        # Set scale dicts in config
        self.cfg.weight_scale_dict = {"dummy.weight_scale": 1.0}
        self.cfg.act_scale_dict = {"dummy.activation_scale": 1.0}

        # Mock GEMM op output
        mock_gemm.return_value = paddle.ones([2, 4])

        # Execute forward pass
        out = self.method.apply(self.layer, x)

        # Verify output and op call
        self.assertIsInstance(out, paddle.Tensor)
        mock_gemm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
