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
from unittest import mock

from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.w4afp8 import (
    QUANT_SCALING_FACTOR,
    W4AFP8Config,
    W4AFP8LinearMethod,
)


class TestW4AFP8(unittest.TestCase):
    def setUp(self):
        self.config = W4AFP8Config(
            weight_scale_dict={"layer.weight_scale": 1.0},
            act_scale_dict={"layer.activation_scale": 1.0},
            is_permuted=False,
            hadamard_block_size=128,
        )
        self.method = W4AFP8LinearMethod(self.config)

        # Mock layer
        self.layer = mock.Mock()
        self.layer.weight_shape = [8, 4]
        self.layer.create_parameter.return_value = "created_weight"
        self.layer.bias = "bias"
        self.layer.add_bias = True
        self.layer._dtype = "float16"
        self.layer.prefix = "layer"

    def test_name(self):
        self.assertEqual(self.config.name(), "w4afp8")

    def test_from_config_defaults(self):
        cfg = W4AFP8Config.from_config({})
        self.assertTrue(cfg.is_permuted)
        self.assertEqual(cfg.hadamard_block_size, 128)

    def test_from_config_full(self):
        cfg = W4AFP8Config.from_config(
            {
                "weight_scale_dict": {"a": 1},
                "act_scale_dict": {"b": 2},
                "is_permuted": False,
                "hadamard_block_size": 64,
            }
        )
        self.assertEqual(cfg.weight_scale_dict["a"], 1)
        self.assertEqual(cfg.hadamard_block_size, 64)
        self.assertFalse(cfg.is_permuted)

    def test_get_quant_method_linear(self):
        # Non-FusedMoE path
        method = self.config.get_quant_method(mock.Mock())
        self.assertIsInstance(method, W4AFP8LinearMethod)

    @mock.patch("fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend.CutlassW4AFP8MoEMethod")
    def test_get_quant_method_moe(self, mock_cutlass):
        # Mock FusedMoE instance
        layer = mock.Mock(spec=FusedMoE)
        type(layer).return_value = None
        result = self.config.get_quant_method(layer)

        mock_cutlass.assert_called_once_with(self.config)
        self.assertEqual(result, mock_cutlass.return_value)

    def test_create_weights(self):
        original_shape = [8, 4]
        self.layer.weight_shape = original_shape.copy()

        self.method.create_weights(self.layer)
        self.assertEqual(self.layer.weight_dtype, "int8")
        self.assertEqual(self.layer.weight, "created_weight")
        self.assertEqual(self.layer.weight_shape, [2, 8])

    @mock.patch("fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize")
    @mock.patch("paddle.view")
    @mock.patch("paddle.cast")
    def test_process_loaded_weights(self, mock_cast, mock_view, mock_quant):
        mock_cast.return_value.cpu.return_value = "cpu_tensor"
        mock_quant.return_value = ("quanted_weight", "weight_scale")
        mock_view.return_value = "reshaped_scale"

        self.layer.weight = mock.Mock()
        self.layer.weight_scale = mock.Mock()

        self.method.process_loaded_weights(self.layer, "weights")

        mock_cast.assert_called_once_with("weights", "float32")
        mock_quant.assert_called_once()
        mock_view.assert_called_once_with("weight_scale", self.layer._dtype)
        self.layer.weight.set_value.assert_called_once_with("quanted_weight")
        self.layer.weight_scale.set_value.assert_called_once_with("reshaped_scale")

    @mock.patch("fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize")
    @mock.patch("paddle.view")
    @mock.patch("paddle.cast")
    def test_process_loaded_weights_with_error(self, mock_cast, mock_view, mock_quant):
        mock_cast.return_value.cpu.return_value = "cpu_tensor"
        mock_quant.return_value = (None, None)
        self.layer.weight = mock.Mock()
        self.layer.weight_scale = mock.Mock()

        self.method.process_loaded_weights(self.layer, "weights")

    @mock.patch("fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16")
    def test_apply_with_bias(self, mock_gemm):
        mock_gemm.return_value = "output"
        x = mock.Mock()
        self.layer.weight = "w"
        self.layer.weight_scale = "s"

        result = self.method.apply(self.layer, x)
        mock_gemm.assert_called_once()
        self.assertEqual(result, "output")

        # Verify out_scale value
        call_args = mock_gemm.call_args.kwargs
        expected_out_scale = 1.0 / (1.0 * QUANT_SCALING_FACTOR * QUANT_SCALING_FACTOR)
        self.assertAlmostEqual(call_args["out_scale"], expected_out_scale)

    @mock.patch("fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16")
    def test_apply_without_bias(self, mock_gemm):
        self.layer.add_bias = False
        mock_gemm.return_value = "out"
        x = "x"

        result = self.method.apply(self.layer, x)
        self.assertEqual(result, "out")
        args = mock_gemm.call_args.kwargs
        self.assertIsNone(args["bias"])

    @mock.patch("fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16")
    def test_apply_prefix_missing_key(self, mock_gemm):
        self.layer.prefix = "unknown"
        x = mock.Mock()
        with self.assertRaises(TypeError):
            self.method.apply(self.layer, x)


if __name__ == "__main__":
    unittest.main()
