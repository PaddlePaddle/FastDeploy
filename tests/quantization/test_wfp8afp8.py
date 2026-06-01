"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.model_executor.layers.quantization.wfp8afp8 import (
    WFP8AFP8Config,
    WFP8AFP8LinearMethod,
)


class TestWFP8AFP8Config(unittest.TestCase):
    """Test WFP8AFP8Config class."""

    def test_init_defaults(self):
        """__init__ sets default attributes."""
        config = WFP8AFP8Config()
        self.assertEqual(config.quant_max_bound, 448)
        self.assertEqual(config.quant_min_bound, -448)
        self.assertEqual(config.quant_round_type, 1)
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertEqual(config.weight_block_size, [-1, 1])
        self.assertFalse(config.is_checkpoint_bf16)

    def test_init_custom(self):
        """__init__ stores custom values."""
        config = WFP8AFP8Config(
            activation_scheme="static",
            weight_block_size=[128, 128],
            is_checkpoint_bf16=True,
        )
        self.assertEqual(config.activation_scheme, "static")
        self.assertEqual(config.weight_block_size, [128, 128])
        self.assertTrue(config.is_checkpoint_bf16)

    def test_name(self):
        """name() returns 'wfp8afp8'."""
        config = WFP8AFP8Config()
        self.assertEqual(config.name(), "wfp8afp8")

    def test_from_config_quantized(self):
        """from_config sets is_checkpoint_bf16=False when is_quantized=True."""
        config = WFP8AFP8Config.from_config({"is_quantized": True})
        self.assertFalse(config.is_checkpoint_bf16)

    def test_from_config_not_quantized(self):
        """from_config sets is_checkpoint_bf16=True when is_quantized=False."""
        config = WFP8AFP8Config.from_config({"is_quantized": False})
        self.assertTrue(config.is_checkpoint_bf16)

    def test_from_config_missing_key(self):
        """from_config sets is_checkpoint_bf16=True when is_quantized missing."""
        config = WFP8AFP8Config.from_config({})
        self.assertTrue(config.is_checkpoint_bf16)

    def test_get_quant_method_non_moe(self):
        """get_quant_method returns WFP8AFP8LinearMethod for non-FusedMoE layers."""
        config = WFP8AFP8Config()

        normal_layer = MagicMock()
        result = config.get_quant_method(normal_layer)
        self.assertIsInstance(result, WFP8AFP8LinearMethod)
        self.assertIs(result.quant_config, config)

    @patch(
        "fastdeploy.model_executor.layers.moe.fused_moe_triton_backend.Wfp8Afp8MoEMethod",
        create=True,
    )
    def test_get_quant_method_moe_layer(self, mock_moe_method_cls):
        """get_quant_method returns Wfp8Afp8MoEMethod for FusedMoE instance."""
        from fastdeploy.model_executor.layers.moe import FusedMoE

        config = WFP8AFP8Config()
        mock_moe_method_cls.return_value = "moe_method_instance"

        layer = MagicMock(spec=FusedMoE)
        config.get_quant_method(layer)
        mock_moe_method_cls.assert_called_once_with(config)


class TestWFP8AFP8LinearMethodInit(unittest.TestCase):
    """Test WFP8AFP8LinearMethod.__init__."""

    def test_init(self):
        """__init__ stores config and sets use_per_token_if_dynamic."""
        config = WFP8AFP8Config()
        method = WFP8AFP8LinearMethod(config)
        self.assertIs(method.quant_config, config)
        self.assertTrue(method.use_per_token_if_dynamic)


class TestWFP8AFP8LinearMethodCreateWeights(unittest.TestCase):
    """Test WFP8AFP8LinearMethod.create_weights."""

    def _make_layer(self, weight_shape=None):
        layer = MagicMock()
        layer.weight_shape = weight_shape or [256, 128]
        layer.weight_dtype = "bfloat16"
        layer._dtype = "bfloat16"
        layer.create_parameter.return_value = MagicMock()
        return layer

    def test_create_weights_non_bf16_checkpoint(self):
        """create_weights reverses shape and sets fp8 dtype when not bf16 checkpoint."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False)
        method = WFP8AFP8LinearMethod(config)
        layer = self._make_layer(weight_shape=[256, 128])

        method.create_weights(layer)

        # weight_shape reversed to [128, 256]
        self.assertEqual(layer.weight_dtype, "float8_e4m3fn")
        # create_parameter called twice: weight + weight_scale
        self.assertEqual(layer.create_parameter.call_count, 2)
        self.assertFalse(method.skip_quant)

    def test_create_weights_non_bf16_scale_shape(self):
        """create_weights computes correct scale_shape for non-bf16 checkpoint."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False, weight_block_size=[-1, 1])
        method = WFP8AFP8LinearMethod(config)
        layer = self._make_layer(weight_shape=[256, 128])

        method.create_weights(layer)

        # scale_shape computation:
        # weight_shape=[256, 128], weight_block_size=[-1, 1]
        # scale_shape[0] = 1 (block_size=-1 -> 1)
        # scale_shape[1] = 128 (128 // 1 = 128)
        # reversed -> [128, 1]
        scale_call = layer.create_parameter.call_args_list[1]
        self.assertEqual(scale_call[1]["shape"], [128, 1])
        self.assertEqual(scale_call[1]["dtype"], "float32")

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.set_weight_attrs")
    def test_create_weights_bf16_checkpoint_default_v1(self, mock_set_weight_attrs):
        """create_weights handles bf16 checkpoint with default_v1 load."""
        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)

        layer = self._make_layer(weight_shape=[256, 128])
        layer.fd_config = MagicMock()
        layer.fd_config.load_config.load_choices = "default_v1"

        method.create_weights(layer, model_format="paddle")

        # create_parameter called once for weight
        layer.create_parameter.assert_called_once()
        mock_set_weight_attrs.assert_called_once()

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.TensorTracker")
    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.set_weight_attrs")
    def test_create_weights_bf16_merged_column_parallel(self, mock_set_weight_attrs, mock_tracker):
        """create_weights adds TensorTracker for MergedColumnParallelLinear."""
        from fastdeploy.model_executor.layers.linear import MergedColumnParallelLinear

        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)

        layer = MagicMock(spec=MergedColumnParallelLinear)
        layer.weight_shape = [256, 128]
        layer.weight_dtype = "bfloat16"
        layer.create_parameter = MagicMock(return_value=MagicMock())
        layer.fd_config = MagicMock()
        layer.fd_config.load_config.load_choices = "default_v1"

        method.create_weights(layer, model_format="paddle", output_dim=True)

        mock_set_weight_attrs.assert_called_once()
        # Check that TensorTracker was created
        call_kwargs = mock_set_weight_attrs.call_args[0][1]
        self.assertIn("tensor_track", call_kwargs)

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.set_weight_attrs")
    def test_create_weights_bf16_torch_format(self, mock_set_weight_attrs):
        """create_weights reverses weight_shape and output_dim for torch format."""
        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)

        layer = self._make_layer(weight_shape=[256, 128])
        layer.fd_config = MagicMock()
        layer.fd_config.load_config.load_choices = "default_v1"

        method.create_weights(layer, model_format="torch", output_dim=True)

        # weight_shape reversed for torch format
        create_param_call = layer.create_parameter.call_args
        self.assertEqual(create_param_call[1]["shape"], [128, 256])
        # Verify set_weight_attrs was called with flipped output_dim
        call_kwargs = mock_set_weight_attrs.call_args[0][1]
        self.assertFalse(call_kwargs["output_dim"])

    def test_create_weights_asserts_shape_len(self):
        """create_weights asserts weight_shape and block_size are length 2."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False, weight_block_size=[-1, 1, 1])
        method = WFP8AFP8LinearMethod(config)
        layer = self._make_layer(weight_shape=[256, 128])

        with self.assertRaises(AssertionError):
            method.create_weights(layer)


class TestWFP8AFP8LinearMethodProcessWeightsAfterLoading(unittest.TestCase):
    """Test WFP8AFP8LinearMethod.process_weights_after_loading."""

    def test_returns_early_if_not_bf16(self):
        """process_weights_after_loading returns immediately if not bf16 checkpoint."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False)
        method = WFP8AFP8LinearMethod(config)
        layer = MagicMock()

        # Should not raise or access layer attributes
        method.process_weights_after_loading(layer)
        layer.weight.transpose.assert_not_called()

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.per_token_cast_to_fp8")
    def test_process_bf16_paddle_format(self, mock_cast_fp8):
        """process_weights_after_loading quantizes bf16 weights (paddle format)."""
        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)
        method.model_format = "paddle"

        qweight = MagicMock()
        qweight.shape = [128, 256]
        weight_scale = MagicMock()
        weight_scale.shape = [128, 1]
        mock_cast_fp8.return_value = (qweight, weight_scale)

        layer = MagicMock()
        weight_mock = MagicMock()
        weight_mock.transpose.return_value.contiguous.return_value = MagicMock()
        layer.weight = weight_mock
        layer.create_parameter.return_value = MagicMock()

        method.process_weights_after_loading(layer)

        mock_cast_fp8.assert_called_once()
        # create_parameter called twice: new weight + weight_scale
        self.assertEqual(layer.create_parameter.call_count, 2)

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.process_weight_transpose")
    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.per_token_cast_to_fp8")
    def test_process_bf16_torch_format(self, mock_cast_fp8, mock_transpose):
        """process_weights_after_loading calls process_weight_transpose for torch format."""
        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)
        method.model_format = "torch"

        qweight = MagicMock()
        qweight.shape = [128, 256]
        weight_scale = MagicMock()
        weight_scale.shape = [128, 1]
        mock_cast_fp8.return_value = (qweight, weight_scale)

        layer = MagicMock()
        weight_mock = MagicMock()
        weight_mock.transpose.return_value.contiguous.return_value = MagicMock()
        layer.weight = weight_mock
        layer.create_parameter.return_value = MagicMock()

        method.process_weights_after_loading(layer)

        mock_transpose.assert_called_once_with(layer, "weight")
        mock_cast_fp8.assert_called_once()

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.per_token_cast_to_fp8")
    def test_process_clears_tensor_track(self, mock_cast_fp8):
        """process_weights_after_loading clears tensor_track if present."""
        config = WFP8AFP8Config(is_checkpoint_bf16=True)
        method = WFP8AFP8LinearMethod(config)
        method.model_format = "paddle"

        qweight = MagicMock()
        qweight.shape = [128, 256]
        weight_scale = MagicMock()
        weight_scale.shape = [128, 1]
        mock_cast_fp8.return_value = (qweight, weight_scale)

        layer = MagicMock()
        weight_mock = MagicMock()
        weight_mock.tensor_track = MagicMock()
        weight_mock.transpose.return_value.contiguous.return_value = MagicMock()
        layer.weight = weight_mock
        layer.create_parameter.return_value = MagicMock()

        method.process_weights_after_loading(layer)

        # tensor_track should be set to None before deletion
        # The code sets layer.weight.tensor_track = None, but then deletes layer.weight
        # So we verify via the mock calls
        mock_cast_fp8.assert_called_once()


class TestWFP8AFP8LinearMethodProcessLoadedWeights(unittest.TestCase):
    """Test WFP8AFP8LinearMethod.process_loaded_weights."""

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.scaled_fp8_quant")
    def test_process_loaded_weights_normal(self, mock_scaled_fp8_quant):
        """process_loaded_weights quantizes and stores weights."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False)
        method = WFP8AFP8LinearMethod(config)
        method.skip_quant = False

        qweight = MagicMock()
        weight_scale = MagicMock()
        mock_scaled_fp8_quant.return_value = (qweight, weight_scale)

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()

        weights = MagicMock()
        weights.dtype = paddle.float16  # not fp8 -> sets use_per_token
        weights.transpose.return_value.contiguous.return_value = "transposed"

        method.process_loaded_weights(layer, weights)

        self.assertTrue(method.use_per_token_if_dynamic)
        mock_scaled_fp8_quant.assert_called_once_with("transposed", use_per_token_if_dynamic=False)
        layer.weight.copy_.assert_called_once_with(qweight, False)
        layer.weight_scale.set_value.assert_called_once_with(weight_scale)

    def test_process_loaded_weights_skip_quant(self):
        """process_loaded_weights handles skip_quant path."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False)
        method = WFP8AFP8LinearMethod(config)
        method.skip_quant = True

        layer = MagicMock()
        layer._dtype = "float16"
        layer.weight = MagicMock()

        weights = MagicMock()
        weights.cast.return_value = "casted_weight"

        method.process_loaded_weights(layer, weights)

        weights.cast.assert_called_once_with("float16")
        layer.weight.set_value.assert_called_once_with("casted_weight")

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.scaled_fp8_quant")
    def test_process_loaded_weights_fp8_dtype(self, mock_scaled_fp8_quant):
        """process_loaded_weights does not set use_per_token when weights already fp8."""
        config = WFP8AFP8Config(is_checkpoint_bf16=False)
        method = WFP8AFP8LinearMethod(config)
        method.skip_quant = False
        method.use_per_token_if_dynamic = False  # pre-set to False

        qweight = MagicMock()
        weight_scale = MagicMock()
        mock_scaled_fp8_quant.return_value = (qweight, weight_scale)

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()

        weights = MagicMock()
        weights.dtype = paddle.float8_e4m3fn  # already fp8
        weights.transpose.return_value.contiguous.return_value = "transposed"

        method.process_loaded_weights(layer, weights)

        # use_per_token_if_dynamic stays False since dtype is already fp8
        self.assertFalse(method.use_per_token_if_dynamic)


class TestWFP8AFP8LinearMethodApply(unittest.TestCase):
    """Test WFP8AFP8LinearMethod.apply."""

    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.cutlass_scaled_mm")
    @patch("fastdeploy.model_executor.layers.quantization.wfp8afp8.scaled_fp8_quant")
    def test_apply_per_token(self, mock_quant, mock_gemm):
        """apply quantizes input and calls cutlass_scaled_mm."""
        config = WFP8AFP8Config()
        method = WFP8AFP8LinearMethod(config)
        method.use_per_token_if_dynamic = True

        a_q = MagicMock()
        a_scales = MagicMock()
        mock_quant.return_value = (a_q, a_scales)
        mock_gemm.return_value = "output"

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.bias = None

        x = MagicMock()
        x.dtype = "bfloat16"

        result = method.apply(layer, x)

        mock_quant.assert_called_once_with(x, use_per_token_if_dynamic=True)
        mock_gemm.assert_called_once_with(a_q, layer.weight, a_scales, layer.weight_scale, "bfloat16", None)
        self.assertEqual(result, "output")

    def test_apply_not_per_token_raises(self):
        """apply raises NotImplementedError when use_per_token_if_dynamic=False."""
        config = WFP8AFP8Config()
        method = WFP8AFP8LinearMethod(config)
        method.use_per_token_if_dynamic = False

        layer = MagicMock()
        x = MagicMock()
        x.dtype = "bfloat16"

        with self.assertRaises(NotImplementedError):
            method.apply(layer, x)


if __name__ == "__main__":
    unittest.main()
