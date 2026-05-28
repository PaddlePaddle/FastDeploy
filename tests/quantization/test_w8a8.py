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

import numpy as np
import paddle

from fastdeploy.model_executor.layers.quantization.w8a8 import (
    SmoothQuantLinearMethod,
    W8A8Config,
    W8A8LinearMethod,
)


class TestW8A8Config(unittest.TestCase):
    """Test W8A8Config class."""

    def test_init(self):
        """__init__ sets all attributes correctly."""
        weight_scale_dict = {"layer.weight_scale": np.array([1.0])}
        act_scale_dict = {"layer.activation_scale": np.array([0.5])}
        config = W8A8Config(weight_scale_dict, act_scale_dict, True, False)
        self.assertEqual(config.weight_scale_dict, weight_scale_dict)
        self.assertEqual(config.act_scale_dict, act_scale_dict)
        self.assertTrue(config.use_gemm_dequant)
        self.assertFalse(config.use_smooth_quant)
        self.assertEqual(config.quant_max_bound, 127)
        self.assertEqual(config.quant_min_bound, -127)
        self.assertEqual(config.quant_round_type, 0)

    def test_name(self):
        """name() returns 'w8a8'."""
        config = W8A8Config({}, {}, False, False)
        self.assertEqual(config.name(), "w8a8")

    def test_from_config(self):
        """from_config extracts keys from config dict."""
        cfg_dict = {
            "weight_scale_dict": {"k": 1.0},
            "act_scale_dict": {"a": 0.5},
            "use_gemm_dequant": True,
            "use_smooth_quant": False,
        }
        # Note: source code from_config doesn't pass use_smooth_quant,
        # so we test with a patched cls to verify key extraction logic
        with patch.object(W8A8Config, "__init__", return_value=None) as mock_init:
            W8A8Config.from_config(cfg_dict)
            mock_init.assert_called_once_with({"k": 1.0}, {"a": 0.5}, True)

    def test_get_quant_method(self):
        """get_quant_method returns W8A8LinearMethod instance."""
        config = W8A8Config({}, {}, False, False)
        method = config.get_quant_method(None)
        self.assertIsInstance(method, W8A8LinearMethod)
        self.assertIs(method.quant_config, config)


class TestW8A8LinearMethodInit(unittest.TestCase):
    """Test W8A8LinearMethod.__init__."""

    def test_init(self):
        """__init__ stores config and creates smooth_quant_method."""
        config = W8A8Config({}, {}, False, False)
        method = W8A8LinearMethod(config)
        self.assertIs(method.quant_config, config)
        self.assertIsInstance(method.smooth_quant_method, SmoothQuantLinearMethod)


class TestW8A8LinearMethodCreateWeights(unittest.TestCase):
    """Test W8A8LinearMethod.create_weights."""

    def _make_layer(self, prefix="model.layer0", embed_dim=64, weight_shape=None):
        layer = MagicMock()
        layer.prefix = prefix
        layer.embed_dim = embed_dim
        layer.weight_shape = weight_shape or [64, 128]
        layer._dtype = "float16"
        layer.create_parameter.return_value = MagicMock()
        return layer

    @patch("fastdeploy.model_executor.layers.quantization.w8a8.convert_to_npu_dequant_scale")
    def test_create_weights_with_scales(self, mock_convert):
        """create_weights creates weight and linear_out_scale when scales exist."""
        mock_convert.side_effect = lambda x: x

        weight_scale = np.array([2.0])
        act_scale = np.array([0.5])
        config = W8A8Config(
            weight_scale_dict={"model.layer0.weight_scale": weight_scale},
            act_scale_dict={"model.layer0.activation_scale": act_scale},
            use_gemm_dequant=True,
            use_smooth_quant=False,
        )
        method = W8A8LinearMethod(config)
        layer = self._make_layer()

        method.create_weights(layer)

        # weight_shape reversed
        self.assertEqual(layer.weight_shape, [128, 64])
        self.assertEqual(layer.weight_dtype, "int8")
        self.assertFalse(method.skip_quant)
        # create_parameter called twice: weight + linear_out_scale
        self.assertEqual(layer.create_parameter.call_count, 2)
        mock_convert.assert_called_once()

    def test_create_weights_skip_quant_no_weight_scale(self):
        """create_weights sets skip_quant=True when weight_scale missing."""
        config = W8A8Config(
            weight_scale_dict={},
            act_scale_dict={"model.layer0.activation_scale": np.array([0.5])},
            use_gemm_dequant=False,
            use_smooth_quant=False,
        )
        method = W8A8LinearMethod(config)
        layer = self._make_layer()

        method.create_weights(layer)

        self.assertTrue(method.skip_quant)
        layer.create_parameter.assert_not_called()

    def test_create_weights_skip_quant_no_act_scale(self):
        """create_weights sets skip_quant=True when act_scale missing."""
        config = W8A8Config(
            weight_scale_dict={"model.layer0.weight_scale": np.array([1.0])},
            act_scale_dict={},
            use_gemm_dequant=False,
            use_smooth_quant=False,
        )
        method = W8A8LinearMethod(config)
        layer = self._make_layer()

        method.create_weights(layer)

        self.assertTrue(method.skip_quant)
        layer.create_parameter.assert_not_called()

    @patch("fastdeploy.model_executor.layers.quantization.w8a8.convert_to_npu_dequant_scale")
    def test_create_weights_with_smooth_quant(self, mock_convert):
        """create_weights calls smooth_quant_method.create_weights when use_smooth_quant=True."""
        mock_convert.side_effect = lambda x: x

        config = W8A8Config(
            weight_scale_dict={"model.layer0.weight_scale": np.array([1.0])},
            act_scale_dict={"model.layer0.activation_scale": np.array([1.0])},
            use_gemm_dequant=False,
            use_smooth_quant=True,
        )
        method = W8A8LinearMethod(config)
        method.smooth_quant_method = MagicMock()
        layer = self._make_layer()

        method.create_weights(layer)

        method.smooth_quant_method.create_weights.assert_called_once_with(layer)


class TestW8A8LinearMethodProcessLoadedWeights(unittest.TestCase):
    """Test W8A8LinearMethod.process_loaded_weights."""

    def test_process_loaded_weights_skip_quant(self):
        """process_loaded_weights handles skip_quant path."""
        config = W8A8Config({}, {}, False, False)
        method = W8A8LinearMethod(config)
        method.skip_quant = True

        layer = MagicMock()
        layer.prefix = "model.layer0"
        layer._dtype = "float16"
        layer.weight = MagicMock()

        weights = paddle.ones([4, 8], dtype="float32")

        method.process_loaded_weights(layer, weights)

        layer.weight.set_value.assert_called_once()
        # Should cast to layer._dtype
        set_value_arg = layer.weight.set_value.call_args[0][0]
        self.assertEqual(set_value_arg.dtype, paddle.float16)

    def test_process_loaded_weights_quantized(self):
        """process_loaded_weights transposes and casts to int8 when not skip_quant."""
        config = W8A8Config({}, {}, False, False)
        method = W8A8LinearMethod(config)
        method.skip_quant = False

        layer = MagicMock()
        layer.prefix = "model.layer0"
        layer.weight = MagicMock()

        weights = paddle.ones([4, 8], dtype="float32")

        with patch.object(config, "use_smooth_quant", False):
            method.process_loaded_weights(layer, weights)

        layer.weight.set_value.assert_called_once()
        set_value_arg = layer.weight.set_value.call_args[0][0]
        self.assertEqual(set_value_arg.dtype, paddle.int8)
        self.assertEqual(list(set_value_arg.shape), [8, 4])

    def test_process_loaded_weights_with_smooth_quant(self):
        """process_loaded_weights calls smooth_quant when enabled."""
        config = W8A8Config({}, {}, False, True)
        method = W8A8LinearMethod(config)
        method.skip_quant = False
        method.smooth_quant_method = MagicMock()

        layer = MagicMock()
        layer.prefix = "model.layer0"
        layer.weight = MagicMock()

        weights = paddle.ones([4, 8], dtype="float32")

        method.process_loaded_weights(layer, weights)

        method.smooth_quant_method.process_loaded_weights.assert_called_once_with(layer, weights)


class TestW8A8LinearMethodApply(unittest.TestCase):
    """Test W8A8LinearMethod.apply."""

    def test_apply_skip_quant(self):
        """apply does paddle.matmul when skip_quant=True."""
        config = W8A8Config({}, {}, False, False)
        method = W8A8LinearMethod(config)
        method.skip_quant = True

        layer = MagicMock()
        layer.weight = paddle.ones([8, 4], dtype="float16")

        x = paddle.ones([2, 4], dtype="float16")
        result = method.apply(layer, x)

        self.assertEqual(list(result.shape), [2, 8])

    @patch("fastdeploy.model_executor.ops.gpu.gemm_dequant")
    def test_apply_gemm_dequant(self, mock_gemm_dequant):
        """apply uses gemm_dequant when use_gemm_dequant=True."""
        mock_gemm_dequant.return_value = paddle.zeros([2, 8], dtype="float16")

        config = W8A8Config({}, {}, True, False)
        method = W8A8LinearMethod(config)
        method.skip_quant = False

        layer = MagicMock()
        layer.weight = paddle.ones([8, 4], dtype="int8")
        layer.linear_out_scale = paddle.ones([8], dtype="float32")
        layer._dtype = "float16"

        x = paddle.ones([2, 4], dtype="int8")
        result = method.apply(layer, x)

        mock_gemm_dequant.assert_called_once_with(x, layer.weight, layer.linear_out_scale, "float16")
        self.assertEqual(list(result.shape), [2, 8])

    @patch("fastdeploy.model_executor.ops.gpu.dequant_int8")
    def test_apply_dequant_int8(self, mock_dequant_int8):
        """apply uses matmul + dequant_int8 when use_gemm_dequant=False."""
        mock_dequant_int8.return_value = paddle.zeros([2, 8], dtype="float16")

        config = W8A8Config({}, {}, False, False)
        method = W8A8LinearMethod(config)
        method.skip_quant = False

        layer = MagicMock()
        layer.weight = paddle.ones([8, 4], dtype="int8")
        layer.linear_out_scale = paddle.ones([8], dtype="float32")
        layer._dtype = "float16"

        x = paddle.ones([2, 4], dtype="int8")
        result = method.apply(layer, x)

        mock_dequant_int8.assert_called_once()
        self.assertEqual(list(result.shape), [2, 8])


class TestSmoothQuantLinearMethodInit(unittest.TestCase):
    """Test SmoothQuantLinearMethod.__init__."""

    def test_init(self):
        """__init__ stores quant_config."""
        config = MagicMock()
        method = SmoothQuantLinearMethod(config)
        self.assertIs(method.quant_config, config)


class TestSmoothQuantLinearMethodCreateWeights(unittest.TestCase):
    """Test SmoothQuantLinearMethod.create_weights."""

    def test_create_weights(self):
        """create_weights creates linear_shift and linear_smooth parameters."""
        config = MagicMock()
        method = SmoothQuantLinearMethod(config)
        # SmoothQuantLinearMethod calls self.create_parameter (line 147)
        # This is inherited from QuantMethodBase -> ABC, so it doesn't exist
        # We need to mock it
        method.create_parameter = MagicMock(return_value=MagicMock())

        layer = MagicMock()
        layer.output_size = 256
        layer._dtype = "float16"

        method.create_weights(layer)

        # self.create_parameter called once for linear_shift
        method.create_parameter.assert_called_once_with(
            shape=[256],
            dtype="float16",
            is_bias=False,
        )
        # layer.create_parameter called once for linear_smooth
        layer.create_parameter.assert_called_once_with(
            shape=[256],
            dtype="float16",
            is_bias=False,
        )


class TestSmoothQuantLinearMethodProcessLoadedWeights(unittest.TestCase):
    """Test SmoothQuantLinearMethod.process_loaded_weights."""

    @patch("fastdeploy.model_executor.layers.quantization.w8a8.get_tensor")
    def test_process_loaded_weights_with_keys_present(self, mock_get_tensor):
        """process_loaded_weights loads shift and smooth from state_dict."""
        mock_get_tensor.return_value = paddle.ones([64], dtype="float32")

        config = MagicMock()
        method = SmoothQuantLinearMethod(config)

        layer = MagicMock()
        layer.shift_key = "model.shift"
        layer.smooth_key = "model.smooth"
        layer.state_dict = {
            "model.shift": paddle.ones([64], dtype="float32"),
            "model.smooth": paddle.ones([64], dtype="float32"),
        }
        layer.linear_shift = MagicMock()
        layer.linear_smooth = MagicMock()

        weights = paddle.ones([64, 64], dtype="float32")

        with patch("paddle.get_default_dtype", return_value="float32"):
            method.process_loaded_weights(layer, weights)

        layer.linear_shift.set_value.assert_called_once()
        layer.linear_smooth.set_value.assert_called_once()
        self.assertEqual(mock_get_tensor.call_count, 2)

    def test_process_loaded_weights_keys_missing(self):
        """process_loaded_weights uses zeros/ones when keys not in state_dict."""
        config = MagicMock()
        method = SmoothQuantLinearMethod(config)

        layer = MagicMock()
        layer.shift_key = "model.shift"
        layer.smooth_key = "model.smooth"
        layer.state_dict = {}  # No keys
        layer.linear_shift_shape = [64]
        layer.linear_smooth_shape = 64
        layer.linear_shift = MagicMock()
        layer.linear_smooth = MagicMock()

        weights = paddle.ones([64, 64], dtype="float32")

        with patch("paddle.get_default_dtype", return_value="float32"):
            method.process_loaded_weights(layer, weights)

        layer.linear_shift.set_value.assert_called_once()
        layer.linear_smooth.set_value.assert_called_once()
        # Verify zeros for shift
        shift_val = layer.linear_shift.set_value.call_args[0][0]
        self.assertTrue(paddle.all(shift_val == 0).item())
        # Verify ones for smooth
        smooth_val = layer.linear_smooth.set_value.call_args[0][0]
        self.assertTrue(paddle.all(smooth_val == 1).item())


class TestSmoothQuantLinearMethodApply(unittest.TestCase):
    """Test SmoothQuantLinearMethod.apply."""

    def test_apply_returns_none(self):
        """apply is a no-op (returns None)."""
        config = MagicMock()
        method = SmoothQuantLinearMethod(config)
        result = method.apply(None, None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
