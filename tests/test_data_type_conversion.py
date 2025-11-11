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
Unit tests for data type conversion functionality in FastDeploy.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle

from fastdeploy.model_executor.layers.quantization.kv_cache import (
    KVCacheMethodBase,
    KvCacheQuantConfig,
)
from fastdeploy.model_executor.layers.quantization.tensor_wise_fp8 import (
    TensorWiseFP8Config,
    TensorWiseFP8LinearMethod,
)
from fastdeploy.model_executor.layers.quantization.wfp8afp8 import (
    WFP8AFP8Config,
    WFP8AFP8LinearMethod,
)
from fastdeploy.model_executor.layers.utils import (
    get_tensor,
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)
from fastdeploy.model_executor.utils import default_weight_loader, temporary_dtype
from fastdeploy.utils import is_list_of, optional_type, parse_type


class TestTypeParsing(unittest.TestCase):
    """Test type parsing and conversion utilities."""

    def test_parse_type_int(self):
        """Test integer type parsing."""
        parser = parse_type(int)
        self.assertEqual(parser("123"), 123)
        self.assertEqual(parser("-456"), -456)

        with self.assertRaises(ValueError):
            parser("abc")

    def test_parse_type_float(self):
        """Test float type parsing."""
        parser = parse_type(float)
        self.assertEqual(parser("3.14"), 3.14)
        self.assertEqual(parser("-2.5"), -2.5)

        with self.assertRaises(ValueError):
            parser("abc")

    def test_parse_type_bool(self):
        """Test boolean type parsing."""
        parser = parse_type(bool)
        self.assertEqual(parser("True"), True)
        self.assertEqual(parser("False"), False)
        self.assertEqual(parser("1"), True)
        self.assertEqual(parser("0"), False)

    def test_optional_type(self):
        """Test optional type parsing."""
        parser = optional_type(int)
        self.assertEqual(parser("123"), 123)
        self.assertEqual(parser("None"), None)
        self.assertEqual(parser(""), None)
        self.assertEqual(parser("0"), 0)

    def test_is_list_of(self):
        """Test list type checking."""
        # Test "first" mode
        self.assertTrue(is_list_of([1, 2, 3], int, check="first"))
        self.assertTrue(is_list_of([], int, check="first"))
        self.assertFalse(is_list_of(["a", 2, 3], int, check="first"))

        # Test "all" mode
        self.assertTrue(is_list_of([1, 2, 3], int, check="all"))
        self.assertFalse(is_list_of([1, "a", 3], int, check="all"))

        # Test non-list input
        self.assertFalse(is_list_of("not a list", int))


class TestTensorUtils(unittest.TestCase):
    """Test tensor utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=paddle.float32)
        self.test_np_array = np.array([1, 2, 3, 4], dtype=np.float32)

    def test_get_tensor_paddle_tensor(self):
        """Test get_tensor with paddle tensor input."""
        result = get_tensor(self.test_tensor)
        self.assertIsInstance(result, paddle.Tensor)
        paddle.testing.assert_allclose(result, self.test_tensor)

    def test_get_tensor_numpy_array(self):
        """Test get_tensor with numpy array input."""
        result = get_tensor(self.test_np_array)
        self.assertIsInstance(result, paddle.Tensor)
        expected = paddle.to_tensor(self.test_np_array)
        paddle.testing.assert_allclose(result, expected)

    def test_get_tensor_string_path(self):
        """Test get_tensor with string path (mocked)."""
        with patch("fastdeploy.model_executor.layers.utils.load_reordered_experts") as mock_load:
            mock_tensor = paddle.to_tensor([1, 2, 3])
            mock_load.return_value = mock_tensor

            result = get_tensor("test_path", model_path="test_model")
            mock_load.assert_called_once_with("test_model", "test_path")
            paddle.testing.assert_allclose(result, mock_tensor)

    def test_temporary_dtype(self):
        """Test temporary dtype context manager."""
        original_dtype = paddle.get_default_dtype()

        with temporary_dtype("float32"):
            self.assertEqual(paddle.get_default_dtype(), paddle.float32)

        # Should revert back to original dtype
        self.assertEqual(paddle.get_default_dtype(), original_dtype)

    def test_temporary_dtype_no_change(self):
        """Test temporary dtype with None input."""
        original_dtype = paddle.get_default_dtype()

        with temporary_dtype(None):
            self.assertEqual(paddle.get_default_dtype(), original_dtype)

        self.assertEqual(paddle.get_default_dtype(), original_dtype)


class TestFP8Conversion(unittest.TestCase):
    """Test FP8 conversion functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_tensor = paddle.randn([128, 256], dtype=paddle.float32)
        self.test_token_tensor = paddle.randn([10, 64], dtype=paddle.float32)

    def test_per_token_cast_to_fp8(self):
        """Test per-token FP8 conversion."""
        result_tensor, result_scale = per_token_cast_to_fp8(self.test_token_tensor)

        # Check output types
        self.assertIsInstance(result_tensor, paddle.Tensor)
        self.assertIsInstance(result_scale, paddle.Tensor)

        # Check FP8 dtype
        self.assertEqual(result_tensor.dtype, paddle.float8_e4m3fn)

        # Check shape consistency
        self.assertEqual(result_tensor.shape, self.test_token_tensor.shape)

        # Check scale shape
        self.assertEqual(result_scale.shape, [self.test_token_tensor.shape[0]])

        # Check FP8 range constraints
        self.assertTrue(paddle.all(result_tensor >= -448))
        self.assertTrue(paddle.all(result_tensor <= 448))

    def test_per_block_cast_to_fp8(self):
        """Test per-block FP8 conversion."""
        block_size = [64, 64]
        result_tensor, result_scale = per_block_cast_to_fp8(self.test_tensor, block_size)

        # Check output types
        self.assertIsInstance(result_tensor, paddle.Tensor)
        self.assertIsInstance(result_scale, paddle.Tensor)

        # Check FP8 dtype
        self.assertEqual(result_tensor.dtype, paddle.float8_e4m3fn)

        # Check scale shape
        expected_scale_shape = [
            (self.test_tensor.shape[0] + block_size[0] - 1) // block_size[0],
            (self.test_tensor.shape[1] + block_size[1] - 1) // block_size[1],
        ]
        self.assertEqual(result_scale.shape, expected_scale_shape)

        # Check FP8 range constraints
        self.assertTrue(paddle.all(result_tensor >= -448))
        self.assertTrue(paddle.all(result_tensor <= 448))


class TestQuantizationTypeConversion(unittest.TestCase):
    """Test quantization-related type conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_layer = Mock()
        self.mock_layer.weight_shape = [1024, 1024]
        self.mock_layer.weight_dtype = paddle.float32

    def test_kv_cache_quant_config_type_conversion(self):
        """Test KV cache quant config type handling."""
        config = KvCacheQuantConfig("int8", True, False)

        # Test config creation and type attributes
        self.assertEqual(config.quant_type, "int8")
        self.assertTrue(config.is_channel_wise)
        self.assertFalse(config.has_zero_point)
        self.assertEqual(config.max_bound, 127.0)

    def test_kv_cache_method_base_type_conversion(self):
        """Test KV cache method base type conversion."""
        config = KvCacheQuantConfig("int8", True, False)
        method = KVCacheMethodBase(config)

        # Mock layer and state_dict for testing
        self.mock_layer.cache_k_zp = Mock()
        self.mock_layer.cache_v_zp = Mock()
        self.mock_layer.cache_k_scale = Mock()
        self.mock_layer.cache_v_scale = Mock()

        state_dict = {
            "cache_k_zp": paddle.to_tensor([1.0, 2.0], dtype=paddle.float32),
            "cache_v_zp": paddle.to_tensor([3.0, 4.0], dtype=paddle.float32),
            "cache_k_scale": paddle.to_tensor([5.0, 6.0], dtype=paddle.float32),
            "cache_v_scale": paddle.to_tensor([7.0, 8.0], dtype=paddle.float32),
        }

        # Test type casting in load_zp and load_scale
        method.load_zp(self.mock_layer, state_dict)
        method.load_scale(self.mock_layer, state_dict)

        # Verify the type casting was applied
        self.assertEqual(self.mock_layer.cache_k_zp.set_value.call_count, 1)
        self.assertEqual(self.mock_layer.cache_v_zp.set_value.call_count, 1)
        self.assertEqual(self.mock_layer.cache_k_scale.set_value.call_count, 1)
        self.assertEqual(self.mock_layer.cache_v_scale.set_value.call_count, 1)


class TestWeightOnlyQuantizationTypeConversion(unittest.TestCase):
    """Test weight-only quantization type conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_layer = Mock()
        self.mock_layer.weight_shape = [1024, 1024]
        self.mock_layer._dtype = paddle.float32
        self.mock_layer.fd_config = Mock()
        self.mock_layer.fd_config.load_config = Mock()
        self.mock_layer.fd_config.load_config.load_choices = "default_v0"

    def test_tensor_wise_fp8_config_type_conversion(self):
        """Test tensor-wise FP8 config type handling."""
        config = TensorWiseFP8Config()

        # Test config properties
        self.assertEqual(config.name(), "tensor_wise_fp8")
        self.assertEqual(config.quant_max_bound, 448)
        self.assertEqual(config.quant_min_bound, -448)

        # Test method creation
        method = config.get_quant_method(Mock())
        self.assertIsInstance(method, TensorWiseFP8LinearMethod)

    def test_wfp8afp8_config_type_conversion(self):
        """Test WFP8AFP8 config type handling."""
        config = WFP8AFP8Config()

        # Test config properties
        self.assertEqual(config.name(), "wfp8afp8")
        self.assertEqual(config.quant_max_bound, 448)
        self.assertEqual(config.quant_min_bound, -448)

        # Test method creation
        method = config.get_quant_method(Mock())
        self.assertIsInstance(method, WFP8AFP8LinearMethod)


class TestWeightLoaderTypeConversion(unittest.TestCase):
    """Test weight loader type conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.fd_config = Mock()
        self.fd_config.parallel_config = Mock()
        self.fd_config.parallel_config.tensor_parallel_size = 1
        self.weight_loader = default_weight_loader(self.fd_config)

        self.mock_param = Mock()
        self.mock_param.dtype = paddle.float32
        self.mock_param.shape = [1024, 1024]
        self.mock_param.output_dim = None
        self.mock_param.weight_need_transpose = False
        self.mock_param.copy_ = Mock()

    def test_weight_loader_same_dtype(self):
        """Test weight loader with same dtype."""
        loaded_weight = paddle.randn([1024, 1024], dtype=paddle.float32)

        self.weight_loader(self.mock_param, loaded_weight)

        # Verify tensor was copied without modification
        self.mock_param.copy_.assert_called_once()
        call_args = self.mock_param.copy_.call_args
        paddle.testing.assert_allclose(call_args[0][0], loaded_weight)

    def test_weight_loader_dtype_conversion(self):
        """Test weight loader with dtype conversion."""
        loaded_weight = paddle.randn([1024, 1024], dtype=paddle.int8)
        self.mock_param.dtype = paddle.float32

        self.weight_loader(self.mock_param, loaded_weight)

        # Verify dtype conversion was applied
        self.mock_param.copy_.assert_called_once()
        call_args = self.mock_param.copy_.call_args
        self.assertEqual(call_args[0][0].dtype, paddle.float32)

    def test_weight_loader_fp8_conversion(self):
        """Test weight loader with FP8 conversion."""
        loaded_weight = paddle.randint(-127, 127, [1024, 1024], dtype=paddle.int8)
        self.mock_param.dtype = paddle.float8_e4m3fn

        self.weight_loader(self.mock_param, loaded_weight)

        # Verify FP8 conversion using view
        self.mock_param.copy_.assert_called_once()
        call_args = self.mock_param.copy_.call_args
        self.assertEqual(call_args[0][0].dtype, paddle.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
