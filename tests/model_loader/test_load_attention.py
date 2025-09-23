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

import unittest
from unittest.mock import Mock

import numpy as np
import paddle

from fastdeploy.config import CacheConfig, FDConfig, ModelConfig, ParallelConfig
from fastdeploy.model_executor.layers.attention.attention import Attention


class MockQuantMethod:
    """Mock quantization method for testing."""

    def __init__(self, has_zero_point=False, max_bound=1.0):
        self.cache_quant_config = Mock()
        self.cache_quant_config.has_zero_point = has_zero_point
        self.cache_quant_config.max_bound = max_bound
        self.create_weights_called = False
        self.create_weights_args = None

    def create_weights(self, layer, weight_loader):
        self.create_weights_called = True
        self.create_weights_args = (layer, weight_loader)

    def process_loaded_weights(self, layer, state_dict):
        pass


class TestAttentionInitWeight(unittest.TestCase):
    """Test cases for Attention.init_weight method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.model_config = Mock(spec=ModelConfig)
        self.model_config.num_attention_heads = 32
        self.model_config.head_dim = 128
        self.model_config.num_key_value_heads = 8
        self.model_config.model = "test_model"
        self.model_config.num_hidden_layers = 12

        self.parallel_config = Mock(spec=ParallelConfig)
        self.parallel_config.tensor_parallel_size = 1
        self.parallel_config.tensor_parallel_rank = 0
        self.parallel_config.max_num_seqs = 8

        self.cache_config = Mock(spec=CacheConfig)

        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.model_config = self.model_config
        self.fd_config.parallel_config = self.parallel_config
        self.fd_config.cache_config = self.cache_config
        self.fd_config.quant_config = None
        self.fd_config.plas_attention_config = None

    def test_init_weight_without_quantization(self):
        """Test init_weight without quantization."""
        # Test case 1: No quantization, no qk_norm
        attention = Attention(fd_config=self.fd_config, layer_id=0, use_qk_norm=False)

        # Check that q_norm_weight and k_norm_weight are not created
        self.assertFalse(hasattr(attention, "q_norm_weight"))
        self.assertFalse(hasattr(attention, "k_norm_weight"))

    def test_init_weight_with_qk_norm(self):
        """Test init_weight with qk_norm enabled."""
        # Test case 2: No quantization, with qk_norm
        attention = Attention(fd_config=self.fd_config, layer_id=0, use_qk_norm=True, rms_norm_eps=1e-6)

        # Check that q_norm_weight and k_norm_weight are created
        self.assertTrue(hasattr(attention, "q_norm_weight"))
        self.assertTrue(hasattr(attention, "k_norm_weight"))

        # Check parameter shapes
        self.assertEqual(attention.q_norm_weight.shape, [attention.qk_head_dim])
        self.assertEqual(attention.k_norm_weight.shape, [attention.qk_head_dim])

        # Check parameter dtype
        self.assertEqual(attention.q_norm_weight.dtype, paddle.float32)
        self.assertEqual(attention.k_norm_weight.dtype, paddle.float32)

        # Check initial values (should be zeros)
        np.testing.assert_array_equal(
            attention.q_norm_weight.numpy(), np.zeros(attention.qk_head_dim, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            attention.k_norm_weight.numpy(), np.zeros(attention.qk_head_dim, dtype=np.float32)
        )

    def test_init_weight_with_quantization(self):
        """Test init_weight with quantization enabled."""
        # Test case 3: With quantization
        mock_quant_method = MockQuantMethod()
        self.fd_config.quant_config = Mock()
        self.fd_config.quant_config.get_quant_method = Mock(return_value=mock_quant_method)

        attention = Attention(fd_config=self.fd_config, layer_id=0, use_qk_norm=False)

        # Check that quant_method.create_weights was called
        self.assertTrue(mock_quant_method.create_weights_called)
        self.assertEqual(mock_quant_method.create_weights_args[0], attention)
        # Check that weight_loader is passed correctly
        self.assertIsNotNone(mock_quant_method.create_weights_args[1])


class TestAttentionWeightLoader(unittest.TestCase):
    """Test cases for Attention.weight_loader method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.model_config = Mock(spec=ModelConfig)
        self.model_config.num_attention_heads = 32
        self.model_config.head_dim = 128
        self.model_config.num_key_value_heads = 8
        self.model_config.model = "test_model"
        self.model_config.num_hidden_layers = 12

        self.parallel_config = Mock(spec=ParallelConfig)
        self.parallel_config.tensor_parallel_size = 1
        self.parallel_config.tensor_parallel_rank = 0
        self.parallel_config.max_num_seqs = 8

        self.cache_config = Mock(spec=CacheConfig)

        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.model_config = self.model_config
        self.fd_config.parallel_config = self.parallel_config
        self.fd_config.cache_config = self.cache_config
        self.fd_config.plas_attention_config = None

        # Create mock quant method
        self.mock_quant_method = MockQuantMethod()
        self.fd_config.quant_config = Mock()
        self.fd_config.quant_config.get_quant_method = Mock(return_value=self.mock_quant_method)

        # Create attention layer
        self.attention = Attention(fd_config=self.fd_config, layer_id=0, use_qk_norm=False)

    def test_weight_loader_without_zero_point(self):
        """Test weight_loader without zero point."""
        # Test case 1: No zero point
        mock_quant_method = MockQuantMethod(has_zero_point=False, max_bound=8.0)
        self.attention.quant_method = mock_quant_method

        # Create mock parameter
        param = paddle.zeros([10], dtype=paddle.float32)

        # Create mock loaded weight
        loaded_weight = np.array([2.0, 4.0, 8.0, 1.0, 0.5, 2.0, 4.0, 8.0, 1.0, 0.5])

        # Call weight_loader
        self.attention.weight_loader(param, loaded_weight)

        # Check that the parameter is updated correctly
        expected_value = 8.0 / loaded_weight
        np.testing.assert_array_almost_equal(param.numpy(), expected_value.astype(np.float32))

    def test_weight_loader_with_zero_point(self):
        """Test weight_loader with zero point."""
        # Test case 2: With zero point
        mock_quant_method = MockQuantMethod(has_zero_point=True, max_bound=8.0)
        self.attention.quant_method = mock_quant_method

        # Create mock parameter
        param = paddle.zeros([10], dtype=paddle.float32)

        # Create mock loaded weight
        loaded_weight = np.array([2.0, 4.0, 8.0, 1.0, 0.5, 2.0, 4.0, 8.0, 1.0, 0.5])

        # Call weight_loader
        self.attention.weight_loader(param, loaded_weight)

        # Check that the parameter is updated correctly
        expected_value = 1.0 / loaded_weight
        np.testing.assert_array_almost_equal(param.numpy(), expected_value.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
