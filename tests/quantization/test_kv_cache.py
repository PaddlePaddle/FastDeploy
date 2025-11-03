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

import sys
import unittest

import numpy as np
import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.kv_cache import (
    KVCacheMethodBase,
    KvCacheQuantConfig,
    KvCacheQuantzationTypes,
)

sys.path.append("../")
from tests.utils import get_default_test_fd_config


class MockLayer(nn.Layer):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.fd_config = get_default_test_fd_config()
        self.fd_config.model_config.num_key_value_heads = 1
        self.head_dim = 1
        self.kv_num_heads = 1
        self.prefix = "mock_layer"
        self.cache_k_scale = None
        self.cache_v_scale = None
        self.cache_k_out_scale = None
        self.cache_v_out_scale = None
        self.cache_k_zp = None
        self.cache_v_zp = None


class TestKVCacheMethodBase(unittest.TestCase):
    def setUp(self):
        self.layer = MockLayer()

    def test_create_weights_int8(self):
        # Test INT8 without zero point
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT8, is_channel_wise=False, has_zero_point=False
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        self.assertEqual(self.layer.cache_quant_type_str, "cache_int8")
        self.assertEqual(self.layer.quant_max_bound, 127.0)
        self.assertEqual(self.layer.quant_min_bound, -127.0)
        self.assertIsNotNone(self.layer.cache_k_scale)
        self.assertIsNotNone(self.layer.cache_v_scale)
        self.assertIsNotNone(self.layer.cache_k_out_scale)
        self.assertIsNotNone(self.layer.cache_v_out_scale)
        self.assertIsNone(self.layer.cache_k_zp)
        self.assertIsNone(self.layer.cache_v_zp)
        self.assertEqual(self.layer.cache_k_scale.shape, [1])

    def test_create_weights_int8_channel_wise(self):
        # Test INT8 with channel wise
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT8, is_channel_wise=True, has_zero_point=False
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        self.assertEqual(self.layer.cache_k_scale.shape, [1])

    def test_create_weights_int4_zp(self):
        # Test INT4 with zero point
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT4_ZP, is_channel_wise=False, has_zero_point=True
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        self.assertEqual(self.layer.cache_quant_type_str, "cache_int4_zp")
        self.assertEqual(self.layer.quant_max_bound, 7.0)
        self.assertEqual(self.layer.quant_min_bound, -7.0)
        self.assertIsNotNone(self.layer.cache_k_zp)
        self.assertIsNotNone(self.layer.cache_v_zp)

    def test_process_loaded_weights_int8(self):
        # Test process INT8 weights
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT8, is_channel_wise=False, has_zero_point=False
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        state_dict = {
            "mock_layer.cachek_matmul.activation_scale": np.array([2.0], dtype=np.float32),
            "mock_layer.cachev_matmul.activation_scale": np.array([3.0], dtype=np.float32),
        }
        method.process_loaded_weights(self.layer, state_dict)

        self.assertAlmostEqual(self.layer.cache_k_scale.numpy()[0], 127.0 / 2.0, places=3)
        self.assertAlmostEqual(self.layer.cache_v_scale.numpy()[0], 127.0 / 3.0, places=3)
        self.assertAlmostEqual(self.layer.cache_k_out_scale.numpy()[0], 2.0 / 127.0, places=3)
        self.assertAlmostEqual(self.layer.cache_v_out_scale.numpy()[0], 3.0 / 127.0, places=3)

    def test_process_loaded_weights_int4_zp(self):
        # Test process INT4 with zero point weights
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT4_ZP, is_channel_wise=False, has_zero_point=True
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        state_dict = {
            "mock_layer.cachek_matmul.activation_scale": np.array([2.0], dtype=np.float32),
            "mock_layer.cachev_matmul.activation_scale": np.array([3.0], dtype=np.float32),
            "mock_layer.cachek_matmul.activation_zero_point": np.array([1.0], dtype=np.float32),
            "mock_layer.cachev_matmul.activation_zero_point": np.array([2.0], dtype=np.float32),
        }
        method.process_loaded_weights(self.layer, state_dict)

        self.assertAlmostEqual(self.layer.cache_k_scale.numpy()[0], 1.0 / 2.0, places=3)
        self.assertAlmostEqual(self.layer.cache_v_scale.numpy()[0], 1.0 / 3.0, places=3)
        self.assertAlmostEqual(self.layer.cache_k_out_scale.numpy()[0], 2.0)
        self.assertAlmostEqual(self.layer.cache_v_out_scale.numpy()[0], 3.0)
        self.assertAlmostEqual(self.layer.cache_k_zp.numpy()[0], 1.0)
        self.assertAlmostEqual(self.layer.cache_v_zp.numpy()[0], 2.0)

    def test_process_weights_after_loading_initialized(self):
        # Test process weights after loading when scale is initialized
        config = KvCacheQuantConfig(
            kv_cache_quant_type=KvCacheQuantzationTypes.INT8, is_channel_wise=False, has_zero_point=False
        )
        method = KVCacheMethodBase(config)
        method.create_weights(self.layer)

        # Simulate initialized scale
        self.layer.cache_k_scale.set_value(paddle.to_tensor([2.0], dtype="float32"))
        self.layer.cache_v_scale.set_value(paddle.to_tensor([3.0], dtype="float32"))

        method.process_weights_after_loading(self.layer)

        self.assertAlmostEqual(self.layer.cache_k_out_scale.numpy()[0], 0.5)
        self.assertAlmostEqual(self.layer.cache_v_out_scale.numpy()[0], 1.0 / 3.0, places=3)


if __name__ == "__main__":
    unittest.main()
