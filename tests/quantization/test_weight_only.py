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

import numpy as np
import paddle

from fastdeploy.model_executor.layers.quantization.weight_only import (
    GPUWeightOnlyLinearMethod,
    WeightOnlyConfig,
)


class DummyLinearLayer:
    """A dummy linear layer"""

    def __init__(self, in_features, out_features, add_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = [out_features, in_features]
        self._dtype = "float16"
        self.add_bias = add_bias
        self.bias = paddle.to_tensor(np.random.randn(out_features).astype("float16")) if add_bias else None

        self.fd_config = type("config", (), {})()
        self.fd_config.load_config = type("load_config", (), {"load_choices": "default_v1"})()

        # Initialize weights (int8) and weight_scale (float16)
        weight_int32 = paddle.randint(low=-128, high=127, shape=self.weight_shape, dtype="int32")
        self.weight = weight_int32.astype("int8")
        self.weight_scale = paddle.ones([self.in_features], dtype="float16")

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        if default_initializer is None:
            return paddle.create_parameter(shape=shape, dtype=dtype)
        else:
            return paddle.create_parameter(shape=shape, dtype=dtype, default_initializer=default_initializer)


class TestWeightOnlyLinearMethodCUDA(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.out_features = 16
        self.layer = DummyLinearLayer(self.in_features, self.out_features)
        self.quant_config = WeightOnlyConfig(algo="weight_only_int8")
        self.method = GPUWeightOnlyLinearMethod(self.quant_config)

    def test_weight_and_scale_shapes(self):
        """Test that weights and scales have the correct shapes."""
        self.assertEqual(list(self.layer.weight.shape), [self.out_features, self.in_features])
        self.assertEqual(list(self.layer.weight_scale.shape), [self.in_features])

    def test_apply_output_shape(self):
        """Test that applying the layer produces output of expected shape."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x)
        self.assertEqual(out.shape, [2, self.out_features])

    def test_apply_nonzero_values(self):
        """Test that the output is not all zeros."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x)
        self.assertFalse(np.allclose(out.numpy(), 0))

    def test_apply_reasonable_range(self):
        """Test that quantized outputs are within a reasonable range."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x).numpy()
        self.assertFalse(np.allclose(out, 0))

        max_expected = 128 * np.max(self.layer.weight_scale.numpy())
        self.assertTrue(np.all(np.abs(out) <= max_expected))


if __name__ == "__main__":
    unittest.main()
