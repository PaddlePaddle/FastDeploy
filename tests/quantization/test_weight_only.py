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


class DummyLinearLayerForWeightOnly:
    def __init__(self, in_features, out_features, add_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = [out_features, in_features]
        self._dtype = "float16"
        self.add_bias = add_bias

        # Optional bias
        self.bias = paddle.to_tensor(np.random.randn(out_features).astype("float16")) if add_bias else None

        # Dummy config for FastDeploy
        self.fd_config = type("config", (), {})()
        self.fd_config.load_config = type("load_config", (), {"load_choices": "default_v1"})()

        # float32 weights
        weight_fp32 = np.random.randn(*self.weight_shape).astype("float32")

        # Per-channel scale
        max_abs = np.max(np.abs(weight_fp32), axis=1, keepdims=True) + 1e-6
        scale = (max_abs / 127.0).astype("float32")

        # Int8 quantization
        weight_int8 = np.clip(np.round(weight_fp32 / scale), -128, 127).astype("int8")

        # Store tensors
        self.weight = paddle.to_tensor(weight_int8, dtype="int8")
        self.weight_scale = paddle.to_tensor(scale.squeeze(-1).astype("float16"))

        # Keep FP32 weight for reference
        self.weight_fp32 = paddle.to_tensor(weight_fp32.astype("float32"))

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        if default_initializer is None:
            return paddle.create_parameter(shape=shape, dtype=dtype)
        else:
            return paddle.create_parameter(shape=shape, dtype=dtype, default_initializer=default_initializer)


class TestGPUWeightOnlyLinearMethod(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.out_features = 16
        self.layer = DummyLinearLayerForWeightOnly(self.in_features, self.out_features)
        self.quant_config = WeightOnlyConfig(algo="weight_only_int8")
        self.method = GPUWeightOnlyLinearMethod(self.quant_config)

    def test_weight_and_scale_shapes(self):
        """Test weight and scale tensor shapes"""
        self.assertEqual(list(self.layer.weight.shape), [self.out_features, self.in_features])
        self.assertEqual(list(self.layer.weight_scale.shape), [self.out_features])

    def test_apply_output_shape(self):
        """Test output shape of apply method"""
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertEqual(out.shape, [2, self.out_features])

    def test_apply_nonzero_output(self):
        """Test that apply output is non-zero"""
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertFalse(np.allclose(out.numpy(), 0))

    def test_apply_numerical_precision(self):
        """Test numerical precision of quantized output"""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))

        # Reference FP32 output
        ref_out = paddle.matmul(
            x.astype("float32"),
            (self.layer.weight.astype("float32") * self.layer.weight_scale.astype("float32")).transpose([1, 0]),
        )
        if self.layer.bias is not None:
            ref_out += self.layer.bias.astype("float32")

        # Manual quantized output
        weight_f32 = self.layer.weight.astype("float32")
        x_f32 = x.astype("float32")
        quant_out = paddle.matmul(x_f32, weight_f32 * self.layer.weight_scale.astype("float32"), transpose_y=True)
        if self.layer.bias is not None:
            quant_out += self.layer.bias.astype("float32")
        quant_out = quant_out.astype("float16")

        np.testing.assert_allclose(ref_out.numpy(), quant_out.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
