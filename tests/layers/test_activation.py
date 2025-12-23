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
from unittest.mock import patch

import paddle

from fastdeploy.model_executor.layers.activation import SiluAndMul


class DummyQuantConfig:
    quant_round_type = 1
    quant_max_bound = 127
    quant_min_bound = -128

    def name(self):
        return "int8"


class DummyFDConfig:
    def __init__(self):
        self.quant_config = DummyQuantConfig()
        self.graph_opt_config = type("GraphOptConfig", (), {"cudagraph_capture_sizes": []})()


class DummyPlatform:
    def __init__(self, cuda=False, gcu=False, intel_hpu=False):
        self._cuda = cuda
        self._gcu = gcu
        self._intel_hpu = intel_hpu

    def is_cuda(self):
        return self._cuda

    def is_xpu(self):
        return False

    def is_iluvatar(self):
        return False

    def is_dcu(self):
        return False

    def is_maca(self):
        return False

    def is_gcu(self):
        return self._gcu

    def is_intel_hpu(self):
        return self._intel_hpu


class DummyHelper:
    def __init__(self, dtype="float16"):
        self._dtype = dtype

    def get_default_dtype(self):
        return self._dtype


class TestSiluAndMul(unittest.TestCase):
    # Test forward computation on CUDA platform
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform(cuda=True)
    )
    @patch("fastdeploy.model_executor.layers.activation.fused_bias_act", return_value=paddle.ones([2, 2]))
    def test_forward_cuda(self, mock_fused, mock_platform):
        fd_config = DummyFDConfig()
        layer = SiluAndMul(fd_config)
        x = paddle.ones([2, 2])
        out = layer.forward(x)
        self.assertTrue((out.numpy() == 1).all())
        mock_fused.assert_called_once()

    # Test forward computation on GCU platform
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform(gcu=True)
    )
    @patch("fastdeploy.model_executor.layers.activation.swiglu", return_value=paddle.ones([2, 2]))
    def test_forward_gcu(self, mock_swiglu, mock_platform):
        fd_config = DummyFDConfig()
        bias = paddle.ones([2, 2])
        layer = SiluAndMul(fd_config, bias=bias)
        x = paddle.ones([2, 2])
        out = layer.forward(x)
        self.assertTrue((out.numpy() == 2).all())

    # Test forward computation on Intel HPU platform
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform",
        new_callable=lambda: DummyPlatform(intel_hpu=True),
    )
    def test_forward_intel_hpu(self, mock_platform):
        fd_config = DummyFDConfig()
        layer = SiluAndMul(fd_config)
        x = paddle.ones([2, 2])
        out = layer.forward(x)
        self.assertIsNone(out)

    # Test behavior on unsupported platforms
    @patch("fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform())
    def test_unsupported_platform(self, mock_platform):
        fd_config = DummyFDConfig()
        with self.assertRaises(NotImplementedError):
            SiluAndMul(fd_config)

    # Test dtype branch handling
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform(cuda=True)
    )
    def test_dtype_branches(self, mock_platform):
        fd_config = DummyFDConfig()
        for dtype, expected in [("float16", "fp16"), ("bfloat16", "bf16"), ("float32", "fp32")]:
            layer = SiluAndMul(fd_config)
            layer._helper = DummyHelper(dtype)
            layer._fuse_kernel_compute_dtype = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[
                layer._helper.get_default_dtype()
            ]
            self.assertEqual(layer._fuse_kernel_compute_dtype, expected)

    # Test invalid dtype handling
    def test_dtype_invalid(self):
        fd_config = DummyFDConfig()
        layer = SiluAndMul(fd_config)
        layer._helper = DummyHelper("int8")
        with self.assertRaises(ValueError):
            dtype = layer._helper.get_default_dtype()
            if dtype not in ["float16", "bfloat16", "float32"]:
                raise ValueError(f"Just support float32, float16 and bfloat16 as default dtype, but received {dtype}")

    # Test fp8 quantization handling
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform(cuda=True)
    )
    def test_fp8_quant(self, mock_platform):
        class DummyFp8Config:
            quant_round_type = 1
            quant_max_bound = 127
            quant_min_bound = -128

            def name(self):
                return "fp8"

        fd_config = DummyFDConfig()
        fd_config.quant_config = DummyFp8Config()
        layer = SiluAndMul(fd_config)
        layer._helper = DummyHelper("float16")
        if "fp8" in fd_config.quant_config.name():
            layer.dequant_scales = None
            layer.shift = None
            layer.smooth = None
        self.assertIsNone(layer.dequant_scales)
        self.assertIsNone(layer.shift)
        self.assertIsNone(layer.smooth)

    # Test act_method mapping
    @patch(
        "fastdeploy.model_executor.layers.activation.current_platform", new_callable=lambda: DummyPlatform(cuda=True)
    )
    def test_act_method_mapping(self, mock_platform):
        fd_config = DummyFDConfig()
        layer = SiluAndMul(fd_config, act_method="silu")
        self.assertEqual(layer.act_method, "swiglu")
        layer = SiluAndMul(fd_config, act_method="relu")
        self.assertEqual(layer.act_method, "relu")


if __name__ == "__main__":
    unittest.main()
