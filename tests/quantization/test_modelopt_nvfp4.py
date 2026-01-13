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
from unittest import mock

import paddle

from fastdeploy.flashinfer import has_flashinfer
from fastdeploy.model_executor.layers.linear import QKVParallelLinear
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.nvfp4 import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
    ModelOptNvFp4LinearMethod,
)


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or get_sm_version() < 100,
    "Nvfp4 do not support sm < 100.",
)
class TestModelOptNvFp4Config(unittest.TestCase):
    def setUp(self):
        prop = paddle.device.cuda.get_device_properties()
        self.sm_version = prop.major * 10 + prop.minor

        self.raw_config = {
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"],
                }
            },
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.34.1.dev85+g7a72957d"},
            "quant_method": "modelopt",
        }

        self.config = ModelOptNvFp4Config.from_config(self.raw_config)

    def test_name(self):
        """Test name() method"""
        self.assertEqual(self.config.name(), "modelopt_fp4")

    def test_from_config(self):
        """Test from_config with full dict"""
        cfg = ModelOptNvFp4Config.from_config(self.raw_config)
        self.assertFalse(cfg.is_checkpoint_bf16)
        self.assertTrue(cfg.is_checkpoint_nvfp4_serialized)
        self.assertEqual(cfg.group_size, 16)
        self.assertEqual(cfg.exclude_modules, [])
        self.assertEqual(cfg.kv_cache_quant_algo, None)
        self.assertEqual(cfg.quant_max_bound, 6)
        self.assertEqual(cfg.quant_min_bound, -6)
        self.assertEqual(cfg.quant_round_type, 1)

    @unittest.skipIf(not has_flashinfer(), "Skip if no FlashInfer available")
    def test_get_quant_method_linear(self):
        """Test get_quant_method with a linear layer"""
        layer = mock.Mock(spec=QKVParallelLinear)
        method = self.config.get_quant_method(layer)
        assert isinstance(method, ModelOptNvFp4LinearMethod)

    @unittest.skipIf(not has_flashinfer(), "Skip if no FlashInfer available")
    def test_get_quant_method_fused_moe(self):
        """Test get_quant_method with a moe layer"""
        layer = mock.Mock(spec=FusedMoE)
        method = self.config.get_quant_method(layer)
        assert isinstance(method, ModelOptNvFp4FusedMoE)


if __name__ == "__main__":
    unittest.main()
