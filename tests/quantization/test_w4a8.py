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

from fastdeploy.model_executor.layers.quantization.w4a8 import W4A8Config
from fastdeploy.platforms import current_platform


class TestW4A8Config(unittest.TestCase):
    def setUp(self):
        self.config = W4A8Config(is_permuted=False, hadamard_block_size=128)

    def test_name(self):
        """Test name() method"""
        self.assertEqual(self.config.name(), "w4a8")

    def test_from_config_defaults(self):
        """Test from_config with empty dict uses defaults"""
        cfg = W4A8Config.from_config({})
        self.assertTrue(cfg.is_permuted)
        self.assertEqual(cfg.hadamard_block_size, 128)

    def test_from_config_full(self):
        """Test from_config with full dict"""
        cfg = W4A8Config.from_config({"is_permuted": False, "hadamard_block_size": 64})
        self.assertFalse(cfg.is_permuted)
        self.assertEqual(cfg.hadamard_block_size, 64)

    def test_get_quant_method_cuda(self):
        """Test get_quant_method returns CUDA method when on CUDA platform"""
        with (
            mock.patch.object(current_platform, "is_cuda", return_value=True),
            mock.patch(
                "fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend.CutlassW4A8MoEMethod"
            ) as mock_cuda,
        ):
            layer = mock.Mock()
            method = self.config.get_quant_method(layer)
            mock_cuda.assert_called_once_with(self.config)
            self.assertEqual(method, mock_cuda.return_value)

    @unittest.skipIf(not hasattr(current_platform, "is_xpu") or not current_platform.is_xpu(), "No XPU, skip test")
    def test_get_quant_method_xpu(self):
        """Test get_quant_method returns XPU method when on XPU platform"""
        with mock.patch("fastdeploy.model_executor.layers.backends.xpu.moe.fused_moe.XPUW4A8MoEMethod") as mock_xpu:
            layer = mock.Mock()
            method = self.config.get_quant_method(layer)
            mock_xpu.assert_called_once_with(self.config)
            self.assertEqual(method, mock_xpu.return_value)

    def test_get_quant_method_unsupported(self):
        """Test that unsupported platform raises ValueError"""
        with (
            mock.patch.object(current_platform, "is_cuda", return_value=False),
            mock.patch.object(current_platform, "is_xpu", return_value=False),
        ):
            layer = mock.Mock()
            with self.assertRaises(ValueError):
                self.config.get_quant_method(layer)


if __name__ == "__main__":
    unittest.main()
