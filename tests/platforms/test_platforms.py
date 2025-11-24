"""
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

import unittest
from unittest.mock import patch

from fastdeploy.platforms.base import _Backend
from fastdeploy.platforms.cpu import CPUPlatform
from fastdeploy.platforms.cuda import CUDAPlatform
from fastdeploy.platforms.dcu import DCUPlatform
from fastdeploy.platforms.gcu import GCUPlatform
from fastdeploy.platforms.intel_hpu import INTEL_HPUPlatform
from fastdeploy.platforms.maca import MACAPlatform
from fastdeploy.platforms.npu import NPUPlatform
from fastdeploy.platforms.xpu import XPUPlatform


class TestCPUPlatform(unittest.TestCase):
    """Test suite for CPUPlatform"""

    def setUp(self):
        self.platform = CPUPlatform()

    @patch("paddle.device.get_device", return_value="cpu")
    def test_is_cpu_and_available(self, mock_get_device):
        """Verify is_cpu() returns True and platform is available"""
        self.assertTrue(self.platform.is_cpu())
        self.assertTrue(self.platform.available())

    def test_attention_backend(self):
        """Verify get_attention_backend_cls returns empty string for CPU"""
        self.assertEqual(self.platform.get_attention_backend_cls(None), "")


class TestCUDAPlatform(unittest.TestCase):
    """Test suite for CUDAPlatform"""

    def setUp(self):
        self.platform = CUDAPlatform()

    @patch("paddle.is_compiled_with_cuda", return_value=True)
    @patch("paddle.device.get_device", return_value="cuda")
    @patch("paddle.static.cuda_places", return_value=[0])
    def test_is_cuda_and_available(self, mock_cuda_places, mock_is_cuda, mock_get_device):
        """Verify is_cuda() returns True and platform is available"""
        self.assertTrue(self.platform.is_cuda())
        self.assertTrue(self.platform.available())

    def test_attention_backend_valid(self):
        """Verify valid attention backends return correct class names"""
        self.assertIn("PaddleNativeAttnBackend", self.platform.get_attention_backend_cls(_Backend.NATIVE_ATTN))
        self.assertIn("AppendAttentionBackend", self.platform.get_attention_backend_cls(_Backend.APPEND_ATTN))
        self.assertIn("MLAAttentionBackend", self.platform.get_attention_backend_cls(_Backend.MLA_ATTN))
        self.assertIn("FlashAttentionBackend", self.platform.get_attention_backend_cls(_Backend.FLASH_ATTN))

    def test_attention_backend_invalid(self):
        """Verify invalid backend raises ValueError"""
        with self.assertRaises(ValueError):
            self.platform.get_attention_backend_cls("INVALID_BACKEND")


class TestMACAPlatform(unittest.TestCase):
    """Test suite for MACAPlatform"""

    @patch("paddle.static.cuda_places", return_value=[0, 1])
    def test_available_true(self, mock_cuda_places):
        """Verify available() returns True when GPUs exist"""
        self.assertTrue(MACAPlatform.available())
        mock_cuda_places.assert_called_once()

    @patch("paddle.static.cuda_places", side_effect=Exception("No GPU"))
    def test_available_false(self, mock_cuda_places):
        """Verify available() returns False when no GPUs"""
        self.assertFalse(MACAPlatform.available())
        mock_cuda_places.assert_called_once()

    def test_get_attention_backend_native(self):
        """Verify NATIVE_ATTN returns correct backend class"""
        self.assertIn("PaddleNativeAttnBackend", MACAPlatform.get_attention_backend_cls(_Backend.NATIVE_ATTN))

    def test_get_attention_backend_append(self):
        """Verify APPEND_ATTN returns correct backend class"""
        self.assertIn("FlashAttentionBackend", MACAPlatform.get_attention_backend_cls(_Backend.APPEND_ATTN))

    def test_get_attention_backend_invalid(self):
        """Verify invalid backend raises ValueError"""
        with self.assertRaises(ValueError):
            MACAPlatform.get_attention_backend_cls("INVALID_BACKEND")


class TestINTELHPUPlatform(unittest.TestCase):
    """Test suite for INTEL_HPUPlatform"""

    @patch("paddle.base.core.get_custom_device_count", return_value=1)
    def test_available_true(self, mock_get_count):
        """Verify available() returns True when HPU exists"""
        self.assertTrue(INTEL_HPUPlatform.available())
        mock_get_count.assert_called_with("intel_hpu")

    @patch("paddle.base.core.get_custom_device_count", side_effect=Exception("No HPU"))
    @patch("fastdeploy.utils.console_logger.warning")
    def test_available_false(self, mock_logger_warn, mock_get_count):
        """Verify available() returns False and warns when no HPU"""
        self.assertFalse(INTEL_HPUPlatform.available())
        mock_logger_warn.assert_called()
        self.assertIn("No HPU", mock_logger_warn.call_args[0][0])

    def test_attention_backend_native(self):
        """Verify NATIVE_ATTN returns correct backend class"""
        self.assertIn("PaddleNativeAttnBackend", INTEL_HPUPlatform.get_attention_backend_cls(_Backend.NATIVE_ATTN))

    def test_attention_backend_hpu(self):
        """Verify HPU_ATTN returns correct backend class"""
        self.assertIn("HPUAttentionBackend", INTEL_HPUPlatform.get_attention_backend_cls(_Backend.HPU_ATTN))

    @patch("fastdeploy.utils.console_logger.warning")
    def test_attention_backend_other(self, mock_logger_warn):
        """Verify invalid backend logs warning and returns None"""
        self.assertIsNone(INTEL_HPUPlatform.get_attention_backend_cls("INVALID_BACKEND"))
        mock_logger_warn.assert_called()


class TestNPUPlatform(unittest.TestCase):
    """Test suite for NPUPlatform"""

    def setUp(self):
        self.platform = NPUPlatform()

    def test_device_name(self):
        """Verify device_name is set to 'npu'"""
        self.assertEqual(self.platform.device_name, "npu")


class TestDCUPlatform(unittest.TestCase):
    """Test suite for DCUPlatform"""

    def setUp(self):
        self.platform = DCUPlatform()

    @patch("paddle.static.cuda_places", return_value=[0])
    def test_available_with_gpu(self, mock_cuda_places):
        """Verify available() returns True when GPU exists"""
        self.assertTrue(self.platform.available())

    @patch("paddle.static.cuda_places", side_effect=Exception("No GPU"))
    def test_available_no_gpu(self, mock_cuda_places):
        """Verify available() returns False when no GPU"""
        self.assertFalse(self.platform.available())

    def test_attention_backend_native(self):
        """Verify NATIVE_ATTN returns correct backend class"""
        self.assertIn("PaddleNativeAttnBackend", self.platform.get_attention_backend_cls(_Backend.NATIVE_ATTN))

    def test_attention_backend_block(self):
        """Verify BLOCK_ATTN returns correct backend class"""
        self.assertIn("BlockAttentionBackend", self.platform.get_attention_backend_cls(_Backend.BLOCK_ATTN))

    def test_attention_backend_invalid(self):
        """Verify invalid backend returns None"""
        self.assertIsNone(self.platform.get_attention_backend_cls("INVALID_BACKEND"))


class TestGCUPlatform(unittest.TestCase):
    """Test suite for GCUPlatform"""

    def setUp(self):
        self.platform = GCUPlatform()

    @patch("paddle.base.core.get_custom_device_count", return_value=1)
    def test_available_with_gcu(self, mock_get_count):
        """Verify available() returns True when GCU exists"""
        self.assertTrue(self.platform.available())

    @patch("paddle.base.core.get_custom_device_count", side_effect=Exception("No GCU"))
    def test_available_no_gcu(self, mock_get_count):
        """Verify available() returns False when no GCU"""
        self.assertFalse(self.platform.available())

    def test_attention_backend_native(self):
        """Verify NATIVE_ATTN returns correct backend class"""
        self.assertIn("GCUMemEfficientAttnBackend", self.platform.get_attention_backend_cls(_Backend.NATIVE_ATTN))

    def test_attention_backend_append(self):
        """Verify APPEND_ATTN returns correct backend class"""
        self.assertIn("GCUFlashAttnBackend", self.platform.get_attention_backend_cls(_Backend.APPEND_ATTN))

    def test_attention_backend_invalid(self):
        """Verify invalid backend raises ValueError"""
        with self.assertRaises(ValueError):
            self.platform.get_attention_backend_cls("INVALID_BACKEND")


class TestXPUPlatform(unittest.TestCase):
    """Test suite for XPUPlatform"""

    @patch("paddle.is_compiled_with_xpu", return_value=True)
    @patch("paddle.static.xpu_places", return_value=[0])
    def test_available_true(self, mock_places, mock_xpu):
        """Verify available() returns True when XPU is compiled and available"""
        self.assertTrue(XPUPlatform.available())

    @patch("paddle.is_compiled_with_xpu", return_value=False)
    @patch("paddle.static.xpu_places", return_value=[])
    def test_available_false(self, mock_places, mock_xpu):
        """Verify available() returns False when XPU is unavailable"""
        self.assertFalse(XPUPlatform.available())

    def test_get_attention_backend_cls(self):
        """Verify NATIVE_ATTN returns correct XPU backend class"""
        expected_cls = "fastdeploy.model_executor.layers.attention.XPUAttentionBackend"
        self.assertEqual(XPUPlatform.get_attention_backend_cls(_Backend.NATIVE_ATTN), expected_cls)


if __name__ == "__main__":
    unittest.main()
