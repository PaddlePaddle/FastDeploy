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


class TestCPUPlatform(unittest.TestCase):
    def setUp(self):
        self.platform = CPUPlatform()

    @patch("paddle.device.get_device", return_value="cpu")
    def test_is_cpu_and_available(self, mock_get_device):
        """
        Check hardware type (CPU) and availability
        """
        self.assertTrue(self.platform.is_cpu())
        self.assertTrue(self.platform.available())

    def test_attention_backend(self):
        """CPUPlatform attention_backend should return empty string"""
        self.assertEqual(self.platform.get_attention_backend_cls(None), "")


class TestCUDAPlatform(unittest.TestCase):
    def setUp(self):
        self.platform = CUDAPlatform()

    @patch("paddle.is_compiled_with_cuda", return_value=True)
    @patch("paddle.device.get_device", return_value="cuda")
    @patch("paddle.static.cuda_places", return_value=[0])
    def test_is_cuda_and_available(self, mock_get_device, mock_is_cuda, mock_cuda_places):
        """
        Check hardware type (CUDA) and availability
        """
        self.assertTrue(self.platform.is_cuda())
        self.assertTrue(self.platform.available())

    def test_attention_backend_valid(self):
        """
        CUDAPlatform should return correct backend class name for valid backends
        """
        self.assertIn(
            "PaddleNativeAttnBackend",
            self.platform.get_attention_backend_cls(_Backend.NATIVE_ATTN),
        )
        self.assertIn(
            "AppendAttentionBackend",
            self.platform.get_attention_backend_cls(_Backend.APPEND_ATTN),
        )
        self.assertIn(
            "MLAAttentionBackend",
            self.platform.get_attention_backend_cls(_Backend.MLA_ATTN),
        )
        self.assertIn(
            "FlashAttentionBackend",
            self.platform.get_attention_backend_cls(_Backend.FLASH_ATTN),
        )

    def test_attention_backend_invalid(self):
        """
        CUDAPlatform should raise ValueError for invalid backend
        """
        with self.assertRaises(ValueError):
            self.platform.get_attention_backend_cls("INVALID_BACKEND")


if __name__ == "__main__":
    unittest.main()
