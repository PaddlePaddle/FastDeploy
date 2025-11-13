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

import ctypes
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.distributed.custom_all_reduce import cuda_wrapper


class TestCudaRTLibrary(unittest.TestCase):

    @patch("fastdeploy.distributed.custom_all_reduce.cuda_wrapper.find_loaded_library")
    @patch("ctypes.CDLL")
    def test_basic_init_and_function_calls(self, mock_cdll, mock_find_lib):
        """Test initialization and basic function calls of CudaRTLibrary"""
        mock_find_lib.return_value = "/usr/local/cuda/lib64/libcudart.so"
        mock_lib = MagicMock()
        mock_cdll.return_value = mock_lib

        # Mock all exported functions to return success (0)
        for func in cuda_wrapper.CudaRTLibrary.exported_functions:
            setattr(mock_lib, func.name, MagicMock(return_value=0))
        mock_lib.cudaGetErrorString.return_value = b"no error"

        lib = cuda_wrapper.CudaRTLibrary()
        ptr = lib.cudaMalloc(64)
        lib.cudaMemset(ptr, 1, 64)
        lib.cudaMemcpy(ptr, ptr, 64)
        lib.cudaFree(ptr)
        lib.cudaSetDevice(0)
        lib.cudaDeviceSynchronize()
        lib.cudaDeviceReset()
        handle = lib.cudaIpcGetMemHandle(ptr)
        lib.cudaIpcOpenMemHandle(handle)
        lib.cudaStreamIsCapturing(ctypes.c_void_p(0))

        self.assertTrue(mock_lib.cudaMalloc.called)
        self.assertTrue(mock_lib.cudaFree.called)

    @patch("builtins.open", create=True)
    def test_find_loaded_library_found(self, mock_open):
        """Test find_loaded_library returns correct path when library is found"""
        # Simulate maps file containing libcudart path
        mock_open.return_value.__enter__.return_value = ["7f... /usr/local/cuda/lib64/libcudart.so.11.0\n"]
        result = cuda_wrapper.find_loaded_library("libcudart")
        self.assertIn("libcudart.so.11.0", result)

    def test_find_loaded_library_not_found(self):
        """Test find_loaded_library returns None when library is not found"""
        with patch("builtins.open", unittest.mock.mock_open(read_data="")):
            path = cuda_wrapper.find_loaded_library("libcudart")
            self.assertIsNone(path)

    def test_cudart_check_raises_error(self):
        """Test CUDART_CHECK raises RuntimeError for non-zero error codes"""
        lib = MagicMock()
        lib.cudaGetErrorString.return_value = b"mock error"
        fake = cuda_wrapper.CudaRTLibrary.__new__(cuda_wrapper.CudaRTLibrary)
        fake.funcs = {"cudaGetErrorString": lib.cudaGetErrorString}

        with self.assertRaises(RuntimeError):
            fake.CUDART_CHECK(1)  # Non-zero error code triggers exception

    @patch("fastdeploy.distributed.custom_all_reduce.cuda_wrapper.find_loaded_library")
    @patch("ctypes.CDLL")
    def test_cache_path_reuse(self, mock_cdll, mock_find_lib):
        """Test library path caching is reused between instances"""
        mock_find_lib.return_value = "/usr/local/cuda/lib64/libcudart.so"
        mock_lib = MagicMock()
        mock_cdll.return_value = mock_lib

        for func in cuda_wrapper.CudaRTLibrary.exported_functions:
            setattr(mock_lib, func.name, MagicMock(return_value=0))
        mock_lib.cudaGetErrorString.return_value = b"ok"

        first = cuda_wrapper.CudaRTLibrary()
        second = cuda_wrapper.CudaRTLibrary()

        self.assertIs(first.funcs, second.funcs)
        self.assertIs(first.lib, second.lib)


if __name__ == "__main__":
    unittest.main()
