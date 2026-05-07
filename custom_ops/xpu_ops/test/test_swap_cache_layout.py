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

import ctypes
import random
import time
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import (
    cuda_host_alloc,
    cuda_host_free,
    swap_cache_layout,
)


class TestAllocCachePinned(unittest.TestCase):
    """Verify xpu_host_alloc/xpu_host_free and basic host memory access."""

    def test_alloc_free(self):
        size = 16 * 1024 * 1024
        ptr = cuda_host_alloc(size)
        self.assertNotEqual(ptr, 0, "cuda_host_alloc returned null")

        try:
            buf = (ctypes.c_uint8 * 4).from_address(ptr)
            buf[0], buf[1], buf[2], buf[3] = 0xDE, 0xAD, 0xBE, 0xEF
            self.assertEqual(list(buf), [0xDE, 0xAD, 0xBE, 0xEF])
        finally:
            cuda_host_free(ptr)


class TestSwapCacheLayout(unittest.TestCase):
    layer_num = 8
    block_num = 128
    head_num = 4
    block_size = 16
    head_dim = 64
    swap_block_num = 16

    def setUp(self):
        self.cache_shape = [self.block_num, self.head_num, self.block_size, self.head_dim]
        self.block_stride = self.head_num * self.block_size * self.head_dim
        self.block_bytes = self.block_stride * 2

        buffer_total_bytes = self.swap_block_num * self.layer_num * self.block_bytes
        self.cpu_buffer = cuda_host_alloc(buffer_total_bytes)

        self.xpu_block_ids = random.sample(range(self.block_num), self.swap_block_num)
        self.cpu_block_ids = list(range(self.swap_block_num))

    def tearDown(self):
        cuda_host_free(self.cpu_buffer)

    def _make_cache(self, fill_value=None):
        cache = []
        for layer_idx in range(self.layer_num):
            value = float(layer_idx) if fill_value is None else float(fill_value)
            cache.append(paddle.full(self.cache_shape, fill_value=value, dtype=paddle.float16))
        paddle.device.synchronize()
        return cache

    def test_roundtrip(self):
        src = self._make_cache()
        dst = self._make_cache(fill_value=-1)

        swap_cache_layout(
            src,
            self.cpu_buffer,
            self.cache_shape,
            self.xpu_block_ids,
            self.cpu_block_ids,
            0,
            0,
        )
        swap_cache_layout(
            dst,
            self.cpu_buffer,
            self.cache_shape,
            self.xpu_block_ids,
            self.cpu_block_ids,
            0,
            1,
        )

        for layer_idx in range(self.layer_num):
            got = dst[layer_idx][self.xpu_block_ids].numpy()
            expected = np.full_like(got, float(layer_idx))
            self.assertTrue(
                np.allclose(got, expected, atol=1e-2),
                f"roundtrip mismatch at layer={layer_idx}",
            )

    def _run_and_report(self, mode, label):
        cache = self._make_cache()
        total_gb = self.swap_block_num * self.layer_num * self.block_bytes / 1073741824

        start = time.time()
        swap_cache_layout(
            cache,
            self.cpu_buffer,
            self.cache_shape,
            self.xpu_block_ids,
            self.cpu_block_ids,
            0,
            mode,
        )
        paddle.device.synchronize()
        cost_time = time.time() - start
        print(
            f"swap cache layout ({label}), total_gb: {total_gb:.6f}GB, "
            f"cost_time: {cost_time:.6f}s, speed: {total_gb / cost_time:.6f}GB/s"
        )

    def test_performance(self):
        for _ in range(3):
            self._run_and_report(0, "device to host")
        for _ in range(3):
            self._run_and_report(1, "host to device")


if __name__ == "__main__":
    unittest.main(verbosity=2)
