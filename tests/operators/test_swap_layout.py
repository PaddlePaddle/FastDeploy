import random
import time
import unittest

import paddle

from fastdeploy.cache_manager.ops import cuda_host_alloc, cuda_host_free
from fastdeploy.model_executor.ops.gpu import swap_cache_layout


class Test(unittest.TestCase):
    def setUp(self):
        self.layer_num = 30
        self.block_num = 3000
        self.head_num = 4
        self.block_size = 64
        self.head_dim = 128

        self.swap_block_num = 100
        self.cache_shape = [self.block_num, self.head_num, self.block_size, self.head_dim]
        assert self.swap_block_num <= self.block_num

        # cache layout: layer_num * [block_num, head_num, block_size, head_dim]
        # buffer layout: [block_num, layer_num, head_num, block_size, head_dim]
        # self.gpu_cache_tensors = self._init_gpu_cache()

        self.block_bytes = self.head_num * self.block_size * self.head_dim * 2
        buffer_total_bytes = self.swap_block_num * self.layer_num * self.block_bytes
        self.cpu_buffer = cuda_host_alloc(buffer_total_bytes)

        self.gpu_block_ids = random.sample(list(range(self.block_num)), self.swap_block_num)
        self.cpu_block_ids = list(range(self.swap_block_num))

    def tearDown(self) -> None:
        cuda_host_free(self.cpu_buffer)

    def _init_gpu_cache(self, fill_value=None):
        gpu_cache_tensors = []
        for i in range(self.layer_num):
            if fill_value is None:
                value = i
            else:
                value = float(fill_value)
            gpu_cache_tensors.append(paddle.full(self.cache_shape, fill_value=value, dtype=paddle.float16))
        paddle.device.synchronize()
        return gpu_cache_tensors

    def _swap_cache_layout(self):
        self.gpu_cache_tensors = self._init_gpu_cache()

        ss = time.time()
        swap_cache_layout(
            self.gpu_cache_tensors,
            self.cpu_buffer,
            self.cache_shape,
            self.gpu_block_ids,
            self.cpu_block_ids,
            0,
            0,
        )
        cost_time = time.time() - ss
        total_gb = self.block_bytes * self.swap_block_num * self.layer_num / 1073741824
        speed = total_gb / cost_time
        print(
            f"swap cache layout (device to host), total_gb: {total_gb:.6f}GB, cost_time: {cost_time:.6f}s, speed: {speed:.6f}GB/s"
        )

        self.gpu_cache_tensors = self._init_gpu_cache(-1)

        ss = time.time()
        swap_cache_layout(
            self.gpu_cache_tensors,
            self.cpu_buffer,
            self.cache_shape,
            self.gpu_block_ids,
            self.cpu_block_ids,
            0,
            1,
        )
        cost_time = time.time() - ss
        speed = total_gb / cost_time
        print(
            f"swap cache layout (host to device), total_gb: {total_gb:.6f}GB, cost_time: {cost_time:.6f}s, speed: {speed:.6f}GB/s"
        )

        for i in range(self.layer_num):
            gpu_cache = self.gpu_cache_tensors[i][self.gpu_block_ids]
            assert paddle.allclose(gpu_cache, paddle.ones_like(gpu_cache) * i)

    def test_swap_cache_layout(self):
        """test swap cache layout"""
        for _ in range(5):
            self._swap_cache_layout()


if __name__ == "__main__":
    unittest.main()
