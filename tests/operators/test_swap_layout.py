import time
import unittest

import numpy as np
import paddle

from fastdeploy.cache_manager.ops import cuda_host_alloc, cuda_host_free
from fastdeploy.model_executor.ops.gpu import swap_cache_layout


class Test(unittest.TestCase):
    def setUp(self):

        self.cache_shape = [8, 64, 4, 128]
        self.layer_num = 10
        self.block_ids = np.arange(self.cache_shape[0])
        self.key_register_buffer = cuda_host_alloc(np.prod(self.cache_shape) * 2)

    def release_buffer(self):
        cuda_host_free(self.key_register_buffer)

    def test_swap_cache_layout(self):

        gpu_key_register_buffer = []
        for i in range(self.layer_num):
            gpu_key_register_buffer.append(paddle.full(self.cache_shape, fill_value=i, dtype=paddle.float16))

        ss = time.time()
        swap_cache_layout(
            gpu_key_register_buffer,
            self.key_register_buffer,
            self.cache_shape,
            self.block_ids,
            0,
            0,
        )
        print("swap cache layout (host to device): ", time.time() - ss)
        ss = time.time()
        del gpu_key_register_buffer
        gpu_key_register_buffer = []
        for i in range(self.layer_num):
            gpu_key_register_buffer.append(paddle.zeros(self.cache_shape, dtype=paddle.float16))
        swap_cache_layout(
            gpu_key_register_buffer,
            self.key_register_buffer,
            self.cache_shape,
            self.block_ids,
            0,
            1,
        )
        for i in range(self.layer_num):
            assert gpu_key_register_buffer[i].numpy()[0, 0, 0, 0] == i
        print("swap cache layout（device to host):", time.time() - ss)

        self.release_buffer()


if __name__ == "__main__":
    unittest.main()
