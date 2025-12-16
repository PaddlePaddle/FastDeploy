import os
import time

import numpy as np
import paddle

from fastdeploy.cache_manager.ops import cuda_host_alloc, cuda_host_free
from fastdeploy.cache_manager.transfer_factory import MooncakeStore
from fastdeploy.model_executor.ops.gpu import swap_cache_layout

MOONCAKE_CONFIG_PATH = "./mooncake_config.json"


class TestMooncakeStore:
    def __init__(self):
        os.environ["MOONCAKE_CONFIG_PATH"] = os.getenv("MOONCAKE_CONFIG_PATH", MOONCAKE_CONFIG_PATH)
        self.storage_backend = MooncakeStore()
        self.cache_shape = [4, 64, 128]
        self.lay_number = 80
        self.max_model_len = 64 * 1024
        self.block_number = self.max_model_len // self.cache_shape[1]

    def test_register_buffer(self):
        need_to_allocate_bytes = self.max_model_len * self.cache_shape[0] * self.lay_number * self.cache_shape[2] * 2
        self.cache_stride = self.cache_shape[1] * self.cache_shape[2] * self.cache_shape[0] * self.lay_number
        print(f"creating cpu cache for alllayers {self.lay_number}: {need_to_allocate_bytes / 1024 ** 3:.2f}GB")
        self.key_register_buffer = cuda_host_alloc(need_to_allocate_bytes)
        self.storage_backend.register_buffer(self.key_register_buffer, need_to_allocate_bytes)

    def _init_gpu_blocks(self):
        self.key_gpu_cache = []
        for i in range(self.lay_number):
            key_cache = paddle.full(
                shape=[self.block_number, self.cache_shape[0], self.cache_shape[1], self.cache_shape[2]],
                fill_value=0,
                dtype=paddle.bfloat16,
            )
            self.key_gpu_cache.append(key_cache)

    def test_write_storage(self, test_block_num=16):
        start_time = time.time()
        keys = [f"test_key_{i}" for i in range(test_block_num)]
        gpu_block_ids = np.arange(test_block_num)

        swap_cache_layout(self.key_gpu_cache, self.key_register_buffer, gpu_block_ids, 0, 1)  # gpu ==> cpu
        print("swap_cache_layout done", time.time() - start_time)
        # import pdb; pdb.set_trace()
        target_location = [self.key_register_buffer + i * self.cache_stride for i in range(test_block_num)]

        target_sizes = [self.cache_stride] * test_block_num

        self.storage_backend.set(keys, target_location=target_location, target_sizes=target_sizes)
        print("write storage time: ", time.time() - start_time)

    def test_read_storage(self, test_block_num=16):
        start_time = time.time()
        keys = [f"test_key_{i}" for i in range(test_block_num)]
        gpu_block_ids = np.arange(test_block_num)
        target_location = [self.key_register_buffer + i * self.cache_stride for i in range(test_block_num)]

        target_sizes = [self.cache_stride] * test_block_num

        self.storage_backend.get(keys, target_location=target_location, target_sizes=target_sizes)

        swap_cache_layout(self.key_gpu_cache, self.key_register_buffer, gpu_block_ids, 0, 0)  # cpu ==> gpu
        print("read storage time: ", time.time() - start_time)

    def free(self):
        cuda_host_free(self.key_register_buffer)


if __name__ == "__main__":

    tester = TestMooncakeStore()
    tester.test_register_buffer()
    tester._init_gpu_blocks()
    tester.test_write_storage(64)
    tester.test_read_storage(64)
    tester.free()
