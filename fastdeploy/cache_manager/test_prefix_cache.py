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

import unittest

import numpy as np

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager


class DummyInnerCacheConfig:
    def __init__(self):
        self.prefill_kvcache_block_num = 4
        self.max_cache_size = 1024
        self.use_cuda_graph = False
        self.num_cpu_blocks = 2
        self.num_gpu_blocks = 2
        self.bytes_per_layer_per_block = 64


class DummySpeculativeConfig:
    def __init__(self):
        self.enable_speculation = False
        self.some_other_field = 0


class DummyCacheConfig:
    def __init__(self):
        self.cache_config = DummyInnerCacheConfig()
        self.speculative_config = DummySpeculativeConfig()


class DummyTask:
    def __init__(self, token_ids, request_id):
        self.prompt_token_ids = token_ids
        self.request_id = request_id


class TestPrefixCacheManagerSwapWithCPUHit(unittest.TestCase):
    """Test GPU–CPU cache swapping behavior with CPU hits."""

    def setUp(self):
        """Initialize cache manager and patch request_block_ids to simulate CPU/GPU allocation."""
        config = DummyCacheConfig()
        self.cache_manager = PrefixCacheManager(config, tensor_parallel_size=1)
        self.cpu_cache = [None] * self.cache_manager.cache_config.num_cpu_blocks

        orig_request_block_ids = self.cache_manager.request_block_ids

        def patched_request_block_ids(task, block_size, dec_token_num):
            total_blocks_needed = len(task.prompt_token_ids) // block_size

            gpu_avail = min(total_blocks_needed, self.cache_manager.cache_config.num_gpu_blocks)
            cpu_avail = total_blocks_needed - gpu_avail

            common, unique, hit_info = orig_request_block_ids(task, block_size, dec_token_num)

            cpu_hits = []
            cpu_unique = []
            for _ in range(cpu_avail):
                hit_found = False
                for idx, val in enumerate(self.cpu_cache):
                    if val == task.request_id:
                        cpu_hits.append(f"CPU-{idx}")
                        hit_found = True
                        break
                if not hit_found:
                    for idx, val in enumerate(self.cpu_cache):
                        if val is None:
                            self.cpu_cache[idx] = task.request_id
                            cpu_unique.append(f"CPU-{idx}")
                            break
            return common, unique + cpu_hits + cpu_unique, hit_info

        self.cache_manager.request_block_ids = patched_request_block_ids

    def request_and_report(self, task, block_size, dec_token_num):
        """Helper to request cache blocks and print hit/alloc status."""
        common, unique, _ = self.cache_manager.request_block_ids(task, block_size, dec_token_num)
        total_blocks = len(task.prompt_token_ids) // block_size

        gpu_hits = [idx for idx in common if isinstance(idx, int)]
        cpu_hits = [idx for idx in common if isinstance(idx, str) and idx.startswith("CPU")]

        gpu_unique = [idx for idx in unique if isinstance(idx, int)]
        cpu_unique = [idx for idx in unique if isinstance(idx, str) and idx.startswith("CPU")]

        hit_rate_gpu = len(gpu_hits) / total_blocks if total_blocks > 0 else 1.0
        hit_rate_cpu = len(cpu_hits) / total_blocks if total_blocks > 0 else 0.0
        total_hit_rate = (len(gpu_hits) + len(cpu_hits)) / total_blocks if total_blocks > 0 else 1.0

        gpu_status = ["." for _ in range(self.cache_manager.cache_config.num_gpu_blocks)]
        for idx in gpu_hits + gpu_unique:
            gpu_status[idx % self.cache_manager.cache_config.num_gpu_blocks] = str(task.request_id)

        cpu_status = ["." for _ in range(self.cache_manager.cache_config.num_cpu_blocks)]
        for idx, val in enumerate(self.cpu_cache):
            if val is not None:
                cpu_status[idx] = str(val)

        print("\n" + "=" * 60)
        print(f"Task {task.request_id} request (total blocks={total_blocks}):")
        print("-" * 60)
        print(f"GPU Hits      : {gpu_hits}")
        print(f"CPU Hits      : {cpu_hits}")
        print(f"GPU Allocated : {gpu_unique}")
        print(f"CPU Allocated : {cpu_unique}")
        print("-" * 60)
        print(f"Hit Rate (GPU) : {hit_rate_gpu:.2f}")
        print(f"Hit Rate (CPU) : {hit_rate_cpu:.2f}")
        print(f"Total Hit Rate : {total_hit_rate:.2f}")
        print("-" * 60)
        print(f"GPU Blocks : {''.join(gpu_status)}   (numbers=request_id, .=free)")
        print(f"CPU Blocks : {''.join(cpu_status)}   (numbers=request_id, .=free)")
        print("=" * 60)

        return gpu_hits, cpu_hits, gpu_unique, cpu_unique, total_hit_rate

    def test_gpu_cpu_swap_with_cpu_hit(self):
        """Task 1 uses GPU, Task 2 triggers CPU fallback, then Task 1 hits GPU/CPU."""
        block_size = 4
        dec_token_num = 0

        task1 = DummyTask(np.arange(8, dtype=np.int64), request_id=1)
        self.request_and_report(task1, block_size, dec_token_num)

        task2 = DummyTask(np.arange(12, dtype=np.int64), request_id=2)
        self.request_and_report(task2, block_size, dec_token_num)

        self.request_and_report(task1, block_size, dec_token_num)


if __name__ == "__main__":
    unittest.main()
