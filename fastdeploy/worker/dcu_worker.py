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

import time

import paddle

from fastdeploy.config import FDConfig
from fastdeploy.utils import get_logger
from fastdeploy.worker.gpu_worker import GpuWorker

logger = get_logger("dcu_worker", "dcu_worker.log")


class DcuWorker(GpuWorker):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        local_rank: int,
        rank: int,
    ):
        super().__init__(
            fd_config=fd_config,
            local_rank=local_rank,
            rank=rank,
        )
        pass

    def determine_available_memory(self) -> int:
        """
        Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # 1. Record memory state before profile run
        Gb = 1024**3
        start_time = time.perf_counter()
        paddle.device.cuda.reset_max_memory_reserved(self.local_rank)
        paddle.device.cuda.reset_max_memory_allocated(self.local_rank)
        paddle_reserved_mem_before_run = paddle.device.cuda.max_memory_reserved(self.local_rank)
        paddle_allocated_mem_before_run = paddle.device.cuda.max_memory_allocated(self.local_rank)  # not reserved

        total_gpu_memory = paddle.device.cuda.get_device_properties(self.local_rank).total_memory
        before_used_gpu_memory = paddle.device.cuda.memory_allocated(self.local_rank)

        logger.info(
            (
                "Before running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {total_gpu_memory / Gb}",
                f"\nDevice used memory: {before_used_gpu_memory / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_before_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_before_run / Gb}",
            )
        )

        # 2. Profile run
        self.model_runner.profile_run()

        # 3. Statistical memory information
        paddle_reserved_mem_after_run = paddle.device.cuda.max_memory_reserved(self.local_rank)
        paddle_allocated_mem_after_run = paddle.device.cuda.max_memory_allocated(self.local_rank)

        after_used_gpu_memory = paddle.device.cuda.memory_allocated(self.local_rank)

        # v0 worker
        model_block_memory_used = self.cal_theortical_kvcache()
        paddle.device.cuda.empty_cache()
        paddle_peak_increase = paddle_reserved_mem_after_run - paddle_allocated_mem_before_run
        available_kv_cache_memory = (
            total_gpu_memory * self.parallel_config.gpu_memory_utilization
            - after_used_gpu_memory
            - paddle_peak_increase
        )
        available_kv_cache_memory += model_block_memory_used * self.parallel_config.total_block_num

        end_time = time.perf_counter()
        logger.info(
            (
                "After running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {total_gpu_memory / Gb}",
                f"\nDevice used memory: {after_used_gpu_memory / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_after_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_after_run / Gb}",
                f"\nAvailable KV Cache meomory: {available_kv_cache_memory / Gb}",
                f"Profile time: {end_time - start_time}",
            )
        )

        return available_kv_cache_memory  # return to caculate the block num in this device
