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

import gc
from typing import List, Optional

import paddle
from paddle import nn
import time
from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.utils import get_logger, set_random_seed
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase
from fastdeploy.worker.xpu_model_runner import XPUModelRunner

logger = get_logger("xpu_worker", "xpu_worker.log")


class XpuWorker(WorkerBase):
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

    def init_device(self):
        """Initialize device and Construct model runner"""
        if paddle.is_compiled_with_xpu():
            # Set evironment variable
            self.device = f"xpu:{self.local_rank}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.parallel_config.dtype)
            self.device_ids = self.parallel_config.device_ids.split(",")

            gc.collect()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: XPUModelRunner = XPUModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            rank=self.rank,
            local_rank=self.local_rank,
        )

    def graph_optimize_and_warm_up_model(self) -> None:
        """
        Perform the warm-up and the graph optimization
        """
        if self.model_runner.graph_opt_level >= 1:
            self.model_runner.sot_warmup()

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
        start_time = time.perf_counter()
        Gb = 1024**3
        local_rank = self.local_rank % 8
        paddle.device.xpu.reset_max_memory_reserved(local_rank)
        paddle.device.xpu.reset_max_memory_allocated(local_rank)
        paddle_reserved_mem_before_run = paddle.device.xpu.max_memory_reserved(local_rank)
        paddle_allocated_mem_before_run = paddle.device.xpu.max_memory_allocated(local_rank)  # not reserved

        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_ids[local_rank]))
        # before_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        logger.info(
            (
                "Before running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {paddle.device.xpu.memory_total(local_rank) / Gb}",
                f"\nDevice used memory: {paddle.device.xpu.memory_used(local_rank) / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_before_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_before_run / Gb}",
            )
        )

        # 2. Profile run
        # self.model_runner.profile_run()
        # set_random_seed(self.fd_config.model_config.seed)
        self.model_runner.prepare_profile()
        self.model_runner.profile_run()
        set_random_seed(self.fd_config.model_config.seed)

        # 3. Statistical memory information
        paddle_reserved_mem_after_run = paddle.device.xpu.max_memory_reserved(local_rank)
        paddle_allocated_mem_after_run = paddle.device.xpu.max_memory_allocated(local_rank)

        model_block_memory_used = self.cal_theortical_kvcache()
        paddle_peak_increase = paddle_reserved_mem_after_run - paddle_allocated_mem_before_run

        paddle.device.xpu.empty_cache()

        # after_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # pynvml.nvmlShutdown()
        after_run_meminfo_total = paddle.device.xpu.memory_total(local_rank)
        after_run_meminfo_used = paddle.device.xpu.memory_used(local_rank)
        available_kv_cache_memory = (
            after_run_meminfo_total * 0.999#self.cache_config.gpu_memory_utilization
            - after_run_meminfo_used
            - paddle_peak_increase
        )
        first = after_run_meminfo_total * 0.999#self.cache_config.gpu_memory_utilization
        # logger.info(
        #     f"\n first: {first}")
        # logger.info(
        #     f"\n second: {after_run_meminfo_used}")
        # logger.info(
        #     f"\n third: {paddle_peak_increase}")
        # logger.info(
        #     f"\n result: {available_kv_cache_memory}"
        #     )
        available_kv_cache_memory += model_block_memory_used * self.parallel_config.total_block_num
        logger.info(f"\n result_2: {available_kv_cache_memory}")
        end_time = time.perf_counter()
        logger.info(
            (
                "After running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {after_run_meminfo_total / Gb}",
                f"\nDevice used memory: {after_run_meminfo_used / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_after_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_after_run / Gb}",
                f"\nAvailable KV Cache meomory: {available_kv_cache_memory / Gb}",
                f"Profile time: {end_time - start_time}",
            )
        )

        return available_kv_cache_memory  # return to caculate the block num in this device

    def cal_theortical_kvcache(self) -> int:
        """ """
        return self.model_runner.cal_theortical_kvcache()

    def load_model(self) -> None:
        """ """
        self.model_runner.load_model()

    def get_model(self) -> nn.Layer:
        """ """
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """ """
        self.model_runner.update_share_input_block_num(num_gpu_blocks=num_gpu_blocks)

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        is_dummy_run: bool = False,
        num_running_requests: Optional[int] = None,
    ) -> Optional[ModelRunnerOutput]:
        """ """

        output = self.model_runner.execute_model(model_forward_batch)

        return output

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        return self.model_runner.exist_prefill()

    def preprocess_new_task(self, req_dicts: List[Request], num_running_requests: int = -1) -> None:
        """Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.model_runner.insert_tasks_v1(req_dicts=req_dicts)
        else:
            self.model_runner.process_prefill_inputs(req_dicts=req_dicts)

    def check_health(self) -> bool:
        """ """
        return True
