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
import time
from typing import List, Optional

import paddle
import paddle.nn as nn
import pynvml

from fastdeploy.config import LLMConfig
from fastdeploy.engine.request import Request
from fastdeploy.scheduler.scheduler_batch import ModelForwardBatch
from fastdeploy.utils import get_logger
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.V1.gpu_model_runner import GPUModelRunner
from fastdeploy.worker.V1.worker_base import WorkerBase

logger = get_logger("gpu_worker", "gpu_worker.log")


class GpuWorker(WorkerBase):
    """ """

    def __init__(
        self,
        llm_config: LLMConfig,
        local_rank: int,
        rank: int,
    ):
        super.__init__(
            llm_config=llm_config,
            local_rank=local_rank,
            rank=rank,
        )
        pass

    def init_device(self):
        """  """
        if self.device_config.device.type == "cuda" and paddle.device.is_compiled_with_cuda(
        ):
            # Set evironment variable
            self.device = f"gpu:{self.local_rank}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.model_config.dtype)

            # Get free memory info
            pynvml.nvmlInit()
            handler = pynvml.nvmlDeviceGetHandleByIndex(self.local_rank)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            pynvml.nvmlShutdown()

            self.free_gpu_memory = meminfo.free
            gc.collect()
            paddle.device.cuda.empty_cache()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        # Construct model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.llm_config, self.device)

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
        paddle.device.cuda.reset_max_memory_reserved(self.local_rank)
        paddle.device.cuda.reset_max_memory_allocated(self.local_rank)
        paddle_reserved_mem_before_run = paddle.device.cuda.max_memory_reserved(
            self.local_rank)
        paddle_allocated_mem_before_run = paddle.device.cuda.max_memory_allocated(
            self.local_rank)  # not reserved

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            int(self.device_ids[self.rank]))
        before_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        logger.info((
            "Before running the profile, the memory usage info is as follows:",
            f"\tDevice Total memory: {before_run_meminfo.total}",
            f"\tDevice used memory: {before_run_meminfo.used}",
            f"\tDevice free memory: {before_run_meminfo.free}",
            f"\tPaddle reserved memory: {paddle_reserved_mem_before_run}",
            f"\tPaddle allocated memory: {paddle_allocated_mem_before_run}"))

        # 2. Profile run
        self.model_runner.profile_run()

        # 3. Statistical memory information
        paddle_reserved_mem_after_run = paddle.device.cuda.max_memory_reserved(
            self.local_rank)
        paddle_allocated_mem_after_run = paddle.device.cuda.max_memory_allocated(
            self.local_rank)

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            int(self.device_ids[self.rank]))
        after_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        not_paddle_use_mem = after_run_meminfo.used - paddle_reserved_mem_after_run
        peak_memory = paddle_allocated_mem_after_run + not_paddle_use_mem

        available_kv_cache_memory = after_run_meminfo.total * self.cache_config.gpu_memory_utilization - peak_memory

        end_time = time.perf_counter()
        logger.info(
            ("After running the profile, the memory usage info is as follows:",
             f"\tDevice Total memory: {after_run_meminfo.total}",
             f"\tDevice used memory: {after_run_meminfo.used}",
             f"\tDevice free memory: {after_run_meminfo.free}",
             f"\tPaddle reserved memory: {paddle_reserved_mem_after_run}",
             f"\tPaddle allocated memory: {paddle_allocated_mem_after_run}",
             f"\tAvailable KV Cache meomory: {available_kv_cache_memory}",
             f"Profile time: {end_time - start_time}"))

        return available_kv_cache_memory  # return to caculate the block num in this device

    def load_model(self) -> nn.Layer:
        """ """
        pass

    def get_model(self) -> nn.Layer:
        """ """
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """ """
        pass

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request], ModelForwardBatch],
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch)

        assert isinstance(output, ModelRunnerOutput)
        return output

    def preprocess_new_task(self, req_dicts: List[Request]) -> None:
        """ Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        self.model_runner.process_prefill_inputs(req_dicts=req_dicts)

    def get_kv_cache_spec(self) -> dict[str, paddle.Tensor]:
        """ """
        pass

    def graph_optimize_and_warm_up_model(self) -> None:
        """ """
        pass

    def check_health(self) -> bool:
        """ """
        return True
