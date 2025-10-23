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
import pynvml
from paddle import nn

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.platforms import current_platform
from fastdeploy.plugins.model_runner import load_model_runner_plugins
from fastdeploy.utils import get_logger, set_random_seed
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("gpu_worker", "gpu_worker.log")

try:
    ModelRunner = load_model_runner_plugins()
except:
    from fastdeploy.worker.gpu_model_runner import GPUModelRunner as ModelRunner


class GpuWorker(WorkerBase):
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
        """
        Initialize device and construct model runner
        """
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if self.device_config.device_type == "cuda" and paddle.device.is_compiled_with_cuda():
            # Set environment variable
            self.device_ids = self.parallel_config.device_ids.split(",")
            self.device = f"gpu:{self.local_rank % self.max_chips_per_node}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.model_config.dtype)

            gc.collect()
            paddle.device.cuda.empty_cache()
            if (
                not self.parallel_config.disable_custom_all_reduce
                and self.parallel_config.tensor_parallel_size > 1
                and paddle.is_compiled_with_cuda()
            ):
                from fastdeploy.distributed.communication import use_custom_allreduce

                use_custom_allreduce()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: ModelRunnerBase = ModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            device_id=int(self.device_ids[self.local_rank % self.max_chips_per_node]),
            rank=self.rank,
            local_rank=self.local_rank,
        )

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        return self.model_runner.exist_prefill()

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
        local_rank = self.local_rank % self.max_chips_per_node
        paddle.device.cuda.reset_max_memory_reserved(local_rank)
        paddle.device.cuda.reset_max_memory_allocated(local_rank)
        paddle_reserved_mem_before_run = paddle.device.cuda.max_memory_reserved(local_rank)
        paddle_allocated_mem_before_run = paddle.device.cuda.max_memory_allocated(local_rank)  # not reserved

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_ids[local_rank]))
        before_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        logger.info(
            (
                "Before running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {before_run_meminfo.total / Gb}",
                f"\nDevice used memory: {before_run_meminfo.used / Gb}",
                f"\nDevice free memory: {before_run_meminfo.free / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_before_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_before_run / Gb}",
            )
        )

        # 2. Profile run
        self.model_runner.profile_run()
        set_random_seed(self.fd_config.model_config.seed)

        # 3. Statistical memory information
        paddle_reserved_mem_after_run = paddle.device.cuda.max_memory_reserved(local_rank)
        paddle_allocated_mem_after_run = paddle.device.cuda.max_memory_allocated(local_rank)

        model_block_memory_used = self.cal_theortical_kvcache()
        paddle_peak_increase = paddle_allocated_mem_after_run - paddle_allocated_mem_before_run

        paddle.device.cuda.empty_cache()

        after_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        available_kv_cache_memory = (
            after_run_meminfo.total * self.cache_config.gpu_memory_utilization
            - after_run_meminfo.used
            - paddle_peak_increase
        )
        available_kv_cache_memory += model_block_memory_used * self.cache_config.total_block_num

        end_time = time.perf_counter()
        logger.info(
            (
                "After running the profile, the memory usage info is as follows:",
                f"\nDevice Total memory: {after_run_meminfo.total / Gb}",
                f"\nDevice used memory: {after_run_meminfo.used / Gb}",
                f"\nDevice free memory: {after_run_meminfo.free / Gb}",
                f"\nPaddle reserved memory: {paddle_reserved_mem_after_run / Gb}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_after_run / Gb}",
                f"\nAvailable KV Cache meomory: {available_kv_cache_memory / Gb}",
                f"Profile time: {end_time - start_time}",
            )
        )

        return available_kv_cache_memory  # return to calculate the block num in this device

    def load_model(self) -> None:
        """Load model"""
        self.model_runner.load_model()

    def get_model(self) -> nn.Layer:
        """Get current model"""
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initizlize the KV Cache with accurate num_gpu_blocks"""
        # accurate cache size
        self.model_runner.update_share_input_block_num(num_gpu_blocks=num_gpu_blocks)

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_request: int = None,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch, num_running_request)
        return output

    def preprocess_new_task(self, req_dicts: List[Request], num_running_requests: int) -> None:
        """Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.model_runner.insert_tasks_v1(req_dicts=req_dicts, num_running_requests=num_running_requests)
        else:
            self.model_runner.insert_prefill_inputs(req_dicts=req_dicts, num_running_requests=num_running_requests)

    def graph_optimize_and_warm_up_model(self) -> None:
        """
        Perform the warm-up and the graph optimization
        """
        if self.fd_config.graph_opt_config.graph_opt_level >= 1 and not self.model_runner.use_cudagraph:
            self.model_runner.sot_warmup()
        # Trigger cuda graph capture
        self.model_runner.capture_model()

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """Calculate the block memory required"""
        return self.model_runner.cal_theortical_kvcache()
