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
import os
import time
from typing import List, Optional

import paddle
import paddle.nn as nn
from paddle.base import core

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.utils import get_logger, set_random_seed
from fastdeploy.worker.hpu_model_runner import HPUModelRunner
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("hpu_worker", "hpu_worker.log")


def max_memory_allocated(device_id: int) -> int:
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device_id: int) -> int:
    return core.device_memory_stat_peak_value("Reserved", device_id)


def reset_max_memory_allocated(device_id: int) -> None:
    core.device_memory_stat_reset_peak_value("Allocated", device_id)


def reset_max_memory_reserved(device_id: int) -> None:
    core.device_memory_stat_reset_peak_value("Reserved", device_id)


class HpuWorker(WorkerBase):
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
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            # Set environment variable
            self.device_ids = self.parallel_config.device_ids.split(",")
            logger.info(
                f"Using Intel HPU device with local rank => device id: {int(self.device_ids[self.local_rank])} as module id"
            )
            intel_hpus_module_id = int(self.device_ids[self.local_rank])
            self.device = f"intel_hpu:{intel_hpus_module_id}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.model_config.dtype)

            gc.collect()
            paddle.device.cuda.empty_cache()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: HPUModelRunner = HPUModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            device_id=self.device_ids[self.local_rank],
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
        module_id = int(self.device_ids[self.local_rank])
        reset_max_memory_allocated(module_id)
        reset_max_memory_reserved(module_id)
        paddle_reserved_mem_before_run = max_memory_reserved(module_id)
        paddle_allocated_mem_before_run = max_memory_allocated(module_id)  # not reserved

        logger.info(
            (
                "Before running the profile, the memory usage info is as follows:",
                f"\nPaddle reserved memory: {paddle_reserved_mem_before_run}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_before_run}",
            )
        )

        # 2. Profile run
        self.model_runner.profile_run()

        # 3. Statistical memory information
        paddle_reserved_mem_after_run = max_memory_reserved(module_id)
        paddle_allocated_mem_after_run = max_memory_allocated(module_id)

        one_mb = 1024 * 1024
        one_gb = 1024 * one_mb
        hpu_reserved_memory = 768 * one_mb  # 768MB reserved for not paddle use memory
        hpu_total_memory = 96 * one_gb  # 96GB HPU memory
        peak_memory = paddle_allocated_mem_after_run + hpu_reserved_memory
        available_kv_cache_memory = hpu_total_memory * self.cache_config.gpu_memory_utilization - peak_memory

        end_time = time.perf_counter()
        logger.info(
            (
                "After running the profile, the memory usage info is as follows:",
                f"\nPaddle reserved memory: {paddle_reserved_mem_after_run}",
                f"\nPaddle allocated memory: {paddle_allocated_mem_after_run}",
                f"\nAvailable KV Cache meomory: {available_kv_cache_memory}",
                f"Profile time: {end_time - start_time}",
            )
        )

        return available_kv_cache_memory  # return to caculate the block num in this device

    def load_model(self) -> None:
        """Load model"""
        self.model_runner.load_model()

    def get_model(self) -> nn.Layer:
        """Get current model"""
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initialize the KV Cache with accurate num_gpu_blocks"""
        # accurate cache size
        self.model_runner.update_share_input_block_num(num_gpu_blocks=num_gpu_blocks)

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_request: int = None,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch)
        return output

    def preprocess_new_task(self, req_dicts: List[Request], num_running_requests: int) -> None:
        """Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        self.model_runner.insert_prefill_inputs(req_dicts=req_dicts, num_running_requests=num_running_requests)

    def graph_optimize_and_warm_up_model(self) -> None:
        """
        Perform the warm-up and the graph optimization
        """
        # wait for all cards loading model completely.
        if self.rank > 1:
            paddle.distributed.barrier()
        # 1. Warm up model
        # NOTE(gongshaotian): may be not need warm_up at this place
        if int(os.environ.get("HPU_WARMUP_BUCKET", 0)) == 1:
            logger.info("Warmup bucket is enabled, start warmup bucket")
            self.model_runner.is_warmuping = True
            self.model_runner.warm_up_bucket()
            self.model_runner.is_warmuping = False
        else:
            logger.info("Skipping warmup bucket, please set HPU_WARMUP_BUCKET=1 to enable it.")

        # 2. Triger cuda grpah capture
        self.model_runner.capture_model()

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """Calculate the block memory required"""
        return self.model_runner.cal_theortical_kvcache()
