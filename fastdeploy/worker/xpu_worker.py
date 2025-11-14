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

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.platforms import current_platform
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
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if paddle.is_compiled_with_xpu():
            # Set environment variable
            self.device_ids = self.parallel_config.device_ids.split(",")
            self.device = f"xpu:{self.local_rank % self.max_chips_per_node}"
            paddle.device.set_device(self.device)
            self.device_id = int(self.device_ids[self.local_rank % self.max_chips_per_node])
            assert (
                self.device_id is not None
            ), f"device_id is none for rank {self.local_rank % self.max_chips_per_node}"
            assert len(self.device_ids) > (
                self.local_rank % self.max_chips_per_node
            ), f"device number must be greater than local rank, but get device number is {len(self.device_ids)}, rank is {self.local_rank % self.max_chips_per_node}"
            paddle.set_default_dtype(self.model_config.dtype)

            gc.collect()
            paddle.device.xpu.empty_cache()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: XPUModelRunner = XPUModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            rank=self.rank,
            device_id=self.device_id,
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
        from fastdeploy.model_executor.ops.xpu import (
            xpu_get_free_global_memory,
            xpu_get_total_global_memory,
            xpu_get_used_global_memory,
        )

        total_memory = xpu_get_total_global_memory(self.device_id)
        used_memory = xpu_get_used_global_memory(self.device_id)
        free_memory = xpu_get_free_global_memory(self.device_id)

        logger.info(
            f"Before warm up, total_memory: {total_memory / 1024**3}GB, "
            f"used_memory: {used_memory / 1024**3}GB, "
            f"free_memory: {free_memory / 1024**3}GB."
        )

        if self.parallel_config.use_ep:
            logger.warning("EP mode does not support profile run.")
        else:
            self.model_runner.profile_run()
        set_random_seed(self.fd_config.model_config.seed)

        total_available_memory = int(total_memory * self.cache_config.gpu_memory_utilization)
        used_memory = xpu_get_used_global_memory(self.device_id)
        available_kv_cache_memory = total_available_memory - used_memory
        model_block_memory_used = self.cal_theortical_kvcache()
        available_kv_cache_memory += model_block_memory_used * self.cache_config.total_block_num
        if self.parallel_config.use_ep:
            available_kv_cache_memory = int(available_kv_cache_memory * 0.6)

        self.model_runner.clear_block_table()

        logger.info(
            f"After warm up, total_available_memory: {total_available_memory / 1024**3}GB, "
            f"used_memory: {used_memory / 1024**3}GB, "
            f"available_kv_cache_memory: {available_kv_cache_memory / 1024**3}GB."
        )
        paddle.device.xpu.empty_cache()
        return available_kv_cache_memory  # approximate value

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
        num_running_requests: Optional[int] = None,
        is_dummy_run: bool = False,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        return self.model_runner.execute_model(model_forward_batch, num_running_requests, is_dummy_run)

    def preprocess_new_task(self, req_dicts: List[Request], num_running_requests: int = -1) -> None:
        """Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.model_runner.insert_tasks_v1(req_dicts=req_dicts)
        else:
            self.model_runner.insert_prefill_inputs(req_dicts=req_dicts)

    def graph_optimize_and_warm_up_model(self) -> None:
        """
        Perform the warm-up and the graph optimization
        """
        if self.model_runner.graph_opt_level >= 1:
            self.model_runner.sot_warmup()

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """Calculate the block memory required"""
        return self.model_runner.cal_theortical_kvcache()
