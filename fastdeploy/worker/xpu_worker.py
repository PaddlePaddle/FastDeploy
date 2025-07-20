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

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.utils import get_logger
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

        # Construct model runner
        self.model_runner: XPUModelRunner = XPUModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            rank=self.rank,
            local_rank=self.local_rank,
        )

    def graph_optimize_and_warm_up_model(self) -> None:
        """
        Optimizes the inference graph using the specified optimization options.
        """
        logger.warn("XPU current could not graph optimize and warm up model")

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

        total_memory = xpu_get_total_global_memory(self.local_rank)
        used_memory = xpu_get_used_global_memory(self.local_rank)
        free_memory = xpu_get_free_global_memory(self.local_rank)

        logger.info(
            f"Before warm up, total_memory: {total_memory}, \
                    used_memory: {used_memory}, free_memory: {free_memory}"
        )

        self.model_runner.prepare_profile()
        self.model_runner.profile_run()

        total_available_memory = int(total_memory * self.parallel_config.gpu_memory_utilization)
        used_memory = xpu_get_used_global_memory(self.local_rank)
        available_kv_cache_memory = total_available_memory - used_memory
        model_block_memory_used = self.cal_theortical_kvcache()
        available_kv_cache_memory += model_block_memory_used * self.parallel_config.total_block_num

        self.model_runner.clear_block_table()

        logger.info(
            f"After warm up, total_available_memory: {total_available_memory}, \
                    used_memory: {used_memory}, available_kv_cache_memory: {available_kv_cache_memory}"
        )
        paddle.device.xpu.empty_cache()
        return available_kv_cache_memory  # approximate value

    def cal_theortical_kvcache(self) -> int:
        """ """
        return self.model_runner.cal_theortical_kvcache()

    def load_model(self) -> None:
        """ """
        self.model_runner.load_model()

    def get_model(self) -> nn.Layer:
        """ """
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """ """
        pass

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch)
        return output

    def prefill_finished(self):
        """
        check whether prefill stage finished
        """
        return self.model_runner.prefill_finished()

    def preprocess_new_task(self, req_dicts: List[Request]) -> None:
        """Process new requests and then start the decode loop
        TODO(gongshaotian):The scheduler should schedule the handling of prefill,
        and workers and modelrunners should not perceive it.
        """
        self.model_runner.process_prefill_inputs(req_dicts=req_dicts)

    def check_health(self) -> bool:
        """ """
        return True

    def reinitialize_kv_cache(self, num_gpu_blocks: int) -> None:
        """ """
        self.model_runner.update_share_input_block_num(num_gpu_blocks=num_gpu_blocks)
