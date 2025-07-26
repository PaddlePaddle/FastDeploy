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
from fastdeploy.utils import get_logger
from fastdeploy.worker.metax_model_runner import MetaxModelRunner
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("gpu_worker", "gpu_worker.log")


class MetaxWorker(WorkerBase):
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
        self.max_chips_per_node = 8
        if paddle.is_compiled_with_custom_device("metax_gpu"):
            # Set evironment variable
            self.device_ids = self.parallel_config.device_ids.split(",")
            self.device = f"metax_gpu:{self.local_rank % self.max_chips_per_node}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.parallel_config.dtype)

            gc.collect()
            paddle.device.cuda.empty_cache()
            if self.parallel_config.enable_custom_all_reduce:
                from fastdeploy.distributed.communication import use_custom_allreduce

                use_custom_allreduce()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct model runner
        self.model_runner: MetaxModelRunner = MetaxModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            device_id=self.device_ids[self.local_rank % self.max_chips_per_node],
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
        """Will implement later"""
        self.model_runner.profile_run()
        return 1024**3*16


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
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch)
        return output

    def preprocess_new_task(self, req_dicts: List[Request]) -> None:
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
        # Triger cuda grpah capture
        self.model_runner.capture_model()

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """Calculate the block memory required"""
        return self.model_runner.cal_theortical_kvcache()
