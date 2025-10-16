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
from fastdeploy.utils import get_logger, set_random_seed
from fastdeploy.worker.gcu_model_runner import GCUModelRunner
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("gcu_worker", "gcu_worker.log")


class GcuWorker(WorkerBase):
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
        if paddle.is_compiled_with_custom_device("gcu"):
            # Set environment variable
            self.device_ids = self.parallel_config.device_ids.split(",")
            self.device = f"gcu:{self.local_rank}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.model_config.dtype)
            logger.info(f"GcuWorker init_device:{self.device}, device_ids:{self.device_ids}")

            gc.collect()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: GCUModelRunner = GCUModelRunner(
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
        Then, it calculate the maximum possible number of GCU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GCU memory
            by adjusting the `gcu_memory_utilization` parameter.
        """
        raise NotImplementedError

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
        num_running_requests: int = None,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        output = self.model_runner.execute_model(model_forward_batch, num_running_requests)
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
        # 1. Warm up model
        # NOTE(gongshaotian): may be not need warm_up at this place
        if self.model_runner.graph_opt_level >= 1:
            self.model_runner.sot_warmup()
        # 2. Trigger cuda graph capture
        self.model_runner.capture_model()
        set_random_seed(self.fd_config.model_config.seed)

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """ """
        return self.model_runner.cal_theortical_kvcache()
