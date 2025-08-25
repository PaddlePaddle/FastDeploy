import gc
from typing import List, Optional

import paddle
import paddle.nn as nn
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.utils import get_logger
from fastdeploy.worker.npu_model_runner import NPUModelRunner
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.worker_base import WorkerBase
from fastdeploy import envs

logger = get_logger("npu_worker", "npu_worker.log")


class NpuWorker(WorkerBase):
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

    def init_device(self):
        """Initialize device and Construct model runner"""
        if paddle.is_compiled_with_custom_device("npu"):
            # Set evironment variable
            self.device = f"npu:{self.local_rank}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.parallel_config.dtype)
            self.device_ids = self.parallel_config.device_ids.split(",")

            gc.collect()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device_type}"
            )

        # Construct model runner
        self.model_runner: NPUModelRunner = NPUModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            device_id=self.device_ids[self.local_rank],
            rank=self.rank,
            local_rank=self.local_rank,
        )

    def prefill_finished(self):
        """
        check whether prefill stage finished
        """
        return self.model_runner.prefill_finished()

    def determine_available_memory(self) -> int:
        # TODO: guozr 这里因为缺失 api 导致无法计算真实的显存
        # return 60 * 0.1 * 1024**3
        return 1024**3

    def load_model(self) -> None:
        """ """
        self.model_runner.load_model()

    def get_model(self) -> nn.Layer:
        """ """
        return self.model_runner.get_model()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initizlize the KV Cache with accurate num_gpu_blocks"""
        # accurate cache size
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
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.model_runner.insert_tasks_v1(req_dicts=req_dicts, num_running_requests=num_running_requests)
        else:
            self.model_runner.insert_prefill_inputs(req_dicts=req_dicts, num_running_requests=num_running_requests)

    def graph_optimize_and_warm_up_model(self) -> None:
        """ """
        pass

    def check_health(self) -> bool:
        """ """
        return True

    def cal_theortical_kvcache(self) -> int:
        """ """
        return self.model_runner.cal_theortical_kvcache()

    def reinitialize_kv_cache(self, num_gpu_blocks: int) -> None:
        """ """
        self.model_runner.update_share_input_block_num(num_gpu_blocks=num_gpu_blocks)
        # TODO:
        # pass
