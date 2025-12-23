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

import numpy as np
import paddle

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import get_logger, set_random_seed
from fastdeploy.worker.gpu_worker import GpuWorker
from fastdeploy.worker.iluvatar_model_runner import IluvatarModelRunner
from fastdeploy.worker.worker_process import PaddleDisWorkerProc

logger = get_logger("iluvatar_worker", "iluvatar_worker.log")


class IluvatarWorker(GpuWorker):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        local_rank: int,
        rank: int,
    ):
        if fd_config.model_config.enable_mm:
            paddle.set_flags({"FLAGS_enable_ixattnbkd": True, "FLAGS_enable_ixdnn_attn": False})
        super(IluvatarWorker, self).__init__(
            fd_config=fd_config,
            local_rank=local_rank,
            rank=rank,
        )

    def init_device(self):
        """
        Initialize device and construct model runner
        """
        if paddle.is_compiled_with_custom_device("iluvatar_gpu"):
            # Set environment variable
            self.device = f"iluvatar_gpu:{self.local_rank}"
            paddle.device.set_device(self.device)
            paddle.set_default_dtype(self.model_config.dtype)
            self.device_ids = self.parallel_config.device_ids.split(",")

            gc.collect()
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        set_random_seed(self.fd_config.model_config.seed)
        # Construct model runner
        self.model_runner: IluvatarModelRunner = IluvatarModelRunner(
            fd_config=self.fd_config,
            device=self.device,
            device_id=self.device_ids[self.local_rank],
            rank=self.rank,
            local_rank=self.local_rank,
        )

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
        return int(float(os.getenv("FD_ILUVATAR_KVCACHE_MEM", "3")) * 1024**3)


# TODO (yuzhe.wu): move it int work_process.py after baidu reconstructs the logic of workproc
class IluvatarPaddleDisWorkerProc(PaddleDisWorkerProc):
    """
    Paddle Distributed wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executes an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(self, fd_config: FDConfig, ranks: int = 1, local_rank: int = 0):
        super(IluvatarPaddleDisWorkerProc, self).__init__(
            fd_config=fd_config,
            ranks=ranks,
            local_rank=local_rank,
        )

    def initialize_kv_cache(self) -> None:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if self.fd_config.parallel_config.do_profile:
            # 1. Get available memory(bytes)
            available_kv_cache_memory = self.worker.determine_available_memory()
            logger.info(f"------- available_kv_cache_memory:{available_kv_cache_memory / 1024**3} GB --------")

            # 2. Calculate the appropriate number of blocks
            model_block_memory_used = self.worker.cal_theortical_kvcache()
            num_blocks_local = int(available_kv_cache_memory // model_block_memory_used)
            # NOTE(liuzichang): Too many block will lead to illegal memory access
            # We will develop dynamic limits in future.
            if num_blocks_local > 40000:
                logger.info(f"------- Reset num_blocks_local {num_blocks_local} to 40000")
                num_blocks_local = min(40000, num_blocks_local)
            logger.info(f"------- model_block_memory_used:{model_block_memory_used} --------")
            logger.info(f"------- num_blocks_local:{num_blocks_local} --------")

            # NOTE(yuzhe.wu): Using the old version of the calculation num_blocks_global method,
            # because the new version that adopting allreduce min will report a bad request error
            # when running 300b model. The Relation commit:
            # https://github.com/PaddlePaddle/FastDeploy/commit/2f74e93d7e87aa3ffec3fc6966bf11ab5363b956

            # 3. Send IPCSignal
            get_profile_block_num = np.zeros(shape=[self.ranks], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.parallel_config.engine_pid,
                create=False,
            )
            self.get_profile_block_num_signal.value[self.local_rank] = num_blocks_local

            # Wait all worker send the signal
            while np.any(self.get_profile_block_num_signal.value <= 0):
                time.sleep(0.01)
            num_blocks_global = self.get_profile_block_num_signal.value.min().item()

            if num_blocks_global < 0:
                logger.error(
                    "The total number of blocks cannot be less than zero."
                    "Please increase gpu_memory_utilization"
                    "Or decrease max_num_batched_tokens(max model length) "
                )
                raise ValueError(
                    "The total number of blocks cannot be less than zero."
                    "Please increase gpu_memory_utilization"
                    "Or decrease max_num_batched_tokens(max model length) "
                )

            self.get_profile_block_num_signal.value[self.local_rank] = num_blocks_global
        else:
            num_blocks_global = self.fd_config.cache_config.total_block_num
        # 4. init kv_cache with accurate num_blocks
        logger.info(f"------- num_blocks_global:{num_blocks_global} --------")
        self.worker.initialize_cache(num_gpu_blocks=num_blocks_global)
