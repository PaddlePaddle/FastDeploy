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

import argparse
import json
import os
import time
from typing import Tuple

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from fastdeploy import envs
from fastdeploy.config import (
    CacheConfig,
    DeviceConfig,
    EarlyStopConfig,
    EPLBConfig,
    ErnieArchitectures,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    PlasAttentionConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
)
from fastdeploy.eplb.async_expert_loader import (
    MODEL_MAIN_NAME,
    REARRANGE_EXPERT_MAGIC_NUM,
    create_mmap,
    load_tensor_from_shm_mem,
)
from fastdeploy.eplb.experts_manager import RedundantExpertManager
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import (
    ExistTaskStatus,
    IPCSignal,
    ModelWeightsStatus,
    RearrangeExpertStatus,
)
from fastdeploy.model_executor.layers.quantization import parse_quant_config
from fastdeploy.model_executor.utils import v1_loader_support
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import get_logger, optional_type
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("worker_process", "worker_process.log")


def get_worker(fd_config: FDConfig, local_rank: int, rank: int) -> WorkerBase:
    """
    get worker of different device
    """
    if fd_config.model_config.enable_logprob and not current_platform.is_cuda():
        raise NotImplementedError("Only CUDA platform supports logprob.")
    if current_platform.is_dcu():
        from fastdeploy.worker.dcu_worker import DcuWorker

        return DcuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_cuda():
        from fastdeploy.worker.gpu_worker import GpuWorker

        return GpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_xpu():
        from fastdeploy.worker.xpu_worker import XpuWorker

        return XpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_iluvatar():
        from fastdeploy.worker.iluvatar_worker import IluvatarWorker

        return IluvatarWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_gcu():
        from fastdeploy.worker.gcu_worker import GcuWorker

        return GcuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_maca():
        from fastdeploy.worker.metax_worker import MetaxWorker

        return MetaxWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)
    if current_platform.is_intel_hpu():
        from fastdeploy.worker.hpu_worker import HpuWorker

        return HpuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)


def init_distributed_environment(seed: int = 20) -> Tuple[int, int]:
    """Initialize Paddle Fleet and get rank of worker"""
    # Global rank
    ranks = dist.get_world_size()
    dist_strategy = fleet.DistributedStrategy()
    if ranks > 0:
        dist_strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": ranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        dist_strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=dist_strategy)

        # Local rank
        local_rank = fleet.worker_index()
    else:
        local_rank = 0
    return ranks, local_rank


def update_fd_config_for_mm(fd_config: FDConfig) -> None:
    architectures = fd_config.model_config.architectures
    if fd_config.model_config.enable_mm and ErnieArchitectures.contains_ernie_arch(architectures):
        fd_config.model_config.tensor_parallel_degree = fd_config.parallel_config.tensor_parallel_size
        fd_config.model_config.tensor_parallel_rank = fd_config.parallel_config.tensor_parallel_rank
        fd_config.model_config.vision_config.dtype = fd_config.model_config.dtype


class PaddleDisWorkerProc:
    """
    Paddle Distributed wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executes an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(self, fd_config: FDConfig, ranks: int = 1, local_rank: int = 0) -> None:
        """
        Initialize a distributed worker and task queue for single-node multi-GPU setup.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        self.ranks = ranks
        self.local_rank = local_rank
        self.fd_config = fd_config
        self.parallel_config = fd_config.parallel_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        self.eplb_config = fd_config.eplb_config

        # TODO(gongshaotian): Use worker factory to get worker
        self.worker = get_worker(fd_config=fd_config, local_rank=self.local_rank, rank=self.ranks)

        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8

    def init_health_status(self) -> None:
        """
        Initialize the health status of the worker.
        Worker Status:
            worker_ready_signal:
            worker_healthy_live_signal:
            exist_task_signal:
            exist_swapped_task_signal:
            model_weights_status:
        """
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if self.parallel_config.data_parallel_size > 1 and not envs.FD_ENABLE_MULTI_API_SERVER:
            launched_expert_service_signal_data = np.zeros(
                shape=[self.parallel_config.data_parallel_size // self.fd_config.nnode], dtype=np.int32
            )
            self.launched_expert_service_signal = IPCSignal(
                name="launched_expert_service_signal",
                array=launched_expert_service_signal_data,
                dtype=np.int32,
                suffix=self.parallel_config.engine_pid,
                create=False,
            )
            while (
                self.launched_expert_service_signal.value[
                    self.parallel_config.local_data_parallel_id % self.max_chips_per_node
                ]
                == 0
            ):
                pass

        # init worker_ready_signal
        array_size = min(
            self.max_chips_per_node,
            self.parallel_config.tensor_parallel_size * self.parallel_config.data_parallel_size,
        )

        workers_ready = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=workers_ready,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False,
        )
        self.worker_ready_signal.value[self.local_rank % self.max_chips_per_node] = 1
        # init worker_healthy_live_signal
        workers_alive = np.zeros(shape=[min(array_size, self.parallel_config.tensor_parallel_size)], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=workers_alive,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size
        self.worker_healthy_live_signal.value[local_rank % self.max_chips_per_node] = int(time.time())

        # init model_weights_status
        workers_model_weights = np.zeros(shape=[1], dtype=np.int32)
        self.model_weights_status = IPCSignal(
            name="model_weights_status",
            array=workers_model_weights,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )

        # init exist_task_signal
        workers_exist_task = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=workers_exist_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )

        # init exist_swapped_task_signal
        workers_swapped_task = np.zeros(shape=[1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=workers_swapped_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )

        # init exist_prefill_task_signal
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )

    def update_weights_from_tensor(self, mmap_infos):
        """
        update_weights_from_tensor
        """
        import time

        while True:
            if self.experts_manager.tensor_infos is None:
                time.sleep(0.1)
            else:
                break
        state_dicts = load_tensor_from_shm_mem(self.experts_manager.tensor_infos, mmap_infos[MODEL_MAIN_NAME], logger)
        rank_expert_list, logical_to_physical_map, expert_count = self.experts_manager.get_ep_rank_to_expert_id_list()
        self.worker.get_model().redundant_table_manger.update_expert_rank_table(
            rank_expert_list, logical_to_physical_map, expert_count
        )
        # TO BE FIXED
        self.worker.get_model().update_state_dict(state_dicts)
        self.experts_manager.tensor_infos = None

    def _broadcast_model_weights_signal(self, src: int, group) -> int:
        model_weights_signal_tensor = paddle.full(shape=[1], fill_value=self.model_weights_signal[0], dtype="int32")
        paddle.distributed.broadcast(model_weights_signal_tensor, src=src, group=group)
        return model_weights_signal_tensor.item()

    def _tp_barrier_wait(self):
        if current_platform.is_xpu():
            self.task_queue.worker_process_tp_barrier.wait()
        else:
            paddle.distributed.barrier(self.parallel_config.tp_group)

    def _init_eplb_signal(self):
        if not self.eplb_config.enable_eplb:
            return

        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size
        self.last_dump_expert_workload_ts = 0
        self.experts_manager = RedundantExpertManager(
            rank=self.local_rank,
            ep_size=self.ranks,
            fd_config=self.fd_config,
            ipc_signal_suffix=self.parallel_config.engine_worker_queue_port,
        )

        dp_ipc_signal_suffix = (
            f"{self.parallel_config.engine_worker_queue_port}_dp{self.parallel_config.local_data_parallel_id}"
        )
        if local_rank == 0:  # master rank0
            signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
            self.signal_update_weight_from_tensor_array = IPCSignal(
                name="signal_update_weight_from_tensor",
                array=signal_update_weight_from_tensor,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

            rearrange_experts_status = np.zeros([1], dtype=np.int32)
            self.rearrange_experts_signal = IPCSignal(
                name="rearrange_experts_status",
                array=rearrange_experts_status,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{local_rank}"
        experts_token_stats = np.zeros(
            (self.fd_config.model_config.num_hidden_layers, self.fd_config.model_config.moe_num_experts),
            dtype=np.int32,
        )
        self.local_experts_token_stats_array = IPCSignal(
            name="local_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        clear_experts_token_stats = np.zeros([1], dtype=np.int32)
        self.signal_clear_experts_token_stats = IPCSignal(
            name="signal_clear_experts_token_stats",
            array=clear_experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        self.mmap_infos = create_mmap(
            [MODEL_MAIN_NAME],
            self.local_rank,
            self.ranks,
            shm_uuid=self.parallel_config.engine_worker_queue_port,
            eplb_config=self.eplb_config,
            logger=logger,
        )

    def _run_eplb(self, tp_rank):
        """internal call to run eplb"""
        if not self.eplb_config.enable_eplb:
            return

        rearrange_time = time.time()
        # Get expert load
        if self.local_experts_token_stats_array.value is not None and (
            int(rearrange_time) - self.last_dump_expert_workload_ts
            > self.eplb_config.redundant_expert_dump_workload_interval
        ):
            self.last_dump_expert_workload_ts = int(rearrange_time)
            clear_stat = False
            if self.signal_clear_experts_token_stats.value[0] == 1:
                clear_stat = True
                self.signal_clear_experts_token_stats.value[0] = 0
            (
                new_stats_array,
                _,
                _,
                _,
            ) = self.worker.get_model().redundant_table_manger.get_expert_tokens_stats(clear_stat=clear_stat)
            self.local_experts_token_stats_array.value[:] = new_stats_array[:]
        elif self.local_experts_token_stats_array.value is None:
            logger.warning("redundant_expert: local_experts_token_stats not init")

        # All DP synchronously update weights
        broadcast_value = 0
        if tp_rank == 0 and self.signal_update_weight_from_tensor_array.value[0] == 1:
            logger.info("redundant_expert: update_weight_from_tensor broadcast signal")
            self.signal_update_weight_from_tensor_array.value[0] = 0
            broadcast_value = REARRANGE_EXPERT_MAGIC_NUM
        data = paddle.to_tensor([broadcast_value])
        paddle.distributed.broadcast(data, 0)
        if data[0] == REARRANGE_EXPERT_MAGIC_NUM:
            self.update_weights_from_tensor(self.mmap_infos)
            logger.info(
                f"redundant_expert: update_weight_from_tensor success, cost {(time.time() - rearrange_time)*1000}ms"
            )
            paddle.distributed.barrier()
            if tp_rank == 0:
                self.rearrange_experts_signal.value[0] = RearrangeExpertStatus.DONE.value
            logger.info("redundant_expert: done")

    def event_loop_normal(self) -> None:
        """Main event loop for Paddle Distributed Workers.
        TODO(gongshaotian): support remote calling of functions that control worker.
        """
        # init eplb signal
        self._init_eplb_signal()
        tp_size = self.parallel_config.tensor_parallel_size
        # Currently, only support single node
        self.nnode = int((tp_size + 7) // 8)
        req_ids = []
        num_running_requests = 0
        tp_rank = self.local_rank % tp_size

        self.model_weights_signal = np.zeros([1], dtype=np.int32)
        while True:
            # run eplb
            self._run_eplb(tp_rank)
            if tp_rank == 0:
                if self.model_weights_status.value[0] != ModelWeightsStatus.NORMAL:
                    self.model_weights_signal[0] = int(self.model_weights_status.value[0])
                if self.fd_config.load_config.dynamic_load_weight and self.parallel_config.enable_expert_parallel:
                    self.model_weights_signal[0] = self._broadcast_model_weights_signal(
                        src=0, group=self.parallel_config.ep_group
                    )
            if self.fd_config.load_config.dynamic_load_weight and tp_size > 1:
                self.model_weights_signal[0] = self._broadcast_model_weights_signal(
                    src=0, group=self.parallel_config.tp_group
                )

            self.insert_step = False
            req_dicts = None
            self.worker_healthy_live_signal.value[tp_rank % self.max_chips_per_node] = int(time.time())

            # The first worker detects whether there are tasks in the task queue
            if tp_rank == 0:
                if self.task_queue.num_tasks() > 0:
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER or not (
                        self.fd_config.model_config.enable_mm and self.worker.exist_prefill()
                    ):
                        if self.nnode > 1 and tp_size > self.max_chips_per_node:
                            self.task_queue.read_finish_flag.set(1)
                        else:
                            self.exist_task_signal.value[0] = ExistTaskStatus.EXIST

            if tp_size > 1:
                # Synchronize the signal for other workers
                self._tp_barrier_wait()

            if self.fd_config.load_config.dynamic_load_weight:
                if self.parallel_config.enable_expert_parallel:
                    paddle.distributed.barrier(self.parallel_config.ep_group)
                else:
                    paddle.distributed.barrier(self.parallel_config.tp_group)
                if self.model_weights_signal[0] != ModelWeightsStatus.NORMAL:
                    logger.info(
                        f"Rank: {self.local_rank} to update or clear parameters, signal is {self.model_weights_signal[0]}, [-1:clear, 1:update]"
                    )
                    from fastdeploy.rl.dynamic_weight_manager import (
                        DynamicWeightManager,
                    )

                    self.model_weights_status.value[0] = self.model_weights_signal[0]
                    DynamicWeightManager.check_model_weights_status(
                        self.model_weights_status,
                        # model_weights_signal
                        self.worker.model_runner,
                        self.parallel_config.engine_worker_queue_port,
                    )
                    logger.info(f"current task queue data: {self.task_queue.num_tasks()}")
                    self.task_queue.clear_data()
                    self.model_weights_signal[0] = ModelWeightsStatus.NORMAL
                    logger.info(f"Rank: {self.local_rank} has updated or cleared parameters.")

            if self.exist_task_signal.value[0] == ExistTaskStatus.EXIST or self.task_queue.read_finish_flag.get() == 1:
                logger.info(f"Rank: {self.local_rank} Detected new requests.")
                self.insert_step = True

                tasks, read_finish = self.task_queue.get_tasks()
                if read_finish:
                    # Ensure that every worker get the task
                    self.exist_task_signal.value[0] = ExistTaskStatus.EMPTY
                    self.task_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)
                    req_dicts.extend(req_dict)

                req_ids = [req.request_id for req in req_dicts]
                logger.info(
                    f"Rank: {self.local_rank}, num_running_requests: {num_running_requests}, "
                    f"num_insert_requests: {len(req_dicts)}, req_ids: {req_ids}"
                )

                # Process prefill inputs
                self.worker.preprocess_new_task(req_dicts, num_running_requests)

            if (not self.parallel_config.use_ep) and (not self.worker.model_runner.not_need_stop()):
                if self.ranks > 1:
                    self._tp_barrier_wait()

                time.sleep(0.001)
                continue

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            start_execute_time = time.time()
            self.worker.execute_model(req_dicts, num_running_requests)
            self.exist_prefill_task_signal.value[0] = self.worker.exist_prefill()
            logger.debug(f"execute model cost: {time.time()-start_execute_time:.5f} s")

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
            logger.info(f"------- model_block_memory_used:{model_block_memory_used / 1024**3} GB --------")
            logger.info(f"------- num_blocks_local:{num_blocks_local} --------")

            if num_blocks_local <= 0:
                raise ValueError(
                    "The total number of blocks cannot be less than zero. "
                    "Please increase gpu_memory_utilization "
                    "Or decrease max_num_batched_tokens(max model length)."
                )

            if self.ranks > 1:
                num_blocks_local = paddle.full(shape=[1], fill_value=num_blocks_local, dtype="int32")
                dist.all_reduce(num_blocks_local, op=dist.ReduceOp.MIN)
                num_blocks_local = num_blocks_local.item()

            if self.local_rank % self.max_chips_per_node == 0:
                # 3. Send IPCSignal
                get_profile_block_num = np.zeros(shape=[1], dtype=np.int32)
                self.get_profile_block_num_signal = IPCSignal(
                    name="get_profile_block_num",
                    array=get_profile_block_num,
                    dtype=np.int32,
                    suffix=self.parallel_config.engine_pid,
                    create=False,
                )
                self.get_profile_block_num_signal.value[0] = num_blocks_local
        else:
            num_blocks_local = self.fd_config.cache_config.total_block_num
        logger.info(f"------- num_blocks_global: {num_blocks_local} --------")

        # 4. init kv_cache with accurate num_blocks
        self.worker.initialize_cache(num_gpu_blocks=num_blocks_local)

    def graph_optimize_and_warm_up_model(self) -> None:
        self.worker.graph_optimize_and_warm_up_model()
        # reset cache_messager prefilled_step signal
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER and self.scheduler_config.splitwise_role == "prefill":
            gpu_id = self.worker.model_runner.device_id
            prefilled_step_name = f"splitwise_complete_prefilled_step_{self.local_rank}"
            prefilled_step_idx_data = np.zeros(shape=[1], dtype=np.int32)
            step_shm_value = IPCSignal(
                name=prefilled_step_name, array=prefilled_step_idx_data, dtype=np.int32, suffix=gpu_id, create=False
            )
            step_shm_value.value[0] = -1

    def init_device(self) -> None:
        """Initialize device and Construct model runner"""
        self.worker.init_device()

    def start_task_queue_service(self):
        # Initialize task queue
        if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
            task_address = (
                self.parallel_config.pod_ip,
                self.parallel_config.engine_worker_queue_port,
            )
        else:
            task_address = f"/dev/shm/fd_task_queue_{self.parallel_config.engine_worker_queue_port}.sock"
        logger.info(f"connect task queue address {task_address}")
        self.task_queue = TaskQueue(
            address=task_address,
            is_server=False,
            num_client=self.parallel_config.tensor_parallel_size,
            client_id=self.parallel_config.tensor_parallel_rank,
            local_data_parallel_id=self.parallel_config.local_data_parallel_id,
        )

    def load_model(self) -> None:
        """Load weights and create model"""

        self.worker.load_model()
        loaded_model_signal_data = np.zeros(shape=[1], dtype=np.int32)
        self.loaded_model_signal = IPCSignal(
            name="loaded_model_signal",
            array=loaded_model_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False,
        )
        if self.ranks > 1:
            paddle.distributed.barrier()
        self.loaded_model_signal.value[0] = 1


def parse_args():
    """
    Parse args from command line
    """
    parser = argparse.ArgumentParser("FastDeploy LLM Inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./output",
        help="model dir",
    )
    parser.add_argument("-mbs", "--max_num_seqs", type=int, default=34, help="max batch size")
    parser.add_argument("--num_gpu_blocks_override", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--pod_ip", type=str, default="127.0.0.1")
    parser.add_argument("--engine_worker_queue_port", type=str, default="9923")
    parser.add_argument("--max_model_len", type=int, default=3072, help="max model len")
    parser.add_argument("--device_ids", type=str, default="0", help="cuda visible devices")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="input dtype")
    parser.add_argument("--enc_dec_block_num", type=int, default=1, help="encoder's decoder num")
    parser.add_argument(
        "--kv_cache_ratio",
        type=float,
        default=0.7,
        help="kv cache ratio for input",
    )
    parser.add_argument("--first_token_id", type=int, default=1, help="first token id")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="gpu memory utilization",
    )
    parser.add_argument("--engine_pid", type=int, default=None, help="Process ID of engine")
    parser.add_argument("--do_profile", action="store_true", help="do profile or not")
    parser.add_argument("--pad_token_id", type=int, default=-1, help="pad token id")
    parser.add_argument("--eos_tokens_lens", type=int, default=2, help="eos token lens")
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        help="enable chunked prefill",
    )
    parser.add_argument(
        "--use_internode_ll_two_stage",
        action="store_true",
        help="enable internode_ll_two_stage",
    )
    parser.add_argument(
        "--speculative_config",
        type=json.loads,
        default=None,
        help="Configuration of SpeculativeConfig.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=2048,
        help="max num batched tokens",
    )

    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="enable prefix cache",
    )
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="enable custom all-reduce",
    )
    parser.add_argument(
        "--disable_sequence_parallel_moe",
        action="store_true",
        help="disable sequence parallel moe",
    )
    parser.add_argument("--splitwise_role", type=str, default="mixed", help="splitwise role")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size",
    )
    parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=1,
        help="expert parallel size",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="data parallel size",
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action="store_true",
        help="enable expert parallel",
    )
    parser.add_argument("--ori_vocab_size", type=int, default=None)
    parser.add_argument("--think_end_id", type=int, default=-1)
    parser.add_argument("--image_patch_id", type=int, default=-1)
    parser.add_argument("--line_break_id", type=int, default=-1)

    parser.add_argument(
        "--quantization",
        type=json.loads,
        default=None,
        help="Quantization name for the model, currently support "
        "'wint4', 'wint8',"
        "default is None. The priority of this configuration "
        "is lower than that of the config file. "
        "More complex quantization methods need to be configured via the config file.",
    )
    parser.add_argument(
        "--graph_optimization_config",
        type=json.loads,
        default=None,
        help="Configuration of Graph optimization backend.",
    )
    parser.add_argument(
        "--plas_attention_config",
        type=json.loads,
        default=None,
        help="Configation of plas attention.",
    )
    parser.add_argument(
        "--guided_decoding_backend",
        type=str,
        default="off",
        help="guided decoding backend",
    )
    parser.add_argument(
        "--disable_any_whitespace",
        action="store_true",
        help="Disable any whitespace for guided decoding.",
    )
    parser.add_argument(
        "--dynamic_load_weight",
        action="store_true",
        help="Enable dynamic weight loading strategy",
    )
    parser.add_argument(
        "--load_strategy",
        type=str,
        choices=["ipc", "ipc_snapshot", "meta", "normal"],
        default="ipc_snapshot",
        help="Weight loading method when dynamic loading is enabled: "
        "'ipc': real-time IPC streaming with automatic resharding, "
        "'ipc_snapshot': load from disk snapshot of IPC weights.",
    )
    parser.add_argument(
        "--enable_logprob",
        action="store_true",
        help="Enable output of token-level log probabilities.",
    )
    parser.add_argument(
        "--max_logprobs",
        type=int,
        default=20,
        help="Maximum number of log probabilities.",
    )
    parser.add_argument(
        "--logprobs_mode",
        type=str,
        default="raw_logprobs",
        help="Indicates the content returned in the logprobs.",
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default=None,
        help="Flag specifies the reasoning parser to use for extracting reasoning content from the model output",
    )
    parser.add_argument(
        "--early_stop_config",
        type=json.loads,
        default=None,
        help="Configuration of early stop.",
    )

    parser.add_argument(
        "--load_choices",
        type=str,
        default="default",
        help="The format of the model weights to load. default/new_loader.",
    )

    parser.add_argument(
        "--ips",
        type=str,
        default=None,
        help="The ips of multinode deployment.",
    )

    parser.add_argument(
        "--lm_head_fp32",
        action="store_true",
        help="Flag to specify dtype of lm_head as FP32",
    )

    parser.add_argument(
        "--max_encoder_cache",
        type=int,
        help="Maximum encoder cache tokens(use 0 to disable).",
    )

    parser.add_argument(
        "--cache-transfer-protocol",
        type=str,
        default="ipc",
        help="support protocol list, comma separated, default is ipc",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="auto",
        help="The type of model runner to use.Each FD instance only supports one model runner.even if the same model can be used for multiple types.",
    )

    parser.add_argument(
        "--convert",
        type=str,
        default="auto",
        help="Convert the model using adapters. The most common use case is to adapt a text generation model to be used for pooling tasks.",
    )

    parser.add_argument(
        "--override-pooler-config",
        type=optional_type(json.loads),
        default=None,
        help="Override configuration for the pooler.",
    )

    parser.add_argument(
        "--logits-processors",
        type=str,
        nargs="+",
        default=[],
        help="FQCNs (Fully Qualified Class Names) of logits processors supported by the service.",
    )

    parser.add_argument(
        "--eplb_config",
        type=json.loads,
        default=None,
        help="EPLB Configuration.",
    )

    args = parser.parse_args()
    return args


def initialize_fd_config(args, ranks: int = 1, local_rank: int = 0) -> FDConfig:
    """Initialize FDConfig from either RolloutModelConfig or argparse.Namespace

    Args:
        config: Configuration object containing all parameters (either RolloutModelConfig or argparse.Namespace)

    Returns:
        FDConfig: Initialized FastDeploy configuration object
    """
    # RL rollout
    paddle.set_default_dtype(args.dtype)
    model_config = ModelConfig(vars(args))
    device_config = DeviceConfig(vars(args))
    speculative_config = SpeculativeConfig(args.speculative_config)
    parallel_config = ParallelConfig(vars(args))
    cache_config = CacheConfig(vars(args))
    scheduler_config = SchedulerConfig(vars(args))

    parallel_config.tensor_parallel_rank = local_rank % parallel_config.tensor_parallel_size
    parallel_config.data_parallel_rank = local_rank // parallel_config.tensor_parallel_size
    # config for EP
    if parallel_config.expert_parallel_size > 1:
        expert_parallel_rank = int(local_rank % parallel_config.expert_parallel_size)
        if isinstance(model_config.moe_num_experts, list):
            num_experts = model_config.moe_num_experts[0]
        else:
            num_experts = model_config.moe_num_experts
        num_experts_per_rank = num_experts // parallel_config.expert_parallel_size
        num_experts_start_offset = expert_parallel_rank * num_experts_per_rank
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        parallel_config.local_data_parallel_id = parallel_config.data_parallel_rank % (
            max_chips_per_node // parallel_config.tensor_parallel_size
        )

        parallel_config.expert_parallel_rank = expert_parallel_rank
        parallel_config.num_experts_per_rank = num_experts_per_rank
        parallel_config.num_experts_start_offset = num_experts_start_offset

    if args.load_strategy != "meta":
        parallel_config.engine_worker_queue_port = parallel_config.engine_worker_queue_port[
            parallel_config.local_data_parallel_id
        ]
    parallel_config.set_communicate_group()

    load_config = LoadConfig(vars(args))

    graph_opt_config = GraphOptimizationConfig(args.graph_optimization_config)

    plas_attention_config = PlasAttentionConfig(args.plas_attention_config)

    early_stop_config = EarlyStopConfig(args.early_stop_config)
    eplb_config = EPLBConfig(args.eplb_config)

    structured_outputs_config: StructuredOutputsConfig = StructuredOutputsConfig(args=vars(args))

    # Note(tangbinhan): used for load_checkpoint
    model_config.pretrained_config.tensor_parallel_rank = parallel_config.tensor_parallel_rank
    model_config.pretrained_config.tensor_parallel_degree = parallel_config.tensor_parallel_size
    model_config.pretrained_config.is_mtp = False
    model_config.pretrained_config.head_dim = model_config.head_dim

    logger.info(f"parallel_config.use_ep {parallel_config.use_ep}")
    logger.info(f"parallel_config.tensor_parallel_size {parallel_config.tensor_parallel_size}")
    logger.info(f"parallel_config.tensor_parallel_rank {parallel_config.tensor_parallel_rank}")
    logger.info(f"parallel_config.engine_worker_queue_port {parallel_config.engine_worker_queue_port}")

    if getattr(model_config, "num_hidden_layers", None) is None:
        raise ValueError("num_hidden_layers is None")

    quant_config = parse_quant_config(
        args,
        model_config,
        is_ernie=ErnieArchitectures.contains_ernie_arch(model_config.architectures),
        is_v1_loader=load_config.load_choices == "default_v1",
    )

    # Log quantization info
    logger.info("===========quantization_config==============")
    if quant_config is not None:
        if model_config.is_quantized:
            logger.info("Model Status: Offline Quantized (pre-quantized weights loaded)")
        else:
            logger.info("Model Status: Original (will apply online quantization)")

        logger.info(f"{model_config.quantization_config}")
    else:
        logger.info("No quantization config found and use original weight and act dtype.")

    logger.info(f"- Dynamic load weight: {load_config.dynamic_load_weight}")
    logger.info(f"- Load strategy: {load_config.load_strategy}")

    if not (current_platform.is_cuda() or current_platform.is_xpu() or current_platform.is_maca()):
        logger.info("Set ENABLE_V1_KVCACHE_SCHEDULER to 0 due to not supported.")
        envs.ENABLE_V1_KVCACHE_SCHEDULER = 0

    if envs.ENABLE_V1_KVCACHE_SCHEDULER and args.splitwise_role == "prefill":
        os.environ["PREFILL_NODE_ONE_STEP_STOP_V1"] = "1"

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=device_config,
        load_config=load_config,
        quant_config=quant_config,
        graph_opt_config=graph_opt_config,
        early_stop_config=early_stop_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        ips=args.ips,
        plas_attention_config=plas_attention_config,
        structured_outputs_config=structured_outputs_config,
        eplb_config=eplb_config,
    )
    update_fd_config_for_mm(fd_config)
    if fd_config.load_config.load_choices == "default_v1" and not v1_loader_support(fd_config):
        fd_config.load_config.load_choices = "default"

    architecture = fd_config.model_config.architectures[0]
    if "PaddleOCR" in architecture:
        envs.FD_ENABLE_MAX_PREFILL = 1
        fd_config.cache_config.enable_prefix_caching = False
        fd_config.cache_config.max_encoder_cache = 0
    return fd_config


def run_worker_proc() -> None:
    """
    start worker process
    """
    # Get args form Engine
    args = parse_args()

    ranks, local_rank = init_distributed_environment()

    # Get fd_config
    fd_config = initialize_fd_config(args, ranks, local_rank)

    # Create worker process
    if current_platform.is_iluvatar():
        from fastdeploy.worker.iluvatar_worker import IluvatarPaddleDisWorkerProc

        worker_proc = IluvatarPaddleDisWorkerProc(fd_config, ranks, local_rank)
    else:
        worker_proc = PaddleDisWorkerProc(fd_config, ranks, local_rank)

    # Initialize device and create model runner
    worker_proc.init_device()

    # Load model
    worker_proc.load_model()
    # Initialize KV Cache
    worker_proc.initialize_kv_cache()

    # Trigger CUDAGraph capture
    worker_proc.graph_optimize_and_warm_up_model()

    # Initialize health status
    worker_proc.init_health_status()

    worker_proc.start_task_queue_service()

    # Start event loop
    worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()
