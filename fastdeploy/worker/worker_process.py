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
import time
from typing import List

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from fastdeploy import envs
from fastdeploy.config import (DecodingConfig, DeviceConfig, FDConfig,
                               GraphOptimizationConfig, LoadConfig,
                               ModelConfig, MoEConfig, MoEPhase,
                               ParallelConfig, SpeculativeConfig)
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.model_executor.layers.quantization import \
    get_quantization_config
from fastdeploy.platforms import current_platform
from fastdeploy.utils import get_logger, none_or_str
from fastdeploy.worker.worker_base import WorkerBase

logger = get_logger("worker_process", "worker_process.log")


def get_worker(fd_config: FDConfig, local_rank: int, rank: int) -> WorkerBase:
    """
    get worker of different device
    """
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
        return IluvatarWorker(fd_config=fd_config,
                              local_rank=local_rank,
                              rank=rank)
    if current_platform.is_gcu():
        from fastdeploy.worker.gcu_worker import GcuWorker
        return GcuWorker(fd_config=fd_config, local_rank=local_rank, rank=rank)


class PaddleDisWorkerProc():
    """
    Paddle Distrubuted wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executea an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(
        self,
        fd_config: FDConfig,
    ) -> None:
        """
        Initialize a distributed worker and task queue for single-node multi-GPU setup.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        self.fd_config = fd_config
        self.parallel_config = fd_config.parallel_config

        # Initialize distributed enviroment
        (self.ranks, self.local_rank) = self.init_distributed_enviroment()

        assert self.parallel_config.tensor_parallel_degree * self.parallel_config.expert_parallel_degree == self.ranks

        self.fd_config.parallel_config.tensor_parallel_rank = \
            self.local_rank % self.parallel_config.tensor_parallel_degree
        self.fd_config.parallel_config.expert_parallel_rank = \
            int(self.local_rank / self.parallel_config.tensor_parallel_degree)

        if self.fd_config.parallel_config.use_ep:
            self.fd_config.moe_config.num_experts_per_rank = \
                self.fd_config.moe_config.num_experts // self.parallel_config.expert_parallel_degree
            self.fd_config.moe_config.num_experts_start_offset = \
                self.fd_config.parallel_config.expert_parallel_rank * self.fd_config.moe_config.num_experts_per_rank

        # For auto TP split
        self.fd_config.model_config.tensor_parallel_degree = self.parallel_config.tensor_parallel_degree
        self.fd_config.model_config.tensor_parallel_rank = self.parallel_config.tensor_parallel_rank
        self.fd_config.model_config.use_ep = self.parallel_config.use_ep

        if self.fd_config.parallel_config.use_ep:
            self.fd_config.model_config.num_experts_per_rank = self.fd_config.moe_config.num_experts_per_rank
            self.fd_config.model_config.num_experts_start_offset = self.fd_config.moe_config.num_experts_start_offset

        # TODO(gongshaotian): Use worker factory to get worker
        self.worker = get_worker(fd_config=fd_config,
                                 local_rank=self.local_rank,
                                 rank=self.ranks)

        # Initialize task queue
        task_address = (self.parallel_config.pod_ip,
                        self.parallel_config.engine_worker_queue_port)

        self.task_queue = TaskQueue(
            address=task_address,
            is_server=False,
            num_client=self.parallel_config.tensor_parallel_degree,
            client_id=self.parallel_config.tensor_parallel_rank,
            local_data_parallel_id=self.fd_config.parallel_config.
            expert_parallel_rank)

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
        # init worker_ready_signal
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        array_size = min(
            max_chips_per_node, self.parallel_config.tensor_parallel_degree *
            self.parallel_config.expert_parallel_degree)
        workers_ready = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=workers_ready,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)
        self.worker_ready_signal.value[self.local_rank %
                                       max_chips_per_node] = 1

        # init worker_healthy_live_signal
        workers_alive = np.zeros(shape=[self.ranks], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=workers_alive,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)
        self.worker_healthy_live_signal.value[self.local_rank % 8] = int(
            time.time())

        # init model_weights_status
        workers_model_weights = np.zeros(shape=[1], dtype=np.int32)
        self.model_weights_status = IPCSignal(
            name="model_weights_status",
            array=workers_model_weights,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

        # init exist_task_signal
        workers_exist_task = np.zeros(
            [self.parallel_config.expert_parallel_degree], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=workers_exist_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

        # init exist_swapped_task_signal
        workers_swapped_task = np.zeros(
            shape=[self.parallel_config.expert_parallel_degree],
            dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=workers_swapped_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

        # init exist_prefill_task_signal
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

    def event_loop_ep(self) -> None:
        """
        Tmp loop function for ep utill DP is supported
        """
        while True:
            self.worker_healthy_live_signal.value[self.local_rank] = int(
                time.time())

            if self.fd_config.parallel_config.tensor_parallel_rank == 0 and self.task_queue.num_tasks(
            ) > 0:
                tasks, read_finish = self.task_queue.get_tasks()

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)
                    req_dicts.extend(req_dict)
                logger.info(f"Rank: {self.local_rank}, num_running_requests: {num_running_requests}, " \
                            f"num_insert_requests: {len(req_dicts)}")
                # Process prefill inputs
                self.worker.preprocess_new_task(req_dicts)

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            self.worker.execute_model()

    def event_loop_normal(self) -> None:
        """ Main event loop for Paddle Distrubuted Workers.
        TODO(gongshaotian): support remote calling of functions that control worker.
        """
        # Currently, only support single node
        self.nnode = int((self.parallel_config.tensor_parallel_degree + 7) // 8)
        mp_num_per_node = self.parallel_config.tensor_parallel_degree // self.nnode
        req_ids = []
        while True:
            if self.local_rank == 0:
                if self.model_weights_status.value[0] != 0:
                    self.exist_task_signal.value[0] = 2
                else:
                    self.exist_task_signal.value[0] = 0

            if self.parallel_config.tensor_parallel_degree > 1:
                # Synchronize before updating weights
                paddle.distributed.barrier()

            self.insert_step = False
            self.worker_healthy_live_signal.value[self.local_rank] = int(
                time.time())

            # The first worker detects whether there are tasks in the task queue
            if self.local_rank %  mp_num_per_node == 0:
                if self.task_queue.num_tasks() > 0:
                    if self.nnode > 1:
                        self.task_queue.read_finish_flag.set(1)
                    else:
                        self.exist_task_signal.value[
                            self.fd_config.parallel_config.
                            expert_parallel_rank] = 1

            if self.parallel_config.tensor_parallel_degree > 1:
                # Synchronize the signal for other workers
                # TODO(@wufeisheng): Split TP group and EP group
                paddle.distributed.barrier()

            if self.fd_config.load_config.dynamic_load_weight:
                if self.exist_task_signal.value[0] == 2:
                    from fastdeploy.rl.dynamic_weight_manager import \
                        DynamicWeightManager
                    DynamicWeightManager.check_model_weights_status(
                        self.model_weights_status, self.worker.model_runner,
                        self.parallel_config.engine_pid)

            if self.exist_task_signal.value[
                    self.fd_config.parallel_config.expert_parallel_rank] == 1 or \
                    self.task_queue.read_finish_flag.get() == 1:
                logger.info(f"Rank: {self.local_rank} Detected new requests.")
                self.insert_step = True

                tasks, read_finish = self.task_queue.get_tasks()
                if read_finish:
                    # Ensure that every worker get the task
                    self.exist_task_signal.value[self.fd_config.parallel_config
                                                 .expert_parallel_rank] = 0
                    self.task_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)
                    req_dicts.extend(req_dict)

                req_ids = [req.request_id for req in req_dicts]
                logger.info(f"Rank: {self.local_rank}, num_running_requests: {num_running_requests}, " \
                            f"num_insert_requests: {len(req_dicts)}, req_ids: {req_ids}")

                # Process prefill inputs
                self.worker.preprocess_new_task(req_dicts)

            if not self.worker.model_runner.not_need_stop():
                if self.ranks > 1:
                    paddle.distributed.barrier()

                time.sleep(0.001)
                continue

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            self.worker.execute_model(req_dicts)

            self.exist_prefill_task_signal.value[
                0] = self.worker.prefill_finished()

    def init_distributed_enviroment(self, seed: int = 20) -> List[int]:
        """ Initialize Paddle Fleet and get rank of worker """
        # Global rank
        self.ranks = dist.get_world_size()
        dist_strategy = fleet.DistributedStrategy()

        dist_strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.ranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        dist_strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=dist_strategy)

        # Local rank
        self.local_rank = fleet.worker_index()

        return self.ranks, self.local_rank

    def determine_num_available_blocks(self) -> None:
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
            available_kv_cache_memory = self.worker.determine_available_memory(
            )
            logger.info(
                f"------- available_kv_cache_memory:{available_kv_cache_memory / 1024**3} GB --------"
            )

            # 2. Calculate the appropriate number of blocks
            model_block_memory_used = self.worker.cal_theortical_kvcache()
            num_blocks_local = int(available_kv_cache_memory //
                                   model_block_memory_used)
            # NOTE(liuzichang): Too many block will lead to illegal memory access
            # We will develop dynamic limits in future.
            if num_blocks_local > 20000:
                logger.info(
                    f"------- Reset num_blocks_local {num_blocks_local} to 20000"
                )
                num_blocks_local = min(20000, num_blocks_local)
            logger.info(
                f"------- model_block_memory_used:{model_block_memory_used} --------"
            )
            logger.info(
                f"------- num_blocks_local:{num_blocks_local} --------")

            logger.info(
                f"self.fd_config.parallel_config.do_profile:{self.fd_config.parallel_config.do_profile}"
            )

            # 3. Send IPCSignal
            get_profile_block_num = np.zeros(shape=[self.ranks],
                                             dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.parallel_config.engine_pid,
                create=False)
            self.get_profile_block_num_signal.value[
                self.local_rank] = num_blocks_local

            # Wait all worker send the signal
            while np.any(self.get_profile_block_num_signal.value <= 0):
                time.sleep(0.01)
            num_blocks_global = self.get_profile_block_num_signal.value.min(
            ).item()
            self.get_profile_block_num_signal.value[
                self.local_rank] = num_blocks_global
        else:
            num_blocks_global = self.fd_config.parallel_config.max_block_num
        # NOTE(liuzichang): Too big num_blocks_global will lead to error 700
        # 4. Updata share inputs
        self.worker.reinitialize_kv_cache(num_gpu_blocks=num_blocks_global)

    def init_device(self) -> None:
        """ Initialize device and Construct model runner """
        self.worker.init_device()

    def load_model(self) -> None:
        """ Load weights and create model """
        self.worker.load_model()


def parse_args():
    """
    Parse args from command line
    """
    parser = argparse.ArgumentParser("FastDeploy LLM Inference")
    parser.add_argument("-m",
                        "--model_name_or_path",
                        type=str,
                        default="./output",
                        help="model dir")
    parser.add_argument("-mbs",
                        "--max_num_seqs",
                        type=int,
                        default=34,
                        help="max batch size")
    parser.add_argument("--total_block_num", type=int, default=2000)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--pod_ip", type=str, default="127.0.0.1")
    parser.add_argument("--engine_worker_queue_port", type=int, default=9923)
    parser.add_argument("--max_model_len",
                        type=int,
                        default=3072,
                        help="max model len")
    parser.add_argument("--device_ids",
                        type=str,
                        default="0",
                        help="cuda visible devices")
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        help="input dtype")
    parser.add_argument("--enc_dec_block_num",
                        type=int,
                        default=1,
                        help="encoder's decoder num")
    parser.add_argument("--kv_cache_ratio",
                        type=float,
                        default=0.7,
                        help="kv cache ratio for input")
    parser.add_argument("--first_token_id",
                        type=int,
                        default=1,
                        help="first token id")
    parser.add_argument("--gpu_memory_utilization",
                        type=float,
                        default=0.9,
                        help="gpu memory utilization")
    parser.add_argument("--engine_pid",
                        type=int,
                        default=None,
                        help="Process ID of engine")
    parser.add_argument("--do_profile",
                        action='store_true',
                        help="do profile or not")
    parser.add_argument("--pad_token_id",
                        type=int,
                        default=-1,
                        help="pad token id")
    parser.add_argument("--eos_tokens_lens",
                        type=int,
                        default=2,
                        help="eos token lens")
    parser.add_argument("--enable_chunked_prefill",
                        action='store_true',
                        help="enable chunked prefill")
    parser.add_argument(
        "--speculative_method",
        default=None,
        type=none_or_str,
        choices=[
            None,
            "ngram",
            "mtp",
        ],
    )
    parser.add_argument(
        "--speculative_max_draft_token_num",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--speculative_model_name_or_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--speculative_model_quantization",
        default="WINT8",
        type=str,
    )
    parser.add_argument(
        "--speculative_benchmark_mode",
        default="false",
        type=str,
    )
    parser.add_argument("--max_num_batched_tokens",
                        type=int,
                        default=2048,
                        help="max num batched tokens")

    parser.add_argument("--enable_prefix_caching",
                        action='store_true',
                        help="enable prefix cache")
    parser.add_argument("--enable-custom-all-reduce",
                        action='store_true',
                        help="enable custom all-reduce")
    parser.add_argument("--splitwise_role",
                        type=str,
                        default="mixed",
                        help="splitwise role")
    parser.add_argument("--tensor_parallel_size",
                        type=int,
                        default=1,
                        help="tensor parallel size")
    parser.add_argument("--expert_parallel_size",
                        type=int,
                        default=1,
                        help="expert parallel size")
    parser.add_argument("--enable_expert_parallell",
                        action='store_true',
                        help="enable expert parallell")
    parser.add_argument("--ori_vocab_size", type=int, default=None)

    parser.add_argument("--quantization",
                        type=str,
                        default="None",
                        help="Quantization name for the model, currentlly support " \
                            "'wint4', 'wint8'," \
                            "default is None. The priority of this configuration "\
                            "is lower than that of the config file. " \
                            "More complex quantization methods need to be configured via the config file.")
    parser.add_argument("--enable_static_graph_inference",
                        action='store_true',
                        help="Whether to use static mode; if enabled, " \
                             "'paddle.to_static' will be used to convert dynamic to static.")
    parser.add_argument("--use_cudagraph",
                        action='store_true',
                        help="Flags to enable cuda graph.")
    parser.add_argument("--max_capture_batch_size",
                        type=int,
                        default=64,
                        help="Maximum Batch Size for Cuda Graph Capture. " \
                        "If max_capture_batch_size set 64, FastDeploy will capture batch size in [1, 64]")
    parser.add_argument("--guided_decoding_backend",
                        type=str,
                        default="off",
                        help="guided decoding backend")
    parser.add_argument("--disable_any_whitespace",
                        action='store_false',
                        help="Disable any whitespace for guided decoding.")
    parser.add_argument("--dynamic_load_weight",
                        action='store_true',
                        help="Enable dynamic weight loading strategy")
    parser.add_argument(
        "--load_strategy",
        type=str,
        choices=['ipc', 'ipc_no_reshard', 'ipc_snapshot', 'meta', 'normal'],
        default='meta',
        help="Weight loading method when dynamic loading is enabled: "
        "'ipc': real-time IPC streaming with automatic resharding, "
        "'ipc_no_reshard': IPC streaming without weight processing, "
        "'ipc_snapshot': load from disk snapshot of IPC weights, "
        "'meta': provide RL traing worker, no_weights_load"
        "'normal':normal load weight")

    args = parser.parse_args()
    return args


def initialize_fd_config(config_or_args) -> FDConfig:
    """Initialize FDConfig from either RolloutModelConfig or argparse.Namespace

    Args:
        config: Configuration object containing all parameters (either RolloutModelConfig or argparse.Namespace)

    Returns:
        FDConfig: Initialized FastDeploy configuration object
    """
    # Get model config from model directory
    model_config_dict, _ = ModelConfig.get_config_dict(config_or_args.model_name_or_path)



    # Handle MoE related configs
    if 'num_experts' in model_config_dict:
        model_config_dict['moe_num_experts'] = model_config_dict.pop('num_experts')
    if 'num_experts_per_tok' in model_config_dict:
        model_config_dict['moe_topk'] = model_config_dict.pop('num_experts_per_tok')


    # Set default values for model config
    model_config_dict["head_dim"] = model_config_dict.get(
        "head_dim", model_config_dict["hidden_size"] // model_config_dict["num_attention_heads"])
    model_config_dict["rope_theta"] = model_config_dict.get("rope_theta", 10000.0)

    # Create model config object
    model_config = ModelConfig.from_dict(model_config_dict)
    model_config.head_dim = model_config_dict["head_dim"]
    paddle.set_default_dtype(config_or_args.dtype)
    if 'tie_word_embeddings' in model_config_dict:
        model_config_dict['tie_word_embeddings'] = model_config_dict.pop('tie_word_embeddings')

    # Initialize all config components
    device_config = DeviceConfig()
    decoding_config = DecodingConfig()
    speculative_config = SpeculativeConfig()
    parallel_config = ParallelConfig()
    load_config = LoadConfig()
    moe_config = MoEConfig()

    # Handle graph optimization config (check for attribute existence for backward compatibility)
    enable_static_graph_inference = getattr(config_or_args, 'enable_static_graph_inference', False)
    use_cudagraph = getattr(config_or_args, 'use_cudagraph', False)
    max_capture_batch_size = getattr(config_or_args, 'max_capture_batch_size', 0)

    graph_opt_config = GraphOptimizationConfig(
        enable_static_graph_inference,
        use_cudagraph,
        max_capture_batch_size
    )

    # Handle quantization (check for attribute existence)
    model_config.quantization = getattr(config_or_args, 'quantization', None)

    # Update speculative config_or_args
    speculative_config.method = getattr(config_or_args, 'speculative_method', None)
    speculative_config.num_speculative_tokens = getattr(config_or_args, 'speculative_max_draft_token_num', 0)
    speculative_config.model_name_or_path = getattr(config_or_args, 'speculative_model_name_or_path', None)
    speculative_config.quantization = getattr(config_or_args, 'speculative_model_quantization', None)
    speculative_config.benchmark_mode = (
        getattr(config_or_args, "speculative_benchmark_mode", "false").lower() == "true"
    )

    # Update parallel config
    parallel_config.engine_pid = getattr(config_or_args, 'engine_pid', None)
    parallel_config.model_name_or_path = config_or_args.model_name_or_path
    parallel_config.max_num_seqs = getattr(config_or_args, 'max_num_seqs', 0)
    parallel_config.max_block_num = getattr(config_or_args, 'total_block_num', 0)
    parallel_config.block_size = getattr(config_or_args, 'block_size', 64)
    parallel_config.pod_ip = getattr(config_or_args, 'pod_ip', None)
    parallel_config.engine_worker_queue_port = getattr(config_or_args, 'engine_worker_queue_port', 0)
    parallel_config.max_model_len = getattr(config_or_args, 'max_model_len', 0)
    model_config.max_seq_len = getattr(config_or_args, 'max_model_len', 0)
    model_config.max_length = getattr(config_or_args, 'max_model_len', 0)
    parallel_config.device_ids = getattr(config_or_args, 'device_ids', [])
    parallel_config.dtype = config_or_args.dtype
    parallel_config.enc_dec_block_num = getattr(config_or_args, 'enc_dec_block_num', 0)
    parallel_config.kv_cache_ratio = getattr(config_or_args, 'kv_cache_ratio', 1.0)
    parallel_config.first_token_id = getattr(config_or_args, 'first_token_id', None)
    parallel_config.gpu_memory_utilization = getattr(config_or_args, 'gpu_memory_utilization', 0.9)
    parallel_config.engine_pid = getattr(config_or_args, 'engine_pid', None)
    parallel_config.do_profile = getattr(config_or_args, 'do_profile', False)
    parallel_config.dynamic_load_weight = getattr(config_or_args, 'dynamic_load_weight', False)
    parallel_config.pad_token_id = getattr(config_or_args, 'pad_token_id', None)
    parallel_config.eos_tokens_lens = getattr(config_or_args, 'eos_tokens_lens', 0)
    parallel_config.enable_chunked_prefill = getattr(config_or_args, 'enable_chunked_prefill', False)
    parallel_config.max_num_batched_tokens = getattr(config_or_args, 'max_num_batched_tokens', 0)
    parallel_config.enable_prefix_caching = getattr(config_or_args, 'enable_prefix_caching', False)
    parallel_config.enable_custom_all_reduce = getattr(config_or_args, 'enable_custom_all_reduce', False)
    parallel_config.use_ep = getattr(config_or_args, 'enable_expert_parallell', False)
    parallel_config.tensor_parallel_degree = getattr(config_or_args, 'tensor_parallel_size', 1)
    parallel_config.expert_parallel_degree = getattr(config_or_args, 'expert_parallel_size', 1)
    parallel_config.splitwise_role = getattr(config_or_args, 'splitwise_role', None)
    parallel_config.guided_decoding_backend = getattr(config_or_args, 'guided_decoding_backend', None)
    parallel_config.disable_any_whitespace = getattr(config_or_args, 'disable_any_whitespace', False)

    # Log parallel config info
    logger.info(f"parallel_config.use_ep {parallel_config.use_ep}")
    logger.info(f"parallel_config.tensor_parallel_degree {parallel_config.tensor_parallel_degree}")
    logger.info(f"splitwise_role {parallel_config.splitwise_role}")

    # Set MoE phase based on splitwise role
    if parallel_config.splitwise_role == "mixed":
        parallel_config.moe_phase = MoEPhase.PREFILL
    elif parallel_config.splitwise_role == "prefill":
        parallel_config.moe_phase = MoEPhase.PREFILL
    elif parallel_config.splitwise_role == "decode":
        parallel_config.moe_phase = MoEPhase.DECODER
    elif parallel_config.splitwise_role is not None:
        raise NotImplementedError

    # Handle model architecture specific configurations
    num_key_value_heads = model_config_dict.get("num_key_value_heads", -1)
    if num_key_value_heads is None:
        num_key_value_heads = -1

    # Calculate FFN hidden size
    if model_config_dict.get("ffn_hidden_size", None) is not None:
        ffn_hidden_size = model_config_dict["ffn_hidden_size"]
    elif model_config_dict.get("intermediate_size", None) is not None:
        ffn_hidden_size = model_config_dict["intermediate_size"]
    else:
        ffn_hidden_size = 4 * model_config_dict["hidden_size"]
        if model_config_dict["hidden_act"].lower() == "swiglu":
            if paddle.distributed.get_world_size() > 1:
                multiple_of = 8 * model_config_dict["num_attention_heads"]
            else:
                multiple_of = 4 * model_config_dict["num_attention_heads"]
            ffn_hidden_size = multiple_of * (
                (int(2 * ffn_hidden_size / 3) + multiple_of - 1) //
                multiple_of)

    # Get number of layers
    num_layers = model_config_dict.get("num_layers", None) or model_config_dict.get(
        "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(f"num_layers<{num_layers}> is invalid")

    use_moe = model_config_dict.get("moe_layer_start_index", num_layers) < num_layers

    # Update model config
    model_config.ffn_hidden_size = ffn_hidden_size
    model_config.num_layers = num_layers
    model_config.num_key_value_heads = num_key_value_heads
    model_config.start_layer_index = model_config_dict.get("start_layer_index", 0)

    # Update MoE config
    moe_config.num_experts = model_config_dict.get("moe_num_experts", None)
    moe_config.moe_intermediate_size = model_config_dict.get("moe_intermediate_size", None)
    moe_config.top_k = model_config_dict.get("moe_k", model_config_dict.get("moe_topk", 8))
    moe_config.moe_num_shared_experts = model_config_dict.get("moe_num_shared_experts", 0)
    moe_config.moe_layer_start_index = model_config_dict.get("moe_layer_start_index", 0)
    moe_config.num_max_dispatch_tokens_per_rank = model_config_dict.get(
        "num_max_dispatch_tokens_per_rank", 256)
    moe_config.moe_use_aux_free = model_config_dict.get("moe_use_aux_free", False)

    # Handle vocabulary size
    model_config.ori_vocab_size = model_config_dict.get("vocab_size", -1)
    if "Ernie4_5_ForCausalLM" in model_config_dict.get("architectures", []):
        model_config.ori_vocab_size = getattr(config_or_args, 'ori_vocab_size', model_config.ori_vocab_size)

    # Handle DeepseekV3 specific config
    if "DeepseekV3ForCausalLM" in model_config_dict.get("architectures", []):
        from paddleformers.transformers import AutoConfig
        model_config.deepseekv3 = AutoConfig.from_pretrained(
            config_or_args.model_name_or_path)

    # Handle quantization config
    quantization_config = model_config_dict.get("quantization_config", None)
    if not model_config.is_quantized:
        if quantization_config is not None:
            if "kv_cache_quant_type" not in quantization_config:
                model_config.is_quantized = True

    quant_config_name = None
    if quantization_config is not None and quantization_config.get(
            "quantization", None) is None:
        raise ValueError(
            "quantization_config should have a key named 'quantization' for specify quant config."
        )

    if quantization_config is not None:
        quant_config_name = quantization_config["quantization"]
    elif getattr(config_or_args, 'quantization', None) != "None":
        quantization_config = {}
        quant_config_name = getattr(config_or_args, 'quantization', None)
        quantization_config["quantization"] = quant_config_name
        # Special handling for Ernie models
        is_ernie = "Ernie4_5_ForCausalLM" in model_config_dict.get("architectures", []) or \
                   "Ernie4_5_MoeForCausalLM" in model_config_dict.get("architectures", [])
        if use_moe and quant_config_name == "wint4" and is_ernie:
            quantization_config["dense_quant_type"] = "wint8"
            quantization_config["moe_quant_type"] = "wint4"
            quantization_config["quantization"] = "mix_quant"
            quant_config_name = "mix_quant"
    else:
        quant_config_name = None

    if quant_config_name is None:
        quant_config = None
    else:
        quant_cls = get_quantization_config(quant_config_name)
        quant_config = quant_cls.from_config(quantization_config)

    # Log quantization info
    logger.info("===========quantization_config==============")
    if quant_config is not None:
        if model_config.is_quantized:
            logger.info(
                "Model Status: Offline Quantized (pre-quantized weights loaded)"
            )
        else:
            logger.info(
                "Model Status: Original (will apply online quantization)")

        logger.info(f"Quantization Method: {getattr(config_or_args, 'quantization', 'None')}")
    else:
        logger.info(
            "No quantization config found and use original weight and act dtype."
        )

    model_config.architectures = model_config_dict.get("architectures")

    # Update load config
    logger.info("===========load_config==============")
    # Handle load config (check for environment variable)
    load_config.use_fastsafetensor = int(envs.FD_USE_FASTSAFETENSOR) == 1
    load_config.dynamic_load_weight = getattr(config_or_args, 'dynamic_load_weight', False)
    load_config.load_strategy = getattr(config_or_args, 'load_strategy', None)
    logger.info(f"- Dynamic load weight: {load_config.dynamic_load_weight}")
    logger.info(f"- Load strategy: {load_config.load_strategy}")
    logger.info(f"- Use fastsafetensor: {load_config.use_fastsafetensor}")

    # Create and return FDConfig
    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=device_config,
        load_config=load_config,
        moe_config=moe_config,
        decoding_config=decoding_config,
        quant_config=quant_config,
        graph_opt_config=graph_opt_config
    )

    return fd_config


def run_worker_proc() -> None:
    """
    start worker process
    """
    # Get args form Engine
    args = parse_args()

    # Get fd_config
    fd_config = initialize_fd_config(args)

    # Create worker process
    worker_proc = PaddleDisWorkerProc(fd_config)

    # Initialize device and create model runner
    worker_proc.init_device()

    # Load model
    worker_proc.load_model()
    logger.info("determine_num_available_blocks")
    worker_proc.determine_num_available_blocks()

    # Trigger CUDAGraph capture
    worker_proc.worker.graph_optimize_and_warm_up_model()

    # Initialize health status
    worker_proc.init_health_status()

    # Start event loop
    if fd_config.parallel_config.use_ep:
        # TODO(wufeisheng): Delete this branch
        worker_proc.event_loop_ep()
    else:
        worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()
