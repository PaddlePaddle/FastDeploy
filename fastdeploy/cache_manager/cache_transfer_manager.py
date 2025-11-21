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
import concurrent.futures
import gc
import json
import queue
import threading
import time
import traceback

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.cache_manager.cache_data import CacheStatus
from fastdeploy.cache_manager.ops import (
    cuda_host_alloc,
    cuda_host_free,
    memory_allocated,
    set_data_ipc,
    set_device,
    share_external_data_,
    swap_cache_all_layers,
    unset_data_ipc,
)
from fastdeploy.config import SpeculativeConfig
from fastdeploy.inter_communicator import EngineCacheQueue, IPCSignal, KVCacheStatus
from fastdeploy.platforms import current_platform
from fastdeploy.utils import get_logger


def parse_args():
    """
    从命令行解析参数
    """
    parser = argparse.ArgumentParser("Cache transfer manager")
    parser.add_argument(
        "--splitwise_role",
        type=str,
        default="mixed",
        help="splitwise role, can be decode, prefill or mixed",
    )
    parser.add_argument("--rank", type=int, default=0, help="current rank")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--num_layers", type=int, default=1, help="model num layers")
    parser.add_argument("--mp_num", type=int, default=1, help="number of model parallel")
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="bfloat16",
        choices=["uint8", "bfloat16", "block_wise_fp8"],
        help="cache dtype",
    )
    parser.add_argument("--key_cache_shape", type=str, default="", help="key cache shape")
    parser.add_argument("--value_cache_shape", type=str, default="", help="value cache shape")
    parser.add_argument("--cache_queue_port", type=int, default=9923, help="cache queue port")
    parser.add_argument("--enable_splitwise", type=int, default=0, help="enable splitwise ")
    parser.add_argument("--pod_ip", type=str, default="0.0.0.0", help="pod ip")
    parser.add_argument(
        "--engine_worker_queue_port",
        type=int,
        default=9923,
        help="engine worker queue port",
    )
    parser.add_argument("--num_cpu_blocks", type=int, default=4, help="cpu cache block number")
    parser.add_argument("--engine_pid", type=str, default=None, help="engine pid")
    parser.add_argument(
        "--protocol",
        type=str,
        default="ipc",
        help="cache transfer protocol, only support ipc now",
    )
    parser.add_argument("--local_data_parallel_id", type=int, default=0)
    parser.add_argument("--rdma_port", type=str, default="", help="rmda port")
    parser.add_argument(
        "--speculative_config",
        type=json.loads,
        default="{}",
        help="speculative config",
    )
    parser.add_argument("--create_cache_tensor", action="store_true")

    args = parser.parse_args()
    return args


class CacheTransferManager:
    """
    管理CPU和GPU之间缓存的交换传输
    """

    def __init__(self, args):
        """
        初始化CacheTransferManager
        """
        device = args.device_id
        rank = args.rank
        self.gpu_cache_kvs = {}
        self.cpu_cache_kvs = {}
        self.gpu_cache_k_tensors = []
        self.gpu_cache_v_tensors = []
        self.gpu_cache_scales_k_tensors = []
        self.gpu_cache_scales_v_tensors = []
        self.speculative_config = SpeculativeConfig(args.speculative_config)
        self.key_cache_shape = [int(i) for i in args.key_cache_shape.split(",")]
        self.value_cache_shape = []
        if args.value_cache_shape:
            self.value_cache_shape = [int(i) for i in args.value_cache_shape.split(",")]
        self.num_gpu_blocks = self.key_cache_shape[0]
        self.num_extra_layers = self.speculative_config.num_extra_cache_layer
        self.num_extra_layer_gpu_blocks = int(self.num_gpu_blocks * self.speculative_config.num_gpu_block_expand_ratio)

        self.swap_to_cpu_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.swap_to_gpu_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.transfer_task_queue = queue.Queue()  # 用来接收传输任务
        self.tansfer_done_queue = queue.Queue()  # 用来告知任务执行完毕
        self.n_ranks = args.mp_num
        self.rank = rank
        self.device = device
        self.engine_pid = args.engine_pid
        self.cache_dtype = args.cache_dtype

        address = (args.pod_ip, args.cache_queue_port)
        self.cache_task_queue = EngineCacheQueue(
            address=address,
            is_server=False,
            num_client=args.mp_num,
            client_id=rank,
            local_data_parallel_id=args.local_data_parallel_id,
        )

        cache_ready_signal_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=self.engine_pid,
            create=False,
        )
        swap_space_ready_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.swap_space_ready_signal = IPCSignal(
            name="swap_space_ready_signal",
            array=swap_space_ready_data,
            dtype=np.int32,
            suffix=self.engine_pid,
            create=False,
        )

        self.num_cpu_blocks = args.num_cpu_blocks

        self._init_gpu_cache(args)
        if self.num_cpu_blocks > 0:
            self._init_cpu_cache(args)

        cache_task_broadcast_data = np.zeros(shape=[1], dtype=np.int32)
        self.cache_task_broadcast_signal = IPCSignal(
            name="cache_task_broadcast_signal",
            array=cache_task_broadcast_data,
            dtype=np.int32,
            suffix=args.engine_pid,
            create=False,
        )

        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        array_size = min(max_chips_per_node, args.mp_num)
        worker_healthy_live_array = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_array,
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
            create=False,
        )
        threading.Thread(target=self.clear_or_update_caches, args=[args], daemon=True).start()

    def _init_gpu_cache(self, args):

        try:
            assert not args.create_cache_tensor
        except:
            logger.warn(
                f"In current implementation, cache transfer manager do not create cache tensors at all, "
                f"meaning create_cache_tensor should be False, while we got {args.create_cache_tensor}. "
                f"Cache tensor creation will occur in: 1) model runner in case of mixed deployment; "
                f"or 2) cache messager in case of disaggregation deployment. "
                f"Please check the codes and make sure they work correctly."
            )
        if not args.create_cache_tensor:
            logger.info(f"[rank {self.rank}/{self.n_ranks}] Waiting for runners or messagers to create kv cache.")
            while self.cache_ready_signal.value[self.rank] != 1:
                time.sleep(0.1)
            logger.info(f"[rank {self.rank}/{self.n_ranks}] OK! Stop waiting.")

        if args.cache_dtype == "block_wise_fp8":
            cache_type = "uint8"
        else:
            cache_type = args.cache_dtype

        logger.info(f"[rank {self.rank}/{self.n_ranks}] Initializing kv cache for all layers.")
        set_device(self.device)
        for i in range(args.num_layers + self.num_extra_layers):
            num_gpu_blocks = self.num_gpu_blocks if i < args.num_layers else self.num_extra_layer_gpu_blocks
            key_name = f"key_caches_{i}_rank{self.rank}.device{self.device}"
            val_name = f"value_caches_{i}_rank{self.rank}.device{self.device}"
            key_cache_scales_name = f"key_cache_scales_{i}_rank{self.rank}.device{self.device}"
            value_cache_scales_name = f"value_cache_scales_{i}_rank{self.rank}.device{self.device}"
            key_cache_shape = [
                num_gpu_blocks,
                self.key_cache_shape[1],
                self.key_cache_shape[2],
                self.key_cache_shape[3],
            ]
            value_cache_shape = []
            if self.value_cache_shape:
                value_cache_shape = [
                    num_gpu_blocks,
                    self.value_cache_shape[1],
                    self.value_cache_shape[2],
                    self.value_cache_shape[3],
                ]
            if args.create_cache_tensor:
                logger.info(
                    f"[rank {self.rank}/{self.n_ranks}] ..creating kv cache for layer {i}: {key_cache_shape} {value_cache_shape}"
                )
                key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_type)
                set_data_ipc(key_cache, key_name)

                if args.cache_dtype == "block_wise_fp8":
                    key_cache_scales = paddle.full(
                        shape=[num_gpu_blocks, self.key_cache_shape[1], self.key_cache_shape[2]],
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
                    set_data_ipc(key_cache_scales, key_cache_scales_name)
                if self.value_cache_shape:
                    val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_type)
                    set_data_ipc(val_cache, val_name)

                    if args.cache_dtype == "block_wise_fp8":
                        value_cache_scales = paddle.full(
                            shape=[num_gpu_blocks, self.value_cache_shape[1], self.value_cache_shape[2]],
                            fill_value=0,
                            dtype=paddle.get_default_dtype(),
                        )
                        set_data_ipc(value_cache_scales, value_cache_scales_name)
            else:
                logger.info(
                    f"[rank {self.rank}/{self.n_ranks}] ..attaching kv cache for layer {i}: {key_cache_shape} {value_cache_shape}"
                )
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                val_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache = share_external_data_(key_cache, key_name, key_cache_shape, True)
                if args.cache_dtype == "block_wise_fp8":
                    key_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                    key_cache_scales = share_external_data_(
                        key_cache_scales,
                        key_cache_scales_name,
                        [num_gpu_blocks, self.key_cache_shape[1], self.key_cache_shape[2]],
                        True,
                    )
                if self.value_cache_shape:
                    val_cache = share_external_data_(val_cache, val_name, value_cache_shape, True)
                    if args.cache_dtype == "block_wise_fp8":
                        value_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                        value_cache_scales = share_external_data_(
                            value_cache_scales,
                            value_cache_scales_name,
                            [num_gpu_blocks, self.value_cache_shape[1], self.value_cache_shape[2]],
                            True,
                        )

            self.gpu_cache_kvs[key_name] = key_cache
            self.gpu_cache_k_tensors.append(self.gpu_cache_kvs[key_name])
            if args.cache_dtype == "block_wise_fp8":
                self.gpu_cache_kvs[key_cache_scales_name] = key_cache_scales
                self.gpu_cache_scales_k_tensors.append(self.gpu_cache_kvs[key_cache_scales_name])
            if args.value_cache_shape:
                self.gpu_cache_kvs[val_name] = val_cache
                self.gpu_cache_v_tensors.append(self.gpu_cache_kvs[val_name])
                if args.cache_dtype == "block_wise_fp8":
                    self.gpu_cache_kvs[value_cache_scales_name] = value_cache_scales
                    self.gpu_cache_scales_v_tensors.append(self.gpu_cache_kvs[value_cache_scales_name])

        if args.create_cache_tensor:
            logger.info(f"[rank {self.rank}/{self.n_ranks}] ✅ kv cache is ready!")
            self.cache_ready_signal.value[self.rank] = 1

        cache_kv_size_byte = sum([tmp.numel() * 1 for key, tmp in self.gpu_cache_kvs.items()])
        logger.info(f"[rank {self.rank}/{self.n_ranks}] device :{self.device}")
        logger.info(f"[rank {self.rank}/{self.n_ranks}] cache_kv_size_byte : {cache_kv_size_byte}")
        logger.info(f"[rank {self.rank}/{self.n_ranks}] done init cache (full) gmem alloc : {memory_allocated()}")

    def _init_cpu_cache(self, args):
        key_cache_size = self.key_cache_shape[1] * self.key_cache_shape[2] * self.key_cache_shape[3]
        if args.value_cache_shape:
            value_cache_size = self.value_cache_shape[1] * self.value_cache_shape[2] * self.value_cache_shape[3]
        else:
            value_cache_size = 0
        if args.cache_dtype == "bfloat16":
            cache_bytes = 2
        elif args.cache_dtype == "uint8" or args.cache_dtype == "block_wise_fp8":
            cache_bytes = 1
        else:
            raise ValueError(f"Unsupported cache dtype: {args.cache_dtype}")
        key_need_to_allocate_bytes = args.num_cpu_blocks * cache_bytes * key_cache_size
        value_need_to_allocate_bytes = args.num_cpu_blocks * cache_bytes * value_cache_size
        if args.cache_dtype == "block_wise_fp8":
            cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
            cache_scales_size = self.key_cache_shape[1] * self.key_cache_shape[2]
            scales_key_need_to_allocate_bytes = args.num_cpu_blocks * cache_scales.element_size() * cache_scales_size
            scales_value_need_to_allocate_bytes = args.num_cpu_blocks * cache_scales.element_size() * cache_scales_size
        logger.info(
            f"[rank {self.rank}/{self.n_ranks}] ..swap space size : {(key_need_to_allocate_bytes + value_need_to_allocate_bytes) / 1024 ** 3:.2f}GB"
        )
        if args.num_cpu_blocks == 0:
            logger.info(f"[rank {self.rank}/{self.n_ranks}] 💡 no swap space (cpu cache) is specified.")
            self.swap_space_ready_signal.value[self.rank] = 1
            return
        logger.info(f"[rank {self.rank}/{self.n_ranks}] Initializing swap space (cpu cache) for all layers.")
        paddle.set_device("cpu")
        self.k_dst_ptrs = []
        self.v_dst_ptrs = []
        self.k_scales_ptrs = []
        self.v_scales_ptrs = []
        for i in range(args.num_layers + self.num_extra_layers):
            key_name = f"key_caches_{i}_rank{self.rank}"
            val_name = f"value_caches_{i}_rank{self.rank}"
            key_cache_scales_name = f"key_cache_scales_{i}_rank{self.rank}"
            value_cache_scales_name = f"value_cache_scales_{i}_rank{self.rank}"
            logger.info(
                f"[rank {self.rank}/{self.n_ranks}] ..creating cpu cache for layer {i}: {(key_need_to_allocate_bytes + value_need_to_allocate_bytes) / 1024 ** 3:.2f}GB"
            )
            self.cpu_cache_kvs[key_name] = cuda_host_alloc(key_need_to_allocate_bytes)
            self.k_dst_ptrs.append(self.cpu_cache_kvs[key_name])
            if args.cache_dtype == "block_wise_fp8":
                self.cpu_cache_kvs[key_cache_scales_name] = cuda_host_alloc(scales_key_need_to_allocate_bytes)
                self.k_scales_ptrs.append(self.cpu_cache_kvs[key_cache_scales_name])
            if value_need_to_allocate_bytes > 0:
                self.cpu_cache_kvs[val_name] = cuda_host_alloc(value_need_to_allocate_bytes)
                self.v_dst_ptrs.append(self.cpu_cache_kvs[val_name])
                if args.cache_dtype == "block_wise_fp8":
                    self.cpu_cache_kvs[value_cache_scales_name] = cuda_host_alloc(scales_value_need_to_allocate_bytes)
                    self.v_scales_ptrs.append(self.cpu_cache_kvs[value_cache_scales_name])
        logger.info(f"[rank {self.rank}/{self.n_ranks}] ✅ swap space (cpu cache) is ready!")
        self.swap_space_ready_signal.value[self.rank] = 1

    def _do_swap_to_cpu_task(
        self,
        swap_node_ids,
        gpu_block_id,
        cpu_block_id,
        event_type,
        transfer_task_id,
    ):
        """
        swap cache GPU->CPU
        """
        self.cache_task_queue.swap_to_cpu_barrier1.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_cpu_barrier1.reset()
        result = self._transfer_data(
            swap_node_ids,
            gpu_block_id,
            cpu_block_id,
            event_type,
            transfer_task_id,
        )
        self.cache_task_queue.swap_to_cpu_barrier2.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_cpu_barrier2.reset()
            self.cache_task_queue.put_transfer_done_signal(result)
            logger.debug(f"_do_swap_to_cpu_task: put_transfer_done_signal {result}")
            logger.info(f"_do_swap_to_cpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}")

    def _do_swap_to_gpu_task(
        self,
        swap_node_ids,
        gpu_block_id,
        cpu_block_id,
        event_type,
        transfer_task_id,
    ):
        """
        swap cache CPU->GPU
        """
        self.cache_task_queue.swap_to_gpu_barrier1.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_gpu_barrier1.reset()
        result = self._transfer_data(
            swap_node_ids,
            gpu_block_id,
            cpu_block_id,
            event_type,
            transfer_task_id,
        )
        self.cache_task_queue.swap_to_gpu_barrier2.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_gpu_barrier2.reset()
            self.cache_task_queue.put_transfer_done_signal(result)
            logger.debug(f"_do_swap_to_gpu_task: put_transfer_done_signal {result}")
            logger.info(f"_do_swap_to_gpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}")

    def check_work_status(self, time_interval_threashold=envs.FD_CACHE_PROC_EXIT_TIMEOUT):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def do_data_transfer(self):
        """
        do data transfer task
        """

        consecutive_error_count = 0
        max_errors = (
            envs.FD_CACHE_PROC_ERROR_COUNT
        )  # After this many consecutive errors, check if the worker process exists.

        while True:
            try:
                if self.rank == 0:
                    if not self.cache_task_queue.empty():
                        self.cache_task_broadcast_signal.value[0] = 1
                if self.n_ranks > 1:
                    self.cache_task_queue.barrier1.wait()
                    if self.rank == 0:
                        self.cache_task_queue.barrier1.reset()
                if self.cache_task_broadcast_signal.value[0] == 1:
                    data, read_finish = self.cache_task_queue.get_transfer_task()
                    logger.debug(f"transfer data: get_transfer_task {data}")
                    if read_finish:
                        self.cache_task_broadcast_signal.value[0] = 0
                    (
                        swap_node_ids,
                        gpu_block_id,
                        cpu_block_id,
                        event_type,
                        transfer_task_id,
                    ) = data
                    if event_type.value == CacheStatus.SWAP2CPU.value:
                        self.swap_to_cpu_thread_pool.submit(
                            self._do_swap_to_cpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                    else:
                        self.swap_to_gpu_thread_pool.submit(
                            self._do_swap_to_gpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                else:
                    if self.n_ranks > 1:
                        self.cache_task_queue.barrier2.wait()
                        if self.rank == 0:
                            self.cache_task_queue.barrier2.reset()
                    continue

                if self.n_ranks > 1:
                    self.cache_task_queue.barrier3.wait()
                    if self.rank == 0:
                        self.cache_task_queue.barrier3.reset()

                consecutive_error_count = 0

            except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                # When a cache_transfer_manager process remains, it keeps printing error logs and may exhaust disk space.
                # Add a check to see if the worker process is alive; if it has ended, exit the loop to stop continuous logging.
                logger.error(f"[CacheTransferManager] Connection broken: {e}")
                consecutive_error_count += 1
                if consecutive_error_count > max_errors:
                    try:
                        status, msg = self.check_work_status()
                    except Exception:
                        status = True

                    if status is False:
                        logger.critical(
                            f"The Worker process has been inactive for over {envs.FD_CACHE_PROC_EXIT_TIMEOUT} seconds, and the Cache process will automatically terminate (the waiting timeout can be extended via FD_CACHE_PROC_EXIT_TIMEOUT)."
                        )
                        break
                time.sleep(1)
                continue

            except Exception as e:
                logger.info(f"do_data_transfer: error: {e}, {str(traceback.format_exc())}")

    def _transfer_data(
        self,
        swap_node_ids,
        task_gpu_block_id,
        task_cpu_block_id,
        event_type,
        transfer_task_id,
    ):
        """
        transfer data
        task_gpu_block_id format: [[block_id0, [fold_block_id0, fold_block_id1]],
            [block_id1, [fold_block_id0, fold_block_id1]], ...]
        """
        logger.debug(
            f"transfer data: transfer_task_id {transfer_task_id}: swap_node_ids {swap_node_ids}"
            + f"task_gpu_block_id {task_gpu_block_id} task_cpu_block_id {task_cpu_block_id} event_type {event_type}"
        )
        start_time = time.time()
        try:
            # transform block id
            assert len(task_gpu_block_id) == len(task_cpu_block_id)
            gpu_block_ids = task_gpu_block_id
            cpu_block_ids = task_cpu_block_id

            if event_type.value == CacheStatus.SWAP2CPU.value:
                swap_cache_all_layers(
                    self.gpu_cache_k_tensors,
                    self.k_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    0,
                )
                swap_cache_all_layers(
                    self.gpu_cache_v_tensors,
                    self.v_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    0,
                )
                if self.cache_dtype == "block_wise_fp8":
                    swap_cache_all_layers(
                        self.gpu_cache_scales_k_tensors,
                        self.k_scales_ptrs,
                        self.num_cpu_blocks,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        0,
                    )
                    swap_cache_all_layers(
                        self.gpu_cache_scales_v_tensors,
                        self.v_scales_ptrs,
                        self.num_cpu_blocks,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        0,
                    )

            elif event_type.value == CacheStatus.SWAP2GPU.value:
                swap_cache_all_layers(
                    self.gpu_cache_k_tensors,
                    self.k_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    1,
                )
                swap_cache_all_layers(
                    self.gpu_cache_v_tensors,
                    self.v_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    1,
                )
                if self.cache_dtype == "block_wise_fp8":
                    swap_cache_all_layers(
                        self.gpu_cache_scales_k_tensors,
                        self.k_scales_ptrs,
                        self.num_cpu_blocks,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        1,
                    )
                    swap_cache_all_layers(
                        self.gpu_cache_scales_v_tensors,
                        self.v_scales_ptrs,
                        self.num_cpu_blocks,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        1,
                    )
            else:
                logger.warning(
                    f"transfer data: Get unexpected event type {event_type}, only SWAP2CPU and SWAP2GPU supported"
                )
        except Exception as e:
            logger.error(f"transfer data: error: {e}")
            raise e
        end_time = time.time()
        elasped_time = end_time - start_time
        logger.info(
            f"transfer data: transfer_task_id {transfer_task_id} event_type {event_type}: "
            + f"transfer {len(gpu_block_ids)} blocks done  elapsed_time {elasped_time:.4f}"
        )
        return (
            swap_node_ids,
            task_gpu_block_id,
            task_cpu_block_id,
            event_type,
            transfer_task_id,
        )

    def clear_or_update_caches(self, args):
        # TODO XPU support RL
        if unset_data_ipc is None:
            return
        logger.info("Start a thread to clear/restore kv cache when model weights are cleared/updated.")
        logger.info(f"FD_ENABLE_SWAP_SPACE_CLEARING={envs.FD_ENABLE_SWAP_SPACE_CLEARING}")
        kv_cache_status = np.zeros([1], dtype=np.int32)
        kv_cache_status_signal = IPCSignal(
            name="kv_cache_status",
            array=kv_cache_status,
            dtype=np.int32,
            suffix=self.engine_pid,
            create=False,
        )
        while True:
            if kv_cache_status_signal.value[0] == KVCacheStatus.CLEARING:
                assert args.splitwise_role == "mixed", "Only mixed mode supports clearing cache."
                try:
                    logger.info(
                        f"[rank {self.rank}/{self.n_ranks}] Start clearing caches {self.cache_ready_signal.value}"
                    )
                    # clear cpu caches
                    if envs.FD_ENABLE_SWAP_SPACE_CLEARING:
                        paddle.set_device("cpu")
                        for ptrs in self.k_dst_ptrs + self.v_dst_ptrs:
                            cuda_host_free(ptrs)
                        self.cpu_cache_kvs.clear()
                        self.k_dst_ptrs.clear()
                        self.v_dst_ptrs.clear()
                        gc.collect()
                        # reset swap_space_ready_signal
                        self.swap_space_ready_signal.value[self.rank] = 0
                        while np.sum(self.swap_space_ready_signal.value) != 0:
                            time.sleep(0.1)

                    # clear gpu caches
                    set_device(self.device)
                    for name, tensor in self.gpu_cache_kvs.items():
                        unset_data_ipc(tensor, name, True, False)
                    self.gpu_cache_kvs.clear()
                    self.gpu_cache_k_tensors.clear()
                    self.gpu_cache_v_tensors.clear()

                    # reset cache_ready_signal
                    self.cache_ready_signal.value[self.rank] = 0
                    logger.info(
                        f"[rank {self.rank}/{self.n_ranks}] Finish clearing caches {self.cache_ready_signal.value}"
                    )

                    # wait for all ranks caches to be cleared
                    if np.sum(self.cache_ready_signal.value) != 0:
                        time.sleep(0.1)

                    # reset kv_cache_status_signal
                    kv_cache_status_signal.value[0] = KVCacheStatus.CLEARED
                    logger.info("All ranks finish clearing caches")

                except Exception as e:
                    logger.error(f"[rank {self.rank}/{self.n_ranks}] Failed to clear caches: {e}")

            elif kv_cache_status_signal.value[0] == KVCacheStatus.UPDATING:
                assert args.splitwise_role == "mixed", "Only mixed mode supports updating cache."
                try:
                    logger.info(
                        f"[rank {self.rank}/{self.n_ranks}] Start restoring caches {self.cache_ready_signal.value}"
                    )
                    # restore cpu cache
                    if envs.FD_ENABLE_SWAP_SPACE_CLEARING:
                        self._init_cpu_cache(args)
                        while np.sum(self.swap_space_ready_signal.value) != args.mp_num:
                            time.sleep(0.1)

                    # restore gpu cache and set cache_ready_signal
                    self._init_gpu_cache(args)
                    logger.info(
                        f"[rank {self.rank}/{self.n_ranks}] Finish restoring caches {self.cache_ready_signal.value}"
                    )

                    # wait for all ranks caches to be ready
                    while np.sum(self.cache_ready_signal.value) != args.mp_num:
                        time.sleep(0.1)

                    # set kv_cache_status_signal
                    logger.info("All ranks finish restoring caches")
                    kv_cache_status_signal.value[0] = KVCacheStatus.NORMAL

                except Exception as e:
                    logger.error(f"[rank {self.rank}/{self.n_ranks}] Failed to restore caches: {e}")

            time.sleep(0.1)


def main():
    """
    启动cache manager
    """

    cache_manager = CacheTransferManager(args)

    cache_manager.do_data_transfer()


if __name__ == "__main__":

    args = parse_args()
    rank_id = args.rank + args.local_data_parallel_id * args.mp_num
    logger = get_logger("cache_transfer_manager", f"cache_transfer_manager_rank{rank_id}.log")
    set_device(args.device_id)
    main()
