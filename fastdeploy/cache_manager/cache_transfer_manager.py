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
import json
import queue
import time

import numpy as np
import paddle

from fastdeploy.cache_manager.cache_data import CacheStatus
from fastdeploy.engine.config import SpeculativeConfig
from fastdeploy.inter_communicator import EngineCacheQueue, IPCSignal
from fastdeploy.model_executor.ops.gpu import (cuda_host_alloc, set_data_ipc,
                                               swap_cache_all_layers)
from fastdeploy.utils import get_logger


def parse_args():
    """
    从命令行解析参数
    """
    parser = argparse.ArgumentParser("Cache transfer manager")
    parser.add_argument("--splitwise_role",
                        type=str,
                        default="mixed",
                        help="splitwise role, can be decode, prefill or mixed")
    parser.add_argument("--rank", type=int, default=0, help="current rank")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="model num layers")
    parser.add_argument("--head_dim",
                        type=int,
                        default=1,
                        help="model head dim")
    parser.add_argument("--kv_num_head",
                        type=int,
                        default=1,
                        help="model kv num head")
    parser.add_argument("--rdma_port", type=str, default="", help="rmda port")
    parser.add_argument("--mp_num",
                        type=int,
                        default=1,
                        help="number of model parallel")
    parser.add_argument("--protocol",
                        type=str,
                        default="ipc",
                        help="cache transfer protocol, only surport ipc now")
    parser.add_argument("--enable_splitwise",
                        type=int,
                        default=0,
                        help="enable splitwise ")
    parser.add_argument("--cache_queue_port",
                        type=int,
                        default=9923,
                        help="cache queue port")
    parser.add_argument("--engine_worker_queue_port",
                        type=int,
                        default=9923,
                        help="engine worker queue port")
    parser.add_argument("--engine_pid",
                        type=str,
                        default=None,
                        help="engine pid")

    parser.add_argument("--num_gpu_blocks",
                        type=int,
                        default=1,
                        help="gpu cache block number")
    parser.add_argument("--num_cpu_blocks",
                        type=int,
                        default=4,
                        help="cpu cache block number")
    parser.add_argument("--block_size",
                        type=int,
                        default=64,
                        help="cache block size(tokens)")
    parser.add_argument("--bytes_per_layer_per_block",
                        type=int,
                        default=1024,
                        help="per layer per block bytes")
    parser.add_argument("--cache_dtype",
                        type=str,
                        default="bfloat16",
                        choices=["uint8", "bfloat16"],
                        help="cache dtype")
    parser.add_argument("--speculative_config",
                        type=json.loads,
                        default="{}",
                        help="speculative config")
    parser.add_argument("--local_data_parallel_id", type=int, default=0)

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
        paddle.set_device(f"gpu:{device}")
        self.gpu_cache_kvs = {}
        self.cpu_cache_kvs = {}
        self.gpu_cache_k_tensors = []
        self.gpu_cache_v_tensors = []
        self.speculative_config = SpeculativeConfig(**args.speculative_config)
        self.num_extra_layers = self.speculative_config.num_extra_cache_layer
        self.num_extra_layer_gpu_blocks = \
            int(args.num_gpu_blocks * \
                self.speculative_config.num_gpu_block_expand_ratio)

        self.swap_to_cpu_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        self.swap_to_gpu_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        self.transfer_task_queue = queue.Queue()  # 用来接收传输任务
        self.tansfer_done_queue = queue.Queue()  # 用来告知任务执行完毕
        self.n_ranks = args.mp_num
        self.rank = rank
        self.device = device

        address = ('0.0.0.0', args.cache_queue_port)
        self.cache_task_queue = EngineCacheQueue(
            address=address,
            is_server=False,
            num_client=args.mp_num,
            client_id=rank,
            local_data_parallel_id=args.local_data_parallel_id)

        self.num_cpu_blocks = args.num_cpu_blocks

        cache_type = args.cache_dtype
        for i in range(args.num_layers + self.num_extra_layers):
            num_gpu_blocks = args.num_gpu_blocks if i < args.num_layers else \
                            self.num_extra_layer_gpu_blocks

            self.gpu_cache_kvs["key_caches_{}_rank{}_device{}".format(
                i, rank, device)] = paddle.full(
                    shape=[
                        num_gpu_blocks,
                        args.kv_num_head,
                        args.block_size,
                        args.head_dim,
                    ],
                    fill_value=0,
                    dtype=cache_type,
                )
            self.gpu_cache_k_tensors.append(
                self.gpu_cache_kvs["key_caches_{}_rank{}_device{}".format(
                    i, rank, device)])
            self.gpu_cache_kvs["value_caches_{}_rank{}_device{}".format(
                i, rank, device)] = paddle.full(
                    shape=[
                        num_gpu_blocks,
                        args.kv_num_head,
                        args.block_size,
                        args.head_dim,
                    ],
                    fill_value=0,
                    dtype=cache_type,
                )
            self.gpu_cache_v_tensors.append(
                self.gpu_cache_kvs["value_caches_{}_rank{}_device{}".format(
                    i, rank, device)])

            set_data_ipc(
                self.gpu_cache_kvs["key_caches_{}_rank{}_device{}".format(
                    i, rank, device)],
                "key_caches_{}_rank{}.device{}".format(i, rank, device))
            set_data_ipc(
                self.gpu_cache_kvs["value_caches_{}_rank{}_device{}".format(
                    i, rank, device)],
                "value_caches_{}_rank{}.device{}".format(i, rank, device))
        cache_kv_size_byte = sum(
            [tmp.numel() * 1 for key, tmp in self.gpu_cache_kvs.items()])
        logger.info(f"device :{self.device}")
        logger.info(f"cache_kv_size_byte : {cache_kv_size_byte}")
        logger.info(
            f"done init cache (full) gmem alloc : {paddle.device.cuda.memory_allocated()}"
        )

        paddle.set_device("cpu")
        self.k_dst_ptrs = []
        self.v_dst_ptrs = []
        for i in range(args.num_layers + self.num_extra_layers):
            self.cpu_cache_kvs["key_caches_{}_rank{}".format(
                i, rank)] = cuda_host_alloc(args.num_cpu_blocks *
                                            args.bytes_per_layer_per_block)
            self.k_dst_ptrs.append(
                self.cpu_cache_kvs["key_caches_{}_rank{}".format(i, rank)])
            self.cpu_cache_kvs["value_caches_{}_rank{}".format(
                i, rank)] = cuda_host_alloc(args.num_cpu_blocks *
                                            args.bytes_per_layer_per_block)
            self.v_dst_ptrs.append(
                self.cpu_cache_kvs["value_caches_{}_rank{}".format(i, rank)])

        cache_ready_signal_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(name="cache_ready_signal",
                                            array=cache_ready_signal_data,
                                            dtype=np.int32,
                                            suffix=args.engine_pid,
                                            create=False)
        self.cache_ready_signal.value[self.rank] = 1

        paddle.set_device(f"gpu:{device}")
        if args.enable_splitwise:
            logger.debug("create cache messager...")
            logger.info(f"{args}")
            from fastdeploy.cache_manager.cache_messager import CacheMessager

            self.cache_messager = CacheMessager(
                splitwise_role=args.splitwise_role,
                transfer_protocol=args.protocol,
                engine_worker_queue_port=args.engine_worker_queue_port,
                local_data_parallel_id=args.local_data_parallel_id,
                gpu_cache_kvs=self.gpu_cache_kvs,
                rank=self.rank,
                nranks=args.mp_num,
                num_layers=args.num_layers + self.num_extra_layers,
                gpu_id=self.device,
                rdma_port=args.rdma_port,
            )
            logger.info("successfully create cache messager")
        logger.info(
            f"done init CacheMessager gmem alloc : {paddle.device.cuda.memory_allocated()}"
        )

        cache_task_broadcast_data = np.zeros(shape=[1], dtype=np.int32)
        self.cache_task_broadcast_signal = IPCSignal(
            name="cache_task_broadcast_signal",
            array=cache_task_broadcast_data,
            dtype=np.int32,
            suffix=args.engine_pid,
            create=False)

    def _do_swap_to_cpu_task(self, swap_node_ids, gpu_block_id, cpu_block_id,
                             event_type, transfer_task_id):
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
            logger.debug(
                f"_do_swap_to_cpu_task: put_transfer_done_signal {result}")
            logger.info(
                f"_do_swap_to_cpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}"
            )

    def _do_swap_to_gpu_task(self, swap_node_ids, gpu_block_id, cpu_block_id,
                             event_type, transfer_task_id):
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
            logger.debug(
                f"_do_swap_to_gpu_task: put_transfer_done_signal {result}")
            logger.info(
                f"_do_swap_to_gpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}"
            )

    def do_data_transfer(self):
        """
        do data transfer task
        """
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
                    data, read_finish = self.cache_task_queue.get_transfer_task(
                    )
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
            except Exception as e:
                logger.info(f"do_data_transfer: error: {e}")

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
            +
            f"task_gpu_block_id {task_gpu_block_id} task_cpu_block_id {task_cpu_block_id} event_type {event_type}"
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
            +
            f"transfer {len(gpu_block_ids)} blocks done  elapsed_time {elasped_time:.4f}"
        )
        return (
            swap_node_ids,
            task_gpu_block_id,
            task_cpu_block_id,
            event_type,
            transfer_task_id,
        )


def main():
    """
    启动cache manager
    """

    cache_manager = CacheTransferManager(args)

    cache_manager.do_data_transfer()


if __name__ == "__main__":

    args = parse_args()
    logger = get_logger("cache_transfer_manager", "cache_transfer_manager.log")
    main()
