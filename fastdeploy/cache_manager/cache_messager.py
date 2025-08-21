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
import math
import threading
import time

import numpy as np
import paddle

from fastdeploy.cache_manager.transfer_factory import IPCCommManager, RDMACommManager
from fastdeploy.config import SpeculativeConfig
from fastdeploy.inter_communicator import EngineWorkerQueue, IPCSignal
from fastdeploy.model_executor.ops.gpu import set_data_ipc
from fastdeploy.utils import get_logger


def parse_args():
    """
    从命令行解析参数
    """
    parser = argparse.ArgumentParser("Cache Messager")
    parser.add_argument(
        "--splitwise_role",
        type=str,
        default="mixed",
        help="splitwise role, can be decode, prefill or mixed",
    )
    parser.add_argument("--rank", type=int, default=0, help="current rank")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="model num layers")
    parser.add_argument("--head_dim", type=int, default=1, help="model head dim")
    parser.add_argument("--kv_num_head", type=int, default=1, help="model kv num head")
    parser.add_argument("--rdma_port", type=str, default="", help="rmda port")
    parser.add_argument("--mp_num", type=int, default=1, help="number of model parallel")
    parser.add_argument("--engine_pid", type=str, default=None, help="engine pid")
    parser.add_argument(
        "--protocol",
        type=str,
        default="ipc",
        help="cache transfer protocol, only surport ipc now",
    )
    parser.add_argument("--pod_ip", type=str, default="0.0.0.0", help="pod ip")
    parser.add_argument(
        "--engine_worker_queue_port",
        type=int,
        default=9923,
        help="engine worker queue port",
    )
    parser.add_argument("--num_gpu_blocks", type=int, default=1, help="gpu cache block number")
    parser.add_argument("--block_size", type=int, default=64, help="cache block size(tokens)")
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="bfloat16",
        choices=["uint8", "bfloat16"],
        help="cache dtype",
    )
    parser.add_argument(
        "--speculative_config",
        type=json.loads,
        default="{}",
        help="speculative config",
    )
    parser.add_argument("--local_data_parallel_id", type=int, default=0)

    args = parser.parse_args()
    return args


class CacheMessager:
    """
    CacheMessager is used to send the cache data between the engine worker and the cache server.
    """

    def __init__(
        self,
        splitwise_role,
        transfer_protocol,
        pod_ip,
        engine_worker_queue_port,
        local_data_parallel_id,
        gpu_cache_kvs,
        rank,
        nranks,
        num_hidden_layers,
        gpu_id=0,
        rdma_port=None,
    ):
        """
        Initialize the CacheMessager object.

        Args:
            splitwise_role (str): splitwise_role only can be 'prefill' or 'decode'.
            transfer_protocol (str): support ipc and rdma
            engine_worker_queue_port (int): engine_worker_queue port
            gpu_cache_kvs (dict): GPU kv cache
            rank (int): current rank
            nranks (int): global rank number
            num_hidden_layers (int): model layer number
            gpu_id (int, optional): GPU ID
            rdma_port (int, optional): RDMA port

        Returns:
            None
        """

        assert splitwise_role in [
            "prefill",
            "decode",
        ], "splitwise_role must be prefill or decode"
        self.splitwise_role = splitwise_role
        self.gpu_cache_kvs = gpu_cache_kvs
        self.rank = rank
        self.nranks = nranks
        address = (pod_ip, engine_worker_queue_port + local_data_parallel_id)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            num_client=self.nranks,
            client_id=self.rank,
            local_data_parallel_id=local_data_parallel_id,
        )
        transfer_protocol = transfer_protocol.split(",")

        logger.info(f"splitwise role: {splitwise_role}, {transfer_protocol}" f"rank: {rank}")

        # 1. initialize the cache_k_ptr_list and cache_v_ptr_list
        self.num_hidden_layers = num_hidden_layers
        cache_k_ptr_list = []
        cache_v_ptr_list = []
        cache_k = []
        cache_v = []
        self.messager = {}
        for layer_idx in range(self.num_hidden_layers):
            key_cache = self.gpu_cache_kvs[f"key_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            val_cache = self.gpu_cache_kvs[f"value_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            cache_k.append(key_cache)
            cache_v.append(val_cache)
            cache_k_ptr_list.append(key_cache.data_ptr())
            cache_v_ptr_list.append(val_cache.data_ptr())
        cache_k_ptr_list = np.array(cache_k_ptr_list)
        cache_v_ptr_list = np.array(cache_v_ptr_list)

        # 2. initialize the block_bytes
        cache_shape = key_cache.shape
        max_block_num = cache_shape[0]
        block_bytes = math.prod(cache_shape[1:])
        if key_cache.dtype == paddle.bfloat16:
            block_bytes *= 2
        logger.info(
            f"layers {num_hidden_layers} cache_shape: {cache_shape}, max_block_num: {max_block_num}, "
            f"block_bytes: {block_bytes}, dtype: {key_cache.dtype}"
        )
        self.block_bytes = block_bytes

        # 3. initialize the messager
        for protocol in transfer_protocol:
            if protocol == "ipc":
                self.messager[protocol] = IPCCommManager(
                    self.rank,
                    gpu_id,
                    cache_k,
                    cache_v,
                )
                local_device_id = int(str(cache_k[0].place)[-2])
                logger.info(f"done create ipc_comm with local_device_id:{local_device_id}, ")

            elif protocol == "rdma":
                logger.info(f"splitwise_role rdma: {self.splitwise_role}, rank: {self.rank}, gpu_id: {gpu_id}")

                self.messager[protocol] = RDMACommManager(
                    splitwise_role,
                    rank,
                    gpu_id,
                    cache_k_ptr_list,
                    cache_v_ptr_list,
                    max_block_num,
                    block_bytes,
                    rdma_port,
                )

        self.gpu_id = gpu_id
        self.cache_info = dict()
        self.rank_id = (
            self.rank + local_data_parallel_id * self.nranks
        )  # align with engine worker rank (paddle.distributed.launch)

        connect_rdma_thread = threading.Thread(target=self._handle_connect_task)
        connect_rdma_thread.daemon = True
        connect_rdma_thread.start()

        logger.info(f"cache messager init finished, use {transfer_protocol}")

    def prefill_layerwise_send_cache_thread(self):
        """
        layerwise_send_cache_thread:
        send cache to other instance
        """
        try:
            prefilled_step_idx_data = np.zeros(shape=[1], dtype=np.int32)
            prefilled_layer_idx_data = np.zeros(shape=[1], dtype=np.int32)
            try:
                step_shm_value = IPCSignal(
                    name=f"splitwise_complete_prefilled_step_{self.rank_id}",
                    array=prefilled_step_idx_data,
                    dtype=np.int32,
                    suffix=self.gpu_id,
                    create=True,
                )
                layer_shm_value = IPCSignal(
                    name=f"splitwise_complete_prefilled_layer_{self.rank_id}",
                    array=prefilled_layer_idx_data,
                    dtype=np.int32,
                    suffix=self.gpu_id,
                    create=True,
                )
            except:
                step_shm_value = IPCSignal(
                    name=f"splitwise_complete_prefilled_step_{self.rank_id}",
                    array=prefilled_step_idx_data,
                    dtype=np.int32,
                    suffix=self.gpu_id,
                    create=False,
                )
                layer_shm_value = IPCSignal(
                    name=f"splitwise_complete_prefilled_layer_{self.rank_id}",
                    array=prefilled_layer_idx_data,
                    dtype=np.int32,
                    suffix=self.gpu_id,
                    create=False,
                )

            step_shm_value.value[0] = -1
            layer_shm_value.value[0] = -1

            self.last_step_idx = -1
            self.last_layer_idx = -1  # int32

            max_step_idx = 100003
            engine_recycled_count = 0

            while True:

                cache_info = self.engine_worker_queue.get_cache_info()

                if cache_info:
                    logger.info(f"cache info {cache_info}")
                    for info in cache_info:
                        if info["request_id"] in self.cache_info:
                            self.cache_info[info["request_id"]].update(info)
                            current_info = self.cache_info[info["request_id"]]
                            if "dest_block_ids" in current_info and "src_block_ids" in current_info:
                                current_src_blocks = current_info["src_block_ids"][
                                    -len(current_info["dest_block_ids"]) :
                                ]
                                current_info["src_block_ids"] = current_src_blocks
                                current_info["current_layer_ids"] = 0
                                current_info["status"] = "init"
                                logger.info(f"start cache_infos: {current_info}")
                            self.cache_info[info["request_id"]] = current_info
                        else:
                            self.cache_info[info["request_id"]] = info
                prefilled_layer_idx = layer_shm_value.value[0]
                prefilled_step_idx = step_shm_value.value[0]
                if prefilled_layer_idx == self.num_hidden_layers - 1:
                    time.sleep(0.001)
                    prefilled_layer_idx = layer_shm_value.value[0]
                    prefilled_step_idx = step_shm_value.value[0]

                if prefilled_step_idx == -1:
                    time.sleep(0.001)
                    continue
                if not self.cache_info:
                    time.sleep(0.001)
                    continue
                if self.last_step_idx > prefilled_step_idx:
                    engine_recycled_count += 1
                self.last_step_idx = prefilled_step_idx  # only copy value read from shm memory
                prefilled_step_idx = (
                    prefilled_step_idx + max_step_idx * engine_recycled_count
                )  # remap prefilled_step_idx for comparison

                logger.debug(
                    f"prefilled_layer_idx: {prefilled_layer_idx}, prefilled_step_idx in shm: {self.last_step_idx},"
                    f"prefilled_step_idx: {prefilled_step_idx} engine_recycled_count {engine_recycled_count}"
                )

                for req_id, item in list(self.cache_info.items()):
                    if "status" not in item:
                        continue
                    if "layer_idx" not in item:
                        item["layer_idx"] = 0
                    if item["status"] == "error":
                        del self.cache_info[req_id]
                        continue
                    if item["current_id"] > prefilled_step_idx:
                        continue
                    current_transfer_protocol = item["transfer_protocol"]
                    if item["transfer_protocol"] == "rdma":
                        target_ip = item["ip"]
                        target_id = int(item["rdma_ports"][self.rank])
                        status = self.messager[current_transfer_protocol].connect(target_ip, target_id)
                        if not status:
                            logger.info(f"connect to {target_ip}:{target_id} failed")
                            item["status"] = "error"
                            self.engine_worker_queue.finish_request_barrier.wait()
                            if self.rank == 0:
                                self.engine_worker_queue.put_finished_req([(item["request_id"], "connect error")])
                            continue
                    elif item["transfer_protocol"] == "ipc":
                        target_ip = "0.0.0.0"
                        target_id = int(item["device_ids"][self.rank])
                    src_block_ids = paddle.to_tensor(item["src_block_ids"], dtype="int32", place="cpu")
                    dest_block_ids = paddle.to_tensor(item["dest_block_ids"], dtype="int32", place="cpu")
                    if item["current_id"] < prefilled_step_idx:
                        current_layer_idx = self.num_hidden_layers
                    else:
                        if item["current_id"] == prefilled_step_idx:
                            current_layer_idx = prefilled_layer_idx + 1

                    for layer_idx in range(item["layer_idx"], current_layer_idx):
                        tic = time.time()
                        return_code = self.messager[current_transfer_protocol].write_cache(
                            target_ip,
                            target_id,
                            src_block_ids,
                            dest_block_ids,
                            layer_idx,
                        )
                        if return_code != 0:
                            item["status"] = "error"
                            self.engine_worker_queue.finish_request_barrier.wait()
                            if self.rank == 0:
                                self.engine_worker_queue.put_finished_req([(item["request_id"], "write cache error")])
                            logger.info(
                                f"write cache failed, layer_idx: {layer_idx}, "
                                f"req_id: {item['request_id']}, dest_ip: {target_ip}"
                            )
                            break

                        tok = time.time()
                        cost_time = tok - tic
                        block_num = len(src_block_ids)
                        avg_time_per_block = cost_time * 1000 / block_num  # ms
                        send_cache_speed = block_num * self.block_bytes / 1073741824 / cost_time  # GB/s
                        logger.info(
                            f"finish write cache for a layer, {item['request_id']}, {layer_idx}"
                            f" {current_transfer_protocol}"
                            f"block_num: {block_num}, send_cache_speed(GB/s): {round(send_cache_speed, 5)},"
                            f"avg_time per block(ms): {round(avg_time_per_block, 5)}"
                        )
                    item["layer_idx"] = current_layer_idx
                    if item["layer_idx"] == self.num_hidden_layers:
                        if item["transfer_protocol"] == "ipc":
                            self.messager["ipc"].write_block_by_sync(target_id)
                        logger.info(f"finish write cache {item['request_id']}")
                        self.engine_worker_queue.finish_request_barrier.wait()
                        if self.rank == 0:
                            self.engine_worker_queue.put_finished_req([(item["request_id"], "finished")])
                            logger.info(f"put write cache {item['request_id']}")
                        del self.cache_info[req_id]
                self.last_layer_idx = prefilled_layer_idx

        except Exception as e:
            logger.info(f"prefill layerwise send cache thread has exception: {e}")

    def _handle_connect_task(self):
        while True:
            try:
                task = self.engine_worker_queue.get_connect_rdma_task()
                if task is None:
                    time.sleep(0.001)
                    continue
                logger.info(f"_handle_connect_task recv task: {task}")
                task_id = task["task_id"]
                ip, rdma_port = task["ip"], task["rdma_port"]
                status = self.messager["rdma"].connect(ip, rdma_port)
                if not status:
                    response = {"task_id": task_id, "success": False}
                else:
                    response = {"task_id": task_id, "success": True}
                self.engine_worker_queue.put_connect_rdma_task_response(response)
            except Exception as e:
                logger.error(f"handle_connect_task has exception: {e}")


def main():
    device = args.device_id
    rank = args.rank
    paddle.set_device(f"gpu:{device}")
    cache_type = args.cache_dtype
    speculative_config = SpeculativeConfig(args.speculative_config)
    num_extra_layers = speculative_config.num_extra_cache_layer
    num_extra_layer_gpu_blocks = int(args.num_gpu_blocks * speculative_config.num_gpu_block_expand_ratio)
    gpu_cache_kvs = {}
    gpu_cache_k_tensors = []
    gpu_cache_v_tensors = []

    for i in range(args.num_hidden_layers + num_extra_layers):
        num_gpu_blocks = args.num_gpu_blocks if i < args.num_hidden_layers else num_extra_layer_gpu_blocks

        gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"] = paddle.full(
            shape=[
                num_gpu_blocks,
                args.kv_num_head,
                args.block_size,
                args.head_dim,
            ],
            fill_value=0,
            dtype=cache_type,
        )
        gpu_cache_k_tensors.append(gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"])
        gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"] = paddle.full(
            shape=[
                num_gpu_blocks,
                args.kv_num_head,
                args.block_size,
                args.head_dim,
            ],
            fill_value=0,
            dtype=cache_type,
        )
        gpu_cache_v_tensors.append(gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"])

        set_data_ipc(
            gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"],
            f"key_caches_{i}_rank{rank}.device{device}",
        )
        set_data_ipc(
            gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"],
            f"value_caches_{i}_rank{rank}.device{device}",
        )
    cache_kv_size_byte = sum([tmp.numel() * 1 for key, tmp in gpu_cache_kvs.items()])
    logger.info(f"device :{device}")
    logger.info(f"cache_kv_size_byte : {cache_kv_size_byte}")
    logger.info(f"done init cache (full) gmem alloc : {paddle.device.cuda.memory_allocated()}")

    cache_messager = CacheMessager(
        splitwise_role=args.splitwise_role,
        transfer_protocol=args.protocol,
        pod_ip=args.pod_ip,
        engine_worker_queue_port=args.engine_worker_queue_port,
        local_data_parallel_id=args.local_data_parallel_id,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=rank,
        nranks=args.mp_num,
        num_hidden_layers=args.num_hidden_layers + num_extra_layers,
        gpu_id=device,
        rdma_port=args.rdma_port,
    )

    cache_ready_signal_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
    cache_ready_signal = IPCSignal(
        name="cache_ready_signal",
        array=cache_ready_signal_data,
        dtype=np.int32,
        suffix=args.engine_pid,
        create=False,
    )
    cache_ready_signal.value[rank] = 1
    cache_messager.prefill_layerwise_send_cache_thread()


if __name__ == "__main__":

    args = parse_args()
    rank_id = args.rank + args.local_data_parallel_id * args.mp_num
    logger = get_logger("cache_messager", f"cache_messager_rank{rank_id}.log")

    logger.info("create cache messager...")
    logger.info(f"{args}")
    main()
