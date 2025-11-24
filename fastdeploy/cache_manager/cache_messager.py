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
import queue
import threading
import time
import traceback

import numpy as np
import paddle

from fastdeploy.cache_manager.ops import (
    get_output_kv_signal,
    get_peer_mem_addr,
    memory_allocated,
    set_data_ipc,
    set_device,
)
from fastdeploy.cache_manager.transfer_factory import IPCCommManager, RDMACommManager
from fastdeploy.config import SpeculativeConfig
from fastdeploy.inter_communicator import (
    EngineWorkerQueue,
    IPCSignal,
    shared_memory_exists,
)
from fastdeploy.utils import envs, get_logger

logger = get_logger("cache_messager", "cache_messager.log")


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
    parser.add_argument("--num_layers", type=int, default=1, help="model num layers")
    parser.add_argument("--key_cache_shape", type=str, default="", help="key cache shape")
    parser.add_argument("--value_cache_shape", type=str, default="", help="value cache shape")
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
    parser.add_argument("--cache_queue_port", type=int, default=9924, help="cache queue port")
    parser.add_argument(
        "--engine_worker_queue_port",
        type=int,
        default=9923,
        help="engine worker queue port",
    )
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
        num_layers,
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
            num_layers (int): model layer number
            gpu_id (int, optional): GPU ID
            rdma_port (int, optional): RDMA port

        Returns:
            None
        """
        self.splitwise_role = splitwise_role
        self.gpu_cache_kvs = gpu_cache_kvs
        self.rank = rank
        self.nranks = nranks
        if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
            address = (pod_ip, engine_worker_queue_port)
        else:
            address = f"/dev/shm/fd_task_queue_{engine_worker_queue_port}.sock"
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
        self.num_layers = num_layers
        cache_k_ptr_list = []
        cache_v_ptr_list = []
        cache_k = []
        cache_v = []
        self.messager = {}
        for layer_idx in range(self.num_layers):
            key_cache = self.gpu_cache_kvs[f"key_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            val_cache = self.gpu_cache_kvs[f"value_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            cache_k.append(key_cache)
            cache_v.append(val_cache)
            if paddle.is_compiled_with_xpu():
                cache_k_ptr_list.append(get_peer_mem_addr(key_cache.data_ptr()))
                cache_v_ptr_list.append(get_peer_mem_addr(val_cache.data_ptr()))
            else:
                cache_k_ptr_list.append(key_cache.data_ptr())
                cache_v_ptr_list.append(val_cache.data_ptr())
        cache_k_ptr_list = np.array(cache_k_ptr_list)
        cache_v_ptr_list = np.array(cache_v_ptr_list)

        # 2. initialize the block_bytes
        cache_shape = key_cache.shape
        max_block_num = cache_shape[0]
        block_bytes = math.prod(cache_shape[1:])
        if key_cache.dtype == paddle.bfloat16 or key_cache.dtype == paddle.float16:
            block_bytes *= 2
        logger.info(
            f"layers {num_layers} cache_shape: {cache_shape}, max_block_num: {max_block_num}, "
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
        self.rank_id = self.rank + local_data_parallel_id * self.nranks

        if self.splitwise_role != "mixed":
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
            prefilled_layer_name = f"splitwise_complete_prefilled_layer_{self.rank_id}.{self.gpu_id}"
            prefilled_step_name = f"splitwise_complete_prefilled_step_{self.rank_id}.{self.gpu_id}"
            step_shm_value = IPCSignal(
                name=f"splitwise_complete_prefilled_step_{self.rank_id}",
                array=prefilled_step_idx_data,
                dtype=np.int32,
                suffix=self.gpu_id,
                create=not shared_memory_exists(prefilled_step_name),
            )
            layer_shm_value = IPCSignal(
                name=f"splitwise_complete_prefilled_layer_{self.rank_id}",
                array=prefilled_layer_idx_data,
                dtype=np.int32,
                suffix=self.gpu_id,
                create=not shared_memory_exists(prefilled_layer_name),
            )
            logger.info(f"splitwise_complete_prefilled_step_{self.rank_id}, gpu_id: {self.gpu_id}")

            step_shm_value.value[0] = -1
            layer_shm_value.value[0] = -1

            self.last_step_idx = -1
            self.last_layer_idx = -1  # int32

            max_step_idx = 100003
            engine_recycled_count = 0

            while True:
                cache_info = self.engine_worker_queue.get_cache_info()
                if cache_info:
                    logger.debug(f"cache info {cache_info}")
                    self.engine_worker_queue.cache_info_barrier.wait()
                    for info in cache_info:
                        if info["request_id"] in self.cache_info:
                            self.cache_info[info["request_id"]].update(info)
                            current_info = self.cache_info[info["request_id"]]
                            if "dest_block_ids" in current_info and "src_block_ids" in current_info:
                                current_src_blocks = current_info["src_block_ids"][
                                    -len(current_info["dest_block_ids"]) :
                                ]
                                current_info["src_block_ids"] = current_src_blocks
                                current_info["status"] = "init"
                                logger.info(f"start cache_infos: {current_info}")
                            self.cache_info[info["request_id"]] = current_info
                        else:
                            self.cache_info[info["request_id"]] = info
                prefilled_layer_idx = layer_shm_value.value[0]
                prefilled_step_idx = step_shm_value.value[0]
                if prefilled_layer_idx == self.num_layers - 1:
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
                    if item["current_id"] > prefilled_step_idx:
                        continue
                    current_transfer_protocol = item["transfer_protocol"]
                    if item["transfer_protocol"] == "rdma":
                        target_ip = item["ip"]
                        target_id = int(item["rdma_ports"][self.rank])
                        status = self.messager[current_transfer_protocol].connect(target_ip, target_id)
                        if not status:
                            logger.error(f"connect to {target_ip}:{target_id} failed")
                            item["status"] = "connect error"
                    elif item["transfer_protocol"] == "ipc":
                        target_ip = "0.0.0.0"
                        target_id = int(item["device_ids"][self.rank])
                    src_block_ids = paddle.to_tensor(item["src_block_ids"], dtype="int32", place="cpu")
                    dest_block_ids = paddle.to_tensor(item["dest_block_ids"], dtype="int32", place="cpu")
                    if item["current_id"] < prefilled_step_idx:
                        current_layer_idx = self.num_layers
                    else:
                        current_layer_idx = prefilled_layer_idx + 1
                    if "error" not in item["status"]:
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
                                item["status"] = "write cache error"
                                logger.error(
                                    f"write cache failed, layer_idx: {layer_idx}, "
                                    f"req_id: {item['request_id']}, dest_ip: {target_ip}"
                                )
                                break

                            tok = time.time()
                            cost_time = tok - tic
                            block_num = len(src_block_ids)
                            avg_time_per_block = cost_time * 1000 / block_num  # ms
                            send_cache_speed = block_num * self.block_bytes / 1073741824 / cost_time  # GB/s
                            logger.debug(
                                f"finish write cache for a layer, {item['request_id']}, {layer_idx}"
                                f" {current_transfer_protocol}"
                                f"block_num: {block_num}, send_cache_speed(GB/s): {round(send_cache_speed, 5)},"
                                f"avg_time per block(ms): {round(avg_time_per_block, 5)}"
                            )
                    item["layer_idx"] = current_layer_idx
                    if item["layer_idx"] == self.num_layers:
                        if "error" not in item["status"]:
                            item["status"] = "finished"
                        if item["transfer_protocol"] == "ipc":
                            self.messager["ipc"].write_block_by_sync(target_id)
                        logger.info(f"finish write cache {item['request_id']}")
                        self.engine_worker_queue.finish_send_cache_barrier.wait()
                        self.engine_worker_queue.put_finished_req([[item["request_id"], item["status"]]])
                        logger.info(f"put write cache {item['request_id']}, status {item['status']}")
                        del self.cache_info[req_id]
                self.last_layer_idx = prefilled_layer_idx

        except Exception as e:
            logger.error(f"prefill layerwise send cache thread has exception: {e}, {str(traceback.format_exc())}")

    def _handle_connect_task(self):
        while True:
            try:
                task, _ = self.engine_worker_queue.get_connect_rdma_task()
                if task is None:
                    time.sleep(0.001)
                    continue
                else:
                    self.engine_worker_queue.connect_task_barrier.wait()
                logger.info(f"_handle_connect_task recv task: {task}")
                task_id = task["task_id"]
                ip, rdma_port = task["ip"], task["rdma_ports"][self.rank]
                status = self.messager["rdma"].connect(ip, rdma_port)
                if not status:
                    response = {"task_id": task_id, "success": False}
                else:
                    response = {"task_id": task_id, "success": True}
                self.engine_worker_queue.connect_task_response_barrier.wait()
                self.engine_worker_queue.put_connect_rdma_task_response(response)
            except Exception as e:
                time.sleep(0.001)
                logger.error(f"handle_connect_task has exception: {e}, {str(traceback.format_exc())}")


class CacheMessagerV1:
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
        num_layers,
        gpu_id=0,
        block_size=64,
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
            num_layers (int): model layer number
            gpu_id (int, optional): GPU ID
            rdma_port (int, optional): RDMA port

        Returns:
            None
        """
        self.splitwise_role = splitwise_role
        self.gpu_cache_kvs = gpu_cache_kvs
        self.rank = rank
        self.nranks = nranks
        if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
            address = (pod_ip, engine_worker_queue_port)
        else:
            address = f"/dev/shm/fd_task_queue_{engine_worker_queue_port}.sock"
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            num_client=self.nranks,
            client_id=self.rank,
            local_data_parallel_id=local_data_parallel_id,
        )
        self.block_size = block_size
        transfer_protocol = transfer_protocol.split(",")

        logger.info(f"splitwise role: {splitwise_role}, {transfer_protocol}" f"rank: {rank}")

        # 1. initialize the cache_k_ptr_list and cache_v_ptr_list
        self.num_layers = num_layers
        cache_k_ptr_list = []
        cache_v_ptr_list = []
        cache_k = []
        cache_v = []
        self.messager = {}
        for layer_idx in range(self.num_layers):
            key_cache = self.gpu_cache_kvs[f"key_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            val_cache = self.gpu_cache_kvs[f"value_caches_{layer_idx}_rank{self.rank}_device{gpu_id}"]
            cache_k.append(key_cache)
            cache_v.append(val_cache)
            if paddle.is_compiled_with_xpu():
                cache_k_ptr_list.append(get_peer_mem_addr(key_cache.data_ptr()))
                cache_v_ptr_list.append(get_peer_mem_addr(val_cache.data_ptr()))
            else:
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
            f"layers {num_layers} cache_shape: {cache_shape}, max_block_num: {max_block_num}, "
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
        self.rank_id = self.rank + local_data_parallel_id * self.nranks
        self.engine_cache_task_thread_lock = threading.Lock()
        self.engine_cache_tasks = [dict() for _ in range(512)]
        self.idx_cache_task_dict = {}
        self.cache_prefilled_engine_ids_queue = queue.Queue()  # keep batch slot index for each prefill step
        if splitwise_role == "prefill":
            consume_signals_thread = threading.Thread(target=self.consume_signals)
            consume_signals_thread.daemon = True
            consume_signals_thread.start()
            add_cache_task_thread = threading.Thread(target=self._add_cache_task_thread)
            add_cache_task_thread.daemon = True
            add_cache_task_thread.start()

        if self.splitwise_role != "mixed":
            connect_rdma_thread = threading.Thread(target=self._handle_connect_task)
            connect_rdma_thread.daemon = True
            connect_rdma_thread.start()

        logger.info(f"cache messager init finished, use {transfer_protocol}")

    def _add_cache_task_thread(self):
        while True:
            try:
                cache_info = self.engine_worker_queue.get_cache_info()
                finished_add_cache_task_req_ids = []
                if cache_info:
                    self.engine_worker_queue.cache_info_barrier.wait()
                    for info in cache_info:
                        if info["request_id"] in self.cache_info:
                            self.cache_info[info["request_id"]].update(info)
                            current_info = self.cache_info[info["request_id"]]
                            assert "dest_block_ids" in current_info and "src_block_ids" in current_info
                            finished_add_cache_task_req_ids.append(info["request_id"])
                            decode_cached_block_num = len(current_info["src_block_ids"]) - len(
                                current_info["dest_block_ids"]
                            )
                            padding_decode_block_ids = [-1 for i in range(decode_cached_block_num)] + current_info[
                                "dest_block_ids"
                            ]
                            current_info["dest_block_ids"] = padding_decode_block_ids
                            current_info["decode_cached_tokens"] = decode_cached_block_num * self.block_size
                            current_info["sended_layer_id"] = -1
                            current_info["sended_block_num"] = current_info["decode_cached_tokens"] // self.block_size
                            current_info["status"] = "init"
                            logger.info(f"Get cache info from P: finish add cache task: {current_info}")
                            self.cache_info[info["request_id"]] = current_info
                            self.idx_cache_task_dict[current_info["current_id"]] = current_info
                        else:
                            logger.info(f"Get cache info from D: {info}")
                            self.cache_info[info["request_id"]] = info

                    if finished_add_cache_task_req_ids:
                        self.engine_worker_queue.put_finished_add_cache_task_req(finished_add_cache_task_req_ids)
                    self.engine_worker_queue.finish_add_cache_task_barrier.wait()
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.info(f"add cache task occured error: {e},  {traceback.format_exc()!s}.")

    def prefill_layerwise_send_cache_thread(self):
        """
        layerwise_send_cache_thread:
        send cache to other instance
        """
        while True:
            try:
                batch_engine_signals = self.cache_prefilled_engine_ids_queue.get()
                self.engine_worker_queue.begin_send_cache_barrier.wait()
                block_start_end_list = []
                current_prefilled_token_num_list = []
                for engine_index, current_step_prefilled_token_num in batch_engine_signals:
                    assert (
                        engine_index in self.idx_cache_task_dict
                    ), f"engine_index {engine_index} not in self.idx_cache_task_dict {self.idx_cache_task_dict}"
                    block_id_start = self.idx_cache_task_dict[engine_index]["sended_block_num"]
                    prefilled_token_num = current_step_prefilled_token_num
                    if (
                        prefilled_token_num == self.idx_cache_task_dict[engine_index]["need_prefill_tokens"]
                    ):  # all chunks have been prefilled
                        block_id_end = len(self.idx_cache_task_dict[engine_index]["src_block_ids"])
                    else:
                        block_id_end = prefilled_token_num // self.block_size  # [block_id_start, block_id_end)
                    block_start_end_list.append((block_id_start, block_id_end))
                    current_prefilled_token_num_list.append(prefilled_token_num)
                while True:  # from layer0 to last layer
                    sended_layer_idx = self.idx_cache_task_dict[batch_engine_signals[0][0]]["sended_layer_id"]
                    start_layer_idx = sended_layer_idx + 1
                    with self.engine_cache_task_thread_lock:  # to check end_layer_idx
                        prefilled_layer_idx = self.engine_cache_tasks[batch_engine_signals[0][0]][
                            "prefilled_layer_idx"
                        ]
                        if sended_layer_idx > prefilled_layer_idx:  # computation must in next chunk
                            logger.info(
                                f"current_prefilled_token_num_list[0] {current_prefilled_token_num_list[0]} prefilled_token_num {self.engine_cache_tasks[batch_engine_signals[0][0]]['prefilled_token_num']}"
                            )

                            assert (
                                current_prefilled_token_num_list[0]
                                < self.engine_cache_tasks[batch_engine_signals[0][0]]["prefilled_token_num"]
                            ), "when sended_layer_idx > prefilled_layer_idx, must be in next chunk, but not, sth wrong"
                            end_layer_idx = self.num_layers - 1  # [start_layer_idx, end_layer_idx)
                        else:
                            end_layer_idx = prefilled_layer_idx
                    if sended_layer_idx == prefilled_layer_idx:  # computation not in next layer
                        time.sleep(0.01)
                    for layer_idx in range(start_layer_idx, end_layer_idx + 1):
                        for i, (block_id_start, block_id_end) in enumerate(block_start_end_list):
                            engine_index = batch_engine_signals[i][0]
                            task = self.idx_cache_task_dict[engine_index]
                            req_id = task["request_id"]
                            if (
                                block_id_start >= block_id_end
                            ):  # no blocks need to transfer for this request in this chunk
                                task["sended_layer_id"] += 1
                                assert task["sended_layer_id"] == layer_idx
                                if task["sended_layer_id"] == self.num_layers - 1:
                                    task["sended_layer_id"] = -1
                                continue
                            else:
                                current_transfer_protocol = task["transfer_protocol"]
                                if task["transfer_protocol"] == "rdma":
                                    target_ip = task["ip"]
                                    target_id = int(task["rdma_ports"][self.rank])
                                    if "error" in task["status"]:
                                        continue

                                    # TODO: use is connected to check if the connection is still alive
                                    logger.debug(f"rdma, start connect decode, {target_ip}:{target_id}")
                                    status = self.messager[current_transfer_protocol].connect(target_ip, target_id)
                                    if status:
                                        logger.info(f"connect to {target_ip}:{target_id} success")
                                    else:
                                        logger.error(f"connect to {target_ip}:{target_id} failed")
                                        task["status"] = "connection error"
                                        continue
                                elif task["transfer_protocol"] == "ipc":
                                    target_ip = "0.0.0.0"
                                    target_id = int(task["device_ids"][self.rank])

                                src_block_ids = task["src_block_ids"][block_id_start:block_id_end]
                                dest_block_ids = task["dest_block_ids"][block_id_start:block_id_end]
                                src_block_ids = paddle.to_tensor(src_block_ids, dtype="int32", place="cpu")
                                dest_block_ids = paddle.to_tensor(dest_block_ids, dtype="int32", place="cpu")

                                logger.info(
                                    f"start write cache for a layer, {req_id}, {layer_idx}, {target_ip}, {target_id}, block_id_start {block_id_start} block_id_end {block_id_end}"
                                )
                                tic = time.time()
                                return_code = self.messager[current_transfer_protocol].write_cache(
                                    target_ip,
                                    target_id,
                                    src_block_ids,
                                    dest_block_ids,
                                    layer_idx,
                                )
                                if return_code != 0:
                                    task["status"] = "write cache error"
                                    logger.error(
                                        f"write cache failed, layer_idx: {layer_idx}, req_id: {req_id}, dest_ip: {target_ip}, block_id_start {block_id_start} block_id_end {block_id_end}"
                                    )
                                tok = time.time()
                                cost_time = tok - tic
                                block_num = len(src_block_ids)
                                avg_time_per_block = cost_time * 1000 / block_num  # ms
                                send_cache_speed = block_num * self.block_bytes / 1073741824 / cost_time  # GB/s
                                logger.debug(
                                    f"finish write cache for a layer, {req_id}, {layer_idx}, {target_ip}, {target_id},"
                                    f"block_num: {block_num}, send_cache_speed(GB/s): {round(send_cache_speed, 5)},"
                                    f"avg_time per block(ms): {round(avg_time_per_block, 5)} block_id_start {block_id_start} block_id_end {block_id_end}"
                                )

                                task["sended_layer_id"] += 1
                                assert task["sended_layer_id"] == layer_idx
                                if task["sended_layer_id"] == self.num_layers - 1:
                                    self.idx_cache_task_dict[engine_index]["sended_block_num"] += (
                                        block_id_end - block_id_start
                                    )
                                    if current_prefilled_token_num_list[i] == task["need_prefill_tokens"]:
                                        if "error" not in task["status"]:
                                            task["status"] = "finished"
                                            logger.info(
                                                f"finish write cache for all layers, req_id: {req_id}, block_id_end {block_id_end} need_prefill_tokens {task['need_prefill_tokens']}"
                                            )
                                    else:
                                        task["sended_layer_id"] = -1
                    if end_layer_idx == self.num_layers - 1:
                        with self.engine_cache_task_thread_lock:
                            for engine_idx, _ in batch_engine_signals:
                                task = self.idx_cache_task_dict[engine_idx]
                                if task["status"] == "finished" or ("error" in task["status"]):
                                    if task["transfer_protocol"] == "ipc":
                                        target_id = int(task["device_ids"][self.rank])
                                        self.messager["ipc"].write_block_by_sync(target_id)
                                    self.engine_worker_queue.finish_send_cache_barrier.wait()
                                    self.engine_worker_queue.put_finished_req([[task["request_id"], task["status"]]])
                                    logger.info(f"put write cache {task['request_id']}, status {task['status']}")
                                    self.engine_cache_tasks[task["current_id"]] = dict()
                                    del self.cache_info[task["request_id"]]
                                    del self.idx_cache_task_dict[task["current_id"]]
                        break
            except Exception as e:
                logger.error(f"prefill layerwise send cache thread has exception: {e} {traceback.format_exc()!s}")
                time.sleep(0.01)

    def consume_signals(self):
        paddle.device.set_device("cpu")
        kv_signal_data = paddle.full(shape=[512 * 3 + 2], fill_value=-1, dtype="int32")
        while True:
            try:
                get_output_kv_signal(kv_signal_data, self.rank_id, 0)  # wait_flag
                if not self.cache_info:
                    time.sleep(0.01)
                    continue
                tasks_count = kv_signal_data[0]
                if tasks_count == -1:
                    time.sleep(0.001)
                    continue
                layer_id = kv_signal_data[1].numpy().tolist()
                if layer_id == self.num_layers - 1:
                    logger.info(f"tasks_count: {tasks_count}, layer_id: {layer_id} self.rank_id {self.rank_id}")
                batch_engine_signals = []
                # format for signal to put in cache_prefilled_engine_ids_queue: [(engine_idx1, prefilled_token_num1), (engine_idx2, prefilled_token_num2)]
                with self.engine_cache_task_thread_lock:
                    for bi in range(tasks_count):
                        engine_idx = kv_signal_data[3 * bi + 2].numpy().tolist()
                        chuck_token_offset = kv_signal_data[3 * bi + 3].numpy().tolist()
                        current_seq_len = kv_signal_data[3 * bi + 4].numpy().tolist()
                        self.engine_cache_tasks[engine_idx]["prefilled_layer_idx"] = layer_id
                        self.engine_cache_tasks[engine_idx]["prefilled_token_num"] = (
                            chuck_token_offset + current_seq_len
                        )
                        batch_engine_signals.append((engine_idx, chuck_token_offset + current_seq_len))
                    if layer_id == 0:
                        logger.info(
                            f"Put batch_engine_signals {batch_engine_signals} into cache_prefilled_engine_ids_queue"
                        )
                        self.cache_prefilled_engine_ids_queue.put(batch_engine_signals)
            except Exception as e:
                logger.error(f"Consume signals get exception: {e}")

    def _handle_connect_task(self):
        while True:
            try:
                task, _ = self.engine_worker_queue.get_connect_rdma_task()
                if task is None:
                    time.sleep(0.001)
                    continue
                else:
                    self.engine_worker_queue.connect_task_barrier.wait()
                logger.info(f"_handle_connect_task recv task: {task}")
                task_id = task["task_id"]
                ip, rdma_port = task["ip"], task["rdma_ports"][self.rank]
                status = self.messager["rdma"].connect(ip, rdma_port)
                if not status:
                    response = {"task_id": task_id, "success": False}
                else:
                    response = {"task_id": task_id, "success": True}
                self.engine_worker_queue.connect_task_response_barrier.wait()
                self.engine_worker_queue.put_connect_rdma_task_response(response)
            except Exception as e:
                logger.error(f"handle_connect_task has exception: {e}, {traceback.format_exc()}")


def main():
    device = args.device_id
    rank = args.rank
    set_device(device)
    cache_type = args.cache_dtype
    speculative_config = SpeculativeConfig(args.speculative_config)
    num_extra_layers = speculative_config.num_extra_cache_layer
    key_cache_shape_list = [int(i) for i in args.key_cache_shape.split(",")]
    value_cache_shape_list = []
    if args.value_cache_shape:
        value_cache_shape_list = [int(i) for i in args.value_cache_shape.split(",")]
    total_gpu_blocks = key_cache_shape_list[0]
    num_extra_layer_gpu_blocks = int(total_gpu_blocks * speculative_config.num_gpu_block_expand_ratio)
    gpu_cache_kvs = {}
    gpu_cache_k_tensors = []
    gpu_cache_v_tensors = []

    logger.info(f"[rank {rank}/{args.mp_num}] Initializing kv cache for all layers.")
    for i in range(args.num_layers + num_extra_layers):
        num_gpu_blocks = total_gpu_blocks if i < args.num_layers else num_extra_layer_gpu_blocks
        key_cache_shape = [
            num_gpu_blocks,
            key_cache_shape_list[1],
            key_cache_shape_list[2],
            key_cache_shape_list[3],
        ]
        value_cache_shape = []
        if value_cache_shape_list:
            value_cache_shape = [
                num_gpu_blocks,
                value_cache_shape_list[1],
                value_cache_shape_list[2],
                value_cache_shape_list[3],
            ]
        logger.info(
            f"[rank {rank}/{args.mp_num}] ..creating kv cache for layer {i}: {key_cache_shape} {value_cache_shape}"
        )

        gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"] = paddle.full(
            shape=key_cache_shape,
            fill_value=0,
            dtype=cache_type,
        )
        gpu_cache_k_tensors.append(gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"])
        set_data_ipc(
            gpu_cache_kvs[f"key_caches_{i}_rank{rank}_device{device}"],
            f"key_caches_{i}_rank{rank}.device{device}",
        )
        if value_cache_shape_list:
            gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"] = paddle.full(
                shape=value_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
            gpu_cache_v_tensors.append(gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"])

            set_data_ipc(
                gpu_cache_kvs[f"value_caches_{i}_rank{rank}_device{device}"],
                f"value_caches_{i}_rank{rank}.device{device}",
            )
    cache_kv_size_byte = sum([tmp.numel() * 1 for key, tmp in gpu_cache_kvs.items()])
    logger.info(f"device :{device}")
    logger.info(f"cache_kv_size_byte : {cache_kv_size_byte}")
    logger.info(f"done init cache (full) gmem alloc : {memory_allocated}")

    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
        cache_messager = CacheMessagerV1(
            splitwise_role=args.splitwise_role,
            transfer_protocol=args.protocol,
            pod_ip=args.pod_ip,
            engine_worker_queue_port=args.engine_worker_queue_port,
            local_data_parallel_id=args.local_data_parallel_id,
            gpu_cache_kvs=gpu_cache_kvs,
            rank=rank,
            nranks=args.mp_num,
            num_layers=args.num_layers + num_extra_layers,
            gpu_id=device,
            rdma_port=args.rdma_port,
        )
    else:
        cache_messager = CacheMessager(
            splitwise_role=args.splitwise_role,
            transfer_protocol=args.protocol,
            pod_ip=args.pod_ip,
            engine_worker_queue_port=args.engine_worker_queue_port,
            local_data_parallel_id=args.local_data_parallel_id,
            gpu_cache_kvs=gpu_cache_kvs,
            rank=rank,
            nranks=args.mp_num,
            num_layers=args.num_layers + num_extra_layers,
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
    logger.info(f"[rank {rank}/{args.mp_num}] ✅ kv cache is ready!")
    if args.splitwise_role == "mixed":
        while True:
            time.sleep(1)
    cache_messager.prefill_layerwise_send_cache_thread()


if __name__ == "__main__":

    args = parse_args()
    rank_id = args.rank + args.local_data_parallel_id * args.mp_num
    logger = get_logger("cache_messager", f"cache_messager_rank{rank_id}.log")
    logger.info("create cache messager...")
    logger.info(f"{args}")
    main()
