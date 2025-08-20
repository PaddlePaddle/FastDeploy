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

import math
import threading
import time
import traceback

import numpy as np
import paddle

from fastdeploy.cache_manager.transfer_factory import IPCCommManager, RDMACommManager
from fastdeploy.inter_communicator import EngineWorkerQueue
from fastdeploy.model_executor.ops.gpu import get_output_kv_signal
from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


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

        assert splitwise_role in [
            "prefill",
            "decode",
        ], "splitwise_role must be prefill or decode"
        self.splitwise_role = splitwise_role
        self.gpu_cache_kvs = gpu_cache_kvs
        self.rank = rank
        self.nranks = nranks
        address = (pod_ip, engine_worker_queue_port)
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
        self.cache_tasks_list = []  # 支持每个元素是一个列表
        self.engine_cache_task_thread_lock = threading.Lock()
        self.engine_cache_tasks = [dict() for _ in range(512)]

        logger.info(f"cache messager init finished, use {transfer_protocol}")

    def prefill_layerwise_send_cache_thread(self):
        """
        layerwise_send_cache_thread:
        send cache to other instance
        """
        while True:
            try:
                cache_info = self.engine_worker_queue.get_cache_info()
                if cache_info:
                    for info in cache_info:
                        if info["request_id"] in self.cache_info:
                            self.cache_info[info["request_id"]].update(info)
                            current_info = self.cache_info[info["request_id"]]
                            if "dest_block_ids" in current_info and "src_block_ids" in current_info:
                                decode_cached_block = len(current_info["src_block_ids"]) - len(
                                    current_info["dest_block_ids"]
                                )
                                current_src_blocks = current_info["src_block_ids"][
                                    -len(current_info["dest_block_ids"]) :
                                ]
                                current_info["send_finished_tokens"] = decode_cached_block * self.block_size
                                current_info["current_tokens"] = current_info["send_finished_tokens"]
                                current_info["src_block_ids"] = current_src_blocks
                                current_info["current_layer_ids"] = -1
                                current_info["current_block_num"] = 0
                                current_info["status"] = "init"
                                logger.info(f"current info: {current_info}")
                                self.cache_info[info["request_id"]] = current_info
                        else:
                            self.cache_info[info["request_id"]] = info
                if not self.cache_info:
                    time.sleep(0.005)
                    continue

                for req_id, item in list(self.cache_info.items()):
                    if "status" not in item:
                        continue
                    if "prefilled_token_num" not in self.engine_cache_tasks[item["current_id"]]:
                        continue
                    if (
                        self.engine_cache_tasks[item["current_id"]]["prefilled_token_num"]
                        <= item["send_finished_tokens"]
                    ):
                        time.sleep(0.005)
                        continue
                    if (
                        self.engine_cache_tasks[item["current_id"]]["prefilled_token_num"] == item["current_tokens"]
                        and self.engine_cache_tasks[item["current_id"]]["prefilled_layer_idx"]
                        == item["current_layer_ids"]
                    ):
                        continue

                    prefill_layer = self.engine_cache_tasks[item["current_id"]]["prefilled_layer_idx"]
                    prefill_tokens = self.engine_cache_tasks[item["current_id"]]["prefilled_token_num"]
                    if prefill_tokens > item["current_tokens"] and item["current_block_num"] != 0:
                        prefill_tokens = item["current_tokens"]
                        prefill_layer = self.num_layers - 1
                        current_block_num = item["current_block_num"]
                    else:
                        current_block_num = (prefill_tokens - item["send_finished_tokens"]) // self.block_size
                        if prefill_tokens == item["total_tokens"]:
                            current_block_num = len(item["src_block_ids"])

                        item["current_block_num"] = current_block_num

                    current_transfer_protocol = item["transfer_protocol"]
                    if item["transfer_protocol"] == "rdma":
                        target_ip = item["ip"]
                        target_id = int(item["rdma_ports"][self.rank])
                        status = self.messager[current_transfer_protocol].connect(target_ip, target_id)
                        if not status:
                            logger.error(f"connect to {target_ip}:{target_id} failed")
                            item["status"] = "error"
                            self.engine_worker_queue.finish_request_barrier.wait()
                            if self.rank == 0:
                                self.engine_worker_queue.put_finished_req([(item["request_id"], "connect error")])
                            continue
                    elif item["transfer_protocol"] == "ipc":
                        target_ip = "0.0.0.0"
                        target_id = int(item["device_ids"][self.rank])

                    src_block_ids = item["src_block_ids"][:current_block_num]
                    dest_block_ids = item["dest_block_ids"][:current_block_num]
                    src_block_ids = paddle.to_tensor(src_block_ids, dtype="int32", place="cpu")
                    dest_block_ids = paddle.to_tensor(dest_block_ids, dtype="int32", place="cpu")
                    logger.debug(
                        f"src_block_ids: {src_block_ids.shape}, dest_block_ids: {dest_block_ids.shape}"
                        f"req_id: {item['request_id']}, current_tokens: {item['current_tokens']}, prefill tokens {prefill_tokens}"
                        f"send_finished_tokens: {item['send_finished_tokens']}, "
                        f"current_layer_ids: {item['current_layer_ids']}, "
                        f"prefilled_layer_idx: {prefill_layer}"
                    )

                    for layer_idx in range(item["current_layer_ids"] + 1, prefill_layer + 1):
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
                            logger.error(
                                f"write cache failed, layer_idx: {layer_idx}, "
                                f"req_id: {item['request_id']}, dest_ip: {target_ip}"
                            )
                            self.engine_cache_tasks[item["current_id"]] = dict()
                            del self.cache_info[req_id]

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
                        item["current_layer_ids"] = layer_idx
                    item["current_tokens"] = prefill_tokens
                    if item["current_layer_ids"] == self.num_layers - 1:
                        if item["transfer_protocol"] == "ipc":
                            self.messager["ipc"].write_block_by_sync(target_id)
                        if prefill_tokens == item["total_tokens"]:
                            logger.info(f"finish write cache {item['request_id']}")
                            self.engine_worker_queue.finish_request_barrier.wait()
                            if self.rank == 0:
                                self.engine_worker_queue.put_finished_req([(item["request_id"], "finished")])
                                logger.info(f"put write cache {item['request_id']}")
                            self.engine_cache_tasks[item["current_id"]] = dict()
                            del self.cache_info[req_id]

                        else:
                            item["current_layer_ids"] = -1
                        item["src_block_ids"] = item["src_block_ids"][current_block_num:]
                        item["dest_block_ids"] = item["dest_block_ids"][current_block_num:]
                        item["send_finished_tokens"] = prefill_tokens
                        item["current_block_num"] = 0

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
                    logger.info(f"tasks_count: {tasks_count}, layer_id: {layer_id}")

                for bi in range(tasks_count):
                    engine_idx = kv_signal_data[3 * bi + 2].numpy().tolist()
                    chuck_token_offset = kv_signal_data[3 * bi + 3].numpy().tolist()
                    current_seq_len = kv_signal_data[3 * bi + 4].numpy().tolist()
                    self.engine_cache_tasks[engine_idx]["prefilled_layer_idx"] = layer_id
                    self.engine_cache_tasks[engine_idx]["prefilled_token_num"] = chuck_token_offset + current_seq_len
            except Exception as e:
                logger.error(f"Consume signals get exception: {e}")
