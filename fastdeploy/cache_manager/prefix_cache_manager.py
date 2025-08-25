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

import heapq
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np

from fastdeploy import envs
from fastdeploy.cache_manager.cache_data import BlockNode, CacheStatus
from fastdeploy.cache_manager.cache_metrics import CacheMetrics
from fastdeploy.inter_communicator import EngineCacheQueue, IPCSignal
from fastdeploy.utils import get_logger

logger = get_logger("prefix_cache_manager", "prefix_cache_manager.log")


class PrefixCacheManager:
    """
    PrefixCacheManager is used to manage the prefix tree and the cache.
    """

    def __init__(
        self,
        config,
        tensor_parallel_size,
        splitwise_role="mixed",
        local_data_parallel_id=0,
    ):
        """
        initialize the PrefixCacheManager
        """

        self.metrics = CacheMetrics()

        if splitwise_role != "mixed":
            self.enable_splitwise = 1
        else:
            self.enable_splitwise = 0
        self.splitwise_role = splitwise_role

        self.cache_config = config.cache_config
        self.speculative_config = config.speculative_config
        self.local_data_parallel_id = local_data_parallel_id

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.num_gpu_blocks = self.cache_config.total_block_num
        else:
            self.num_gpu_blocks = self.cache_config.prefill_kvcache_block_num
        self.num_cpu_blocks = self.cache_config.num_cpu_blocks
        self.gpu_free_block_list = list(range(self.num_gpu_blocks - 1, -1, -1))
        if self.num_cpu_blocks > 0:
            self.cpu_free_block_list = list(range(self.num_cpu_blocks - 1, -1, -1))
        else:
            self.cpu_free_block_list = []
        heapq.heapify(self.gpu_free_block_list)
        heapq.heapify(self.cpu_free_block_list)
        self.node_id_pool = list(range(self.num_gpu_blocks + self.num_cpu_blocks))

        self.radix_tree_root = BlockNode(-1, [], 0, 0, -1, 0, None, None, None)

        # gpu cache data structure
        self.gpu_lru_leaf_heap = []
        self.gpu_lru_leaf_set = set()

        # cpu cache data structure
        self.cpu_lru_leaf_heap = []
        self.cpu_lru_leaf_set = set()

        # swap in/out data structure
        self.request_release_lock = Lock()
        self.task_swapping_event = {}

        self.node_map = {}
        self.req_leaf_map = {}  # {request_id: leaf node}
        self.leaf_req_map = defaultdict(set)
        self.unfilled_req_block_map = defaultdict(list)
        self.cache_info = {}

        self.executor_pool = ThreadPoolExecutor(max_workers=1)
        self.free_gpu_executor_pool = ThreadPoolExecutor(max_workers=1)
        self.free_cpu_executor_pool = ThreadPoolExecutor(max_workers=1)
        self.gpu_free_task_future = None
        self.cache_status_lock = Lock()

        logger.info(
            f"num_gpu_blocks_server_owned {self.num_gpu_blocks} num_cpu_blocks "
            + f"{self.num_cpu_blocks}, bytes_per_layer_per_block {self.cache_config.bytes_per_layer_per_block}"
        )

    def launch_cache_manager(
        self,
        cache_config,
        tensor_parallel_size,
        device_ids,
        pod_ip,
        engine_worker_queue_port,
        pid_suffix,
    ):
        """
        launch_cache_manager function used to initialize the cache manager.
        """
        broadcast_cache_task_flag_array = np.zeros([1], dtype=np.int32)

        self.shm_cache_task_flag_broadcast = IPCSignal(
            name="cache_task_broadcast_signal",
            array=broadcast_cache_task_flag_array,
            dtype=np.int32,
            suffix=pid_suffix,
            create=True,
        )

        self.cache_task_queue = EngineCacheQueue(
            address=(pod_ip, cache_config.cache_queue_port),
            authkey=b"cache_queue_service",
            is_server=False,
            num_client=tensor_parallel_size,
            client_id=0,
            local_data_parallel_id=self.local_data_parallel_id,
        )

        current_dir_path = os.path.split(os.path.abspath(__file__))[0]
        filename = "cache_transfer_manager.py"
        py_path = os.path.join(current_dir_path, filename)

        if (
            hasattr(cache_config.model_cfg, "num_key_value_heads")
            and hasattr(cache_config.model_cfg, "num_key_value_heads")
            and cache_config.model_cfg.num_key_value_heads is not None
            and int(cache_config.model_cfg.num_key_value_heads) > 0
        ):
            kv_num_head = int(cache_config.model_cfg.num_key_value_heads) // tensor_parallel_size
        else:
            kv_num_head = cache_config.model_cfg.num_attention_heads // tensor_parallel_size

        cache_ready_signal_data = np.zeros(shape=[tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=pid_suffix,
            create=True,
        )
        log_dir = envs.FD_LOG_DIR
        cache_manager_processes = []
        for i in range(tensor_parallel_size):
            launch_cmd = (
                "FLAGS_allocator_strategy=auto_growth CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
                + " NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=0"
                + f" {sys.executable} {py_path}"
                + f" --device_id {int(device_ids[i])}"
                + f" --rank {i}"
                + f" --splitwise_role {self.splitwise_role}"
                + f" --num_layers {cache_config.model_cfg.num_hidden_layers}"
                + f" --head_dim {cache_config.model_cfg.head_dim}"
                + f" --kv_num_head {kv_num_head}"
                + f" --mp_num {tensor_parallel_size}"
                + f" --cache_dtype {cache_config.cache_dtype}"
                + f" --cache_queue_port {cache_config.cache_queue_port}"
                + f" --enable_splitwise {int(self.enable_splitwise)}"
                + f" --pod_ip {pod_ip}"
                + f" --engine_worker_queue_port {engine_worker_queue_port}"
                + f" --num_gpu_blocks {cache_config.total_block_num}"
                + f" --num_cpu_blocks {cache_config.num_cpu_blocks}"
                + f" --bytes_per_layer_per_block {cache_config.bytes_per_layer_per_block}"
                + f" --block_size {cache_config.block_size}"
                + f" --engine_pid {pid_suffix}"
                + f" --protocol {cache_config.cache_transfer_protocol}"
                + f" --local_data_parallel_id {self.local_data_parallel_id}"
                + f" --rdma_port {cache_config.rdma_comm_ports[i] if cache_config.rdma_comm_ports is not None else '0'}"
                + f" --speculative_config '{self.speculative_config.to_json_string()}'"
                + f" >{log_dir}/launch_cache_manager_{int(device_ids[i])}.log 2>&1"
            )
            logger.info(f"Launch cache transfer manager, command:{launch_cmd}")
            cache_manager_processes.append(subprocess.Popen(launch_cmd, shell=True, preexec_fn=os.setsid))
        # 等待cache初始化完毕
        logger.info("Waiting for cache transfer manager ready...")
        while np.sum(self.cache_ready_signal.value) != tensor_parallel_size:
            time.sleep(1)
        exit_code = cache_manager_processes[-1].poll()
        if exit_code is None:
            logger.info("Launch cache transfer manager successful")
        else:
            logger.info("Launch cache transfer manager failed, see launch_cache_manager.log for more information")

        if cache_config.enable_hierarchical_cache and self.num_cpu_blocks > 0:
            logger.info("Enable hierarchical cache.")
            self._enable_cpu_cache()
        return cache_manager_processes

    def update_cache_config(self, cache_config):
        """
        update cache config
        """
        self.cache_config = cache_config
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.num_gpu_blocks = cache_config.total_block_num
            self.gpu_free_block_list = list(
                range(self.num_gpu_blocks - 1, -1, -1)
            )  # All gpu blocks are managed by cache manager
        else:
            self.num_gpu_blocks = cache_config.prefill_kvcache_block_num
            self.gpu_free_block_list = list(
                range(self.num_gpu_blocks - 1, -1, -1)
            )  # Only block table divided for prefill managed by server

        heapq.heapify(self.gpu_free_block_list)
        self.node_id_pool = list(range(self.num_gpu_blocks + self.num_cpu_blocks))

    def _enable_cpu_cache(self):
        """
        _enable_cpu_cache function used to enable cpu cache.
        """

        # ipc_cache_queue_port = self.cache_config.cache_queue_port
        # self.cache_task_queue = CacheQueueManager(
        #     rank=0,
        #     mp_num=tensor_parallel_size,
        #     port=ipc_cache_queue_port,
        # )
        # 开启获取传输任务结果的监听线程
        self.transfer_recv_thread = threading.Thread(target=self.recv_data_transfer_result)
        self.transfer_recv_thread.start()

    def can_allocate_gpu_blocks(self, num_blocks: int):
        """
        Check if num_blocks gpu blocks can be allocated.
        """
        if len(self.gpu_free_block_list) < num_blocks:
            return False
        else:
            return True

    def allocate_gpu_blocks(self, num_blocks):
        """
        allocate gpu blocks.
        """
        assert num_blocks <= len(
            self.gpu_free_block_list
        ), f"gpu free block num: {len(self.gpu_free_block_list)} < needed number {num_blocks}"
        allocated_block_ids = [heapq.heappop(self.gpu_free_block_list) for i in range(num_blocks)]
        logger.info(
            f"allocate_gpu_blocks: {allocated_block_ids}, len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )
        return allocated_block_ids

    def recycle_gpu_blocks(self, gpu_block_ids):
        """
        recycle gpu blocks.
        """
        logger.info(
            f"recycle_gpu_blocks: {gpu_block_ids}, len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )
        if isinstance(gpu_block_ids, list):
            for gpu_block_id in gpu_block_ids:
                heapq.heappush(self.gpu_free_block_list, gpu_block_id)
        else:
            heapq.heappush(self.gpu_free_block_list, gpu_block_ids)

    def allocate_cpu_blocks(self, num_blocks):
        """
        allocate cpu blocks.
        """
        assert num_blocks <= len(
            self.cpu_free_block_list
        ), f"cpu free block num: {len(self.cpu_free_block_list)} < needed number {num_blocks}"
        allocated_block_ids = [heapq.heappop(self.cpu_free_block_list) for i in range(num_blocks)]
        logger.info(
            f"allocate_cpu_blocks: {allocated_block_ids}, len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        return allocated_block_ids

    def recycle_cpu_blocks(self, cpu_block_ids):
        """
        recycle cpu blocks.
        """
        logger.info(
            f"recycle_cpu_blocks: {cpu_block_ids}, len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        if isinstance(cpu_block_ids, list):
            for cpu_block_id in cpu_block_ids:
                heapq.heappush(self.cpu_free_block_list, cpu_block_id)
        else:
            heapq.heappush(self.cpu_free_block_list, cpu_block_ids)

    def issue_swap_task(
        self,
        transfer_task_id,
        swap_node_ids,
        gpu_block_ids,
        cpu_block_ids,
        event_type,
        is_sync=True,
    ):
        """
        start data swap task
        args:
            transfer_task_id: transfer task id
            swap_node_ids:    to swap node id list
            gpu_block_ids:    to swap gpu block id list
            cpu_block_ids:    to swap cpu block id list
            event_type:       CacheStatus.SWAP2GPU or CacheStatus.SWAP2CPU
            is_sync:          bool, whether to wait for the result of the swap task
        """

        self.task_swapping_event[transfer_task_id] = Event()
        self.cache_task_queue.put_transfer_task(
            (
                swap_node_ids,
                gpu_block_ids,
                cpu_block_ids,
                event_type,
                transfer_task_id,
            )
        )
        if is_sync:
            self.sync_swap_task(transfer_task_id)

    def sync_swap_task(self, transfer_task_id):
        """
        sync swap task
        """
        self.task_swapping_event[transfer_task_id].wait()
        del self.task_swapping_event[transfer_task_id]

    def _check_validity(self, req_id, match_gpu_blocks_num, expected_block_num):
        """
        check enough gpu memory to allocate cache
        """
        if expected_block_num - match_gpu_blocks_num > len(self.gpu_free_block_list):
            msg = (
                f"request_block_ids: request block for req_id {req_id} failed. "
                + f"matched gpu block num: {match_gpu_blocks_num} require extra gpu block num: "
                + f"{expected_block_num - match_gpu_blocks_num} > free block num: {len(self.gpu_free_block_list)}"
            )
            logger.info(msg)
            raise Exception("Not enough GPU memory to allocate cache")

    def _prepare_cpu_cache(
        self,
        req_id,
        swap_node_ids,
        gpu_recv_block_ids,
        cpu_recv_block_ids,
        match_cpu_block_ids,
    ):
        """
        将cpu cache转移到GPU
        """
        transfer_task_id = req_id
        need_transfer_task_gpu_block_ids = []
        need_transfer_task_cpu_block_ids = []

        for tmp_gpu_block_id in gpu_recv_block_ids:
            need_transfer_task_gpu_block_ids.append(tmp_gpu_block_id)
        for tmp_cpu_block_id in match_cpu_block_ids:
            need_transfer_task_cpu_block_ids.append(tmp_cpu_block_id)

        assert len(need_transfer_task_gpu_block_ids) == len(need_transfer_task_cpu_block_ids)
        logger.info(f"request_block_ids: req_id {req_id} issue_swap_task transfer_task_id {transfer_task_id}")
        self.issue_swap_task(
            transfer_task_id,
            swap_node_ids,
            need_transfer_task_gpu_block_ids,
            need_transfer_task_cpu_block_ids,
            CacheStatus.SWAP2GPU,
            True,
        )

    def _prepare_cache(
        self,
        req_id,
        input_ids,
        block_size,
        expected_block_num,
        match_gpu_block_ids,
        match_cpu_block_ids,
        match_node_ids,
    ):
        """
        prepare cache for request
        """

        match_gpu_blocks_num = len(match_gpu_block_ids)
        match_cpu_blocks_num = len(match_cpu_block_ids)
        matched_block_num = match_gpu_blocks_num + match_cpu_blocks_num

        cpu_recv_block_ids = []
        gpu_recv_block_ids = []
        gpu_extra_block_ids = []

        # allocate gpu cache for matched cpu blocks
        if match_cpu_blocks_num > 0:
            gpu_recv_block_ids = self.allocate_gpu_blocks(match_cpu_blocks_num)
        # allocate gpu cache
        gpu_extra_block_num = expected_block_num - matched_block_num
        if gpu_extra_block_num > 0:
            gpu_extra_block_ids = self.allocate_gpu_blocks(gpu_extra_block_num)

        if len(gpu_recv_block_ids) > 0:
            self._prepare_cpu_cache(
                req_id,
                match_node_ids,
                gpu_recv_block_ids,
                cpu_recv_block_ids,
                match_cpu_block_ids,
            )

        return gpu_recv_block_ids, gpu_extra_block_ids

    def get_required_block_num(self, input_token_num, block_size):
        """
        get required block num by input token num and block size
        """
        return (input_token_num + block_size - 1) // block_size

    def update_cache_blocks(self, task, block_size):
        """
        update cache blocks for a task.
        # TODO(chengyanfu): support async update

        Parameters:
        - task: Task
        - block_size: Size per block (in tokens)
        """
        try:
            req_id = task.request_id
            num_cached_tokens = task.num_cached_tokens
            block_tables = task.block_tables

            last_node, input_ids = self.cache_info[req_id]
            left_input_ids = input_ids[num_cached_tokens:]
            gpu_extra_block_ids = block_tables[num_cached_tokens // block_size :]

            with self.request_release_lock:
                current_time = time.time()
                leaf_node = self.build_path(
                    req_id=req_id,
                    current_time=current_time,
                    input_ids=input_ids,
                    left_input_ids=left_input_ids,
                    gpu_block_ids=gpu_extra_block_ids,
                    block_size=block_size,
                    last_node=last_node,
                    reverved_dec_block_num=0,
                )
                self.req_leaf_map[req_id] = leaf_node
                self.leaf_req_map[leaf_node].add(req_id)
                self.cache_info[req_id] = (leaf_node, input_ids)
        except Exception as e:
            logger.error(f"update_cache_blocks, error: {type(e)} {e}, {str(traceback.format_exc())}")
            raise e

    def request_match_blocks(self, task, block_size, *args):
        """
        get match blocks info for a task.
        This is a synchronous interface. If CPU-to-GPU data transfer occurs,
        it will block until synchronization completes.
        Callers requiring asynchronous behavior should invoke this via a thread pool.

        Note: This function may allocate GPU blocks for matched CPU Cache

        Parameters:
        - task: Task dictionary
        - block_size: Size per block (in tokens)

        Returns:
        - common_block_ids: List of matched shared blocks
        - unique_block_ids: List of exclusively allocated blocks
        """
        with self.request_release_lock:
            try:
                hit_info = {}
                hit_info["gpu_cache_blocks"] = 0
                hit_info["cpu_cache_blocks"] = 0
                self.metrics.req_count += 1
                input_ids = task.prompt_token_ids
                req_id = task.request_id
                logger.info(f"request_block_ids: start to allocate blocks for req_id {req_id}")
                input_token_num = len(input_ids)
                common_block_ids = []
                # 1. match block
                (
                    match_gpu_block_ids,
                    match_cpu_block_ids,
                    swap_node_ids,
                    match_block_node,
                    gpu_match_token_num,
                    cpu_match_token_num,
                ) = self.match_block(req_id, input_ids, block_size)

                #  update matched node info
                self._update_matched_node_info(req_id, match_block_node, current_time=time.time())

                # 2. prepare cache
                #  allocate gpu cache for matched cpu blocks
                gpu_recv_block_ids = []
                match_cpu_blocks_num = len(match_cpu_block_ids)
                if self.can_allocate_gpu_blocks(num_blocks=match_cpu_blocks_num):
                    if match_cpu_blocks_num > 0:
                        gpu_recv_block_ids = self.allocate_gpu_blocks(match_cpu_blocks_num)
                        if len(gpu_recv_block_ids) > 0:
                            self._prepare_cpu_cache(
                                req_id=req_id,
                                swap_node_ids=swap_node_ids,
                                gpu_recv_block_ids=gpu_recv_block_ids,
                                match_cpu_block_ids=match_cpu_block_ids,
                                cpu_recv_block_ids=[],
                            )
                else:
                    raise Exception("Not enough GPU memory to allocate cache for matched CPU Cache")

                #  record request cache info
                self.cache_info[req_id] = (match_block_node, input_ids)

                # 3. update metrics
                matched_token_num = gpu_match_token_num + cpu_match_token_num
                common_block_ids = match_gpu_block_ids + gpu_recv_block_ids
                if matched_token_num > 0:
                    self.metrics.hit_req_count += 1
                self.metrics.calculate_hit_metrics(
                    req_id,
                    cpu_match_token_num,
                    gpu_match_token_num,
                    input_token_num,
                )
                hit_info["gpu_cache_blocks"] = gpu_match_token_num // block_size
                hit_info["cpu_cache_blocks"] = cpu_match_token_num // block_size
                self.metrics._update_history_hit_metrics()
                if self.metrics.req_count % 10000 == 0:
                    self.metrics.reset_metrics()
                logger.info(
                    f"request_block_ids: request block for req_id {req_id}: common_block_ids {common_block_ids}"
                )
                return common_block_ids, matched_token_num, hit_info
            except Exception as e:
                logger.error(f"request_block_ids: error: {type(e)} {e}, {str(traceback.format_exc())}")
                raise e

    def request_block_ids(self, task, block_size, dec_token_num, *args):
        """
        Allocate blocks for a task.
        This is a synchronous interface. If CPU-to-GPU data transfer occurs,
        it will block until synchronization completes.
        Callers requiring asynchronous behavior should invoke this via a thread pool.

        Parameters:
        - task: Task dictionary
        - block_size: Size per block (in tokens)
        - dec_token_num: Number of tokens reserved for decoding on the server side

        Returns:
        - common_block_ids: List of matched shared blocks
        - unique_block_ids: List of exclusively allocated blocks
        """
        with self.request_release_lock:
            try:
                hit_info = {}
                hit_info["gpu_cache_blocks"] = 0
                hit_info["cpu_cache_blocks"] = 0
                self.metrics.req_count += 1
                input_ids = task.prompt_token_ids
                req_id = task.request_id
                logger.info(f"request_block_ids: start to allocate blocks for req_id {req_id}")
                input_token_num = len(input_ids)
                common_block_ids = []
                unique_block_ids = []
                # 1. match block
                (
                    match_gpu_block_ids,
                    match_cpu_block_ids,
                    swap_node_ids,
                    match_block_node,
                    gpu_match_token_num,
                    cpu_match_token_num,
                ) = self.match_block(req_id, input_ids, block_size)
                match_gpu_blocks_num = len(match_gpu_block_ids)
                matched_token_num_in_cpu_and_gpu = gpu_match_token_num + cpu_match_token_num
                # check enough gpu memory to allocate cache
                block_num = (input_token_num + block_size - 1 + dec_token_num) // block_size
                self._check_validity(req_id, match_gpu_blocks_num, block_num)
                # update matched node info
                current_time = time.time()
                self._update_matched_node_info(req_id, match_block_node, current_time)
                # 2. prepare cache
                (
                    gpu_recv_block_ids,
                    gpu_extra_block_ids,
                ) = self._prepare_cache(
                    req_id,
                    input_ids,
                    block_size,
                    block_num,
                    match_gpu_block_ids,
                    match_cpu_block_ids,
                    swap_node_ids,
                )
                # update matched token num
                matched_block_num = gpu_match_token_num + cpu_match_token_num

                common_block_ids = match_gpu_block_ids + gpu_recv_block_ids
                unique_block_ids = gpu_extra_block_ids

                dec_block_num = dec_token_num // block_size
                left_input_ids = input_ids[matched_token_num_in_cpu_and_gpu:]  # 没在前缀树中的token
                gpu_build_path_block_ids = []

                gpu_build_path_block_ids = gpu_extra_block_ids

                leaf_node = self.build_path(
                    req_id,
                    current_time,
                    input_ids,
                    left_input_ids,
                    gpu_build_path_block_ids,
                    block_size,
                    match_block_node,
                    dec_block_num,
                )
                self.req_leaf_map[req_id] = leaf_node
                self.leaf_req_map[leaf_node].add(req_id)
                # 3. update metrics
                if matched_block_num > 0:
                    self.metrics.hit_req_count += 1
                self.metrics.calculate_hit_metrics(
                    req_id,
                    cpu_match_token_num,
                    gpu_match_token_num,
                    input_token_num,
                )
                hit_info["gpu_cache_blocks"] = gpu_match_token_num // block_size
                hit_info["cpu_cache_blocks"] = cpu_match_token_num // block_size
                self.metrics._update_history_hit_metrics()
                if self.metrics.req_count % 10000 == 0:
                    self.metrics.reset_metrics()
                logger.info(
                    f"request_block_ids: request block for req_id {req_id}: common_block_ids "
                    + f"{common_block_ids}, unique_block_ids {unique_block_ids}"
                )
                return common_block_ids, unique_block_ids, hit_info
            except Exception as e:
                logger.error(f"request_block_ids: error: {type(e)} {e}, {str(traceback.format_exc())}")
                raise e

    def release_block_ids_async(self, task):
        """
        async release block ids
        """
        return self.executor_pool.submit(self.release_block_ids, task)

    def release_block_ids(self, task):
        """
        release block ids
        """
        with self.request_release_lock:
            try:
                req_id = task.request_id
                leaf_node = self.req_leaf_map.pop(req_id)
                if leaf_node in self.leaf_req_map:
                    self.leaf_req_map[leaf_node].remove(req_id)
                    if not (self.leaf_req_map[leaf_node]):
                        del self.leaf_req_map[leaf_node]
                node = leaf_node
                while node != self.radix_tree_root:
                    if req_id in node.req_id_set:
                        node.req_id_set.remove(req_id)
                    node.decrement_shared_count()
                    node = node.parent

                if req_id in self.cache_info:
                    del self.cache_info[req_id]

                logger.info(f"release_block_ids: req_id {req_id} leaf_node {leaf_node}")

                if leaf_node == self.radix_tree_root:
                    self.recycle_gpu_blocks(self.unfilled_req_block_map[req_id])
                    del self.unfilled_req_block_map[req_id]
                    return

                if leaf_node in self.gpu_lru_leaf_set:
                    return
                if leaf_node.shared_count == 0 and leaf_node.is_gpu_leaf_node and leaf_node.is_persistent is False:
                    self.gpu_lru_leaf_set.add(leaf_node)
                    heapq.heappush(self.gpu_lru_leaf_heap, leaf_node)
                logger.info(
                    f"release_block_ids: req_id {req_id} has been finished, "
                    + f"current gpu_lru_leaf_heap length {len(self.gpu_lru_leaf_heap)}"
                )
                return
            except Exception as e:
                logger.error(f"release_block_ids: error: {type(e)} {e}, {str(traceback.format_exc())}")
                raise e

    def _handle_free_gpu_node_without_cpu(self, node):
        """
        GPU node eviction
        """
        node.cache_status = CacheStatus.CPU

        self.node_id_pool.append(node.node_id)
        if node.node_id in self.node_map:
            del self.node_map[node.node_id]
        logger.info(f"free_block_ids_async: free node {node}")

        self.recycle_gpu_blocks(node.reverved_dec_block_ids)
        node.reverved_dec_block_ids = []
        self.recycle_gpu_blocks(node.block_id)

    def _handle_free_gpu_node_with_cpu(
        self,
        node,
        hash_value_input_ids_map,
        hash_value_depth_map,
        need_recycle_gpu_block_ids,
        hash_value_gpu_block_ids_map,
        hash_value_swap_node_ids_map,
    ):
        """
        GPU node eviction in hierarchical cache layers
        """

        self.recycle_gpu_blocks(node.reverved_dec_block_ids)
        node.reverved_dec_block_ids = []

        need_recycle_gpu_block_ids.append(node.block_id)
        hash_value_gpu_block_ids_map[node.input_hash_value].append(node.block_id)
        hash_value_swap_node_ids_map[node.input_hash_value].append(node.node_id)

    def _evict_cache_async(
        self,
        future,
        total_gpu_free_count,
        hash_value_gpu_block_ids_map,
        hash_value_block_ids_map,
        hash_value_swap_node_ids_map,
        hash_value_input_ids_map,
        hash_value_depth_map,
    ):
        """
        evict cache async (GPU --> CPU)
        """
        if future is not None:
            future.result()
        transfer_task_id = str(uuid.uuid4())
        swap_node_ids = []
        need_transfer_task_gpu_block_ids = []
        need_transfer_task_cpu_block_ids = []
        cpu_block_ids = self.allocate_cpu_blocks(total_gpu_free_count)
        for input_hash_value in hash_value_gpu_block_ids_map.keys():
            need_transfer_task_gpu_block_ids.extend(reversed(hash_value_gpu_block_ids_map[input_hash_value]))
            all_allocated_cpu_block_ids = []
            for _ in reversed(hash_value_gpu_block_ids_map[input_hash_value]):
                cpu_block_id_t = cpu_block_ids.pop(0)
                all_allocated_cpu_block_ids.append(cpu_block_id_t)
                need_transfer_task_cpu_block_ids.append(cpu_block_id_t)

            swap_node_ids.extend(reversed(hash_value_swap_node_ids_map[input_hash_value]))
        logger.info(
            "free_block_ids_async: issue transfer task: "
            + f"transfer_task_id {transfer_task_id}: "
            + f"swap_node_ids {swap_node_ids} need_transfer_task_gpu_block_ids "
            + f"{need_transfer_task_gpu_block_ids}, need_transfer_task_cpu_block_ids "
            + f"{need_transfer_task_cpu_block_ids}, CacheStatus.SWAP2CPU"
        )
        self.issue_swap_task(
            transfer_task_id,
            swap_node_ids,
            need_transfer_task_gpu_block_ids,
            need_transfer_task_cpu_block_ids,
            CacheStatus.SWAP2CPU,
            True,
        )

        logger.info(
            "free_block_ids_async: after free, " + f"len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )

    def free_block_ids_async(self, need_block_num):
        """
        free block ids async
        args：
            need_query_block_num: max number of gpu blocks to free
        """
        with self.request_release_lock:
            if self.gpu_free_task_future is not None:
                if not self.gpu_free_task_future.done():
                    return
                else:
                    self.gpu_free_task_future.result()
                    self.gpu_free_task_future = None
            try:
                need_recycle_gpu_block_ids = []

                hash_value_input_ids_map = {}
                hash_value_block_ids_map = defaultdict(list)
                hash_value_depth_map = {}

                hash_value_swap_node_ids_map = defaultdict(list)
                hash_value_gpu_block_ids_map = defaultdict(list)
                total_gpu_free_count = 0

                while True:
                    if len(self.gpu_lru_leaf_heap) == 0:
                        break
                    if total_gpu_free_count >= need_block_num:
                        break
                    node = heapq.heappop(self.gpu_lru_leaf_heap)
                    self.gpu_lru_leaf_set.remove(node)
                    if (
                        not self.cache_config.enable_hierarchical_cache
                        or self.cache_config.num_cpu_blocks < need_block_num
                    ):
                        if node.shared_count == 0 and node.is_gpu_leaf_node:  # 直接回收
                            self._handle_free_gpu_node_without_cpu(node)
                            total_gpu_free_count += 1
                            cur_node = node
                            node = node.parent
                            if cur_node.hash_value in node.children:
                                del node.children[cur_node.hash_value]
                            if not node.children:
                                if node in self.gpu_lru_leaf_set:
                                    continue
                                if (
                                    node != self.radix_tree_root
                                    and node.shared_count == 0
                                    and node.is_gpu_leaf_node
                                    and node.is_persistent is False
                                ):
                                    heapq.heappush(self.gpu_lru_leaf_heap, node)
                                    self.gpu_lru_leaf_set.add(node)
                        else:
                            continue
                    else:
                        if node.shared_count == 0 and node.is_gpu_leaf_node:
                            node.cache_status = CacheStatus.SWAP2CPU
                        else:
                            continue
                        self._handle_free_gpu_node_with_cpu(
                            node,
                            hash_value_input_ids_map,
                            hash_value_depth_map,
                            need_recycle_gpu_block_ids,
                            hash_value_gpu_block_ids_map,
                            hash_value_swap_node_ids_map,
                        )
                        total_gpu_free_count += 1

                        node = node.parent
                        if node in self.gpu_lru_leaf_set:
                            continue
                        if (
                            node != self.radix_tree_root
                            and node.shared_count == 0
                            and node.is_gpu_leaf_node
                            and node.is_persistent is False
                        ):
                            heapq.heappush(self.gpu_lru_leaf_heap, node)
                            self.gpu_lru_leaf_set.add(node)

                # swap cache to cpu
                if hash_value_gpu_block_ids_map:
                    cpu_free_future = None
                    if total_gpu_free_count > len(self.cpu_free_block_list):
                        cpu_free_count = total_gpu_free_count
                        if cpu_free_count < need_block_num:
                            cpu_free_count = need_block_num
                        cpu_free_future = self.free_cpu_executor_pool.submit(self.free_cpu_block_ids, cpu_free_count)
                    self.gpu_free_task_future = self.free_gpu_executor_pool.submit(
                        self._evict_cache_async,
                        cpu_free_future,
                        total_gpu_free_count,
                        hash_value_gpu_block_ids_map,
                        hash_value_block_ids_map,
                        hash_value_swap_node_ids_map,
                        hash_value_input_ids_map,
                        hash_value_depth_map,
                    )
                else:
                    self.gpu_free_task_future = None
            except Exception as e:
                logger.error(f"free_block_ids_async: error: {type(e)} {e}, {str(traceback.format_exc())}")
                raise e

    def free_cpu_block_ids(self, need_block_num):
        """
        Evict CPU blocks (at least need_block_num blocks)
        Parameters:
        - need_block_num: Number of CPU blocks required to evict

        Returns:
        - freed_block_num: Number of CPU blocks successfully evicted
        """
        hash_value_block_ids_map = defaultdict(list)
        total_cpu_free_count = 0
        with self.request_release_lock:
            while True:
                if len(self.cpu_lru_leaf_heap) == 0:
                    break
                if total_cpu_free_count >= need_block_num:
                    break

                node = heapq.heappop(self.cpu_lru_leaf_heap)
                self.cpu_lru_leaf_set.remove(node)
                tmp_block_ids = []
                if node.shared_count == 0 and node.cache_status == CacheStatus.CPU and node.is_cpu_leaf_node:

                    self.recycle_cpu_blocks(node.block_id)
                    hash_value_block_ids_map[node.input_hash_value].extend(reversed(tmp_block_ids))
                    logger.info(f"free_cpu_block_ids: free node {node}")

                    self.node_id_pool.append(node.node_id)
                    total_cpu_free_count += 1
                    if node.node_id in self.node_map:
                        del self.node_map[node.node_id]
                    cur_node = node
                    node = node.parent
                    if cur_node.hash_value in node.children:
                        del node.children[cur_node.hash_value]
                    if not node.children:
                        if node in self.cpu_lru_leaf_set:
                            continue
                        if (
                            node != self.radix_tree_root
                            and node.shared_count == 0
                            and node.is_cpu_leaf_node
                            and node.cache_status == CacheStatus.CPU
                        ):
                            heapq.heappush(self.cpu_lru_leaf_heap, node)
                            self.cpu_lru_leaf_set.add(node)
        logger.info(
            "free_cpu_block_ids: after free, " + f"len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        return total_cpu_free_count

    def cal_block_hash(self, block):
        """
        calculate hash value of a block
        """
        return hash(tuple(block))

    def match_block(self, req_id, input_ids, block_size):
        """
        Args:
            req_id: Task request ID
            input_ids: Input token IDs
            block_size: Size of each block

        Returns:
            match_gpu_block_ids: List of matched GPU block IDs
            match_cpu_block_ids: List of matched CPU block IDs
            swap_node_ids: List of node IDs requiring swap operations
            match_block_node: Last matched node in the path
            gpu_match_token_num: Number of tokens matched in GPU blocks
            cpu_match_token_num: Number of tokens matched in CPU blocks
        """

        total_token_num = len(input_ids)
        current_match_node = self.radix_tree_root  # 从根节点开始搜
        match_gpu_block_ids = []
        match_cpu_block_ids = []
        match_node_ids = []
        match_token_num = 0
        cpu_match_token_num = 0
        gpu_match_token_num = 0
        swap_node_ids = []
        matche_nodes = []
        has_modified_gpu_lru_leaf_heap = False
        has_modified_cpu_lru_leaf_heap = False

        with self.cache_status_lock:
            while match_token_num < total_token_num:
                token_block = input_ids[match_token_num : match_token_num + block_size]
                token_num = len(token_block)
                if token_num != block_size:
                    break
                hash_value = self.cal_block_hash(token_block)
                if hash_value in current_match_node.children:
                    child = current_match_node.children[hash_value]
                    matche_nodes.append(child)
                    match_node_ids.append(child.node_id)
                    if child in self.gpu_lru_leaf_set:
                        self.gpu_lru_leaf_set.remove(child)
                        self.gpu_lru_leaf_heap.remove(child)
                        has_modified_gpu_lru_leaf_heap = True
                    elif child in self.cpu_lru_leaf_set:
                        self.cpu_lru_leaf_set.remove(child)
                        self.cpu_lru_leaf_heap.remove(child)
                        has_modified_cpu_lru_leaf_heap = True
                    if child.has_in_gpu:
                        match_gpu_block_ids.append(child.block_id)
                        gpu_match_token_num += block_size
                    else:
                        if child.cache_status == CacheStatus.SWAP2CPU:
                            logger.info(
                                f"match_block: req_id {req_id} matched node"
                                + f" {child.node_id} which is being SWAP2CPU"
                            )
                            child.cache_status = CacheStatus.GPU
                            match_gpu_block_ids.append(child.block_id)
                            gpu_match_token_num += block_size
                        elif child.cache_status == CacheStatus.CPU:
                            child.cache_status = CacheStatus.SWAP2GPU
                            match_cpu_block_ids.append(child.block_id)
                            cpu_match_token_num += block_size
                            swap_node_ids.append(child.node_id)
                    match_token_num = match_token_num + block_size
                    current_match_node = child
                else:
                    break

        if has_modified_gpu_lru_leaf_heap:
            heapq.heapify(self.gpu_lru_leaf_heap)
        if has_modified_cpu_lru_leaf_heap:
            heapq.heapify(self.cpu_lru_leaf_heap)

        logger.info(f"match_block: req_id {req_id} matched nodes: {match_node_ids}")
        return (
            match_gpu_block_ids,
            match_cpu_block_ids,
            swap_node_ids,
            current_match_node,
            gpu_match_token_num,
            cpu_match_token_num,
        )

    def _update_matched_node_info(self, req_id, last_node, current_time):
        """
        Update the shared count and last used time of the matched nodes
        """
        node = last_node
        while node != self.radix_tree_root:
            node.increment_shared_count()
            node.last_used_time = current_time
            node.req_id_set.add(req_id)
            node = node.parent

    def build_path(
        self,
        req_id,
        current_time,
        input_ids,
        left_input_ids,
        gpu_block_ids,
        block_size,
        last_node,
        reverved_dec_block_num,
    ):
        """
        Build path for blocks beyond the common prefix
            Parameters:
            - req_id: Request ID of the task
            - left_input_ids: Remaining input tokens not found in the prefix tree
            - gpu_block_ids: List of available GPU block IDs for new node allocation
            - block_size: Token capacity per block
            - last_node: Last successfully matched node
            - reserved_dec_block_num: Number of blocks reserved for decoding

            Returns:
            - leaf_node: The constructed leaf node
        """
        gpu_block_ids = gpu_block_ids.copy()
        node = last_node
        reverved_dec_block_ids = []
        input_hash_value = self.cal_block_hash(input_ids)

        token_num = len(left_input_ids)
        if token_num == 0:
            for i in range(reverved_dec_block_num):
                reverved_dec_block_ids.append(gpu_block_ids.pop(0))
            last_node.reverved_dec_block_ids.extend(reverved_dec_block_ids)
            return last_node
        node = last_node
        unique_node_ids = []
        new_last_node = last_node
        has_unfilled_block = False

        for i in range(0, token_num, block_size):
            current_block = left_input_ids[i : i + block_size]
            current_block_size = len(current_block)  # 最后一个block可能没填满
            if current_block_size != block_size:
                has_unfilled_block = True
            else:
                hash_value = self.cal_block_hash(current_block)
                allocated_block_id = gpu_block_ids.pop(0)
                node_id = self.node_id_pool.pop()
                unique_node_ids.append(node_id)
                new_last_node = BlockNode(
                    node_id,
                    input_ids,
                    input_hash_value,
                    node.depth + 1,
                    allocated_block_id,
                    current_block_size,
                    hash_value,
                    current_time,
                    parent=node,
                    shared_count=1,
                    reverved_dec_block_ids=[],
                )
                new_last_node.req_id_set.add(req_id)
                self.node_map[node_id] = new_last_node
                node.children[hash_value] = new_last_node
                node = new_last_node
        if has_unfilled_block is True:
            reverved_dec_block_ids.append(gpu_block_ids.pop(0))

        for i in range(reverved_dec_block_num):
            reverved_dec_block_ids.append(gpu_block_ids.pop(0))
        if new_last_node == self.radix_tree_root:
            self.unfilled_req_block_map[req_id] = reverved_dec_block_ids
        else:
            new_last_node.reverved_dec_block_ids.extend(reverved_dec_block_ids)
        logger.info(f"build_path: allocate unique node ids {unique_node_ids} for req_id {req_id}")
        return new_last_node

    def _handle_swap_result(self, swap_node_id, task_gpu_block_id, task_cpu_block_id, event_type):
        """
        handle swap resuha
        """
        if swap_node_id is None:
            return
        with self.cache_status_lock:
            if event_type.value == CacheStatus.SWAP2CPU.value:
                gpu_block_id = task_gpu_block_id
                cpu_block_id = task_cpu_block_id
                node = self.node_map[swap_node_id]
                if node.cache_status.value == CacheStatus.GPU.value:

                    logger.info(
                        f"recv_data_transfer_result: node {node.node_id} "
                        + f"has been reused when SWAP2CPU, recycle cpu block id {cpu_block_id}"
                    )
                    self.recycle_cpu_blocks(cpu_block_id)
                else:
                    node.cache_status = CacheStatus.CPU
                    node.block_id = cpu_block_id
                    if (
                        node != self.radix_tree_root
                        and node.shared_count == 0
                        and node.is_cpu_leaf_node
                        and node.cache_status == CacheStatus.CPU
                    ):
                        if node not in self.cpu_lru_leaf_set:
                            heapq.heappush(self.cpu_lru_leaf_heap, node)
                            self.cpu_lru_leaf_set.add(node)

                    self.recycle_gpu_blocks(gpu_block_id)
                    logger.info(f"recv_data_transfer_result: after SWAP2CPU, node {node}")

            elif event_type.value == CacheStatus.SWAP2GPU.value:
                gpu_block_id = task_gpu_block_id
                cpu_block_id = task_cpu_block_id

                node = self.node_map[swap_node_id]
                node.cache_status = CacheStatus.GPU
                node.block_id = gpu_block_id

                self.recycle_cpu_blocks(cpu_block_id)
                logger.info(f"recv_data_transfer_result: after SWAP2GPU, node {node}")
            else:
                logger.warning(
                    f"recv_data_transfer_result: Get unexpected event type {event_type}"
                    + ", only SWAP2CPU and SWAP2GPU supported"
                )

    def recv_data_transfer_result(self):
        """
        recv data transfer result
        """
        while True:

            try:
                data = self.cache_task_queue.get_transfer_done_signal()
                if data is None:
                    time.sleep(0.001)
                    continue
                (
                    swap_node_ids,
                    task_gpu_block_id,
                    task_cpu_block_id,
                    event_type,
                    transfer_task_id,
                ) = data
                length = len(task_gpu_block_id)
                for i in range(length):
                    self._handle_swap_result(
                        swap_node_ids[i],
                        task_gpu_block_id[i],
                        task_cpu_block_id[i],
                        event_type,
                    )
                if transfer_task_id in self.task_swapping_event:
                    self.task_swapping_event[transfer_task_id].set()
                logger.info(
                    f"recv_data_transfer_result: transfer_task_id {transfer_task_id}: "
                    + f"task_node_ids {swap_node_ids} task_gpu_block_id {task_gpu_block_id} "
                    + f"task_cpu_block_id {task_cpu_block_id} event_type {event_type} done"
                )
            except Exception as e:
                logger.warning(f"recv_data_transfer_result: error: {e}, {str(traceback.format_exc())}")
                raise e
