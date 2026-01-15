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
from fastdeploy.cache_manager.ops import get_all_visible_devices
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator import EngineCacheQueue, IPCSignal, PrefixTreeStatus
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import get_hash_str, get_logger

logger = get_logger("prefix_cache_manager", "cache_manager.log")


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
        self.config = config
        self.tensor_parallel_size = tensor_parallel_size
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

        self.key_cache_shape = []
        self.val_cache_shape = []
        self.node_id_pool = list(range(self.num_gpu_blocks + self.num_cpu_blocks))

        self.radix_tree_root = BlockNode(-1, [], 0, 0, -1, 0, None, None, None)

        # prams for cache storage
        self.kvcache_storage_backend = self.cache_config.kvcache_storage_backend
        self.write_policy = self.cache_config.write_policy
        self.task_write_back_event = {}
        self.task_prefetch_event = {}
        self.storage_prefetch_block_ids = {}

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
        self.req_to_radix_tree_info = {}  # {request_id: (last_match_node, num_cached_tokens_in_raidx_tree)}

        self.executor_pool = ThreadPoolExecutor(max_workers=1)
        self.free_gpu_executor_pool = ThreadPoolExecutor(max_workers=1)
        self.free_cpu_executor_pool = ThreadPoolExecutor(max_workers=1)
        self.gpu_free_task_future = None
        self.cache_status_lock = Lock()

        logger.info(
            f"num_gpu_blocks_server_owned {self.num_gpu_blocks} num_cpu_blocks "
            + f"{self.num_cpu_blocks}, bytes_per_layer_per_block {self.cache_config.bytes_per_layer_per_block}"
        )

        main_process_metrics.max_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.available_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.free_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.available_gpu_resource.set(1.0)

    def _get_kv_cache_shape(self, max_block_num):
        from fastdeploy.model_executor.layers.attention import get_attention_backend

        attn_cls = get_attention_backend()
        num_heads = self.config.model_config.num_attention_heads // self.config.parallel_config.tensor_parallel_size
        kv_num_heads = max(
            1,
            int(self.config.model_config.num_key_value_heads) // self.config.parallel_config.tensor_parallel_size,
        )
        head_dim = self.config.model_config.head_dim

        kv_cache_quant_type = None
        if (
            self.config.quant_config
            and hasattr(self.config.quant_config, "kv_cache_quant_type")
            and self.config.quant_config.kv_cache_quant_type is not None
        ):
            kv_cache_quant_type = self.config.quant_config.kv_cache_quant_type

        # Initialize AttentionBackend buffers
        encoder_block_shape_q = 64
        decoder_block_shape_q = 16
        key_cache_shape, value_cache_shape = attn_cls(
            self.config,
            kv_num_heads=kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
        ).get_kv_cache_shape(max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type)
        logger.info(f"key_cache_shape {key_cache_shape} value_cache_shape {value_cache_shape}")
        return key_cache_shape, value_cache_shape

    @property
    def available_gpu_resource(self):
        return len(self.gpu_free_block_list) / self.num_gpu_blocks if self.num_gpu_blocks > 0 else 0.0

    def launch_cache_manager(
        self,
        cache_config,
        tensor_parallel_size,
        device_ids,
        pod_ip,
        engine_worker_queue_port,
        ipc_suffix,
        create_cache_tensor,
    ):
        """
        launch_cache_manager function used to initialize the cache manager.
        """
        broadcast_cache_task_flag_array = np.zeros([1], dtype=np.int32)

        self.shm_cache_task_flag_broadcast = IPCSignal(
            name="cache_task_broadcast_signal",
            array=broadcast_cache_task_flag_array,
            dtype=np.int32,
            suffix=engine_worker_queue_port,
            create=True,
        )

        self.cache_task_queue = EngineCacheQueue(
            address=(pod_ip, cache_config.local_cache_queue_port),
            authkey=b"cache_queue_service",
            is_server=False,
            num_client=tensor_parallel_size,
            client_id=0,
            local_data_parallel_id=self.local_data_parallel_id,
        )

        current_dir_path = os.path.split(os.path.abspath(__file__))[0]
        filename = "cache_transfer_manager.py"
        py_path = os.path.join(current_dir_path, filename)

        cache_messager_processes = []
        key_cache_shape, val_cache_shape = self._get_kv_cache_shape(cache_config.total_block_num)
        key_cache_shape = ",".join([str(i) for i in key_cache_shape])
        val_cache_shape = ",".join([str(i) for i in val_cache_shape])
        logger.info(f"key_cache_shape {key_cache_shape} value_cache_shape {val_cache_shape}")
        if self.enable_splitwise:
            cache_messager_processes = self.launch_cache_messager(
                cache_config,
                tensor_parallel_size,
                device_ids,
                key_cache_shape,
                val_cache_shape,
                pod_ip,
                engine_worker_queue_port,
                ipc_suffix,
            )
            if cache_messager_processes is None:
                raise RuntimeError("Launch cache messager failed")
                return []

        cache_ready_signal_data = np.zeros(shape=[tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=engine_worker_queue_port,
            create=False,
        )
        swap_space_ready_data = np.zeros(shape=[tensor_parallel_size], dtype=np.int32)
        self.swap_space_ready_signal = IPCSignal(
            name="swap_space_ready_signal",
            array=swap_space_ready_data,
            dtype=np.int32,
            suffix=engine_worker_queue_port,
            create=False,
        )
        prefix_tree_status = np.zeros([1], dtype=np.int32)
        self.prefix_tree_status_signal = IPCSignal(
            name="prefix_tree_status",
            array=prefix_tree_status,
            dtype=np.int32,
            suffix=engine_worker_queue_port,
            create=False,
        )

        # Run command to launch cache transfer managers
        log_dir = envs.FD_LOG_DIR
        cache_manager_processes = []
        visible_devices = get_all_visible_devices()

        val_cache_arg_str = ""
        if val_cache_shape:
            if isinstance(val_cache_shape, list):
                val_shape_str = ",".join(map(str, val_cache_shape))
            else:
                val_shape_str = str(val_cache_shape)
            val_cache_arg_str = f" --value_cache_shape {val_shape_str}"
        if cache_config.kvcache_storage_backend:
            kvcache_storage_backend_str = cache_config.kvcache_storage_backend
        else:
            kvcache_storage_backend_str = "none"

        if self.cache_config.swap_space or self.cache_config.kvcache_storage_backend:
            for i in range(tensor_parallel_size):
                launch_cmd = (
                    "FLAGS_allocator_strategy=auto_growth "
                    + visible_devices
                    + " NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=0"
                    + f" FD_ENABLE_SWAP_SPACE_CLEARING={envs.FD_ENABLE_SWAP_SPACE_CLEARING}"
                    + f" {sys.executable} {py_path}"
                    + f" --device_id {int(device_ids[i])}"
                    + f" --rank {i}"
                    + f" --splitwise_role {self.splitwise_role}"
                    + f" --num_layers {cache_config.model_cfg.num_hidden_layers}"
                    + f" --mp_num {tensor_parallel_size}"
                    + f" --cache_dtype {cache_config.cache_dtype}"
                    + f" --key_cache_shape {key_cache_shape}"
                    + val_cache_arg_str
                    + f" --cache_queue_port {cache_config.local_cache_queue_port}"
                    + f" --enable_splitwise {int(self.enable_splitwise)}"
                    + f" --pod_ip {pod_ip}"
                    + f" --engine_worker_queue_port {engine_worker_queue_port}"
                    + f" --num_cpu_blocks {cache_config.num_cpu_blocks}"
                    + f" --ipc_suffix {ipc_suffix}"
                    + f" --protocol {cache_config.cache_transfer_protocol}"
                    + f" --local_data_parallel_id {self.local_data_parallel_id}"
                    + f" --rdma_port {cache_config.local_rdma_comm_ports[i] if cache_config.local_rdma_comm_ports is not None else '0'}"
                    + f" --speculative_config '{self.speculative_config.to_json_string()}'"
                    + f" --default_dtype '{self.config.model_config.dtype}'"
                    + (" --create_cache_tensor" if create_cache_tensor else "")
                    + f" --kvcache_storage_backend {kvcache_storage_backend_str}"
                    + f" --write_policy {cache_config.write_policy}"
                    + f" --max_model_len {self.config.model_config.max_model_len}"
                    + f" >{log_dir}/launch_cache_transfer_manager_{int(device_ids[i])}.log 2>&1"
                )
                logger.info(f"Launch cache transfer manager, command:{launch_cmd}")
                cache_manager_processes.append(subprocess.Popen(launch_cmd, shell=True, preexec_fn=os.setsid))

        logger.info("PrefixCacheManager is waiting for kv cache to be initialized.")
        while np.sum(self.cache_ready_signal.value) != tensor_parallel_size:
            time.sleep(1)

        if self.num_cpu_blocks > 0:
            while np.sum(self.swap_space_ready_signal.value) != tensor_parallel_size:
                time.sleep(1)

        if cache_manager_processes:
            exit_code = cache_manager_processes[-1].poll()
            if exit_code is None:
                logger.info("Launch cache transfer manager successful")
            else:
                logger.info(
                    "Launch cache transfer manager failed, see launch_cache_transfer_manager.log for more information"
                )

        # Start additional threads
        if cache_config.kvcache_storage_backend or self.num_cpu_blocks > 0:
            logger.info("Enable hierarchical cache.")
            threading.Thread(target=self.recv_data_transfer_result).start()
        if cache_config.enable_prefix_caching:
            threading.Thread(target=self.clear_prefix_cache, daemon=True).start()

        all_cache_processes = cache_messager_processes + cache_manager_processes
        return all_cache_processes

    def launch_cache_messager(
        self,
        cache_config,
        tensor_parallel_size,
        device_ids,
        key_cache_shape,
        value_cache_shape,
        pod_ip,
        engine_worker_queue_port,
        ipc_suffix,
    ):
        """
        launch_cache_messager function used to initialize the cache messager.
        """
        current_dir_path = os.path.split(os.path.abspath(__file__))[0]
        filename = "cache_messager.py"

        cache_ready_signal_data = np.zeros(shape=[tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=ipc_suffix,
            create=False,
        )

        py_path = os.path.join(current_dir_path, filename)
        log_dir = envs.FD_LOG_DIR
        cache_messager_processes = []
        visible_devices = get_all_visible_devices()

        val_cache_arg_str = ""
        if value_cache_shape:
            if isinstance(value_cache_shape, list):
                val_shape_str = ",".join(map(str, value_cache_shape))
            else:
                val_shape_str = str(value_cache_shape)
            val_cache_arg_str = f" --value_cache_shape {val_shape_str}"

        for i in range(tensor_parallel_size):
            launch_cmd = (
                "FLAGS_allocator_strategy=auto_growth "
                + visible_devices
                + " NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=0"
                + f" {sys.executable} {py_path}"
                + f" --device_id {int(device_ids[i])}"
                + f" --rank {i}"
                + f" --splitwise_role {self.splitwise_role}"
                + f" --num_layers {cache_config.model_cfg.num_hidden_layers}"
                + f" --mp_num {tensor_parallel_size}"
                + f" --cache_dtype {cache_config.cache_dtype}"
                + f" --key_cache_shape {key_cache_shape}"
                + val_cache_arg_str
                + f" --pod_ip {pod_ip}"
                + f" --default_dtype '{self.config.model_config.dtype}'"
                + f" --cache_queue_port {cache_config.local_cache_queue_port}"
                + f" --engine_worker_queue_port {engine_worker_queue_port}"
                + f" --protocol {cache_config.cache_transfer_protocol}"
                + f" --local_data_parallel_id {self.local_data_parallel_id}"
                + f" --ipc_suffix {ipc_suffix}"
                + f" --rdma_port {cache_config.local_rdma_comm_ports[i] if cache_config.local_rdma_comm_ports is not None else '0'}"
                + f" --speculative_config '{self.speculative_config.to_json_string()}'"
                + f" >{log_dir}/launch_cache_messager_tprank{i}.log 2>&1"
            )
            logger.info(f"Launch cache messager, command:{launch_cmd}")
            cache_messager_processes.append(subprocess.Popen(launch_cmd, shell=True, preexec_fn=os.setsid))

        logger.info("Waiting for cache ready...")
        while np.sum(self.cache_ready_signal.value) != tensor_parallel_size:
            time.sleep(1)
        exit_code = cache_messager_processes[-1].poll()
        if exit_code is None:
            logger.info("Launch cache messager successful")
        else:
            logger.info("Launch cache messager failed, see launch_cache_messager.log for more information")
            cache_messager_processes = None
        return cache_messager_processes

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

        main_process_metrics.max_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.available_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.free_gpu_block_num.set(self.num_gpu_blocks)
        main_process_metrics.available_gpu_resource.set(1.0)

    def can_allocate_gpu_blocks(self, num_blocks: int):
        """
        Check if num_blocks gpu blocks can be allocated.
        """
        if len(self.gpu_free_block_list) < num_blocks:
            if self.cache_config.enable_prefix_caching:
                self.free_block_ids(num_blocks)
            if len(self.gpu_free_block_list) < num_blocks:
                return False
            else:
                return True
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
        main_process_metrics.free_gpu_block_num.set(len(self.gpu_free_block_list))
        main_process_metrics.available_gpu_resource.set(self.available_gpu_resource)
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
        main_process_metrics.free_gpu_block_num.set(len(self.gpu_free_block_list))
        main_process_metrics.available_gpu_resource.set(self.available_gpu_resource)

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
            (event_type, transfer_task_id, swap_node_ids, gpu_block_ids, cpu_block_ids)
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

    def update_cache_blocks(self, task, block_size, num_computed_tokens):
        """
        update cache blocks for a task.
        # TODO(chengyanfu): support async update

        Parameters:
        - task: Task
        - block_size: Size per block (in tokens)
        """
        try:
            req_id = task.request_id
            last_node, num_cached_tokens = self.req_to_radix_tree_info[req_id]
            can_cache_computed_tokens = num_computed_tokens - num_computed_tokens % block_size
            if req_id in self.leaf_req_map[last_node]:  # delete old leaf record, update later
                self.leaf_req_map[last_node].remove(req_id)
            logger.debug(
                f"update_cache_blocks: req_id {req_id}, num_cached_tokens {num_cached_tokens}, "
                f"can_cache_computed_tokens {can_cache_computed_tokens}"
            )

            with self.request_release_lock:
                leaf_node = self.mm_build_path(
                    request=task,
                    num_computed_tokens=num_computed_tokens,
                    block_size=block_size,
                    last_node=last_node,
                    num_cached_tokens=num_cached_tokens,
                )
                self.req_leaf_map[req_id] = leaf_node
                self.leaf_req_map[leaf_node].add(req_id)
                self.req_to_radix_tree_info[req_id] = [leaf_node, can_cache_computed_tokens]
                task.num_cached_blocks = can_cache_computed_tokens // block_size
        except Exception as e:
            logger.error(f"update_cache_blocks, error: {type(e)} {e}, {str(traceback.format_exc())}")
            raise e

    def is_chunked_mm_input(self, mm_inputs, matched_token_num):
        """
        check if mm_inputs is chunked
        """
        if mm_inputs is None or "mm_positions" not in mm_inputs or len(mm_inputs["mm_positions"]) == 0:
            return False, 0

        for idx in range(len(mm_inputs["mm_positions"])):
            position = mm_inputs["mm_positions"][idx]
            if position.offset < matched_token_num < position.offset + position.length:
                return True, idx
            elif matched_token_num < position.offset:
                break
        return False, 0

    def request_match_blocks(self, task: Request, block_size, *args):
        """
        Match and fetch cache for a task.
        This is a synchronous interface. If CPU-to-GPU data transfer occurs,
        it will block until synchronization completes.
        Callers requiring asynchronous behavior should invoke this via a thread pool.

        Note: This function may allocate GPU blocks for matched CPU Cache and Storage Cache

        Parameters:
        - task: Task dictionary
        - block_size: Size per block (in tokens)

        Returns:
        - common_block_ids: List of matched shared blocks
        - match_token_num: Number of matched tokens
        - metrics: Dictionary of metrics
        """
        with self.request_release_lock:
            try:
                metrics = {
                    "gpu_match_token_num": 0,
                    "cpu_match_token_num": 0,
                    "storage_match_token_num": 0,
                    "match_gpu_block_ids": [],
                    "gpu_recv_block_ids": [],
                    "match_storage_block_ids": [],
                    "cpu_cache_prepare_time": 0,
                    "storage_cache_prepare_time": 0,
                }
                self.metrics.req_count += 1
                if isinstance(task.prompt_token_ids, np.ndarray):
                    prompt_token_ids = task.prompt_token_ids.tolist()
                else:
                    prompt_token_ids = task.prompt_token_ids
                req_id = task.request_id
                logger.info(f"request_match_blocks: start to process req {req_id}")
                input_token_ids = prompt_token_ids + task.output_token_ids
                input_token_num = len(input_token_ids)
                common_block_ids = []
                # 1. match block
                (
                    match_gpu_block_ids,
                    match_cpu_block_ids,
                    swap_node_ids,
                    match_block_node,
                    gpu_match_token_num,
                    cpu_match_token_num,
                ) = self.mm_match_block(task, block_size)

                #  update matched node info
                self._update_matched_node_info(req_id, match_block_node, current_time=time.time())

                # 2. prepare cpu cache: allocate gpu cache for matched cpu blocks, wait for data transfer to complete
                gpu_recv_block_ids = []
                match_cpu_blocks_num = len(match_cpu_block_ids)
                if self.can_allocate_gpu_blocks(num_blocks=match_cpu_blocks_num):
                    if match_cpu_blocks_num > 0:
                        logger.debug(
                            f"request_match_blocks: req_id {req_id}, allocate {match_cpu_blocks_num} block to receive cpu cache"
                        )
                        gpu_recv_block_ids = self.allocate_gpu_blocks(match_cpu_blocks_num)
                        if len(gpu_recv_block_ids) > 0:
                            start_time = time.time()
                            self._prepare_cpu_cache(
                                req_id=req_id,
                                swap_node_ids=swap_node_ids,
                                gpu_recv_block_ids=gpu_recv_block_ids,
                                match_cpu_block_ids=match_cpu_block_ids,
                                cpu_recv_block_ids=[],
                            )
                            cost_time = time.time() - start_time
                            metrics["cpu_cache_prepare_time"] = cost_time
                else:
                    raise Exception(
                        "request_match_blocks: Not enough GPU memory to allocate cache for matched CPU Cache"
                    )

                # 3. match and prefetch cache from storage
                match_token_num = gpu_match_token_num + cpu_match_token_num
                no_match_token_num = input_token_num - match_token_num
                no_match_block_num = (no_match_token_num + block_size - 1) // block_size
                gpu_recv_storage_block_ids = []
                storage_match_token_num = 0
                match_storage_block_ids = []

                if self.kvcache_storage_backend and no_match_token_num >= block_size:
                    if not self.can_allocate_gpu_blocks(num_blocks=no_match_block_num):
                        raise Exception(
                            "request_match_blocks: Not enough GPU memory to allocate cache for matched Storage Cache"
                        )

                    logger.debug(
                        f"request_match_blocks: req_id {req_id}, allocate {no_match_block_num} block to receive storage cache"
                    )
                    gpu_recv_storage_block_ids = self.allocate_gpu_blocks(no_match_block_num)

                    prefix_block_key = [] if match_block_node.hash_value is None else [match_block_node.hash_value]
                    cur_token_idx = match_token_num
                    no_match_block_keys = []
                    while cur_token_idx <= input_token_num - block_size:
                        cur_block_token_ids = input_token_ids[cur_token_idx : cur_token_idx + block_size]
                        cur_block_key = get_hash_str(cur_block_token_ids, prefix_block_key)
                        no_match_block_keys.append(cur_block_key)
                        cur_token_idx += block_size
                        prefix_block_key = [cur_block_key]

                    logger.info(
                        f"start prefetch cache from storage, req_id: {req_id}, block num: {len(no_match_block_keys)}"
                    )
                    start_time = time.time()
                    storage_matched_block_ids = self.issue_prefetch_storage_task(
                        req_id, no_match_block_keys, gpu_recv_storage_block_ids
                    )
                    storage_matched_block_num = len(storage_matched_block_ids)
                    storage_match_token_num = storage_matched_block_num * block_size
                    cost_time = time.time() - start_time
                    metrics["storage_cache_prepare_time"] = cost_time
                    logger.info(
                        f"finish prefetch cache from storage, req_id: {req_id}, "
                        f"matched block num: {storage_matched_block_num}, cost_time:{cost_time:.6f}s"
                    )

                    match_storage_block_ids = gpu_recv_storage_block_ids[:storage_matched_block_num]
                    self.recycle_gpu_blocks(gpu_recv_storage_block_ids[storage_matched_block_num:])

                # 4. update metrics
                match_token_num = gpu_match_token_num + cpu_match_token_num + storage_match_token_num
                common_block_ids = match_gpu_block_ids + gpu_recv_block_ids + match_storage_block_ids
                if match_token_num > 0:
                    self.metrics.hit_req_count += 1
                self.metrics.calculate_hit_metrics(
                    req_id,
                    cpu_match_token_num,
                    gpu_match_token_num,
                    storage_match_token_num,
                    input_token_num,
                )
                metrics["gpu_match_token_num"] = gpu_match_token_num
                metrics["cpu_match_token_num"] = cpu_match_token_num
                metrics["storage_match_token_num"] = storage_match_token_num
                metrics["match_gpu_block_ids"] = match_gpu_block_ids
                metrics["gpu_recv_block_ids"] = gpu_recv_block_ids
                metrics["match_storage_block_ids"] = match_storage_block_ids
                self.metrics._update_history_hit_metrics()
                if self.metrics.req_count % 10000 == 0:
                    self.metrics.reset_metrics()
                logger.debug(f"request_match_blocks: req_id {req_id}, matched_block_ids_num {len(common_block_ids)}")
                logger.debug(f"request_match_blocks: req_id {req_id}, matched_block_ids {common_block_ids}")

                # set leaf node temporarily, then update it in update_cache_blocks
                self.req_leaf_map[req_id] = match_block_node
                self.leaf_req_map[match_block_node].add(req_id)
                # record request cache info in radix tree, note that the block ids for receiving storage cache
                # are recorded into radix tree in update_cache_blocks
                self.req_to_radix_tree_info[req_id] = [match_block_node, gpu_match_token_num + cpu_match_token_num]
                task.num_cached_blocks = len(common_block_ids)
                return common_block_ids, match_token_num, metrics
            except Exception as e:
                logger.error(f"request_match_blocks: request_block_ids: error: {type(e)} {e}")
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
                (gpu_recv_block_ids, gpu_extra_block_ids) = self._prepare_cache(
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
                    0,
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

    def free_block_ids(self, need_block_num):
        self.free_block_ids_async(need_block_num)
        while (self.gpu_free_task_future is not None) and (not self.gpu_free_task_future.done()):
            time.sleep(0.001)

    def release_block_ids(self, task):
        """
        release block ids
        """
        with self.request_release_lock:
            try:
                req_id = task.request_id
                keys = []
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
                    keys.append(node.hash_value)
                    node = node.parent

                if req_id in self.req_to_radix_tree_info:
                    del self.req_to_radix_tree_info[req_id]

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

    def write_cache_to_storage(self, request: Request):
        """
        For finished request, write cache to storage.
        NOTE: this function does not modify the global params
        """
        if self.kvcache_storage_backend is None:
            return

        req_id = request.request_id
        keys = []
        node = self.req_leaf_map[req_id]
        while node != self.radix_tree_root:
            keys.append(node.hash_value)
            node = node.parent
        keys = list(reversed(keys))
        if not keys:
            return

        gpu_block_ids = request.block_tables[: len(keys)]
        logger.info(f"start write cache back to storage, req_id: {req_id}, block num: {len(keys)}")
        tic = time.time()
        self.issue_write_back_storage_task(req_id=req_id, hash_keys=keys, gpu_block_ids=gpu_block_ids, is_sync=True)
        cost_time = time.time() - tic
        logger.info(f"finish write cache back to storage, req_id: {req_id}, cost_time: {cost_time:.6f}s")

    def issue_write_back_storage_task(self, req_id, hash_keys, gpu_block_ids, is_sync=True, timeout=0.5):
        if self.kvcache_storage_backend is None:
            return

        if len(hash_keys) != len(gpu_block_ids):
            err_msg = f"write_back_storage error: hash_keys({len(hash_keys)}) != gpu_block_ids({len(gpu_block_ids)})"
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.task_write_back_event[req_id] = Event()
        self.cache_task_queue.put_transfer_task((CacheStatus.GPU2STORAGE, req_id, hash_keys, gpu_block_ids, timeout))
        if is_sync:
            self.wait_write_storage_task(req_id)

    def wait_write_storage_task(self, req_id):
        """
        Sync write back task
        """
        if req_id in self.task_write_back_event:
            self.task_write_back_event[req_id].wait()
            del self.task_write_back_event[req_id]

    def issue_prefetch_storage_task(self, req_id, hash_keys, gpu_block_ids, is_sync=True, timeout=0.5):
        """
        Prefetch cache from storage task
        """
        storage_block_ids = []
        self.task_prefetch_event[req_id] = Event()
        # issue task to cache_transfer_manager
        self.cache_task_queue.put_transfer_task((CacheStatus.STORAGE2GPU, req_id, hash_keys, gpu_block_ids, timeout))
        if is_sync:
            storage_block_ids = self.wait_prefetch_storage_task(req_id)
        return storage_block_ids

    def wait_prefetch_storage_task(self, req_id):
        """
        Wait for prefetch cache from storage task to finish
        """
        if req_id not in self.task_prefetch_event:
            return None

        self.task_prefetch_event[req_id].wait()
        storage_block_ids = self.storage_prefetch_block_ids[req_id]
        del self.task_prefetch_event[req_id]
        del self.storage_prefetch_block_ids[req_id]
        return storage_block_ids

    def free_nodes_directly(self, node):
        with self.request_release_lock:
            try:
                total_gpu_free_count = 0
                while True:
                    if node in self.gpu_lru_leaf_heap:
                        self.gpu_lru_leaf_heap.remove(node)
                        self.gpu_lru_leaf_set.remove(node)
                    if node.shared_count == 0 and node.is_gpu_leaf_node:  # 直接回收
                        self._handle_free_gpu_node_without_cpu(node)
                        logger.info(f"free_nodes_directly: node {node}")
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
                            break
                    else:
                        break
            except Exception as e:
                logger.error(f"free_nodes_directly: error: {type(e)} {e}")
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
                        logger.info("free_block_ids_async: no more gpu leaf node available.")
                        break
                    if total_gpu_free_count >= need_block_num:
                        break
                    node = heapq.heappop(self.gpu_lru_leaf_heap)
                    self.gpu_lru_leaf_set.remove(node)
                    if self.cache_config.num_cpu_blocks < need_block_num:
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
                logger.info(
                    f"free_block_ids_async: need_block_num {need_block_num}, free_block_num {total_gpu_free_count}."
                )

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

    def get_block_hash_extra_keys(self, request, start_idx, end_idx, mm_idx):
        """
        Retrieves additional hash keys for block identification.

        Args:
            request: The input request object containing the data to be processed.
            start_idx (int): The starting index of the block segment to hash.
            end_idx (int): The ending index of the block segment to hash.
            mm_idx: The multimodal index identifier for specialized content handling.

        Returns:
            mm_idx: next multimodal index
            hash_keys: A list of additional hash keys
        """
        hash_keys = []
        mm_inputs = request.multimodal_inputs
        if (
            mm_inputs is None
            or "mm_positions" not in mm_inputs
            or "mm_hashes" not in mm_inputs
            or len(mm_inputs["mm_positions"]) == 0
        ):
            return mm_idx, hash_keys

        assert start_idx < end_idx, f"start_idx {start_idx} >= end_idx {end_idx}"
        assert (
            start_idx >= 0 and start_idx < request.num_total_tokens
        ), f"start_idx {start_idx} out of range {request.num_total_tokens}"
        assert (
            end_idx >= 0 and end_idx <= request.num_total_tokens
        ), f"end_idx {end_idx} out of range {request.num_total_tokens}"
        assert len(mm_inputs["mm_positions"]) == len(
            mm_inputs["mm_hashes"]
        ), f"mm_positions {len(mm_inputs['mm_positions'])} != mm_hashes {len(mm_inputs['mm_hashes'])}"
        assert mm_idx >= 0 and mm_idx < len(
            mm_inputs["mm_hashes"]
        ), f"mm_idx {mm_idx} out of range {len(mm_inputs['mm_hashes'])}"

        if mm_inputs["mm_positions"][-1].offset + mm_inputs["mm_positions"][-1].length < start_idx:
            # non images in current block
            return mm_idx, hash_keys

        for img_idx in range(mm_idx, len(mm_inputs["mm_positions"])):
            image_offset = mm_inputs["mm_positions"][img_idx].offset
            image_length = mm_inputs["mm_positions"][img_idx].length

            if image_offset + image_length < start_idx:
                # image before block
                continue
            elif image_offset >= end_idx:
                # image after block
                return img_idx, hash_keys
            elif image_offset + image_length > end_idx:
                hash_keys.append(mm_inputs["mm_hashes"][img_idx])
                return img_idx, hash_keys
            else:
                hash_keys.append(mm_inputs["mm_hashes"][img_idx])
        return len(mm_inputs["mm_positions"]) - 1, hash_keys

    def mm_match_block(self, request, block_size):
        """
        Match and retrieve cached blocks for multimodal requests using a radix tree structure.

        Args:
            request: The multimodal request object containing prompt and output token IDs.
            block_size (int): The size of each token block for matching and processing.

        Returns:
            tuple: A tuple containing:
                - match_gpu_block_ids (list): List of block IDs matched in GPU cache
                - match_cpu_block_ids (list): List of block IDs matched in CPU cache
                - swap_node_ids (list): List of node IDs scheduled for GPU-CPU swapping
                - current_match_node: The last matched node in the radix tree traversal
                - gpu_match_token_num (int): Total number of tokens matched in GPU cache
                - cpu_match_token_num (int): Total number of tokens matched in CPU cache
        """
        if isinstance(request.prompt_token_ids, np.ndarray):
            prompt_token_ids = request.prompt_token_ids.tolist()
        else:
            prompt_token_ids = request.prompt_token_ids
        input_ids = prompt_token_ids + request.output_token_ids
        total_token_num = len(input_ids)
        current_match_node = self.radix_tree_root  # 从根节点开始搜
        match_gpu_block_ids = []
        match_cpu_block_ids = []
        match_node_ids = []
        mm_idx = 0
        match_token_num = 0
        cpu_match_token_num = 0
        gpu_match_token_num = 0
        swap_node_ids = []
        matche_nodes = []
        has_modified_gpu_lru_leaf_heap = False
        has_modified_cpu_lru_leaf_heap = False
        prefix_block_key = []

        with self.cache_status_lock:
            while match_token_num < total_token_num:
                token_block = input_ids[match_token_num : match_token_num + block_size]
                token_num = len(token_block)
                if token_num != block_size:
                    break
                mm_idx, extra_keys = self.get_block_hash_extra_keys(
                    request=request,
                    start_idx=match_token_num,
                    end_idx=match_token_num + block_size,
                    mm_idx=mm_idx,
                )
                prefix_block_key.extend(extra_keys)
                hash_value = get_hash_str(token_block, prefix_block_key)
                prefix_block_key = [hash_value]

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
                                f"match_block: req_id {request.request_id} matched node"
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

        logger.info(f"match_block: req_id {request.request_id} matched nodes: {match_node_ids}")
        return (
            match_gpu_block_ids,
            match_cpu_block_ids,
            swap_node_ids,
            current_match_node,
            gpu_match_token_num,
            cpu_match_token_num,
        )

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
        prefix_block_key = []

        with self.cache_status_lock:
            while match_token_num < total_token_num:
                token_block = input_ids[match_token_num : match_token_num + block_size]
                token_num = len(token_block)
                if token_num != block_size:
                    break
                hash_value = get_hash_str(token_block, prefix_block_key)
                prefix_block_key = [hash_value]
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
                    #  record request cache info
                    self.req_to_radix_tree_info[req_id] = [child, match_token_num]
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

    def mm_build_path(self, request, num_computed_tokens, block_size, last_node, num_cached_tokens):
        """
        Constructs a caching path in radix tree for multimodal requests by processing computed tokens.

        Args:
            request: The inference request object containing:
                - prompt_token_ids: Original input tokens (List[int] or np.ndarray)
                - output_token_ids: Generated tokens (List[int])
                - mm_positions: Optional image positions for multimodal content
            num_computed_tokens: Total tokens processed so far (cached + newly computed)
            block_size: Fixed size of token blocks (must match cache configuration)
            last_node: The deepest existing BlockNode in the radix tree for this request
            num_cached_tokens: Number of tokens already cached

        Returns:
            BlockNode: The new deepest node in the constructed path
        """
        if isinstance(request.prompt_token_ids, np.ndarray):
            prompt_token_ids = request.prompt_token_ids.tolist()
        else:
            prompt_token_ids = request.prompt_token_ids
        input_ids = prompt_token_ids + request.output_token_ids
        can_cache_computed_tokens = num_computed_tokens - num_computed_tokens % block_size
        if num_cached_tokens == can_cache_computed_tokens:
            return last_node

        mm_idx = 0
        node = last_node
        unique_node_ids = []
        new_last_node = last_node
        has_unfilled_block = False
        current_time = time.time()

        input_hash_value = get_hash_str(input_ids)
        gpu_block_ids = request.block_tables[num_cached_tokens // block_size :].copy()
        prefix_block_key = [] if last_node.hash_value is None else [last_node.hash_value]

        for i in range(num_cached_tokens, can_cache_computed_tokens, block_size):
            current_block = input_ids[i : i + block_size]
            current_block_size = len(current_block)  # 最后一个block可能没填满
            if current_block_size != block_size:
                has_unfilled_block = True
            else:
                mm_idx, extra_keys = self.get_block_hash_extra_keys(
                    request=request,
                    start_idx=i,
                    end_idx=i + block_size,
                    mm_idx=mm_idx,
                )
                prefix_block_key.extend(extra_keys)
                hash_value = get_hash_str(current_block, prefix_block_key)
                prefix_block_key = [hash_value]
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
                new_last_node.req_id_set.add(request.request_id)
                self.node_map[node_id] = new_last_node
                node.children[hash_value] = new_last_node
                node = new_last_node

        reverved_dec_block_ids = []
        if has_unfilled_block is True:
            reverved_dec_block_ids.append(gpu_block_ids.pop(0))

        if new_last_node == self.radix_tree_root:
            self.unfilled_req_block_map[request.request_id] = reverved_dec_block_ids
        else:
            new_last_node.reverved_dec_block_ids.extend(reverved_dec_block_ids)
        logger.info(f"build_path: allocate unique node ids {unique_node_ids} for req_id {request.request_id}")
        return new_last_node

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
        input_hash_value = get_hash_str(input_ids)

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
        prefix_block_key = [] if last_node.hash_value is None else [last_node.hash_value]

        for i in range(0, token_num, block_size):
            current_block = left_input_ids[i : i + block_size]
            current_block_size = len(current_block)  # 最后一个block可能没填满
            if current_block_size != block_size:
                has_unfilled_block = True
            else:
                hash_value = get_hash_str(current_block, prefix_block_key)
                prefix_block_key = [hash_value]
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
                event_type = data[0]

                if event_type.value == CacheStatus.STORAGE2GPU.value:
                    logger.info(f"recv_data_transfer_result: {data}")
                    task_id, hash_keys, block_ids = data[1:]
                    if task_id not in self.storage_prefetch_block_ids:
                        self.storage_prefetch_block_ids[task_id] = []
                    saved_block_ids = self.storage_prefetch_block_ids[task_id]
                    saved_block_ids.append(block_ids)
                    if len(saved_block_ids) == self.tensor_parallel_size:
                        self.storage_prefetch_block_ids[task_id] = min(saved_block_ids, key=len)
                        if task_id in self.task_prefetch_event:
                            self.task_prefetch_event[task_id].set()
                elif event_type.value == CacheStatus.GPU2STORAGE.value:
                    logger.info(f"recv_data_transfer_result: {data}")
                    task_id, hash_keys, block_ids = data[1:]
                    if task_id in self.task_write_back_event:
                        self.task_write_back_event[task_id].set()
                else:
                    (
                        event_type,
                        transfer_task_id,
                        swap_node_ids,
                        task_gpu_block_id,
                        task_cpu_block_id,
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

    def reset(self):
        """
        Reset the RadixTree.
        """

        if len(self.node_map) == 0:
            return

        logger.info("Resetting the RadixTree!")

        # wait for swap tasks to finish
        if self.gpu_free_task_future is not None:
            self.gpu_free_task_future.result()
            self.gpu_free_task_future = None
        for event in list(self.task_swapping_event.values()):
            event.wait()
        self.task_swapping_event.clear()

        # clear node map
        self.node_map.clear()
        self.req_leaf_map.clear()
        self.leaf_req_map.clear()
        self.unfilled_req_block_map.clear()
        self.req_to_radix_tree_info.clear()

        # reset gpu cache data structure
        self.gpu_lru_leaf_heap.clear()
        self.gpu_lru_leaf_set.clear()

        # reset cpu cache data structure
        self.cpu_lru_leaf_heap.clear()
        self.cpu_lru_leaf_set.clear()

        # reset gpu/cpu free block list
        self.gpu_free_block_list = list(range(self.num_gpu_blocks - 1, -1, -1))
        if self.num_cpu_blocks > 0:
            self.cpu_free_block_list = list(range(self.num_cpu_blocks - 1, -1, -1))
        else:
            self.cpu_free_block_list = []
        heapq.heapify(self.gpu_free_block_list)
        heapq.heapify(self.cpu_free_block_list)

        # reset node/tree
        self.node_id_pool = list(range(self.num_gpu_blocks + self.num_cpu_blocks))
        self.radix_tree_root = BlockNode(-1, [], 0, 0, -1, 0, None, None, None)

        # reset metrics
        self.metrics.reset_metrics()
        main_process_metrics.free_gpu_block_num.set(len(self.gpu_free_block_list))
        main_process_metrics.available_gpu_block_num.set(len(self.gpu_free_block_list))
        main_process_metrics.available_gpu_resource.set(self.available_gpu_resource)

    def clear_prefix_cache(self):
        """
        If the model weights status is updating or clearing, reset prefix cache tree
        """
        logger.info("Start a thread to clear prefix cache when model weights are cleared.")
        prefix_tree_status_signal = self.prefix_tree_status_signal
        while True:
            if prefix_tree_status_signal.value[0] == PrefixTreeStatus.CLEARING:
                self.reset()
                prefix_tree_status_signal.value[0] = PrefixTreeStatus.CLEARED
                logger.info("Prefix cache tree is cleared.")
            if prefix_tree_status_signal.value[0] == PrefixTreeStatus.UPDATING:
                prefix_tree_status_signal.value[0] = PrefixTreeStatus.NORMAL
                logger.info("Prefix cache tree is updated.")
            time.sleep(0.01)
