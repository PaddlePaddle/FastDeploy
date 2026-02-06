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
import os
import queue
import threading
import time
import traceback
from typing import List

import numpy as np
import paddle
import yaml

from fastdeploy import envs
from fastdeploy.cache_manager.cache_data import CacheStatus
from fastdeploy.cache_manager.cache_tasks import ReadStorageTask, WriteStorageTask
from fastdeploy.cache_manager.ops import (
    cuda_host_alloc,
    cuda_host_free,
    memory_allocated,
    set_data_ipc,
    set_device,
    share_external_data_,
    swap_cache_all_layers,
    swap_cache_layout,
    unset_data_ipc,
)
from fastdeploy.cache_manager.transfer_factory import (
    AttentionStore,
    FileStore,
    MooncakeStore,
)
from fastdeploy.config import SpeculativeConfig
from fastdeploy.inter_communicator import EngineCacheQueue, IPCSignal, KVCacheStatus
from fastdeploy.platforms import current_platform
from fastdeploy.utils import console_logger, get_logger


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
    parser.add_argument("--rank", type=int, default=0, help="local tp rank")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--max_model_len", type=int, default=32768, help="max model length")
    parser.add_argument("--num_layers", type=int, default=1, help="model num layers")
    parser.add_argument("--mp_num", type=int, default=1, help="number of model parallel")
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="bfloat16",
        choices=["uint8", "bfloat16", "block_wise_fp8"],
        help="cache dtype",
    )
    parser.add_argument(
        "--default_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "uint8"],
        help="paddle default dtype, swap_cache_batch only support float16、bfloat16 and uint8 now",
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
    parser.add_argument("--ipc_suffix", type=str, default=None, help="engine pid")
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
    parser.add_argument(
        "--kvcache_storage_backend",
        type=str,
        default=None,
        choices=["mooncake", "attention_store", "file"],
        help="The storage backend for kvcache storage. If not set, storage backend is disabled.",
    )
    parser.add_argument(
        "--write_policy",
        type=str,
        choices=["write_through"],
        default="write_through",
        help="KVCache write policy",
    )
    parser.add_argument("--model_path", type=str, help="The path of model")

    args = parser.parse_args()
    return args


def get_key_prefix_from_version(version_file_path):
    # the format of version string is RL-STEP{xx}-{timestamp}-{uuid4}
    with open(version_file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        version = data["version"]
        parts = version.split("-", 2)
        key_prefix = "-".join(parts[:2])
        return key_prefix


class CacheTransferManager:
    """
    管理CPU和GPU之间缓存的交换传输
    """

    def __init__(self, args):
        """
        初始化CacheTransferManager
        """
        self.gpu_cache_kvs = {}
        self.cpu_cache_kvs = {}
        self.gpu_cache_k_tensors = []
        self.gpu_cache_v_tensors = []
        self.gpu_cache_scales_k_tensors = []
        self.gpu_cache_scales_v_tensors = []
        self.speculative_config = SpeculativeConfig(args.speculative_config)

        # parse kv cache shape
        self.key_cache_shape = [int(i) for i in args.key_cache_shape.split(",")]
        self.value_cache_shape = []
        if args.value_cache_shape:
            self.value_cache_shape = [int(i) for i in args.value_cache_shape.split(",")]

        # extract kv cache shape into fields
        self.num_gpu_blocks = self.key_cache_shape[0]
        self.head_num = self.key_cache_shape[1]
        self.block_size = self.key_cache_shape[2]
        self.head_dim = self.key_cache_shape[3]

        # compute cache bytes
        self.cache_dtype = args.cache_dtype
        self.cache_item_bytes = self._get_cache_item_bytes(self.cache_dtype)
        self.scale_item_bytes = self._get_cache_item_bytes(paddle.get_default_dtype())
        self.has_cache_scale = self.cache_dtype == "block_wise_fp8"
        if self.has_cache_scale:
            self.cache_scale_shape = [self.num_gpu_blocks, self.head_num, self.block_size]

        # extract other arg values
        self.model_id = os.path.basename(args.model_path.rstrip("/"))
        self.n_ranks = args.mp_num
        self.rank = args.rank
        self.device = args.device_id
        self.num_layers = args.num_layers
        self.ipc_suffix = args.ipc_suffix
        self.local_data_parallel_id = args.local_data_parallel_id
        self.num_extra_layers = self.speculative_config.num_extra_cache_layer
        self.num_extra_layer_gpu_blocks = int(self.num_gpu_blocks * self.speculative_config.num_gpu_block_expand_ratio)
        paddle.set_default_dtype(args.default_dtype)

        self.swap_to_cpu_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.swap_to_gpu_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.read_storage_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.write_back_storage_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.timeout_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.transfer_task_queue = queue.Queue()  # 用来接收传输任务
        self.tansfer_done_queue = queue.Queue()  # 用来告知任务执行完毕

        address = (args.pod_ip, args.cache_queue_port)
        self.cache_task_queue = EngineCacheQueue(
            address=address,
            is_server=False,
            num_client=args.mp_num,
            client_id=self.rank,
            local_data_parallel_id=args.local_data_parallel_id,
        )

        cache_ready_signal_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
            create=False,
        )
        swap_space_ready_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.swap_space_ready_signal = IPCSignal(
            name="swap_space_ready_signal",
            array=swap_space_ready_data,
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
            create=False,
        )

        self.num_cpu_blocks = args.num_cpu_blocks

        self._init_gpu_cache(args)
        if self.num_cpu_blocks > 0:
            self._init_cpu_cache(args)
        self._init_storage(args)

        cache_task_broadcast_data = np.zeros(shape=[1], dtype=np.int32)
        self.cache_task_broadcast_signal = IPCSignal(
            name="cache_task_broadcast_signal",
            array=cache_task_broadcast_data,
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
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

        # Initialize update/clear signals for RL
        self.kv_cache_status_signal = IPCSignal(
            name="kv_cache_status",
            array=np.zeros([1], dtype=np.int32),
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
            create=False,
        )
        threading.Thread(target=self.check_cache_status, args=[args], daemon=True).start()

        cache_transfer_inited_signal_data = np.zeros(shape=[args.mp_num], dtype=np.int32)
        self.cache_transfer_inited_signal = IPCSignal(
            name="cache_transfer_inited_signal",
            array=cache_transfer_inited_signal_data,
            dtype=np.int32,
            suffix=args.engine_worker_queue_port,
            create=False,
        )
        self.cache_transfer_inited_signal.value[self.rank] = 1

    def _init_storage(self, args):
        self.storage_backend_type = args.kvcache_storage_backend

        try:
            # TODO: support cache scale for other backend
            if self.has_cache_scale:
                if self.storage_backend_type not in ["mooncake"]:
                    raise ValueError(
                        f"Unsupported storage backend ({self.storage_backend_type}) "
                        "when cache quantization is block_wise_fp8"
                    )

            if self.storage_backend_type is None:
                self.storage_backend = None
            elif self.storage_backend_type == "mooncake":
                logger.info("Start initialize mooncake store...")
                self.storage_backend = MooncakeStore(tp_rank=self.rank)
                self._init_storage_buffer(args)
                logger.info("Initialized mooncake store successfully")
            elif self.storage_backend_type == "attention_store":
                logger.info("Start initialize attention store...")
                # TODO: support different model version in rl
                self.storage_backend = AttentionStore(
                    namespace=self.model_id,
                    shard_id=self.rank,
                    shard_num=self.n_ranks,
                    layer_num=self.num_layers + self.num_extra_layers,
                    block_token_size=self.block_size,
                    bytes_per_shard_layer_per_block=self.head_num
                    * self.block_size
                    * self.head_dim
                    * self.cache_item_bytes,
                    device_id=self.device,
                    dp_id=self.local_data_parallel_id,
                )
                logger.info("Initialized attention store successfully!")
            elif args.kvcache_storage_backend == "file":
                logger.info("Start initialize file store...")
                self.storage_backend = FileStore(
                    namespace=self.model_id,
                    tp_rank=self.rank,
                    tp_size=self.n_ranks,
                )
                self._init_storage_buffer(args)
                logger.info("Initialized file store successfully")
            else:
                raise NotImplementedError(f"Unsupported storage backend: {self.storage_backend_type}")
        except Exception as e:
            err_msg = f"Fail to initialize storage backend, {e}, traceback: {traceback.format_exc()}"
            logger.error(err_msg)
            console_logger.error(err_msg)  # print error message to console
            raise

        if args.write_policy not in ["write_through"]:
            raise ValueError(f"Invalid write policy: {args.write_policy}")
        self.write_policy = args.write_policy

        self.key_prefix = ""
        version_file_path = os.path.join(args.model_path, "version.yaml")
        if os.path.exists(version_file_path):
            self.key_prefix = get_key_prefix_from_version(version_file_path)
        logger.info(f"The key_prefix of cache storage is {self.key_prefix}")

        logger.info("Initialize cache storage successfully")

    def _init_storage_buffer(self, args):
        """
        Initialize pinned memory buffer that can hold the cache for a longest request
        cache layout: layer_num * [block_num, head_num, block_size, head_dim]
        scale layout: layer_num * [block_num, head_num, block_size]
        cache buffer layout: [block_num, layer_num, head_num, block_size, head_dim]
        scale buffer layout: [block_num, layer_num, head_num, block_size]
        """
        layer_num = self.num_layers + self.num_extra_layers
        block_num = (args.max_model_len + self.block_size - 1) // self.block_size
        logger.info(
            f"Creating cache buffer for storage with shape: "
            f"[{block_num}, {layer_num}, {self.head_num}, {self.block_size}, {self.head_dim}]"
        )

        self.cache_buffer_stride_bytes = (
            layer_num * self.head_num * self.block_size * self.head_dim * self.cache_item_bytes
        )
        cache_buffer_total_bytes = block_num * self.cache_buffer_stride_bytes * 2  # key and value

        logger.info(f"Creating cache cpu buffer for all layers: {cache_buffer_total_bytes / 1024 ** 3:.2f}GB")
        read_buffer = cuda_host_alloc(cache_buffer_total_bytes)
        self.storage_key_read_buffer = read_buffer
        self.storage_value_read_buffer = read_buffer + cache_buffer_total_bytes // 2
        self.storage_backend.register_buffer(read_buffer, cache_buffer_total_bytes)

        write_buffer = cuda_host_alloc(cache_buffer_total_bytes)
        self.storage_key_write_buffer = write_buffer
        self.storage_value_write_buffer = write_buffer + cache_buffer_total_bytes // 2
        self.storage_backend.register_buffer(write_buffer, cache_buffer_total_bytes)

        if self.has_cache_scale:
            self.scale_buffer_stride_bytes = layer_num * self.head_num * self.block_size * self.scale_item_bytes
            scale_buffer_total_bytes = block_num * self.scale_buffer_stride_bytes * 2
            logger.info(
                f"Creating scale cpu buffer cache for all layers: {scale_buffer_total_bytes / 1024 ** 3:.2f}GB"
            )

            read_buffer = cuda_host_alloc(scale_buffer_total_bytes)
            self.storage_key_scale_read_buffer = read_buffer
            self.storage_value_scale_read_buffer = read_buffer + scale_buffer_total_bytes // 2
            self.storage_backend.register_buffer(read_buffer, scale_buffer_total_bytes)

            write_buffer = cuda_host_alloc(scale_buffer_total_bytes)
            self.storage_key_scale_write_buffer = write_buffer
            self.storage_value_scale_write_buffer = write_buffer + scale_buffer_total_bytes // 2
            self.storage_backend.register_buffer(write_buffer, scale_buffer_total_bytes)

    def _init_gpu_cache(self, args):

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
        for i in range(self.num_layers + self.num_extra_layers):
            # NOTE: num_extra_layer_gpu_blocks is usually equal to num_gpu_blocks
            num_gpu_blocks = self.num_gpu_blocks if i < self.num_layers else self.num_extra_layer_gpu_blocks
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
        cache_item_bytes = self._get_cache_item_bytes(self.cache_dtype)
        key_need_to_allocate_bytes = args.num_cpu_blocks * cache_item_bytes * key_cache_size
        value_need_to_allocate_bytes = args.num_cpu_blocks * cache_item_bytes * value_cache_size
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
        for i in range(self.num_layers + self.num_extra_layers):
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

    def _get_cache_item_bytes(self, cache_dtype):
        if cache_dtype == "float32":
            bytes = 4
        elif cache_dtype in ("bfloat16", "float16"):
            bytes = 2
        elif cache_dtype in ["uint8", "block_wise_fp8"]:
            bytes = 1
        else:
            raise ValueError(f"Unsupported cache dtype: {cache_dtype}")
        return bytes

    def _run_read_storage(
        self,
        task_id: str,
        token_ids: List[int],
        start_read_block_idx: int,
        k_cache_keys: List[str],
        v_cache_keys: List[str],
        k_scale_keys: List[str],
        v_scale_keys: List[str],
        gpu_block_ids: List[int],
        cpu_block_ids: List[int],
        timeout: float,
    ):
        """
        Read storage data from the given blocks to the corresponding cache tensors on the current rank's GPU.
        """
        try:
            if self.storage_backend_type in ("mooncake", "file"):
                block_num = len(gpu_block_ids)
                keys = k_cache_keys + v_cache_keys
                k_cache_ptrs = [
                    self.storage_key_read_buffer + i * self.cache_buffer_stride_bytes for i in cpu_block_ids
                ]
                v_cache_ptrs = [
                    self.storage_value_read_buffer + i * self.cache_buffer_stride_bytes for i in cpu_block_ids
                ]
                target_locations = k_cache_ptrs + v_cache_ptrs
                target_sizes = [self.cache_buffer_stride_bytes] * block_num * 2  # key and value
                if k_scale_keys and v_scale_keys:
                    keys.extend(k_scale_keys + v_scale_keys)
                    k_scale_ptrs = [
                        self.storage_key_scale_read_buffer + i * self.scale_buffer_stride_bytes for i in cpu_block_ids
                    ]
                    v_scale_ptrs = [
                        self.storage_value_scale_read_buffer + i * self.scale_buffer_stride_bytes
                        for i in cpu_block_ids
                    ]
                    target_locations.extend(k_scale_ptrs + v_scale_ptrs)
                    target_sizes.extend([self.scale_buffer_stride_bytes] * block_num * 2)

                start_time = time.time()
                result = self.storage_backend.batch_get(
                    keys=keys, target_locations=target_locations, target_sizes=target_sizes
                )
                read_cost_time = time.time() - start_time

                if k_scale_keys and v_scale_keys:
                    k_result, v_result = result[:block_num], result[block_num : 2 * block_num]
                    k_scale_result, v_scale_result = result[2 * block_num : 3 * block_num], result[3 * block_num :]
                    success_block_num = 0
                    for k, v, k_scale, v_scale in zip(k_result, v_result, k_scale_result, v_scale_result):
                        if not (k > 0 and v > 0 and k_scale > 0 and v_scale > 0):
                            break
                        success_block_num += 1
                else:
                    k_result, v_result = result[:block_num], result[block_num : 2 * block_num]
                    success_block_num = 0
                    for k, v in zip(k_result, v_result):
                        if not (k > 0 and v > 0):
                            break
                        success_block_num += 1
                logger.debug(f"_run_read_storage, success_block_num: {success_block_num}")
                valid_gpu_block_ids = gpu_block_ids[:success_block_num]
                valid_cpu_block_ids = cpu_block_ids[:success_block_num]

                mode = 1  # cpu ==> gpu
                start_time = time.time()
                swap_cache_layout(
                    self.gpu_cache_k_tensors,
                    self.storage_key_read_buffer,
                    self.key_cache_shape,
                    valid_gpu_block_ids,
                    valid_cpu_block_ids,
                    self.device,
                    mode,
                )
                swap_cache_layout(
                    self.gpu_cache_v_tensors,
                    self.storage_value_read_buffer,
                    self.value_cache_shape,
                    valid_gpu_block_ids,
                    valid_cpu_block_ids,
                    self.device,
                    mode,
                )
                if k_scale_keys and v_scale_keys:
                    swap_cache_layout(
                        self.gpu_cache_scales_k_tensors,
                        self.storage_key_scale_read_buffer,
                        self.cache_scale_shape,
                        valid_gpu_block_ids,
                        valid_cpu_block_ids,
                        self.device,
                        mode,
                    )
                    swap_cache_layout(
                        self.gpu_cache_scales_v_tensors,
                        self.storage_value_scale_read_buffer,
                        self.cache_scale_shape,
                        valid_gpu_block_ids,
                        valid_cpu_block_ids,
                        self.device,
                        mode,
                    )
                swap_cost_time = time.time() - start_time
                logger.debug(
                    f"_run_read_storage, swap_cost_time: {swap_cost_time:.6f}s, read_cost_time: {read_cost_time:.6f}s"
                )

            elif self.storage_backend_type == "attention_store":
                key_cache = []
                val_cache = []
                for i in range(self.num_layers + self.num_extra_layers):
                    key_cache.append(self.gpu_cache_kvs[f"key_caches_{i}_rank{self.rank}.device{self.device}"])
                    val_cache.append(self.gpu_cache_kvs[f"value_caches_{i}_rank{self.rank}.device{self.device}"])

                start_time = time.time()
                read_block_num = self.storage_backend.read(
                    task_id, key_cache, val_cache, token_ids, gpu_block_ids, start_read_block_idx, timeout
                )
                read_cost_time = time.time() - start_time
                valid_gpu_block_ids = gpu_block_ids[:read_block_num]
                logger.debug(f"_run_read_storage, read_cost_time: {read_cost_time:.6f}s")

            return valid_gpu_block_ids

        except Exception as e:
            logger.error(
                f"An error occurred in _run_read_storage, " f"error: {e}, traceback:\n{traceback.format_exc()}"
            )
            raise

    def read_storage_task(self, task: ReadStorageTask):
        """Read cache from the storage backend to the GPU memory."""
        assert (
            self.storage_backend
        ), f"storage_backend not initialized, storage_backend_type: {self.storage_backend_type}"

        try:
            gpu_block_ids = task.gpu_block_ids.copy()
            cpu_block_ids = [i for i in range(len(gpu_block_ids))]
            k_cache_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_key" for key in task.keys]
            v_cache_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_value" for key in task.keys]
            if not self.has_cache_scale:
                k_scale_keys = None
                v_scale_keys = None
            else:
                k_scale_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_key_scale" for key in task.keys]
                v_scale_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_value_scale" for key in task.keys]

            match_block_num = 0
            if self.storage_backend_type in ("mooncake", "file"):
                match_block_num = self.storage_backend.query(
                    k_cache_keys, v_cache_keys, k_scale_keys, v_scale_keys, task.timeout
                )
            elif self.storage_backend_type == "attention_store":
                match_block_num = self.storage_backend.query(
                    task.task_id, task.token_ids, task.start_read_block_idx, task.timeout
                )
            logger.info(f"Matched {match_block_num} blocks in cache storage for read task {task.task_id}")

            k_cache_keys = k_cache_keys[:match_block_num]
            v_cache_keys = v_cache_keys[:match_block_num]
            k_scale_keys = k_scale_keys[:match_block_num] if k_scale_keys else None
            v_scale_keys = v_scale_keys[:match_block_num] if v_scale_keys else None
            gpu_block_ids = gpu_block_ids[:match_block_num]
            cpu_block_ids = cpu_block_ids[:match_block_num]
            valid_gpu_block_ids = []
            if match_block_num > 0:
                # TODO: support timeout with actual block count
                try:
                    valid_gpu_block_ids = self._run_read_storage(
                        task.task_id,
                        task.token_ids[: match_block_num * self.block_size],
                        task.start_read_block_idx,
                        k_cache_keys,
                        v_cache_keys,
                        k_scale_keys,
                        v_scale_keys,
                        gpu_block_ids,
                        cpu_block_ids,
                        task.timeout,
                    )
                    logger.info(
                        f"Successfully read {len(valid_gpu_block_ids)} blocks from cache storage for task {task.task_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to read cache for task {task.task_id}, error: {e}, traceback: {traceback.format_exc()}"
                    )
                    valid_gpu_block_ids = []
                finally:
                    try:
                        if (self.rank == 0) and self.storage_backend_type == "attention_store":
                            self.storage_backend.flush_token_index(task.task_id, task.token_ids, 0, True)
                        logger.info(f"Report cache index in HBM to cache storage for task {task.task_id}")
                    except Exception as e:
                        logger.info(
                            f"Failed to report cache index in HBM to cache storage for task {task.task_id}, error: {e}"
                        )

            result = (CacheStatus.STORAGE2GPU, task.task_id, task.keys, valid_gpu_block_ids)
            self.cache_task_queue.swap_storage_to_gpu_barrier.wait()
            self.cache_task_queue.swap_storage_to_gpu_barrier.reset()
            self.cache_task_queue.put_transfer_done_signal(result)
            logger.debug(f"read_storage_task: put transfer done signal for {task.task_id}")

        except Exception as e:
            logger.error(
                f"An error occurred in read_storage_task: "
                f"task_id: {task.task_id}, error:{e}, {traceback.format_exc()}"
            )

    def _run_write_back_storage(
        self,
        task_id,
        token_ids,
        start_write_block_idx,
        k_cache_keys,
        v_cache_keys,
        k_scale_keys,
        v_scale_keys,
        gpu_block_ids,
        cpu_block_ids,
        timeout,
    ):
        try:
            if self.storage_backend_type in ("mooncake", "file"):
                mode = 0  # gpu ==> cpu
                start_time = time.time()
                swap_cache_layout(
                    self.gpu_cache_k_tensors,
                    self.storage_key_write_buffer,
                    self.key_cache_shape,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    mode,
                )
                swap_cache_layout(
                    self.gpu_cache_v_tensors,
                    self.storage_value_write_buffer,
                    self.key_cache_shape,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    mode,
                )
                if k_scale_keys and v_scale_keys:
                    swap_cache_layout(
                        self.gpu_cache_scales_k_tensors,
                        self.storage_key_scale_write_buffer,
                        self.cache_scale_shape,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        mode,
                    )
                    swap_cache_layout(
                        self.gpu_cache_scales_v_tensors,
                        self.storage_value_scale_write_buffer,
                        self.cache_scale_shape,
                        gpu_block_ids,
                        cpu_block_ids,
                        self.device,
                        mode,
                    )
                swap_cost_time = time.time() - start_time

                block_num = len(gpu_block_ids)
                keys = k_cache_keys + v_cache_keys
                k_cache_ptrs = [
                    self.storage_key_write_buffer + i * self.cache_buffer_stride_bytes for i in cpu_block_ids
                ]
                v_cache_ptrs = [
                    self.storage_value_write_buffer + i * self.cache_buffer_stride_bytes for i in cpu_block_ids
                ]
                target_locations = k_cache_ptrs + v_cache_ptrs
                target_sizes = [self.cache_buffer_stride_bytes] * block_num * 2  # key and value
                if k_scale_keys and v_scale_keys:
                    keys.extend(k_scale_keys + v_scale_keys)
                    k_scale_ptrs = [
                        self.storage_key_scale_write_buffer + i * self.scale_buffer_stride_bytes for i in cpu_block_ids
                    ]
                    v_scale_ptrs = [
                        self.storage_value_scale_write_buffer + i * self.scale_buffer_stride_bytes
                        for i in cpu_block_ids
                    ]
                    target_locations.extend(k_scale_ptrs + v_scale_ptrs)
                    target_sizes.extend([self.scale_buffer_stride_bytes] * block_num * 2)

                start_time = time.time()
                self.storage_backend.batch_set(keys=keys, target_locations=target_locations, target_sizes=target_sizes)
                write_cost_time = time.time() - start_time

                logger.debug(
                    f"_run_write_back_storage, swap_cost_time: {swap_cost_time:.6f}s, write_cost_time: {write_cost_time:.6f}s"
                )
                return block_num

            elif self.storage_backend_type == "attention_store":
                key_cache = []
                val_cache = []
                for i in range(self.num_layers + self.num_extra_layers):
                    key_cache.append(self.gpu_cache_kvs[f"key_caches_{i}_rank{self.rank}.device{self.device}"])
                    val_cache.append(self.gpu_cache_kvs[f"value_caches_{i}_rank{self.rank}.device{self.device}"])

                start_time = time.time()
                write_block_num = self.storage_backend.write(
                    task_id, key_cache, val_cache, token_ids, gpu_block_ids, start_write_block_idx, timeout
                )
                write_cost_time = time.time() - start_time
                logger.debug(f"_run_write_back_storage, write_cost_time: {write_cost_time:.6f}s")
                return write_block_num

        except Exception as e:
            logger.error(
                f"An error occurred in _run_write_back_storage, " f"error: {e}, traceback:\n{traceback.format_exc()}"
            )
            return 0

    def write_back_storage_task(self, task: WriteStorageTask):
        """
        Write cache to the storage backend from the GPU memory.
        """
        assert (
            self.storage_backend
        ), f"storage_backend not initialized, storage_backend_type: {self.storage_backend_type}"

        try:
            gpu_block_ids = task.gpu_block_ids.copy()
            cpu_block_ids = [i for i in range(len(gpu_block_ids))]
            k_cache_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_key" for key in task.keys]
            v_cache_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_value" for key in task.keys]
            if not self.has_cache_scale:
                k_scale_keys = None
                v_scale_keys = None
            else:
                k_scale_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_key_scale" for key in task.keys]
                v_scale_keys = [f"prefix{self.key_prefix}_{key}_{self.rank}_value_scale" for key in task.keys]

            match_block_num = 0
            if self.storage_backend_type == ("mooncake", "file"):
                match_block_num = self.storage_backend.query(
                    k_cache_keys, v_cache_keys, k_scale_keys, v_scale_keys, task.timeout
                )
            elif self.storage_backend_type == "attention_store":
                match_block_num = self.storage_backend.query(task.task_id, task.token_ids, 0, task.timeout)
            logger.info(f"Matched {match_block_num} blocks in cache storage for write task {task.task_id}")

            if match_block_num >= len(k_cache_keys):
                logger.info(f"No uncached keys found for task {task.task_id}")
                gpu_block_ids = []
            else:
                try:
                    k_cache_keys = k_cache_keys[match_block_num:]
                    v_cache_keys = v_cache_keys[match_block_num:]
                    k_scale_keys = k_scale_keys[match_block_num:] if k_scale_keys else None
                    v_scale_keys = v_scale_keys[match_block_num:] if v_scale_keys else None
                    gpu_block_ids = gpu_block_ids[match_block_num:]
                    cpu_block_ids = cpu_block_ids[match_block_num:]
                    # TODO: support timeout with actual block count
                    write_block_num = self._run_write_back_storage(
                        task.task_id,
                        task.token_ids,
                        match_block_num,
                        k_cache_keys,
                        v_cache_keys,
                        k_scale_keys,
                        v_scale_keys,
                        gpu_block_ids,
                        cpu_block_ids,
                        task.timeout,
                    )
                    logger.info(
                        f"Successfully wrote {write_block_num} blocks to cache storage for task {task.task_id}"
                    )
                except Exception as e:
                    logger.error(f"Error in write back storage task: {e}, traceback:{traceback.format_exc()}")
                    gpu_block_ids = []
                finally:
                    try:
                        if (self.rank == 0) and self.storage_backend_type == "attention_store":
                            self.storage_backend.flush_token_index(task.task_id, task.token_ids, 0, False)
                        logger.info(f"Report cache index out HBM to cache storage for task {task.task_id}")
                    except Exception as e:
                        logger.info(
                            f"Failed to report cache index out HBM to cache storage for task {task.task_id}, error: {e}"
                        )

            result = (CacheStatus.GPU2STORAGE, task.task_id, task.keys, gpu_block_ids)
            self.cache_task_queue.swap_to_storage_barrier.wait()
            if self.rank == 0:  # 只有当rank为0时执行同步操作
                self.cache_task_queue.swap_to_storage_barrier.reset()
                self.cache_task_queue.put_transfer_done_signal(result)  # 发送传输完成信号
                logger.debug(f"write_back_storage_task: put_transfer_done_signal {result}")
        except Exception as e:
            logger.error(
                f"An error occurred in write_back_storage_task, " f"error: {e}, traceback:\n{traceback.format_exc()}"
            )

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
                    logger.debug(f"do_data_transfer: {data}")
                    if read_finish:
                        self.cache_task_broadcast_signal.value[0] = 0
                    event_type, event_args = data[0], data[1:]
                    if event_type.value == CacheStatus.SWAP2CPU.value:
                        transfer_task_id, swap_node_ids, gpu_block_id, cpu_block_id = event_args
                        self.swap_to_cpu_thread_pool.submit(
                            self._do_swap_to_cpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                    elif event_type.value == CacheStatus.SWAP2GPU.value:
                        transfer_task_id, swap_node_ids, gpu_block_id, cpu_block_id = event_args
                        self.swap_to_gpu_thread_pool.submit(
                            self._do_swap_to_gpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                    elif event_type.value == CacheStatus.STORAGE2GPU.value:
                        read_storage_task = event_args[0]
                        self.read_storage_thread_pool.submit(
                            self.read_storage_task,
                            read_storage_task,
                        )
                    elif event_type.value == CacheStatus.GPU2STORAGE.value:
                        write_storage_task = event_args[0]
                        self.write_back_storage_thread_pool.submit(
                            self.write_back_storage_task,
                            write_storage_task,
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
            event_type,
            transfer_task_id,
            swap_node_ids,
            task_gpu_block_id,
            task_cpu_block_id,
        )

    def check_cache_status(self, args):
        # TODO XPU support RL
        if unset_data_ipc is None:
            return
        logger.info("[RL] Launch a thread to clear/restore kv cache when model weights are cleared/updated.")
        while True:
            # handle cache clearing/restoring
            if self.kv_cache_status_signal.value[0] == KVCacheStatus.CLEARING:
                assert args.splitwise_role == "mixed", "Only mixed mode supports clearing cache."
                try:
                    # clear cpu caches
                    logger.info("[RL] start clearing caches")
                    logger.debug("[RL] start clearing cpu caches")
                    if self.num_cpu_blocks > 0 and envs.FD_ENABLE_SWAP_SPACE_CLEARING:
                        paddle.set_device("cpu")
                        for ptrs in self.k_dst_ptrs + self.v_dst_ptrs:
                            cuda_host_free(ptrs)
                        self.cpu_cache_kvs.clear()
                        self.k_dst_ptrs.clear()
                        self.v_dst_ptrs.clear()
                        if self.cache_dtype == "block_wise_fp8":
                            self.k_scales_ptrs.clear()
                            self.v_scales_ptrs.clear()
                        gc.collect()
                        logger.debug("[RL] successfully cleared cpu caches")
                        # reset swap_space_ready_signal
                        self.swap_space_ready_signal.value[self.rank] = 0
                        while np.sum(self.swap_space_ready_signal.value) != 0:
                            time.sleep(0.1)
                        logger.debug("[RL] all ranks cleared cpu caches")
                    else:
                        logger.debug("[RL] skip clearing cpu caches")

                    # clear gpu caches
                    logger.debug("[RL] start clearing gpu caches")
                    if args.create_cache_tensor:
                        logger.info("[RL] waiting for gpu runner to unlink cuda ipc")
                        while self.cache_ready_signal.value[self.rank] != 0:
                            time.sleep(0.1)
                        logger.info("[RL] stop waiting! gpu runner has unlinked cuda ipc")
                        paddle.set_device(f"gpu:{self.device}")
                        self.gpu_cache_kvs.clear()
                        self.gpu_cache_k_tensors.clear()
                        self.gpu_cache_v_tensors.clear()
                        if self.cache_dtype == "block_wise_fp8":
                            self.gpu_cache_scales_k_tensors.clear()
                            self.gpu_cache_scales_v_tensors.clear()
                        paddle.device.cuda.empty_cache()
                        logger.debug("[RL] successfully cleared gpu caches")
                    else:
                        for name, tensor in self.gpu_cache_kvs.items():
                            unset_data_ipc(tensor, name, True, False)
                        logger.debug("[RL] successfully unlinked gpu caches cuda ipc")
                        self.cache_ready_signal.value[self.rank] = 0

                    while np.sum(self.cache_ready_signal.value) != 0:
                        time.sleep(0.1)
                    logger.info("[RL] all ranks cleared caches!")

                    # reset kv_cache_status_signal
                    self.kv_cache_status_signal.value[0] = KVCacheStatus.CLEARED

                    self._log_memory("after clearing caches")

                except Exception as e:
                    logger.error(f"[RL] failed to clear caches: {e}")

            elif self.kv_cache_status_signal.value[0] == KVCacheStatus.UPDATING:
                assert args.splitwise_role == "mixed", "Only mixed mode supports updating cache."
                try:
                    # restore cpu cache
                    logger.info("[RL] start restoring caches")
                    logger.debug("[RL] start restoring cpu caches")
                    if self.num_cpu_blocks > 0 and envs.FD_ENABLE_SWAP_SPACE_CLEARING:
                        self._init_cpu_cache(args)
                        logger.debug("[RL] successfully restored cpu caches")
                        while np.sum(self.swap_space_ready_signal.value) != args.mp_num:
                            time.sleep(0.1)
                        logger.debug("[RL] all ranks restored cpu caches")
                    else:
                        logger.debug("[RL] skip restoring cpu caches")

                    # restore gpu cache and set cache_ready_signal
                    logger.debug("[RL] start restoring gpu caches")
                    self._init_gpu_cache(args)
                    logger.debug("[RL] successfully restored gpu caches")

                    if self.storage_backend_type is not None:
                        # use key_prefix to distinguish cache for different version of weight in rl
                        version_file_path = os.path.join(args.model_path, "version.yaml")
                        assert os.path.exists(version_file_path), f"version.yaml not found at {version_file_path}"
                        self.key_prefix = get_key_prefix_from_version(version_file_path)
                        logger.info(f"Update key_prefix of cache storage to {self.key_prefix}")

                    # wait for all ranks caches to be ready
                    while np.sum(self.cache_ready_signal.value) != args.mp_num:
                        time.sleep(0.1)
                    logger.info("[RL] all ranks restored caches!")

                    # set kv_cache_status_signal
                    self.kv_cache_status_signal.value[0] = KVCacheStatus.NORMAL

                    self._log_memory("after restoring caches")
                except Exception as e:
                    logger.error(f"[RL] failed to restore caches: {e}")

            time.sleep(0.1)

    def _log_memory(self, context: str):
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024**3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024**3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024**3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024**3)

        logger.warning(
            f"GPU memory usage {context}:"
            f"max_allocated: {max_alloc:.2f}GB "
            f"max_reserved: {max_reserved:.2f}GB "
            f"current_allocated: {curr_alloc:.2f}GB "
            f"current_reserved: {curr_reserved:.2f}GB"
        )


def main():
    """
    启动cache manager
    """
    cache_manager = CacheTransferManager(args)
    cache_manager.do_data_transfer()


if __name__ == "__main__":

    args = parse_args()
    rank_id = args.rank + args.local_data_parallel_id * args.mp_num
    if args.mp_num > 1:
        logger = get_logger("cache_transfer", f"cache_transfer_{rank_id}.log")
    else:
        logger = get_logger("cache_transfer", "cache_transfer.log")

    logger.info(f"args: {vars(args)}")
    set_device(args.device_id)
    try:
        main()
    except Exception as e:
        logger.error(f"cache_transfer_manager failed with error: {e}, traceback: {traceback.format_exc()}")
        raise
