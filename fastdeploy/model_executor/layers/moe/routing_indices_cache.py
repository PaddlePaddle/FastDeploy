"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import asyncio
import atexit
import functools
import multiprocessing
import os
import shutil
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from typing import Dict, Optional, TypedDict

import numpy as np
import paddle
import paddle.distributed as dist
import triton
import triton.language as tl
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, RoutingReplayConfig


@triton.jit
def _save_routing_kernel(
    ROUTING_REPLAY_TABLE_PTR,
    TOPK_IDS_PTR,
    BATCH_ID_PER_TOKEN_PTR,
    CU_SEQLENS_Q_PTR,
    SEQ_LENS_DECODER_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_HIDDEN_LAYERS,
    MAX_MODEL_LEN,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM

    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    k_mask = k_offsets < TOP_K

    topk_ids_ptrs = TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :]
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(topk_ids_ptrs, mask=load_mask)

    batch_ids = tl.load(BATCH_ID_PER_TOKEN_PTR + token_offsets, mask=token_mask)
    pad_mask = token_mask & (batch_ids != -1)
    # [0, 3, 4, 10, 12][0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3]
    # -> [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] - [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # -> [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1]
    start_offsets = tl.load(CU_SEQLENS_Q_PTR + batch_ids, mask=pad_mask)
    token_relative_index = token_offsets - start_offsets

    # [BLOCK_SIZE_M]
    len_decoder = tl.load(SEQ_LENS_DECODER_PTR + batch_ids, mask=pad_mask)
    token_seq_pos = len_decoder + token_relative_index

    STRIDE_BUF_SEQ = MAX_MODEL_LEN * NUM_HIDDEN_LAYERS * TOP_K
    STRIDE_BUF_TOKEN = NUM_HIDDEN_LAYERS * TOP_K
    STRIDE_BUF_LAYER = TOP_K

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    output_ptrs = (
        ROUTING_REPLAY_TABLE_PTR
        + batch_ids[:, None] * STRIDE_BUF_SEQ
        + token_seq_pos[:, None] * STRIDE_BUF_TOKEN
        + LAYER_IDX * STRIDE_BUF_LAYER
        + k_offsets[None, :]
    )

    pos_mask = token_seq_pos < MAX_MODEL_LEN
    pos_mask = pos_mask & pad_mask

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    pos_mask = pos_mask[:, None] & k_mask[None, :]

    final_mask = load_mask & pos_mask

    tl.store(output_ptrs, topk_vals, mask=final_mask)


def save_routing_to_buffer(
    routing_replay_table: paddle.Tensor,  # [max_num_seqs, num_layers, max_len, top_k]
    topk_ids: paddle.Tensor,  # [token_num, top_k]
    batch_id_per_token: paddle.Tensor,  # [token_num, 1]
    seq_lens_decoder: paddle.Tensor,  # [max_num_seqs, 1]
    cu_seqlens_q: paddle.Tensor,  # [max_num_seqs + 1, 1]
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
):
    if tp_size > 1 and ep_size > 1:
        token_num_per_rank = topk_ids.shape[0]
        if token_num_per_rank == 0:
            return
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        topk_ids = topk_ids_all[: batch_id_per_token.shape[0], :]

    token_num, top_k = topk_ids.shape
    max_num_seqs, max_model_len, num_hidden_layers, _ = routing_replay_table.shape
    assert token_num > 0
    assert topk_ids.shape[1] == routing_replay_table.shape[3], (topk_ids.shape[1], routing_replay_table.shape[3])
    assert batch_id_per_token.shape[0] == token_num, (batch_id_per_token.shape[0], token_num)
    assert seq_lens_decoder.shape[0] == max_num_seqs, (seq_lens_decoder.shape[0], max_num_seqs)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)  # top_k

    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel[grid](
        routing_replay_table,
        topk_ids,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_decoder,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_HIDDEN_LAYERS=num_hidden_layers,
        MAX_MODEL_LEN=max_model_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


class RoutingReplayManager:
    """Request level routing replay table manager"""

    def __init__(self, fd_config: FDConfig, block_table, total_block_num):
        self.fd_config = fd_config
        self.block_table = block_table
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.max_model_len = fd_config.model_config.max_model_len
        self.num_moe_layers = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index
        self.only_last_turn = fd_config.routing_replay_config.only_last_turn
        self.use_fused_put = fd_config.routing_replay_config.use_fused_put
        if fd_config.model_config.architectures[0] == "Glm4MoeForCausalLM":
            self.moe_top_k = fd_config.model_config.num_experts_per_tok
        else:
            self.moe_top_k = fd_config.model_config.moe_k
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        # Initialize the routing replay table and routing cache
        self.routing_batch_to_request: Dict[int, str] = {}
        num_experts = fd_config.model_config.moe_num_experts + fd_config.model_config.moe_num_shared_experts
        self.routing_dtype = self.get_routing_dtype(num_experts=num_experts)
        self._init_routing_cache(dtype=self.routing_dtype, total_block_num=total_block_num)

        # Initialize routing store wrapper
        if self.tp_rank == 0:
            self._store_wrapper = StoreWrapper(
                fd_config=fd_config,
            )
            self._store_wrapper.start_store_warpper()

    def _init_routing_cache(self, dtype: str, total_block_num: int):
        """Initialize the device buffer and host buffer."""

        max_num_kv_tokens = total_block_num * self.fd_config.cache_config.block_size

        self._host_cache = paddle.full(
            shape=[max_num_kv_tokens, self.num_moe_layers, self.moe_top_k], fill_value=-1, dtype=dtype, device="cpu"
        )

        self.routing_replay_table = paddle.full(
            shape=[self.max_num_seqs, self.max_model_len, self.num_moe_layers, self.moe_top_k],
            fill_value=-1,
            dtype=dtype,
        )
        logger.info(
            f"[R3] The host cache size is:{self._host_cache.shape}, device cache size is: {self.routing_replay_table.shape}"
        )

    def get_routing_dtype(self, num_experts: int, reserved_fill_value: int = 1) -> str:
        """Calculate the minimum number of bits required for storage routing."""
        if num_experts <= 0:
            raise ValueError(f"num_experts must be greater than 0 but got {num_experts}, please check model config.")
        dtype = "uint8"
        total_number = num_experts + reserved_fill_value
        if total_number <= 255:  # uint8: 0~255
            dtype = "uint8"
        elif total_number <= 65535:  # uint16: 0~65,535
            dtype = "uint16"
        elif total_number <= 4294967295:  # uint32: 0~4,294,967,295
            dtype = "uint32"
        else:
            raise ValueError(
                f"The number of experts {num_experts} exceeds the representation range of uint32, please check model config."
            )
        logger.info(f"[R3] Routing replay table dtype: {dtype}")
        return dtype

    def update_host_cache(self, positions: paddle.Tensor, slot_mapping: paddle.Tensor):
        """Update the host cache with new tokens"""
        for batch_id, position in enumerate(positions):
            if len(position) > 0 and len(slot_mapping[batch_id]) > 0:
                routing_ids = self.routing_replay_table[batch_id, position, :, :].contiguous()
                routing_ids = routing_ids.cpu()

                self._host_cache[slot_mapping[batch_id], :, :] = routing_ids

    def get_token_positions(self, seq_lens_decoder, seq_lens_this_time):
        """Get token position of each sequence in a batch."""
        starts = seq_lens_decoder.numpy()[:, 0]
        increase_num = seq_lens_this_time.numpy()[:, 0]

        positions = []
        for i in range(self.max_num_seqs):
            if seq_lens_this_time[i] == 0:
                positions.append([])
                continue
            repeated_base = np.repeat(starts[i], increase_num[i])
            positions.append(repeated_base + np.arange(0, increase_num[i]))

        return positions

    def compute_slot_mapping(self, positions: np.ndarray):
        """Compute the mapping between token ids and kvcache slots"""
        slot_mapping = []
        for batch_id, position in enumerate(positions):
            if len(position) == 0:
                slot_mapping.append([])
                continue
            block_table_indices = position // self.fd_config.cache_config.block_size
            token_block_ids = self.block_table[batch_id, block_table_indices]
            block_offset = position % self.fd_config.cache_config.block_size

            token_cache_ids = np.array(token_block_ids) * self.fd_config.cache_config.block_size + block_offset
            slot_mapping.append(token_cache_ids)

        return slot_mapping

    def _get_routing_from_cache(self, finished_batch_ids, seq_lens_decoder):
        """
        When request is finished or cleared the length of the request is recorded at seq_lens_decoder
            1. finish the step: after update input, lens = seq_lens_decoder_buffer
            2. clear parameter: after update input, lens = seq_lens_decoder_buffer
        """
        # Get the slot mapping of the request cache.
        current_token_nums = seq_lens_decoder.numpy()[:, 0]
        positions = []
        for batch_id in range(self.max_num_seqs):
            position = []
            if batch_id in finished_batch_ids:
                position = np.arange(0, current_token_nums[batch_id])
            positions.append(position)

        # Collection the cached routing information
        token_cache_ids = self.compute_slot_mapping(positions=positions)
        for slot_map in token_cache_ids:
            if len(slot_map) > 0:
                token_cached_routing = self._host_cache[slot_map, :, :]
                return paddle.transpose(token_cached_routing, [1, 0, 2])
        raise ValueError("No cached routing found")

    def put_finished_batch(
        self,
        finished_batch_ids,
        seq_lens_decoder,
    ):
        finished_batch_ids_list = finished_batch_ids.cpu().tolist()
        for batch_id, finished in enumerate(finished_batch_ids_list):
            if finished:
                assert batch_id in self.routing_batch_to_request.keys()
                # Deregister the request
                request_id = self._deregister_request(batch_id)
                # Put the routing of finished request to store
                self._put_request_to_store(
                    batch_id=batch_id,
                    request_id=request_id,
                    seq_lens_decoder=seq_lens_decoder,
                )
                # Clear the slot of the finished batch
                self._clear_table_slot(batch_id)

    def register_request(self, batch_id: int, request_id: str):
        """
        Register a new request to routing replay table
        Args:
            batch_id: The batch ID of this request
            request_id: The global ID of the request is usually executed by the training process in RL
        """
        # The chunked prefill tasks will be registered repeatedly
        if batch_id in self.routing_batch_to_request:
            if self.routing_batch_to_request[batch_id] == request_id:
                logger.warning(f"[R3] Request {request_id} has been registered at {batch_id}.")
                return
            else:
                raise RuntimeError(
                    f"[R3] The Batch {batch_id} has been registered by request {self.routing_batch_to_request[batch_id]}, now robed by {request_id},"
                )

        # Register the new request
        self.routing_batch_to_request[batch_id] = request_id
        logger.info(f"[R3] Register request {request_id} with batch id {batch_id}")

    def _deregister_request(self, batch_id: int) -> str:
        """
        Deregister a request from routing replay table
        """
        assert batch_id in self.routing_batch_to_request
        return self.routing_batch_to_request.pop(batch_id)

    def _put_request_to_store(
        self,
        batch_id: int,
        request_id: str,
        seq_lens_decoder,
    ):
        if self.tp_rank == 0:
            before_put_request_time = time.perf_counter()

            # Collect the routing of finished request
            batch_buffer = self._get_routing_from_cache(
                finished_batch_ids=[batch_id], seq_lens_decoder=seq_lens_decoder
            )
            rollout_id = self.split_request_id(request_id)

            if self.use_fused_put:
                self._store_wrapper.submit_put_task(routing_indices=batch_buffer, rollout_id=rollout_id)
            else:
                for layer_id in range(self.num_moe_layers):
                    layer_buffer = batch_buffer[layer_id]
                    self._store_wrapper.submit_put_task(
                        routing_indices=layer_buffer, rollout_id=rollout_id, layer_idx=layer_id
                    )

            # Only store the routing of last turn
            if self.only_last_turn:
                self._store_wrapper.submit_clear_prefix_batch_task(rollout_id=rollout_id)

            logger.info(f"[R3] Submit {request_id} time cost: {time.perf_counter() - before_put_request_time}")

    def clear_request(self, batch_id: int):
        """Clear the routing indices of the request"""
        self._clear_table_slot(batch_id)
        self.routing_batch_to_request.pop(batch_id, None)

    def _clear_table_slot(self, batch_id: int):
        assert 0 <= batch_id < self.max_num_seqs
        self.routing_replay_table[batch_id].fill_(-1)

    def get_routing_table(self) -> paddle.Tensor:
        return self.routing_replay_table

    def split_request_id(self, request_id: str):
        """
        Split the request id to get rollout id.

        request_id: "chatcmpl-request.user-uuid"
        rollout_id: "request.user"
            example: "chatcmpl-xxx_xxx_epoch_15:2:2:1-d9f16c5c-65f6-4815-b44d-14e2c581907c_0" -> "xxx_xxx_epoch_15:2:2:1"
        """
        chat_type, tmp_str = request_id.split("-", 1)
        # NOTE(gongshaotian): only support chatcmpl now
        assert (
            chat_type == "chatcmpl"
        ), "Rollout Routing Replay only supports chatcmpl. Please check whether the request type and userid settings are correct."
        reversed_tmp_str = tmp_str[::-1].split("-", 5)
        rollout_id = reversed_tmp_str[-1][::-1]
        return rollout_id


class StoreWrapper(object):
    def __init__(self, fd_config: False) -> None:
        super().__init__()
        self.fd_config = fd_config

        # Initialize task queue
        moe_layer_num = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index
        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.queue_max_size = moe_layer_num * max_num_seqs * 10

        self.manager = multiprocessing.Manager()
        self._task_queue = self.manager.Queue(maxsize=self.queue_max_size)

        self._monitor_thread: threading.Thread = None
        self._stop_monitor = threading.Event()

        # Initialize consumer process
        self._routing_store_process = StoreProcess(
            task_queue=self._task_queue,
            routing_replay_config=self.fd_config.routing_replay_config,
            max_model_len=self.fd_config.model_config.max_model_len,
        )
        self._sotre_process_running = False

        # Register atexit handler
        atexit.register(self.shutdown)

    def shutdown(self):
        """ """
        if not self._sotre_process_running:
            return
        self._sotre_process_running = False

        # Stop the monitor thread
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

        # Put a sentinel value to signal the consumer to stop
        if self._routing_store_process and self._routing_store_process.is_alive():
            try:
                self._task_queue.put_nowait(None)
            except Exception as e:
                logger.info(f"Could not put sentinel into queue: {e}")

        if self._routing_store_process and self._routing_store_process.is_alive():
            # Wait for all tasks to be processed
            self._routing_store_process.join(timeout=10.0)
            if self._routing_store_process.is_alive():
                self._routing_store_process.close()
                self._routing_store_process.join()

        self._task_queue.join()
        self.manager.shutdown()
        self._sotre_process_running = False

    def start_store_warpper(self):
        """ """
        if self._sotre_process_running:
            return
        self._sotre_process_running = True

        # Start monitor thread
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_queue_load, daemon=True)
        self._monitor_thread.start()

        # Start Routing Store Wrapper in sub process
        self._routing_store_process.start()

    def _monitor_queue_load(self):
        """ """
        while not self._stop_monitor.is_set():
            time.sleep(2.0)
            if not self._sotre_process_running:
                break
            qsize = self._task_queue.qsize()

            # Alarm when the task exceeds 80% of the queue capacity
            if qsize > self.queue_max_size * 0.8:
                logger.warning(
                    f"[Monitor] Queue load is HIGH: {qsize}/{self.queue_max_size}. "
                    f"Dropped tasks so far: {self._dropped_tasks}. "
                    "Consider increasing max_workers or queue_max_size."
                )
            logger.debug(f"[Monitor] Queue load: {qsize}/{self.queue_max_size}")

    def submit_put_task(self, routing_indices: paddle.Tensor, rollout_id: str, layer_idx: int = None) -> None:
        """Submit a put task to the task queue"""
        if not self._sotre_process_running:
            raise RuntimeError("Store not started.")

        start_time = time.perf_counter()
        if layer_idx is not None:
            rdma_rollout_key = f"{rollout_id}_{layer_idx}"
        else:
            rdma_rollout_key = rollout_id

        routing_indices_np = routing_indices.numpy()

        task: StoreTask = {"task_type": "put", "key": rdma_rollout_key, "data": routing_indices_np}

        try:
            self._task_queue.put_nowait(task)
        except Exception:
            raise RuntimeError(f"Queue is FULL. Dropping put task for key: {rdma_rollout_key}. ")
        logger.info(f"[R3] Submit put task for key: {rdma_rollout_key}, cost time: {time.perf_counter()-start_time} s")

    def submit_clear_store_task(self) -> None:
        """Submit clear store task"""
        if not self._sotre_process_running:
            raise RuntimeError("Store not started.")

        start_time = time.perf_counter()
        task: StoreTask = {"task_type": "clear_store", "key": None, "data": None}

        try:
            self._task_queue.put_nowait(task)
            # Wait for the task to be processed
            self._task_queue.join()
        except Exception:
            raise RuntimeError("Queue is FULL. Dropping put task for key: clear_store. ")
        logger.info(f"[R3] Submit clear task, cost time: {time.perf_counter()-start_time} s")

    def submit_clear_prefix_batch_task(self, rollout_id) -> None:
        """Submit clear prefix batch task"""
        if not self._sotre_process_running:
            raise RuntimeError("Store not started.")
        prefix_batch = self.get_needed_clear_ids(rollout_id)

        if prefix_batch is None:
            return
        start_time = time.perf_counter()
        task: StoreTask = {"task_type": "clear_prefix_batch", "key": prefix_batch, "data": None}
        try:
            self._task_queue.put_nowait(task)
        except Exception:
            raise RuntimeError("Queue is FULL. Dropping put task for key: clear_store. ")
        logger.info(
            f"[R3] Submit clear prefix batch task for key: {prefix_batch}, cost time: {time.perf_counter()-start_time} s"
        )

    def get_needed_clear_ids(self, roullout_id: str) -> Optional[str]:
        """
        Generate the prefix IDs for all closed multi-round tasks.
        rollout_id: "xxx_xxx_epoch_15:2:2:1"
            example: xxx_xxx_data_id:gen_id:turn_id:segment_id
        """
        reversed_segment_id, reversed_turn_id, reversed_prefix_gen_id = roullout_id[::-1].split(":", 2)
        prefix_gen_id = reversed_prefix_gen_id[::-1]
        turn_id = eval(reversed_turn_id[::-1])
        segment_id = eval(reversed_segment_id[::-1])

        assert turn_id >= 0 and segment_id >= 0
        prefix_batch = None
        if turn_id > 0:
            prefix_batch = f"{prefix_gen_id}:{(turn_id-1)}:{segment_id}"
        return prefix_batch


class StoreTask(TypedDict):
    task_type: str
    key: str
    data: np.ndarray


class StoreProcess(Process):
    def __init__(self, task_queue: Queue, routing_replay_config: RoutingReplayConfig, max_model_len: int) -> None:
        super().__init__()
        self.max_model_len = max_model_len
        self._task_queue = task_queue
        self.routing_replay_config = routing_replay_config
        self.max_workers = 5
        self._closed = False

        # Note: _routing_store and _event_loop_thread must be initialized in run()
        # because they cannot be properly inherited after fork()
        self._routing_store = None
        self._event_loop_thread = None

    def run(self):
        logger.info(f"[R3] Start Running Store Wrapper in sub process {os.getpid()}")

        # Initialize routing store in subprocess
        self._routing_store = get_routing_store(routing_replay_config=self.routing_replay_config)

        # Initialize event loop thread in subprocess
        self._event_loop_thread = AsyncEventLoopThread()
        self._event_loop_thread.start()
        if not self._event_loop_thread._started_event.wait(timeout=5.0):
            raise RuntimeError("Failed to start async event loop thread in subprocess")

        clear_store_task = StoreTask({"task_type": "clear_store", "key": None, "data": None})
        self._task_queue.put_nowait(clear_store_task)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self._closed:
                try:
                    task = self._task_queue.get()
                    if task is None:  # Sentinel
                        self._task_queue.task_done()
                        break

                    if task["task_type"] == "put":
                        future = executor.submit(self.process_put_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                    elif task["task_type"] == "clear_store":
                        future = executor.submit(self.process_clear_store_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                    elif task["task_type"] == "clear_prefix_batch":
                        future = executor.submit(self.process_clear_prefix_batch_task, task)
                        future.add_done_callback(lambda f: self._task_queue.task_done())
                except Exception as e:
                    self._task_queue.task_done()
                    raise RuntimeError(f"Error during processing task. {e}")

        logger.info(f"[Consumer Process {Process.current_process().pid}] Shutdown.")

    def process_put_task(self, store_task: StoreTask) -> None:
        try:
            # TODO(gongshaotian): delete this after trainer support dynamic len
            store_task["data"] = self.pad_routing_indices(store_task["data"])
            coro_obj = self._routing_store.put(routing_key=store_task["key"], routing_indices=store_task["data"])
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error submitting put task: {e}")
            traceback.print_exc()
            raise

    def process_clear_store_task(self, store_task: StoreTask) -> None:
        try:
            coro_obj = self._routing_store.clear_store()
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error during processing clear store task. {e}")
            traceback.print_exc()
            raise

    def process_clear_prefix_batch_task(self, store_task: StoreTask) -> None:
        try:
            coro_obj = self._routing_store.clear_prefix_batch(routing_prefix_key=store_task["key"])
            future = self._event_loop_thread.submit_coroutine(
                coro_obj, callback=functools.partial(self._on_async_task_completed, store_task)
            )
            return future
        except Exception as e:
            logger.error(f"Error submitting clear_prefix_batch task: {e}")
            traceback.print_exc()
            raise

    def _on_async_task_completed(self, task, future):
        """ """
        try:
            # result = future.result()
            logger.info(f"[R3] Async task completed: {task['task_type']}, key: {task['key']}")
        except Exception as e:
            logger.error(f"[R3] Async task failed: {task['task_type']}, key: {task['key']}, error: {e}")
            traceback.print_exc()
            raise

    def close(self):
        """Close the store process"""
        self._closed = True
        if hasattr(self, "_event_loop_thread"):
            self._event_loop_thread.stop()

    def pad_routing_indices(self, routing_indices: np.ndarray) -> np.ndarray:
        """Pad routing indices of the request levevl to max model len"""
        routing_shape = routing_indices.shape
        if len(routing_shape) == 2:  # [token, topk]
            pad_array = np.full(
                shape=[(self.max_model_len - routing_indices.shape[0]), routing_indices.shape[1]],
                fill_value=-1,
                dtype=routing_indices.dtype,
            )
            return np.concatenate([routing_indices, pad_array], axis=0)

        elif len(routing_shape) == 3:  # [layer, token, topk]
            pad_array = np.full(
                shape=[
                    routing_indices.shape[0],
                    (self.max_model_len - routing_indices.shape[1]),
                    routing_indices.shape[2],
                ],
                fill_value=-1,
                dtype=routing_indices.dtype,
            )
            return np.concatenate([routing_indices, pad_array], axis=1)
        else:
            raise ValueError(f"Invalid routing indices shape: {routing_shape}")


class AsyncEventLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._loop = None
        self._started_event = threading.Event()
        self._closed = False

    def run(self):
        """Run the async event loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Set the event loop to be started
        self._started_event.set()
        logger.info("[EventLoopThread] Event loop started, running forever...")

        try:
            self._loop.run_forever()
            logger.info("[EventLoopThread] Event loop stopped")
        except Exception as e:
            logger.error(f"[EventLoopThread] Event loop exception: {e}")
            traceback.print_exc()
        finally:
            logger.info("[EventLoopThread] Closing event loop")
            self._loop.close()

    def submit_coroutine(self, coro, callback=None):
        """Thread safely submit coroutine to event loop"""
        if self._closed:
            raise RuntimeError("Event loop thread is closed")
        if not self._started_event.wait(timeout=5.0):
            raise RuntimeError("Event loop failed to start within 5 seconds")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        if callback:

            def wrapped_callback(f):
                try:
                    callback(f)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
                    traceback.print_exc()

            future.add_done_callback(wrapped_callback)
        return future

    def stop(self):
        """Stop the event loop"""
        if not self._closed:
            self._closed = True
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)


class RoutingStoreBase(ABC):
    """Base class for routing store"""

    def __init__(self, routing_replay_config: RoutingReplayConfig) -> None:
        self.routing_replay_config = routing_replay_config

    @abstractmethod
    async def put(self, routing_key: str, routing_indices: np.ndarray) -> None:
        """Put the routing indices into store"""
        raise NotImplementedError

    @abstractmethod
    async def clear_store(
        self,
    ):
        """Clear the routing indices store"""
        raise NotImplementedError

    @abstractmethod
    async def clear_prefix_batch(self, routing_prefix_key: str):
        """Clear the routing indices"""
        raise NotImplementedError


class RoutingStoreLocal(RoutingStoreBase):
    """Routing Store using local memory"""

    def __init__(self, routing_replay_config) -> None:
        super().__init__(routing_replay_config=routing_replay_config)
        self.local_store_dir = routing_replay_config.local_store_dir
        os.makedirs(self.local_store_dir, exist_ok=True)

    async def put(
        self,
        routing_key: str,
        routing_indices: np.ndarray,
    ) -> None:
        """Put the routing indices into store"""
        # TODO(gongshaotian) covert ./store_dir/routing_key/layer_id.pdtensor to ./store_dir/routing_key.pdtensor
        time_before_put = time.perf_counter()

        if len(routing_indices.shape) == 2:
            re_layer_id, re_rollout_id = routing_key[::-1].split("_", 1)
            rollout_id = re_rollout_id[::-1]
            layer_id = re_layer_id[::-1]
            request_path = os.path.join(self.local_store_dir, rollout_id)
            file_path = os.path.join(request_path, f"layer_{layer_id}.pdtensor")
        elif len(routing_indices.shape) == 3:
            request_path = os.path.join(self.local_store_dir, routing_key)
            file_path = os.path.join(request_path, f"{routing_key}.pdtensor")
        else:
            raise ValueError(f"Invalid routing indices shape: {routing_indices.shape}")

        paddle.save(routing_indices, file_path)
        logger.info(f"[R3] The routing key {routing_key} put cost is {time.perf_counter()-time_before_put}s")

    async def clear_store(self):
        """Clear the routing indices store"""
        if os.path.isdir(self.local_store_dir):
            shutil.rmtree(self.local_store_dir)

        logger.info("[R3] Clear routing store.")

    async def clear_prefix_batch(self, routing_prefix_key: str):
        """Clear the routing indices"""
        raise NotImplementedError


class RoutingStoreRDMA(RoutingStoreBase):
    """Routing Store using RDMA"""

    def __init__(self, routing_replay_config) -> None:
        super().__init__(routing_replay_config=routing_replay_config)
        try:
            # Only used in RLHF
            from p2pstore import P2PClient, P2PConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError(" RoutingStoreRDMA and p2pstore only support in RLHF. ")

        rdma_store_server = routing_replay_config.rdma_store_server
        p2pConfig = P2PConfig(metadata_server=rdma_store_server)
        self.p2p_client = P2PClient(p2pConfig)

    async def put(self, routing_key: str, routing_indices: np.ndarray) -> None:
        """Put the routing indices into store"""
        time_before_put = time.perf_counter()
        result = await self.p2p_client.put(routing_key, routing_indices)
        logger.info(f"[R3] The routing key {routing_key}, put cost is {time.perf_counter()-time_before_put}s")
        return result

    async def clear_prefix_batch(self, routing_prefix_key: str):
        time_before_clear = time.perf_counter()
        result = await self.p2p_client.delete_prefix_batch([routing_prefix_key])
        logger.info(
            f"[R3] The clear routing prefix key {routing_prefix_key}, cost is {time.perf_counter()-time_before_clear}s"
        )
        return result

    async def clear_store(self):
        """Clear the routing indices store"""
        time_before_clear = time.perf_counter()
        result = await self.p2p_client.clear()
        logger.info(f"[R3] Clear routing store cost is {time.perf_counter()-time_before_clear}s.")
        return result


def get_routing_store(routing_replay_config: RoutingReplayConfig) -> RoutingStoreBase:
    if routing_replay_config.routing_store_type == "local":
        return RoutingStoreLocal(routing_replay_config=routing_replay_config)
    elif routing_replay_config.routing_store_type == "rdma":
        return RoutingStoreRDMA(routing_replay_config=routing_replay_config)
    else:
        raise ValueError(
            f"Invalid routing store type: '{routing_replay_config.routing_store_type}'. "
            "Valid types are: 'local', 'rdma'"
        )
