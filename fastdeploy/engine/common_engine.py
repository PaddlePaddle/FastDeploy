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

from __future__ import annotations

import copy
import os
import threading
import time
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle
import zmq
from opentelemetry import trace

from fastdeploy.engine.request import Request, RequestOutput, RequestType
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import (
    EngineCacheQueue,
    EngineWorkerQueue,
    IPCSignal,
    ZmqIpcServer,
    ZmqTcpServer,
)
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.metrics.trace_util import start_span, start_span_request
from fastdeploy.model_executor.guided_decoding import schema_checker
from fastdeploy.plugins.token_processor import load_token_processor_plugins
from fastdeploy.splitwise.internal_adapter_utils import InternalAdapter
from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector
from fastdeploy.utils import EngineError, envs, get_logger, llm_logger

try:
    TokenProcessor = load_token_processor_plugins()
    llm_logger.info(f"TokenProcessor plugin {TokenProcessor} loaded")
except:
    from fastdeploy.output.token_processor import TokenProcessor


class EngineService:
    """
    Base class containing common engine functionality
    """

    def __init__(self, cfg, start_queue=True):
        """
        Initializes the LLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg
        if cfg.scheduler_config.splitwise_role != "mixed" or cfg.cache_config.enable_prefix_caching:
            if isinstance(self.cfg.cache_config.cache_queue_port, str):
                self.cfg.cache_config.cache_queue_port = self.cfg.cache_config.cache_queue_port.split(",")
            if isinstance(self.cfg.cache_config.cache_queue_port, list):
                self.cfg.cache_config.cache_queue_port = int(
                    self.cfg.cache_config.cache_queue_port[self.cfg.parallel_config.local_data_parallel_id]
                )

        if self.cfg.parallel_config.enable_expert_parallel:
            self.llm_logger = get_logger(
                "fastdeploy", f"fastdeploy_rank{self.cfg.parallel_config.local_data_parallel_id}.log"
            )
        else:
            self.llm_logger = llm_logger

        self.scheduler = cfg.scheduler_config.scheduler()
        self.enable_decode_cache_task = envs.FD_ENABLE_CACHE_TASK == "1"

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.resource_manager = ResourceManagerV1(
                cfg.scheduler_config.max_num_seqs,
                cfg,
                cfg.parallel_config.tensor_parallel_size,
                cfg.scheduler_config.splitwise_role,
                cfg.parallel_config.local_data_parallel_id,
            )
        else:
            self.resource_manager = ResourceManager(
                cfg.scheduler_config.max_num_seqs,
                cfg,
                cfg.parallel_config.tensor_parallel_size,
                cfg.scheduler_config.splitwise_role,
                cfg.parallel_config.local_data_parallel_id,
            )

        self.start_worker_queue_service(start_queue)

        os.environ["INFERENCE_MSG_QUEUE_ID"] = self.cfg.parallel_config.engine_worker_queue_port[
            self.cfg.parallel_config.local_data_parallel_id
        ]

        self.split_connector = SplitwiseConnector(cfg, self.engine_worker_queue, self.resource_manager)
        self.waiting_requests = []
        self.token_processor = TokenProcessor(
            cfg=cfg,
            cached_generated_tokens=self.scheduler,
            engine_worker_queue=self.engine_worker_queue,
            split_connector=self.split_connector,
        )
        self.token_processor.set_resource_manager(self.resource_manager)

        self.partial_chunked_tokens = [0] * (self.cfg.max_num_partial_prefills + 1)
        for idx in range(1, self.cfg.max_num_partial_prefills + 1):
            self.partial_chunked_tokens[idx] = (
                (self.cfg.scheduler_config.max_num_batched_tokens // idx)
                // self.cfg.cache_config.block_size
                * self.cfg.cache_config.block_size
            )

        self.guided_decoding_checker = None
        if self.cfg.structured_outputs_config.guided_decoding_backend != "off":
            self.guided_decoding_checker = schema_checker(
                self.cfg.structured_outputs_config.guided_decoding_backend,
                disable_any_whitespace=self.cfg.structured_outputs_config.disable_any_whitespace,
            )
        self._init_worker_monitor_signals()

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def start(self):
        self.running = True
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.insert_task_to_worker_thread = threading.Thread(target=self._scheduler_task_to_worker_v1, daemon=True)
        else:
            self.insert_task_to_worker_thread = threading.Thread(target=self._insert_task_to_worker, daemon=True)
        self.insert_task_to_worker_thread.start()
        self.token_processor.tasks_queue = self.engine_worker_queue
        self.token_processor.run()
        if self.cfg.scheduler_config.splitwise_role != "mixed":
            self.split_mode_get_tasks()

    def create_data_processor(self):
        self.input_processor = InputPreprocessor(
            self.cfg.model_config,
            self.cfg.structured_outputs_config.reasoning_parser,
            self.cfg.limit_mm_per_prompt,
            self.cfg.mm_processor_kwargs,
            self.cfg.tool_parser,
        )
        self.data_processor = self.input_processor.create_processor()

    def _init_worker_monitor_signals(self):  # exist_task_signal 用于各worker进程感知是否有新Task需要处理
        current_suffix = int(
            self.cfg.parallel_config.engine_worker_queue_port[self.cfg.parallel_config.local_data_parallel_id]
        )
        self.llm_logger.info(f"current_suffix: {current_suffix}")
        exist_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=exist_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # exist_swapped_task_signal 用于engine感知worker中是否存在swapped task
        exist_swapped_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=exist_swapped_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # exist_prefill_task_signal 用于各worker进程感知是否进行prefill
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # worker_live_signal 用于engine感知各worker进程是否存活，记录每个step 时间
        worker_healthy_live_recorded_time_array = np.zeros(
            shape=[min(self.cfg.worker_num_per_node, self.cfg.parallel_config.tensor_parallel_size)], dtype=np.int32
        )
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        cache_ready_signal_data = np.zeros(shape=[self.cfg.parallel_config.tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        swap_space_ready_signal_data = np.zeros(shape=[self.cfg.parallel_config.tensor_parallel_size], dtype=np.int32)
        self.swap_space_ready_signal = IPCSignal(
            name="swap_space_ready_signal",
            array=swap_space_ready_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        prefix_tree_status = np.zeros([1], dtype=np.int32)
        self.prefix_tree_status_signal = IPCSignal(
            name="prefix_tree_status",
            array=prefix_tree_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        kv_cache_status = np.zeros([1], dtype=np.int32)
        self.kv_cache_status_signal = IPCSignal(
            name="kv_cache_status",
            array=kv_cache_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

    def start_worker_queue_service(self, start_queue):
        """
        start queue service for engine worker communication
        """
        address = (
            self.cfg.master_ip,
            int(self.cfg.parallel_config.engine_worker_queue_port[self.cfg.parallel_config.local_data_parallel_id]),
        )

        if start_queue and (self.cfg.host_ip == self.cfg.master_ip or self.cfg.master_ip == "0.0.0.0"):
            self.llm_logger.info(f"Starting engine worker queue server service at {address}")
            self.engine_worker_queue_server = EngineWorkerQueue(
                address=address,
                is_server=True,
                num_client=self.cfg.parallel_config.tensor_parallel_size,
                local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
            )

            if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
                self.cache_task_queue = EngineCacheQueue(
                    address=(
                        self.cfg.master_ip,
                        self.cfg.cache_config.cache_queue_port,
                    ),
                    authkey=b"cache_queue_service",
                    is_server=True,
                    num_client=self.cfg.parallel_config.tensor_parallel_size,
                    client_id=-1,
                    local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                )
        self.llm_logger.info(
            f"local {min(self.cfg.worker_num_per_node * self.cfg.node_rank + self.cfg.parallel_config.local_data_parallel_id,self.cfg.parallel_config.data_parallel_size - 1)}"
        )
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            num_client=self.cfg.parallel_config.tensor_parallel_size,
            client_id=0,
            local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
            local_data_parallel_id=min(
                self.cfg.worker_num_per_node * self.cfg.node_rank + self.cfg.parallel_config.local_data_parallel_id,
                self.cfg.parallel_config.data_parallel_size - 1,
            ),
        )

    def insert_tasks(self, tasks, current_id=-1, allocated=False):
        """
        Insert tasks to engine.
        """
        for task in tasks:
            start_span_request("DEQUEUE", task, trace.SpanKind.CONSUMER)

        # TODO 返回至 scheduler
        if allocated:
            current_tasks = []
            for task in tasks:
                cur_task_idx = self.resource_manager.req_dict[task.request_id]
                del self.resource_manager.req_dict[task.request_id]
                cur_task = self.resource_manager.tasks_list[cur_task_idx]
                if envs.FD_ENABLE_INTERNAL_ADAPTER:
                    if not task.outputs.token_ids:  # first token is eos in Prefill, just recycle resource and continue
                        self.resource_manager.stop_flags[cur_task_idx] = True
                        self.resource_manager.tasks_list[cur_task_idx] = None
                        self.resource_manager._recycle_block_tables(cur_task)
                        if task.request_id in self.token_processor.tokens_counter:
                            del self.token_processor.tokens_counter[task.request_id]
                        self.llm_logger.warning(f"{task.request_id} need not decode after first token")
                        continue
                cur_task.prompt_token_ids[0] = task.outputs.token_ids[0]
                cur_task.num_cached_tokens = task.num_cached_tokens
                if (
                    self.cfg.speculative_config.method in ["mtp"]
                    and self.cfg.scheduler_config.splitwise_role == "decode"
                ):
                    cur_task.draft_token_ids = copy.deepcopy(task.outputs.draft_token_ids)
                if task.error_code != 200:
                    self.resource_manager.stop_flags[cur_task_idx] = True
                    self.resource_manager.tasks_list[cur_task_idx] = None
                    self.resource_manager._recycle_block_tables(cur_task)
                    if task.request_id in self.token_processor.tokens_counter:
                        del self.token_processor.tokens_counter[task.request_id]
                    self.scheduler.put_results([task])
                    self.llm_logger.warning(
                        f"{task.request_id} prefill failed with msg:{task.error_msg}, recycle resource."
                    )
                    continue
                self.token_processor.tokens_counter[task.request_id] = 1
                current_tasks.append(cur_task)
            if current_tasks:
                self.engine_worker_queue.put_tasks((current_tasks, self.resource_manager.real_bsz))
            return True

        self.resource_manager.check_and_free_block_tables()

        if not isinstance(tasks, list):
            tasks = [tasks]

        need_delete_tasks = []
        for task in tasks:
            if self.cfg.scheduler_config.splitwise_role != "mixed":
                status, msg = self.split_connector.check_decode_allocated(task)
                if not status:
                    self.llm_logger.error(f"{task.request_id} prefill failed with msg:{msg}.")
                    self.scheduler.put_results(
                        [
                            RequestOutput(
                                request_id=task.request_id,
                                finished=True,
                                error_code=500,
                                error_msg=msg,
                            )
                        ]
                    )
                    need_delete_tasks.append(task)
                    continue
        for tmp_task in need_delete_tasks:
            tasks.remove(tmp_task)

        for item in tasks:
            item.schedule_start_time = time.time()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            self.llm_logger.error(f"Inserting batch:{len(tasks)} exceeds the available batch:{available_batch}.")
            self.llm_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        req_ids = [t.request_id for t in tasks]

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)

        if not tasks:
            error_msg = f"The request required resources is exceed the limit, request id={req_ids}."
            self.llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=500)
            return False

        self.token_processor.number_of_tasks += len(tasks)

        is_decode = False
        is_prefill = False
        for i in range(len(tasks)):
            if tasks[i].disaggregate_info is not None:
                if tasks[i].disaggregate_info["role"] == "decode":
                    is_decode = True
                else:
                    is_prefill = True
            self.token_processor.number_of_input_tokens += tasks[i].prompt_token_ids_len

        self.split_connector.send_cache_infos(tasks, current_id)
        if not is_decode:
            self.llm_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
            for task in tasks:
                task.inference_start_time = time.time()
            if not is_prefill:
                if not self.cfg.model_config.enable_mm:
                    self.update_requests_chunk_size(tasks)
                else:
                    self.update_mm_requests_chunk_size(tasks)
            self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
            if is_prefill and self.cfg.scheduler_config.name != "splitwise":
                self.engine_worker_queue.available_prefill_instances.put(1)
        return True

    def task_is_finished(self, index):
        """
        judge if the task is finished
        """
        assert index < len(self.resource_manager.stop_flags)
        return self.resource_manager.stop_flags[index]

    def all_tasks_finished(self):
        """
        judge if all tasks are finished
        """
        return np.sum(self.resource_manager.stop_flags) == len(self.resource_manager.stop_flags)

    def update_requests_chunk_size(self, requests):
        """
        update each request's chunk size info
        """

        def update_tokens(idx, chunk_size, update_chunk=False):
            nonlocal remain_batched_tokens, chunk_request_num
            if update_chunk:
                requests_chunk[idx][-1] += chunk_size
            else:
                requests_chunk[idx].append(chunk_size)
            remain_batched_tokens -= chunk_size
            current_request_size[idx] -= chunk_size
            if current_request_size[idx] <= 0:
                chunk_request_num -= 1

        if not self.cfg.cache_config.enable_chunked_prefill or len(requests) == 0:
            return

        current_request_size = [request.prompt_token_ids_len for request in requests]
        requests_chunk = [[] for _ in range(len(requests))]
        chunk_request_num = len(current_request_size)
        while chunk_request_num >= 1:
            remain_batched_tokens = self.cfg.scheduler_config.max_num_batched_tokens
            for idx in range(len(current_request_size)):
                if current_request_size[idx] <= 0:
                    continue
                chunk_size = min(
                    current_request_size[idx],
                    self.partial_chunked_tokens[chunk_request_num],
                )
                update_tokens(idx, chunk_size)

            while remain_batched_tokens >= self.cfg.cache_config.block_size:
                # 当前 max_num_batched_tokens 还有剩余时，优先分配给较短的请求
                waiting_requests = [input_lens for input_lens in current_request_size if input_lens > 0]
                if len(waiting_requests) == 0:
                    break

                available_tokens = (
                    remain_batched_tokens // self.cfg.cache_config.block_size * self.cfg.cache_config.block_size
                )
                append_idx = current_request_size.index(min(waiting_requests))
                chunk_size = min(
                    current_request_size[append_idx],
                    self.partial_chunked_tokens[chunk_request_num],
                    available_tokens,
                )
                update_tokens(append_idx, chunk_size, update_chunk=True)

        for idx in range(len(requests)):
            requests[idx].set("prefill_chunk_info", requests_chunk[idx])

    def update_mm_requests_chunk_size(self, requests):
        """
        update each multimodal request's chunk size info
        """
        if not self.cfg.cache_config.enable_chunked_prefill or len(requests) == 0:
            return

        for request in requests:
            inputs = request.multimodal_inputs
            # 兼容没有图片和视频的情况
            if inputs["images"] is None:
                inputs["image_type_ids"] = np.array([], dtype="int32")
                inputs["grid_thw"] = np.array([], dtype="int64")
                inputs["images"] = np.array([], dtype="uint8")
            input_ids = paddle.to_tensor(inputs["input_ids"], dtype="int64")
            image_type_ids = paddle.to_tensor(inputs["image_type_ids"], dtype="int32")
            image_mask = input_ids == self.data_processor.image_patch_id
            image_token_sum = paddle.full(shape=[len(input_ids) + 1], fill_value=0, dtype="int32")
            image_token_sum[1:] = paddle.cumsum(image_mask.cast("int32"), dtype="int32")
            grid_thw = []
            for one in inputs["grid_thw"]:
                if one[0] == 1:
                    grid_thw.append(one)
                else:
                    grid_thw.extend([[2, one[1], one[2]]] * (one[0] // 2))
            grid_thw = paddle.to_tensor(grid_thw, dtype="int64")

            from fastdeploy.model_executor.ops.gpu import get_mm_split_fuse

            chunk_image_num, chunk_seq_len = get_mm_split_fuse(
                input_ids,
                image_type_ids,
                image_token_sum,
                grid_thw,
                self.data_processor.image_patch_id,
                len(grid_thw),
                0,
                len(input_ids),
                0,
                self.partial_chunked_tokens[1],
                2048,
            )

            grid_thw = grid_thw.numpy().reshape([-1, 3])
            num_chunks = len(chunk_image_num)
            chunks_info = []
            input_ids_st, image_type_ids_st, grid_thw_st, patch_st = 0, 0, 0, 0
            for idx in range(num_chunks):
                chunk_input_ids = inputs["input_ids"][input_ids_st : input_ids_st + chunk_seq_len[idx]]
                chunk_token_type_ids = inputs["token_type_ids"][input_ids_st : input_ids_st + chunk_seq_len[idx]]
                actual_image_num = np.sum(grid_thw[grid_thw_st : grid_thw_st + chunk_image_num[idx], 0])
                chunk_image_type_ids = inputs["image_type_ids"][
                    image_type_ids_st : image_type_ids_st + actual_image_num
                ]
                chunk_grid_thw = grid_thw[grid_thw_st : grid_thw_st + chunk_image_num[idx]]
                chunk_patch_num = np.sum(np.prod(chunk_grid_thw, axis=1))
                chunk_images = inputs["images"][patch_st : patch_st + chunk_patch_num]

                chunks_info.append(
                    {
                        "input_ids": chunk_input_ids,
                        "token_type_ids": chunk_token_type_ids,
                        "image_type_ids": (chunk_image_type_ids if chunk_image_type_ids.shape[0] else None),
                        "grid_thw": (chunk_grid_thw if chunk_grid_thw.shape[0] else None),
                        "images": (chunk_images if chunk_images.shape[0] else None),
                        "position_ids": None,
                    }
                )

                input_ids_st += chunk_seq_len[idx]
                image_type_ids_st += actual_image_num
                grid_thw_st += chunk_image_num[idx]
                patch_st += chunk_patch_num
            request.set("prefill_chunk_info", chunks_info)

    def _insert_task_to_worker(self):
        """
        Insert task to engine thread, monitor scheduler request queue.
        if the engine has resource, insert task to engine
        """
        current_id = 0
        while getattr(self, "running", True):
            try:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue
                if hasattr(self, "exist_prefill_task_signal") and self.exist_prefill_task_signal.value[0] > 0:
                    if (
                        self.cfg.scheduler_config.splitwise_role == "mixed"
                        or self.split_connector.has_splitwise_tasks()
                    ):
                        time.sleep(0.005)
                        continue
                if self.engine_worker_queue.num_cache_infos() > 0:
                    time.sleep(0.001)
                    continue
                if len(self.split_connector.current_request_ids) > 0:
                    time.sleep(0.001)
                    continue

                num_prefill_batch = min(
                    int(self.resource_manager.available_batch()),
                    self.cfg.max_prefill_batch,
                )

                self.resource_manager.check_and_free_block_tables()
                tasks = self.scheduler.get_requests(
                    available_blocks=self.resource_manager.available_block_num(),
                    block_size=self.cfg.cache_config.block_size,
                    reserved_output_blocks=self.cfg.cache_config.enc_dec_block_num,
                    max_num_batched_tokens=self.cfg.scheduler_config.max_num_batched_tokens,
                    batch=num_prefill_batch,
                )

                if len(tasks) == 0:
                    time.sleep(0.001)
                    continue

                if self.cfg.scheduler_config.splitwise_role != "mixed":
                    self.llm_logger.info("Inserting splitwise tasks")
                    self.split_connector.send_splitwise_tasks(tasks, current_id)

                insert_successful = self.insert_tasks(tasks, current_id)
                if insert_successful:
                    current_id = current_id + 1
                else:
                    continue

                main_process_metrics.num_requests_waiting.dec(len(tasks))
                main_process_metrics.num_requests_running.inc(len(tasks))
            except Exception as e:
                err_msg = f"Error happend while insert task to engine: {e}, {traceback.format_exc()!s}."
                self.llm_logger.error(err_msg)

    def _scheduler_task_to_worker_v1(self):
        """
        Insert tasks to worker with scheduler v1 (ENABLE_V1_KVCACHE_SCHEDULER=1).
        """
        get_request_pool = ThreadPoolExecutor(max_workers=1)
        is_fetching = False

        def _fetch_request():
            try:
                nonlocal is_fetching
                is_fetching = True
                num_prefill_batch = min(
                    int(self.resource_manager.available_batch()),
                    self.cfg.max_prefill_batch,
                )
                if self.cfg.model_config.enable_mm:
                    available_blocks = self.resource_manager.available_block_num()
                else:
                    available_blocks = self.cfg.cache_config.max_block_num_per_seq

                tasks = self.scheduler.get_requests(
                    available_blocks=available_blocks,
                    block_size=self.cfg.cache_config.block_size,
                    reserved_output_blocks=self.cfg.cache_config.enc_dec_block_num,
                    max_num_batched_tokens=self.cfg.model_config.max_model_len,
                    batch=num_prefill_batch,
                )
                if self.cfg.scheduler_config.splitwise_role != "mixed":
                    for task in tasks:
                        # assure can allocate block ids in P
                        while not self.resource_manager.preallocate_resource_in_p(task):
                            time.sleep(0.005)
                        self.llm_logger.info(f"ask D resource for req_id: {task.request_id}")
                        self.split_connector.send_splitwise_tasks([task], task.idx)
                    need_delete_tasks = []
                    for task in tasks:
                        if self.cfg.scheduler_config.splitwise_role != "mixed":
                            # assure fetch block ids from D
                            status, msg = self.split_connector.check_decode_allocated(task)
                            if not status:
                                self.llm_logger.error(f"{task.request_id} prefill failed with msg:{msg}.")
                                self.scheduler.put_results(
                                    [
                                        RequestOutput(
                                            request_id=task.request_id,
                                            finished=True,
                                            error_code=500,
                                            error_msg=msg,
                                        )
                                    ]
                                )
                                need_delete_tasks.append(task)
                                continue
                    for tmp_task in need_delete_tasks:
                        tasks.remove(tmp_task)
                        # release resource in P
                        self.resource_manager.prerelease_resource(tmp_task)
                if self.cfg.scheduler_config.splitwise_role == "prefill":
                    # to send cache info to cache messager
                    if tasks:
                        self.split_connector.send_cache_infos(tasks, 0)
                        # ensure cache tasks has sent to cache_messager
                        need_check_req_ids = [task.request_id for task in tasks]
                        while need_check_req_ids:
                            req_ids = self.engine_worker_queue.get_finished_add_cache_task_req()
                            self.llm_logger.info(f"get_finished_add_cache_task_req: {req_ids}")
                            if req_ids:
                                for req_id in req_ids:
                                    assert req_id in need_check_req_ids
                                    need_check_req_ids.remove(req_id)
                            else:
                                time.sleep(0.001)
                # Fetch requests and add them to the scheduling queue
                if tasks:
                    if self.cfg.scheduler_config.splitwise_role == "prefill":
                        self.resource_manager.add_request_in_p(tasks)
                    else:
                        for task in tasks:
                            self.resource_manager.add_request(task)
                is_fetching = False
            except Exception as e:
                self.llm_logger.error(f"fetching request error {e} {str(traceback.format_exc())}")
                is_fetching = False

        while self.running:
            try:
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue
                if self.cfg.scheduler_config.splitwise_role != "mixed":
                    if not is_fetching:
                        get_request_pool.submit(_fetch_request)

                else:
                    if (
                        len(self.resource_manager.waiting) == 0
                        and (not is_fetching)
                        and self.exist_prefill_task_signal.value[0] == 0
                    ):
                        get_request_pool.submit(_fetch_request)
                # 2. Schedule requests
                tasks = self.resource_manager.schedule()
                # 3. Send to engine
                if tasks:
                    if self.cfg.scheduler_config.splitwise_role == "decode":
                        for task in tasks:
                            if task.task_type == RequestType.PREEMPTED:
                                msg = f"{task.request_id} decode not enough blocks, need to be rescheduled."
                                self.llm_logger.error(msg)
                                self.scheduler.put_results(
                                    [
                                        RequestOutput(
                                            request_id=task.request_id,
                                            finished=True,
                                            error_code=500,
                                            error_msg=msg,
                                        )
                                    ]
                                )
                    self.resource_manager.get_real_bsz()
                    self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
                else:
                    time.sleep(0.005)

            except Exception as e:
                err_msg = "Error happend while insert task to engine: {}, {}.".format(e, str(traceback.format_exc()))
                self.llm_logger.error(err_msg)

    def start_zmq_service(self, api_server_pid=None):
        if api_server_pid is None:
            return
        self.api_server_pid = api_server_pid
        if envs.FD_ENABLE_INTERNAL_ADAPTER:
            self.recv_request_server = ZmqTcpServer(port=envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT, mode=zmq.PULL)
            self.send_response_server = ZmqTcpServer(port=envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT, mode=zmq.ROUTER)
            self.internal_adapter = InternalAdapter(
                cfg=self.cfg, engine=self, dp_rank=self.cfg.node_rank * self.cfg.worker_num_per_node
            )
        else:
            self.recv_request_server = ZmqIpcServer(name=api_server_pid, mode=zmq.PULL)
            self.send_response_server = ZmqIpcServer(name=api_server_pid, mode=zmq.ROUTER)
        self.recv_result_handle_thread = threading.Thread(
            target=self.send_response_server.recv_result_handle, daemon=True
        )
        self.recv_result_handle_thread.start()
        time.sleep(3)
        self.insert_task_to_scheduler_thread = threading.Thread(target=self._insert_zmq_task_to_scheduler, daemon=True)
        self.insert_task_to_scheduler_thread.start()

        self.receive_output_thread = threading.Thread(target=self._zmq_send_generated_tokens, daemon=True)
        self.receive_output_thread.start()

    def _insert_zmq_task_to_scheduler(self):
        added_requests: Dict[str, int] = dict()
        if envs.FD_ENABLE_INTERNAL_ADAPTER:
            if self.cfg.scheduler_config.splitwise_role == "decode":
                return
        while self.running:
            try:
                block = True if len(added_requests) == 0 else False
                if not self.cfg.model_config.enable_mm:
                    err, data = self.recv_request_server.receive_json_once(block)
                else:
                    err, data = self.recv_request_server.receive_pyobj_once(block)
                if err is not None:
                    self.llm_logger.error(f"Engine stops inserting zmq task into scheduler, err:{err}")
                    break

                request, insert_task = None, []
                results: List[Tuple[str, Optional[str]]] = list()
                if data:
                    err_msg = None
                    try:
                        request = Request.from_dict(data)
                        start_span("ENQUEUE_ZMQ", data, trace.SpanKind.PRODUCER)
                        main_process_metrics.requests_number.inc()
                        self.llm_logger.debug(f"Receive request: {request}")
                    except Exception as e:
                        self.llm_logger.error(f"Receive request error: {e}, {traceback.format_exc()!s}")
                        err_msg = str(e)
                        results.append((data["request_id"], err_msg))

                    if self.guided_decoding_checker is not None and err_msg is None:
                        request, err_msg = self.guided_decoding_checker.schema_format(request)
                        if err_msg is not None:
                            self.llm_logger.error(f"Receive request error: {err_msg}")
                            results.append((request.request_id, err_msg))

                    if err_msg is None:
                        insert_task.append(request)

                response = self.scheduler.put_requests(insert_task)
                results.extend(response)

                if request:
                    if request.request_id not in added_requests:
                        added_requests[request.request_id] = 0
                    added_requests[request.request_id] += 1

                for request_id, failed in results:
                    if request_id in added_requests:
                        added_requests[request_id] -= 1
                        if added_requests[request_id] == 0:
                            added_requests.pop(request_id)

                    if failed is None:
                        main_process_metrics.num_requests_waiting.inc(1)
                        continue

                    error_result = RequestOutput(
                        request_id=request_id,
                        finished=True,
                        error_code=500,
                        error_msg=failed,
                    )
                    # Since the request is not in scheduler
                    # Send result by zmq directly
                    self.send_response_server.send_response(request_id, [error_result])
            except Exception as e:
                self.llm_logger.error(
                    f"Error happened while receiving new request from zmq, details={e}, "
                    f"traceback={traceback.format_exc()}"
                )

    def _decode_token(self, token_ids, req_id, is_end):
        delta_text = ""
        if envs.FD_ENABLE_RETURN_TEXT:
            delta_text, cum_tokens, _ = self.data_processor.ids2tokens(token_ids, req_id)
            if delta_text != "":
                prefix_offset = self.data_processor.decode_status[req_id][0]
                read_offset = self.data_processor.decode_status[req_id][1]
                token_ids = cum_tokens[prefix_offset:read_offset]
            else:
                token_ids = []
            if is_end:
                del self.data_processor.decode_status[req_id]
        return delta_text, token_ids

    def _zmq_send_generated_tokens(self):
        """
        Recieve output for zmq
        """
        while self.running:
            try:
                results = self.scheduler.get_results()
                if len(results) == 0:
                    time.sleep(0.005)
                    continue
                for request_id, contents in results.items():
                    new_contents = []
                    for content in contents:
                        decode_type = content.outputs.decode_type
                        delta_text = ""
                        if decode_type == 0:
                            delta_text, token_ids = self._decode_token(
                                token_ids=content.outputs.token_ids, req_id=request_id, is_end=content.finished
                            )
                        else:
                            token_ids = content.outputs.token_ids
                        if len(token_ids):
                            content.outputs.token_ids = token_ids
                            content.outputs.text = delta_text
                            new_contents.append(content)
                        elif content.finished:
                            new_contents.append(content)
                        else:
                            llm_logger.warning(
                                f"current tokens need to accumulate, req_id: {request_id} {content.outputs.token_ids}"
                            )
                    if len(new_contents):
                        llm_logger.info(f"Send response for request id: {request_id}")
                        self.send_response_server.send_response(request_id, new_contents)
            except Exception as e:
                llm_logger.error(f"Unexcepted error happend: {e}, {traceback.format_exc()!s}")

    def split_mode_get_tasks(self):
        """
        Split mode get tasks
        """

        def receiver_loop():
            while self.running:
                try:

                    processed_indices = []
                    for idx, task in enumerate(self.waiting_requests):
                        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                            if self.resource_manager.preallocate_resource_in_d(task):
                                self.llm_logger.info(f"Resource available, processing task {task.request_id}")
                                self.split_connector.send_cache_infos([task], -1)
                                processed_indices.append(idx)
                            else:
                                self.llm_logger.debug(f"Still waiting for resources {task.request_id}")
                                break
                        else:
                            if self.resource_manager.is_resource_sufficient(task.prompt_token_ids_len):
                                self.insert_tasks([task])
                                self.llm_logger.info(f"Resource available, processing task {task.request_id}")
                                processed_indices.append(idx)
                            else:
                                self.llm_logger.debug(f"Still waiting for resources {task.request_id}")
                                break

                    for idx in sorted(processed_indices, reverse=True):
                        self.waiting_requests.pop(idx)

                    if not self.engine_worker_queue.disaggregate_queue_empty():
                        items = self.engine_worker_queue.get_disaggregated_tasks()
                        for item in items:
                            role = item[0]
                            tasks = item[1]

                            if role == "prefill":
                                for task in tasks:
                                    task.max_tokens = task.min_tokens = 2
                                self.insert_tasks(tasks)

                            elif role == "decode":
                                if hasattr(tasks[0], "finished"):
                                    if not isinstance(tasks, list):
                                        tasks = [tasks]
                                    for task in tasks:
                                        task.finished = False
                                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                                        for task in tasks:
                                            if envs.FD_ENABLE_INTERNAL_ADAPTER:
                                                if (
                                                    not task.outputs.token_ids
                                                ):  # first token is eos in Prefill, just recycle resource and continue
                                                    cur_task = self.resource_manager.requests[task.request_id]
                                                    self.resource_manager.stop_flags[cur_task.idx] = True
                                                    self.resource_manager.tasks_list[cur_task.idx] = None
                                                    self.resource_manager._free_blocks(cur_task)
                                                    if cur_task.request_id in self.token_processor.tokens_counter:
                                                        del self.token_processor.tokens_counter[task.request_id]
                                                    self.llm_logger.warning(
                                                        f"{task.request_id} need not decode after first token"
                                                    )
                                                    del self.resource_manager.requests[task.request_id]
                                                    del self.resource_manager.req_dict[task.request_id]
                                                    continue
                                            if task.error_code != 200:
                                                cur_task = self.resource_manager.requests[task.request_id]
                                                self.resource_manager.stop_flags[cur_task.idx] = True
                                                self.resource_manager.tasks_list[cur_task.idx] = None
                                                self.resource_manager._free_blocks(cur_task)
                                                if cur_task.request_id in self.token_processor.tokens_counter:
                                                    del self.token_processor.tokens_counter[task.request_id]
                                                self.scheduler.put_results([task])
                                                self.llm_logger.warning(
                                                    f"{task.request_id} prefill failed with msg:{task.error_msg}, recycle resource."
                                                )
                                                continue
                                            self.resource_manager.insert_task_for_decoding(task)

                                    else:
                                        self.insert_tasks(tasks, allocated=True)
                                        if self.cfg.innode_prefill_ports is not None:
                                            self.scheduler.put_results(tasks)
                                else:
                                    if len(self.waiting_requests):
                                        self.llm_logger.info(f"Waiting for resource for task {tasks[0].request_id}")
                                        self.waiting_requests.extend(tasks)
                                    else:
                                        new_waiting = []
                                        for task in tasks:
                                            can_allocate_resource = False
                                            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                                                if self.resource_manager.preallocate_resource_in_d(task):
                                                    self.split_connector.send_cache_infos([task], -1)
                                                    can_allocate_resource = True
                                            else:
                                                if self.resource_manager.is_resource_sufficient(
                                                    task.prompt_token_ids_len
                                                ):
                                                    self.insert_tasks([task])
                                                    can_allocate_resource = True
                                            if can_allocate_resource is False:
                                                if not self.enable_decode_cache_task:
                                                    task.error_msg = "Not enough resources"
                                                new_waiting.append(task)

                                        if new_waiting:
                                            if not self.enable_decode_cache_task:
                                                self.split_connector.send_cache_infos(new_waiting, -1)
                                            else:
                                                self.waiting_requests.extend(new_waiting)
                                                self.llm_logger.info(
                                                    f"Added {len(new_waiting)} tasks to waiting queue"
                                                )

                    else:
                        time.sleep(0.001)

                except Exception as e:
                    self.llm_logger.error(f"Error in main loop: {e}")
                    time.sleep(0.1)

        threading.Thread(target=receiver_loop, daemon=True).start()

    def start_cache_service(self, device_ids, ipc_signal_suffix):
        return self.resource_manager.cache_manager.launch_cache_manager(
            cache_config=self.cfg.cache_config,
            tensor_parallel_size=self.cfg.parallel_config.tensor_parallel_size,
            device_ids=device_ids,
            pod_ip=self.cfg.master_ip,
            engine_worker_queue_port=int(
                self.cfg.parallel_config.engine_worker_queue_port[self.cfg.parallel_config.local_data_parallel_id]
            ),
            pid_suffix=ipc_signal_suffix,
            create_cache_tensor=False,
        )

    def check_and_free_block_tables(self):
        self.resource_manager.check_and_free_block_tables()

    def clear_data(self):
        try:
            llm_logger.info("Clear Data: Start")
            self.token_processor.clear_data()
            self.engine_worker_queue.clear_data()
            self.send_response_server.req_dict.clear()
            self.recv_request_server.req_dict.clear()
            llm_logger.info("Clear Data: Successfully")
            return True
        except Exception as e:
            llm_logger.error(f"Clear data error: {e}")
            return False

    def _exit_sub_services(self):
        """
        exit sub services
        """
        self.running = False
        self.engine_worker_queue_server.cleanup()
        self.exist_task_signal.clear()
        self.exist_swapped_task_signal.clear()
        self.worker_healthy_live_signal.clear()
        self.cache_ready_signal.clear()
        self.swap_space_ready_signal.clear()
        self.exist_prefill_task_signal.clear()
        self.model_weights_status_signal.clear()
        self.prefix_tree_status_signal.clear()
        self.kv_cache_status_signal.clear()
        if hasattr(self, "send_response_server") and self.send_response_server is not None:
            self.send_response_server.close()
        if hasattr(self, "recv_request_server") and self.recv_request_server is not None:
            self.recv_request_server.close()
        if hasattr(self, "recv_control_cmd_server") and self.recv_control_cmd_server is not None:
            self.recv_control_cmd_server.close()
