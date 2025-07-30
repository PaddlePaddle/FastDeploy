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
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle
import zmq
from opentelemetry import trace
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.expert_service import start_expert_service
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import (
    EngineCacheQueue,
    EngineWorkerQueue,
    IPCSignal,
    ZmqClient,
)
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.metrics.trace_util import start_span, start_span_request
from fastdeploy.model_executor.guided_decoding import schema_checker
from fastdeploy.output.token_processor import TokenProcessor, WarmUpTokenProcessor
from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector
from fastdeploy.utils import EngineError, console_logger, envs, llm_logger


class LLMEngine:
    """
    Engine class responsible for managing the Large Language Model (LLM) operations.

    Attributes:
        cfg (Config): Configuration object containing all the parameters.
        cached_generated_tokens (queue.Queue): Queue to store generated tokens.
        scheduler (LocalScheduler or GlobalScheduler): Scheduling tasks.
        input_processor (InputPreprocessor): Preprocessor for input data.
        resource_manager (ResourceManager): Manager for resource allocation.
        token_processor (TokenProcessor): Processor for token generation.
        engine_worker_queue (EngineWorkerQueue): Queue for communication between engine and workers.
        is_started (bool): Flag indicating if the engine has started.
        do_profile (int): Flag indicating if profiling is enabled.
    """

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """
        Creates an LLM engine from the provided engine arguments.

        Args:
            engine_args (EngineArgs): Engine arguments object.

        Returns:
            LLMEngine: Instance of the LLMEngine class.
        """
        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the LLMEngine.
        return cls(cfg=config)

    def __init__(self, cfg):
        """
        Initializes the LLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg
        self.running = True
        self.scheduler = cfg.scheduler_config.scheduler()

        self.input_processor = InputPreprocessor(
            cfg.tokenizer,
            cfg.reasoning_parser,
            cfg.limit_mm_per_prompt,
            cfg.mm_processor_kwargs,
            cfg.enable_mm,
        )

        self.start_queue_service()

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.resource_manager = ResourceManagerV1(
                cfg.max_num_seqs, cfg, cfg.tensor_parallel_size, cfg.splitwise_role
            )
            if cfg.splitwise_role != "mixed":
                raise NotImplementedError(
                    "Currently ENABLE_V1_KVCACHE_SCHEDULER=1 only supported in mixed sampling now."
                )
        else:
            self.resource_manager = ResourceManager(
                cfg.max_num_seqs, cfg, cfg.tensor_parallel_size, cfg.splitwise_role
            )

        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(self.cfg.engine_worker_queue_port)

        self.split_connector = SplitwiseConnector(cfg, self.scheduler, self.engine_worker_queue, self.resource_manager)

        self.token_processor = TokenProcessor(
            cfg=self.cfg,
            cached_generated_tokens=self.scheduler,
            engine_worker_queue=self.engine_worker_queue,
            split_connector=self.split_connector,
        )
        self.token_processor.set_resource_manager(self.resource_manager)

        self.is_started = False

        self.waiting_requests = []

        if self.cfg.cache_config.num_gpu_blocks_override is None:
            self.do_profile = 1
        else:
            self.do_profile = 0

        self.partial_chunked_tokens = [0] * (self.cfg.max_num_partial_prefills + 1)
        for idx in range(1, self.cfg.max_num_partial_prefills + 1):
            self.partial_chunked_tokens[idx] = (
                (self.cfg.max_num_batched_tokens // idx)
                // self.cfg.cache_config.block_size
                * self.cfg.cache_config.block_size
            )
            self.partial_chunked_tokens[idx] = max(1, self.partial_chunked_tokens[idx])

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

        self.guided_decoding_checker = None
        if self.cfg.guided_decoding_backend != "off":
            self.guided_decoding_checker = schema_checker(
                self.cfg.guided_decoding_backend,
                disable_any_whitespace=self.cfg.disable_any_whitespace,
            )

    def start(self, api_server_pid=None):
        """
        Initializes the engine and starts its sub-services.
        If `api_server_pid` is defined, will launch a thread
        to keep getting request from zmq_server.
        """
        assert not self.is_started, "The engine is already started."
        start_time = time.time()

        self.api_server_pid = api_server_pid
        self.engine_pid = os.getpid()
        self.ipc_signal_suffix = self.engine_pid if self.api_server_pid is None else self.api_server_pid
        self._init_worker_signals()

        self.data_processor = self.input_processor.create_processor()

        if api_server_pid is not None:
            self.zmq_server = ZmqClient(name=api_server_pid, mode=zmq.PULL)
            self.zmq_server.start_server()
            self.zmq_server.create_router()
            time.sleep(3)

        if self.do_profile == 0 and (
            self.cfg.cache_config.enable_prefix_caching or self.cfg.splitwise_role != "mixed"
        ):
            device_ids = self.cfg.device_ids.split(",")
            self.cache_manager_processes = self.resource_manager.cache_manager.launch_cache_manager(
                cache_config=self.cfg.cache_config,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                device_ids=device_ids,
                pod_ip=self.cfg.master_ip,
                engine_worker_queue_port=self.cfg.engine_worker_queue_port,
                pid_suffix=self.ipc_signal_suffix,
            )
            self.launched_cache_manager_signal.value[0] = 1

        self.worker_proc = self._start_worker_service()
        console_logger.info("Waitting worker processes ready...")
        time.sleep(5)
        self.worker_init_status = dict()
        if not self.check_worker_initialize_status():
            console_logger.error("Failed to launch worker processes, check log/workerlog.* for more details.")
            return False

        # Start warmup if enabled
        if self.cfg.use_warmup:
            console_logger.info("Starting warmup")
            self._set_warmup_token_processor()
            self.warmup()
            self._del_warmup_token_processor()
            console_logger.info("Warmup finished")

        self.token_processor.tasks_queue = self.engine_worker_queue

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.insert_task_to_worker_thread = threading.Thread(target=self._scheduler_task_to_worker_v1, daemon=True)
        else:
            self.insert_task_to_worker_thread = threading.Thread(target=self._insert_task_to_worker, daemon=True)
        self.insert_task_to_worker_thread.start()

        if self.api_server_pid is not None:
            self.insert_task_to_scheduler_thread = threading.Thread(
                target=self._insert_zmq_task_to_scheduler, daemon=True
            )
            self.insert_task_to_scheduler_thread.start()

            self.receive_output_thread = threading.Thread(target=self._zmq_send_generated_tokens, daemon=True)
            self.receive_output_thread.start()

        # Start TokenProcessor thread
        self.token_processor.run()

        if self.cfg.splitwise_role != "mixed":
            # 单机逻辑
            self.engine_worker_queue.available_prefill_instances.put(1)
            self.split_mode_get_tasks()
            if self.cfg.scheduler_config.name == "splitwise":
                self.splitwise_receive_thread = threading.Thread(target=self.split_connector.start_receiver, args=())
                self.splitwise_receive_thread.daemon = True
                self.splitwise_receive_thread.start()

            self.cfg.init_cache_info()

            role = self.cfg.splitwise_role
            host_ip = self.cfg.host_ip
            disaggregate = self.cfg.disaggregate_info
            if self.cfg.scheduler_config.name == "splitwise":
                self.scheduler.start(role, host_ip, disaggregate)

            time.sleep(1)

            if self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1:
                self.dp_processed = []
                for i in range(
                    1,
                    self.cfg.parallel_config.data_parallel_size // self.cfg.nnode,
                ):
                    time.sleep(1)
                    self.dp_processed.append(
                        multiprocessing.Process(
                            target=start_expert_service,
                            args=(
                                self.cfg,
                                i + self.cfg.node_rank * self.cfg.worker_num_per_node,
                                self.ipc_signal_suffix,
                            ),
                        )
                    )
                    llm_logger.info(
                        f"Engine is initialized successfully with {self.cfg.tensor_parallel_size}"
                        + f" data parallel id {i}"
                    )
                    self.dp_processed[-1].start()

        console_logger.info(f"Worker processes are launched with {time.time() - start_time} seconds.")
        return True

    def _zmq_send_generated_tokens(self):
        """
        Recieve output for zmq
        """
        assert self.api_server_pid is not None
        while self.running:
            try:
                results = self.scheduler.get_results()
                if len(results) == 0:
                    time.sleep(0.005)
                    continue
                for request_id, contents in results.items():
                    self.zmq_server.send_multipart(request_id, contents)

            except Exception as e:
                llm_logger.error(f"Unexcepted error happend: {e}, {traceback.format_exc()!s}")

    def _get_generated_result(self):
        """
        Get result from scheduler, this function is called by generate()
        which is only used in offline inference.
        """
        return self.scheduler.get_results()

    def _insert_task_to_worker(self):
        """
        Insert task to engine thread, monitor scheduler request queue.
        if the engine has resource, insert task to engine
        """
        current_id = -1
        while self.running:
            try:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue
                if self.exist_prefill_task_signal.value[0] > 0:
                    if self.cfg.splitwise_role == "mixed" or self.split_connector.has_splitwise_tasks():
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
                    max_num_batched_tokens=self.cfg.max_num_batched_tokens,
                    batch=num_prefill_batch,
                )

                if len(tasks) == 0:
                    time.sleep(0.001)
                    continue

                current_id = (current_id + 1) % 100003
                if self.cfg.splitwise_role != "mixed":
                    llm_logger.info("Inserting splitwise tasks")
                    self.split_connector.send_splitwise_tasks(tasks, current_id)

                self.insert_tasks(tasks, current_id)

                main_process_metrics.num_requests_waiting.dec(len(tasks))
                main_process_metrics.num_requests_running.inc(len(tasks))
            except Exception as e:
                err_msg = f"Error happend while insert task to engine: {e}, {traceback.format_exc()!s}."
                llm_logger.error(err_msg)

    def _scheduler_task_to_worker_v1(self):
        """
        Insert tasks to worker with scheduler v1 (ENABLE_V1_KVCACHE_SCHEDULER=1).
        """
        get_request_pool = ThreadPoolExecutor(max_workers=1)
        is_fetching = False

        def _fetch_request():
            nonlocal is_fetching
            is_fetching = True
            num_prefill_batch = min(
                int(self.resource_manager.available_batch()),
                self.cfg.max_prefill_batch,
            )
            tasks = self.scheduler.get_requests(
                available_blocks=self.resource_manager.available_block_num(),
                block_size=self.cfg.cache_config.block_size,
                reserved_output_blocks=self.cfg.cache_config.enc_dec_block_num,
                max_num_batched_tokens=self.cfg.max_model_len,
                batch=num_prefill_batch,
            )
            # Fetch requests and add them to the scheduling queue
            for task in tasks:
                self.resource_manager.add_request(task)
            is_fetching = False

        while self.running:
            try:
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue
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
                    self.resource_manager.get_real_bsz()
                    self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
                else:
                    time.sleep(0.005)

            except Exception as e:
                err_msg = "Error happend while insert task to engine: {}, {}.".format(e, str(traceback.format_exc()))
                llm_logger.error(err_msg)

    def _insert_zmq_task_to_scheduler(self):
        if self.api_server_pid is None:
            return

        added_requests: Dict[str, int] = dict()
        while self.running:
            try:
                block = True if len(added_requests) == 0 else False
                if not self.cfg.enable_mm:
                    err, data = self.zmq_server.receive_json_once(block)
                else:
                    err, data = self.zmq_server.receive_pyobj_once(block)
                if err is not None:
                    llm_logger.error("Engine stops inserting zmq task into scheduler")
                    break

                request, insert_task = None, []
                results: List[Tuple[str, Optional[str]]] = list()
                if data:
                    request = Request.from_dict(data)
                    start_span("ENQUEUE_ZMQ", data, trace.SpanKind.PRODUCER)

                    llm_logger.debug(f"Receive request: {request}")

                    err_msg = None
                    if self.guided_decoding_checker is not None:
                        request, err_msg = self.guided_decoding_checker.schema_format(request)

                    if err_msg is not None:
                        llm_logger.error(err_msg)
                        results.append((request.request_id, err_msg))
                    else:
                        insert_task.append(request)

                response = self.scheduler.put_requests(insert_task)
                results.extend(response)

                if request:
                    if request.request_id not in added_requests:
                        added_requests[request.request_id] = 0
                    added_requests[request.request_id] += 1

                for request_id, failed in results:
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
                    self.zmq_server.send_multipart(request_id, error_result)
            except Exception as e:
                llm_logger.error(
                    f"Error happend while receving new request from zmq, details={e}, "
                    f"traceback={traceback.format_exc()}"
                )

    def add_requests(self, task, sampling_params=None, **kwargs):
        """
        Add a new request to the queue.

        Args:
            task: Request A dictionary representing the request.
            sampling_params: A dictionary representing the sampling parameters.

        Returns:
            None
        """
        # TODO 输入输出长度确认

        request = Request.from_dict(task)
        llm_logger.info(f"Receive request {request}")
        if sampling_params is not None:
            sampling_params.update_from_tokenizer(self.data_processor.tokenizer)
            request.sampling_params = sampling_params
        request.preprocess_start_time = time.time()

        enable_thinking = None
        if kwargs is not None:
            enable_thinking = kwargs.get("enable_thinking", None)
        request = self.data_processor.process_request(request, self.cfg.max_model_len, enable_thinking=enable_thinking)
        request.prompt_token_ids_len = len(request.prompt_token_ids)
        input_ids_len = request.prompt_token_ids_len
        request.set(
            "max_tokens",
            min(
                self.cfg.max_model_len - input_ids_len,
                request.get("max_tokens"),
            ),
        )
        if request.get("reasoning_max_tokens") is None:
            default_reasoning_max_tokens = max(int(request.get("max_tokens") * 0.8), 1)
            request.set("reasoning_max_tokens", default_reasoning_max_tokens)
        min_tokens = request.get("min_tokens")
        if input_ids_len + min_tokens >= self.cfg.max_model_len:
            error_msg = (
                f"Input text is too long, length of prompt token({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len "
            )
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.cfg.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.cfg.max_model_len})."
            )
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if self.guided_decoding_checker is not None:
            request, err_msg = self.guided_decoding_checker.schema_format(request)
            if err_msg is not None:
                llm_logger.error(err_msg)
                raise EngineError(err_msg, error_code=400)

        request.preprocess_end_time = time.time()
        self.scheduler.put_requests([request])
        llm_logger.info(f"Cache task with request_id ({request.get('request_id')})")
        llm_logger.debug(f"cache task: {request}")

    def warmup(self):
        """
        construct test tasks and avoid out of memory problem in the worker process
        """
        # get eos_token_id
        pass

    def split_mode_get_tasks(self):
        """
        Split mode get tasks
        """

        def receiver_loop():
            while self.running:
                try:

                    processed_indices = []
                    for idx, task in enumerate(self.waiting_requests):
                        if self.resource_manager.is_resource_sufficient(task.prompt_token_ids_len):
                            self.insert_tasks([task])
                            llm_logger.info(f"Resource available, processing task {task.request_id}")
                            processed_indices.append(idx)
                        else:
                            llm_logger.debug(f"Still waiting for resources {task.request_id}")
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
                                    self.insert_tasks(tasks, allocated=True)

                                    if self.cfg.innode_prefill_ports is not None:
                                        self.scheduler.put_results(tasks)

                                else:
                                    if len(self.waiting_requests):
                                        llm_logger.info(f"Waiting for resource for task {tasks[0].request_id}")
                                        self.waiting_requests.extend(tasks)
                                    else:
                                        new_waiting = []
                                        for task in tasks:
                                            if self.resource_manager.is_resource_sufficient(task.prompt_token_ids_len):
                                                self.insert_tasks([task])
                                            else:
                                                new_waiting.append(task)

                                        if new_waiting:
                                            self.waiting_requests.extend(new_waiting)
                                            llm_logger.info(f"Added {len(new_waiting)} tasks to waiting queue")

                    else:
                        time.sleep(0.001)

                except Exception as e:
                    llm_logger.error(f"Error in main loop: {e}")
                    time.sleep(0.1)

        threading.Thread(target=receiver_loop, daemon=True).start()

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
            remain_batched_tokens = self.cfg.max_num_batched_tokens
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
            image_token_sum[1:] = paddle.cumsum(image_mask.cast("int32"))
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

    def insert_tasks(self, tasks, current_id=-1, allocated=False):
        """
        Insert tasks to engine.
        """
        for task in tasks:
            start_span_request("DEQUEUE", task, trace.SpanKind.CONSUMER)
            if task.sampling_params.bad_words is not None:
                task.sampling_params.update_from_tokenizer(self.data_processor.tokenizer)
        # TODO 返回至 scheduler
        if allocated:
            current_tasks = []
            for task in tasks:
                cur_task_idx = self.resource_manager.req_dict[task.request_id]
                del self.resource_manager.req_dict[task.request_id]
                cur_task = self.resource_manager.tasks_list[cur_task_idx]
                cur_task.prompt_token_ids[0] = task.outputs.token_ids[0]
                if self.cfg.speculative_config.method in ["mtp"] and self.cfg.splitwise_role == "decode":
                    cur_task.draft_token_ids = copy.deepcopy(task.outputs.draft_token_ids)
                if task.error_code != 200:
                    self.resource_manager.stop_flags[cur_task_idx] = True
                    self.resource_manager.tasks_list[cur_task_idx] = None
                    self.resource_manager._recycle_block_tables(cur_task)
                    if task.request_id in self.token_processor.tokens_counter:
                        del self.token_processor.tokens_counter[task.request_id]
                    self.scheduler.put_results([task])
                    llm_logger.warning(
                        f"{task.request_id} prefill failed with msg:{task.error_msg}, recycle resource."
                    )
                    continue
                self.token_processor.tokens_counter[task.request_id] = 1
                current_tasks.append(cur_task)
            self.engine_worker_queue.put_tasks((current_tasks, self.resource_manager.real_bsz))
            return True

        self.resource_manager.check_and_free_block_tables()

        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item.schedule_start_time = time.time()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            llm_logger.error(f"Inserting batch:{len(tasks)} exceeds the available batch:{available_batch}.")
            llm_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        req_ids = [t.request_id for t in tasks]

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)

        if not tasks:
            error_msg = f"The request required resources is exceed the limit, request id={req_ids}."
            llm_logger.error(error_msg)
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
            llm_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
            for task in tasks:
                task.inference_start_time = time.time()
            if not is_prefill:
                if not self.cfg.enable_mm:
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

    def _set_warmup_token_processor(self):
        """
        set token_processor for warmup
        """
        self.token_processor_backup = self.token_processor
        self.token_processor = WarmUpTokenProcessor(self.cfg)
        self.token_processor.set_resource_manager(self.resource_manager)
        self.token_processor.tasks_queue = self.engine_worker_queue

        # start TokenProcessor thread
        self.token_processor.run()

    def _del_warmup_token_processor(self):
        """
        delete token_processor for warmup
        """
        self.token_processor.stop()
        del self.token_processor

        # reset token_processor
        self.token_processor = self.token_processor_backup
        del self.token_processor_backup

    def _worker_processes_ready(self):
        """
        judge if all worker processes are ready

        """
        if np.sum(self.worker_ready_signal.value) == self.cfg.worker_num_per_node:
            return True
        return False

    def _init_worker_signals(self):
        """
        Initialize shared memory to indicate engine status
        """
        # worker_ready_signatensor_parallel_size
        worker_ready_signal_data = np.zeros(shape=[self.cfg.worker_num_per_node], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=worker_ready_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        # exist_task_signal 用于各worker进程感知是否有新Task需要处理
        exist_task_signal_data = np.zeros([self.cfg.parallel_config.data_parallel_size], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=exist_task_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        # exist_swapped_task_signal 用于engine感知worker中是否存在swapped task
        exist_swapped_task_signal_data = np.zeros([self.cfg.parallel_config.data_parallel_size], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=exist_swapped_task_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        # exist_prefill_task_signal 用于各worker进程感知是否进行prefill
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        # launched_cache_manager_signal 用于感知engine是否启动了cache_manager
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.splitwise_role != "mixed":
            launched_cache_manager_signal_data = np.zeros([1], dtype=np.int32)
            self.launched_cache_manager_signal = IPCSignal(
                name="launched_cache_manager_signal",
                array=launched_cache_manager_signal_data,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True,
            )

        # worker_live_signal 用于engine感知各worker进程是否存活，记录每个step 时间
        worker_healthy_live_recorded_time_array = np.zeros(shape=[self.cfg.worker_num_per_node], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        if self.do_profile:
            get_profile_block_num = np.zeros([1], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True,
            )

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

    def _exit_sub_services(self):
        """
        exit sub services
        """
        self.running = False

        if hasattr(self, "cache_manager_processes"):
            self.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear()
            self.resource_manager.cache_manager.cache_ready_signal.clear()
            for p in self.cache_manager_processes:
                llm_logger.info(f"Killing cache manager process {p.pid}")
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception as e:
                    print(f"Error extracting file: {e}")
        self.worker_ready_signal.clear()
        self.exist_task_signal.clear()
        self.exist_swapped_task_signal.clear()
        self.worker_healthy_live_signal.clear()
        self.exist_prefill_task_signal.clear()
        if hasattr(self, "get_profile_block_num_signal"):
            self.get_profile_block_num_signal.clear()
        self.model_weights_status_signal.clear()
        if hasattr(self, "worker_proc") and self.worker_proc is not None:
            try:
                os.killpg(self.worker_proc.pid, signal.SIGTERM)
            except Exception as e:
                print(f"Error extracting sub services: {e}")

        self.engine_worker_queue.cleanup()
        if hasattr(self, "zmq_server") and self.zmq_server is not None:
            self.zmq_server.close()
        if hasattr(self, "dp_processed"):
            for p in self.dp_processed:
                p.join()

    def _setting_environ_variables(self):
        """
        配置环境变量
        """
        variables = {
            "ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY": 0,
            "LOAD_STATE_DICT_THREAD_NUM": len(self.cfg.device_ids.split(",")),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "FLAGS_use_append_attn": 1,
            "NCCL_ALGO": "Ring",
            "FLAGS_max_partition_size": int(os.getenv("FLAGS_max_partition_size", 32768)),
            "FLAGS_hardamard_moe_block_size": int(os.getenv("FLAGS_hardamard_moe_block_size", 128)),
            "FLAGS_hardamard_use_diagonal_block_matrix": int(
                os.getenv("FLAGS_hardamard_use_diagonal_block_matrix", 0)
            ),
        }
        # environment variables needed by Dy2St
        variables.update(
            {
                "SOT_LOG_LEVEL": os.getenv("SOT_LOG_LEVEL", default="0"),
                "SOT_UNSAFE_CACHE_FASTPATH": os.getenv("SOT_UNSAFE_CACHE_FASTPATH", default="1"),
                "SOT_ENABLE_0_SIZE_FALLBACK": os.getenv("SOT_ENABLE_0_SIZE_FALLBACK", default="0"),
                "SOT_SPECIALIZED_DIM_NUMBERS": os.getenv("SOT_SPECIALIZED_DIM_NUMBERS", default="no"),
                "FLAGS_specialize_device_in_dy2st": os.getenv("FLAGS_specialize_device_in_dy2st", default="1"),
                "FLAGS_enable_async_fast_gc": os.getenv("FLAGS_enable_async_fast_gc", default="0"),
                "FLAGS_pir_interpreter_record_stream_for_gc_cache": os.getenv(
                    "FLAGS_pir_interpreter_record_stream_for_gc_cache", default="1"
                ),
                "FLAGS_parameters_persistent_mode_in_dy2st": os.getenv(
                    "FLAGS_parameters_persistent_mode_in_dy2st", default="1"
                ),
            }
        )

        if self.cfg.splitwise_role != "mixed":
            variables["FLAGS_use_pd_disaggregation"] = 1
            # TODO dynamic load environment variable
            if self.cfg.splitwise_role == "prefill":
                variables["FLAGS_fmt_write_cache_completed_signal"] = 1

        if self.cfg.enable_mm:
            variables["FLAGS_max_partition_size"] = 1024

        command_prefix = ""
        for k, v in variables.items():
            command_prefix += f"{k}={v} "
        return command_prefix

    def _start_worker_service(self):
        """
        start gpu worker service

        """
        log_dir = os.getenv("FD_LOG_DIR", default="log")
        command_prefix = self._setting_environ_variables()
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        # TODO
        uncache_worker_stdout = "" if os.getenv("UNCACHE_WORKER_STDOUT", "0") == 1 else "-u"
        pd_cmd = f"{command_prefix} {sys.executable} {uncache_worker_stdout} -m paddle.distributed.launch"
        pd_cmd = pd_cmd + f" --log_dir {log_dir}"

        worker_path = "../worker/worker_process.py"
        py_script = os.path.join(current_dir_path, worker_path)

        ori_vocab_size = (
            len(self.data_processor.tokenizer.sp_model)
            if hasattr(self.data_processor.tokenizer, "sp_model")
            else len(self.data_processor.tokenizer.vocab)
        )

        arguments = (
            f" --devices {self.cfg.device_ids} {py_script}"
            f" --max_num_seqs {self.cfg.max_num_seqs} --max_model_len {self.cfg.max_model_len}"
            f" --gpu_memory_utilization {self.cfg.cache_config.gpu_memory_utilization}"
            f" --model {self.cfg.model_name_or_path!s}"
            f" --device_ids {self.cfg.device_ids}"
            f" --tensor_parallel_size {self.cfg.tensor_parallel_size}"
            f" --engine_worker_queue_port {self.cfg.engine_worker_queue_port!s}"
            f" --pod_ip {self.cfg.master_ip}"
            f" --total_block_num {self.cfg.cache_config.total_block_num}"
            f" --block_size {self.cfg.cache_config.block_size}"
            f" --enc_dec_block_num {self.cfg.cache_config.enc_dec_block_num}"
            f" --eos_tokens_lens {self.data_processor.eos_token_id_len}"
            f" --pad_token_id {self.data_processor.pad_token_id}"
            f" --engine_pid {self.engine_pid}"
            f" --max_num_batched_tokens {self.cfg.max_num_batched_tokens}"
            f" --splitwise_role {self.cfg.splitwise_role}"
            f" --kv_cache_ratio {self.cfg.cache_config.kv_cache_ratio}"
            f" --expert_parallel_size {self.cfg.parallel_config.expert_parallel_size}"
            f" --quantization {self.cfg.model_config.quantization}"
            f" --ori_vocab_size {ori_vocab_size}"
            f" --speculative_config '{self.cfg.speculative_config.to_json_string()}'"
            f" --graph_optimization_config '{self.cfg.graph_optimization_config.to_json_string()}'"
            f" --guided_decoding_backend {self.cfg.guided_decoding_backend}"
            f" --load_strategy {self.cfg.load_config.load_strategy}"
            f" --early_stop_config '{self.cfg.early_stop_config.to_json_string()}'"
            f" --load_choices {self.cfg.load_choices}"
        )

        worker_append_flag = {
            "enable_expert_parallel": self.cfg.parallel_config.enable_expert_parallel,
            "enable_prefix_caching": self.cfg.cache_config.enable_prefix_caching,
            "enable_chunked_prefill": self.cfg.cache_config.enable_chunked_prefill,
            "do_profile": self.do_profile,
            "dynamic_load_weight": self.cfg.load_config.dynamic_load_weight,
            "disable_any_whitespace": self.cfg.disable_any_whitespace,
            "enable_custom_all_reduce": self.cfg.parallel_config.enable_custom_all_reduce,
            "enable_logprob": self.cfg.enable_logprob,
            "enable_mm": self.cfg.enable_mm,
        }
        for worker_flag, value in worker_append_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag}"
        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {','.join(self.cfg.ips)} --nnodes {len(self.cfg.ips)}"
        pd_cmd = pd_cmd + arguments + f" 2>{log_dir}/launch_worker.log"
        llm_logger.info(f"Launch worker service command: {pd_cmd}")
        p = subprocess.Popen(
            pd_cmd,
            stdout=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid,
        )
        return p

    def _format_and_add_data(self, prompts: dict):

        if "request_id" in prompts:
            prompts["request_id"] = prompts["request_id"]

        if "request_id" not in prompts:
            request_id = str(uuid.uuid4())
            prompts["request_id"] = request_id
        query_list = []

        if "context" in prompts:
            for item in prompts["context"]:
                if item["role"] == "system":
                    prompts["system"] = item["utterance"]
                elif item["role"] in ["user", "assistant"]:
                    query_list.append(item["utterance"])
                    prompts["prompt"] = query_list

        if "max_tokens" not in prompts:
            prompts["max_tokens"] = self.cfg.max_model_len

        self.add_requests(prompts)
        return prompts["request_id"]

    def generate(self, prompts, stream):
        """
        Generates a response based on the given prompt using the model.

        Args:
            prompts (dict): The prompt to use for generating the response.
            stream (bool): Whether to stream the output or wait until completion.

        Yields:
            dict: The generated response.
        """
        llm_logger.info(f"Starting generation for prompt: {prompts}")
        try:
            req_id = self._format_and_add_data(prompts)
        except Exception as e:
            llm_logger.error(f"Error happend while adding request, details={e}")
            raise EngineError(str(e), error_code=400)

        # 获取当前请求的结果
        for result in self._get_generated_tokens(req_id):
            is_end = result.finished
            if stream and not is_end:
                processed = self.data_processor.process_response(result)
                if processed is None:
                    continue
                output = processed.to_dict()
                yield output

            # Exit loop if termination condition is met
            if is_end:
                processed = self.data_processor.process_response(result)
                output = processed.to_dict()
                llm_logger.debug(f"Generate result: {output}")
                if not stream:
                    yield output
                else:
                    output["outputs"]["text"] = ""
                    output["outputs"]["reasoning_content"] = ""
                    yield output

                self.resource_manager.check_and_free_block_tables()

    def _stop_profile(self):
        """
        Stop profiling of the model server and reset variables.
        """
        self.do_profile = 0
        while self.get_profile_block_num_signal.value[0] == 0:
            time.sleep(1)
        num_gpu_blocks = self.get_profile_block_num_signal.value[0]
        self.cfg.cache_config.reset(num_gpu_blocks)
        self.resource_manager.reset_cache_config(self.cfg.cache_config)
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.splitwise_role != "mixed":
            device_ids = self.cfg.device_ids.split(",")
            self.cache_manager_processes = self.resource_manager.cache_manager.launch_cache_manager(
                cache_config=self.cfg.cache_config,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                device_ids=device_ids,
                pod_ip=self.cfg.master_ip,
                engine_worker_queue_port=self.cfg.engine_worker_queue_port,
                pid_suffix=self.ipc_signal_suffix,
            )
            self.launched_cache_manager_signal.value[0] = 1

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def check_worker_initialize_status(self):
        """
        Check the initlialize status of workers by stdout logging
        """

        def detect_thread():
            for line in self.worker_proc.stdout:
                line = line.decode("utf-8", errors="ignore")
                if self.worker_init_status.get("finished", False):
                    break
                if match := re.search(
                    r"Loading (?:fastsafetensors |safetensors )?checkpoint shards:\s*(\d+)",
                    line,
                ):
                    self.worker_init_status["weight_loadding"] = eval(match.group(1)) * 1.0 / 100
                elif (match := re.search(r"Start load layer (\d+)", line)) or (
                    match := re.search(r"set state for layer (\d+)", line)
                ):
                    progress = eval(match.group(1)) * 1.0 / self.cfg.model_config.num_hidden_layers
                    self.worker_init_status["layer_loadding"] = progress
                    if self.worker_init_status["layer_loadding"] == self.cfg.model_config.num_hidden_layers - 1:
                        self.worker_init_status["finished"] = True

        self.checking_worker_status_thread = threading.Thread(target=detect_thread, daemon=True)
        self.checking_worker_status_thread.start()
        checking_worker_init_kv_cache_status_thread = None
        if self.do_profile:
            checking_worker_init_kv_cache_status_thread = threading.Thread(target=self._stop_profile, daemon=True)
            checking_worker_init_kv_cache_status_thread.start()

        # display weight loadding progress
        with tqdm(total=100, desc="Loading Weights") as pbar:
            progress = 0
            while progress < 100:
                progress = int(self.worker_init_status.get("weight_loadding", 0) * 100)
                if self.worker_init_status.get("layer_loadding", 0) > 0 or self._worker_processes_ready():
                    progress = 100
                pbar.update(progress - pbar.n)
                pbar.refresh()
                time.sleep(0.5)
                if self.worker_proc.poll() is not None:
                    return False

        # display layer loadding progress
        with tqdm(total=100, desc="Loading Layers") as pbar:
            progress = 0
            while progress < 100:
                progress = int(self.worker_init_status.get("layer_loadding", 0) * 100)
                if self._worker_processes_ready():
                    progress = 100
                pbar.update(progress - pbar.n)
                pbar.refresh()
                time.sleep(0.5)
                if self.worker_proc.poll() is not None:
                    return False

        self.worker_init_status["finished"] = True
        try:
            self.checking_worker_status_thread.join(timeout=1)
            if checking_worker_init_kv_cache_status_thread is not None:
                checking_worker_init_kv_cache_status_thread.join(timeout=1)
        except Exception:
            pass
        return True

    def start_queue_service(self):
        """
        start queue service for engine worker communication
        """
        address = (self.cfg.master_ip, self.cfg.engine_worker_queue_port)
        if self.cfg.host_ip == self.cfg.master_ip or self.cfg.master_ip == "0.0.0.0":
            llm_logger.info(f"Starting engine worker queue server service at {address}")
            self.engine_worker_queue_server = EngineWorkerQueue(
                address=address,
                is_server=True,
                num_client=self.cfg.tensor_parallel_size,
                local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
            )

            if self.cfg.cache_config.enable_prefix_caching or self.cfg.splitwise_role != "mixed":
                self.cache_task_queue = EngineCacheQueue(
                    address=(
                        self.cfg.master_ip,
                        self.cfg.cache_config.cache_queue_port,
                    ),
                    authkey=b"cache_queue_service",
                    is_server=True,
                    num_client=self.cfg.tensor_parallel_size,
                    client_id=-1,
                    local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                )

        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            num_client=self.cfg.tensor_parallel_size,
            client_id=0,
            local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
            local_data_parallel_id=min(
                self.cfg.worker_num_per_node * self.cfg.node_rank,
                self.cfg.parallel_config.data_parallel_size - 1,
            ),
        )
