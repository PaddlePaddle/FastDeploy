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

from typing import List, Tuple, Dict, Optional
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

import numpy as np
import zmq
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import (EngineWorkerQueue, IPCSignal,
                                           ZmqClient)
from fastdeploy.output.token_processor import (TokenProcessor,
                                               WarmUpTokenProcessor)
from fastdeploy.utils import EngineError, console_logger, llm_logger


class LLMEngine(object):
    """
    Main engine class for managing Large Language Model (LLM) inference operations.
    
    This class handles the complete lifecycle of LLM inference including:
    - Initialization and configuration
    - Request processing and scheduling
    - Resource management
    - Communication with worker processes
    - Token generation and output handling
    
    Key Components:
    - Scheduler: Manages request queue and task scheduling
    - ResourceManager: Handles GPU memory allocation and block management
    - TokenProcessor: Processes generated tokens and handles streaming output
    - WorkerQueue: Facilitates communication between engine and worker processes
    
    Attributes:
        cfg (Config): Engine configuration parameters
        scheduler (BaseScheduler): Task scheduler instance
        input_processor (InputPreprocessor): Preprocesses input data
        resource_manager (ResourceManager): Manages GPU resources
        token_processor (TokenProcessor): Handles token generation
        engine_worker_queue (EngineWorkerQueue): Worker communication queue
        is_started (bool): Engine running status flag
        do_profile (int): Profiling mode flag (0=disabled, 1=enabled)
        worker_proc (subprocess.Popen): Worker process handle
        zmq_server (ZmqClient): ZMQ communication server
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
        Initialize the LLM engine with given configuration.
        
        Note: Prefer using from_engine_args() for most use cases as it provides
        better configuration validation.
        
        Sets up:
        - Task scheduler based on configuration
        - Input preprocessing pipeline  
        - Resource management system
        - Token generation processor
        - Worker communication queue
        - Profiling and monitoring systems
        
        Args:
            cfg (Config): Complete engine configuration including:
                         - Model parameters
                         - Parallelism settings
                         - Memory allocation
                         - Performance tuning options
                         
        Raises:
            ValueError: If required configuration parameters are missing or invalid
        """
        self.cfg = cfg
        self.scheduler = cfg.scheduler_config.scheduler()

        self.input_processor = InputPreprocessor(cfg.tokenizer, cfg.enable_mm)
        self.resource_manager = ResourceManager(
            cfg.max_num_seqs, cfg.cache_config)

        self.token_processor = TokenProcessor(
            cfg=self.cfg, cached_generated_tokens=self.scheduler)
        self.token_processor.set_resource_manager(self.resource_manager)
        time.sleep(1)  # TODO: Investigate the purpose of this sleep.

        address = ('0.0.0.0', self.cfg.engine_worker_queue_port)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=True,
            num_client=self.cfg.tensor_parallel_size)

        self.is_started = False

        if self.cfg.cache_config.num_gpu_blocks_override is None:
            self.do_profile = 1
        else:
            self.do_profile = 0

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

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

        self.worker_proc = self._start_worker_service()
        console_logger.info("Waitting worker processes ready...")
        time.sleep(5)
        self.worker_init_status = dict()
        if not self.check_worker_initialize_status():
            console_logger.error(
                "Failed to launch worker processes, check log/workerlog.* for more details."
            )
            return False

        # Start warmup if enabled
        if self.cfg.use_warmup:
            console_logger.info("Starting warmup")
            self._set_warmup_token_processor()
            self.warmup()
            self._del_warmup_token_processor()
            console_logger.info("Warmup finished")

        self.token_processor.tasks_queue = self.engine_worker_queue

        self.insert_task_to_worker_thread = threading.Thread(
            target=self._insert_task_to_worker, args=())
        self.insert_task_to_worker_thread.daemon = True
        self.insert_task_to_worker_thread.start()

        if self.api_server_pid is not None:
            self.insert_task_to_scheduler_thread = threading.Thread(
                target=self._insert_zmq_task_to_scheduler, args=())
            self.insert_task_to_scheduler_thread.daemon = True
            self.insert_task_to_scheduler_thread.start()

            self.receive_output_thread = threading.Thread(
                target=self._zmq_send_generated_tokens, args=())
            self.receive_output_thread.daemon = True
            self.receive_output_thread.start()

        # Start TokenProcessor thread
        self.token_processor.run()

        # self.start_push_sender_thread()
        if self.do_profile:
            self._stop_profile()
        console_logger.info(
            "Worker processes are launched with {} seconds.".format(
                time.time() - start_time))
        return True

    def _zmq_send_generated_tokens(self):
        """
        Recieve output for zmq
        """
        assert self.api_server_pid is not None
        while True:
            try:
                def get_results_handler(request_ids):
                    results = dict()
                    try:
                        results = self.scheduler.get_results(request_ids)
                        for req_id, contents in results.items():
                            results[req_id] = [data.to_dict()
                                               for data in contents]
                    except Exception as e:
                        llm_logger.error(f"Get results handler error: {e}")
                    return results

                self.zmq_server.send_multipart2(get_results_handler)
            except Exception as e:
                llm_logger.error("Unexcepted error happend: {}, {}".format(
                    e, str(traceback.format_exc())))

    def _get_generated_result(self, request_id):
        """
        Get result from scheduler, this function is called by generate()
        which is only used in offline inference.
        """
        try:
            acc = None
            while True:
                results = self.scheduler.get_results([request_id])
                for _, contents in results.items():
                    for result in contents:
                        if acc is None:
                            acc = result
                        else:
                            acc.add(result)

                        if result.finished:
                            yield acc
                            return

                        yield result

        except Exception as e:
            llm_logger.error("Unexcepted error happend: {}, {}".format(
                e, str(traceback.format_exc())))

    def _insert_task_to_worker(self):
        """
        Insert task to engine thread, monitor scheduler request queue.
        if the engine has resource, insert task to engine
        """
        while True:
            try:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue

                num_prefill_batch = min(
                    int(self.resource_manager.available_batch()),
                    self.cfg.max_prefill_batch)

                if self.cfg.enable_chunked_prefill:
                    cur_max_num_batched_tokens = self.cfg.max_model_len * num_prefill_batch
                else:
                    cur_max_num_batched_tokens = self.cfg.max_num_batched_tokens

                tasks = self.scheduler.get_requests(
                    available_blocks=self.resource_manager.available_block_num(
                    ),
                    block_size=self.cfg.cache_config.block_size,
                    reserved_output_blocks=self.cfg.cache_config.
                    enc_dec_block_num,
                    max_num_batched_tokens=cur_max_num_batched_tokens,
                    batch=num_prefill_batch)

                if len(tasks) == 0:
                    time.sleep(0.001)
                    continue

                self.insert_tasks(tasks)
            except Exception as e:
                err_msg = "Error happend while insert task to engine: {}, {}.".format(
                    e, str(traceback.format_exc()))
                llm_logger.error(err_msg)

    def _insert_zmq_task_to_scheduler(self):
        if self.api_server_pid is None:
            return

        added_requests: Dict[str, int] = dict()
        while True:
            try:
                block = True if len(added_requests) == 0 else False
                if not self.cfg.enable_mm:
                    err, data = self.zmq_server.receive_json_once(block)
                else:
                    err, data = self.zmq_server.receive_pyobj_once(block)
                if err is not None:
                    llm_logger.error(
                        "Engine stops inserting zmq task into scheduler")
                    break

                request = None
                if data:
                    request = Request.from_dict(data)
                    llm_logger.info(f"Receive request: {request}")

                results: List[Tuple[str, Optional[str]]] = self.scheduler.put_requests(
                    [] if request is None else [request])

                if request:
                    if request.request_id not in added_requests:
                        added_requests[request.request_id] = 0
                    added_requests[request.request_id] += 1

                for request_id, failed in results:
                    added_requests[request_id] -= 1
                    if added_requests[request_id] == 0:
                        added_requests.pop(request_id)

                if failed is None:
                    continue

                error_result = RequestOutput(request_id=request_id,
                                             finished=True,
                                             error_code=500,
                                             error_msg=failed)
                # Since the request is not in scheduler
                # Send result by zmq directly
                self.zmq_server.send_multipart(
                    request.request_id, error_result)
            except Exception as e:
                llm_logger.error(
                    f"Error happend while receving new request from zmq, details={e}"
                )

    def add_requests(self, task, sampling_params=None):
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
        if sampling_params is not None:
            request.sampling_params = sampling_params
        request.preprocess_start_time = time.time()
        request = self.data_processor.process_request(request, self.cfg.max_model_len)

        request.prompt_token_ids_len = len(request.prompt_token_ids)
        input_ids_len = request.prompt_token_ids_len
        request.set(
            "max_tokens",
            min(self.cfg.max_model_len - input_ids_len,
                request.get("max_tokens")))
        min_tokens = request.get("min_tokens")
        if input_ids_len + min_tokens >= self.cfg.max_model_len:
            error_msg = (
                f"Input text is too long, length of prompt token({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len ")
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.cfg.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.cfg.max_model_len})."
            )
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        request.preprocess_end_time = time.time()
        self.scheduler.put_requests([request])
        llm_logger.info(
            f"Cache task with request_id ({request.get('request_id')})")
        llm_logger.debug(f"cache task: {request}")

    def warmup(self):
        """
        construct test tasks and avoid out of memory problem in the worker process
        """
        # get eos_token_id
        pass

    def insert_tasks(self, tasks):
        """
        Insert tasks to engine.
        """
        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item.schedule_start_time = time.time()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            llm_logger.error(
                "Inserting batch:{} exceeds the available batch:{}.".format(
                    len(tasks), available_batch))
            llm_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        req_ids = [t.request_id for t in tasks]

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)
        if not tasks:
            error_msg = f"The request required resources is exceed the limit, request id={req_ids}."
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=500)

        self.token_processor.number_of_tasks += len(tasks)
        token_chunk_size =(self.cfg.max_num_batched_tokens // len(tasks)) // self.cfg.cache_config.block_size * self.cfg.cache_config.block_size
        for i in range(len(tasks)):
            self.token_processor.number_of_input_tokens += tasks[
                i].prompt_token_ids_len

            tasks[i].set("token_chunk_size", token_chunk_size)

        llm_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
        self.engine_worker_queue.put_tasks(
            (tasks, self.resource_manager.real_bsz))
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
        return np.sum(self.resource_manager.stop_flags) == len(
            self.resource_manager.stop_flags)

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
        if np.sum(self.worker_ready_signal.value) == self.cfg.tp_num_per_node:
            return True
        return False

    def _init_worker_signals(self):
        """
        Initialize shared memory to indicate engine status
        """
        # worker_ready_signal 用于engine感知各worker进程是否Ready

        worker_ready_signal_data = np.zeros(
            shape=[self.cfg.tensor_parallel_size], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(name="worker_ready_singnal",
                                             array=worker_ready_signal_data,
                                             dtype=np.int32,
                                             suffix=self.ipc_signal_suffix,
                                             create=True)

        # exist_task_signal 用于各worker进程感知是否有新Task需要处理
        exist_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(name="exist_task_signal",
                                           array=exist_task_signal_data,
                                           dtype=np.int32,
                                           suffix=self.ipc_signal_suffix,
                                           create=True)

        # exist_swapped_task_signal 用于engine感知worker中是否存在swapped task
        exist_swapped_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=exist_swapped_task_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True)

        # worker_live_signal 用于engine感知各worker进程是否存活，记录每个step 时间
        worker_healthy_live_recorded_time_array = np.zeros(
            shape=[self.cfg.tensor_parallel_size], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True)

        if self.do_profile:
            get_profile_block_num = np.zeros([self.cfg.tensor_parallel_size],
                                             dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True)

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True)

    def _exit_sub_services(self):
        """
        exit sub services
        """
        self.worker_ready_signal.clear()
        self.exist_task_signal.clear()
        self.exist_swapped_task_signal.clear()
        self.worker_healthy_live_signal.clear()
        if hasattr(self, "get_profile_block_num_signal"):
            self.get_profile_block_num_signal.clear()
        self.model_weights_status_signal.clear()
        if hasattr(self, "worker_proc") and self.worker_proc is not None:
            try:
                os.killpg(self.worker_proc.pid, signal.SIGTERM)
            except:
                pass
        if hasattr(self, "zmq_server") and self.zmq_server is not None:
            self.zmq_server.close()

    def _setting_environ_variables(self):
        """
       配置环境变量
       """
        variables = {
            "PADDLE_TRAINER_ID": 0,
            "PADDLE_TRAINERS_NUM": 1,
            "TRAINER_INSTANCES_NUM": 1,
            "TRAINER_INSTANCES": "0.0.0.0",
            "ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY": 0,
            "LOAD_STATE_DICT_THREAD_NUM": len(self.cfg.device_ids.split(',')),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "FLAGS_use_append_attn": 1,
            "NCCL_ALGO": "Ring",
            "ELLM_DYNAMIC_MODE": 1,
        }
        command_prefix = ""
        for k, v in variables.items():
            command_prefix += f"{k}={v} "
        return command_prefix

    def _start_worker_service(self):
        """
        start gpu worker service

        """
        command_prefix = self._setting_environ_variables()
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        # TODO
        uncache_worker_stdout = "" if os.getenv("UNCACHE_WORKER_STDOUT",
                                                "0") == 1 else "-u"
        pd_cmd = f"{command_prefix} {sys.executable} {uncache_worker_stdout} -m paddle.distributed.launch "
        py_script = os.path.join(current_dir_path, "../worker/worker.py")
        arguments = (
            f" --nnodes {str(self.cfg.nnode)}"
            f" --devices {self.cfg.device_ids} {py_script}"
            f" --max_num_seqs {self.cfg.max_num_seqs} --max_model_len {self.cfg.max_model_len}"
            f" --gpu_memory_utilization {self.cfg.cache_config.gpu_memory_utilization}"
            f" --model_name_or_path {str(self.cfg.model_name_or_path)}"
            f" --device_ids {self.cfg.device_ids}"
            f" --engine_worker_queue_port {str(self.cfg.engine_worker_queue_port)}"
            f" --total_block_num {self.cfg.cache_config.total_block_num}"
            f" --block_size {self.cfg.cache_config.block_size}"
            f" --enc_dec_block_num {self.cfg.cache_config.enc_dec_block_num}"
            f" --eos_tokens_lens {self.data_processor.eos_token_id_len}"
            f" --pad_token_id {self.data_processor.pad_token_id}"
            f" --engine_pid {self.engine_pid}"
            f" --do_profile {self.do_profile}"
            f" --dynamic_load_weight {self.cfg.model_config.dynamic_load_weight}"
            f" --max_num_batched_tokens {self.cfg.max_num_batched_tokens}"
            f" --kv_cache_ratio {self.cfg.cache_config.kv_cache_ratio} --dtype {self.cfg.cache_config.cache_dtype}"
        )
        worker_append_flag = {
            "enable_chunked_prefill": self.cfg.enable_chunked_prefill,
        }
        for worker_flag, value in worker_append_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag}"

        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {self.cfg.ips}"
        log_dir = os.getenv("FD_LOG_DIR", default="log")
        pd_cmd = pd_cmd + arguments + f" 2>{log_dir}/launch_worker.log"
        llm_logger.info("Launch worker service command: {}".format(pd_cmd))
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
            llm_logger.error(
                f"Error happend while adding request, details={e}")
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

    def _stop_profile(self):
        """
        Stop profiling of the model server and reset variables.
        """
        self.do_profile = 0
        num_gpu_blocks = -1
        for i in range(self.cfg.tensor_parallel_size):
            while self.get_profile_block_num_signal.value[i] == 0:
                time.sleep(1)
            if num_gpu_blocks < 0:
                num_gpu_blocks = self.get_profile_block_num_signal.value[i]
            else:
                num_gpu_blocks = min(
                    num_gpu_blocks, self.get_profile_block_num_signal.value[i])

        console_logger.info(f"Stop profile, num_gpu_blocks:  {num_gpu_blocks}")
        self.cfg.cache_config.reset(num_gpu_blocks)
        self.resource_manager.reset_cache_config(self.cfg.cache_config)

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - \
                self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def check_worker_initialize_status(self):
        """
        Check the initlialize status of workers by stdout logging
        """

        def detect_thread():
            for line in self.worker_proc.stdout:
                line = line.decode('utf-8', errors='ignore')
                if self.worker_init_status.get("finished", False):
                    break
                if match := re.search(r'Loading checkpoint shards:\s*(\d+)',
                                      line):
                    self.worker_init_status["weight_loadding"] = eval(
                        match.group(1)) * 1.0 / 100
                elif (match := re.search(r'Start load layer (\d+)',
                                         line)) or (match := re.search(
                                             r'set state for layer (\d+)',
                                             line)):
                    progress = eval(match.group(
                        1)) * 1.0 / self.cfg.model_config.num_layers
                    self.worker_init_status["layer_loadding"] = progress
                    if self.worker_init_status[
                            "layer_loadding"] == self.cfg.model_config.num_layers - 1:
                        self.worker_init_status["finished"] = True

        self.checking_worker_status_thread = threading.Thread(
            target=detect_thread, args=())
        self.checking_worker_status_thread.daemon = True
        self.checking_worker_status_thread.start()

        # display weight loadding progress
        with tqdm(total=100, desc="Loading Weights") as pbar:
            progress = 0
            while progress < 100:
                progress = int(
                    self.worker_init_status.get("weight_loadding", 0) * 100)
                if self.worker_init_status.get(
                        "layer_loadding",
                        0) > 0 or self._worker_processes_ready():
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
                progress = int(
                    self.worker_init_status.get("layer_loadding", 0) * 100)
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
        except Exception:
            pass
        return True

