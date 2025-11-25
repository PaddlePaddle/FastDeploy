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

import json
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
from dataclasses import asdict

import numpy as np
import paddle
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.expert_service import start_data_parallel_service
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator import EngineWorkerQueue, IPCSignal
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.platforms import current_platform
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
        self.is_started = False

        self.engine = EngineService(cfg)

        if self.cfg.cache_config.num_gpu_blocks_override is None:
            self.do_profile = 1
        else:
            self.do_profile = 0
        self._finalizer = weakref.finalize(self, self._exit_sub_services)

        main_process_metrics.set_cache_config_info(obj=self.cfg.cache_config)

    def start(self, api_server_pid=None):
        """
        Initializes the engine and starts its sub-services.
        If `api_server_pid` is defined, will launch a thread
        to keep getting request from zmq_server.

        NOTE: To clarify the launch order of the components of the LLM engine:
        1. First, launch splitwise scheduler (if necessary) and expert services (if necessary).
        2. Then, launch common engine, which includes some background threads that inserts tasks and receives ouptuts.
        3. Most importantly, launch workers and cache services. The launch order of them are listed as follows.

            | Profile | Mixed | PrefixCache | Cache -> Worker | Worker -> Cache |
            |---------|-------|-------------|-----------------|-----------------|
            | 1       | 1     | 1           | 0               | 1               |
            | 1       | 1     | 0           | 0               | 0               |
            | 1       | 0     | 1           | 0               | 1               |
            | 1       | 0     | 0           | 0               | 1               |
            | 0       | 1     | 1           | 0               | 1               |
            | 0       | 1     | 0           | 0               | 0               |
            | 0       | 0     | 1           | 1               | 0               |
            | 0       | 0     | 0           | 1               | 0               |

        4. Finally, inform user the engine has successfully started.

        """
        assert not self.is_started, "The engine is already started."
        start_time = time.time()

        self.api_server_pid = api_server_pid
        self.ipc_signal_suffix = self.cfg.parallel_config.engine_worker_queue_port[0]
        self._init_worker_signals()

        self.launch_components()

        self.engine.start()
        self.engine.create_data_processor()
        self.data_processor = self.engine.data_processor

        # If block numer is specified and model is deployed in mixed mode, start cache manager first
        if not self.do_profile and self.cfg.scheduler_config.splitwise_role != "mixed":
            if not current_platform.is_intel_hpu():
                device_ids = self.cfg.parallel_config.device_ids.split(",")
                self.cache_manager_processes = self.engine.start_cache_service(device_ids, self.ipc_signal_suffix)

        # Start workers
        self.worker_proc = self._start_worker_service()
        console_logger.info("Waiting for worker processes to be ready...")
        time.sleep(5)
        self.worker_init_status = dict()

        result_container = {}

        def check_worker_initialize_status_func(res: dict):
            res["worker_is_alive"] = True
            if not self.check_worker_initialize_status():
                console_logger.error("Failed to launch worker processes, check log/workerlog.* for more details.")
                res["worker_is_alive"] = False

        self.check_worker_initialize_status_func_thread = threading.Thread(
            target=check_worker_initialize_status_func, args=(result_container,), daemon=True
        )
        self.check_worker_initialize_status_func_thread.start()

        # Wait model loading
        while self.loaded_model_signal.value[0] == 0:
            # Make sure worker process is alive
            if not self.check_worker_initialize_status_func_thread.is_alive():
                return False
            time.sleep(1)

        # If block number is not specified, let workers do profiling to determine the block number,
        # and then start the cache manager
        if self.do_profile:
            self._stop_profile()
        elif self.cfg.scheduler_config.splitwise_role == "mixed" and self.cfg.cache_config.enable_prefix_caching:
            if not current_platform.is_intel_hpu():
                device_ids = self.cfg.parallel_config.device_ids.split(",")
                self.cache_manager_processes = self.engine.start_cache_service(device_ids, self.ipc_signal_suffix)

        # Launch components: scheduler, cache_manager, expert_service et.al.
        if self.cfg.scheduler_config.splitwise_role != "mixed":
            self.launched_cache_manager_signal.value[0] = 1

        if self.cfg.scheduler_config.splitwise_role != "mixed" and envs.FD_ENABLE_INTERNAL_ADAPTER:
            envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT = envs.FD_ZMQ_RECV_REQUEST_SERVER_PORTS.split(",")[0]
            envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT = envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORTS.split(",")[0]

        if api_server_pid is not None:
            llm_logger.info(f"Start zmq server, api_server_pid: {api_server_pid}")
            self.engine.start_zmq_service(api_server_pid)

        # Worker launched
        self.check_worker_initialize_status_func_thread.join()
        if not result_container["worker_is_alive"]:
            console_logger.error("Failed to launch worker processes, check log/workerlog.* for more details.")
            return False

        console_logger.info(f"Worker processes are launched with {time.time() - start_time} seconds.")

        # Print blocks number & max running requests to console
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            block_size = self.cfg.cache_config.block_size
            num_gpu_blocks = self.cfg.cache_config.num_gpu_blocks_override or self.cfg.cache_config.total_block_num
            num_cpu_blocks = self.cfg.cache_config.num_cpu_blocks
            max_running_requests = min(
                (num_gpu_blocks + num_cpu_blocks) * block_size // self.cfg.model_config.max_model_len,
                self.cfg.scheduler_config.max_num_seqs,
            )
            console_logger.info(
                f"Detected {num_gpu_blocks} gpu blocks and {num_cpu_blocks} cpu blocks in cache (block size: {block_size})."
            )
            console_logger.info(
                f"FastDeploy will be serving {max_running_requests} running requests "
                f"if each sequence reaches its maximum length: {self.cfg.model_config.max_model_len}"
            )

        return True

    def _get_generated_result(self):
        """
        Get result from scheduler, this function is called by generate()
        which is only used in offline inference.
        """
        return self.engine.scheduler.get_results()

    # _insert_task_to_worker moved to CommonEngine

    def _has_guided_input(self, request):
        """
        Check if the request has any guided input.
        """
        return any(
            x is not None
            for x in (
                request.guided_json,
                request.guided_regex,
                request.guided_choice,
                request.structural_tag,
                request.guided_grammar,
                request.guided_json_object,
            )
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

        if sampling_params is not None:
            task.update(asdict(sampling_params))
        request = Request.from_dict(task)
        request.llm_engine_recv_req_timestamp = time.time()
        llm_logger.info(f"Receive request {request}")
        if sampling_params is not None:
            if sampling_params.temperature is not None and abs(sampling_params.temperature) < 1e-06:
                sampling_params.temperature = 1e-06
            request.sampling_params = sampling_params
        request.preprocess_start_time = time.time()
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        chat_template_kwargs["chat_template"] = kwargs.get("chat_template")
        kwargs["chat_template_kwargs"] = chat_template_kwargs
        request = self.engine.data_processor.process_request(request, self.cfg.model_config.max_model_len, **kwargs)
        request.prompt_token_ids_len = len(request.prompt_token_ids)
        request.need_prefill_tokens = request.prompt_token_ids_len
        input_ids_len = request.prompt_token_ids_len
        request.set(
            "max_tokens",
            min(
                self.cfg.model_config.max_model_len - input_ids_len,
                request.get("max_tokens"),
            ),
        )
        min_tokens = request.get("min_tokens")
        if input_ids_len + min_tokens >= self.cfg.model_config.max_model_len:
            error_msg = (
                f"Input text is too long, length of prompt token({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len "
            )
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.cfg.model_config.max_model_len:
            error_msg = f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.cfg.model_config.max_model_len})."
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if request.get("stop_seqs_len") is not None:
            stop_seqs_len = request.get("stop_seqs_len")
            max_stop_seqs_num = envs.FD_MAX_STOP_SEQS_NUM
            if len(stop_seqs_len) > max_stop_seqs_num:
                error_msg = (
                    f"Length of stop ({stop_seqs_len}) exceeds the limit max_stop_seqs_num({max_stop_seqs_num})."
                    "Please reduce the number of stop or set a lager max_stop_seqs_num by `FD_MAX_STOP_SEQS_NUM`"
                )
                llm_logger.error(error_msg)
                raise EngineError(error_msg, error_code=400)
            stop_seqs_max_len = envs.FD_STOP_SEQS_MAX_LEN
            for single_stop_seq_len in stop_seqs_len:
                if single_stop_seq_len > stop_seqs_max_len:
                    error_msg = (
                        f"Length of stop_seqs({single_stop_seq_len}) exceeds the limit stop_seqs_max_len({stop_seqs_max_len})."
                        "Please reduce the length of stop sequences or set a larger stop_seqs_max_len by `FD_STOP_SEQS_MAX_LEN`"
                    )
                    llm_logger.error(error_msg)
                    raise EngineError(error_msg, error_code=400)

        if self._has_guided_input(request):
            err_msg = None
            if self.guided_decoding_checker is None:
                err_msg = (
                    "guided_backend is None, use --guided-decoding-backend to specify the backend at server startup."
                )
            else:
                request, err_msg = self.guided_decoding_checker.schema_format(request)

            if err_msg is not None:
                llm_logger.error(err_msg)
                raise EngineError(err_msg, error_code=400)

        request.preprocess_end_time = time.time()
        self.engine.scheduler.put_requests([request])
        llm_logger.info(f"Cache task with request_id ({request.get('request_id')})")
        llm_logger.debug(f"cache task: {request}")

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
        # worker_ready_signal 用于worker进程感知engine是否启动完成
        worker_ready_signal_data = np.zeros(shape=[self.cfg.worker_num_per_node], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=worker_ready_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        # launched_cache_manager_signal 用于感知engine是否启动了cache_manager
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
            launched_cache_manager_signal_data = np.zeros([1], dtype=np.int32)
            self.launched_cache_manager_signal = IPCSignal(
                name="launched_cache_manager_signal",
                array=launched_cache_manager_signal_data,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True,
            )

        # launched_expert_service_signal: Used to sense whether each expet_servic is started successfully
        if self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1:
            launched_expert_service_signal_data = np.zeros(
                shape=[self.cfg.parallel_config.data_parallel_size // self.cfg.nnode], dtype=np.int32
            )
            self.launched_expert_service_signal = IPCSignal(
                name="launched_expert_service_signal",
                array=launched_expert_service_signal_data,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True,
            )

        # loaded_model_signal: Used to detect whether each worker has completed model loading
        loaded_model_signal_data = np.zeros([1], dtype=np.int32)
        self.loaded_model_signal = IPCSignal(
            name="loaded_model_signal",
            array=loaded_model_signal_data,
            dtype=np.int32,
            suffix=self.ipc_signal_suffix,
            create=True,
        )

        if self.do_profile:
            if paddle.is_compiled_with_custom_device("iluvatar_gpu"):
                get_profile_block_num = np.zeros([self.cfg.worker_num_per_node], dtype=np.int32)
            else:
                get_profile_block_num = np.zeros([1], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.ipc_signal_suffix,
                create=True,
            )

    def _exit_sub_services(self):
        """
        exit sub services
        """
        self.running = False
        llm_logger.info("Engine shut down, exiting sub services...")

        if hasattr(self, "cache_manager_processes"):
            if hasattr(self.engine.resource_manager.cache_manager, "shm_cache_task_flag_broadcast"):
                self.engine.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear()
            if hasattr(self.engine.resource_manager.cache_manager, "cache_ready_signal"):
                self.engine.resource_manager.cache_manager.cache_ready_signal.clear()
            for p in self.cache_manager_processes:
                llm_logger.info(f"Killing cache manager process {p.pid}")
                try:
                    pgid = os.getpgid(p.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception as e:
                    console_logger.error(
                        f"Error killing cache manager process {p.pid}: {e}, {str(traceback.format_exc())}"
                    )
        self.worker_ready_signal.clear()
        self.loaded_model_signal.clear()

        if hasattr(self, "get_profile_block_num_signal"):
            self.get_profile_block_num_signal.clear()

        if hasattr(self, "worker_proc") and self.worker_proc is not None:
            try:
                pgid = os.getpgid(self.worker_proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                console_logger.error(f"Error extracting sub services: {e}, {str(traceback.format_exc())}")

        if hasattr(self, "zmq_server") and self.zmq_server is not None:
            self.zmq_server.close()

        if hasattr(self, "dp_processed"):
            for p in self.dp_processed:
                console_logger.info(f"Waiting for worker {p.pid} to exit")
                p.join()
            for p in self.dp_engine_worker_queue_server:
                p.cleanup()

    def _setting_environ_variables(self):
        """
        配置环境变量
        """
        variables = {
            "ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY": 0,
            "LOAD_STATE_DICT_THREAD_NUM": len(self.cfg.parallel_config.device_ids.split(",")),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "NCCL_ALGO": "Ring",
            "FLAGS_max_partition_size": int(os.getenv("FLAGS_max_partition_size", 1024)),
            "OMP_NUM_THREADS": int(os.getenv("OMP_NUM_THREADS", 3)),
            "FD_ENABLE_PDL": envs.FD_ENABLE_PDL,
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

        if self.cfg.scheduler_config.splitwise_role != "mixed":
            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                variables["FLAGS_use_pd_disaggregation_per_chunk"] = 1
            else:
                variables["FLAGS_use_pd_disaggregation"] = 1
            # TODO dynamic load environment variable
            if self.cfg.scheduler_config.splitwise_role == "prefill":
                variables["FLAGS_fmt_write_cache_completed_signal"] = 1

        if self.cfg.model_config.enable_mm:
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
            len(self.engine.data_processor.tokenizer.sp_model)
            if hasattr(self.engine.data_processor.tokenizer, "sp_model")
            else len(self.engine.data_processor.tokenizer.vocab)
        )

        think_end_id = self.data_processor.tokenizer.get_vocab().get("</think>", -1)
        if think_end_id > 0:
            llm_logger.info(f"Get think_end_id {think_end_id} from vocab.")
        else:
            llm_logger.info("No </think> token found in vocabulary, the model can not do reasoning.")
        image_patch_id = self.data_processor.tokenizer.get_vocab().get("<|IMAGE_PLACEHOLDER|>", -1)
        line_break_id = self.data_processor.tokenizer.get_vocab().get("\n", -1)

        ports = ",".join(self.cfg.parallel_config.engine_worker_queue_port)
        ips = None
        if self.cfg.ips is not None:
            ips = ",".join(self.cfg.ips)
        arguments = (
            f" --devices {self.cfg.parallel_config.device_ids} {py_script}"
            f" --max_num_seqs {self.cfg.scheduler_config.max_num_seqs} --max_model_len {self.cfg.model_config.max_model_len}"
            f" --gpu_memory_utilization {self.cfg.cache_config.gpu_memory_utilization}"
            f" --model {self.cfg.model_config.model!s}"
            f" --device_ids {self.cfg.parallel_config.device_ids}"
            f" --tensor_parallel_size {self.cfg.parallel_config.tensor_parallel_size}"
            f" --engine_worker_queue_port {ports}"
            f" --pod_ip {self.cfg.master_ip}"
            f" --block_size {self.cfg.cache_config.block_size}"
            f" --enc_dec_block_num {self.cfg.cache_config.enc_dec_block_num}"
            f" --eos_tokens_lens {self.engine.data_processor.eos_token_id_len}"
            f" --pad_token_id {self.engine.data_processor.pad_token_id}"
            f" --engine_pid {self.cfg.parallel_config.engine_worker_queue_port[0]}"
            f" --max_num_batched_tokens {self.cfg.scheduler_config.max_num_batched_tokens}"
            f" --splitwise_role {self.cfg.scheduler_config.splitwise_role}"
            f" --kv_cache_ratio {self.cfg.cache_config.kv_cache_ratio}"
            f" --expert_parallel_size {self.cfg.parallel_config.expert_parallel_size}"
            f" --data_parallel_size {self.cfg.parallel_config.data_parallel_size}"
            f" --quantization '{json.dumps(self.cfg.model_config.quantization)}'"
            f" --ori_vocab_size {ori_vocab_size}"
            f" --think_end_id {think_end_id}"
            f" --image_patch_id {image_patch_id}"
            f" --line_break_id {line_break_id}"
            f" --speculative_config '{self.cfg.speculative_config.to_json_string()}'"
            f" --graph_optimization_config '{self.cfg.graph_opt_config.to_json_string()}'"
            f" --guided_decoding_backend {self.cfg.structured_outputs_config.guided_decoding_backend}"
            f" --load_strategy {self.cfg.load_config.load_strategy}"
            f" --early_stop_config '{self.cfg.early_stop_config.to_json_string()}'"
            f" --reasoning_parser {self.cfg.structured_outputs_config.reasoning_parser}"
            f" --load_choices {self.cfg.load_config.load_choices}"
            f" --plas_attention_config '{self.cfg.plas_attention_config.to_json_string()}'"
            f" --ips {ips}"
            f" --max_encoder_cache {self.cfg.cache_config.max_encoder_cache}"
            f" --cache-transfer-protocol {self.cfg.cache_config.cache_transfer_protocol}"
            f" --runner {self.cfg.model_config.runner}"
            f" --convert {self.cfg.model_config.convert}"
            f" --override-pooler-config {self.cfg.model_config.override_pooler_config}"
            f" --logprobs_mode {self.cfg.model_config.logprobs_mode}"
            f" --max_logprobs {self.cfg.model_config.max_logprobs}"
            f" --eplb_config '{self.cfg.eplb_config.to_json_string()}'"
        )
        if self.cfg.structured_outputs_config.logits_processors is not None:
            arguments += f" --logits-processors {' '.join(self.cfg.structured_outputs_config.logits_processors)}"

        worker_store_true_flag = {
            "enable_expert_parallel": self.cfg.parallel_config.enable_expert_parallel,
            "enable_prefix_caching": self.cfg.cache_config.enable_prefix_caching,
            "enable_chunked_prefill": self.cfg.cache_config.enable_chunked_prefill,
            "do_profile": self.do_profile,
            "dynamic_load_weight": self.cfg.load_config.dynamic_load_weight,
            "disable_any_whitespace": self.cfg.structured_outputs_config.disable_any_whitespace,
            "disable_custom_all_reduce": self.cfg.parallel_config.disable_custom_all_reduce,
            "use_internode_ll_two_stage": self.cfg.parallel_config.use_internode_ll_two_stage,
            "disable_sequence_parallel_moe": self.cfg.parallel_config.disable_sequence_parallel_moe,
            "enable_logprob": self.cfg.model_config.enable_logprob,
            "lm_head_fp32": self.cfg.model_config.lm_head_fp32,
        }
        for worker_flag, value in worker_store_true_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag}"

        worker_default_none_flag = {
            "num_gpu_blocks_override": self.cfg.cache_config.num_gpu_blocks_override,
        }
        for worker_flag, value in worker_default_none_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag} {value}"

        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {ips} --nnodes {len(self.cfg.ips)}"
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
            prompts["max_tokens"] = self.cfg.model_config.max_model_len

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
            llm_logger.error(f"Error happened while adding request, details={e}, {str(traceback.format_exc())}")
            raise EngineError(str(e), error_code=400)

        # Get the result of the current request
        for result in self._get_generated_tokens(req_id):
            is_end = result.finished
            if stream and not is_end:
                processed = self.engine.data_processor.process_response(result)
                if processed is None:
                    continue
                output = processed.to_dict()
                yield output

            # Exit loop if termination condition is met
            if is_end:
                processed = self.engine.data_processor.process_response(result)
                output = processed.to_dict()
                llm_logger.debug(f"Generate result: {output}")
                if not stream:
                    yield output
                else:
                    output["outputs"]["text"] = ""
                    output["outputs"]["reasoning_content"] = ""
                    yield output

                self.engine.check_and_free_block_tables()

    def _stop_profile(self):
        """
        Stop profiling of the model server and reset variables.
        """
        self.do_profile = 0
        while self.get_profile_block_num_signal.value[0] == 0:
            time.sleep(1)
        num_gpu_blocks = self.get_profile_block_num_signal.value[0]
        self.cfg.cache_config.reset(num_gpu_blocks)
        self.engine.resource_manager.reset_cache_config(self.cfg.cache_config)
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
            if not current_platform.is_intel_hpu():
                device_ids = self.cfg.parallel_config.device_ids.split(",")
                self.cache_manager_processes = self.engine.start_cache_service(device_ids, self.ipc_signal_suffix)

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.engine.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.engine.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def launch_components(self):
        if self.cfg.scheduler_config.splitwise_role != "mixed":
            self.splitwise_receive_thread = threading.Thread(
                target=self.engine.split_connector.start_receiver, args=()
            )
            self.splitwise_receive_thread.daemon = True
            self.splitwise_receive_thread.start()

        role = self.cfg.scheduler_config.splitwise_role
        host_ip = self.cfg.host_ip
        disaggregate = self.cfg.disaggregate_info
        request_queues_for_dp_ipc = None
        result_queues_for_dp_ipc = None
        if self.cfg.scheduler_config.name == "splitwise":
            self.engine.scheduler.start(role, host_ip, disaggregate)
        elif self.cfg.scheduler_config.name == "dp":
            request_queues_for_dp_ipc = []
            result_queues_for_dp_ipc = []
            for i in range(self.cfg.parallel_config.data_parallel_size):
                request_queues_for_dp_ipc.append(multiprocessing.Queue())
                result_queues_for_dp_ipc.append(multiprocessing.Queue())
            self.engine.scheduler.start(
                self.cfg.node_rank * self.cfg.worker_num_per_node % self.cfg.worker_num_per_node,
                request_queues_for_dp_ipc,
                result_queues_for_dp_ipc,
            )

        if not envs.FD_ENABLE_MULTI_API_SERVER:
            if self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1:
                self.launched_expert_service_signal.value[0] = 1
                self.dp_processed = []
                self.dp_engine_worker_queue_server = []
                for i in range(
                    1,
                    self.cfg.parallel_config.data_parallel_size // self.cfg.nnode,
                ):
                    if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
                        address = (
                            self.cfg.master_ip,
                            int(self.cfg.parallel_config.engine_worker_queue_port[i]),
                        )
                    else:
                        address = f"/dev/shm/fd_task_queue_{self.cfg.parallel_config.engine_worker_queue_port[i]}.sock"

                    llm_logger.info(f"dp start queue service {address}")
                    self.dp_engine_worker_queue_server.append(
                        EngineWorkerQueue(
                            address=address,
                            is_server=True,
                            num_client=self.cfg.parallel_config.tensor_parallel_size,
                            local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                        )
                    )
                    self.dp_processed.append(
                        multiprocessing.Process(
                            target=start_data_parallel_service,
                            args=(
                                self.cfg,
                                i,
                                None,
                                request_queues_for_dp_ipc,
                                result_queues_for_dp_ipc,
                            ),
                        )
                    )
                    llm_logger.info(
                        f"Engine is initialized successfully with {self.cfg.parallel_config.tensor_parallel_size}"
                        + f" data parallel id {i}"
                    )
                    self.dp_processed[-1].start()
                    while self.launched_expert_service_signal.value[i] == 0:
                        time.sleep(1)

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
        except Exception:
            pass
        return True
