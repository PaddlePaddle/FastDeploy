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

import asyncio
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
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import numpy as np
import paddle
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.expert_service import start_data_parallel_service
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import EngineWorkerQueue, IPCSignal
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import EngineError, console_logger, envs, llm_logger


class AsyncRequestQueue:
    """Async request output queue for managing single request output stream"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.queue: asyncio.Queue[Union[RequestOutput, Exception]] = asyncio.Queue()
        self._finished = False
        self._cache_size = 0

    async def put(self, output: RequestOutput) -> None:
        """Put output to queue with memory allocation optimization"""
        if isinstance(output, RequestOutput) and output.finished:
            self._finished = True
        await self.queue.put(output)
        self._cache_size += 1

    async def put_error(self, error: Exception) -> None:
        """Put error to queue"""
        self._finished = True
        await self.queue.put(error)

    async def get(self) -> RequestOutput:
        """Get output, raise exception if it's an error"""
        result = await self.queue.get()
        self._cache_size = max(0, self._cache_size - 1)
        if isinstance(result, Exception):
            raise result
        return result

    def get_nowait(self) -> Optional[RequestOutput]:
        """Non-blocking get output"""
        try:
            result = self.queue.get_nowait()
            self._cache_size = max(0, self._cache_size - 1)
            if isinstance(result, Exception):
                raise result
            return result
        except asyncio.QueueEmpty:
            return None

    @property
    def finished(self) -> bool:
        """Check if request is completed"""
        return self._finished

    @property
    def size(self) -> int:
        """Return queue size for performance monitoring"""
        return self._cache_size


class AsyncOutputProcessor:
    """Async output processor responsible for distributing engine outputs to corresponding request queues"""

    def __init__(self, tokenizer=None):
        self.request_queues: Dict[str, AsyncRequestQueue] = {}
        self.tokenizer = tokenizer

    async def register_request(self, request_id: str, queue: AsyncRequestQueue) -> None:
        """Register request queue"""
        self.request_queues[request_id] = queue

    async def process_outputs(self, outputs: Dict[str, List[RequestOutput]]) -> None:
        """Process engine outputs and distribute to corresponding request queues"""
        if not outputs:
            return

        finished_requests = []

        for request_id, output_list in outputs.items():
            if request_id not in self.request_queues:
                continue

            queue = self.request_queues[request_id]

            # Ensure output_list is in list format
            if not isinstance(output_list, list):
                output_list = [output_list]

            for output in output_list:
                # Process single output
                processed_output = self._process_single_output(output)
                await queue.put(processed_output)

                if processed_output.finished:
                    finished_requests.append(request_id)

        # Clean up completed requests
        for request_id in finished_requests:
            self.request_queues.pop(request_id, None)

    def _process_single_output(self, output: RequestOutput) -> RequestOutput:
        """Process single output for token decoding"""

        try:
            token_ids = output.outputs.token_ids
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            output.outputs.text = decoded_text
        except Exception:
            if not hasattr(output.outputs, "text"):
                output.outputs.text = ""

        return output

    async def abort_request(self, request_id: str) -> None:
        """Abort request and clean up related resources"""
        if request_id in self.request_queues:
            queue = self.request_queues.pop(request_id)
            await queue.put_error(EngineError("Request aborted", error_code=499))

    async def propagate_error(self, error: Exception) -> None:
        """Propagate error to all active request queues"""
        tasks = []
        for queue in list(self.request_queues.values()):
            if not queue.finished:
                tasks.append(queue.put_error(error))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.request_queues.clear()


class AsyncLLMEngine:
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
        Creates an AsyncLLMEngine from the provided engine arguments.

        Args:
            engine_args (EngineArgs): Engine arguments object.

        Returns:
            AsyncLLMEngine: Instance of the AsyncLLMEngine class.
        """
        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the AsyncLLMEngine.
        return cls(cfg=config)

    def __init__(self, cfg):
        """
        Initializes the AsyncLLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg
        self.running = True
        self.is_started = False

        self.input_processor = InputPreprocessor(
            cfg.model_config,
            cfg.structured_outputs_config.reasoning_parser,
            cfg.limit_mm_per_prompt,
            cfg.mm_processor_kwargs,
            cfg.tool_parser,
        )
        self.engine_service = EngineService(cfg)

        if self.cfg.cache_config.num_gpu_blocks_override is None:
            self.do_profile = 1
        else:
            self.do_profile = 0

        # Create async output processor, pass tokenizer for decoding
        tokenizer = None
        if hasattr(self, "input_processor") and hasattr(self.input_processor, "tokenizer"):
            tokenizer = self.input_processor.tokenizer
        elif hasattr(self, "data_processor") and hasattr(self.data_processor, "tokenizer"):
            tokenizer = self.data_processor.tokenizer

        self.output_processor = AsyncOutputProcessor(tokenizer=tokenizer)

        self.output_handler: Optional[asyncio.Task] = None

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

        main_process_metrics.set_cache_config_info(obj=self.cfg.cache_config)

    def start(self):
        """
        Initializes the engine and starts its sub-services.
        """
        assert not self.is_started, "The engine is already started."
        start_time = time.time()

        self.ipc_signal_suffix = self.cfg.parallel_config.engine_worker_queue_port[0]
        self._init_worker_signals()

        self.data_processor = self.input_processor.create_processor()
        self.engine_service.data_processor = self.data_processor

        # Launch components: scheduler, cache_manager, expert_service et.al.
        self.launch_components()

        # Update output processor tokenizer
        if hasattr(self.data_processor, "tokenizer") and self.data_processor.tokenizer:
            self.output_processor.tokenizer = self.data_processor.tokenizer

        self.engine_service.start()

        # If block number is specified and model is deployed in splitwise mode, start cache manager first
        if not self.do_profile and self.cfg.scheduler_config.splitwise_role != "mixed":
            device_ids = self.cfg.parallel_config.device_ids.split(",")
            self.cache_manager_processes = self.engine_service.start_cache_service(device_ids, self.ipc_signal_suffix)

        # Start workers
        self.worker_proc = self._start_worker_service()
        console_logger.info("Waiting worker processes ready...")
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
        elif self.cfg.cache_config.enable_prefix_caching:
            device_ids = self.cfg.parallel_config.device_ids.split(",")
            self.cache_manager_processes = self.engine_service.start_cache_service(device_ids, self.ipc_signal_suffix)

        # Set cache manager signal
        if self.cfg.scheduler_config.splitwise_role != "mixed":
            self.launched_cache_manager_signal.value[0] = 1

        # Worker launched
        self.check_worker_initialize_status_func_thread.join()
        if not result_container["worker_is_alive"]:
            console_logger.error("Failed to launch worker processes, check log/workerlog.* for more details.")
            return False

        console_logger.info(f"Worker processes are launched with {time.time() - start_time} seconds.")

        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._start_output_handler()
        except RuntimeError:
            pass

        self.is_started = True
        return True

    async def get_model_config(self):
        """Get model configuration"""
        return self.cfg.model_config

    async def get_tokenizer(self):
        """Get tokenizer"""
        if hasattr(self, "data_processor"):
            return self.data_processor.tokenizer
        return None

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

    async def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None,
        **kwargs,
    ) -> AsyncRequestQueue:
        """
        Async add request

        Args:
            request_id: Request ID
            prompt: Input prompt
            sampling_params: Sampling parameters
            arrival_time: Arrival time
            **kwargs: Other parameters

        Returns:
            AsyncRequestQueue: Request output queue
        """
        if not self.is_started or self.engine_service is None:
            raise EngineError("Engine not started. Call start() first.", error_code=500)

        if request_id is None:
            request_id = str(uuid.uuid4())

        # Create output queue
        output_queue = AsyncRequestQueue(request_id)

        if arrival_time is None:
            arrival_time = time.time()

        if isinstance(prompt, str):
            prompt = {
                "prompt": prompt,
                "request_id": request_id,
            }
        elif isinstance(prompt, list) and isinstance(prompt[0], int):
            prompt = {
                "prompt_token_ids": prompt,
                "request_id": request_id,
            }
        elif isinstance(prompt, dict):
            prompt["request_id"] = request_id
        else:
            raise TypeError(f"Invalid type for 'prompt': {type(prompt)}, expected one of ['str', 'list', 'dict'].")

        if sampling_params is not None:
            prompt.update(asdict(sampling_params))

        try:
            request = Request.from_dict(prompt)
            request.llm_engine_recv_req_timestamp = time.time()

            # Check if already preprocessed by AsyncEngineClient
            is_preprocessed = prompt.get("_preprocessed", False)

            # Set sampling_params
            if sampling_params is not None:
                request.sampling_params = sampling_params

            # Preprocess request
            request = self.data_processor.process_request(request, self.cfg.model_config.max_model_len, **kwargs)

            prompt_token_ids_len = len(request.prompt_token_ids)
            request.prompt_token_ids_len = prompt_token_ids_len
            request.need_prefill_tokens = prompt_token_ids_len

            if not is_preprocessed:
                request.preprocess_start_time = arrival_time
                input_ids_len = request.prompt_token_ids_len

                request.set(
                    "max_tokens",
                    min(
                        self.cfg.model_config.max_model_len - input_ids_len,
                        request.get("max_tokens"),
                    ),
                )

                if request.get("reasoning_max_tokens") is None:
                    default_reasoning_max_tokens = max(int(request.get("max_tokens") * 0.8), 1)
                    request.set("reasoning_max_tokens", default_reasoning_max_tokens)

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

                request.preprocess_end_time = time.time()

            # Register output queue first, then add request
            await self.output_processor.register_request(request_id, output_queue)

            # TODO: Optimize architecture to implement async transmission to worker
            self.engine_service.scheduler.put_requests([request])

            return output_queue

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Request processing failed: {e}", error_code=400)

    async def generate(
        self,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Async generation interface

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Request ID
            **kwargs: Other parameters

        Yields:
            RequestOutput: Generated output
        """
        if not self.is_started:
            raise EngineError("Engine not started. Call start() first.", error_code=500)

        try:
            # Ensure output processor is running
            self._start_output_handler()

            # Async add request
            output_queue = await self.add_request(request_id, prompt, sampling_params, **kwargs)

            finished = False

            while not finished:
                # Prefer non-blocking get first
                output = output_queue.get_nowait() or await output_queue.get()
                finished = output.finished
                yield output

        except EngineError:
            raise
        except GeneratorExit:
            llm_logger.info(f"Request {request_id} generator exit (outer)")
            return
        except Exception as e:
            await self.abort_request(request_id)
            llm_logger.error(f"Request {request_id} failed: {e}")
            raise EngineError(str(e), error_code=500) from e

    async def abort_request(self, request_id: str) -> None:
        """
        Abort the specified request

        Args:
            request_id: Request ID to abort
        """
        try:
            await self.output_processor.abort_request(request_id)
            llm_logger.info(f"Aborted request {request_id}")
        except Exception as e:
            llm_logger.error(f"Failed to abort request {request_id}: {e}")

    def _start_output_handler(self) -> None:
        """Start background output processing task"""
        if self.output_handler is not None:
            return

        async def output_handler_loop():
            """Background loop: get results from engine service and distribute to corresponding queues"""
            try:
                while self.running:
                    # Check engine service status
                    if self.engine_service is None:
                        await asyncio.sleep(0.001)
                        continue

                    results = self.engine_service.scheduler.get_results()

                    if not results:
                        # No results, minimal delay to yield control
                        await asyncio.sleep(0)
                        continue

                    await self.output_processor.process_outputs(results)

            except GeneratorExit:
                llm_logger.info("Output handler loop received GeneratorExit, shutting down gracefully")
            except asyncio.CancelledError:
                llm_logger.info("Output handler loop cancelled, shutting down gracefully")
            except Exception as e:
                llm_logger.exception("AsyncLLM output_handler failed")
                await self.output_processor.propagate_error(e)
            finally:
                llm_logger.info("Output handler loop finished")

        self.output_handler = asyncio.create_task(output_handler_loop())
        llm_logger.info("Output handler started")

    async def shutdown(self):
        """
        Gracefully shutdown AsyncLLM engine
        """
        llm_logger.info("Starting AsyncLLM shutdown...")

        self.running = False

        # Clean up request queues in output processor (clean queues first to avoid new tasks)
        if hasattr(self, "output_processor"):
            try:
                await self.output_processor.propagate_error(Exception("AsyncLLM shutdown"))
            except Exception as e:
                llm_logger.warning(f"Error while cleaning output processor: {e}")

        # Shutdown async output processor
        if hasattr(self, "output_handler") and self.output_handler and not self.output_handler.done():
            self.output_handler.cancel()
            try:
                await asyncio.wait_for(self.output_handler, timeout=2.0)
            except asyncio.CancelledError:
                llm_logger.info("Output handler cancelled successfully")
            except asyncio.TimeoutError:
                llm_logger.warning("Output handler cancellation timeout, proceeding with cleanup")
            except Exception as e:
                llm_logger.warning(f"Error while cancelling output handler: {e}")
            finally:
                self.output_handler = None

        # Shutdown underlying engine service
        if hasattr(self, "engine_service") and self.engine_service is not None:
            llm_logger.info("Stopping engine service...")
            try:
                if hasattr(self.engine_service, "running"):
                    self.engine_service.running = False

                self._exit_sub_services()
            except Exception as e:
                llm_logger.error(f"Error while stopping engine service: {e}")

        self.is_started = False
        llm_logger.info("AsyncLLM shutdown completed")

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

        if hasattr(self, "cache_manager_processes"):
            self.engine_service.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear()
            self.engine_service.resource_manager.cache_manager.cache_ready_signal.clear()
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
            "FLAGS_use_append_attn": 1,
            "NCCL_ALGO": "Ring",
            "FLAGS_max_partition_size": int(os.getenv("FLAGS_max_partition_size", 1024)),
            "OMP_NUM_THREADS": int(os.getenv("OMP_NUM_THREADS", 3)),
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
        uncache_worker_stdout = "" if os.getenv("UNCACHE_WORKER_STDOUT", "0") == "1" else "-u"
        pd_cmd = f"{command_prefix} {sys.executable} {uncache_worker_stdout} -m paddle.distributed.launch"
        pd_cmd = pd_cmd + f" --log_dir {log_dir}"

        worker_path = "../worker/worker_process.py"
        py_script = os.path.join(current_dir_path, worker_path)

        ori_vocab_size = (
            len(self.data_processor.tokenizer.sp_model)
            if hasattr(self.data_processor.tokenizer, "sp_model")
            else len(self.data_processor.tokenizer.vocab)
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
            f" --eos_tokens_lens {self.data_processor.eos_token_id_len}"
            f" --pad_token_id {self.data_processor.pad_token_id}"
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
            f" --cache-transfer-protocol {self.cfg.cache_config.cache_transfer_protocol}"
            f" --runner {self.cfg.model_config.runner}"
            f" --convert {self.cfg.model_config.convert}"
            f" --override-pooler-config {self.cfg.model_config.override_pooler_config}"
            f" --logprobs_mode {self.cfg.model_config.logprobs_mode}"
            f" --max_logprobs {self.cfg.model_config.max_logprobs}"
            f" --eplb_config '{self.cfg.eplb_config.to_json_string()}'"
        )

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

    def _stop_profile(self):
        """
        Stop profiling of the model server and reset variables.
        """
        self.do_profile = 0
        while self.get_profile_block_num_signal.value[0] == 0:
            time.sleep(1)
        num_gpu_blocks = self.get_profile_block_num_signal.value[0]
        self.cfg.cache_config.reset(num_gpu_blocks)
        self.engine_service.resource_manager.reset_cache_config(self.cfg.cache_config)
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
            device_ids = self.cfg.parallel_config.device_ids.split(",")
            self.cache_manager_processes = self.engine_service.start_cache_service(device_ids, self.ipc_signal_suffix)

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.engine_service.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.engine_service.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def launch_components(self):
        if self.cfg.scheduler_config.splitwise_role != "mixed":
            self.engine_service.split_mode_get_tasks()
            if self.cfg.scheduler_config.name == "splitwise":
                self.splitwise_receive_thread = threading.Thread(
                    target=self.engine_service.split_connector.start_receiver, args=()
                )
                self.splitwise_receive_thread.daemon = True
                self.splitwise_receive_thread.start()

        self.cfg.init_cache_info()

        role = self.cfg.scheduler_config.splitwise_role
        host_ip = self.cfg.host_ip
        disaggregate = self.cfg.disaggregate_info
        if self.cfg.scheduler_config.name == "splitwise":
            self.engine_service.scheduler.start(role, host_ip, disaggregate)

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
