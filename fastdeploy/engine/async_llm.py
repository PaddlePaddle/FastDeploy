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

import inspect
import os
import signal
import time
import traceback
import uuid
import weakref
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import numpy as np
import zmq

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.openai.utils import DealerConnectionManager
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.inter_communicator.zmq_client import ZmqIpcClient
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import EngineError, envs, llm_logger


class AsyncOutputProcessor:
    """Async output processor responsible for distributing engine outputs to corresponding request queues"""

    def __init__(self, data_processor=None):
        """
        Args:
            data_processor: The data processor created by InputPreprocessor,
                used to post-process RequestOutput (decode token_ids, reasoning, tools, etc.).
        """
        self.data_processor = data_processor

    def _process_output(
        self,
        response_item: RequestOutput | Dict[str, Any],
        stream: bool = True,
        enable_thinking: bool = False,
        include_stop_str_in_output: bool = False,
    ) -> Dict[str, Any]:
        """Process a single response dict via data_processor.process_response_dict.

        This mirrors the behavior of ChatResponseProcessor in the OpenAI serving
        path: operate on a dict representation and return a dict. On any error
        we fall back to the original dict and ensure ``outputs.text`` exists to
        avoid cascading failures.
        """

        try:
            processed = self.data_processor.process_response_dict(
                response_item,
                stream=stream,
                enable_thinking=enable_thinking,
                include_stop_str_in_output=include_stop_str_in_output,
            )
            # Some processors may return None when there is no valid text.
            if processed is None:
                outputs = response_item.get("outputs") or {}
                if "text" not in outputs:
                    outputs["text"] = ""
                    response_item["outputs"] = outputs
                return response_item
            return processed
        except Exception:
            outputs = response_item.get("outputs") or {}
            if "text" not in outputs:
                outputs["text"] = ""
                response_item["outputs"] = outputs
            return response_item


class EngineServiceClient:
    """
    Base engine service client, responsible for managing EngineService lifecycle.
    """

    def __init__(self, cfg, pid):
        self.cfg = cfg
        self.engine_process = None
        self.engine_pid = pid
        self._running = False

        llm_logger.info(f"EngineServiceClient initialized with engine_pid: {self.engine_pid}")

    async def start(self):
        """Start engine service process"""
        try:
            # Start independent engine process
            self._start_engine_process()

            # Wait for engine to be ready
            if not self._wait_engine_ready():
                raise EngineError("Engine failed to start within timeout", error_code=500)

            self._running = True
            llm_logger.info("EngineServiceClient started successfully")

        except Exception as e:
            llm_logger.error(f"Failed to start EngineServiceClient: {e}")
            raise
        return True

    def _start_engine_process(self):
        """Start engine process"""
        try:
            import multiprocessing

            self.shutdown_signal = multiprocessing.Value("i", 0)  # 0=running, 1=shutdown

            def run_engine():
                engine = None

                def signal_handler(signum, frame):
                    llm_logger.info(f"Engine process received signal {signum}, initiating shutdown...")
                    if engine:
                        engine.running = False

                # Register signal handlers
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)

                try:
                    engine = EngineService(self.cfg, use_async_llm=True)
                    # Start engine with ZMQ service
                    engine.start(async_llm_pid=self.engine_pid)

                    # Keep engine running until shutdown signal is received
                    while self.shutdown_signal.value == 0 and getattr(engine, "running", True):
                        time.sleep(0.5)

                except Exception as e:
                    llm_logger.error(f"Engine process error: {e}, {str(traceback.format_exc())}")
                finally:
                    if engine and hasattr(engine, "_exit_sub_services"):
                        try:
                            engine._exit_sub_services()
                            llm_logger.info("Engine process cleanup completed")
                        except Exception as e:
                            llm_logger.error(f"Error during engine cleanup: {e}")

            self.engine_process = multiprocessing.Process(target=run_engine)
            self.engine_process.start()

            llm_logger.info(f"Started engine process with PID: {self.engine_process.pid}")

        except Exception as e:
            llm_logger.error(f"Failed to start engine process: {e}")
            raise

    def _wait_engine_ready(self) -> bool:
        """Wait for engine and workers to be fully ready"""
        max_wait_time = 500  # seconds
        wait_interval = 1
        elapsed_time = 0

        llm_logger.info("Waiting for engine and workers to be ready...")

        # Use IPC signals to check engine readiness
        # Get the correct suffix
        ipc_suffix = (
            self.cfg.parallel_config.engine_worker_queue_port[0]
            if hasattr(self.cfg, "parallel_config")
            else self.engine_pid
        )

        # Check if loaded_model_signal exists and is ready
        loaded_model_signal = None

        while elapsed_time < max_wait_time:
            # Try to connect to loaded_model_signal
            if loaded_model_signal is None:
                try:
                    loaded_model_signal = IPCSignal(
                        name="loaded_model_signal",
                        array=np.zeros([1], dtype=np.int32),
                        dtype=np.int32,
                        suffix=ipc_suffix,
                        create=False,
                    )
                except:
                    # Signal not ready yet
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                    continue

            # Check if workers have loaded models
            if loaded_model_signal.value[0] > 0:
                llm_logger.info("Workers have loaded models successfully")
                # Give ZMQ service more time to fully start
                llm_logger.info("Waiting additional time for ZMQ service to be ready...")
                time.sleep(5)  # Wait for ZMQ service startup + recv_result_handle
                return True

            time.sleep(wait_interval)
            elapsed_time += wait_interval

            if elapsed_time % 10 == 0:  # Log every 10 seconds
                llm_logger.info(f"Waiting for workers to load models... ({elapsed_time}s)")

        return False

    def shutdown(self):
        """Shutdown engine service process"""
        llm_logger.info("Shutting down EngineServiceClient...")

        self._running = False

        # Send graceful shutdown signal to engine process
        if hasattr(self, "shutdown_signal"):
            llm_logger.info("Sending shutdown signal to engine process...")
            self.shutdown_signal.value = 1

        # Wait for engine process to shutdown
        if self.engine_process and self.engine_process.is_alive():
            llm_logger.info("Waiting for engine process to shutdown...")
            self.engine_process.terminate()
            self.engine_process.join(timeout=5)
            if self.engine_process.is_alive():
                llm_logger.warning("Force killing engine process...")
                self.engine_process.kill()

        llm_logger.info("EngineServiceClient shutdown completed")


class AsyncLLM(EngineServiceClient):
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
        do_profile (int): Flag indicating if profiling is enabled.
    """

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs, pid):
        """
        Creates an AsyncLLM client from the provided engine arguments.

        Args:
            engine_args (EngineArgs): Engine arguments object.

        Returns:
            AsyncLLM: Instance of the AsyncLLM class.
        """
        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the AsyncLLM client.
        return cls(cfg=config, pid=pid)

    def __init__(self, cfg, pid):
        """
        Initializes the AsyncLLM client with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        super().__init__(cfg, pid)
        self.cfg = cfg
        self.running = True
        self._prompt_metadata: Dict[str, Dict[str, Any]] = {}

        self.input_processor = InputPreprocessor(
            cfg.model_config,
            cfg.structured_outputs_config.reasoning_parser,
            cfg.limit_mm_per_prompt,
            cfg.mm_processor_kwargs,
            cfg.tool_parser,
        )
        # Create data processor
        self.data_processor = self.input_processor.create_processor()

        # Create high-performance async connection manager
        self.connection_manager = None
        self.request_client = None

        # Output processor uses data_processor for post-processing engine outputs
        self.output_processor = AsyncOutputProcessor(self.data_processor)

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

        main_process_metrics.set_cache_config_info(obj=self.cfg.cache_config)

    async def init_connections(self):
        """Initialize high-performance ZMQ connections"""
        try:
            # Create ZMQ client for sending requests
            self.request_client = ZmqIpcClient(name=self.engine_pid, mode=zmq.PUSH)
            self.request_client.connect()

            # Create high-performance async connection manager for receiving responses
            self.connection_manager = DealerConnectionManager(
                pid=self.engine_pid, max_connections=int(os.getenv("FD_DEALER_CONNECTIONS", 50))
            )

            if not self.connection_manager.running:
                await self.connection_manager.initialize()

            llm_logger.info("High-performance ZMQ connections initialized successfully")
        except Exception as e:
            llm_logger.error(f"Failed to initialize ZMQ connections: {e}")
            raise

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
    ):
        """
        Async add request

        Args:
            request_id: Request ID
            prompt: Input prompt
            sampling_params: Sampling parameters
            arrival_time: Arrival time
            **kwargs: Other parameters

        """

        if request_id is None:
            request_id = str(uuid.uuid4())

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
        prompt["metrics"] = {}

        try:
            # Check if already preprocessed by api_server
            is_preprocessed = prompt.get("_preprocessed", False)

            if inspect.iscoroutinefunction(self.data_processor.process_request_dict):
                request = await self.data_processor.process_request_dict(prompt, self.cfg.model_config.max_model_len)
            else:
                request = self.data_processor.process_request_dict(prompt, self.cfg.model_config.max_model_len)

            request["prompt_token_ids_len"] = len(request["prompt_token_ids"])

            # Cache prompt metadata for later enrichment of async responses
            req_id = request.get("request_id")
            self._prompt_metadata[req_id] = {
                "prompt_token_ids": request.get("prompt_token_ids"),
                "prompt_tokens": request.get("prompt_tokens"),
            }
            request["need_prefill_tokens"] = request["prompt_token_ids_len"]

            if not is_preprocessed:
                request["metrics"]["preprocess_start_time"] = arrival_time
                input_ids_len = request["prompt_token_ids_len"]

                request["max_tokens"] = min(
                    self.cfg.model_config.max_model_len - input_ids_len, request.get("max_tokens")
                )

                min_tokens = request.get("min_tokens", 1)
                if input_ids_len + min_tokens >= self.cfg.model_config.max_model_len:
                    error_msg = (
                        f"Input text is too long, length of prompt token({input_ids_len}) "
                        f"+ min_dec_len ({min_tokens}) >= max_model_len "
                    )
                    llm_logger.error(error_msg)
                    raise EngineError(error_msg, error_code=400)

                request["metrics"]["preprocess_end_time"] = time.time()
                preprocess_cost_time = (
                    request["metrics"]["preprocess_end_time"] - request["metrics"]["preprocess_start_time"]
                )
                llm_logger.info(
                    f"Cache request with request_id ({request.get('request_id')}), "
                    f"preprocess time cost {preprocess_cost_time}"
                )
            if not envs.ENABLE_V1_DATA_PROCESSOR and self.cfg.model_config.enable_mm:
                self.request_client.send_pyobj(request)
            else:
                self.request_client.send_json(request)

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"async_llm add request failed: {e}", error_code=400)

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
            sampling_params: Sampling parameters. If `sampling_params.n > 1`,
                will generate `n` completions sequentially.
            request_id: Request ID
            **kwargs: Other parameters

        Yields:
            RequestOutput: Generated output
        """

        num_choices = sampling_params.n if sampling_params is not None and sampling_params.n else 1
        stream = True
        include_stop_str_in_output = False
        enable_thinking = kwargs.pop("enable_thinking", False)

        if isinstance(prompt, dict):
            num_choices = prompt.get("n")
            stream = prompt.get("stream", True)
            include_stop_str_in_output = prompt.get("include_stop_str_in_output", False)

        # Ensure ZMQ client and connection manager are initialized in current process
        if (
            self.request_client is None
            or self.connection_manager is None
            or not getattr(self.connection_manager, "running", False)
        ):
            raise EngineError(
                "AsyncLLM engine not initialized. Call init_connections() before generate.",
                error_code=500,
            )

        # Build request ids and connection key
        if num_choices <= 1:
            # Single-choice: keep user-provided request_id semantics
            child_request_ids = [request_id or str(uuid.uuid4())]
            conn_request_id = child_request_ids[0]
        else:
            # Multi-choice: use unified "cmpl-" base id so DealerConnectionManager
            # can merge cmpl-xxx_0, cmpl-xxx_1, ... back to the same response queue.
            user_request_id = request_id or str(uuid.uuid4())
            conn_request_id = f"cmpl-{user_request_id}"
            child_request_ids = [f"{conn_request_id}_{i}" for i in range(num_choices)]

        try:
            # 1) Send all sub-requests to engine
            for child_request_id in child_request_ids:
                await self.add_request(child_request_id, prompt, sampling_params, **kwargs)

            # 2) Get a shared connection for conn_request_id and handshake all sub-requests
            dealer, response_queue = await self.connection_manager.get_connection(
                request_id=conn_request_id, num_choices=num_choices
            )

            for child_request_id in child_request_ids:
                dealer.write([b"", child_request_id.encode("utf-8")])

            # 3) Stream responses from all choices interleaved
            remaining = num_choices
            while remaining > 0:
                response_list = await response_queue.get()

                for response_item in response_list:
                    if (
                        isinstance(response_item, dict) or isinstance(response_item, Request)
                    ) and "request_id" in response_item:
                        req_id = response_item.get("request_id")

                        # First, use output_processor to post-process the raw dict
                        if hasattr(self, "output_processor"):
                            processed_output = self.output_processor._process_output(
                                response_item,
                                stream=stream,
                                enable_thinking=enable_thinking,
                                include_stop_str_in_output=include_stop_str_in_output,
                            )
                        else:
                            processed_output = response_item
                        if not envs.ENABLE_V1_DATA_PROCESSOR:
                            processed_output = RequestOutput.from_dict(processed_output)
                        # Enrich outputs with prompt metadata on the first packet
                        if req_id:
                            prompt_meta = self._prompt_metadata.get(req_id)
                            if prompt_meta is not None and processed_output.outputs.send_idx == 0:
                                processed_output.prompt_token_ids = prompt_meta.get("prompt_token_ids")
                                processed_output.prompt = prompt_meta.get("prompt_tokens")
                                self._prompt_metadata.pop(req_id, None)

                        if processed_output.finished:
                            remaining -= 1

                        yield processed_output

        except GeneratorExit:
            llm_logger.info(f"Request {conn_request_id} generator exit (outer)")
            return
        except Exception as e:
            llm_logger.error(f"Request {conn_request_id} failed: {e}")
            raise EngineError(str(e), error_code=500) from e
        finally:
            # Ensure request_map/request_num are cleaned up
            try:
                await self.connection_manager.cleanup_request(conn_request_id)
            except Exception:
                pass

    async def abort_request(self, request_id: str) -> None:
        """
        Abort the specified request

        Args:
            request_id: Request ID to abort
        """
        try:
            # Clean up request through DealerConnectionManager
            if hasattr(self, "connection_manager") and self.connection_manager:
                await self.connection_manager.cleanup_request(request_id)
            llm_logger.info(f"Aborted request {request_id}")
        except Exception as e:
            llm_logger.error(f"Failed to abort request {request_id}: {e}")

    async def shutdown(self):
        """
        Gracefully shutdown AsyncLLM engine
        """
        llm_logger.info("Starting AsyncLLM shutdown...")

        self.running = False

        # Close high-performance connection manager
        if hasattr(self, "connection_manager") and self.connection_manager is not None:
            llm_logger.info("Stopping connection manager...")
            try:
                await self.connection_manager.close()
            except Exception as e:
                llm_logger.error(f"Error while stopping connection manager: {e}")

        # Close ZMQ client
        if hasattr(self, "request_client") and self.request_client is not None:
            llm_logger.info("Closing request client...")
            try:
                self.request_client.close()
            except Exception as e:
                llm_logger.warning(f"Error closing request client: {e}")

        # Shutdown engine service process
        try:
            super().shutdown()
        except Exception as e:
            llm_logger.error(f"Error while stopping engine service process: {e}")

        llm_logger.info("AsyncLLM shutdown completed")

    def _exit_sub_services(self):
        """
        Clean up any remaining resources
        """
        pass
