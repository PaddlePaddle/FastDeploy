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

import inspect
import os
import time
import traceback
import uuid

import numpy as np
from filelock import FileLock

from fastdeploy import envs
from fastdeploy.config import ModelConfig
from fastdeploy.entrypoints.openai.utils import DealerConnectionManager
from fastdeploy.envs import FD_SUPPORT_MAX_CONNECTIONS
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import (
    IPCSignal,
    KVCacheStatus,
    ModelWeightsStatus,
    PrefixTreeStatus,
    ZmqIpcClient,
)
from fastdeploy.metrics.work_metrics import work_process_metrics
from fastdeploy.multimodal.registry import MultimodalRegistry
from fastdeploy.platforms import current_platform
from fastdeploy.utils import (
    EngineError,
    ParameterError,
    StatefulSemaphore,
    api_server_logger,
)


class EngineClient:
    """
    EngineClient is a class that handles the communication between the client and the server.
    """

    def __init__(
        self,
        model_name_or_path,
        tokenizer,
        max_model_len,
        tensor_parallel_size,
        pid,
        port,
        limit_mm_per_prompt,
        mm_processor_kwargs,
        # enable_mm=False,
        reasoning_parser=None,
        data_parallel_size=1,
        enable_logprob=False,
        workers=1,
        tool_parser=None,
        enable_prefix_caching=None,
        splitwise_role=None,
    ):
        architectures = ModelConfig({"model": model_name_or_path}).architectures[0]
        if MultimodalRegistry.contains_model(architectures):
            self.enable_mm = True
        else:
            self.enable_mm = False

        input_processor = InputPreprocessor(
            tokenizer,
            reasoning_parser,
            limit_mm_per_prompt,
            mm_processor_kwargs,
            self.enable_mm,
            tool_parser,
        )
        self.enable_logprob = enable_logprob
        self.reasoning_parser = reasoning_parser
        self.data_processor = input_processor.create_processor()
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_splitwise = splitwise_role != "mixed"
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8

        if tensor_parallel_size <= max_chips_per_node:
            self.is_master = True
        else:
            self.is_master = False

        array_size = min(max_chips_per_node, tensor_parallel_size)
        self.worker_healthy_live_recorded_time_array = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=self.worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=port,
            create=False,
        )
        self.semaphore = StatefulSemaphore((FD_SUPPORT_MAX_CONNECTIONS + workers - 1) // workers)
        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=port,
            create=False,
        )
        prefix_tree_status = np.zeros([1], dtype=np.int32)
        self.prefix_tree_status_signal = IPCSignal(
            name="prefix_tree_status",
            array=prefix_tree_status,
            dtype=np.int32,
            suffix=port,
            create=False,
        )
        kv_cache_status = np.zeros([1], dtype=np.int32)
        self.kv_cache_status_signal = IPCSignal(
            name="kv_cache_status",
            array=kv_cache_status,
            dtype=np.int32,
            suffix=port,
            create=False,
        )
        self.connection_manager = DealerConnectionManager(
            pid, max_connections=int(os.getenv("FD_DEALER_CONNECTIONS", 50))
        )
        self.connection_initialized = False
        self.clear_update_lock = FileLock(f"/tmp/fd_weight_clear_update_lock__pid{pid}_port{port}.lock")

    def create_zmq_client(self, model, mode):
        """
        Create a ZMQ client.
        """
        self.zmq_client = ZmqIpcClient(model, mode)
        self.zmq_client.connect()

    async def format_and_add_data(self, prompts: dict):
        """
        Format the request data and send the request to the server.
        """
        if "request_id" not in prompts:
            request_id = str(uuid.uuid4())
            prompts["request_id"] = request_id

        if "max_tokens" not in prompts:
            prompts["max_tokens"] = self.max_model_len - 1

        await self.add_requests(prompts)
        return prompts["prompt_token_ids"]

    async def add_requests(self, task):
        """
        Add a new request to the queue.

        Args:
            task: Request A dictionary representing the request.
            sampling_params: A dictionary representing the sampling parameters.

        Returns:
            None
        """

        task["preprocess_start_time"] = time.time()
        try:
            chat_template_kwargs = task.get("chat_template_kwargs") or {}
            chat_template_kwargs.update({"chat_template": task.get("chat_template"), "tools": task.get("tools")})
            task["chat_template_kwargs"] = chat_template_kwargs
            if inspect.iscoroutinefunction(self.data_processor.process_request_dict):
                await self.data_processor.process_request_dict(task, self.max_model_len)
            else:
                self.data_processor.process_request_dict(task, self.max_model_len)

            task["prompt_token_ids_len"] = len(task["prompt_token_ids"])
            input_ids_len = task["prompt_token_ids_len"]
            task["max_tokens"] = min(self.max_model_len - input_ids_len, task.get("max_tokens"))
            min_tokens = task.get("min_tokens", 1)
            if "messages" in task:
                del task["messages"]
            api_server_logger.info(f"task['max_tokens']:{task['max_tokens']}")
            work_process_metrics.request_params_max_tokens.observe(task["max_tokens"])
            work_process_metrics.prompt_tokens_total.inc(input_ids_len)
            work_process_metrics.request_prompt_tokens.observe(input_ids_len)
        except Exception as e:
            api_server_logger.error(f"add_requests error: {e}, {str(traceback.format_exc())}")
            raise EngineError(str(e), error_code=400)

        if input_ids_len + min_tokens >= self.max_model_len:
            error_msg = (
                f"Input text is too long, input_ids_len ({input_ids_len}) "
                f"+ min_tokens({min_tokens}) >= max_model_len({self.max_model_len})"
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.max_model_len})."
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if "stop_seqs_len" in task:
            stop_seqs_len = task["stop_seqs_len"]
            max_stop_seqs_num = int(envs.FD_MAX_STOP_SEQS_NUM)
            if len(stop_seqs_len) > max_stop_seqs_num:
                error_msg = (
                    f"Length of stop ({stop_seqs_len}) exceeds the limit max_stop_seqs_num({max_stop_seqs_num})."
                    "Please reduce the number of stop or set a lager max_stop_seqs_num by `FD_MAX_STOP_SEQS_NUM`"
                )
                api_server_logger.error(error_msg)
                raise EngineError(error_msg, error_code=400)
            stop_seqs_max_len = int(envs.FD_STOP_SEQS_MAX_LEN)
            for single_stop_seq_len in stop_seqs_len:
                if single_stop_seq_len > stop_seqs_max_len:
                    error_msg = (
                        f"Length of stop_seqs({single_stop_seq_len}) exceeds the limit stop_seqs_max_len({stop_seqs_max_len})."
                        "Please reduce the length of stop sequences or set a larger stop_seqs_max_len by `FD_STOP_SEQS_MAX_LEN`"
                    )
                    api_server_logger.error(error_msg)
                    raise EngineError(error_msg, error_code=400)

        task["preprocess_end_time"] = time.time()
        preprocess_cost_time = task["preprocess_end_time"] - task["preprocess_start_time"]
        api_server_logger.info(
            f"Cache request with request_id ({task.get('request_id')}), "
            f"preprocess time cost {preprocess_cost_time}"
        )

        self.valid_parameters(task)
        api_server_logger.debug(f"Receive task: {task}")
        try:
            if not self.enable_mm:
                self.zmq_client.send_json(task)
            else:
                self.zmq_client.send_pyobj(task)
        except Exception as e:
            api_server_logger.error(f"zmq_client send task error: {e}, {str(traceback.format_exc())}")
            raise EngineError(str(e), error_code=400)

    def valid_parameters(self, data):
        """
        Validate stream options
        超参数（top_p、seed、frequency_penalty、temperature、presence_penalty）的校验逻辑
        前置到了ChatCompletionRequest/CompletionRequest中
        """

        if data.get("n") is not None:
            if data["n"] != 1:
                raise ParameterError("n", "n only support 1.")

        if data.get("max_tokens") is not None:
            if data["max_tokens"] < 1 or data["max_tokens"] >= self.max_model_len:
                raise ParameterError("max_tokens", f"max_tokens can be defined [1, {self.max_model_len}).")

        if data.get("reasoning_max_tokens") is not None:
            if data["reasoning_max_tokens"] < 1:
                raise ParameterError("reasoning_max_tokens", "reasoning_max_tokens must be greater than 1")
            if data["reasoning_max_tokens"] > data["max_tokens"]:
                data["reasoning_max_tokens"] = data["max_tokens"]
                api_server_logger.warning(
                    f"req_id: {data['request_id']}, reasoning_max_tokens exceeds max_tokens, the value of reasoning_max_tokens will be adjusted to match that of max_tokens"
                )

        # logprobs
        logprobs = data.get("logprobs")
        top_logprobs = None

        if isinstance(logprobs, bool) and logprobs:
            if not self.enable_logprob:
                err_msg = "Logprobs is disabled, please enable it in startup config."
                api_server_logger.error(err_msg)
                raise ParameterError("logprobs", err_msg)
            top_logprobs = data.get("top_logprobs")
        elif isinstance(logprobs, int):
            top_logprobs = logprobs
        elif logprobs:
            raise ParameterError("logprobs", "Invalid type for 'logprobs'")

        # enable_logprob
        if top_logprobs:
            if not self.enable_logprob:
                err_msg = "Logprobs is disabled, please enable it in startup config."
                api_server_logger.error(err_msg)
                raise ParameterError("logprobs", err_msg)

            if not isinstance(top_logprobs, int):
                err_type = type(top_logprobs).__name__
                err_msg = f"Invalid type for 'top_logprobs': expected int but got {err_type}."
                api_server_logger.error(err_msg)
                raise ParameterError("top_logprobs", err_msg)

            if top_logprobs < 0:
                err_msg = f"Invalid 'top_logprobs': must be >= 0, got {top_logprobs}."
                api_server_logger.error(err_msg)
                raise ParameterError("top_logprobs", err_msg)

            if top_logprobs > 20:
                err_msg = "Invalid value for 'top_logprobs': must be <= 20."
                api_server_logger.error(err_msg)
                raise ParameterError("top_logprobs", err_msg)

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

    def is_workers_alive(self):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.model_weights_status_signal.value[0] == ModelWeightsStatus.NORMAL:
            return True, ""
        else:
            return False, "No model weight enabled"

    def update_model_weight(self, timeout=300):
        """
        Update the model weight by sending a signal to the server.
        1 : worker receive the signal and start to update model weight
        2 : worker update finish and notify client
        """
        with self.clear_update_lock:
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.NORMAL:
                return True, ""
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.UPDATING:
                return False, "worker is updating model weight already"
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.CLEARING:
                return False, "worker is clearing model weight, cannot update now"

            self.model_weights_status_signal.value[0] = ModelWeightsStatus.UPDATING
            if self.enable_prefix_caching or self.enable_splitwise:
                self.kv_cache_status_signal.value[0] = KVCacheStatus.UPDATING
            if self.enable_prefix_caching:
                self.prefix_tree_status_signal.value[0] = PrefixTreeStatus.UPDATING
            api_server_logger.info(f"start update model weight {self.model_weights_status_signal.value}")
            all_updated = False
            while timeout >= 0 and not all_updated:
                api_server_logger.info(
                    f"Updating model weights.. "
                    f"model_weights_status: {self.model_weights_status_signal.value[0]}, "
                    f"prefix_tree_status: {self.prefix_tree_status_signal.value[0]}, "
                    f"kv_cache_status: {self.kv_cache_status_signal.value[0]} "
                )
                weight_updated = self.model_weights_status_signal.value[0] == ModelWeightsStatus.NORMAL
                cache_updated = self.kv_cache_status_signal.value[0] == KVCacheStatus.NORMAL
                prefix_updated = self.prefix_tree_status_signal.value[0] == PrefixTreeStatus.NORMAL
                if self.enable_prefix_caching or self.enable_splitwise:
                    if self.enable_prefix_caching:
                        all_updated = weight_updated and cache_updated and prefix_updated
                    else:
                        all_updated = weight_updated and cache_updated
                else:
                    all_updated = weight_updated
                time.sleep(1)
                timeout -= 1
            if timeout < 0:
                return False, "Update model weight timeout"
            time.sleep(1)
            return True, ""

    def clear_load_weight(self, timeout=300):
        """
        Clear the load weight status.
        -1 : worker receive the signal and start to clear model weight
        -2 : worker clear finish and notify client
        """

        with self.clear_update_lock:
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.CLEARED:
                return True, ""
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.CLEARING:
                return False, "worker is clearing model weight already"
            if self.model_weights_status_signal.value[0] == ModelWeightsStatus.UPDATING:
                return False, "worker is updating model weight, cannot clear now"

            self.model_weights_status_signal.value[0] = ModelWeightsStatus.CLEARING
            if self.enable_prefix_caching or self.enable_splitwise:
                self.kv_cache_status_signal.value[0] = KVCacheStatus.CLEARING
            if self.enable_prefix_caching:
                self.prefix_tree_status_signal.value[0] = PrefixTreeStatus.CLEARING

            api_server_logger.info(f"start clear model weight {self.model_weights_status_signal.value}")
            all_cleared = False
            while timeout >= 0 and not all_cleared:
                api_server_logger.info(
                    f"Clearing model weights.. "
                    f"model_weights_status: {self.model_weights_status_signal.value[0]}, "
                    f"prefix_tree_status: {self.prefix_tree_status_signal.value[0]}, "
                    f"kv_cache_status: {self.kv_cache_status_signal.value[0]} "
                )
                weight_cleared = self.model_weights_status_signal.value[0] == ModelWeightsStatus.CLEARED
                cache_cleared = self.kv_cache_status_signal.value[0] == KVCacheStatus.CLEARED
                prefix_cleared = self.prefix_tree_status_signal.value[0] == PrefixTreeStatus.CLEARED
                if self.enable_prefix_caching or self.enable_splitwise:
                    if self.enable_prefix_caching:
                        all_cleared = weight_cleared and cache_cleared and prefix_cleared
                    else:
                        all_cleared = weight_cleared and cache_cleared
                else:
                    all_cleared = weight_cleared
                time.sleep(1)
                timeout -= 1

            if timeout < 0:
                return False, "Clear model weight timeout"
            time.sleep(1)
            return True, ""

    def check_model_weight_status(self):
        return self.model_weights_status_signal.value[0] < 0
