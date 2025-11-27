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
from copy import copy
from http import HTTPStatus

import numpy as np
from filelock import FileLock

from fastdeploy import envs
from fastdeploy.entrypoints.openai.utils import DealerConnectionManager
from fastdeploy.envs import FD_SUPPORT_MAX_CONNECTIONS
from fastdeploy.eplb.utils import RedundantExpertWorkload
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import (
    IPCSignal,
    KVCacheStatus,
    ModelWeightsStatus,
    PrefixTreeStatus,
    RearrangeExpertStatus,
    ZmqIpcClient,
)
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.platforms import current_platform
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
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
        config,
        reasoning_parser=None,
        data_parallel_size=1,
        enable_logprob=False,
        workers=1,
        tool_parser=None,
        enable_prefix_caching=None,
        splitwise_role=None,
        max_processor_cache=0,
    ):
        self.config = config
        self.model_config = config.model_config
        self.enable_mm = self.model_config.enable_mm
        enable_processor_cache = self.enable_mm and max_processor_cache > 0
        input_processor = InputPreprocessor(
            self.model_config,
            reasoning_parser,
            limit_mm_per_prompt,
            mm_processor_kwargs,
            tool_parser,
            enable_processor_cache,
        )
        self.enable_logprob = enable_logprob
        self.reasoning_parser = reasoning_parser
        self.data_processor = input_processor.create_processor()
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_splitwise = splitwise_role != "mixed"
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8

        if self.enable_mm and self.enable_prefix_caching:
            from fastdeploy.cache_manager.cache_data import (
                is_mm_model_disable_prefix_cache,
            )

            self.disable_prefix_mm = is_mm_model_disable_prefix_cache(self.model_config)

        if tensor_parallel_size <= max_chips_per_node:
            self.is_master = True
        else:
            self.is_master = False

        if self.config.eplb_config.enable_eplb:
            self.init_eplb_signals(ipc_signal_suffix=port)

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

    def init_eplb_signals(self, ipc_signal_suffix):
        """
        Initialize eplb signals.
        """
        if self.config.parallel_config.tensor_parallel_rank != 0:
            # only TP rank 0 need to init eplb signals, rank 0 manage all EPLB signals for all TP ranks
            return

        self.signal_clear_experts_token_stats_list = []
        self.local_experts_token_stats_array_list = []
        self.expert_tokens_stats_array_list = []
        self.signal_update_weight_from_disk_array_list = []
        self.update_weight_from_disk_result_list = []

        dp_ipc_signal_suffix = f"{ipc_signal_suffix}_dp{self.config.parallel_config.local_data_parallel_id}"
        rearrange_experts_status = np.zeros([1], dtype=np.int32)
        self.rearrange_experts_signal = IPCSignal(
            name="rearrange_experts_status",
            array=rearrange_experts_status,
            dtype=np.int32,
            suffix=dp_ipc_signal_suffix,
            create=False,
        )

        rearrange_experts_ips_size_array = np.zeros([1], dtype=np.int32)
        self.rearrange_experts_ips_size_signal = IPCSignal(
            name="rearrange_experts_ips_size",
            array=rearrange_experts_ips_size_array,
            dtype=np.int32,
            suffix=dp_ipc_signal_suffix,
            create=False,
        )

        self.shm_rearrange_experts_ips_list = IPCSignal(
            name="rearrange_experts_ips_list",
            shm_size=self.config.eplb_config.redundant_expert_ip_shm_size,
            suffix=dp_ipc_signal_suffix,
            create=False,
        )

        signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
        self.signal_update_weight_from_tensor_array = IPCSignal(
            name="signal_update_weight_from_tensor",
            array=signal_update_weight_from_tensor,
            dtype=np.int32,
            suffix=dp_ipc_signal_suffix,
            create=False,
        )

        for tp_rank_id in range(self.config.parallel_config.tensor_parallel_size):
            tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{tp_rank_id}"
            signal_clear_experts_token_stats = np.zeros([1], dtype=np.int32)
            self.signal_clear_experts_token_stats_list.append(
                IPCSignal(
                    name="signal_clear_experts_token_stats",
                    array=signal_clear_experts_token_stats,
                    dtype=np.int32,
                    suffix=tp_ipc_signal_suffix,
                    create=False,
                )
            )

            signal_update_weight_from_disk = np.zeros([1], dtype=np.int32)
            self.signal_update_weight_from_disk_array_list.append(
                IPCSignal(
                    name="signal_update_weight_from_disk",
                    array=signal_update_weight_from_disk,
                    dtype=np.int32,
                    suffix=tp_ipc_signal_suffix,
                    create=False,
                )
            )

            result_update_weight_from_disk = np.zeros([1], dtype=np.int32)
            self.update_weight_from_disk_result_list.append(
                IPCSignal(
                    name="result_update_weight_from_disk",
                    array=result_update_weight_from_disk,
                    dtype=np.int32,
                    suffix=tp_ipc_signal_suffix,
                    create=False,
                )
            )

            experts_token_stats = np.zeros(
                (self.config.model_config.num_hidden_layers, self.config.model_config.moe_num_experts),
                dtype=np.int32,
            )
            self.expert_tokens_stats_array_list.append(
                IPCSignal(
                    name="all_experts_token_stats",
                    array=experts_token_stats,
                    dtype=np.int32,
                    suffix=tp_ipc_signal_suffix,
                    create=False,
                )
            )
            self.local_experts_token_stats_array_list.append(
                IPCSignal(
                    name="local_experts_token_stats",
                    array=experts_token_stats,
                    dtype=np.int32,
                    suffix=tp_ipc_signal_suffix,
                    create=False,
                )
            )

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

    def _check_mm_disable_prefix_cache(self, task):
        is_multimodal_data = False
        if self.disable_prefix_mm:
            multimodal_inputs = task.get("multimodal_inputs", [])
            if multimodal_inputs:
                token_type_ids = multimodal_inputs.get("token_type_ids", [])
                if token_type_ids:
                    is_multimodal_data = np.sum(token_type_ids) > 0
        return is_multimodal_data

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
        trace_print(LoggingEventName.PREPROCESSING_START, task["request_id"], task.get("user", ""))
        try:
            chat_template_kwargs = task.get("chat_template_kwargs") or {}
            chat_template_kwargs.update({"chat_template": task.get("chat_template")})
            task["chat_template_kwargs"] = chat_template_kwargs
            if inspect.iscoroutinefunction(self.data_processor.process_request_dict):
                await self.data_processor.process_request_dict(task, self.max_model_len)
            else:
                self.data_processor.process_request_dict(task, self.max_model_len)

            if self.enable_mm and self.enable_prefix_caching:
                if self._check_mm_disable_prefix_cache(task):
                    api_server_logger.error(
                        "The current service does not support processing requests containing multimodal data when prefix cache is enabled. Please send only text-based requests or disable prefix cache"
                    )
                    raise EngineError(
                        "The current service does not support processing requests containing multimodal data when prefix cache is enabled. Please send only text-based requests or disable prefix cache",
                        error_code=400,
                    )

            task["prompt_token_ids_len"] = len(task["prompt_token_ids"])
            input_ids_len = task["prompt_token_ids_len"]

            task["max_tokens"] = min(self.max_model_len - input_ids_len, task.get("max_tokens"))
            min_tokens = task.get("min_tokens", 1)
            if "messages" in task:
                del task["messages"]
            api_server_logger.info(f"task['max_tokens']:{task['max_tokens']}")
            main_process_metrics.request_params_max_tokens.observe(task["max_tokens"])
            main_process_metrics.prompt_tokens_total.inc(input_ids_len)
            main_process_metrics.request_prompt_tokens.observe(input_ids_len)
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
            max_stop_seqs_num = envs.FD_MAX_STOP_SEQS_NUM
            if len(stop_seqs_len) > max_stop_seqs_num:
                error_msg = (
                    f"Length of stop ({stop_seqs_len}) exceeds the limit max_stop_seqs_num({max_stop_seqs_num})."
                    "Please reduce the number of stop or set a lager max_stop_seqs_num by `FD_MAX_STOP_SEQS_NUM`"
                )
                api_server_logger.error(error_msg)
                raise EngineError(error_msg, error_code=400)
            stop_seqs_max_len = envs.FD_STOP_SEQS_MAX_LEN
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
        n = task.get("n", 1)
        try:
            request_id_idx = task.get("request_id")
            parts = request_id_idx.rsplit("_", 1)
            if len(parts) == 1:
                self._send_task(task)
            else:
                request_id = parts[0]
                index = int(parts[1])
                for i in range(index * n, (index + 1) * n):
                    child_task = copy(task)
                    child_task["request_id"] = f"{request_id}_{i}"
                    self._send_task(child_task)
        except Exception as e:
            api_server_logger.error(f"zmq_client send task error: {e}, {str(traceback.format_exc())}")
            raise EngineError(str(e), error_code=400)

    def _send_task(self, task):
        if not self.enable_mm:
            self.zmq_client.send_json(task)
        else:
            self.zmq_client.send_pyobj(task)

    def valid_parameters(self, data):
        """
        Validate stream options
        超参数（top_p、seed、frequency_penalty、temperature、presence_penalty）的校验逻辑
        前置到了ChatCompletionRequest/CompletionRequest中
        """

        if data.get("max_tokens") is not None:
            if data["max_tokens"] < 1 or data["max_tokens"] >= self.max_model_len:
                api_server_logger.error(
                    f"req_id:{data['request_id']}, max_tokens must be defined [1, {self.max_model_len}), but now it's {data['max_tokens']}."
                )
                raise ValueError(
                    f"max_tokens can be defined [1, {self.max_model_len}), but now it's {data['max_tokens']}."
                )

        if data.get("reasoning_max_tokens") is not None:
            if data["reasoning_max_tokens"] < 1:
                raise ParameterError("reasoning_max_tokens", "reasoning_max_tokens must be greater than 1")
            if data["reasoning_max_tokens"] > data["max_tokens"]:
                data["reasoning_max_tokens"] = data["max_tokens"]
                api_server_logger.warning(
                    f"req_id: {data['request_id']}, reasoning_max_tokens exceeds max_tokens, the value of reasoning_max_tokens will be adjusted to {data['max_tokens']}"
                )
        if data.get("temperature") is not None and abs(data["temperature"]) < 1e-6:
            data["temperature"] = 1e-6
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

    async def rearrange_experts(self, request_dict: dict):
        """
        rearrange experts
        Args:
            request_dict (dict): request body
        Returns:
            tuple: response body, status code
        """
        eplb_config = self.config.eplb_config
        if not eplb_config.enable_eplb:
            content = {"code": 1, "msg": "redundant expert is disabled"}
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        if (
            request_dict.get("user", "") != eplb_config.redundant_expert_api_user
            or request_dict.get("passwd", "") != eplb_config.redundant_expert_api_password
        ):
            content = {"code": 1, "msg": "user or passwd is invalid"}
            status_code = HTTPStatus.UNAUTHORIZED
            return content, status_code

        if self.config.parallel_config.tensor_parallel_rank != 0:
            content = {
                "code": 1,
                "msg": f"actual rank {self.config.parallel_config.tensor_parallel_rank}, expect rank 0",
            }
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        action = request_dict.get("action", "")
        api_server_logger.info(f"redundant_expert: rearrange_experts recv request, action {action}")
        if action == "":
            # action: start rearrange experts
            # params: {'user': 'xxx', 'passwd': 'xxx', 'ips': ['10.54.99.77:8000', '10.54.99.77:8300']}
            if self.rearrange_experts_signal.value[0] != RearrangeExpertStatus.FREE.value:
                content = {
                    "code": 1,
                    "msg": f"rearrange is doing. actual status {self.rearrange_experts_signal.value[0]}, expect status {RearrangeExpertStatus.FREE.value}",
                }
                status_code = HTTPStatus.BAD_REQUEST
            if "ips" not in request_dict and content is None:
                content = {"code": 1, "msg": "ips in request is None"}
                status_code = HTTPStatus.BAD_REQUEST

            if content is not None:
                return content, status_code

            data_bytes = (";".join(request_dict["ips"])).encode("utf-8")
            data_size = len(data_bytes)
            if data_size > eplb_config.redundant_expert_ip_shm_size:
                content = {
                    "code": 1,
                    "msg": f"actual ips size {data_size}, max limit {eplb_config.redundant_expert_ip_shm_size}",
                }
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            else:
                self.rearrange_experts_ips_size_signal.value[0] = data_size
                self.shm_rearrange_experts_ips_list.shm.buf[:data_size] = data_bytes
                content = {"code": 0, "msg": "ok"}
                status_code = HTTPStatus.OK
            return content, status_code
        elif action == "recv_expert_weight":
            # action: receive global expert workload, and begin update weight from disk
            # params: {'user': 'xxx', 'passwd': 'xxx', 'weight': (layers, experts)}
            if "data" not in request_dict or not isinstance(request_dict["data"], list):
                content = {"code": 1, "msg": "data not in request or data is not a list"}
                status_code = HTTPStatus.BAD_REQUEST
            else:
                weight = np.array(request_dict["data"], dtype=np.int32)
                for idx in range(len(self.expert_tokens_stats_array_list)):
                    self.expert_tokens_stats_array_list[idx].value[:] = weight[:]
                    self.signal_update_weight_from_disk_array_list[idx].value[0] = 1

                content = {"code": 0, "msg": "ok"}
                status_code = HTTPStatus.OK
            return content, status_code
        elif action == "update_weight_from_tensor":
            if self.config.scheduler_config.splitwise_role != "prefill" and content is None:
                content = {
                    "code": 1,
                    "msg": f"actual role {self.config.scheduler_config.splitwise_role}, expect role prefill",
                }
                status_code = HTTPStatus.BAD_REQUEST
            if self.rearrange_experts_signal.value[0] != RearrangeExpertStatus.LOAD_SUCC.value and content is None:
                content = {
                    "code": 1,
                    "msg": f"actual status {self.rearrange_experts_signal.value[0]}, expect status {RearrangeExpertStatus.LOAD_SUCC.value}",
                }
                status_code = HTTPStatus.BAD_REQUEST

            if content is None:
                self.signal_update_weight_from_tensor_array.value[0] = 1
                content = {"code": 0, "msg": "ok"}
                status_code = HTTPStatus.OK
            return content, status_code
        else:
            content = {"code": 1, "msg": f"invalid action {action}"}
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

    async def get_per_expert_tokens_stats(self, request_dict: dict):
        """
        get per expert tokens stats

        Args:
            request_dict (dict): request body
        Returns:
            tuple: response body, status code
        """
        eplb_config = self.config.eplb_config
        if not eplb_config.enable_eplb:
            content = {"code": 1, "msg": "redundant expert is disabled"}
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        if (
            request_dict.get("user", "") != eplb_config.redundant_expert_api_user
            or request_dict.get("passwd", "") != eplb_config.redundant_expert_api_password
        ):
            content = {"code": 1, "msg": "user or passwd is invalid"}
            status_code = HTTPStatus.UNAUTHORIZED
            return content, status_code

        if self.config.parallel_config.tensor_parallel_rank != 0:
            content = {
                "code": 1,
                "msg": f"actual rank {self.config.parallel_config.tensor_parallel_rank}, expect rank 0",
            }
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        if "clear_stat" in request_dict and request_dict["clear_stat"]:
            for clear_experts_token_stats in self.signal_clear_experts_token_stats_list:
                clear_experts_token_stats.value[0] = 1

        local_experts_list = []
        for local_experts_token_stats in self.local_experts_token_stats_array_list:
            local_experts_list.append(local_experts_token_stats.value.tolist())
        content = {"code": 0, "msg": "ok", "data": local_experts_list}
        status_code = HTTPStatus.OK
        return content, status_code

    async def check_redundant(self, request_dict: dict):
        """
        check redundant
        Args:
            request_dict (dict): request body
        Returns:
            tuple: response body, status code
        """
        content, status_code = None, HTTPStatus.OK
        eplb_config = self.config.eplb_config

        if not eplb_config.enable_eplb:
            content = {"code": 1, "msg": "redundant expert is disabled"}
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        if (
            request_dict.get("user", "") != eplb_config.redundant_expert_api_user
            or request_dict.get("passwd", "") != eplb_config.redundant_expert_api_password
        ):
            content = {"code": 1, "msg": "user or passwd is invalid"}
            status_code = HTTPStatus.UNAUTHORIZED
            return content, status_code

        if self.config.parallel_config.tensor_parallel_rank != 0:
            content = {
                "code": 1,
                "msg": f"actual rank {self.config.parallel_config.tensor_parallel_rank}, expect rank 0",
            }
            status_code = HTTPStatus.BAD_REQUEST
            return content, status_code

        action = request_dict.get("action", "")
        if action == "":
            status = "unknown"
            try:
                status = RearrangeExpertStatus(self.rearrange_experts_signal.value[0]).name
            except Exception:
                # Ignore errors if status cannot be determined; default to "unknown"
                pass
            content = {"code": 0, "msg": "ok", "status": status}
            get_workloads = False if "check_get_workloads" not in request_dict else request_dict["check_get_workloads"]
            if get_workloads:
                content["data"], content["msg"] = RedundantExpertWorkload(eplb_config.redundant_expert_meta_dir).load()
            status_code = HTTPStatus.OK
        elif action == "check_load_weight_result":
            update_weight_from_disk_list = []
            for update_weight_result in self.update_weight_from_disk_result_list:
                update_weight_from_disk_list.append(update_weight_result.value[0].tolist())
            content = {"code": 0, "msg": "ok", "data": update_weight_from_disk_list}
            status_code = HTTPStatus.OK
        return content, status_code
