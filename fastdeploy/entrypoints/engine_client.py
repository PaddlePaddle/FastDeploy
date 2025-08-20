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

import time
import traceback
import uuid

import numpy as np

from fastdeploy import envs
from fastdeploy.engine.config import ModelConfig
from fastdeploy.envs import FD_SUPPORT_MAX_CONNECTIONS
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import IPCSignal, ZmqClient
from fastdeploy.metrics.work_metrics import work_process_metrics
from fastdeploy.multimodal.registry import MultimodalRegistry
from fastdeploy.platforms import current_platform
from fastdeploy.utils import EngineError, StatefulSemaphore, api_server_logger


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
        limit_mm_per_prompt,
        mm_processor_kwargs,
        # enable_mm=False,
        reasoning_parser=None,
        data_parallel_size=1,
        enable_logprob=False,
        workers=1,
        tool_parser=None,
    ):
        import fastdeploy.model_executor.models  # noqa: F401

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
        max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        array_size = min(max_chips_per_node, tensor_parallel_size * data_parallel_size)
        self.worker_healthy_live_recorded_time_array = np.zeros(shape=[array_size], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=self.worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=pid,
            create=False,
        )
        self.semaphore = StatefulSemaphore((FD_SUPPORT_MAX_CONNECTIONS + workers - 1) // workers)
        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=pid,
            create=False,
        )

    def create_zmq_client(self, model, mode):
        """
        Create a ZMQ client.
        """
        self.zmq_client = ZmqClient(model, mode)
        self.zmq_client.connect()

    def format_and_add_data(self, prompts: dict):
        """
        Format the request data and send the request to the server.
        """
        if "request_id" not in prompts:
            request_id = str(uuid.uuid4())
            prompts["request_id"] = request_id

        if "max_tokens" not in prompts:
            prompts["max_tokens"] = self.max_model_len - 1

        self.add_requests(prompts)
        return prompts["prompt_token_ids"]

    def add_requests(self, task):
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
            self.data_processor.process_request_dict(task, self.max_model_len)

            task["prompt_token_ids_len"] = len(task["prompt_token_ids"])
            input_ids_len = task["prompt_token_ids_len"]
            task["max_tokens"] = min(self.max_model_len - input_ids_len, task.get("max_tokens"))
            if task.get("reasoning_max_tokens", None) is None:
                task["reasoning_max_tokens"] = max(int(task["max_tokens"] * 0.8), 1)
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

        self.vaild_parameters(task)
        api_server_logger.debug(f"Recieve task: {task}")
        try:
            if not self.enable_mm:
                self.zmq_client.send_json(task)
            else:
                self.zmq_client.send_pyobj(task)
        except Exception as e:
            api_server_logger.error(f"zmq_client send task error: {e}, {str(traceback.format_exc())}")
            raise EngineError(str(e), error_code=400)

    def vaild_parameters(self, data):
        """
        Validate stream options
        """

        if data.get("n"):
            if data["n"] != 1:
                raise ValueError("n only support 1.")

        if data.get("max_tokens"):
            if data["max_tokens"] < 1 or data["max_tokens"] >= self.max_model_len:
                raise ValueError(f"max_tokens can be defined [1, {self.max_model_len}).")

        if data.get("reasoning_max_tokens"):
            if data["reasoning_max_tokens"] > data["max_tokens"] or data["reasoning_max_tokens"] < 1:
                raise ValueError("reasoning_max_tokens must be between max_tokens and 1")

        if data.get("top_p"):
            if data["top_p"] > 1 or data["top_p"] < 0:
                raise ValueError("top_p value can only be defined [0, 1].")

        if data.get("frequency_penalty"):
            if not -2.0 <= data["frequency_penalty"] <= 2.0:
                raise ValueError("frequency_penalty must be in [-2, 2]")

        if data.get("temperature"):
            if data["temperature"] < 0:
                raise ValueError("temperature must be non-negative")

        if data.get("presence_penalty"):
            if not -2.0 <= data["presence_penalty"] <= 2.0:
                raise ValueError("presence_penalty must be in [-2, 2]")

        if data.get("seed"):
            if not 0 <= data["seed"] <= 922337203685477580:
                raise ValueError("seed must be in [0, 922337203685477580]")

        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")

        # logprobs
        logprobs = data.get("logprobs")
        top_logprobs = None

        if isinstance(logprobs, bool) and logprobs:
            if not self.enable_logprob:
                err_msg = "Logprobs is disabled, please enable it in startup config."
                api_server_logger.error(err_msg)
                raise ValueError(err_msg)
            top_logprobs = data.get("top_logprobs")
        elif isinstance(logprobs, int):
            top_logprobs = logprobs
        elif logprobs:
            raise ValueError("Invalid type for 'logprobs'")

        # enable_logprob
        if top_logprobs:
            if not self.enable_logprob:
                err_msg = "Logprobs is disabled, please enable it in startup config."
                api_server_logger.error(err_msg)
                raise ValueError(err_msg)

            if not isinstance(top_logprobs, int):
                err_type = type(top_logprobs).__name__
                err_msg = f"Invalid type for 'top_logprobs': expected int but got {err_type}."
                api_server_logger.error(err_msg)
                raise ValueError(err_msg)

            if top_logprobs < 0:
                err_msg = f"Invalid 'top_logprobs': must be >= 0, got {top_logprobs}."
                api_server_logger.error(err_msg)
                raise ValueError(err_msg)

            if top_logprobs > 20:
                err_msg = "Invalid value for 'top_logprobs': must be <= 20."
                api_server_logger.error(err_msg)
                raise ValueError(err_msg)

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
        if self.model_weights_status_signal.value[0] == 0:
            return True, ""
        else:
            return False, "No model weight enabled"

    def update_model_weight(self, timeout=300):
        """
        Update the model weight by sending a signal to the server.
        1 : worker receive the signal and start to update model weight
        2 : worker update finish and notify client
        """
        if self.model_weights_status_signal.value[0] == 0:
            return True, ""
        if self.model_weights_status_signal.value[0] == 1:
            return False, "updating model weight already"

        self.model_weights_status_signal.value[0] = 1
        api_server_logger.info(f"start update model weight {self.model_weights_status_signal.value}")
        while self.model_weights_status_signal.value[0] != 0 and timeout != 0:
            time.sleep(1)
            timeout -= 1
            continue
        if self.model_weights_status_signal.value[0] != 0:
            return False, "Update model weight timeout"
        time.sleep(1)
        return True, ""

    def clear_load_weight(self, timeout=300):
        """
        Clear the load weight status.
        -1 : worker receive the signal and start to clear model weight
        -2 : worker clear finish and notify client
        """
        if self.model_weights_status_signal.value[0] == -2:
            return True, ""
        if self.model_weights_status_signal.value[0] == -1:
            return False, "clearing model weight already"

        self.model_weights_status_signal.value[0] = -1

        api_server_logger.info(f"start clear model weight {self.model_weights_status_signal.value}")
        while self.model_weights_status_signal.value[0] != -2 and timeout != 0:
            time.sleep(1)
            timeout -= 1
            continue
        if self.model_weights_status_signal.value[0] != -2:
            return False, "clear model weight timeout"
        time.sleep(1)
        return True, ""
