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

import zmq
import time
from random import randint
import uuid
import numpy as np

from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator import ZmqClient, IPCSignal
from fastdeploy.utils import api_server_logger, EngineError


class EngineClient:
    """
    EngineClient is a class that handles the communication between the client and the server.
    """
    def __init__(self, tokenizer, max_model_len, tensor_parallel_size, pid, enable_mm=False):
        input_processor =  InputPreprocessor(tokenizer, enable_mm)
        self.enable_mm = enable_mm
        self.data_processor = input_processor.create_processor()
        self.max_model_len = max_model_len
        self.worker_healthy_live_recorded_time_array = np.zeros(shape=[tensor_parallel_size], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(name="worker_healthy_live_signal",
                    array=self.worker_healthy_live_recorded_time_array,
                    dtype=np.int32,
                    suffix=pid,
                    create=False)

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=pid,
            create=False)

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
        if "request_id" in prompts:
            prompts["request_id"] = prompts["request_id"]

        if "request_id" not in prompts:
            request_id = str(uuid.uuid4())
            prompts["request_id"] = request_id
        query_list = []

        if "max_tokens" not in prompts:
            prompts["max_tokens"] = self.max_model_len - 1

        self.add_requests(prompts)

    def add_requests(self, task):
        """
        Add a new request to the queue.

        Args:
            task: Request A dictionary representing the request.
            sampling_params: A dictionary representing the sampling parameters.

        Returns:
            None
        """
        self.vaild_parameters(task)

        task["preprocess_start_time"] = time.time()
        try:
            self.data_processor.process_request_dict(task, self.max_model_len)

            task["prompt_token_ids_len"] = len(task["prompt_token_ids"])
            input_ids_len = task["prompt_token_ids_len"]
            task["max_tokens"] = min(self.max_model_len - input_ids_len , task.get("max_tokens"))
            min_tokens = task.get("min_tokens", 1)
        except Exception as e:
            api_server_logger.error(e)
            raise EngineError(str(e), error_code=400)

        if input_ids_len + min_tokens >= self.max_model_len:
            error_msg = (
                f"Input text is too long, input_ids_len ({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len "
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.max_model_len})."
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        task["preprocess_end_time"] = time.time()
        preprocess_cost_time = task["preprocess_end_time"] - task["preprocess_start_time"]
        api_server_logger.info(
            f"Cache request with request_id ({task.get('request_id')}), "
            f"cost {time.time() - preprocess_cost_time}"
        )
        api_server_logger.debug(f"Recieve task: {task}")
        try:
            if not self.enable_mm:
                self.zmq_client.send_json(task)
            else:
                self.zmq_client.send_pyobj(task)
        except Exception as e:
            api_server_logger.error(e)
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

        if data.get("top_p"):
            if data["top_p"] > 1 or data["top_p"] < 0:
                raise ValueError(
                    "top_p value can only be defined [0, 1].")


        if data.get("frequency_penalty"):
            if  not -2.0 <= data["frequency_penalty"] <= 2.0:
                raise ValueError("frequency_penalty must be in [-2, 2]")

        if data.get("temperature"):
            if data["temperature"] < 0:
                raise ValueError(f"temperature must be non-negative")


        if data.get("presence_penalty"):
            if  not -2.0 <= data["presence_penalty"] <= 2.0:
                raise ValueError("presence_penalty must be in [-2, 2]")



        if data.get("seed"):
            if not 0 <= data["seed"] <= 922337203685477580:
                raise ValueError("seed must be in [0, 922337203685477580]")

        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")



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



    def update_model_weight(self, timeout = 300):
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
        while self.model_weights_status_signal.value[0] != 0  and timeout != 0:
            time.sleep(1)
            timeout -= 1
            continue
        if self.model_weights_status_signal.value[0] != 0:
            return False, "Update model weight timeout"
        time.sleep(1)
        return True, ""



    def clear_load_weight(self, timeout = 300):
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
        while self.model_weights_status_signal.value[0] != -2  and timeout != 0:
            time.sleep(1)
            timeout -= 1
            continue
        if self.model_weights_status_signal.value[0] != -2:
            return False, "clear model weight timeout"
        time.sleep(1)
        return True, ""

