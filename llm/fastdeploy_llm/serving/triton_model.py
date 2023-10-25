# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import json
import os
import uuid
import threading
import time
import numpy as np
import functools
from fastdeploy_llm.serving.serving_model import ServingModel
from fastdeploy_llm.utils.logging_util import logger
from fastdeploy_llm.task import Task, BatchTask
import fastdeploy_llm as fdlm

# Only working in triton python backend
try:
    import triton_python_backend_utils as pb_utils
except:
    pass


def stream_call_back(call_back_task, token_tuple, index, is_last_token,
                     sender):
    out = dict()
    out["result"] = token_tuple[1]
    out["req_id"] = call_back_task.task_id
    out["token_ids"] = [token_tuple[0]]
    out['send_idx'] = index
    out["is_end"] = is_last_token
    out_tensor = pb_utils.Tensor(
        "OUT", np.array(
            [json.dumps(out)], dtype=np.object_))
    if is_last_token:
        sender[call_back_task.task_id].send(
            pb_utils.InferenceResponse([out_tensor]),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        del sender[call_back_task.task_id]
    else:
        sender[call_back_task.task_id].send(
            pb_utils.InferenceResponse([out_tensor]))


def parse(parameters_config, name, default_value=None):
    if name not in parameters_config:
        if default_value:
            return default_value
        else:
            raise Exception(
                "Cannot find key:{} while parsing parameters.".format(name))
    return parameters_config[name]["string_value"]


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(args["model_name"]))

        parameters = self.model_config["parameters"]

        config = fdlm.Config(
            os.path.join(args['model_repository'], args['model_version']))
        config.max_batch_size = int(parse(parameters, "MAX_BATCH_SIZE", 4))
        config.mp_num = int(parse(parameters, "MP_NUM", 1))
        if config.mp_num < 0:
            config.mp_num = None
        config.max_dec_len = int(parse(parameters, "MAX_DEC_LEN", 1024))
        config.max_seq_len = int(parse(parameters, "MAX_SEQ_LEN", 1024))
        config.decode_strategy = parse(parameters, "DECODE_STRATEGY",
                                       "sampling")
        config.stop_threshold = int(parse(parameters, "STOP_THRESHOLD", 2))
        config.disable_dynamic_batching = int(
            parse(parameters, "DISABLE_DYNAMIC_BATCHING", 0))
        config.max_queue_num = int(parse(parameters, "MAX_QUEUE_NUM", 512))
        config.is_ptuning = int(parse(parameters, "IS_PTUNING", 0))
        if config.is_ptuning:
            config.model_prompt_dir_path = parse(parameters,
                                                 "MODEL_PROMPT_DIR_PATH")
            config.max_prefix_len = int(parse(parameters, "MAX_PREFIX_LEN"))
        config.load_environment_variables()

        self.config = config
        self.response_handler = dict()
        self.model = ServingModel(config)
        self.model.model.stream_sender = self.response_handler
        self.model.start()

    def execute(self, requests):
        for request in requests:
            request_tensor = pb_utils.get_input_tensor_by_name(request, "IN")

            # 1. validate the request data
            try:
                data = json.loads(request_tensor.as_numpy()[0])
                if isinstance(data, list):
                    data = data[0]
            except Exception as e:
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "Cannot load json data from request, error={}.".format(
                            e)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue

            # 2. validate the deserializing process
            task = Task()
            try:
                task.from_dict(data)
            except Exception as e:
                error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(
                    "There's error while deserializing data from request, error={}".
                    format(e)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue

            # 3. check if exists task id conflict
            if task.task_id is None:
                task.task_id = str(uuid.uuid4())
            if task.task_id in self.response_handler:
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "Task id conflict with {}.".format(task.task_id)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue

            # 4. validate the parameters in task
            try:
                task.check(self.config.max_dec_len)
            except Exception as e:
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "There's error while checking task, error={}".format(
                            e)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue

            # 5. check if the requests queue is full
            if self.model.requests_queue.qsize() > self.config.max_queue_num:
                error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(
                    "The queue is full now(size={}), please wait for a while.".
                    format(self.model.max_queue_num)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue

            # 6. check if the prefix embedding is exist
            if self.config.is_ptuning and task.model_id is not None:
                np_file_path = os.path.join(self.config.model_prompt_dir_path,
                                            "8-{}".format(task.model_id), "1",
                                            "task_prompt_embeddings.npy")
                if not os.path.exists(np_file_path):
                    error_res = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "There's no prefix embedding for model_id={}.".
                            format(task.model_id)))
                    res_sender = request.get_response_sender()
                    res_sender.send(
                        error_res,
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    continue

            # 7. Add task to requests queue
            task.call_back_func = stream_call_back
            try:
                self.model.add_request(task)
            except Exception as e:
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "There's error while inserting new request, error={}".
                        format(e)))
                res_sender = request.get_response_sender()
                res_sender.send(
                    error_res,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                continue
            self.response_handler[task.task_id] = request.get_response_sender()

    def finalize(self):
        logger.info("The triton server is going to terminating...")
        self.model.stop()
        logger.info("The triton server is terminated, byebye.")
