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
from collections import defaultdict
from fastdeploy_llm.serving.serving_model import ServingModel
from fastdeploy_llm.utils.logging_util import logger, warning_logger
from fastdeploy_llm.utils.logging_util import error_format, ErrorCode,  ErrorType
from fastdeploy_llm.task import Task, BatchTask
import fastdeploy_llm as fdlm
import queue

# Only working in triton python backend
try:
    import triton_python_backend_utils as pb_utils
except:
    pass


response_dict = {}
response_finished_queue = queue.Queue()


def stream_call_back(call_back_task, token_tuple, index, is_last_token,
                     sender):
    if is_last_token:
        all_token_ids = [t[0] for t in call_back_task.result.completion_tokens] + [token_tuple[0]]
        all_strs = "".join([t[1] for t in call_back_task.result.completion_tokens]) + token_tuple[1]
        out = dict()
        out["result"] = all_strs
        out["req_id"] = call_back_task.task_id
        out["token_ids"] = all_token_ids
        out['send_idx'] = 0  # 整句返回
        out["is_end"] = True
        out_tensor = pb_utils.Tensor(
            "OUT", np.array(
                [json.dumps(out)], dtype=np.object_))
        response = pb_utils.InferenceResponse([out_tensor])
        response_dict[call_back_task.task_id] = response
        response_finished_queue.put(call_back_task.task_id)
        
        logger.info("Model output for req_id: {}  results_all: {} tokens_all: {} inference_cost_time: {} ms".format(
            call_back_task.task_id, all_strs, all_token_ids, (time.time() - call_back_task.inference_start_time) * 1000)) 


def parse(parameters_config, name, default_value=None):
    if name not in parameters_config:
        if default_value:
            return default_value
        else:
            raise Exception(
                "Cannot find key:{} while parsing parameters.".format(name))
    return parameters_config[name]["string_value"]


class TritonPythonModelNonStream:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
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

        self.wait_time_out = 60
        self.config = config
        self.response_handler = dict()
        self.model = ServingModel(config)
        self.model.model.stream_sender = self.response_handler
        self.model.start()

    def execute(self, requests):
        responses = []
        inflight_valid_tasks = {}
        request_start_time_dict = {}
        for i, request in enumerate(requests):
            request_tensor = pb_utils.get_input_tensor_by_name(request, "IN")

            # 1. validate the request data
            try:
                data = json.loads(request_tensor.as_numpy()[0])
                if isinstance(data, list):
                    data = data[0]
            except Exception as e:
                error_type = ErrorType.Query
                error_code = ErrorCode.C0000
                error_info = "Cannot load json data from request, received data = {} error={}.".format(request_tensor.as_numpy(), e)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue

            # 2. validate the deserializing process
            task = Task()
            try:
                task.from_dict(data)
                request_start_time = time.time()
                task.set_request_start_time(request_start_time)
            except Exception as e:
                error_type = ErrorType.Query
                error_code = ErrorCode.C0001
                error_info = "There's error while deserializing data from request, received data = {} error={}".format(data, e)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue

            # 3. check if exists task id conflict
            if task.task_id is None:
                task.task_id = str(uuid.uuid4())
            request_start_time_dict[task.task_id] = request_start_time
            if task.task_id in self.response_handler:
                error_type = ErrorType.Query
                error_code = ErrorCode.C0001
                error_info = "Task id conflict with {}.".format(task.task_id)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue

            # 4. validate the parameters in task
            try:
                task.check(self.config.max_dec_len)
            except Exception as e:
                error_type = ErrorType.Query
                error_code = ErrorCode.C0001
                error_info = "There's error while checking task, task={} error={}".format(task, e)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue

            # 5. check if the requests queue is full
            if self.model.requests_queue.qsize() > self.config.max_queue_num:
                error_type = ErrorType.Server
                error_code = ErrorCode.S0000
                error_info = "The queue is full now(size={}), please wait for a while.".format(self.config.max_queue_num)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
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
                    responses.append(error_res)
                    continue

            # 7. Add task to requests queue
            task.call_back_func = stream_call_back
            try:
                self.model.add_request(task)
            except queue.Full as e:
                # Log error for Server
                error_type = ErrorType.Server
                error_code = ErrorCode.S0000
                error_info = "The queue is full now(size={}), please scale service.".format(self.config.max_queue_num)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                # Log error for query
                error_type = ErrorType.Query
                error_code = ErrorCode.C0001
                error_info = "There's error while inserting new request, task={} error={}".format(task, "service too busy")
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue

            except Exception as e:
                error_type = ErrorType.Query
                error_code = ErrorCode.C0001
                error_info = "There's error while inserting new request, task={} error={}".format(task, e)
                error_msg = error_format.format(error_type.name, error_code.name, error_info)
                warning_logger.error(error_msg)
                error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                responses.append(error_res)
                continue
            inflight_valid_tasks[task.task_id] = i
            responses.append(None) # we use None as placeholder, fill it later
            self.response_handler[task.task_id] = None  # for compatibility
        
        while True:
            if len(inflight_valid_tasks) == 0:
                break
            try:
                task_id = response_finished_queue.get(timeout=self.wait_time_out)
                index = inflight_valid_tasks[task_id]
                responses[index] = response_dict[task_id]
                del inflight_valid_tasks[task_id]
                del response_dict[task_id]
                del self.response_handler[task_id]
            except:
                for task_id, index in inflight_valid_tasks.items():
                    error_type = ErrorType.Query
                    error_code = ErrorCode.C0001
                    error_info = "Timeout for getting inference result."
                    error_msg = error_format.format(error_type.name, error_code.name, error_info)
                    warning_logger.error(error_msg)
                    error_res = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                    responses[index] = error_res
                    break
        for task_id, start_time in request_start_time_dict.items():
            logger.info("req_id: {} has sent back to client, request_cost_time: {} ms".format(task_id, (time.time() - start_time) * 1000))
        return responses

    def finalize(self):
        logger.info("The triton server is going to terminating...")
        info_type = ErrorType.Server
        info_code = ErrorCode.S0002
        info_msg = error_format.format(info_type.name, info_code.name, "The triton server is going to terminating...")
        warning_logger.info(info_msg)
        self.model.stop()
        os.system("""
                    bash -c 'pids=$(ps auxww | grep -E "triton_python_backend_stub|multiprocessing.resource_tracker|engine.py" | grep -v grep | awk '"'"'{print $2}'"'"'); 
                    echo $pids; 
                    for pid in ${pids[@]}; do 
                    kill -9 ${pid} 
                    done;'
                    """)
        logger.info("The triton server is terminated, byebye.")

