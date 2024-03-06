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
import queue
import asyncio

from fastapi import Request, HTTPException
from fastapi.responses import Response, JSONResponse
import google.protobuf.text_format as text_format
import google.protobuf.json_format as json_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

from fastdeploy_llm.serving.serving_model import ServingModel
from fastdeploy_llm.utils.logging_util import logger, warning_logger
from fastdeploy_llm.utils.logging_util import error_format, ErrorCode,  ErrorType
from fastdeploy_llm.task import Task, BatchTask
import fastdeploy_llm as fdlm

def pbtxt2json(content: str):
    '''
   Convert protocol messages in text format to json format string.
   '''
    message = text_format.Parse(content, ModelConfig())
    json_string = json_format.MessageToJson(message)
    return json_string
    

request_start_time_dict = {}
response_dict = {}
event_dict = {}
response_checked_dict = {}


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
        response_dict[call_back_task.task_id] = out
        logger.info("Model output for req_id: {}  results_all: {} tokens_all: {} inference_cost_time: {} ms".format(
            call_back_task.task_id, all_strs, all_token_ids, (time.time() - call_back_task.inference_start_time) * 1000)) 


def parse(parameters_config, name, default_value=None):
    if name not in parameters_config:
        if default_value:
            return default_value
        else:
            raise Exception(
                "Cannot find key:{} while parsing parameters.".format(name))
    return parameters_config[name]["stringValue"]


class ModelExecutor:
    def __init__(self, model_dir):
        config = fdlm.Config(model_dir)
        config_pb_path = os.path.join(model_dir, 'config.pbtxt')
        if os.path.exists(config_pb_path):
            with open(config_pb_path, 'r') as f:
                data = f.read()
                json_str = pbtxt2json(data)
                parameters = json.loads(json_str)['parameters']
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

        self.wait_time_out = config.inference_response_timeout
        self.config = config
        self.response_handler = dict()
        self.model = None
        
    
    def prepare_model(self):
        # This method can only called once within all process
        self.model = ServingModel(self.config)
        self.model.model.stream_sender = self.response_handler
        self.model.start()

    def execute(self, req_dict):
        # 1. validate the deserializing process
        task = Task()
        try:
            task.from_dict(req_dict)
            request_start_time = time.time()
            task.set_request_start_time(request_start_time)
        except Exception as e:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "There's error while deserializing data from request, received data = {} error={}".format(req_dict, e)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            response_dict[req_dict['req_id']] = error_msg
            return task

        # 3. check if exists task id conflict
        if task.task_id is None:
            task.task_id = str(uuid.uuid4())
        request_start_time_dict[task.task_id] = request_start_time
        if task.task_id in event_dict:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "Task id conflict with {}.".format(task.task_id)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            return None

        # 4. validate the parameters in task
        try:
            task.check(self.config.max_dec_len)
        except Exception as e:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "There's error while checking task, task={} error={}".format(task, e)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            response_dict[req_dict['req_id']] = error_msg
            return task

        # 5. check if the requests queue is full
        if self.model.requests_queue.qsize() > self.config.max_queue_num:
            error_type = ErrorType.Server
            error_code = ErrorCode.S0000
            error_info = "The queue is full now(size={}), please wait for a while.".format(self.config.max_queue_num)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            response_dict[req_dict['req_id']] = error_msg
            return task

        # 6. check if the prefix embedding is exist
        if self.config.is_ptuning and task.model_id is not None:
            np_file_path = os.path.join(self.config.model_prompt_dir_path,
                                        "8-{}".format(task.model_id), "1",
                                        "task_prompt_embeddings.npy")
            if not os.path.exists(np_file_path):
                response_dict[req_dict['req_id']] = error_msg
                return task

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
            response_dict[req_dict['req_id']] = error_msg
            return task

        except Exception as e:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "There's error while inserting new request, task={} error={}".format(task, e)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            response_dict[req_dict['req_id']] = error_msg
            return task
        
        return task

    async def inference(self, request_in: Request):
        """
        API for generation task.
        """
        start_time = time.time()
        try:
            input_dict = await request_in.json()
            logger.info("recieved req_dict {}".format(input_dict))
        except:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            content = await request_in.body()
            error_info = "request body is not a valid json format, received data = {}".format(content)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        if self.model is None:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "Model is not ready"
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        task = self.execute(input_dict)
        if task is None:  # task id conflict
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "Task id conflict"
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            raise HTTPException(status_code=400, detail=error_msg)

        event_dict[task.task_id] = asyncio.Event() 
        try:
            await asyncio.wait_for(event_dict[task.task_id].wait(), self.wait_time_out)  
        except:
            error_type = ErrorType.Query
            error_code = ErrorCode.C0001
            error_info = "Timeout for getting inference result, task={}".format(task)
            error_msg = error_format.format(error_type.name, error_code.name, error_info)
            warning_logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        result = response_checked_dict[task.task_id]
        del response_checked_dict[task.task_id]
        del event_dict[task.task_id]
        logger.info("req_id: {} has sent back to client, request_cost_time: {} ms".format(task.task_id, (time.time() - start_time) * 1000))
        return JSONResponse(result)

    def check_live(self):
        """
        API for detecting http app status.
        """
        if self.model.model._is_engine_initialized() and (self.model.model.engine_proc.poll() is None):
            # 引擎进程未退出
            # 判断是否推理在正常进行
            # 1. current_start_inference_time表示当前推理的开始时间 previous_start_inference_time表示上次推理的开始时间
            # 代码逻辑约定，当没有新请求或者当前请求推理结束的时候，current_start_inference_time和previous_start_inference_time是相同的
            # 当前请求正在推理的时候，current_start_inference_time和previous_start_inference_time是不同的
            with self.model.model.hang_detection_lock:
                previous_start_inference_time = self.model.model.previous_start_inference_time
                current_start_inference_time = self.model.model.previous_start_inference_time
            if previous_start_inference_time == current_start_inference_time:
                #（1） 没有新请求或者新请求推理完毕：等待10s，两者还相同，判定正常
                time.sleep(10)
                with self.model.model.hang_detection_lock:
                    new_previous_start_inference_time = self.model.model.previous_start_inference_time
                    new_current_start_inference_time = self.model.model.current_start_inference_time
                if new_previous_start_inference_time == new_current_start_inference_time:
                    logger.info("check_live: True") 
                    return Response(status_code=200)
                else:
                    # 两者不同，说明进入了推理阶段
                    # new_current_start_inference_time肯定和current_start_inference_time不同
                    # 再等20s，正常的话肯定在处理新请求或者是旧请求已经结束
                    current_start_inference_time = new_current_start_inference_time
                    time.sleep(20)
                    with self.model.model.hang_detection_lock:
                        new_current_start_inference_time = self.model.model.current_start_inference_time
                        new_previous_start_inference_time = self.model.model.previous_start_inference_time
                    # 推理完了，且没有新的请求
                    if new_previous_start_inference_time == new_current_start_inference_time:
                        logger.info("check_live: True") 
                        return Response(status_code=200)
                    else:
                        # 不同，可能是当前推理没结束，或者结束了又进入了新的推理。第一种情况判定为hang死，第二种情况判定为正常
                        if new_current_start_inference_time == current_start_inference_time:
                            warning_logger.error("check_live: False") 
                            return Response(status_code=500)
                        else:
                            logger.info("check_live: True") 
                            return Response(status_code=200)

            else:
                # 两者不同，说明进入了推理阶段
                # 再等20s，正常的话肯定在处理新请求或者是旧请求已经结束
                time.sleep(20)
                with self.model.model.hang_detection_lock:
                    new_current_start_inference_time = self.model.model.current_start_inference_time
                    new_previous_start_inference_time = self.model.model.previous_start_inference_time
                # 推理完了，且没有新的请求
                if new_previous_start_inference_time == new_current_start_inference_time:
                    logger.info("check_live: True") 
                    return Response(status_code=200)
                else:
                    # 不同，可能是当前推理没结束，或者结束了又进入了新的推理。第一种情况判定为hang死，第二种情况判定为正常
                    if new_current_start_inference_time == current_start_inference_time:
                        warning_logger.error("check_live: False") 
                        return Response(status_code=500)
                    else:
                        logger.info("check_live: True") 
                        return Response(status_code=200)

        else:
            warning_logger.error("check_live: False") 
            return Response(status_code=500)


async def watch_result():
    while True:
        await asyncio.sleep(0.01) # 10ms查询一次结果
        if response_dict:
            for task_id in response_dict:
                if task_id in event_dict:
                    response_checked_dict[task_id] = response_dict[task_id]
                    event_dict[task_id].set()
                
            for task_id in response_checked_dict:
                if task_id in response_dict:
                    del response_dict[task_id]


model_dir = os.getenv("MODEL_DIR", None)
if model_dir is None:
    raise ValueError("Environment variable MODEL_DIR must be set")
model_executor = ModelExecutor(model_dir)


    