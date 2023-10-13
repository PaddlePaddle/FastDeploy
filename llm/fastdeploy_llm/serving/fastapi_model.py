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

import argparse

from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

model = None
server_config = None


def parse(parameters_config, name, default_value=None):
    if name not in parameters_config:
        if default_value:
            return default_value
        else:
            raise Exception(
                "Cannot find key:{} while parsing parameters.".format(name))
    return parameters_config[name]["string_value"]


@app.post("/generate")
async def generate(request: Request) -> Response:
    try:
        request_dict = await request.json()
    except Exception as e:
        logger.info("Got an illegal request, error={}".format(e))
        return Response(status_code=499)

    task = Task()
    task.task_id = str(uuid.uuid4())

    try:
        task.from_dict(request_dict)
    except Exception as e:
        logger.info(
            "There's error while deserializing data from request, error={}".
            format(e))
        return Response(status_code=499)

    try:
        task.check(server_config.max_dec_len)
    except Exception as e:
        logger.info("There's error while checking task, error={}".format(e))
        return Response(status_code=499)

    if model.requests_queue.qsize() > server_config.max_queue_num:
        logger.info("The queue is full now(size={}), please wait for a while.".
                    format(model.max_queue_num))
        return Response(status_code=499)

    results_generator = model.generate(task)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    return JSONResponse(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--disable_dynamic_batching", type=int, default=1)
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    config = fdlm.Config(args.model_path)
    config.max_batch_size = args.max_batch_size
    config.disable_dynamic_batching = args.disable_dynamic_batching

    server_config = config
    response_handler = dict()
    model = ServingModel(config)
    model.model.stream_sender = response_handler
    model.start()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
