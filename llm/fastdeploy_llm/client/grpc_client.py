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

import queue
import json
import sys
from functools import partial
import os
import time
import numpy as np
import subprocess
from fastdeploy_llm.utils.logging_util import logger
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import *
except:
    pass


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class GrpcClient:
    def __init__(self,
                 url: str,
                 model_name: str,
                 model_version: str="1",
                 timeout: int=1000000,
                 openai_port: int=None):
        """
        Args:
            url (`str`): inference server grpc url
            model_name (`str`)
            model_version (`str`): default "1"
            timeout (`int`): inference timeout in seconds
            openai_port (`int`)
        """
        self._model_name = model_name
        self._model_version = model_version
        self.timeout = timeout
        self.url = url

    def generate(self,
                 prompt: str,
                 request_id: str="0",
                 top_p: float=0.0,
                 temperature: float=1.0,
                 max_dec_len: int=1024,
                 min_dec_len: int=2,
                 penalty_score: float=1.0,
                 frequency_score: float=0.99,
                 eos_token_id: int=2,
                 presence_score: float=0.0,
                 stream: bool=False):
        import tritonclient.grpc as grpcclient
        #from tritonclient.utils import *

        user_data = UserData()
        req_dict = {
            "text": prompt,
            "topp": top_p,
            "temperature": temperature,
            "max_dec_len": max_dec_len,
            "min_dec_len": min_dec_len,
            "penalty_score": penalty_score,
            "frequency_score": frequency_score,
            "eos_token_id": eos_token_id,
            "model_test": "test",
            "presence_score": presence_score
        }

        inputs = [
            grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))
        ]
        outputs = [grpcclient.InferRequestedOutput("OUT")]

        in_data = np.array([json.dumps(req_dict)], dtype=np.object_)

        user_data = UserData()
        with grpcclient.InferenceServerClient(
                url=self.url, verbose=False) as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))
            inputs[0].set_data_from_numpy(in_data)
            triton_client.async_stream_infer(
                model_name=self._model_name,
                inputs=inputs,
                request_id=request_id,
                outputs=outputs)
            response = dict()
            response["token_ids"] = list()
            response["token_strs"] = list()
            response["input"] = req_dict
            while True:
                data_item = user_data._completed_requests.get(
                    timeout=self.timeout)
                if type(data_item) == InferenceServerException:
                    logger.error(
                        "Error happend while generating, status={}, msg={}".
                        format(data_item.status(), data_item.message()))
                    response["error_info"] = (data_item.status(),
                                              data_item.message())
                    break
                else:
                    results = data_item.as_numpy("OUT")[0]
                    data = json.loads(results)
                    response["token_ids"] += data["token_ids"]
                    response["token_strs"].append(data["result"])
                    if data.get("is_end", False):
                        break
            return response

    def async_generate(self,
                       prompt: str,
                       request_id: str="0",
                       top_p: float=0.0,
                       temperature: float=1.0,
                       max_dec_len: int=1024,
                       min_dec_len: int=2,
                       penalty_score: float=1.0,
                       frequency_score: float=0.99,
                       eos_token_id: int=2,
                       presence_score: float=0.0,
                       stream: bool=False):
        import tritonclient.grpc as grpcclient
        #from tritonclient.utils import *

        user_data = UserData()
        req_dict = {
            "text": prompt,
            "topp": top_p,
            "temperature": temperature,
            "max_dec_len": max_dec_len,
            "min_dec_len": min_dec_len,
            "penalty_score": penalty_score,
            "frequency_score": frequency_score,
            "eos_token_id": eos_token_id,
            "model_test": "test",
            "presence_score": presence_score
        }

        inputs = [
            grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))
        ]
        outputs = [grpcclient.InferRequestedOutput("OUT")]

        in_data = np.array([json.dumps(req_dict)], dtype=np.object_)

        user_data = UserData()
        with grpcclient.InferenceServerClient(
                url=self.url, verbose=False) as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))
            inputs[0].set_data_from_numpy(in_data)
            triton_client.async_stream_infer(
                model_name=self._model_name,
                inputs=inputs,
                request_id=request_id,
                outputs=outputs)
            while True:
                data_item = user_data._completed_requests.get(
                    timeout=self.timeout)
                if type(data_item) == InferenceServerException:
                    logger.error(
                        "Error happend while generating, status={}, msg={}".
                        format(data_item.status(), data_item.message()))
                    break
                else:
                    results = data_item.as_numpy("OUT")[0]
                    data = json.loads(results)
                    yield data
                    if data.get("is_end", False):
                        break
