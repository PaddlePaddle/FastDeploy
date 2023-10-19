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
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

import api_client


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class grpcClient:

    def __init__(self,
                 base_url: str,
                 model_name: str,
                 model_version: str = "1",
                 timeout: int = 100,
                 openai_port: int = None):
        """
        Args:
            base_url (`str`): inference server grpc url
            model_name (`str`)
            model_version (`str`): default "1"
            timeout (`int`): inference timeout in seconds
            openai_port (`int`) 
        """
        self._model_name = model_name
        self._model_version = model_version
        self.timeout = timeout
        self._client = grpcclient.InferenceServerClient(base_url,
                                                        verbose=False)

        error = self._verify_triton_state(self._client)
        if error:
            raise RuntimeError(
                f"Could not communicate to Triton Server: {error}")

        self.inputs = [
            grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))
        ]
        self.outputs = [grpcclient.InferRequestedOutput("OUT")]
        self.has_init = False
        self.user_data = UserData()

        if openai_port is not None:
            pd_cmd = "python3 api_client.py --url {0} --port {1} --model {2}".format(
                base_url, openai_port, model_name)
            subprocess.Popen(pd_cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid)
            time.sleep(5)

    def _verify_triton_state(self, triton_client):
        if not triton_client.is_server_live():
            return f"Triton server {self._server_url} is not live"
        elif not triton_client.is_server_ready():
            return f"Triton server {self._server_url} is not ready"
        elif not triton_client.is_model_ready(self._model_name,
                                              self._model_version):
            return f"Model {self._model_name}:{self._model_version} is not ready"
        return None

    def generate(self,
                 prompt: str,
                 request_id: str = "0",
                 top_p: float = 0.0,
                 temperature: float = 1.0,
                 max_dec_len: int = 1024,
                 min_dec_len: int = 2,
                 penalty_score: float = 1.0,
                 frequency_score: float = 0.99,
                 eos_token_id: int = 2,
                 presence_score: float = 0.0,
                 stream: bool = False):

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

        try:
            if not self.has_init:
                self._client.start_stream(
                    callback=partial(callback, self.user_data))
                self.has_init = True
            else:
                self.user_data.reset()
                self.inputs = [
                    grpcclient.InferInput("IN", [1],
                                          np_to_triton_dtype(np.object_))
                ]
                self.outputs = [grpcclient.InferRequestedOutput("OUT")]

            in_data = np.array([json.dumps(req_dict)], dtype=np.object_)
            self.inputs[0].set_data_from_numpy(in_data)

            self._client.async_stream_infer(model_name=self._model_name,
                                            inputs=self.inputs,
                                            request_id=request_id,
                                            outputs=self.outputs)
            if stream:
                completion = []
            else:
                completion = ""
            while True:
                data_item = self.user_data._completed_requests.get(
                    timeout=self.timeout)
                if type(data_item) == InferenceServerException:
                    print('Exception:', 'status', data_item.status(), 'msg',
                          data_item.message())
                else:
                    results = data_item.as_numpy("OUT")[0]
                    data = json.loads(results)
                    if stream:
                        completion.append(data["result"])
                    else:
                        completion += data["result"]
                    if data.get("is_end", False):
                        break
            return completion
        except Exception as e:
            print(f"Client infer error: {e}")
            raise e
