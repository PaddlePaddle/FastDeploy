# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import numpy as np
from typing import Optional
import json
import ast

from pprint import pprint
from tritonclient import utils as client_utils
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, service_pb2_grpc, service_pb2

LOGGER = logging.getLogger("run_inference_on_triton")


class SyncGRPCTritonRunner:
    DEFAULT_MAX_RESP_WAIT_S = 120

    def __init__(
            self,
            server_url: str,
            model_name: str,
            model_version: str,
            *,
            verbose=False,
            resp_wait_s: Optional[float]=None, ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._verbose = verbose
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s

        self._client = InferenceServerClient(
            self._server_url, verbose=self._verbose)
        error = self._verify_triton_state(self._client)
        if error:
            raise RuntimeError(
                f"Could not communicate to Triton Server: {error}")

        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} "
            f"are up and ready!")

        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(self._model_name,
                                                         self._model_version)
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        # for tm in model_metadata.inputs:
        #     print("tm:", tm)
        self._inputs = {tm.name: tm for tm in model_metadata.inputs}
        self._input_names = list(self._inputs)
        self._outputs = {tm.name: tm for tm in model_metadata.outputs}
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]

    def Run(self, inputs):
        """
        Args:
            inputs: list, Each value corresponds to an input name of self._input_names
        Returns:
            results: dict, {name : numpy.array}
        """
        infer_inputs = []
        for idx, data in enumerate(inputs):
            data = json.dumps(data)
            data = np.array([[data], ], dtype=np.object_)
            infer_input = InferInput(self._input_names[idx], data.shape,
                                     "BYTES")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)

        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=self._outputs_req,
            client_timeout=self._response_wait_t, )
        # results = {name: results.as_numpy(name) for name in self._output_names}
        results = results.as_numpy(self._output_names[0])
        return results

    def _verify_triton_state(self, triton_client):
        if not triton_client.is_server_live():
            return f"Triton server {self._server_url} is not live"
        elif not triton_client.is_server_ready():
            return f"Triton server {self._server_url} is not ready"
        elif not triton_client.is_model_ready(self._model_name,
                                              self._model_version):
            return f"Model {self._model_name}:{self._model_version} is not ready"
        return None


if __name__ == "__main__":
    model_name = "uie"
    model_version = "1"
    url = "localhost:8001"
    runner = SyncGRPCTritonRunner(url, model_name, model_version)

    print("1. Named Entity Recognition Task--------------")
    schema = ["时间", "选手", "赛事名称"]
    print(f"The extraction schema: {schema}")
    text = ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"]
    print("text=", text)
    print("results:")
    results = runner.Run([text, schema])
    for result in results:
        result = result.decode('utf-8')
        result = ast.literal_eval(result)
        pprint(result)

    print("================================================")
    text = ["2月7日北京冬奥会短道速滑男子1000米决赛中任子威获得冠军！"]
    print("text=", text)
    # while schema is empty, use the schema set up last time.
    schema = []
    results = runner.Run([text, schema])
    print("results:")
    for result in results:
        result = result.decode('utf-8')
        result = ast.literal_eval(result)
        pprint(result)

    print("\n2. Relation Extraction Task")
    schema = {"竞赛名称": ["主办方", "承办方", "已举办次数"]}
    print(f"The extraction schema: {schema}")
    text = [
        "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作"
        "委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"
    ]
    print("text=", text)
    print("results:")
    results = runner.Run([text, schema])
    for result in results:
        result = result.decode('utf-8')
        result = ast.literal_eval(result)
        pprint(result)
