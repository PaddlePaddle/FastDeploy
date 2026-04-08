# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import time
from typing import Any, Union

import pytest
from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    clean_ports,
)


class FDRunner:
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 1,
        max_model_len: int = 1024,
        load_choices: str = "default",
        quantization: str = "None",
        **kwargs,
    ) -> None:
        from fastdeploy.entrypoints.llm import LLM

        clean_ports()
        time.sleep(10)
        graph_optimization_config = {"use_cudagraph": False}
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            load_choices=load_choices,
            quantization=quantization,
            max_num_batched_tokens=max_model_len,
            graph_optimization_config=graph_optimization_config,
            port=FD_API_PORT,
            cache_queue_port=FD_CACHE_QUEUE_PORT,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:

        req_outputs = self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)
        outputs: list[tuple[list[list[int]], list[str]]] = []
        for output in req_outputs:
            outputs.append((output.outputs.token_ids, output.outputs.text))
        return outputs

    def generate_topp0(
        self,
        prompts: Union[list[str]],
        max_tokens: int,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        from fastdeploy.engine.sampling_params import SamplingParams

        topp_params = SamplingParams(temperature=0.0, top_p=0, max_tokens=max_tokens)
        outputs = self.generate(prompts, topp_params, **kwargs)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.llm


@pytest.fixture(scope="session")
def fd_runner():
    return FDRunner
