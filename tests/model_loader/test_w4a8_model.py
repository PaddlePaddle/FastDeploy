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

import os
import weakref

import pytest

from fastdeploy.entrypoints.llm import LLM

bash_path = os.getenv("MODEL_PATH")
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 9961))
FD_ENGINE_QUEUE_PORTS = [
    [FD_ENGINE_QUEUE_PORT, FD_ENGINE_QUEUE_PORT + 1],
    [FD_ENGINE_QUEUE_PORT + 2, FD_ENGINE_QUEUE_PORT + 3],
    [FD_ENGINE_QUEUE_PORT - 1, FD_ENGINE_QUEUE_PORT - 2],
    [FD_ENGINE_QUEUE_PORT - 3, FD_ENGINE_QUEUE_PORT - 4],
]
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))
FD_CACHE_QUEUE_PORTS = [FD_CACHE_QUEUE_PORT, FD_CACHE_QUEUE_PORT + 1, FD_CACHE_QUEUE_PORT + 2, FD_CACHE_QUEUE_PORT + 3]


models = [
    "ernie-4_5-fake-w4a8-unpermuted",
    "ernie-4_5-fake-w4a8-permuted",
    "ernie-4_5-fake-w4afp8-unpermuted",
    "ernie-4_5-fake-w4afp8-permuted",
]

prompts = ["解释下“温故而知新"]


@pytest.fixture(scope="module", params=models)
def llm(request):
    """LLM测试夹具"""
    model_path = os.path.join(bash_path, request.param)
    try:
        port_index = models.index(request.param) % len(FD_ENGINE_QUEUE_PORTS)
        llm_instance = LLM(
            model=model_path,
            tensor_parallel_size=1,
            data_parallel_size=2,
            max_model_len=8192,
            num_gpu_blocks_override=1024,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORTS[port_index],
            cache_queue_port=FD_CACHE_QUEUE_PORTS[port_index],
            load_choices="default",
            enable_expert_parallel=True,
        )
        yield weakref.proxy(llm_instance)
    except Exception as e:
        assert False, f"LLM initialization failed: {e}"


@pytest.mark.timeout(60)
def test_generation(llm):
    print(f"testing generation with model: {llm}")
    # topp_params = SamplingParams(temperature=0.1, top_p=0, max_tokens=20)
    # output = llm.generate(prompts=prompts, sampling_params=topp_params)
    # print(output)
