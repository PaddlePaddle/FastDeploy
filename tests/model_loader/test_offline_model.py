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

import os
import sys

import pytest

prompts = ["解释下'温故而知新'", "who are you?"]

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.model_loader.utils import (
    check_tokens_id_and_text_close,
    form_model_get_output_topp0,
    get_torch_model_path,
    run_with_timeout,
)

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))

expected_output_map = {
    "Qwen3-30B-A3B-FP8": [
        (
            [
                106599,
                9370,
                109091,
                90395,
                107485,
                46944,
                99912,
                111564,
                1773,
                364,
                99416,
                99535,
                68536,
                52183,
                16628,
                6,
                99639,
                99700,
                110434,
                26940,
                67831,
                72881,
                25067,
                9370,
                115040,
                3837,
                111490,
                67338,
                107090,
                100052,
                107232,
                151645,
            ],
            "这句话的含义，并给出一个实际的例子。 '温故而知新'是一句出自《论语》的成语，意思是通过复习旧的知识",
        ),
        (
            [
                358,
                2776,
                264,
                3460,
                4128,
                1614,
                7881,
                553,
                54364,
                14817,
                11,
                323,
                358,
                2776,
                2598,
                1207,
                16948,
                13,
                358,
                646,
                1492,
                498,
                448,
                264,
                8045,
                315,
                9079,
                1741,
                438,
                35764,
                4755,
                151645,
            ],
            " I'm a large language model developed by Alibaba Cloud, and I'm called Qwen. I can help you with a variety of tasks such as answering questions",
        ),
    ],
}

model_param_map = {
    "Qwen3-30B-A3B-FP8": {
        "tensor_parallel_size": 2,
        "quantizations": [
            {
                "quant_type": "None",
                "backend": "triton",
                "env": {"DG_NVCC_OVERRIDE_CPP_STANDARD": "17"},
            },
        ],
    },
}

params = []
for model, cfg in model_param_map.items():
    for q in cfg["quantizations"]:
        if isinstance(q, dict):
            quant, backend, env = q["quant_type"], q.get("backend", "default"), q.get("env", {})
        else:
            quant, backend, env = q, "default", {}
        params.append(
            pytest.param(
                model,
                cfg.get("tensor_parallel_size", 1),
                cfg.get("max_model_len", 1024),
                quant,
                cfg.get("max_tokens", 32),
                env,
                marks=[pytest.mark.core_model],
                id=f"offline_quant_{model}.{quant}.{backend}",
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_model_len,quantization,max_tokens,env",
    params,
)
def test_offline_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
    env,
    monkeypatch,
) -> None:
    torch_model_path = get_torch_model_path(model_name_or_path)
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    fd_outputs = run_with_timeout(
        target=form_model_get_output_topp0,
        args=(
            fd_runner,
            torch_model_path,
            tensor_parallel_size,
            max_model_len,
            max_tokens,
            quantization,
            "default_v1",
            FD_ENGINE_QUEUE_PORT,
            prompts,
            FD_CACHE_QUEUE_PORT,
        ),
    )
    check_tokens_id_and_text_close(
        outputs_0_lst=fd_outputs,
        outputs_1_lst=(expected_output_map[model_name_or_path]),
        name_0="offline_quant_outputs",
        name_1="baseline",
    )
