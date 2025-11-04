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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.model_loader.utils import (
    check_tokens_id_and_text_close,
    form_model_get_output_topp0,
    get_paddle_model_path,
    run_with_timeout,
)

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))

prompts = ["解释下”温故而知新”", "Hello, how are you?"]

# {id,baseline}
baseline = {
    "Qwen3-30B-A3B.block_wise_fp8.triton": [
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
                1036,
                99416,
                99535,
                68536,
                52183,
                16628,
                854,
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
            "这句话的含义，并给出一个实际的例子。 “温故而知新”是一句出自《论语》的成语，意思是通过复习旧的知识",
        ),
        (
            [
                358,
                2776,
                4460,
                311,
                1477,
                279,
                897,
                315,
                279,
                25098,
                315,
                279,
                729,
                282,
                2075,
                8,
                284,
                220,
                16,
                11884,
                87,
                61,
                17,
                488,
                220,
                16,
                8,
                504,
                856,
                284,
                481,
                151645,
            ],
            " I'm trying to find the value of the integral of the function f(x) = 1/(x^2 + 1) from x = -",
        ),
    ],
    "Qwen3-30B-A3B.block_wise_fp8.deepgemm": [
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
                1036,
                99416,
                99535,
                68536,
                52183,
                16628,
                854,
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
            "这句话的含义，并给出一个实际的例子。 “温故而知新”是一句出自《论语》的成语，意思是通过复习旧的知识",
        ),
        (
            [
                358,
                2776,
                4460,
                311,
                11625,
                419,
                3491,
                25,
                330,
                9885,
                279,
                897,
                315,
                279,
                7493,
                25,
                220,
                16,
                15,
                15,
                15,
                14,
                16,
                15,
                15,
                15,
                488,
                220,
                16,
                15,
                15,
                151645,
            ],
            " I'm trying to solve this problem: \"Find the value of the expression: 1000/1000 + 100",
        ),
    ],
}

model_param_map = {
    "Qwen3-30B-A3B": {
        "tensor_parallel_size": 2,
        "max_num_seqs": 1,
        "quantizations": [
            {
                "quant_type": "block_wise_fp8",
                "backend": "triton",
                "env": {"DG_NVCC_OVERRIDE_CPP_STANDARD": "17"},
            },
            {
                "quant_type": "block_wise_fp8",
                "backend": "deepgemm",
                "env": {"DG_NVCC_OVERRIDE_CPP_STANDARD": "17", "FD_USE_DEEP_GEMM": "1"},
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
                cfg.get("torch_model_name_or_path", ""),
                cfg.get("tensor_parallel_size", 1),
                cfg.get("max_num_seqs", 1),
                cfg.get("max_model_len", 1024),
                quant,
                cfg.get("max_tokens", 32),
                env,
                marks=[pytest.mark.core_model],
                id=f"{model}.{quant}.{backend}",
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,torch_model_name_or_path,tensor_parallel_size,max_num_seqs,max_model_len,quantization,max_tokens,env",
    params,
)
def test_common_model(
    fd_runner,
    model_name_or_path: str,
    torch_model_name_or_path: str,
    tensor_parallel_size: int,
    max_num_seqs,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
    env,
    request,
    monkeypatch,
) -> None:
    model_path = get_paddle_model_path(model_name_or_path)
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    form_model_get_output = form_model_get_output_topp0

    fd_outputs_v1 = run_with_timeout(
        target=form_model_get_output,
        args=(
            fd_runner,
            model_path,
            tensor_parallel_size,
            max_num_seqs,
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
        outputs_0_lst=baseline[request.node.callspec.id],
        outputs_1_lst=fd_outputs_v1,
        name_0="default loader",
        name_1="default_v1 loader",
    )
