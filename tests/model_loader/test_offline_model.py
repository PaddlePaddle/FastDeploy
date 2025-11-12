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
    form_model_get_output_topp0,
    get_torch_model_path,
    run_with_timeout,
)

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
        "max_num_seqs": 1,
        "graph_optimization_config": {"use_cudagraph": False},
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
                cfg.get("max_num_seqs", 1),
                cfg.get("max_model_len", 1024),
                quant,
                cfg.get("max_tokens", 32),
                env,
                marks=[pytest.mark.core_model],
                id=f"offline_quant_{model}.{quant}.{backend}",
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_num_seqs,max_model_len,quantization,max_tokens,env",
    params,
)
def test_offline_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_num_seqs: int,
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

    _ = run_with_timeout(
        target=form_model_get_output_topp0,
        args=(
            fd_runner,
            torch_model_path,
            tensor_parallel_size,
            max_num_seqs,
            max_model_len,
            max_tokens,
            quantization,
            "default_v1",
            prompts,
        ),
    )
