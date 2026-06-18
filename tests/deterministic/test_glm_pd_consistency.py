# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from model_loader.utils import (
    compare_p_d_logprobs,
    form_model_pd_consistency_test,
    get_paddle_model_path,
    run_with_timeout,
)

model_param_map = {
    "GLM-4.5-Air-Fake": {
        "tensor_parallel_size": 2,
        "quantizations": [None],
        "max_num_seqs": 1,
        "graph_optimization_config": {"use_cudagraph": False},
        "max_tokens": 20,
        "env": {
            "FLAGS_flash_attn_version": "3",
            "FD_ATTENTION_BACKEND": "FLASH_ATTN",
            "QKRMSNORM_USE_PHI_RMSNORM": "0",
            "FD_USE_PHI_MOE_TOPK": "1",
            "FD_USE_PHI_MOE_PERMUTE": "1",
            "FD_USE_PHI_FP8_QUANT": "1",
            "MOE_PROB_IN_ADVANCE": "1",
            "RL_EB5_NEED_SKIP_QUANT": "1",
            "FD_SiluAndMul_USE_PHI_SWIGLU": "1",
            "FD_USE_GET_SAVE_OUTPUT_V1": "1",
            "FLAGS_use_legacy_linear": "1",
            "FD_ENABLE_RL": "1",
            "FD_DETERMINISTIC_MODE": "1",
            "FD_SKIP_IN_DETERMINISTIC": "1",
        },
    }
}
prompts = ["将英文谚语'Actions speak louder than words'翻译成中文。"]

params = []
for model, cfg in model_param_map.items():
    top_env = cfg.get("env", {})
    for q in cfg["quantizations"]:
        if isinstance(q, dict):
            quant, backend, q_env = q["quant_type"], q.get("backend", "default"), q.get("env", {})
        else:
            quant, backend, q_env = q, "default", {}
        # 合并顶层 env 和 quantization 级别的 env
        merged_env = {**top_env, **q_env}
        params.append(
            pytest.param(
                model,
                cfg.get("tensor_parallel_size", 1),
                cfg.get("max_num_seqs", 1),
                cfg.get("max_model_len", 1024),
                quant,
                cfg.get("max_tokens", 128),
                merged_env,
                marks=[pytest.mark.core_model],
                id=f"{model}.{quant}.{backend}",
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_num_seqs,max_model_len,quantization,max_tokens,env",
    params,
)
def test_model_against_baseline(
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
    """
    Test that model output matches baseline file.
    """
    model_path = get_paddle_model_path(model_name_or_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    d_output, p_output, eos_token_id = run_with_timeout(
        target=form_model_pd_consistency_test,
        kwargs={
            "fd_runner": fd_runner,
            "prompts": prompts,
            "llm_params": {
                "model_name_or_path": str(model_path),
                "tensor_parallel_size": tensor_parallel_size,
                "max_num_seqs": max_num_seqs,
                "max_model_len": max_model_len,
                "quantization": quantization,
                "load_choices": "dummy",
                "enable_logprob": True,
                "enable_prefix_caching": False,
            },
            "d_sampling_params": {
                "top_p": 0.0,
                "max_tokens": max_tokens,
                "min_tokens": max_tokens,
                "temperature": 1,
                "n": 1,
                "logprobs": 0,
                "seed": 1,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
            "p_sampling_params": {
                "top_p": 0.0,
                "max_tokens": 1,
                "min_tokens": 1,
                "temperature": 1,
                "prompt_logprobs": 0,
                "n": 1,
                "logprobs": 0,
                "seed": 1,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
        },
    )
    compare_p_d_logprobs(d_output, p_output, eos_token_id)
