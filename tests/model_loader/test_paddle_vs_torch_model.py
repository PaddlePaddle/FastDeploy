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

import pytest

from tests.model_loader.utils import (
    check_tokens_id_and_text_close,
    form_model_get_output_topp0,
    get_paddle_model_path,
    get_torch_model_path,
    run_with_timeout,
)

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
MAX_WAIT_SECONDS = 60 * 5

prompts = ["解释下“温故而知新", "Hello, how are you?"]

hugging_face_model_param_map = {
    "Qwen2.5-7B-Instruct": {
        "tensor_parallel_size": 2,
        "quantizations": ["None"],
    },
}
hf_params = []
for model, cfg in hugging_face_model_param_map.items():
    for q in cfg["quantizations"]:
        hf_params.append(
            pytest.param(
                model,
                cfg.get("tensor_parallel_size", 1),
                cfg.get("max_model_len", 1024),
                q,
                cfg.get("max_tokens", 32),
                marks=[pytest.mark.core_model],
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_model_len,quantization,max_tokens",
    hf_params,
)
def test_paddle_vs_torch_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
) -> None:
    fd_model_path = get_paddle_model_path(model_name_or_path)
    torch_model_path = get_torch_model_path(model_name_or_path)
    paddle_outputs = run_with_timeout(
        target=form_model_get_output_topp0,
        args=(
            fd_runner,
            fd_model_path,
            tensor_parallel_size,
            max_model_len,
            max_tokens,
            quantization,
            "default",
            FD_ENGINE_QUEUE_PORT,
            prompts,
        ),
    )
    hf_outputs = run_with_timeout(
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
        ),
    )

    check_tokens_id_and_text_close(
        outputs_0_lst=paddle_outputs,
        outputs_1_lst=hf_outputs,
        name_0="Paddle model (default loader)",
        name_1="HuggingFace model (default_v1 loader)",
    )
