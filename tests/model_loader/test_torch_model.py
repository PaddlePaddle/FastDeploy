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
    calculate_diff_rate,
    form_model_get_output_topp0,
    get_torch_model_path,
    run_with_timeout,
)

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
MAX_WAIT_SECONDS = 60 * 5

prompts = ["Where is Tiananmen Square in Beijing?"]


def check_result_against_baseline(outputs, baseline_file, threshold=0.05):
    """
    Check model outputs against baseline file.
    """
    try:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_content = f.read().strip()
    except FileNotFoundError:
        raise AssertionError(f"Baseline file not found: {baseline_file}")

    # Combine all outputs into a single string for comparison
    current_content = ""
    for idx, output in enumerate(outputs):
        # output format: (token_ids, text)
        _, text = output

        if isinstance(text, list):
            text_str = "".join(text)
        else:
            text_str = text

        current_content += text_str

    temp_file = "Qwen2.5-7B-Instruct-current"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(current_content)

    # Calculate difference rate
    diff_rate = calculate_diff_rate(current_content, baseline_content)

    if diff_rate >= threshold:
        raise AssertionError(
            f"Output differs from baseline file by too much ({diff_rate:.4%}):\n"
            f"Current output: {current_content!r}\n"
            f"Baseline content: {baseline_content!r}\n"
            f"Current output saved to: {temp_file}"
        )


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
                cfg.get("max_tokens", 100),
                marks=[pytest.mark.core_model],
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_model_len,quantization,max_tokens",
    hf_params,
)
def test_model_against_baseline(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
) -> None:
    """
    Test that model output matches baseline file.
    """
    torch_model_path = get_torch_model_path(model_name_or_path)

    # Run model
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

    # Determine baseline file path
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        baseline_file = os.path.join(base_path, "Qwen2.5-7B-Instruct-tp2")
    else:
        baseline_file = "Qwen2.5-7B-Instruct-tp2"

    # Compare against baseline file
    check_result_against_baseline(hf_outputs, baseline_file, threshold=0.05)
