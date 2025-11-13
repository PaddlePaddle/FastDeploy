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

os.environ["FD_USE_MACHETE"] = "0"

from tests.model_loader.utils import (
    calculate_diff_rate,
    form_model_get_output_topp0,
    get_torch_model_path,
    run_with_timeout,
)

prompts = ["北京天安门在哪里?"]


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

    temp_file = f"{os.path.basename(baseline_file)}-current"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(current_content)

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
        "max_num_seqs": 1,
        "tensor_parallel_size": 2,
        "quantizations": ["wint8"],
    },
    "Qwen3-30B-A3B": {
        "max_num_seqs": 1,
        "tensor_parallel_size": 2,
        "quantizations": ["wint8"],
    },
}

hf_params = []
for model, cfg in hugging_face_model_param_map.items():
    for q in cfg["quantizations"]:
        hf_params.append(
            pytest.param(
                model,
                cfg.get("tensor_parallel_size", 2),
                cfg.get("max_num_seqs", 1),
                cfg.get("max_model_len", 1024),
                q,
                cfg.get("max_tokens", 100),
                marks=[pytest.mark.core_model],
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_num_seqs,max_model_len,quantization,max_tokens",
    hf_params,
)
def test_model_against_baseline(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_num_seqs: int,
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
            max_num_seqs,
            max_model_len,
            max_tokens,
            quantization,
            "default_v1",
            prompts,
        ),
    )

    # Determine baseline file path based on model name
    base_path = os.getenv("MODEL_PATH", "")

    # Get baseline suffix from config
    model_config = hugging_face_model_param_map.get(model_name_or_path, {})
    baseline_suffix = model_config.get("baseline_suffix", "tp2")
    baseline_filename = f"{model_name_or_path}-{baseline_suffix}"

    if base_path:
        baseline_file = os.path.join(base_path, baseline_filename)
    else:
        baseline_file = baseline_filename

    # Compare against baseline file
    check_result_against_baseline(hf_outputs, baseline_file, threshold=0.05)
