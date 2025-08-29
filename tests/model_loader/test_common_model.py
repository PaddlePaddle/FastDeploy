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
import shutil
import traceback
import warnings
from multiprocessing import Process, Queue

import pytest

os.environ["LOAD_STATE_DICT_THREAD_NUM"] = "1"
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
MAX_WAIT_SECONDS = 60 * 5

prompts = ["解释下“温故而知新", "Hello, how are you?"]
TokensIdText = list[tuple[list[int], str]]
# (token_ids, text)


def get_model_paths(base_model_name: str) -> tuple[str, str]:
    """return (fastdeploy_path, huggingface_path)"""
    # FastDeploy model path
    fd_base_path = os.getenv("MODEL_PATH")
    if fd_base_path:
        fd_model_path = os.path.join(fd_base_path, base_model_name)
    else:
        fd_model_path = base_model_name

    # HuggingFace model path
    torch_model_path = os.path.join(
        fd_base_path,
        "torch",
        base_model_name,
    )

    return fd_model_path, torch_model_path


def clear_logs():
    log_path = os.path.join(os.getcwd(), "log")
    if os.path.exists(log_path):
        try:
            shutil.rmtree(log_path)
            print(f"Deleted log directory: {log_path}")
        except Exception as e:
            print(f"Failed to delete log directory {log_path}: {e}")
    else:
        print(f"No log directory found at {log_path}")


def print_logs():
    log_dir = os.path.join(os.getcwd(), "log")
    log_file = os.path.join(log_dir, "workerlog.0")

    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return

    print(f"\n===== {log_file} start =====")
    with open(log_file, "r") as f:
        for line in f:
            print(line, end="")
    print(f"\n===== {log_file} end =====\n")


def check_tokens_id_and_text_close(
    *,
    outputs_0_lst: TokensIdText,
    outputs_1_lst: TokensIdText,
    name_0: str,
    name_1: str,
    warn_on_mismatch: bool = True,
) -> None:
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (outputs_0, outputs_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        assert len(outputs_0) == len(outputs_1)
        output_ids_0, output_str_0 = outputs_0
        output_ids_1, output_str_1 = outputs_1

        # Loop through generated tokens.
        for idx, (output_id_0, output_id_1) in enumerate(zip(output_ids_0, output_ids_1)):
            is_tok_mismatch = output_id_0 != output_id_1
            if is_tok_mismatch and warn_on_mismatch:
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\nMatched tokens:\t{output_ids_0[:idx]}"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(fail_msg, stacklevel=2)
                break
    else:
        if output_str_0 != output_str_1 and warn_on_mismatch:
            fail_msg = f"Test{prompt_idx}:" f"\n{name_0}:\t{output_str_0!r}" f"\n{name_1}:\t{output_str_1!r}"
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(fail_msg, stacklevel=2)


def form_model_get_output(
    fd_runner,
    model_path,
    tensor_parallel_size,
    max_model_len,
    max_tokens,
    quantization,
    load_choices,
    result_queue,
):
    try:
        with fd_runner(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            load_choices=load_choices,
            quantization=quantization,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
        ) as fd_model:
            fd_outputs = fd_model.generate_topp0(prompts, max_tokens=max_tokens)
            result_queue.put(fd_outputs)
    except Exception:
        print(f"Failed using {load_choices} laoder to load model from {model_path}.")
        traceback.print_exc()
        pytest.fail(f"Failed to initialize LLM model from {model_path}")


def run_with_timeout(target, args, timeout=60 * 5):
    clear_logs()
    result_queue = Queue()
    p = Process(target=target, args=(*args, result_queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        print_logs()
        raise RuntimeError("Worker process hung and was terminated")
    try:
        return result_queue.get(timeout=60)
    except Exception as e:
        raise RuntimeError(f"Failed to get result from worker: {e}")


model_param_map = {
    "Qwen3-0.6B": {
        "quantizations": ["None", "wint4", "wint8"],
    },
    "ernie-4_5-21b-a3b-bf16-paddle": {
        "tensor_parallel_size": 2,
        "quantizations": [
            "wint8",
        ],
    },
    "Qwen2-7B-Instruct": {
        "quantizations": ["wint4"],
    },
    "Qwen3-30B-A3B": {
        "tensor_parallel_size": 2,
        "quantizations": [
            {
                "quant_type": "block_wise_fp8",
                "backend": "triton",
                "env": {"DG_NVCC_OVERRIDE_CPP_STANDARD": "17"},
            },
            {"quant_type": "block_wise_fp8", "backend": "deepgemm", "env": {"DG_NVCC_OVERRIDE_CPP_STANDARD": "17"}},
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
                id=f"{model}.{quant}.{backend}",
            )
        )


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_model_len,quantization,max_tokens,env",
    params,
)
def test_common_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
    env,
    monkeypatch,
) -> None:
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, model_name_or_path)
    else:
        model_path = model_name_or_path
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    fd_outputs_v0 = run_with_timeout(
        target=form_model_get_output,
        args=(fd_runner, model_path, tensor_parallel_size, max_model_len, max_tokens, quantization, "default"),
    )
    fd_outputs_v1 = run_with_timeout(
        target=form_model_get_output,
        args=(fd_runner, model_path, tensor_parallel_size, max_model_len, max_tokens, quantization, "default_v1"),
    )
    check_tokens_id_and_text_close(
        outputs_0_lst=fd_outputs_v0,
        outputs_1_lst=fd_outputs_v1,
        name_0="default loader",
        name_1="default_v1 loader",
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

    fd_model_path, torch_model_path = get_model_paths(model_name_or_path)

    paddle_outputs = run_with_timeout(
        target=form_model_get_output,
        args=(fd_runner, fd_model_path, tensor_parallel_size, max_model_len, max_tokens, quantization, "default"),
    )

    hf_outputs = run_with_timeout(
        target=form_model_get_output,
        args=(
            fd_runner,
            torch_model_path,
            tensor_parallel_size,
            max_model_len,
            max_tokens,
            quantization,
            "default_v1",
        ),
    )

    check_tokens_id_and_text_close(
        outputs_0_lst=paddle_outputs,
        outputs_1_lst=hf_outputs,
        name_0="Paddle model (default loader)",
        name_1="HuggingFace model (default_v1 loader)",
    )
