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
import traceback
import warnings
from multiprocessing import Process, Queue

import pytest

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
MAX_WAIT_SECONDS = 60 * 5

prompts = ["解释下“温故而知新", "Hello, how are you?"]
TokensIdText = list[tuple[list[int], str]]
# (token_ids, text)


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


model_param_map = {
    "Qwen3-0.6B": {
        "quantizations": ["None", "wint4", "wint8"],
    },
    "ernie-4_5-21b-a3b-bf16-paddle": {
        "tensor_parallel_size": 2,
        "quantizations": ["wint8"],
    },
}

params = []
for model, cfg in model_param_map.items():
    for q in cfg["quantizations"]:
        params.append(
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
    params,
)
def test_common_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
) -> None:
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, model_name_or_path)
    else:
        model_path = model_name_or_path
    result_queue = Queue()
    p = Process(
        target=form_model_get_output,
        args=(
            fd_runner,
            model_path,
            tensor_parallel_size,
            max_model_len,
            max_tokens,
            quantization,
            "default",
            result_queue,
        ),
    )
    p.start()
    p.join()
    fd_outputs_v0 = result_queue.get(timeout=60)

    p = Process(
        target=form_model_get_output,
        args=(
            fd_runner,
            model_path,
            tensor_parallel_size,
            max_model_len,
            max_tokens,
            quantization,
            "default_v1",
            result_queue,
        ),
    )
    p.start()
    p.join()
    fd_outputs_v1 = result_queue.get(timeout=60)
    check_tokens_id_and_text_close(
        outputs_0_lst=fd_outputs_v0,
        outputs_1_lst=fd_outputs_v1,
        name_0="default loader",
        name_1="default_v1 loader",
    )
