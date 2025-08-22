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
from multiprocessing import Process, Queue

import pytest

from .utils import check_tokens_id_and_text_close

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
MAX_WAIT_SECONDS = 60 * 5

prompts = ["解释下“温故而知新", "Hello, how are you?"]


def get_model_paths(base_model_name: str) -> tuple[str, str]:
    """return (fastdeploy_path, huggingface_path)"""
    # FastDeploy model path
    fd_base_path = os.getenv("FD_MODEL_PATH")
    if fd_base_path:
        fd_model_path = os.path.join(fd_base_path, base_model_name)
    else:
        fd_model_path = base_model_name

    # HuggingFace model path
    hf_base_path = os.getenv("HF_MODEL_PATH")
    if hf_base_path:
        hf_model_path = os.path.join(hf_base_path, base_model_name)
    else:
        hf_model_path = base_model_name

    return fd_model_path, hf_model_path


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


@pytest.mark.parametrize(
    "model_name_or_path,tensor_parallel_size,max_model_len",
    [
        pytest.param(
            "Qwen2.5-7B-Instruct",
            2,
            1024,
            marks=[pytest.mark.core_model],
        ),
    ],
)
@pytest.mark.parametrize("quantization", ["None"])
@pytest.mark.parametrize("max_tokens", [32])
def test_common_model(
    fd_runner,
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_tokens: int,
    quantization: str,
) -> None:
    fd_model_path, hf_model_path = get_model_paths(model_name_or_path)
    result_queue = Queue()
    p = Process(
        target=form_model_get_output,
        args=(
            fd_runner,
            fd_model_path,
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
            hf_model_path,
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
