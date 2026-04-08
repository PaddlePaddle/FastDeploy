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
import signal
import socket
import subprocess
import time
import traceback

import pytest

from fastdeploy import LLM, SamplingParams

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))
MAX_WAIT_SECONDS = 60


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def format_chat_prompt(messages):
    """
    Format multi-turn conversation into prompt string, suitable for chat models.
    Uses Qwen2 style with <|im_start|> / <|im_end|> tokens.
    """
    prompt = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


@pytest.fixture(scope="module")
def model_path():
    """
    Get model path from environment variable MODEL_PATH,
    default to "./Qwen2-7B-Instruct" if not set.
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        return os.path.join(base_path, "Qwen2-7B-Instruct")
    else:
        return "./Qwen2-7B-Instruct"


@pytest.fixture(scope="module")
def llm(model_path):
    """
    Fixture to initialize the LLM model with a given model path
    """
    try:
        output = subprocess.check_output(f"lsof -i:{FD_ENGINE_QUEUE_PORT} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {FD_ENGINE_QUEUE_PORT}, pid={pid}")
    except subprocess.CalledProcessError:
        pass

    try:
        start = time.time()
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
            cache_queue_port=FD_CACHE_QUEUE_PORT,
            max_model_len=32768,
            quantization="wint8",
            logits_processors=["LogitBiasLogitsProcessor"],
        )

        # Wait for the port to be open
        wait_start = time.time()
        while not is_port_open("127.0.0.1", FD_ENGINE_QUEUE_PORT):
            if time.time() - wait_start > MAX_WAIT_SECONDS:
                pytest.fail(
                    f"Model engine did not start within {MAX_WAIT_SECONDS} seconds on port {FD_ENGINE_QUEUE_PORT}"
                )
            time.sleep(1)

        print(f"Model loaded successfully from {model_path} in {time.time() - start:.2f}s.")
        yield llm
    except Exception:
        print(f"Failed to load model from {model_path}.")
        traceback.print_exc()
        pytest.fail(f"Failed to initialize LLM model from {model_path}")


def test_generate_prompts(llm):
    """
    Test basic prompt generation
    """
    # Only one prompt enabled for testing currently
    prompts = [
        "请介绍一下中国的四大发明。",
        "太阳和地球之间的距离是多少？",
        "写一首关于春天的古风诗。",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )

    try:
        outputs = llm.generate(prompts, sampling_params)

        # Verify basic properties of the outputs
        assert len(outputs) == len(prompts), "Number of outputs should match number of prompts"

        for i, output in enumerate(outputs):
            assert output.prompt == prompts[i], f"Prompt mismatch for case {i + 1}"
            assert isinstance(output.outputs.text, str), f"Output text should be string for case {i + 1}"
            assert len(output.outputs.text) > 0, f"Generated text should not be empty for case {i + 1}"
            assert isinstance(output.finished, bool), f"'finished' should be boolean for case {i + 1}"
            assert output.metrics.model_execute_time > 0, f"Execution time should be positive for case {i + 1}"

            print(f"=== Prompt generation Case {i + 1} Passed ===")

    except Exception:
        print("Failed during prompt generation.")
        traceback.print_exc()
        pytest.fail("Prompt generation test failed")


def test_chat_completion(llm):
    """
    Test chat completion with multiple turns
    """
    chat_cases = [
        [
            {"role": "user", "content": "你好，请介绍一下你自己。"},
        ],
        [
            {"role": "user", "content": "你知道地球到月球的距离是多少吗？"},
            {"role": "assistant", "content": "大约是38万公里左右。"},
            {"role": "user", "content": "那太阳到地球的距离是多少？"},
        ],
        [
            {"role": "user", "content": "请给我起一个中文名。"},
            {"role": "assistant", "content": "好的，你可以叫“星辰”。"},
            {"role": "user", "content": "再起一个。"},
            {"role": "assistant", "content": "那就叫”大海“吧。"},
            {"role": "user", "content": "再来三个。"},
        ],
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )

    for i, case in enumerate(chat_cases):
        prompt = format_chat_prompt(case)
        try:
            outputs = llm.generate(prompt, sampling_params)

            # Verify chat completion properties
            assert len(outputs) == 1, "Should return one output per prompt"
            assert isinstance(outputs[0].outputs.text, str), "Output text should be string"
            assert len(outputs[0].outputs.text) > 0, "Generated text should not be empty"
            assert outputs[0].metrics.model_execute_time > 0, "Execution time should be positive"

            print(f"=== Chat Case {i + 1} Passed ===")

        except Exception:
            print(f"[ERROR] Chat Case {i + 1} failed.")
            traceback.print_exc()
            pytest.fail(f"Chat case {i + 1} failed")


def test_generate_prompts_stream(llm):
    """
    Test basic prompt generation stream outputs
    """

    prompts = [
        "请介绍一下中国的四大发明。",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )

    try:
        outputs = llm.generate(prompts, sampling_params, stream=True)

        # Collect streaming output
        output = []
        for chunk in outputs:
            if chunk[0] is not None:
                output.append(chunk[0].outputs.text)
        assert len(output) > 0

    except Exception:
        print("Failed during prompt generation.")
        traceback.print_exc()
        pytest.fail("Prompt generation test failed")


def test_chat_completion_stream(llm):
    """
    Test chat completion stream outputs
    """
    chat_cases = [
        [
            {"role": "user", "content": "你好，请介绍一下你自己。"},
        ],
        [
            {"role": "user", "content": "你知道地球到月球的距离是多少吗？"},
            {"role": "assistant", "content": "大约是38万公里左右。"},
            {"role": "user", "content": "那太阳到地球的距离是多少？"},
        ],
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
    )

    try:
        outputs = llm.chat(chat_cases, sampling_params, stream=True)

        # Collect streaming output
        output = [[], []]
        for chunks in outputs:
            for req_idx, chunk in enumerate(chunks):
                if chunk is not None:
                    output[req_idx].append(chunk.outputs.text)
        assert len(output[0]) > 0
        assert len(output[1]) > 0

    except Exception:
        print("Failed during prompt chat.")
        traceback.print_exc()
        pytest.fail("Prompt chat test failed")


def test_seed(llm):
    """
    Test chat completion with same seed
    """
    prompt = "请介绍下中国的四大发明，用一句话概述每个发明。"
    sampling_params = SamplingParams(temperature=0.1, seed=1, max_tokens=100)
    num_runs = 5

    results = []
    try:
        for i in range(num_runs):
            outputs = llm.generate(prompt, sampling_params)
            results.append(outputs[0].outputs.text)

        assert all([result == results[0] for result in results]), "Results are not identical."
        print("All results are identical.")

    except Exception:
        print("Failed during prompt generation.")
        traceback.print_exc()
        pytest.fail("Prompt generation test failed")


def test_logits_processors(llm):
    """
    Test LogitBiasLogitsProcessor: token with extremely large logit bias should always be greedy-sampled
    """
    messages = [{"role": "user", "content": "鲁迅是谁"}]
    sampling_params = SamplingParams(
        top_p=0.0,
        max_tokens=128,
    )
    outputs = llm.chat(messages, sampling_params)
    print("generated text:", outputs[0].outputs.text)
    original_generated_text = outputs[0].outputs.text

    # test request with logit bias
    token_id_with_exlarge_bias = 123
    messages = [{"role": "user", "content": "鲁迅是谁"}]
    sampling_params = SamplingParams(
        top_p=0.0,
        max_tokens=128,
        logits_processors_args={"logit_bias": {token_id_with_exlarge_bias: 100000}},
    )
    outputs = llm.chat(messages, sampling_params)
    print("generated text:", outputs[0].outputs.text)
    print("generated token ids:", outputs[0].outputs.token_ids)
    print("expected token id:", token_id_with_exlarge_bias)
    assert all(x == token_id_with_exlarge_bias for x in outputs[0].outputs.token_ids[:-1])

    # test request without logit bias
    messages = [{"role": "user", "content": "鲁迅是谁"}]
    sampling_params = SamplingParams(
        top_p=0.0,
        max_tokens=128,
    )
    outputs = llm.chat(messages, sampling_params)
    print("generated text:", outputs[0].outputs.text)
    current_generated_text = outputs[0].outputs.text
    assert current_generated_text == original_generated_text


if __name__ == "__main__":
    """
    Main entry point for the test script.
    """
    pytest.main(["-sv", __file__])
