# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import pytest
import traceback
from fastdeploy import LLM, SamplingParams
import os
import subprocess
import signal

FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8313))

def format_chat_prompt(messages):
    """
    Format multi-turn conversation into prompt string, suitable for chat models.
    Uses Qwen2 style with <|im_start|> / <|im_end|> tokens.
    """
    prompt = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "user":
            prompt += "<|im_start|>user\n{content}<|im_end|>\n".format(content=content)
        elif role == "assistant":
            prompt += "<|im_start|>assistant\n{content}<|im_end|>\n".format(content=content)
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
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
            max_model_len=4096
        )
        print("Model loaded successfully from {}.".format(model_path))
        yield llm
    except Exception:
        print("Failed to load model from {}.".format(model_path))
        traceback.print_exc()
        pytest.fail("Failed to initialize LLM model from {}".format(model_path))


def test_generate_prompts(llm):
    """
    Test basic prompt generation
    """
    # Only one prompt enabled for testing currently
    prompts = [
        "请介绍一下中国的四大发明。",
        # "太阳和地球之间的距离是多少？",
        # "写一首关于春天的古风诗。",
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
            assert output.prompt == prompts[i], "Prompt mismatch for case {}".format(i + 1)
            assert isinstance(output.outputs.text, str), "Output text should be string for case {}".format(i + 1)
            assert len(output.outputs.text) > 0, "Generated text should not be empty for case {}".format(i + 1)
            assert isinstance(output.finished, bool), "'finished' should be boolean for case {}".format(i + 1)
            assert output.metrics.model_execute_time > 0, "Execution time should be positive for case {}".format(i + 1)

            print("=== Prompt generation Case {} Passed ===".format(i + 1))

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

            print("=== Chat Case {} Passed ===".format(i + 1))

        except Exception:
            print("[ERROR] Chat Case {} failed.".format(i + 1))
            traceback.print_exc()
            pytest.fail("Chat case {} failed".format(i + 1))


if __name__ == "__main__":
    """
    Main entry point for the test script.
    """
    pytest.main(["-sv", __file__])