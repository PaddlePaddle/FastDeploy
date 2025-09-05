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
MAX_WAIT_SECONDS = 60

os.environ["LD_LIBRARY_PATH"] = "/usr/local/nccl/"
# enbale get_save_output_v1
os.environ["FD_USE_GET_SAVE_OUTPUT_V1"] = "1"


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


@pytest.fixture(scope="module")
def model_path():
    """
    Get model path from environment variable MODEL_PATH,
    default to "./ERNIE-4.5-0.3B-Paddle" if not set.
    """
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        return os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")
    else:
        return "./ERNIE-4.5-0.3B-Paddle"


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
            tensor_parallel_size=2,
            num_gpu_blocks_override=1024,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
            max_model_len=8192,
            seed=1,
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


if __name__ == "__main__":
    """
    Main entry point for the test script.
    """
    pytest.main(["-sv", __file__])
