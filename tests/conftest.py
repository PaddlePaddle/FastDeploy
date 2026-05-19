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

import glob
import os
import re
import time
from typing import Any, Union

import pytest
from e2e.utils.serving_utils import (  # noqa: E402
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    clean_ports,
)


def pytest_configure(config):
    """
    Configure pytest:
    - Register custom markers
    - Ensure log directory exists
    """
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU platform")

    log_dir = os.environ.get("FD_LOG_DIR", "log")
    os.makedirs(log_dir, exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """
    Skip tests marked with 'gpu' if no GPU device is detected.

    IMPORTANT:
    Do NOT import paddle or fastdeploy here.
    This hook runs during test collection (before process fork).
    Importing CUDA-related libraries will initialize CUDA runtime,
    causing forked subprocesses to fail with:
    OSError: CUDA error(3), initialization error.
    """
    has_gpu = len(glob.glob("/dev/nvidia[0-9]*")) > 0

    if has_gpu:
        return

    skip_marker = pytest.mark.skip(reason="Test requires GPU platform, skipping on non-GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)


class FDRunner:
    """
    Wrapper for FastDeploy LLM serving process.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 1,
        max_model_len: int = 1024,
        load_choices: str = "default",
        quantization: str = "None",
        **kwargs,
    ) -> None:
        from fastdeploy.entrypoints.llm import LLM

        clean_ports()
        time.sleep(10)
        graph_optimization_config = {"use_cudagraph": False}
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            load_choices=load_choices,
            quantization=quantization,
            max_num_batched_tokens=max_model_len,
            graph_optimization_config=graph_optimization_config,
            port=FD_API_PORT,
            cache_queue_port=FD_CACHE_QUEUE_PORT,
            engine_worker_queue_port=FD_ENGINE_QUEUE_PORT,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:
        """
        Run generation and return token IDs and generated texts.
        """
        req_outputs = self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)
        outputs: list[tuple[list[list[int]], list[str]]] = []
        for output in req_outputs:
            outputs.append((output.outputs.token_ids, output.outputs.text))
        return outputs

    def generate_topp0(
        self,
        prompts: Union[list[str]],
        max_tokens: int,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        """
        Generate outputs with deterministic sampling (top_p=0, temperature=0).
        """
        from fastdeploy.engine.sampling_params import SamplingParams

        topp_params = SamplingParams(temperature=0.0, top_p=0, max_tokens=max_tokens)
        outputs = self.generate(prompts, topp_params, **kwargs)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.llm


@pytest.fixture(scope="session")
def fd_runner():
    """Provide FDRunner as a pytest fixture."""
    return FDRunner


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Capture failed test cases and save error logs to FD_LOG_DIR.

    Only logs failures during the test execution phase.
    """
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        log_dir = os.environ.get("FD_LOG_DIR", "log")
        os.makedirs(log_dir, exist_ok=True)

        case_name = re.sub(r"_+", "_", re.sub(r"[^\w\-.]", "_", item.nodeid.split("::", 1)[-1])).strip("_")[:200]

        error_log_file = os.path.join(log_dir, f"pytest_{case_name}_error.log")

        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write(f"Case name: {item.nodeid}\n")
            f.write(f"Outcome: {report.outcome}\n")
            f.write(f"Duration: {report.duration:.4f}s\n")
            f.write("-" * 80 + "\n")

            if report.longrepr:
                f.write(str(report.longrepr))
