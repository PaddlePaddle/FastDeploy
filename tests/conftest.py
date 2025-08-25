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
from typing import Any, Union

import pytest


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses `lsof` to find process ids and sends SIGKILL.
    """
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        pass


def clean_ports(ports_to_clean: list[int]):
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    for port in ports_to_clean:
        kill_process_on_port(port)


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


class FDRunner:
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 1024,
        load_choices: str = "default",
        quantization: str = "None",
        **kwargs,
    ) -> None:
        from fastdeploy.entrypoints.llm import LLM

        ports_to_clean = []
        if "engine_worker_queue_port" in kwargs:
            ports_to_clean.append(kwargs["engine_worker_queue_port"])
        clean_ports(ports_to_clean)
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            load_choices=load_choices,
            quantization=quantization,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:

        req_outputs = self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)
        outputs: list[tuple[list[list[int]], list[str]]] = []
        sample_output_ids: list[list[int]] = []
        sample_output_strs: list[str] = []
        for output in req_outputs:
            sample_output_ids.append(output.outputs.token_ids)
            sample_output_strs.append(output.outputs.text)
            outputs.append((sample_output_ids, sample_output_strs))
        return outputs

    def generate_topp0(
        self,
        prompts: Union[list[str]],
        max_tokens: int,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        from fastdeploy.engine.sampling_params import SamplingParams

        topp_params = SamplingParams(temperature=0.1, top_p=0, max_tokens=max_tokens)
        outputs = self.generate(prompts, topp_params, **kwargs)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.llm


@pytest.fixture(scope="session")
def fd_runner():
    return FDRunner
