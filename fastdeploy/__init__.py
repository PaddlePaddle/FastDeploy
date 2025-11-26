"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import os
import subprocess
import sys
import uuid

# suppress warning log from paddlepaddle
os.environ["GLOG_minloglevel"] = "2"
# suppress log from aistudio
os.environ["AISTUDIO_LOG"] = "critical"
# set prometheus dir
if os.getenv("PROMETHEUS_MULTIPROC_DIR", "") == "":
    prom_dir = f"/tmp/fd_prom_{str(uuid.uuid4())}"
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
    if os.path.exists(prom_dir):
        os.rmdir(prom_dir)
    os.mkdir(prom_dir)

import typing

# first import prometheus setup to set PROMETHEUS_MULTIPROC_DIR
# otherwise, the Prometheus package will be imported first,
# which will prevent correct multi-process setup
from fastdeploy.metrics.prometheus_multiprocess_setup import (
    setup_multiprocess_prometheus,
)

setup_multiprocess_prometheus()

from paddleformers.utils.log import logger as pf_logger

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.utils import current_package_version, envs

if envs.FD_DEBUG != 1:
    import logging

    pf_logger.logger.setLevel(logging.INFO)

try:
    import use_triton_in_paddle

    use_triton_in_paddle.make_triton_compatible_with_paddle()
except ImportError:
    pass
# TODO(tangbinhan): remove this code

__version__ = current_package_version()


def _patch_fastsafetensors():
    try:
        file_path = (
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import fastsafetensors, os; \
             print(os.path.join(os.path.dirname(fastsafetensors.__file__), \
             'frameworks', '_paddle.py'))",
                ]
            )
            .decode()
            .strip()
        )

        with open(file_path, "r") as f:
            content = f.read()
        if "DType.U16: DType.BF16," in content and "DType.U8: paddle.uint8," in content:
            return

        modified = False
        if "DType.U16: DType.BF16," not in content:
            lines = content.splitlines()
            new_lines = []
            inside_block = False
            for line in lines:
                new_lines.append(line)
                if "need_workaround_dtypes: Dict[DType, DType] = {" in line:
                    inside_block = True
                elif inside_block and "}" in line:
                    new_lines.insert(-1, "    DType.U16: DType.BF16,")
                    inside_block = False
                    modified = True
            content = "\n".join(new_lines)

        if "DType.I8: paddle.uint8," in content:
            content = content.replace("DType.I8: paddle.uint8,", "DType.U8: paddle.uint8,")
            modified = True

        if modified:
            with open(file_path, "w") as f:
                f.write(content + "\n")

    except Exception as e:
        print(f"Failed to patch fastsafetensors: {e}")


_patch_fastsafetensors()


MODULE_ATTRS = {"ModelRegistry": ".model_executor.models.model_base:ModelRegistry", "version": ".utils:version"}


if typing.TYPE_CHECKING:
    from fastdeploy.model_executor.models.model_base import ModelRegistry
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            try:
                module_name, attr_name = MODULE_ATTRS[name].split(":")
                module = import_module(module_name, __package__)
                return getattr(module, attr_name)
            except ModuleNotFoundError:
                print(f"Module {MODULE_ATTRS[name]} not found.")
        else:
            print(f"module {__package__} has no attribute {name}")


__all__ = ["LLM", "SamplingParams", "ModelRegistry", "version"]
