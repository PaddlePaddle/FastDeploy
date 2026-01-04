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

import paddle

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
from fastdeploy.utils import console_logger, current_package_version, envs, get_version_info

paddle.compat.enable_torch_proxy(scope={"triton"})
# paddle.compat.enable_torch_proxy(scope={"triton"}) enables the torch proxy
# specifically for the 'triton' module. This means `import torch` inside 'triton'
# will actually import paddle's compatibility layer (acting as torch).
#
# 'scope' acts as an allowlist. To add other modules, you can do:
# paddle.compat.enable_torch_proxy(scope={"triton", "new_module"})
#
# Note: Ensure that any torch APIs used in 'new_module' are already implemented in Paddle.


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

# Version check mechanism: Check if the Paddle version used at runtime matches the one used during FastDeploy compilation
try:
    version_info = get_version_info()
    if version_info is not None and "paddle_commit" in version_info:
        build_paddle_commit = version_info["paddle_commit"]
        runtime_paddle_commit = paddle.version.commit
        
        if build_paddle_commit != runtime_paddle_commit:
            console_logger.warning(
                f"The Paddle version in the current runtime environment is inconsistent with the Paddle code version "
                f"used during FastDeploy compilation. This may cause errors. "
                f"It is recommended to install the corresponding Paddle version.\n"
                f"  Build-time Paddle commit: {build_paddle_commit}\n"
                f"  Runtime Paddle commit: {runtime_paddle_commit}"
            )
except Exception as e:
    # Version check failure should not affect FastDeploy's normal operation
    console_logger.debug(f"Version check failed: {e}")


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
