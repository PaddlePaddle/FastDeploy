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

# Configure root logger first to unify log formats
# This must be done before importing any modules that may use the logger
import logging
import os
from contextlib import contextmanager

# Create standard format (without color)
_root_formatter = logging.Formatter(
    "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
)

# Save original getLogger before any patching
_original_getLogger = logging.getLogger


@contextmanager
def _intercept_paddle_loggers():
    """Intercept and configure paddle loggers during import."""

    def _patched(name=None):
        if name and str(name).startswith("paddle"):
            # Configure paddle logger immediately on first access
            return _configure_logger(name)
        return _original_getLogger(name)

    logging.getLogger = _patched
    try:
        yield
    finally:
        logging.getLogger = _original_getLogger


def _configure_logger(name=None):
    """Configure logger with unified format.

    Args:
        name: Logger name. If None, configures root logger.
    """
    # Use original getLogger to avoid recursion when interceptor is active
    logger = _original_getLogger(name)
    logger.setLevel(logging.DEBUG if envs.FD_DEBUG else logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(_root_formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


from fastdeploy.utils import _is_package_installed, envs

# Configure root logger
_configure_logger()

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

with _intercept_paddle_loggers():
    import paddle

# first import prometheus setup to set PROMETHEUS_MULTIPROC_DIR
# otherwise, the Prometheus package will be imported first,
# which will prevent correct multi-process setup
from fastdeploy.metrics.prometheus_multiprocess_setup import (
    setup_multiprocess_prometheus,
)

setup_multiprocess_prometheus()


from paddleformers.utils.log import logger as pf_logger

# Configure paddleformers loggers with unified format
_configure_logger("paddleformers")

# Also configure pf_logger.logger (if it is a Logger object)
if hasattr(pf_logger, "logger") and isinstance(pf_logger.logger, logging.Logger):
    _configure_logger(pf_logger.logger.name)

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.utils import console_logger, current_package_version, get_version_info

# We can use enable_compat only when torch is not installed, otherwise it will
# cause some unexpected issues in triton kernels. We use enable_compat_on_triton_kernel
# for these cases.
if not _is_package_installed("torch"):
    paddle.enable_compat(scope={"triton"})

if envs.FD_DEBUG != 1:
    # Log level has been configured above
    pass

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
