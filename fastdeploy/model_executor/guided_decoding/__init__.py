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

# from fastdeploy.config import FDConfig
from fastdeploy.model_executor.guided_decoding.base_guided_decoding import (
    BackendBase,
    BaseChecker,
    LogitsProcessorBase,
)

__all__ = ["get_guided_backend", "schema_checker", "LogitsProcessorBase", "BackendBase", "BaseChecker"]


def get_guided_backend(
    fd_config,
    **kwargs,
):
    """
    Get the guided decoding backend instance based on configuration.

    Args:
        fd_config (FDConfig): FastDeploy configuration object containing backend settings
        **kwargs: Additional arguments passed to the backend constructor

    Returns:
        BaseBackend: An instance of the specified guided decoding backend

    Raises:
        ValueError: If the specified backend is not supported
    """
    if fd_config.structured_outputs_config.guided_decoding_backend.lower() == "xgrammar":
        from fastdeploy.model_executor.guided_decoding.xgrammar_backend import (
            XGrammarBackend,
        )

        return XGrammarBackend(
            fd_config=fd_config,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Get unsupported backend {fd_config.structured_outputs_config.guided_decoding_backend},"
            f" please check your configuration."
        )


def schema_checker(backend_name: str, **kwargs):
    """
    Get the schema checker instance for the specified backend.

    Args:
        backend_name (str): Name of the backend (e.g. "xgrammar")
        **kwargs: Additional arguments passed to the checker constructor

    Returns:
        BaseChecker: An instance of the specified schema checker

    Raises:
        ValueError: If the specified backend is not supported
    """
    if backend_name.lower() == "xgrammar":
        from fastdeploy.model_executor.guided_decoding.xgrammar_backend import (
            XGrammarChecker,
        )

        return XGrammarChecker(**kwargs)
    else:
        raise ValueError(f"Get unsupported backend {backend_name}, please check your configuration.")
