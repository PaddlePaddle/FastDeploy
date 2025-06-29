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
"""
Environment variables used by FastDeploy.
"""

import os
from typing import Any, Callable

environment_variables: dict[str, Callable[[], Any]] = {
    # Whether to use BF16 on CPU.
    "FD_CPU_USE_BF16":
    lambda: os.getenv("FD_CPU_USE_BF16", "False"),

    # Cuda architecture to build FastDeploy.This is a list of strings
    # such as [80,90].
    "FD_BUILDING_ARCS":
    lambda: os.getenv("FD_BUILDING_ARCS", "[]"),

    # Log directory.
    "FD_LOG_DIR":
    lambda: os.getenv("FD_LOG_DIR", "log"),

    # Whether to use debug mode, can set 0 or 1
    "FD_DEBUG":
    lambda: os.getenv("FD_DEBUG", "0"),

    # Number of days to keep fastdeploy logs.
    "FD_LOG_BACKUP_COUNT":
    lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),

    # Model download cache directory.
    "FD_MODEL_CACHE":
    lambda: os.getenv("FD_MODEL_CACHE", None),

    # Maximum number of stop sequences.
    "FD_MAX_STOP_SEQS_NUM":
    lambda: os.getenv("FD_MAX_STOP_SEQS_NUM", "5"),

    # Maximum length of stop sequences.
    "FD_STOP_SEQS_MAX_LEN":
    lambda: os.getenv("FD_STOP_SEQS_MAX_LEN", "8"),

    # GPU devices that will be used. This is a string that
    # splited by comma, such as 0,1,2.
    "CUDA_VISIBLE_DEVICES":
    lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),

    # Whether to use HuggingFace tokenizer.
    "FD_USE_HF_TOKENIZER":
    lambda: os.getenv("FD_USE_HF_TOKENIZER", 0),

    # Set the high watermark (HWM) for receiving data during ZMQ initialization
    "FD_ZMQ_SNDHWM":
    lambda: os.getenv("FD_ZMQ_SNDHWM", 10000),

    # cache kv quant params directory
    "FD_CACHE_PARAMS":
    lambda: os.getenv("FD_CACHE_PARAMS", "none"),

    # Set attention backend. "NATIVE_ATTN", "APPEND_ATTN"
    # and "MLA_ATTN" can be set currently.
    "FD_ATTENTION_BACKEND":
    lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),

    # Set sampling class. "base", "air" and "rejection" can be set currently.
    "FD_SAMPLING_CLASS":
    lambda: os.getenv("FD_SAMPLING_CLASS", "base"),

    # Set moe backend."cutlass","marlin" and "triton" can be set currently.
    "FD_MOE_BACKEND":
    lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),

    # Set triton kernel JIT compilation directory.
    "FD_TRITON_KERNEL_CACHE_DIR":
    lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),

    # Whether transition from standalone PD decoupling to centralized inference
    "FD_PD_CHANGEABLE":
    lambda: os.getenv("FD_PD_CHANGEABLE", "1"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
