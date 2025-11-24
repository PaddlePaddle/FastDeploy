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
    "FD_CPU_USE_BF16": lambda: os.getenv("FD_CPU_USE_BF16", "False"),
    # Cuda architecture to build FastDeploy.This is a list of strings
    # such as [80,90].
    "FD_BUILDING_ARCS": lambda: os.getenv("FD_BUILDING_ARCS", "[]"),
    # Log directory.
    "FD_LOG_DIR": lambda: os.getenv("FD_LOG_DIR", "log"),
    # Whether to use debug mode, can set 0 or 1
    "FD_DEBUG": lambda: int(os.getenv("FD_DEBUG", "0")),
    # Number of days to keep fastdeploy logs.
    "FD_LOG_BACKUP_COUNT": lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),
    # Model download source, can set "AISTUDIO", "MODELSCOPE" or "HUGGINGFACE".
    "FD_MODEL_SOURCE": lambda: os.getenv("FD_MODEL_SOURCE", "AISTUDIO"),
    # Model download cache directory.
    "FD_MODEL_CACHE": lambda: os.getenv("FD_MODEL_CACHE", None),
    # Maximum number of stop sequences.
    "FD_MAX_STOP_SEQS_NUM": lambda: int(os.getenv("FD_MAX_STOP_SEQS_NUM", "5")),
    # Maximum length of stop sequences.
    "FD_STOP_SEQS_MAX_LEN": lambda: int(os.getenv("FD_STOP_SEQS_MAX_LEN", "8")),
    # GPU devices that will be used. This is a string that
    # splited by comma, such as 0,1,2.
    "CUDA_VISIBLE_DEVICES": lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),
    # Whether to use HuggingFace tokenizer.
    "FD_USE_HF_TOKENIZER": lambda: bool(int(os.getenv("FD_USE_HF_TOKENIZER", "0"))),
    # Set the high watermark (HWM) for receiving data during ZMQ initialization
    "FD_ZMQ_SNDHWM": lambda: os.getenv("FD_ZMQ_SNDHWM", 0),
    # cache kv quant params directory
    "FD_CACHE_PARAMS": lambda: os.getenv("FD_CACHE_PARAMS", "none"),
    # Set attention backend. "NATIVE_ATTN", "APPEND_ATTN"
    # and "MLA_ATTN" can be set currently.
    "FD_ATTENTION_BACKEND": lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),
    # Set sampling class. "base", "base_non_truncated", "air" and "rejection" can be set currently.
    "FD_SAMPLING_CLASS": lambda: os.getenv("FD_SAMPLING_CLASS", "base"),
    # Set moe backend."cutlass","marlin" and "triton" can be set currently.
    "FD_MOE_BACKEND": lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),
    # Whether to use Machete for wint4 dense gemm.
    "FD_USE_MACHETE": lambda: os.getenv("FD_USE_MACHETE", "1"),
    # Set whether to disable recompute the request when the KV cache is full.
    "FD_DISABLED_RECOVER": lambda: os.getenv("FD_DISABLED_RECOVER", "0"),
    # Set triton kernel JIT compilation directory.
    "FD_TRITON_KERNEL_CACHE_DIR": lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),
    # Whether transition from standalone PD decoupling to centralized inference
    "FD_PD_CHANGEABLE": lambda: os.getenv("FD_PD_CHANGEABLE", "0"),
    # Whether to use fastsafetensor load weight (0 or 1)
    "FD_USE_FASTSAFETENSOR": lambda: bool(int(os.getenv("FD_USE_FASTSAFETENSOR", "0"))),
    # Whether to use DeepGemm for FP8 blockwise MoE.
    "FD_USE_DEEP_GEMM": lambda: bool(int(os.getenv("FD_USE_DEEP_GEMM", "0"))),
    # Whether to use aggregate send.
    "FD_USE_AGGREGATE_SEND": lambda: bool(int(os.getenv("FD_USE_AGGREGATE_SEND", "0"))),
    # Whether to open Trace.
    "TRACES_ENABLE": lambda: os.getenv("TRACES_ENABLE", "false"),
    # set traec Server name.
    "FD_SERVICE_NAME": lambda: os.getenv("FD_SERVICE_NAME", "FastDeploy"),
    # set traec host name.
    "FD_HOST_NAME": lambda: os.getenv("FD_HOST_NAME", "localhost"),
    # set traec exporter.
    "TRACES_EXPORTER": lambda: os.getenv("TRACES_EXPORTER", "console"),
    # set traec exporter_otlp_endpoint.
    "EXPORTER_OTLP_ENDPOINT": lambda: os.getenv("EXPORTER_OTLP_ENDPOINT"),
    # set traec exporter_otlp_headers.
    "EXPORTER_OTLP_HEADERS": lambda: os.getenv("EXPORTER_OTLP_HEADERS"),
    # enable kv cache block scheduler v1 (no need for kv_cache_ratio)
    "ENABLE_V1_KVCACHE_SCHEDULER": lambda: int(os.getenv("ENABLE_V1_KVCACHE_SCHEDULER", "1")),
    # set prealloc block num for decoder
    "FD_ENC_DEC_BLOCK_NUM": lambda: int(os.getenv("FD_ENC_DEC_BLOCK_NUM", "2")),
    # enbale max prefill of one execute step
    "FD_ENABLE_MAX_PREFILL": lambda: int(os.getenv("FD_ENABLE_MAX_PREFILL", "0")),
    # Whether to use PLUGINS.
    "FD_PLUGINS": lambda: None if "FD_PLUGINS" not in os.environ else os.environ["FD_PLUGINS"].split(","),
    # set trace attribute job_id.
    "FD_JOB_ID": lambda: os.getenv("FD_JOB_ID"),
    # support max connections
    "FD_SUPPORT_MAX_CONNECTIONS": lambda: int(os.getenv("FD_SUPPORT_MAX_CONNECTIONS", "1024")),
    # Offset for Tensor Parallelism group GID.
    "FD_TP_GROUP_GID_OFFSET": lambda: int(os.getenv("FD_TP_GROUP_GID_OFFSET", "1000")),
    # enable multi api server
    "FD_ENABLE_MULTI_API_SERVER": lambda: bool(int(os.getenv("FD_ENABLE_MULTI_API_SERVER", "0"))),
    "FD_FOR_TORCH_MODEL_FORMAT": lambda: bool(int(os.getenv("FD_FOR_TORCH_MODEL_FORMAT", "0"))),
    # force disable default chunked prefill
    "FD_DISABLE_CHUNKED_PREFILL": lambda: bool(int(os.getenv("FD_DISABLE_CHUNKED_PREFILL", "0"))),
    # Whether to use new get_output and save_output method (0 or 1)
    "FD_USE_GET_SAVE_OUTPUT_V1": lambda: bool(int(os.getenv("FD_USE_GET_SAVE_OUTPUT_V1", "0"))),
    # Whether to enable model cache feature
    "FD_ENABLE_MODEL_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_CACHE", "0"))),
    # enable internal module to access LLMEngine.
    "FD_ENABLE_INTERNAL_ADAPTER": lambda: int(os.getenv("FD_ENABLE_INTERNAL_ADAPTER", "0")),
    # LLMEngine receive requests port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORT": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORT", "8200"),
    # LLMEngine send response port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORT": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORT", "8201"),
    # LLMEngine receive requests port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "8200"),
    # LLMEngine send response port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "8201"),
    # LLMEngine receive control command port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_CONTROL_CMD_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_CONTROL_CMD_SERVER_PORTS", "8202"),
    # Whether to enable cache task in decode node
    "FD_ENABLE_CACHE_TASK": lambda: os.getenv("FD_ENABLE_CACHE_TASK", "1"),
    # Batched token timeout in EP
    "FD_EP_BATCHED_TOKEN_TIMEOUT": lambda: float(os.getenv("FD_EP_BATCHED_TOKEN_TIMEOUT", "0.1")),
    # Max pre-fetch requests number in PD
    "FD_EP_MAX_PREFETCH_TASK_NUM": lambda: int(os.getenv("FD_EP_MAX_PREFETCH_TASK_NUM", "8")),
    # Enable or disable model caching.
    # When enabled, the quantized model is stored as a cache for future inference to improve loading efficiency.
    "FD_ENABLE_MODEL_LOAD_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_LOAD_CACHE", "0"))),
    # Whether to clear cpu cache when clearing model weights.
    "FD_ENABLE_SWAP_SPACE_CLEARING": lambda: int(os.getenv("FD_ENABLE_SWAP_SPACE_CLEARING", "0")),
    # enable return text, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ENABLE_RETURN_TEXT": lambda: bool(int(os.getenv("FD_ENABLE_RETURN_TEXT", "0"))),
    # Used to truncate the string inserted during thinking when reasoning in a model. (</think> for ernie-45-vl, \n</think>\n\n for ernie-x1)
    "FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR": lambda: os.getenv("FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR", "</think>"),
    # Timeout for cache_transfer_manager process exit
    "FD_CACHE_PROC_EXIT_TIMEOUT": lambda: int(os.getenv("FD_CACHE_PROC_EXIT_TIMEOUT", "600")),
    # Count for cache_transfer_manager process error
    "FD_CACHE_PROC_ERROR_COUNT": lambda: int(os.getenv("FD_CACHE_PROC_ERROR_COUNT", "10")),
    # API_KEY required for service authentication
    "FD_API_KEY": lambda: [] if "FD_API_KEY" not in os.environ else os.environ["FD_API_KEY"].split(","),
    # The AK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_AK": lambda: os.getenv("ENCODE_FEATURE_BOS_AK"),
    # The SK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_SK": lambda: os.getenv("ENCODE_FEATURE_BOS_SK"),
    # The ENDPOINT of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_ENDPOINT": lambda: os.getenv("ENCODE_FEATURE_ENDPOINT"),
    # Enable offline perf test mode for PD disaggregation
    "FD_OFFLINE_PERF_TEST_FOR_PD": lambda: int(os.getenv("FD_OFFLINE_PERF_TEST_FOR_PD", "0")),
    "FD_ENABLE_E2W_TENSOR_CONVERT": lambda: int(os.getenv("FD_ENABLE_E2W_TENSOR_CONVERT", "0")),
    "FD_ENGINE_TASK_QUEUE_WITH_SHM": lambda: int(os.getenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")),
    "FD_FILL_BITMASK_BATCH": lambda: int(os.getenv("FD_FILL_BITMASK_BATCH", "4")),
    "FD_ENABLE_PDL": lambda: int(os.getenv("FD_ENABLE_PDL", "1")),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_unique_name(self, name):
    """
    Get unique name for config
    """
    shm_uuid = os.getenv("SHM_UUID", "")
    return name + f"_{shm_uuid}"


def __setattr__(name: str, value: Any):
    assert name in environment_variables
    environment_variables[name] = lambda: value


def __dir__():
    return list(environment_variables.keys())
