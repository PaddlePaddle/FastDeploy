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

import paddle

from fastdeploy.platforms import current_platform
from fastdeploy.utils import llm_logger as logger

try:
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import (
            swap_cache_per_layer,  # 单层 KV cache 换入算子（同步）
        )
        from fastdeploy.model_executor.ops.gpu import (
            swap_cache_per_layer_async,  # 单层 KV cache 换入算子（异步，无强制 sync）
        )
        from fastdeploy.model_executor.ops.gpu import (
            cuda_host_alloc,
            cuda_host_free,
            get_data_ptr_ipc,
            get_output_kv_signal,
            ipc_sent_key_value_cache_by_remote_ptr,
            ipc_sent_key_value_cache_by_remote_ptr_block_sync,
            set_data_ipc,
            share_external_data,
            swap_cache_all_layers,
            swap_cache_layout,
            unset_data_ipc,
        )

        memory_allocated = paddle.device.cuda.memory_allocated

        def get_peer_mem_addr(*args, **kwargs):
            raise RuntimeError("CUDA no need of get_peer_mem_addr!")

    elif current_platform.is_maca():
        from fastdeploy.model_executor.ops.gpu import (  # get_output_kv_signal,; ipc_sent_key_value_cache_by_remote_ptr_block_sync,
            cuda_host_alloc,
            cuda_host_free,
            get_data_ptr_ipc,
            ipc_sent_key_value_cache_by_remote_ptr,
            set_data_ipc,
            share_external_data,
            swap_cache_all_layers,
            unset_data_ipc,
        )

        memory_allocated = paddle.device.memory_allocated

        def get_peer_mem_addr(*args, **kwargs):
            raise RuntimeError("CUDA no need of get_peer_mem_addr!")

        def get_output_kv_signal(*args, **kwargs):
            raise RuntimeError("Metax get_output_kv_signal UNIMPLEMENTED!")

        def ipc_sent_key_value_cache_by_remote_ptr_block_sync(*args, **kwargs):
            raise RuntimeError("Metax ipc_sent_key_value_cache_by_remote_ptr_block_sync UNIMPLEMENTED!")

        def swap_cache_per_layer(*args, **kwargs):  # 单层 KV cache 换入算子（同步）
            raise RuntimeError("Metax swap_cache_per_layer UNIMPLEMENTED")

        def swap_cache_per_layer_async(*args, **kwargs):  # 单层 KV cache 换入算子（异步）
            raise RuntimeError("Metax swap_cache_per_layer_async UNIMPLEMENTED")

        def swap_cache_layout(*args, **kwargs):
            raise RuntimeError("Metax swap_cache_layout UNIMPLEMENTED")

    elif current_platform.is_xpu():
        from fastdeploy.model_executor.ops.xpu import (
            cuda_host_alloc,
            cuda_host_free,
            get_output_kv_signal,
            get_peer_mem_addr,
            set_data_ipc,
            share_external_data,
            swap_cache_all_layers,
        )

        unset_data_ipc = None
        swap_cache_layout = None
        memory_allocated = paddle.device.xpu.memory_allocated

        def get_data_ptr_ipc(*args, **kwargs):
            raise RuntimeError("XPU get_data_ptr_ipc UNIMPLEMENTED!")

        def ipc_sent_key_value_cache_by_remote_ptr(*args, **kwargs):
            raise RuntimeError("XPU ipc_sent_key_value_cache_by_remote_ptr UNIMPLEMENTED")

        def ipc_sent_key_value_cache_by_remote_ptr_block_sync(*args, **kwargs):
            raise RuntimeError("XPU No ipc_sent_key_value_cache_by_remote_ptr UNIMPLEMENTED")

        def swap_cache_per_layer(*args, **kwargs):  # 单层 KV cache 换入算子（同步）
            raise RuntimeError("XPU swap_cache_per_layer UNIMPLEMENTED")

        def swap_cache_per_layer_async(*args, **kwargs):  # 单层 KV cache 换入算子（异步）
            raise RuntimeError("XPU swap_cache_per_layer_async UNIMPLEMENTED")

    else:
        raise RuntimeError("Prefix cache ops only supported CUDA nor XPU platform ")

    def set_device(device):
        if current_platform.is_cuda():
            paddle.set_device(f"gpu:{device}")
        elif current_platform.is_maca():
            paddle.set_device(f"metax_gpu:{device}")
        elif current_platform.is_xpu():
            paddle.set_device(f"xpu:{device}")
        else:
            raise RuntimeError("No supported platform")

    def share_external_data_(cache, cache_name, cache_shape, use_ipc):
        if current_platform.is_cuda():
            cache = share_external_data(cache, cache_name, cache_shape)
        elif current_platform.is_maca():
            cache = share_external_data(cache, cache_name, cache_shape)
        elif current_platform.is_xpu():
            cache = share_external_data(cache, cache_name, cache_shape, use_ipc)
        else:
            raise RuntimeError("No supported platform")
        return cache

    def get_all_visible_devices():
        if current_platform.is_xpu():
            return "XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
        elif current_platform.is_maca():
            return f'MACA_VISIBLE_DEVICES={os.environ.get("MACA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")}'
        else:
            return "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

except Exception as e:
    logger.warning(f"Failed to import cache manager ops: {e}")
    cuda_host_alloc = None
    cuda_host_free = None
    set_data_ipc = None
    share_external_data_ = None
    swap_cache_all_layers = None
    swap_cache_per_layer = None  # 单层 KV cache 换入算子（同步）
    swap_cache_per_layer_async = None  # 单层 KV cache 换入算子（异步）
    unset_data_ipc = None
    set_device = None
    memory_allocated = None
    get_output_kv_signal = None
    get_data_ptr_ipc = None
    ipc_sent_key_value_cache_by_remote_ptr = None
    ipc_sent_key_value_cache_by_remote_ptr_block_sync = None
    get_peer_mem_addr = None
    get_all_visible_devices = None
    swap_cache_layout = None


__all__ = [
    "cuda_host_alloc",
    "cuda_host_free",
    "set_data_ipc",
    "share_external_data_",
    "swap_cache_all_layers",
    "swap_cache_per_layer",  # 单层 KV cache 换入算子（同步）
    "swap_cache_per_layer_async",  # 单层 KV cache 换入算子（异步，无强制 sync）
    "unset_data_ipc",  # XPU是 None
    "set_device",
    "memory_allocated",
    "get_output_kv_signal",
    "get_data_ptr_ipc",
    "ipc_sent_key_value_cache_by_remote_ptr",
    "ipc_sent_key_value_cache_by_remote_ptr_block_sync",
    "get_peer_mem_addr",
    "get_all_visible_devices",
    "swap_cache_layout",
]
