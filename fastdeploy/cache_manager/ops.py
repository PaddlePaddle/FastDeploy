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

import paddle

from fastdeploy.platforms import current_platform

try:
    if current_platform.is_cuda():
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
            unset_data_ipc,
        )

        memory_allocated = paddle.device.cuda.memory_allocated

        def get_peer_mem_addr(*args, **kwargs):
            raise RuntimeError("CUDA no need of get_peer_mem_addr!")

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
        memory_allocated = paddle.device.xpu.memory_allocated

        def get_data_ptr_ipc(*args, **kwargs):
            raise RuntimeError("XPU get_data_ptr_ipc UNIMPLENENTED!")

        def ipc_sent_key_value_cache_by_remote_ptr(*args, **kwargs):
            raise RuntimeError("XPU ipc_sent_key_value_cache_by_remote_ptr UNIMPLENENTED")

        def ipc_sent_key_value_cache_by_remote_ptr_block_sync(*args, **kwargs):
            raise RuntimeError("XPU No ipc_sent_key_value_cache_by_remote_ptr UNIMPLENENTED")

    else:
        raise RuntimeError("Prefix cache ops only supported CUDA nor XPU platform ")

    def set_device(device):
        if current_platform.is_cuda():
            paddle.set_device(f"gpu:{device}")
        elif current_platform.is_xpu():
            paddle.set_device(f"xpu:{device}")
        else:
            raise RuntimeError("No supported platform")

    def share_external_data_(cache, cache_name, cache_shape, use_ipc):
        if current_platform.is_cuda():
            cache = share_external_data(cache, cache_name, cache_shape)
        elif current_platform.is_xpu():
            cache = share_external_data(cache, cache_name, cache_shape, use_ipc)
        else:
            raise RuntimeError("No supported platform")
        return cache

    def get_all_visible_devices():
        if current_platform.is_xpu():
            return "XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
        else:
            return "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

except:
    cuda_host_alloc = None
    cuda_host_free = None
    set_data_ipc = None
    share_external_data_ = None
    swap_cache_all_layers = None
    unset_data_ipc = None
    set_device = None
    memory_allocated = None
    get_output_kv_signal = None
    get_data_ptr_ipc = None
    ipc_sent_key_value_cache_by_remote_ptr = None
    ipc_sent_key_value_cache_by_remote_ptr_block_sync = None
    get_peer_mem_addr = None
    get_all_visible_devices = None


__all__ = [
    "cuda_host_alloc",
    "cuda_host_free",
    "set_data_ipc",
    "share_external_data_",
    "swap_cache_all_layers",
    "unset_data_ipc",  # XPU是 None
    "set_device",
    "memory_allocated",
    "get_output_kv_signal",
    "get_data_ptr_ipc",
    "ipc_sent_key_value_cache_by_remote_ptr",
    "ipc_sent_key_value_cache_by_remote_ptr_block_sync",
    "get_peer_mem_addr",
    "get_all_visible_devices",
]
