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

from fastdeploy.cache_manager.ops import (
    get_data_ptr_ipc,
    ipc_sent_key_value_cache_by_remote_ptr,
    ipc_sent_key_value_cache_by_remote_ptr_block_sync,
)
from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


class IPCConnector:
    """
    IPC communication class.
    """

    def __init__(self, rank_id_, remote_gpu_id_, layer_num, local_gpu_id_, cache_dtype):
        """
        Args:
        rank_id_: rank id
        remote_gpu_id_: remote gpu id

        """
        self.remote_key_tensor_ptr_list = []
        self.remote_value_tensor_ptr_list = []
        self.remote_key_scale_tensor_ptr_list = []
        self.remote_value_scale_tensor_ptr_list = []
        self.remote_gpu_id = int(remote_gpu_id_)
        self.rank_id = rank_id_
        self.local_gpu_id = int(local_gpu_id_)
        self.cache_dtype = cache_dtype
        tmp = paddle.ones([1, 1])
        logger.info(
            f"init ipc rank{self.rank_id} with remote {self.remote_gpu_id} {self.local_gpu_id}, cache dtype {self.cache_dtype}"
        )
        for layer_id in range(layer_num):
            key_unique_name = f"key_caches_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
            value_unique_name = f"value_caches_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
            self.remote_key_tensor_ptr_list.append(get_data_ptr_ipc(tmp, key_unique_name))
            self.remote_value_tensor_ptr_list.append(get_data_ptr_ipc(tmp, value_unique_name))
            if self.cache_dtype == "block_wise_fp8":
                key_scale_name = f"key_cache_scales_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
                val_scale_name = f"value_cache_scales_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
                self.remote_key_scale_tensor_ptr_list.append(get_data_ptr_ipc(tmp, key_scale_name))
                self.remote_value_scale_tensor_ptr_list.append(get_data_ptr_ipc(tmp, val_scale_name))
        self.write_stream = paddle.device.Stream(f"gpu:{self.local_gpu_id}")


class IPCCommManager:
    """
    IPC communication manager, used to initialize ipc and cache transmission.
    """

    def __init__(
        self,
        rank_id_,
        gpu_idx_,
        local_key_cache_tensor_list,  # tensor list
        local_value_cache_tensor_list,  # tensor
        local_key_cache_scale_list,
        local_value_cache_scale_list,
        cache_dtype,
    ):
        """
        Args:
            rank_id_: Rank id of the current process.
            gpu_idx_: Local GPU index used for cache communication.
            local_key_cache_tensor_list: List of local key cache tensors, one per layer.
            local_value_cache_tensor_list: List of local value cache tensors, one per layer.
            local_key_cache_scale_list: List of per-layer scale tensors for key caches,
                used when cache quantization (e.g. dy-fp8) is enabled.
            local_value_cache_scale_list: List of per-layer scale tensors for value caches,
                used when cache quantization (e.g. dy-fp8) is enabled.
            cache_dtype: String indicating the data type/format of the cache
                (for example, "bfloat16" or "block_wise_fp8").
        """
        self.rank_id = rank_id_
        self.gpu_idx = gpu_idx_
        self.cache_dtype = cache_dtype
        # local cache to tensor
        self.local_key_cache_tensor_list = local_key_cache_tensor_list
        self.local_value_cache_tensor_list = local_value_cache_tensor_list
        self.layer_num = len(self.local_key_cache_tensor_list)
        self.local_key_cache_scale_list = local_key_cache_scale_list
        self.local_value_cache_scale_list = local_value_cache_scale_list
        # record connected ipc info
        self.comm_map = {}

    def connect(self, remote_gpu_id_=0):
        """
        Connect to remote gpu.
        """
        logger.info(f"{self.rank_id}: connect to remote_gpu_id:{remote_gpu_id_} {self.layer_num} {self.gpu_idx}")
        if self.is_connected(remote_gpu_id_):
            return True
        else:
            self.comm_map[remote_gpu_id_] = IPCConnector(
                self.rank_id, remote_gpu_id_, self.layer_num, self.gpu_idx, self.cache_dtype
            )
            return True

    def is_connected(self, remote_gpu_id_=0):
        """
        Check if remote gpu is connected.
        """
        if remote_gpu_id_ in self.comm_map.keys():
            return True
        else:
            return False

    def write_cache(self, ip, remote_gpu_id, local_block_ids, remote_block_ids, layer_idx):
        """
        Connect to remote gpu and write cache.
        """
        block_num = len(local_block_ids)
        if not self.is_connected(remote_gpu_id):
            self.connect(remote_gpu_id)
        comm = self.comm_map[remote_gpu_id]
        with paddle.device.stream_guard(comm.write_stream):
            ipc_sent_key_value_cache_by_remote_ptr(
                self.local_key_cache_tensor_list[layer_idx],
                self.local_value_cache_tensor_list[layer_idx],
                local_block_ids,
                remote_block_ids,
                comm.remote_key_tensor_ptr_list[layer_idx],
                comm.remote_value_tensor_ptr_list[layer_idx],
                block_num,
                self.gpu_idx,
                comm.remote_gpu_id,
                comm.write_stream.stream_base.cuda_stream,
                False,
            )
            if self.cache_dtype == "block_wise_fp8":
                logger.debug(f"IPC write cache scales for layer: {layer_idx}")
                ipc_sent_key_value_cache_by_remote_ptr(
                    self.local_key_cache_scale_list[layer_idx],
                    self.local_value_cache_scale_list[layer_idx],
                    local_block_ids,
                    remote_block_ids,
                    comm.remote_key_scale_tensor_ptr_list[layer_idx],
                    comm.remote_value_scale_tensor_ptr_list[layer_idx],
                    block_num,
                    self.gpu_idx,
                    comm.remote_gpu_id,
                    comm.write_stream.stream_base.cuda_stream,
                    True,
                )
        return 0

    def write_block_by_sync(self, remote_gpu_id):
        """
        check finish event and wait for it
        """
        paddle.set_device(f"gpu:{self.gpu_idx}")
        comm = self.comm_map[remote_gpu_id]
        ipc_sent_key_value_cache_by_remote_ptr_block_sync(
            self.local_key_cache_tensor_list[0],  # tensor no use
            self.local_value_cache_tensor_list[0],  # tensor no use
            comm.write_stream.stream_base.cuda_stream,
        )
