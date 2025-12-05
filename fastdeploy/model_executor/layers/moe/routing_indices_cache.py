"""
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

import copy
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import paddle
import paddle.distributed as dist
import triton
import triton.language as tl

from fastdeploy.config import FDConfig


@triton.jit
def _save_routing_kernel(
    ROUTING_REPLAY_TABLE_PTR,
    TOPK_IDS_PTR,
    BATCH_ID_PER_TOKEN_PTR,
    CU_SEQLENS_Q_PTR,
    SEQ_LENS_DECODER_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_HIDDEN_LAYERS,
    MAX_MODEL_LEN,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM

    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    k_mask = k_offsets < TOP_K

    topk_ids_ptrs = TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :]
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(topk_ids_ptrs, mask=load_mask)

    batch_ids = tl.load(BATCH_ID_PER_TOKEN_PTR + token_offsets, mask=token_mask)
    pad_mask = token_mask & (batch_ids != -1)
    # [0, 3, 4, 10, 12][0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3]
    # -> [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] - [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 10, 10]
    # -> [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1]
    start_offsets = tl.load(CU_SEQLENS_Q_PTR + batch_ids, mask=pad_mask)
    token_relative_index = token_offsets - start_offsets

    # [BLOCK_SIZE_M]
    len_decoder = tl.load(SEQ_LENS_DECODER_PTR + batch_ids, mask=pad_mask)
    token_seq_pos = len_decoder + token_relative_index

    STRIDE_BUF_SEQ = NUM_HIDDEN_LAYERS * MAX_MODEL_LEN * TOP_K
    STRIDE_BUF_LAYER = MAX_MODEL_LEN * TOP_K
    STRIDE_BUF_TOKEN = TOP_K

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    output_ptrs = (
        ROUTING_REPLAY_TABLE_PTR
        + batch_ids[:, None] * STRIDE_BUF_SEQ
        + LAYER_IDX * STRIDE_BUF_LAYER
        + token_seq_pos[:, None] * STRIDE_BUF_TOKEN
        + k_offsets[None, :]
    )

    pos_mask = token_seq_pos < MAX_MODEL_LEN
    pos_mask = pos_mask & pad_mask

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    pos_mask = pos_mask[:, None] & k_mask[None, :]

    final_mask = load_mask & pos_mask

    tl.store(output_ptrs, topk_vals, mask=final_mask)


def save_routing_to_buffer(
    routing_replay_table: paddle.Tensor,  # [max_num_seqs, num_layers, max_len, top_k]
    topk_ids: paddle.Tensor,  # [token_num, top_k]
    batch_id_per_token: paddle.Tensor,  # [token_num, 1]
    seq_lens_decoder: paddle.Tensor,  # [max_num_seqs, 1]
    cu_seqlens_q: paddle.Tensor,  # [max_num_seqs + 1, 1]
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
):
    if tp_size > 1 and ep_size > 1:
        token_num_per_rank = topk_ids.shape[0]
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        topk_ids = topk_ids_all[: batch_id_per_token.shape[0], :]

    token_num, top_k = topk_ids.shape
    max_num_seqs, num_hidden_layers, max_model_len, _ = routing_replay_table.shape
    assert token_num > 0

    assert topk_ids.shape[1] == routing_replay_table.shape[3], (topk_ids.shape[1], routing_replay_table.shape[3])
    assert batch_id_per_token.shape[0] == token_num, (batch_id_per_token.shape[0], token_num)
    assert seq_lens_decoder.shape[0] == max_num_seqs, (seq_lens_decoder.shape[0], max_num_seqs)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)  # top_k

    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel[grid](
        routing_replay_table,
        topk_ids,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_decoder,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_HIDDEN_LAYERS=num_hidden_layers,
        MAX_MODEL_LEN=max_model_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


class RoutingReplayManager:
    """Request level routing replay table manager"""

    def __init__(
        self,
        fd_config: FDConfig,
    ):
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.max_model_len = fd_config.model_config.max_model_len
        self.num_moe_layers = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index

        if fd_config.model_config.architectures[0] == "Glm4MoeForCausalLM":
            self.moe_top_k = fd_config.model_config.num_experts_per_tok
        else:
            self.moe_top_k = fd_config.model_config.moe_k
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        self.routing_store = get_routing_store(fd_config=fd_config)
        self.routing_batch_to_request: Dict[int, str] = {}
        self.routing_replay_table = paddle.full(
            shape=[self.max_num_seqs, self.num_moe_layers, self.max_model_len, self.moe_top_k],
            fill_value=-1,
            dtype="int32",
        )

    def register_request(self, batch_id: int, request_id: str):
        """
        Register a new request to routing replay table
        Args:
            batch_id: The batch ID of this request
            request_id: The global ID of the request is usually executed by the training process in RL
        """
        # Save requests that have been finished for the current slot
        if batch_id in self.routing_batch_to_request:
            pre_request_id = self._deregister_request(batch_id)
            self._put_request_to_store(batch_id, pre_request_id)
        # Register the new request
        self.routing_batch_to_request[batch_id] = request_id

    def _deregister_request(self, batch_id: int) -> str:
        """
        Deregister a request from routing replay table
        """
        assert batch_id in self.routing_batch_to_request
        return self.routing_batch_to_request.pop(batch_id)

    def _put_request_to_store(
        self,
        batch_id: int,
        request_id: str,
    ):
        if self.tp_rank == 0:
            batch_buffer = self.routing_replay_table[batch_id]
            for layer_id in range(self.num_moe_layers):
                layer_buffer = batch_buffer[layer_id]
                rollout_id = self.split_request_id(request_id)
                self.routing_store.put(routing_indices=layer_buffer, rollout_id=rollout_id, layer_idx=layer_id)

        self._clear_table_slot(batch_id)

    def put_table_to_store(self):
        """Put the routing table"""
        batch_ids = copy.deepcopy(list(self.routing_batch_to_request.keys()))
        for batch_id in batch_ids:
            request_id = self._deregister_request(batch_id)
            self._put_request_to_store(batch_id, request_id)

    def _clear_table_slot(self, batch_id: int):
        assert 0 <= batch_id < self.max_num_seqs
        self.routing_replay_table[batch_id].fill_(-1)

    def clear_routing_table(self):
        """Clear all slots of the routing replay table"""
        self.routing_replay_table.fill_(-1)

    def _clear_store(self):
        """Clear routing store"""
        self.routing_store.clear_store()

    def _clear_request_of_store(self, request_id):
        """Clear one request of routing store"""
        rollout_id = self.split_request_id(request_id)
        for layer_idx in range(self.num_moe_layers):
            self.routing_store.clear(rollout_id=rollout_id, layer_idx=layer_idx)

    def get_request_from_store(self, request_id: str) -> List[paddle.Tensor]:
        """Get the routing indices of the request from store"""
        routing_list = []
        rollout_id = self.split_request_id(request_id)
        for layer_idx in range(self.num_moe_layers):
            one_layer_routing = self.routing_store.get(rollout_id, layer_idx)
            routing_list.append(one_layer_routing)

        return routing_list

    def get_routing_table(self) -> paddle.Tensor:
        return self.routing_replay_table

    def split_request_id(self, request_id: str):
        """Split the request id to get rollout id"""
        chat_type, tmp_str = request_id.split("-", 1)
        # NOTE(gongshaotian): only support chatcmpl now
        # assert chat_type == "chatcmpl"
        reversed_tmp_str = tmp_str[::-1].split("-", 5)
        rollout_id = reversed_tmp_str[-1][::-1]
        return rollout_id


class RoutingStoreBase(ABC):
    """Base class for routing store"""

    def __init__(self, fd_config: FDConfig) -> None:
        self.fd_config = fd_config

    @abstractmethod
    def put(self, routing_indices: paddle.Tensor, rollout_id: str, layer_idx: Optional[int] = None) -> None:
        """Put the routing indices into store"""
        raise NotImplementedError

    @abstractmethod
    def get(self, rollout_id: str, layer_idx: Optional[int] = None) -> paddle.Tensor:
        """Get the routing indices from store"""
        raise NotImplementedError

    @abstractmethod
    def clear(self, rollout_id: str, layer_idx: Optional[int] = None) -> None:
        """Clear the routing indices of the request"""
        raise NotImplementedError

    @abstractmethod
    def clear_store(
        self,
    ):
        """Clear the routing indices store"""
        raise NotImplementedError


class RoutingStoreLocal(RoutingStoreBase):
    """Routing Store using local memory"""

    def __init__(self, fd_config) -> None:
        super().__init__(fd_config=fd_config)
        self.local_store_dir = fd_config.routing_replay_config.local_store_dir

    def put(self, routing_indices: paddle.Tensor, rollout_id: str, layer_idx: int) -> None:
        """Put the routing indices into store"""
        dir_path = os.path.join(self.local_store_dir, f"{rollout_id}")
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"layer_{layer_idx}.pdtensor")
        paddle.save(routing_indices, file_path)

    def get(
        self,
        rollout_id: str,
        layer_idx: int = None,
    ) -> paddle.Tensor:
        """Get the routing indices from store"""
        dir_path = os.path.join(self.local_store_dir, f"{rollout_id}")
        file_path = os.path.join(dir_path, f"layer_{layer_idx}.pdtensor")
        assert os.path.exists(file_path), f"File not found: {file_path}"
        layer_routing_indices = paddle.load(file_path)

        return layer_routing_indices

    def clear(
        self,
        rollout_id: str,
        layer_idx: int = None,
    ) -> None:
        """Clear the routing indices of the request"""
        dir_path = os.path.join(self.local_store_dir, f"{rollout_id}")
        file_path = os.path.join(dir_path, f"layer_{layer_idx}.pdtensor")
        assert os.path.exists(file_path), f"File not found: {file_path}"
        os.remove(file_path)

        # Delete empty directory
        if len(os.listdir(dir_path)) == 0:
            os.rmdir(dir_path)

    def clear_store(self):
        """Clear the routing indices store"""
        if os.path.isdir(self.local_store_dir):
            for file_name in os.listdir(self.local_store_dir):
                file_path = os.path.join(self.local_store_dir, file_name)
                shutil.rmtree(file_path)


class RoutingStoreRDMA(RoutingStoreBase):
    """Routing Store using RDMA"""

    def __init__(self) -> None:
        super().__init__()


def get_routing_store(fd_config: FDConfig) -> RoutingStoreBase:
    if fd_config.routing_replay_config.routing_store_type == "local":
        return RoutingStoreLocal(fd_config=fd_config)
    elif fd_config.routing_replay_config.routing_store_type == "rdma":
        return RoutingStoreRDMA(fd_config=fd_config)
    else:
        raise ValueError(
            f"Invalid routing store type: '{fd_config.routing_replay_config.routing_store_type}'. "
            "Valid types are: 'local', 'rdma'"
        )
