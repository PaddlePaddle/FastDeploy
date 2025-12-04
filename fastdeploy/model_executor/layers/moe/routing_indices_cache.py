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
from typing import Dict

import paddle
import paddle.distributed as dist
import triton
import triton.language as tl

from fastdeploy.config import FDConfig


@triton.jit
def _save_routing_kernel(
    ROUTING_TABLE_BUFFER_PTR,
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
    topk_ids_ptrs = TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :]
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    topk_vals = tl.load(topk_ids_ptrs, mask=token_mask[:, None])

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
        ROUTING_TABLE_BUFFER_PTR
        + batch_ids[:, None] * STRIDE_BUF_SEQ
        + LAYER_IDX * STRIDE_BUF_LAYER
        + token_seq_pos[:, None] * STRIDE_BUF_TOKEN
        + k_offsets[None, :]
    )

    pos_mask = token_seq_pos < MAX_MODEL_LEN
    pos_mask = pos_mask & pad_mask
    final_mask = token_mask[:, None] & pos_mask[:, None]

    tl.store(output_ptrs, topk_vals, mask=final_mask)


def save_routing_to_buffer(
    routing_table_buffer: paddle.Tensor,  # [max_num_seqs, num_layers, max_len, top_k]
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
    max_num_seqs, num_hidden_layers, max_model_len, _ = routing_table_buffer.shape
    assert token_num > 0
    assert topk_ids.shape[1] == routing_table_buffer.shape[3], (topk_ids.shape[1], routing_table_buffer.shape[3])
    assert batch_id_per_token.shape[0] == token_num, (batch_id_per_token.shape[0], token_num)
    assert seq_lens_decoder.shape[0] == max_num_seqs, (seq_lens_decoder.shape[0], max_num_seqs)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = top_k  # 值一般很小，直接设为 top_k

    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel[grid](
        routing_table_buffer,
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


# max_num_seqs = 4
# num_layers = 1
# max_len = 10
# top_k = 8
# token_num = 12

# routing_table_buffer = paddle.full([max_num_seqs, num_layers, max_len, top_k], -1, dtype="int32")
# topk_ids = paddle.randint(0, 384, [token_num, top_k], dtype="int32")
# batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3], dtype="int32").reshape([-1, 1])
# seq_lens_decoder = paddle.to_tensor([0, 2, 0, 3], dtype="int32").reshape([-1, 1])
# cu_seqlens_q = paddle.to_tensor([0, 3, 4, 10, 12], dtype="int32").reshape([-1, 1])
# current_layer_idx = 0

# save_routing_to_buffer(
#     routing_table_buffer=routing_table_buffer,
#     topk_ids=topk_ids,
#     batch_id_per_token=batch_id_per_token,
#     seq_lens_decoder=seq_lens_decoder,
#     cu_seqlens_q=cu_seqlens_q,
#     layer_idx=current_layer_idx,
# )


class RoutingReplayManager:
    def __init__(
        self,
        fd_config: FDConfig,
        output_dir: str = "./routing_replay_output",
    ):
        self.max_num_seqs = fd_config.parallel_config.max_num_seqs
        self.max_model_len = fd_config.model_config.max_model_len
        self.num_moe_layers = fd_config.model_config.num_hidden_layers - fd_config.model_config.moe_layer_start_index
        self.moe_top_k = fd_config.model_config.moe_k
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        self.output_dir = output_dir

        self.routing_batch_to_request: Dict[int, str] = {}

        self.routing_table_buffer = paddle.full(
            shape=[self.max_num_seqs, self.num_moe_layers, self.max_model_len, self.moe_top_k],
            fill_value=-1,
            dtype="int32",
        )

    def _deregister_request(self, batch_id: int) -> str:
        assert batch_id in self.routing_batch_to_request
        return self.routing_batch_to_request.pop(batch_id)

    def _clear_buffer_slot(self, batch_id: int):
        assert 0 <= batch_id < self.max_num_seqs
        self.routing_table_buffer[batch_id].fill_(-1)

    def _save_routing_to_file(
        self,
        batch_id: int,
        request_id: str,
    ):
        if self.tp_rank == 0:
            dir_path = os.path.join(self.output_dir, f"{request_id}")
            os.makedirs(dir_path, exist_ok=True)
            batch_buffer = self.routing_table_buffer[batch_id]
            for layer_id in range(self.num_moe_layers):
                layer_buffer = batch_buffer[layer_id]
                print(f"{layer_id=}, {layer_buffer=}")
                file_path = os.path.join(
                    dir_path, f"layer_{layer_id}_shape_{self.max_model_len}x{self.moe_top_k}.pdtensor"
                )
                paddle.save(layer_buffer, file_path)

        self._clear_buffer_slot(batch_id)

    def clear_buffer(self):
        self.routing_table_buffer.fill_(-1)

    def register_request(self, batch_id: int, request_id: str):
        if batch_id in self.routing_batch_to_request:
            pre_request_id = self._deregister_request(batch_id)
            self._save_routing_to_file(batch_id, pre_request_id)

        self.routing_batch_to_request[batch_id] = request_id

    def get_buffer(self) -> paddle.Tensor:
        return self.routing_table_buffer

    def save_tail_routing(self):
        batch_ids = copy.deepcopy(list(self.routing_batch_to_request.keys()))
        for batch_id in batch_ids:
            request_id = self._deregister_request(batch_id)
            self._save_routing_to_file(batch_id, request_id)
