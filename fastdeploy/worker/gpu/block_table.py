"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
import triton
import triton.language as tl

from fastdeploy.utils import ceil_div
from fastdeploy.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
    ):
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.max_num_blocks = ceil_div(self.max_model_len, self.block_size)

        self.block_table = StagedWriteTensor(
            (self.max_num_seqs, self.max_num_blocks), dtype=paddle.int32, init_value=-1
        )
        self.num_blocks = UvaBackedTensor(self.max_num_seqs, dtype=paddle.int32, init_value=0)

        self.input_block_table = paddle.full(self.block_table.shape, -1, dtype=paddle.int32)
        self.slot_mappings = paddle.full(
            self.max_num_batched_tokens,
            0,
            dtype=paddle.int64,
        )

    def append_block_ids(
        self,
        req_idx: int,
        block_ids: list[int],
    ) -> None:
        self.block_table.stage_write(req_idx, 0, block_ids)
        self.num_blocks.np[req_idx] = len(block_ids)

    def apply_staged_writes(self) -> None:
        self.block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: paddle.Tensor,
    ) -> tuple[paddle.Tensor, ...]:
        """
        纯纯的重排
        """
        num_seqs = idx_mapping.shape[0]
        _gather_block_tables_kernel[(self.max_num_seqs,)](
            idx_mapping,
            self.block_table.gpu,
            self.input_block_table,
            self.block_table.gpu.stride(0),
            self.num_blocks.gpu,
            num_seqs,
            self.max_num_blocks,
            BLOCK_SIZE=1024,
        )
        return self.input_block_table

    def compute_slot_mappings(
        self,
        idx_mapping: paddle.Tensor,
        query_start_loc: paddle.Tensor,
        positions: paddle.Tensor,
    ) -> paddle.Tensor:
        num_seqs = idx_mapping.shape[0]
        # TODO 根据cuda graph的分桶做padding
        _compute_slot_mappings_kernel[(num_seqs + 1,)](
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            BLOCK_SIZE=1024,
        )
        return self.slot_mappings


@triton.jit(do_not_specialize=["num_seqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,
    src_block_table_ptr,
    dst_block_table_ptr,
    block_table_stride,
    num_blocks_ptr,
    num_seqs,
    max_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dst_row_ptr = dst_block_table_ptr + batch_idx * block_table_stride

    if batch_idx >= num_seqs:
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, -1, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    num_blocks = tl.load(num_blocks_ptr + req_idx)
    src_row_ptr = src_block_table_ptr + req_idx * block_table_stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_seqs]
    query_start_loc,  # [num_seqs + 1]
    pos,  # [num_tokens]
    block_table_ptr,  # [max_num_seqs, max_num_blocks]
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens]
    slot_mappings_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx == tl.num_programs(0) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, -1, mask=offset < max_num_tokens)
        return

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // block_size
        block_offsets = positions % block_size
        block_numbers = tl.load(block_table_ptr + req_state_idx * block_table_stride + block_indices)
        slot_ids = block_numbers * block_size + block_offsets
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
