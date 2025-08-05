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

# refer to https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py

from typing import List, Optional

import paddle

try:
    import triton
    import triton.language as tl
except ImportError as err:
    raise ImportError("Triton is not installed") from err


@triton.jit
def apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    num_rows,
    vocab_size,
    logits_strides,
    bitmask_strides,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for in-place logits masking using bitwise compression.

    Processes logits tensor in blocks, applying bitmask to restrict vocabulary access.
    Masked positions are set to -inf to ensure zero probability during sampling.

    Note:
    - Bitmask uses 32:1 compression (1 bit per vocabulary token)
    - Optimized for GPU parallel processing with configurable block size
    """
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    for work_id in tl.range(pid, num_rows * num_blocks, NUM_SMS):
        row_id = work_id // num_blocks
        block_offset = (work_id % num_blocks) * BLOCK_SIZE
        batch_id = row_id if indices_ptr is None else tl.load(indices_ptr + row_id)
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_strides
        packed_bitmask = tl.load(bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets, packed_bitmask_mask)
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        tl.store(logits_ptr + batch_id * logits_strides + offsets, -float("inf"), vocab_mask & bitmask)


def apply_token_bitmask_inplace_triton(
    logits: paddle.Tensor,
    bitmask: paddle.Tensor,
    vocab_size: Optional[int] = None,
    indices: Optional[List[int]] = None,
):
    """Applies vocabulary mask to logits tensor using Triton GPU kernel.

    Args:
        logits: Input logits tensor of shape [batch_size, vocab_size]
        bitmask: Compressed mask tensor (int32) where each bit represents a token
        vocab_size: Optional explicit vocabulary size (defaults to auto-detected)
        indices: Optional list of batch indices to apply mask to

    Note:
        Requires CUDA GPU with Triton support
        Bitmask must be int32 tensor with shape [batch_size, ceil(vocab_size/32)]
    """
    NUM_SMS = paddle.device.cuda.get_device_properties().multi_processor_count
    BLOCK_SIZE = 4096

    assert bitmask.dtype == paddle.int32, "bitmask must be of type int32"

    detected_vocab_size = min(logits.shape[-1], bitmask.shape[-1] * 32)
    if vocab_size is None:
        vocab_size = detected_vocab_size
    else:
        assert (
            vocab_size <= detected_vocab_size
        ), f"vocab_size {vocab_size} is larger than the detected vocab_size {detected_vocab_size}"

    num_rows = len(indices) if indices is not None else logits.shape[0] if logits.ndim == 2 else 1

    if indices is not None:
        indices = paddle.to_tensor(indices, dtype=paddle.int32, place=logits.place)

    grid = (NUM_SMS,)

    apply_token_bitmask_inplace_kernel[grid](
        logits,
        bitmask,
        indices,
        num_rows,
        vocab_size,
        logits.shape[-1],
        bitmask.shape[-1],
        NUM_SMS,
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32 // (16 // logits.element_size()),
        num_stages=3,
    )
