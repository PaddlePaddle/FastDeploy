# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
FLA sequence chunking index utility functions.

Porting notes:
  - Replaced torch with paddle
  - torch.cat([...]) → paddle.concat([...])
  - torch.arange(n) → paddle.arange(n)
  - torch.stack([a,b], 1) → paddle.stack([a,b], axis=1)
  - tensor.eq(0) → (tensor == 0)
  - .cumsum(0) → .cumsum(axis=0)
  - cu_seqlens.new_tensor([0]) → paddle.to_tensor([0], dtype=cu_seqlens.dtype)
"""

import paddle
import triton

from fastdeploy.model_executor.ops.triton_ops.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: paddle.Tensor) -> paddle.Tensor:
    """Compute the length of each sequence [N]."""
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: paddle.Tensor, chunk_size: int) -> paddle.Tensor:
    """
    Generate (seq_idx, chunk_in_seq_idx) pairs for each chunk.
    Returns shape: [num_chunks, 2], dtype matches cu_seqlens.
    """
    indices = paddle.concat([paddle.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    # (indices == 0) marks the first chunk of each sequence
    seq_ids = (indices == 0).cast(paddle.int64).cumsum(axis=0) - 1
    return paddle.stack([seq_ids, indices], axis=1).cast(cu_seqlens.dtype)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: paddle.Tensor, chunk_size: int) -> paddle.Tensor:
    """
    Compute the chunk start offset for each sequence (cumulative chunk count per sequence).
    Returns shape: [N+1], dtype matches cu_seqlens.
    """
    lens_in_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    return paddle.concat([paddle.to_tensor([0], dtype=cu_seqlens.dtype), lens_in_chunks]).cumsum(axis=0)
