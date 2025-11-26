"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License,
 Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
 software
# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle

from fastdeploy.platforms import current_platform


def reorder_split_prefill_and_decode_python(x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder):
    """
    Python implementation for reordering tokens by putting decode tokens first and prefill tokens after.

    Args:
        x_remove_padding: Tensor of shape [total_tokens]
        batch_id_per_token: Tensor of shape [total_tokens]
        cu_seqlens_q: Tensor of shape [batch_size + 1]
        seq_lens_encoder: Tensor of shape [batch_size]

    Returns:
        Tuple of (x_reorder, batch_id_reorder, num_decode_tokens)
    """
    total_tokens = x_remove_padding.shape[0]

    if total_tokens < 1:
        raise ValueError("Input cannot be empty")

    # Calculate sequence lengths and decode counts using vectorized operations
    seq_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    decode_counts = seq_lengths - seq_lens_encoder
    total_decode = paddle.sum(decode_counts)

    # Create sequence indices for all tokens
    # Generate batch indices for each token using cumsum and searchsorted
    token_indices = paddle.arange(total_tokens, dtype="int32")
    batch_for_token = paddle.searchsorted(cu_seqlens_q, token_indices, right=True) - 1

    # Calculate relative positions within each sequence
    token_pos_in_seq = token_indices - cu_seqlens_q[batch_for_token]

    # Get prefill lengths for each token
    prefill_lengths = paddle.gather(seq_lens_encoder, batch_for_token)

    # Create masks using vectorized operations
    decode_mask = token_pos_in_seq >= prefill_lengths
    prefill_mask = token_pos_in_seq < prefill_lengths

    # Reorder tokens using masks
    x_reorder = paddle.concat([x_remove_padding[decode_mask], x_remove_padding[prefill_mask]])

    batch_id_reorder = paddle.concat([batch_id_per_token[decode_mask], batch_id_per_token[prefill_mask]])

    num_decode_tokens = paddle.to_tensor([total_decode], dtype="int64")

    return x_reorder, batch_id_reorder, num_decode_tokens


def reorder_split_prefill_and_decode(x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder):
    """
    Unified interface for reordering tokens, automatically selects GPU or CPU implementation.

    Args:
        x_remove_padding: Tensor of shape [total_tokens]
        batch_id_per_token: Tensor of shape [total_tokens]
        cu_seqlens_q: Tensor of shape [batch_size + 1]
        seq_lens_encoder: Tensor of shape [batch_size]

    Returns:
        Tuple of (x_reorder, batch_id_reorder, num_decode_tokens)
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import (
            reorder_split_prefill_and_decode as reorder_split_prefill_and_decode_cuda,
        )

        return reorder_split_prefill_and_decode_cuda(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
        )
    else:
        return reorder_split_prefill_and_decode_python(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
        )
