"""Paddle port of ``Xattention_prefill_dim4`` (BS=1 / varlen path).

Pipeline:
  q,k: [1, H, T, D]   (post k_norm, post RoPE)
  -> chunked Triton GEMM + softmax block-sum  -> [1,H,Qb,Kb] block sum
  -> find_blocks_chunked (threshold)          -> [1,H,Qb,Kb] bool mask
  -> block_sparse_attn (paddle custom op)     -> attn_out [T,H,D]
  -> reshape back to [1,H,T,D]
"""

from __future__ import annotations

import math

import paddle
import paddle.nn.functional as F

from .block_sparse_attn import block_sparse_attn_paddle
from .find_blocks import find_blocks_chunked
from .xattention_triton import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum


def _pad_seq(x: paddle.Tensor, num_to_pad: int) -> paddle.Tensor:
    """Pad seq dim (axis=2) of [B,H,T,D] with zeros."""
    if num_to_pad <= 0:
        return x
    return F.pad(x, [0, 0, 0, num_to_pad], value=0, data_format="NCHW")


def xattn_estimate(
    query_states: paddle.Tensor,  # [1, H, q_len, D]
    key_states: paddle.Tensor,    # [1, H, k_len, D]
    block_size: int,
    stride: int,
    norm: float = 1.0,
    threshold: float = 0.9,
    chunk_size: int = 16384,
    use_triton: bool = True,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
):
    assert use_triton, "paddle port only supports the Triton path"
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    _, num_kv_head, k_len, _ = key_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    assert k_chunk_num >= q_chunk_num

    pad_q = _pad_seq(query_states, q_num_to_pad) if q_num_to_pad > 0 else query_states
    pad_k = _pad_seq(key_states, k_num_to_pad) if k_num_to_pad > 0 else key_states

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    attn_sum_list = []
    simple_mask_list = []

    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride / norm

    for chunk_idx in range(q_chunk_num):
        q_start = chunk_idx * reshaped_chunk_size * stride
        q_end = q_start + reshaped_chunk_size * stride
        chunk_q = pad_q[:, :, q_start:q_end, :]

        attn_weights_slice = flat_group_gemm_fuse_reshape(
            chunk_q,
            pad_k,
            stride,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
            is_causal=causal,
        )
        attn_sum = softmax_fuse_block_sum(
            attn_weights_slice,
            reshaped_block_size,
            min(4096, reshaped_block_size),
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
            k_reshaped_seq_len - k_reshaped_num_to_pad,
            scale,
            is_causal=causal,
        )

        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )
        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

    attn_sums = paddle.concat(attn_sum_list, axis=-2)
    simple_masks = paddle.concat(simple_mask_list, axis=-2)

    if causal:
        mask_size = min(q_block_num, simple_masks.shape[-1])
        if mask_size > 0:
            tri = paddle.triu(
                paddle.ones([mask_size, mask_size], dtype="bool"), diagonal=1
            )
            causal_block_mask = paddle.logical_not(tri)
            sub = simple_masks[:, :, -mask_size:, -mask_size:]
            simple_masks[:, :, -mask_size:, -mask_size:] = paddle.logical_and(sub, causal_block_mask)
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye = paddle.eye(q_block_num, dtype="int32").astype("bool").unsqueeze(0).unsqueeze(0)
        eye = eye.expand([1, num_kv_head, q_block_num, q_block_num])
        sub = simple_masks[:, :, -q_block_num:, -q_block_num:]
        simple_masks[:, :, -q_block_num:, -q_block_num:] = paddle.where(eye, paddle.ones_like(sub), sub)

    return attn_sums, simple_masks


@paddle.no_grad()
def Xattention_prefill_dim4(
    query_states: paddle.Tensor,  # [1, H, T, D]
    key_states: paddle.Tensor,    # [1, H, T, D]
    value_states: paddle.Tensor,  # [1, H, T, D]
    stride: int,
    cu_seq_lens: paddle.Tensor,   # int32 [B+1]; BS=1 => [0, T]
    norm: float = 1.0,
    threshold: float = 0.8,
    block_size: int = 128,
    use_triton: bool = True,
    causal: bool = True,
    chunk_size: int | None = None,
    keep_sink: bool = False,
    keep_recent: bool = False,
    head_mask_type: paddle.Tensor | None = None,
    sink_num: int = 1,
    local_num: int = 16,
) -> paddle.Tensor:
    batch_size, num_heads, max_q_len, head_dim = query_states.shape
    _, _, max_k_len, _ = key_states.shape
    assert batch_size == 1, "this paddle port targets BS=1 only (FastDeploy varlen)"

    valid_len = int(cu_seq_lens[1].item()) - int(cu_seq_lens[0].item())

    cur_q = query_states[:, :, :valid_len, :]
    cur_k = key_states[:, :, :valid_len, :]
    cur_klen = cur_k.shape[2]
    if chunk_size is None:
        chunk_size = max(
            min(
                max(2048, 1 << (cur_klen - 1).bit_length()),
                128 * 1024 * 2048 // (1 << (cur_klen - 1).bit_length()),
            ),
            2048,
        )

    _, approx_mask = xattn_estimate(
        cur_q, cur_k,
        block_size=block_size, stride=stride, norm=norm, threshold=threshold,
        use_triton=True, causal=causal, chunk_size=chunk_size,
        keep_sink=keep_sink, keep_recent=keep_recent,
    )

    valid_q_blocks = (valid_len + block_size - 1) // block_size
    valid_k_blocks = (valid_len + block_size - 1) // block_size
    approx_mask[:, :, valid_q_blocks:, :] = False
    approx_mask[:, :, :, valid_k_blocks:] = False

    # ---- BSA expects [total_T, H, D] varlen layout ----
    total_T = int(cu_seq_lens[-1].item())
    # query/key/value [1,H,T,D] -> [T,H,D]
    q_var = query_states.squeeze(0).transpose([1, 0, 2])[:total_T].contiguous()
    k_var = key_states.squeeze(0).transpose([1, 0, 2])[:total_T].contiguous()
    v_var = value_states.squeeze(0).transpose([1, 0, 2])[:total_T].contiguous()

    if head_mask_type is None:
        head_mask_type = paddle.ones([num_heads], dtype="int32")

    max_q_block_num = (max_q_len + block_size - 1) // block_size
    max_k_block_num = (max_k_len + block_size - 1) // block_size

    sparse_mask_idx = paddle.nonzero(head_mask_type == 1).reshape([-1])
    if sparse_mask_idx.shape[0] > 0:
        blockmask = paddle.index_select(approx_mask, sparse_mask_idx, axis=1)
        blockmask = blockmask[:, :, :max_q_block_num, :max_k_block_num].contiguous()
    else:
        # No sparse heads -- BSA still wants a tensor; pass empty.
        blockmask = paddle.ones(
            [1, 0, max_q_block_num, max_k_block_num], dtype="bool"
        )

    streaming_info = paddle.to_tensor(
        [sink_num, local_num] * num_heads, dtype="int32"
    )

    attn_out = block_sparse_attn_paddle(
        q_var, k_var, v_var,
        cu_seq_lens, cu_seq_lens,
        head_mask_type, streaming_info, blockmask,
        max_q_len, max_k_len,
        p_dropout=0.0, deterministic=True, is_causal=causal,
        m_block_dim=block_size, n_block_dim=block_size,
    )  # [total_T, H, D]

    # Back to [1,H,T,D] padded.
    out = paddle.zeros([1, num_heads, max_q_len, head_dim], dtype=attn_out.dtype)
    out[0, :, :total_T, :] = attn_out.transpose([1, 0, 2])
    return out
