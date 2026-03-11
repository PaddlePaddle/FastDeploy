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

Triton kernel: fused RoPE + paged KV cache write.

Replaces the C++ gqa_rope_write_cache kernel for deterministic + CUDA Graph path.
Single pass over all tokens, performing:
  - Q heads: apply RoPE, write to q_out buffer
  - K heads: apply RoPE, write to paged cache_k via block_tables
  - V heads: copy to paged cache_v via block_tables (no RoPE)

Supports two RoPE styles (via ROPE_STYLE constexpr):
  - ROPE_STYLE=0: standard interleaved (LLaMA, Qwen2)
    cos/sin shape: [max_seq_len, head_dim/2]
    math: out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
          out[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]
  - ROPE_STYLE=1: neox full (Qwen3)
    cos/sin shape: [max_seq_len, head_dim]
    math: out[i]        = x[i]*cos[i] - x[i+D/2]*sin[i]
          out[i+D/2]    = x[i+D/2]*cos[i] + x[i]*sin[i]
"""

import triton
import triton.language as tl


@triton.jit
def _rope_and_cache_write_kernel(
    qkv_ptr,  # [token_num, total_heads * head_dim]
    q_out_ptr,  # [max_tokens, q_heads, head_dim]
    cache_k_ptr,  # [num_blocks, kv_heads, block_size, head_dim]
    cache_v_ptr,  # [num_blocks, kv_heads, block_size, head_dim]
    cos_ptr,  # [max_seq_len, emb_dim]
    sin_ptr,  # [max_seq_len, emb_dim]
    batch_id_per_token_ptr,  # [token_num]
    cu_seqlens_q_ptr,  # [bs+1]
    seq_lens_encoder_ptr,  # [bs]
    seq_lens_decoder_ptr,  # [bs]
    block_tables_ptr,  # [bs, max_blocks_per_seq]
    q_num_heads: tl.constexpr,
    kv_num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_blocks_per_seq,
    ROPE_STYLE: tl.constexpr,  # 0=standard interleaved, 1=neox full
):
    """
    Grid: (token_num, q_num_heads + 2*kv_num_heads)
    Each program handles one (token, head) pair over HEAD_DIM/2 elements
    (processing pairs for standard, or halves for neox).
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    ori_bi = tl.load(batch_id_per_token_ptr + token_idx)
    if ori_bi == -1:
        return

    # Position calculation (matches C++ exactly)
    cache_kv_len = tl.load(seq_lens_decoder_ptr + ori_bi)
    enc_len = tl.load(seq_lens_encoder_ptr + ori_bi)
    cache_kv_len = tl.where(enc_len == 0, 0, cache_kv_len)  # FA3 workaround
    cu_seqlen_q = tl.load(cu_seqlens_q_ptr + ori_bi)
    ori_seq_id = (token_idx - cu_seqlen_q) + cache_kv_len

    total_heads: tl.constexpr = q_num_heads + 2 * kv_num_heads
    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    qkv_row_stride = total_heads * HEAD_DIM
    head_base = token_idx * qkv_row_stride + head_idx * HEAD_DIM

    is_q = head_idx < q_num_heads
    is_k = (head_idx >= q_num_heads) & (head_idx < q_num_heads + kv_num_heads)
    needs_rope = is_q | is_k

    half_offs = tl.arange(0, HALF_DIM)

    if ROPE_STYLE == 0:
        # Standard interleaved: load even/odd pairs
        even_offs = half_offs * 2  # [0, 2, 4, ...]
        odd_offs = even_offs + 1  # [1, 3, 5, ...]
        x_a = tl.load(qkv_ptr + head_base + even_offs).to(tl.float32)
        x_b = tl.load(qkv_ptr + head_base + odd_offs).to(tl.float32)

        # cos/sin: [max_seq_len, head_dim/2]
        emb_base = ori_seq_id * HALF_DIM + half_offs
        cos_val = tl.load(cos_ptr + emb_base)
        sin_val = tl.load(sin_ptr + emb_base)
    elif ROPE_STYLE == 1:
        # Neox full: load left half / right half
        x_a = tl.load(qkv_ptr + head_base + half_offs).to(tl.float32)
        x_b = tl.load(qkv_ptr + head_base + half_offs + HALF_DIM).to(tl.float32)

        # cos/sin: [max_seq_len, head_dim], use first half_dim
        emb_base = ori_seq_id * HEAD_DIM + half_offs
        cos_val = tl.load(cos_ptr + emb_base)
        sin_val = tl.load(sin_ptr + emb_base)

    # RoPE math (same for both styles):
    #   out_a = a * cos - b * sin
    #   out_b = b * cos + a * sin
    out_a = tl.where(needs_rope, x_a * cos_val - x_b * sin_val, x_a)
    out_b = tl.where(needs_rope, x_b * cos_val + x_a * sin_val, x_b)

    # Cast back to bf16
    out_a_bf = out_a.to(tl.bfloat16)
    out_b_bf = out_b.to(tl.bfloat16)

    # --- Write output ---
    if is_q:
        q_base = token_idx * q_num_heads * HEAD_DIM + head_idx * HEAD_DIM
        if ROPE_STYLE == 0:
            tl.store(q_out_ptr + q_base + even_offs, out_a_bf)
            tl.store(q_out_ptr + q_base + odd_offs, out_b_bf)
        elif ROPE_STYLE == 1:
            tl.store(q_out_ptr + q_base + half_offs, out_a_bf)
            tl.store(q_out_ptr + q_base + half_offs + HALF_DIM, out_b_bf)
    elif is_k:
        kv_head_idx = head_idx - q_num_heads
        block_table_row = ori_bi * max_blocks_per_seq
        block_id = tl.load(block_tables_ptr + block_table_row + ori_seq_id // BLOCK_SIZE)
        block_offset = ori_seq_id % BLOCK_SIZE
        cache_base = (
            block_id * kv_num_heads * BLOCK_SIZE * HEAD_DIM
            + kv_head_idx * BLOCK_SIZE * HEAD_DIM
            + block_offset * HEAD_DIM
        )
        if ROPE_STYLE == 0:
            tl.store(cache_k_ptr + cache_base + even_offs, out_a_bf)
            tl.store(cache_k_ptr + cache_base + odd_offs, out_b_bf)
        elif ROPE_STYLE == 1:
            tl.store(cache_k_ptr + cache_base + half_offs, out_a_bf)
            tl.store(cache_k_ptr + cache_base + half_offs + HALF_DIM, out_b_bf)
    else:
        # V: no RoPE, direct copy
        v_head_idx = head_idx - q_num_heads - kv_num_heads
        block_table_row = ori_bi * max_blocks_per_seq
        block_id = tl.load(block_tables_ptr + block_table_row + ori_seq_id // BLOCK_SIZE)
        block_offset = ori_seq_id % BLOCK_SIZE
        cache_base = (
            block_id * kv_num_heads * BLOCK_SIZE * HEAD_DIM
            + v_head_idx * BLOCK_SIZE * HEAD_DIM
            + block_offset * HEAD_DIM
        )
        # Load full HEAD_DIM for V (need both halves)
        d_offs = tl.arange(0, HEAD_DIM)
        v_data = tl.load(qkv_ptr + head_base + d_offs)
        tl.store(cache_v_ptr + cache_base + d_offs, v_data)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def triton_rope_and_cache_write(
    qkv,  # [token_num, (q_heads + 2*kv_heads) * head_dim], bfloat16
    cache_k,  # [num_blocks, kv_heads, block_size, head_dim], bfloat16
    cache_v,  # [num_blocks, kv_heads, block_size, head_dim], bfloat16
    q_out,  # [max_tokens, q_heads, head_dim], bfloat16 (pre-allocated)
    rotary_embs,  # [2, 1, max_seq_len, 1, emb_dim], float32
    batch_id_per_token,  # [token_num], int32
    cu_seqlens_q,  # [bs+1], int32
    seq_lens_encoder,  # [bs], int32
    seq_lens_decoder,  # [bs], int32
    block_tables,  # [bs, max_blocks_per_seq], int32
    q_num_heads,
    kv_num_heads,
    head_dim,
    block_size,
    use_neox_rotary_style=False,
):
    """
    Fused RoPE + paged cache write (Triton).

    Writes q_out[:token_num], cache_k, cache_v in-place.
    Returns q_out for convenience.
    """
    token_num = qkv.shape[0]
    max_blocks_per_seq = block_tables.shape[1]

    # Determine RoPE style
    emb_last_dim = rotary_embs.shape[4]
    if use_neox_rotary_style and emb_last_dim == head_dim:
        rope_style = 1  # neox full (Qwen3)
    else:
        rope_style = 0  # standard interleaved

    # Extract cos/sin as contiguous 2D tensors.
    # Slice of [2, 1, max_seq_len, 1, emb_dim] may be non-contiguous;
    # Triton kernel assumes linear addressing, so ensure contiguity.
    cos = rotary_embs[0, 0, :, 0, :].contiguous()  # [max_seq_len, emb_dim]
    sin = rotary_embs[1, 0, :, 0, :].contiguous()  # [max_seq_len, emb_dim]

    total_heads = q_num_heads + 2 * kv_num_heads
    grid = (token_num, total_heads)

    _rope_and_cache_write_kernel[grid](
        qkv,
        q_out,
        cache_k,
        cache_v,
        cos,
        sin,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        ROPE_STYLE=rope_style,
    )

    return q_out
