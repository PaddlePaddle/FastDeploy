import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def do_rope_kernel(
    qkv_out,
    cos_emb,
    sin_emb,
    cu_seqlens_q,
    seq_lens_decoder,
    batch_id_per_token,
    qkv_size,
    rope_last_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim_k: tl.constexpr,
    head_dim_k_next_power_of_2: tl.constexpr,
):
    token_id = tl.program_id(0)

    qkv_out += token_id * qkv_size

    batch_id = tl.load(batch_id_per_token + token_id)
    if batch_id < 0:
        return

    kv_len = tl.load(seq_lens_decoder + batch_id)
    cu_q_len = tl.load(cu_seqlens_q + batch_id)
    token_id_in_this_batch = token_id - cu_q_len + kv_len

    cos_emb += token_id_in_this_batch * rope_last_dim
    sin_emb += token_id_in_this_batch * rope_last_dim

    offset_emb = tl.arange(0, rope_last_dim)
    cos = tl.load(cos_emb + offset_emb)
    sin = tl.load(sin_emb + offset_emb)

    offset = tl.arange(0, rope_last_dim)

    for head_id in range(num_q_heads + num_kv_heads):

        x0 = tl.load(qkv_out + offset).to(tl.float32)
        x1 = tl.load(qkv_out + rope_last_dim + offset).to(tl.float32)

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        tl.store(qkv_out + offset, y0)
        tl.store(qkv_out + rope_last_dim + offset, y1)

        offset += head_dim_k


def do_rope(
    qkv_out,
    cos_emb,
    sin_emb,
    cu_seqlens_q,
    seq_lens_decoder,
    batch_id_per_token,
    cache_k,
    cache_v,
):
    assert qkv_out.ndim == 2
    assert cache_k.ndim == 4
    assert cache_v.ndim == 4

    head_dim_k = cache_k.shape[-1]
    num_kv_heads = cache_k.shape[1]
    head_dim_v = cache_k.shape[-1]
    qkv_size = qkv_out.shape[-1]
    num_q_heads = (qkv_size - head_dim_v * num_kv_heads) // head_dim_k - num_kv_heads

    M = qkv_out.shape[0]
    grid = (M,)

    do_rope_kernel[grid](
        qkv_out,
        cos_emb,
        sin_emb,
        cu_seqlens_q,
        seq_lens_decoder,
        batch_id_per_token,
        qkv_size,
        cos_emb.shape[-1],
        num_q_heads,
        num_kv_heads,
        head_dim_k,
        head_dim_k_next_power_of_2=triton.next_power_of_2(head_dim_k),
    )
