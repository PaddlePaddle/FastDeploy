import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def write_cache_kernel(
    qkv_out,
    cache_k,
    cache_v,
    cu_seqlens_q,
    seq_lens_decoder,
    batch_id_per_token,
    page_table,
    q_size,
    page_size: tl.constexpr,
    max_page_per_seq: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim_k: tl.constexpr,
    head_dim_v: tl.constexpr,
    head_dim_k_next_power_of_2: tl.constexpr,
    head_dim_v_next_power_of_2: tl.constexpr,
):
    token_id = tl.program_id(0)

    qkv_out += token_id * (q_size + num_kv_heads * (head_dim_k + head_dim_v))
    qkv_out += q_size

    batch_id = tl.load(batch_id_per_token + token_id)
    if batch_id < 0:
        return
    kv_len = tl.load(seq_lens_decoder + batch_id)
    cu_q_len = tl.load(cu_seqlens_q + batch_id)
    token_id_in_this_batch = token_id - cu_q_len + kv_len
    physical_id = tl.load(page_table + batch_id * max_page_per_seq + token_id_in_this_batch // page_size)
    offset = token_id_in_this_batch % page_size

    cache_k += physical_id * num_kv_heads * page_size * head_dim_k
    cache_v += physical_id * num_kv_heads * page_size * head_dim_v

    for head_id in range(num_kv_heads):
        offset_k = tl.arange(0, head_dim_k_next_power_of_2)
        offset_v = tl.arange(0, head_dim_v_next_power_of_2)
        mask_k = offset_k < head_dim_k
        mask_v = offset_v < head_dim_v

        k = tl.load(qkv_out + head_id * head_dim_k + offset_k, mask_k)
        tl.store(cache_k + offset * head_dim_k + offset_k, k, mask_k)

        v = tl.load(qkv_out + num_kv_heads * head_dim_k + head_id * head_dim_v + offset_v, mask_v)
        tl.store(cache_v + offset * head_dim_v + offset_v, v, mask_v)

        cache_k += page_size * head_dim_k
        cache_v += page_size * head_dim_v


def write_cache(
    qkv_out,
    cache_k,
    cache_v,
    cu_seqlens_q,
    seq_lens_decoder,
    batch_id_per_token,
    page_table,
):
    assert qkv_out.ndim == 2
    assert cache_k.ndim == 4
    assert cache_v.ndim == 4
    assert cache_k.shape[:3] == cache_v.shape[:3]
    page_size = cache_k.shape[2]
    head_dim_k = cache_k.shape[-1]
    head_dim_v = cache_v.shape[-1]
    num_kv_heads = cache_k.shape[1]
    q_size = qkv_out.shape[-1] - (head_dim_k + head_dim_v) * num_kv_heads

    M = qkv_out.shape[0]

    grid = (M,)

    write_cache_kernel[grid](
        qkv_out,
        cache_k,
        cache_v,
        cu_seqlens_q,
        seq_lens_decoder,
        batch_id_per_token,
        page_table,
        q_size,
        page_size,
        page_table.shape[1],
        num_kv_heads,
        head_dim_k,
        head_dim_v,
        head_dim_k_next_power_of_2=triton.next_power_of_2(head_dim_k),
        head_dim_v_next_power_of_2=triton.next_power_of_2(head_dim_v),
    )
