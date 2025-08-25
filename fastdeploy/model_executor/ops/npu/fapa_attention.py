import inspect

import paddle
from paddle.base import core


def fused_fapa_attention_npu(
    qkv_weight,
    rope_emb,
    cache_k,
    cache_v,
    seq_lens_encoder,
    seq_lens_decoder,
    block_tables,
    q_num_head,
    kv_num_head,
    head_dim,
    max_seq_len,
    block_size,
):

    rope_emb=paddle.cast(rope_emb,paddle.bfloat16)
    cos,sin=paddle.chunk(rope_emb, chunks=2, axis=-1)

    out = core.eager._run_custom_op(
        "fused_fapa_attention_op",
        qkv_weight,
        cos,
        sin,
        cache_k,
        cache_v,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        q_num_head,
        kv_num_head,
        head_dim,
        1,
        max_seq_len,
        block_size,
        False,  # use_neox_rotary_style
        False,
    )
    return out
