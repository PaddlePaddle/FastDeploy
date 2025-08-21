import math
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
)

paddle.seed(0)

max_seq_len = 32768
encoder_max_partition_size = max_seq_len
max_partition_size = max_seq_len

max_dec_len = 1024
bsz = 64
run_time = 10
warm_up = 2
block_size = 64
head_dim = 128
num_q_head = 20
num_kv_head = 4
dtype = "bfloat16"

rope_3d = False
use_neox_rotary_style = False
CURRENT_Q = [None]
TOTAL_K = []
TOTAL_V = []


def split_qkv(qkv, bsz, seq_len, num_q_head, num_kv_head, head_dim):
    # [token_num, (num_q_head + 2 * num_kv_head) * head_dim]
    qkv = qkv.reshape([bsz, seq_len, -1, head_dim])
    q = qkv[:, :, :num_q_head, :]
    # [bsz,  seq_len, num_q_head, head_dim]
    CURRENT_Q[0] = q

    # [bsz,  seq_len, num_kv_head, head_dim]
    k = qkv[:, :, num_q_head : num_q_head + num_kv_head, :]
    TOTAL_K.append(k)

    # [bsz,  seq_len, num_kv_head, head_dim]
    v = qkv[:, :, num_q_head + num_kv_head :, :]
    TOTAL_V.append(v)


def get_padding_offset(bsz, seq_lens_this_time, seq_lens_decoder):
    batch_id_per_token = []
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_seq_len_q = 0
    cum_seq_len_k = 0
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        seq_len_dec_now = seq_lens_decoder[i]
        for j in range(seq_len_now):
            batch_id_per_token.append(i)
        cum_seq_len_q += seq_len_now
        cum_seq_len_k += seq_len_now + seq_len_dec_now
        cu_seqlens_q[i + 1] = cum_seq_len_q
        cu_seqlens_k[i + 1] = cum_seq_len_k
    return paddle.to_tensor(batch_id_per_token, dtype="int32"), cu_seqlens_q, cu_seqlens_k


# block_table
block_num_per_seq = (max_seq_len + block_size - 1) // block_size
max_block_num = block_num_per_seq * bsz
cache_shape = (
    max_block_num,
    num_kv_head,
    block_size,
    head_dim,
)

cache_k = paddle.zeros(shape=cache_shape).astype(dtype)
cache_v = paddle.zeros(shape=cache_shape).astype(dtype)

block_tables = paddle.zeros(shape=(bsz, block_num_per_seq), dtype="int32")

free_list = list(range(max_block_num - 1, -1, -1))

for i in range(bsz):
    need_block_num = (max_seq_len + block_size - 1) // block_size
    for j in range(need_block_num):
        block_id = free_list.pop()
        block_tables[i, j] = block_id


def ref_attention(q, k, v, num_q_head, num_kv_head, head_dim, mask):
    q = q.transpose([0, 2, 1, 3])
    if len(k) > 1:
        k = paddle.concat(k, axis=1)
    else:
        k = k[0]
    k = k.transpose([0, 2, 1, 3])
    if len(v) > 1:
        v = paddle.concat(v, axis=1)
    else:
        v = v[0]
    v = v.transpose([0, 2, 1, 3])
    total_len = k.shape[2]

    scores = q.reshape([bsz, num_kv_head, -1, head_dim]) @ k.transpose([0, 1, 3, 2]) * (1.0 / math.sqrt(head_dim))
    scores = scores.reshape([bsz, num_q_head, -1, total_len])

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,q_len,kv_len]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [bsz,1,q_len,kv_len]
        scores = paddle.add(scores, mask)
    weights = F.softmax(scores, axis=-1)

    o = weights.reshape([bsz, num_kv_head, -1, total_len]) @ v
    return o.reshape([bsz, num_q_head, -1, head_dim]).transpose([0, 2, 1, 3]).reshape([-1, num_q_head, head_dim])


def clear_param():
    global CURRENT_Q, TOTAL_K, TOTAL_V
    CURRENT_Q = [None]
    TOTAL_K = []
    TOTAL_V = []


def test_append_c16_attention(q_len, kv_len, prefill=False, attn_mask=None):
    if prefill:
        seq_lens_enc = [
            q_len,
        ] * bsz
    else:
        seq_lens_enc = [
            0,
        ] * bsz

    seq_lens_dec = [
        kv_len,
    ] * bsz
    seq_lens_cur = [
        q_len,
    ] * bsz
    token_num = sum(seq_lens_cur)
    decoder_step_token_num = 1 if prefill else q_len

    seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_cur, "int32")
    seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")

    batch_id_per_token, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, seq_lens_this_time, seq_lens_decoder)

    # random data
    qkv_varlen_shape = [token_num, (num_q_head + 2 * num_kv_head) * head_dim]

    rotary_embs_shape = [2, 1, max_seq_len, 1, head_dim if use_neox_rotary_style else head_dim // 2]
    # qkv_bias_shape = [num_q_head + 2 * num_kv_head, head_dim]

    qkv = paddle.randn(shape=qkv_varlen_shape).astype(dtype)

    # save q, k, v for ref
    split_qkv(qkv, bsz, q_len, num_q_head, num_kv_head, head_dim)

    rotary_embs = paddle.randn(shape=rotary_embs_shape).astype("float32")
    rotary_embs[0, :, :, :, :] = 1
    rotary_embs[1, :, :, :, :] = 0

    # qkv_scale = None
    # qkv_bias = None

    cache_k_scale = None
    cache_v_scale = None
    cache_k_out_scale = None
    cache_v_out_scale = None
    # shift_bias = None
    # smooth_weight = None

    encoder_block_shape_q = 64
    decoder_block_shape_q = 16

    decode_max_tile_size = (
        bsz
        * (decoder_step_token_num * (num_q_head // num_kv_head) + decoder_block_shape_q - 1)
        / decoder_block_shape_q
    )
    decoder_batch_ids = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
    decoder_tile_ids_per_batch = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
    decoder_num_blocks = paddle.full([1], 0, dtype="int32").pin_memory()
    max_len_tensor_cpu = paddle.full([8], 0, dtype="int32").cpu()
    paddle.device.synchronize()
    (
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        max_len_kv,
    ) = get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks,
        max_len_tensor_cpu,
        encoder_block_shape_q,
        decoder_block_shape_q,
        num_q_head // num_kv_head,
        block_size,
        decoder_step_token_num,
    )
    s_time = 0
    for i in range(run_time + warm_up):
        if i == warm_up:
            s_time = time.time()
        out = append_attention(
            qkv,
            cache_k,
            cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_len_tensor_cpu,
            max_len_kv,
            rotary_embs,
            attn_mask,  # attn_mask
            None,
            None,
            cache_k_scale,
            cache_v_scale,
            cache_k_out_scale,
            cache_v_out_scale,
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,
            None,
            None,
            None,
            None,
            None,
            1e-6,
            "bf16",
            "none",  # cache_quant_type
            use_neox_rotary_style,
            rope_3d,
            max_seq_len,
            0.0,
            0.0,
            -1.0,  # out_linear_in_scale
            encoder_block_shape_q,  # encoder_block_shape_q
            decoder_block_shape_q,  # decoder_block_shape_q
            max_partition_size,  # max_partition_size
            encoder_max_partition_size,  # encoder_max_partition_size
            decoder_step_token_num,  # speculate_max_draft_token_num
            True,  # causal
            decoder_step_token_num > 1,  # speculate_decoder
        )
        paddle.device.synchronize()
    e_time = time.time()
    print(f"mean infer time: {np.mean((e_time - s_time) * 1000 / run_time):.2f}")
    return out[0].reshape([token_num, num_q_head, head_dim])


def test_naive_speculative_decoding(num_q_head, num_kv_head, head_dim):
    prefill_len = 8192
    dec_len_q = 5
    total_len = prefill_len + dec_len_q
    mask = paddle.tril(paddle.ones((bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
    mask = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))
    test_append_c16_attention(prefill_len, 0, True)
    dec_out = test_append_c16_attention(dec_len_q, prefill_len, False)

    ref_out = ref_attention(CURRENT_Q[0], TOTAL_K, TOTAL_V, num_q_head, num_kv_head, head_dim, mask)
    np.testing.assert_allclose(
        ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
    )


def test_mask(num_q_head, num_kv_head, head_dim):
    prefill_len = 8192
    dec_len_q = 5
    total_len = prefill_len + dec_len_q
    mask = paddle.tril(paddle.ones((bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
    mask_ref = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))

    mask_append_attn = mask[:, :, prefill_len:]
    mask_append_attn = paddle.where(
        mask_append_attn == 1,
        paddle.full_like(mask_append_attn, fill_value=False, dtype=bool),
        paddle.full_like(mask_append_attn, fill_value=True, dtype=bool),
    )

    test_append_c16_attention(prefill_len, 0, True)
    dec_out = test_append_c16_attention(dec_len_q, prefill_len, False, mask_append_attn)

    ref_out = ref_attention(CURRENT_Q[0], TOTAL_K, TOTAL_V, num_q_head, num_kv_head, head_dim, mask_ref)

    np.testing.assert_allclose(
        ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
    )


def test_tree_mask(num_q_head, num_kv_head, head_dim):
    # tree
    #       [N,   N+1,    N+1,    N+2,    N+2]
    # N     [0,   -inf,   -inf,   -inf,   -inf]
    # N+1   [0,   0,      -inf,   -inf,   -inf]
    # N+1   [0,   -inf,   0,      -inf,   -inf]
    # N+2   [0,   0,      -inf,   0,      -inf]
    # N+2   [0,   -inf,   0,      -inf,   0]
    prefill_len = 8192
    dec_len_q = 5
    total_len = prefill_len + dec_len_q
    mask = paddle.tril(paddle.ones((bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
    mask[:, 2, prefill_len + 1] = 0
    mask[:, 3, prefill_len + 2] = 0
    mask[:, 4, prefill_len + 1] = 0
    mask[:, 4, prefill_len + 3] = 0

    mask_ref = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))

    mask_append_attn = mask[:, :, prefill_len:]
    mask_append_attn = paddle.where(
        mask_append_attn == 1,
        paddle.full_like(mask_append_attn, fill_value=False, dtype=bool),
        paddle.full_like(mask_append_attn, fill_value=True, dtype=bool),
    )

    test_append_c16_attention(prefill_len, 0, True)
    dec_out = test_append_c16_attention(dec_len_q, prefill_len, False, mask_append_attn)
    ref_out = ref_attention(CURRENT_Q[0], TOTAL_K, TOTAL_V, num_q_head, num_kv_head, head_dim, mask_ref)
    np.testing.assert_allclose(
        ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
    )


if __name__ == "__main__":

    test_naive_speculative_decoding(num_q_head, num_kv_head, head_dim)
    clear_param()

    test_mask(num_q_head, num_kv_head, head_dim)
    clear_param()

    test_tree_mask(num_q_head, num_kv_head, head_dim)
