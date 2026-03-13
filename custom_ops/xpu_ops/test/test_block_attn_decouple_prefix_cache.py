# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import block_attn, split_rope_kvcache, block_attn_decouple, get_infer_param


def equal_all(x, y):
    elementwise_equality = paddle.equal(x, y)
    is_equal = paddle.any(elementwise_equality).numpy()
    return is_equal

def decouple_block_attn(
    qkv,
    key_cache,
    value_cache,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_scales,
    v_scales,
    k_scales_inv,
    v_scales_inv,
    k_zeros,
    v_zeros,
    shift,
    smooth,
    q_norm_weight,
    k_norm_weight,
    kv_signal_data_cpu,
    cachekv_signal_thread_cpu,
    use_neox_rotary_style,
    rope_3d):
    
    is_cache_int8 = key_cache.dtype == paddle.int8
    has_zp = k_zeros is not None and v_zeros is not None
    is_prefix_cache = len_info_cpu[5] > 0
    
    token_num = qkv.shape[0]
    head_dim = key_cache.shape[3]
    total_num_head = qkv.shape[-1] // head_dim
    kv_num_heads = key_cache.shape[1]
    num_heads = total_num_head - 2 * kv_num_heads
    hidden_dim = num_heads * head_dim

    enc_batch = len_info_cpu[0]
    dec_batch = len_info_cpu[1]
    total_enc_len = len_info_cpu[2]
    total_dec_len = token_num - total_enc_len
    
    q_enc, k_enc, v_enc, q_dec, k_dec, v_dec = split_rope_kvcache(
        qkv,
        key_cache,
        value_cache,
        cum_offsets,
        rotary_embs,
        block_tables,
        len_info_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_map_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        decoder_batch_map_cpu,
        prefix_len_cpu,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_map,
        decoder_context_len,
        decoder_context_len_cache,
        decoder_batch_map,
        prefix_len,
        k_scales,
        v_scales,
        k_zeros,
        v_zeros,
        q_norm_weight,
        k_norm_weight,
        kv_signal_data_cpu,
        cachekv_signal_thread_cpu,
        use_neox_rotary_style,
        rope_3d)
    
    # q = q * k_scales_inv
    if is_cache_int8 and has_zp:
        if enc_batch > 0 and is_prefix_cache:
            origin_shape = q_enc.shape
            q_enc_reshaped = paddle.view(
                q_enc,
                [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
            q_enc_reshaped = q_enc_reshaped * paddle.view(k_scales_inv, [1, kv_num_heads, 1, head_dim])
            q_enc = paddle.view(q_enc_reshaped, origin_shape)
            
            # q_enc_reshaped = paddle.reshape(
            #     q_enc,
            #     [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
            # q_enc_reshaped = q_enc_reshaped * paddle.reshape(k_scales_inv, [1, kv_num_heads, 1, head_dim])
            # q_enc = paddle.reshape(q_enc_reshaped, q_enc.shape)
        if dec_batch > 0:
            origin_shape = q_dec.shape
            q_dec_reshaped = paddle.view(
                q_dec,
                [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
            q_dec_reshaped = q_dec_reshaped * paddle.view(k_scales_inv, [1, kv_num_heads, 1, head_dim])
            q_dec = paddle.view(q_dec_reshaped, origin_shape)
            
            # q_dec_reshaped = paddle.reshape(
            #     q_dec,
            #     [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
            # q_dec_reshaped = q_dec_reshaped * paddle.reshape(k_scales_inv, [1, kv_num_heads, 1, head_dim])
            # q_dec = paddle.reshape(q_dec_reshaped, q_dec.shape)
            
    out = block_attn_decouple(
        q_enc,
        k_enc,
        v_enc,
        q_dec,
        k_dec,
        v_dec,
        key_cache,
        value_cache,
        block_tables,
        prefix_block_tables,
        len_info_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        encoder_batch_map_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        decoder_batch_map_cpu,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        encoder_batch_map,
        decoder_context_len,
        decoder_batch_map,
        k_scales_inv,
        v_scales_inv,
        k_zeros,
        v_zeros)
    
    if enc_batch > 0:
        if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
            sliced_out = out[:total_enc_len, :]
            origin_shape = sliced_out.shape
        if is_cache_int8 and has_zp and is_prefix_cache:
            # out = (out - v_zeros) * v_scales_inv
            out_reshaped = paddle.view(
                sliced_out,
                [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim]) - paddle.view(v_zeros, [1, kv_num_heads, 1, head_dim])
            out_reshaped = out_reshaped * paddle.view(v_scales_inv, [1, kv_num_heads, 1, head_dim])
            sliced_out = paddle.view(out_reshaped, origin_shape)
        if shift:
            sliced_out = sliced_out + shift
        if smooth:
            sliced_out = sliced_out * smooth
        if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
            out[:total_enc_len, :] = sliced_out
            
        # if is_cache_int8 and has_zp and is_prefix_cache:
        #     # out = (out - v_zeros) * v_scales_inv
        #     out_reshaped = paddle.reshape(
        #         out[:total_enc_len, :],
        #         [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim]) - paddle.reshape(v_zeros, [1, kv_num_heads, 1, head_dim])
        #     out_reshaped = out_reshaped * paddle.reshape(v_scales_inv, [1, kv_num_heads, 1, head_dim])
        #     out[:total_enc_len, :] = paddle.reshape(out_reshaped, out[:total_enc_len, :].shape)
        # if shift:
        #     out[:total_enc_len, :] = out[:total_enc_len, :] + shift
        # if smooth:
        #     out[:total_enc_len, :] = out[:total_enc_len, :] * smooth
    if dec_batch > 0:
        if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
            sliced_out = out[total_enc_len:, :]
            origin_shape = sliced_out.shape
        if is_cache_int8 and has_zp:
            # out = (out - v_zeros) * v_scales_inv
            out_reshaped = paddle.view(
                sliced_out,
                [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
            if v_zeros is not None:
                out_reshaped = out_reshaped - paddle.view(v_zeros, [1, kv_num_heads, 1, head_dim])
            out_reshaped = out_reshaped * paddle.view(v_scales_inv, [1, kv_num_heads, 1, head_dim])
            sliced_out = paddle.view(out_reshaped, origin_shape)
        if shift:
            sliced_out = sliced_out + shift
        if smooth:
            sliced_out = sliced_out * smooth
        if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
            out[total_enc_len:, :] = sliced_out
            
        # if is_cache_int8 and has_zp:
        #     # out = (out - v_zeros) * v_scales_inv
        #     out_reshaped = paddle.reshape(
        #         out[total_enc_len:, :],
        #         [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
        #     if v_zeros is not None:
        #         out_reshaped = out_reshaped - paddle.reshape(v_zeros, [1, kv_num_heads, 1, head_dim])
        #     out_reshaped = out_reshaped * paddle.reshape(v_scales_inv, [1, kv_num_heads, 1, head_dim])
        #     out[total_enc_len:, :] = paddle.reshape(out_reshaped, out[total_enc_len:, :].shape)
        # if shift:
        #     out[total_enc_len:, :] = out[total_enc_len:, :] + shift
        # if smooth:
        #     out[total_enc_len:, :] = out[total_enc_len:, :] * smooth
    return out

head_num = 64
kv_head_num = 8
head_dim = 128
seq_len = 128
block_batch = 5
max_block_per_seq = 128
block_size = 64

seq_lens_encoder = paddle.to_tensor([128, 0, 0, 0, 0], dtype="int32")
seq_lens_decoder = paddle.to_tensor([0, 0, 0, 0, 0], dtype="int32")
seq_lens_this_time = paddle.to_tensor([128, 0, 0, 0, 0], dtype="int32")
block_tables = paddle.arange(0, block_batch * max_block_per_seq, dtype="int32")
block_tables = block_tables.reshape((block_batch, max_block_per_seq))
(
    encoder_batch_map,
    decoder_batch_map,
    encoder_batch_idx,
    decoder_batch_idx,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    prefix_len,
    decoder_context_len,
    decoder_context_len_cache,
    prefix_block_tables,
    encoder_batch_map_cpu,
    decoder_batch_map_cpu,
    encoder_batch_idx_cpu,
    decoder_batch_idx_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    prefix_len_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    len_info_cpu,
) = get_infer_param(
    seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_tables, 64
)  # block_size

qkv = paddle.uniform(
    shape=[seq_len, (head_num + 2 * kv_head_num) * head_dim],
    dtype="bfloat16",
    min=-1.0,
    max=1.0,
)

cum_offsets = paddle.zeros(shape=[block_batch], dtype="bfloat16")
rotary_embs = paddle.uniform(shape=[2, 1, 8192, 1, head_dim], dtype="float32", min=-1.0, max=1.0)
key_cache = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)
value_cache = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)
# C8
key_cache_int8 = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="int8",
)
value_cache_int8 = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="int8",
)
scale_tensor_k = paddle.uniform(shape=[kv_head_num * head_dim], dtype="bfloat16", min=1.0, max=1.0)  # max
scale_tensor_v = paddle.uniform(shape=[kv_head_num * head_dim], dtype="bfloat16", min=1.0, max=1.0)  # max
k_quant_scale = 127.0 / scale_tensor_k  # for C8 per channel means 127 / max
v_quant_scale = 127.0 / scale_tensor_v  # for C8 per channel means 127 / max
k_dequant_scale = paddle.cast(scale_tensor_k, dtype="float32")  # for C8 per channel means max
v_dequant_scale = paddle.cast(scale_tensor_v, dtype="float32")  # for C8 per channel means max
k_dequant_scale_zp = 1 / k_quant_scale  # for C8 per channel zp means max
v_dequant_scale_zp = 1 / v_quant_scale  # for C8 per channel zp means max

k_zp = paddle.zeros(shape=[kv_head_num * head_dim], dtype="bfloat16")
v_zp = paddle.zeros(shape=[kv_head_num * head_dim], dtype="bfloat16")

attn_out = block_attn(
    qkv,
    key_cache,
    value_cache,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)
attn_out_decouple = decouple_block_attn(
    qkv,
    key_cache,
    value_cache,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)
is_equal = equal_all(attn_out, attn_out_decouple)
if is_equal:
    print("\ntest_block_attn_decouple PASSED.")
else:
    print("\ntest_block_attn_decouple FAILED.")
assert is_equal

attn_out_C8 = block_attn(
    qkv,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale,
    v_dequant_scale,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)
attn_out_C8_decouple = decouple_block_attn(
    qkv,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale,
    v_dequant_scale,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

is_equal = equal_all(attn_out_C8, attn_out_C8_decouple)
if is_equal:
    print("\ntest_block_attn_decouple C8 PASSED.")
else:
    print("\ntest_block_attn_decouple C8 FAILED.")
assert is_equal

attn_out_C8_zp = block_attn(
    qkv,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale_zp,
    v_dequant_scale_zp,
    k_zp,
    v_zp,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

attn_out_C8_zp_decouple = decouple_block_attn(
    qkv,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale_zp,
    v_dequant_scale_zp,
    k_zp,
    v_zp,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

is_equal = equal_all(attn_out_C8_zp, attn_out_C8_zp_decouple)
if is_equal:
    print("\ntest_block_attn_decouple C8 zp PASSED.")
else:
    print("\ntest_block_attn_decouple C8 zp FAILED.")
assert is_equal

# prefix cache : hit 71 tokens
hit_prefix_len = 71
seq_lens_encoder = paddle.to_tensor([seq_len - hit_prefix_len, 0, 0, 0, 0], dtype="int32")
# 71 means prefix len
seq_lens_decoder = paddle.to_tensor([hit_prefix_len, 0, 0, 0, 0], dtype="int32")
(
    encoder_batch_map,
    decoder_batch_map,
    encoder_batch_idx,
    decoder_batch_idx,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    prefix_len,
    decoder_context_len,
    decoder_context_len_cache,
    prefix_block_tables,
    encoder_batch_map_cpu,
    decoder_batch_map_cpu,
    encoder_batch_idx_cpu,
    decoder_batch_idx_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    prefix_len_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    len_info_cpu,
) = get_infer_param(
    seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_tables, 64
)  # block_size
qkv_prefix = qkv[hit_prefix_len:]

attn_out_prefix_cache = block_attn(
    qkv_prefix,
    key_cache,
    value_cache,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

attn_out_prefix_cache_decouple = decouple_block_attn(
    qkv_prefix,
    key_cache,
    value_cache,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

is_equal = equal_all(attn_out_prefix_cache, attn_out_prefix_cache_decouple)
if is_equal:
    print("\ntest_block_attn_decouple prefix_cache PASSED.")
else:
    print("\ntest_block_attn_decouple prefix_cache FAILED.")
assert is_equal

attn_out_C8_prefix_cache = block_attn(
    qkv_prefix,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale,
    v_dequant_scale,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

attn_out_C8_prefix_cache_decouple = decouple_block_attn(
    qkv_prefix,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale,
    v_dequant_scale,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

is_equal = equal_all(attn_out_C8_prefix_cache, attn_out_C8_prefix_cache_decouple)
if is_equal:
    print("\ntest_block_attn_decouple prefix_cache C8 PASSED.")
else:
    print("\ntest_block_attn_decouple prefix_cache C8 FAILED.")
assert is_equal

attn_out_C8_zp_prefix_cache = block_attn(
    qkv_prefix,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale_zp,
    v_dequant_scale_zp,
    k_zp,
    v_zp,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

attn_out_C8_zp_prefix_cache_decouple = decouple_block_attn(
    qkv_prefix,
    key_cache_int8,
    value_cache_int8,
    cum_offsets,
    rotary_embs,
    block_tables,
    prefix_block_tables,
    len_info_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    encoder_batch_map_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    decoder_batch_map_cpu,
    prefix_len_cpu,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    encoder_batch_map,
    decoder_context_len,
    decoder_context_len_cache,
    decoder_batch_map,
    prefix_len,
    k_quant_scale,
    v_quant_scale,
    k_dequant_scale_zp,
    v_dequant_scale_zp,
    k_zp,
    v_zp,
    None,
    None,
    None,
    None,
    None,
    None,
    False,
    False
)

is_equal = equal_all(attn_out_C8_zp_prefix_cache, attn_out_C8_zp_prefix_cache_decouple)
if is_equal:
    print("\ntest_block_attn_decouple prefix_cache C8 zp PASSED.")
else:
    print("\ntest_block_attn_decouple prefix_cache C8 zp FAILED.")
assert is_equal

print("\n")
print("-- C16 prefix cache test --")
print("attn_out_decouple[hit_prefix_len:]'s mean:", attn_out_decouple[hit_prefix_len:].mean().item())
print("attn_out_prefix_cache_decouple's mean: ", attn_out_prefix_cache_decouple.mean().item())
attn_out_prefix_cache_np = attn_out_prefix_cache_decouple.astype("float32").numpy()
attn_out_np = attn_out_decouple[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_prefix_cache_np, attn_out_np, rtol=1e-2, atol=1e-3
), f"C16 prefix cache != No prefix cache,\n attn_out_decouple[hit_prefix_len:]: {attn_out_np},\nattn_out_prefix_cache_decouple: {attn_out_prefix_cache_np}"


print("\n-- C8 per channel prefix cache test --")
print(
    "attn_out_C8_decouple[hit_prefix_len:]'s mean:",
    attn_out_C8_decouple[hit_prefix_len:].mean().item(),
)
print("attn_out_C8_prefix_cache_decouple's mean: ", attn_out_C8_prefix_cache_decouple.mean().item())
attn_out_C8_prefix_cache_np = attn_out_C8_prefix_cache_decouple.astype("float32").numpy()
attn_out_C8_np = attn_out_C8_decouple[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_C8_prefix_cache_np, attn_out_C8_np, rtol=1e-1, atol=1e-2
), f"C8 per channel prefix cache != No prefix cache,\n attn_out_C8_decouple[hit_prefix_len:]: {attn_out_C8_np},\nattn_out_C8_prefix_cache_decouple: {attn_out_C8_prefix_cache_np}"

print("\n-- C8 per channel zp prefix cache test --")
print(
    "attn_out_C8_zp_decouple[hit_prefix_len:]'s mean:",
    attn_out_C8_zp_decouple[hit_prefix_len:].mean().item(),
)
print(
    "attn_out_C8_zp_prefix_cache_decouple's mean: ",
    attn_out_C8_zp_prefix_cache_decouple.mean().item(),
)
attn_out_C8_zp_prefix_cache_np = attn_out_C8_zp_prefix_cache_decouple.astype("float32").numpy()
attn_out_C8_zp_np = attn_out_C8_zp_decouple[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_C8_zp_prefix_cache_np, attn_out_C8_zp_np, rtol=1e-1, atol=1e-2
), f"C8 per channel zp prefix cache != No prefix cache,\n attn_out_C8_zp_decouple[hit_prefix_len:]: {attn_out_C8_zp_np},\nattn_out_C8_zp_prefix_cache_decouple: {attn_out_C8_zp_prefix_cache_np}"
