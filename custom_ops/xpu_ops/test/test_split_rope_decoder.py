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

import os
import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import block_attn, split_rope_kvcache, block_attn_decouple, get_infer_param

if os.getenv("decoder_splice", None) == "1":
    only_decoder = True
else:
    only_decoder = False


def all_close(x, y, rtol, atol):
    if x.dtype == paddle.bfloat16:
        x = paddle.cast(x, paddle.float32)
        y = paddle.cast(y, paddle.float32)
    x_np = x.numpy()
    y_np = y.numpy()
    return np.allclose(x_np, y_np, rtol=rtol, atol=atol)


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
        
    return q_enc, k_enc, v_enc, q_dec, k_dec, v_dec, out

head_num = 2
kv_head_num = 2
head_dim = 64
if only_decoder:
    seq_len = 1
    encoder_seq_len = 0
    decoder_seq_len = 1
else:
    seq_len = 128
    encoder_seq_len = 128
    decoder_seq_len = 0
block_batch = 5
max_block_per_seq = 128
block_size = 64

seq_lens_encoder = paddle.to_tensor([encoder_seq_len, 0, 0, 0, 0], dtype="int32")
seq_lens_decoder = paddle.to_tensor([decoder_seq_len, 0, 0, 0, 0], dtype="int32")
seq_lens_this_time = paddle.to_tensor([seq_len, 0, 0, 0, 0], dtype="int32")
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

qkv_shape = [seq_len, (head_num + 2 * kv_head_num) * head_dim]
rotary_embs_shape = [2, 1, 8192, 1, head_dim]
# qkv = paddle.uniform(
#     shape=qkv_shape,
#     dtype="bfloat16",
#     min=-1.0,
#     max=1.0,
# )
# rotary_embs = paddle.uniform(shape=rotary_embs_shape, dtype="float32", min=-1.0, max=1.0)
qkv = paddle.full(qkv_shape, 2.0, dtype=paddle.bfloat16)
rotary_embs = paddle.full(rotary_embs_shape, 3.0, dtype=paddle.float32)

cum_offsets = paddle.zeros(shape=[block_batch], dtype="bfloat16")
key_cache = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)
value_cache = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)
key_cache_decouple = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)
value_cache_decouple = paddle.zeros(
    shape=[block_batch * max_block_per_seq, kv_head_num, block_size, head_dim],
    dtype="bfloat16",
)

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
q_enc, k_enc, v_enc, q_dec, k_dec, v_dec, attn_out_decouple = decouple_block_attn(
    qkv,
    key_cache_decouple,
    value_cache_decouple,
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

rtol = 1e-6
atol = rtol

is_equal = all_close(value_cache.to(paddle.float32), value_cache_decouple.to(paddle.float32), rtol, atol)
if is_equal:
    print("\ntest_block_attn_decouple value_cache PASSED.")
else:
    print(f"\ntest_block_attn_decouple value_cache FAILED, with rtol = {rtol}, atol = {atol}.")
    print(f"mark_debug: value_cache: {value_cache.to(paddle.float32)}")
    print(f"mark_debug: value_cache_decouple: {value_cache_decouple.to(paddle.float32)}")
assert is_equal

is_equal = all_close(key_cache.to(paddle.float32), key_cache_decouple.to(paddle.float32), rtol, atol)
if is_equal:
    print("\ntest_block_attn_decouple key_cache PASSED.")
else:
    print(f"\ntest_block_attn_decouple key_cache FAILED, with rtol = {rtol}, atol = {atol}.")
    print(f"mark_debug: key_cache: {key_cache.to(paddle.float32)}")
    print(f"mark_debug: key_cache_decouple: {key_cache_decouple.to(paddle.float32)}")
assert is_equal

is_equal = all_close(attn_out.to(paddle.float32), attn_out_decouple.to(paddle.float32), rtol, atol)
if is_equal:
    print("\ntest_block_attn_decouple attn_out PASSED.")
else:
    print(f"\ntest_block_attn_decouple attn_out FAILED, with rtol = {rtol}, atol = {atol}.")
    print(f"mark_debug: attn_out: {attn_out.to(paddle.float32)}")
    print(f"mark_debug: attn_out_decouple: {attn_out_decouple.to(paddle.float32)}")
assert is_equal