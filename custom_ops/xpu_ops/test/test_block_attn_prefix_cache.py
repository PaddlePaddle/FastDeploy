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

from fastdeploy.model_executor.ops.xpu import block_attn, get_infer_param

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
)
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
)
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
)

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
)

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
)

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
)
print("-- C16 prefix cache test --")
print("attn_out[hit_prefix_len:]'s mean:", attn_out[hit_prefix_len:].mean().item())
print("attn_out_prefix_cache's mean: ", attn_out_prefix_cache.mean().item())
attn_out_prefix_cache_np = attn_out_prefix_cache.astype("float32").numpy()
attn_out_np = attn_out[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_prefix_cache_np, attn_out_np, rtol=1e-2, atol=1e-3
), f"C16 prefix cache != No prefix cache,\n attn_out[hit_prefix_len:]: {attn_out_np},\nattn_out_prefix_cache: {attn_out_prefix_cache_np}"


print("\n-- C8 per channel prefix cache test --")
print(
    "attn_out_C8[hit_prefix_len:]'s mean:",
    attn_out_C8[hit_prefix_len:].mean().item(),
)
print("attn_out_C8_prefix_cache's mean: ", attn_out_C8_prefix_cache.mean().item())
attn_out_C8_prefix_cache_np = attn_out_C8_prefix_cache.astype("float32").numpy()
attn_out_C8_np = attn_out_C8[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_C8_prefix_cache_np, attn_out_C8_np, rtol=1e-1, atol=1e-2
), f"C8 per channel prefix cache != No prefix cache,\n attn_out_C8[hit_prefix_len:]: {attn_out_C8_np},\nattn_out_C8_prefix_cache: {attn_out_C8_prefix_cache_np}"

print("\n-- C8 per channel zp prefix cache test --")
print(
    "attn_out_C8_zp[hit_prefix_len:]'s mean:",
    attn_out_C8_zp[hit_prefix_len:].mean().item(),
)
print(
    "attn_out_C8_zp_prefix_cache's mean: ",
    attn_out_C8_zp_prefix_cache.mean().item(),
)
attn_out_C8_zp_prefix_cache_np = attn_out_C8_zp_prefix_cache.astype("float32").numpy()
attn_out_C8_zp_np = attn_out_C8_zp[hit_prefix_len:].astype("float32").numpy()
assert np.allclose(
    attn_out_C8_zp_prefix_cache_np, attn_out_C8_zp_np, rtol=1e-1, atol=1e-2
), f"C8 per channel zp prefix cache != No prefix cache,\n attn_out_C8_zp[hit_prefix_len:]: {attn_out_C8_zp_np},\nattn_out_C8_zp_prefix_cache: {attn_out_C8_zp_prefix_cache_np}"
