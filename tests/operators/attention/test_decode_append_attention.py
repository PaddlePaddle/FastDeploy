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

import copy
import random
import unittest

import numpy as np
import paddle
from paddle.incubate.nn.functional import fused_rms_norm

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    config_for_attention,
    decode_append_attention,
    decoder_write_cache_with_rope,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    pre_cache_len_concat,
)

seed = 1000

random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


class RopeEmbedding:
    def __init__(self, use_neox_rotary_style=False):
        self.use_neox_rotary_style = use_neox_rotary_style
        self.base = 10000

    def get_neox_style_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def get_rotary_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, D/2]
        emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, head_dim // 2))
        # shape: [B, S, 1, D/2]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def _apply_rope(self, rotary_emb, q, k, cache_len):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        # sin, cos = paddle.chunk(rp, 2, axis=-1)
        seq, head_dim = q.shape[2], q.shape[3]
        cos, sin = paddle.chunk(rotary_emb, 2, axis=0)
        cos = cos[:, :, cache_len : cache_len + seq, ...]
        sin = sin[:, :, cache_len : cache_len + seq, ...]
        cos = paddle.squeeze(cos, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        sin = paddle.squeeze(sin, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]

        if self.use_neox_rotary_style:
            sin_pos = sin
            cos_pos = cos
            # NeoX Stype：前后半部分分块旋转
            rotate_half_q = paddle.reshape(
                paddle.concat(
                    [
                        -q[:, :, :, q.shape[-1] // 2 :],
                        q[:, :, :, : q.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(q),
            )
            rotate_half_k = paddle.reshape(
                paddle.concat(
                    [
                        -k[:, :, :, k.shape[-1] // 2 :],
                        k[:, :, :, : k.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(k),
            )
        else:
            sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), [1, 1, seq, head_dim])
            # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), [1, 1, seq, head_dim])
            # GPT Stype：奇偶位置分块旋转
            rotate_half_q = paddle.reshape(
                paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
                paddle.shape(q),
            )
            rotate_half_k = paddle.reshape(
                paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
                paddle.shape(k),
            )

        query = paddle.add(paddle.multiply(q, cos_pos), paddle.multiply(rotate_half_q, sin_pos))

        key = paddle.add(paddle.multiply(k, cos_pos), paddle.multiply(rotate_half_k, sin_pos))

        return paddle.cast(query, q.dtype), paddle.cast(key, k.dtype)


def create_attn_mask(mask_type, batch_size, seq_lens, pre_cache_length=0, sliding_window=0):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        # [batch_size, 1, max_seq_len, max_seq_len + pre_cache_length],
        [batch_size, 1, max_seq_len, max_seq_len],
        dtype=mask_type,
    )
    mask[:, :, :, :pre_cache_length] = 1
    for i in range(batch_size):
        seq_len = seq_lens[i]
        ones_tensor = paddle.ones(shape=(seq_len, seq_len), dtype=mask_type)
        if sliding_window <= 0:
            mask[i, 0, :seq_len, :seq_len] = (paddle.tril(ones_tensor) - 1) * 1e4
        else:
            tmp_triu = paddle.triu(ones_tensor, -(sliding_window - 1))
            mask[i, 0, :seq_len, :seq_len] = (paddle.tril(ones_tensor) * tmp_triu - 1) * 1e4
    return mask


def naive_attention_impl(
    query,
    key,
    value,
    pre_key=None,
    pre_value=None,
    mask=None,
    scale=1.0,
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    use_cachekv_int8="None",
    q_norm_weight=None,
    k_norm_weight=None,
    sinks=None,
):
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    if pre_key is not None:
        pre_key = pre_key.reshape([batch, kv_head, 1, -1, head_dim])
        pre_key = paddle.tile(pre_key, [1, 1, heads // kv_head, 1, 1])
        pre_key = pre_key.reshape([batch, heads, -1, head_dim])
        key = paddle.concat([pre_key, key], axis=2)

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])

    if pre_value is not None:
        pre_value = pre_value.reshape([batch, kv_head, 1, -1, head_dim])
        pre_value = paddle.tile(pre_value, [1, 1, heads // kv_head, 1, 1])
        pre_value = pre_value.reshape([batch, heads, -1, head_dim])
        value = paddle.concat([pre_value, value], axis=2)

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    if mask is not None:
        attention = attention + mask

    if sinks is not None:
        kv_len = attention.shape[-1]
        sinks_tiled = sinks.unsqueeze([0, 2, 3]).expand([batch, heads, seq_len, 1])
        attention = paddle.concat([attention, sinks_tiled], axis=-1)
        softmax_result = paddle.nn.functional.softmax(attention, -1)[:, :, :, :kv_len]
    else:
        softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(paddle.cast(softmax_result, dtype=value.dtype), value)
    return result


def get_padding_offset(bsz, seq_lens_this_time):
    token_num = paddle.sum(seq_lens_this_time)
    batch_id_per_token = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    index = 0
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i].item()
        for j in range(seq_len_now):
            batch_id_per_token[index] = i
            index += 1
        cu_seqlens_q[i + 1] = index
        cu_seqlens_k[i + 1] = index
    return batch_id_per_token, cu_seqlens_q, cu_seqlens_k


def remove_padding(seq_lens, cu_seq_lens, inputs, token_num):
    bsz, num_head, seq_len, head_dim = inputs.shape
    output = paddle.zeros(shape=[token_num, num_head * head_dim], dtype=inputs.dtype)
    inputs = inputs.transpose([0, 2, 1, 3]).reshape([bsz, seq_len, -1])
    for i in range(bsz):
        seq_len_now = seq_lens[i]
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        output[start_idx:end_idx, :] = inputs[i, :seq_len_now, :]
    return output


def get_qkv_and_qkv_concat_tensor(bs, q_num_head, kv_num_head, seq_len, head_dim, place, dtype):
    query = np.random.random([bs, q_num_head, seq_len, head_dim])
    q = paddle.to_tensor(query, place=place, dtype=dtype, stop_gradient=False) - 0.5
    key = np.random.random([bs, kv_num_head, seq_len, head_dim])
    k = paddle.to_tensor(key, place=place, dtype=dtype, stop_gradient=False) - 0.5
    value = np.random.random([bs, kv_num_head, seq_len, head_dim])
    v = paddle.to_tensor(value, place=place, dtype=dtype, stop_gradient=False) - 0.5
    token_num = bs * seq_len

    qkv = paddle.concat(
        [
            q.transpose([0, 2, 1, 3]).reshape([token_num, q_num_head * head_dim]),
            k.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * head_dim]),
            v.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * head_dim]),
        ],
        axis=1,
    ).reshape([token_num, -1])
    return q, k, v, qkv


class TestDecodeAppendAttention(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 1
        self.max_tokens_per_batch = 1
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()

    def init_tensor(self):
        # seq_lens
        if self.seq_len_dec is None:
            self.seq_lens_dec = [
                self.cache_len,
            ] * self.batch_size
        else:
            self.batch_size = len(self.seq_lens_dec)
        self.seq_lens_decoder = paddle.to_tensor(
            self.seq_lens_dec,
            "int32",
        )
        if self.seq_lens_this_time is None:
            self.seq_lens_this_time = [
                self.max_tokens_per_batch,
            ] * self.batch_size
        self.token_num = sum(self.seq_lens_this_time)
        self.seq_lens_this_time = paddle.to_tensor(self.seq_lens_this_time, "int32")

        self.seq_lens_enc = [0] * self.batch_size

        self.seq_lens_encoder = paddle.to_tensor(
            self.seq_lens_enc,
            "int32",
        )

        # self.qkv = paddle.rand([self.token_num, (self.q_num_head + 2 * self.kv_num_head) * self.head_dim], dtype=self.dtype)
        self.q, self.k, self.v, self.qkv = get_qkv_and_qkv_concat_tensor(
            self.batch_size,
            self.q_num_head,
            self.kv_num_head,
            self.max_tokens_per_batch,
            self.head_dim,
            self.place,
            self.dtype,
        )
        self.qkv = paddle.to_tensor(self.qkv, dtype=self.dtype)

        # qk_norm
        self.q_norm_weight = None
        self.k_norm_weight = None
        if self.use_qk_norm:
            q_norm_weight_np = np.random.random([self.head_dim]) / 10
            k_norm_weight_np = np.random.random([self.head_dim]) / 10
            self.q_norm_weight = paddle.to_tensor(q_norm_weight_np, dtype="float32")
            self.k_norm_weight = paddle.to_tensor(k_norm_weight_np, dtype="float32")

        # rotary embedding
        self.rope = RopeEmbedding(False)
        tmp_position_ids = paddle.arange(self.max_model_len).reshape((1, -1))
        self.rotary_embs = self.rope.get_rotary_position_embedding(tmp_position_ids, self.head_dim)

        # block_table
        self.block_num_per_seq = (self.max_model_len + self.block_size - 1) // self.block_size
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))
        self.block_tables = paddle.zeros(shape=(self.batch_size, self.block_num_per_seq), dtype="int32")
        for i in range(self.batch_size):
            need_block_num = (self.max_model_len + self.block_size - 1) // self.block_size
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()

        # cache_kv && scale
        self.cache_shape = (
            self.max_block_num,
            self.kv_num_head,
            self.block_size,
            self.head_dim,
        )

        if self.use_dynamic_quant:
            self.cache_scale_shape = (
                self.max_block_num,
                self.kv_num_head,
                self.block_size,
            )
            self.cache_k = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_v = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_k_T = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
            self.cache_v_T = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
            self.cache_k_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.cache_v_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.cache_k_out_scale = None
            self.cache_k_out_scale = None
        else:
            self.cache_k_scale = self.quant_max_bound / self.k.transpose([1, 0, 2, 3]).reshape(
                [self.kv_num_head, -1]
            ).abs().max(axis=1)
            self.cache_v_scale = self.quant_max_bound / self.v.transpose([1, 0, 2, 3]).reshape(
                [self.kv_num_head, -1]
            ).abs().max(axis=1)

            self.cache_k_out_scale = (
                self.k.transpose([1, 0, 2, 3]).reshape([self.kv_num_head, -1]).max(axis=1) / self.quant_max_bound
            )
            self.cache_v_out_scale = (
                self.v.transpose([1, 0, 2, 3]).reshape([self.kv_num_head, -1]).max(axis=1) / self.quant_max_bound
            )

            self.cache_k = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_v = paddle.zeros(shape=self.cache_shape, dtype="uint8")

        (
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, self.seq_lens_this_time)

        # mask
        if self.mask_matrix:
            self.attn_mask = create_attn_mask(
                self.dtype,
                self.batch_size,
                [
                    self.max_tokens_per_batch,
                ]
                * self.batch_size,
                sliding_window=self.sliding_window,
            )
        else:
            self.attn_mask = None

        # mask offset
        self.mask_offset = None
        if self.use_mask_offset:
            self.mask_offset = paddle.full(self.batch_size * 2, 0, "int32")
            for i in range(self.batch_size):
                self.mask_offset[i * 2] = 0
                self.mask_offset[i * 2 + 1] = self.seq_lens_dec[i] + 1

        if self.use_sinks:
            self.sinks = paddle.to_tensor(
                np.random.random([self.q_num_head]), place=self.place, dtype=self.dtype, stop_gradient=False
            )
        else:
            self.sinks = None

        # buffer
        self.buffer = {}
        min_chunk_size = 128
        max_num_chunk = (self.max_model_len + min_chunk_size - 1) // min_chunk_size
        self.group_size = self.q_num_head // self.kv_num_head
        q_tile_size = 16 if self.max_tokens_per_batch * self.group_size <= 16 else 32
        q_tile_num = (self.max_tokens_per_batch * self.group_size + q_tile_size - 1) // q_tile_size
        self.buffer["max_len_tensor_cpu"] = paddle.full([6], 0, dtype="int32").cpu()
        # block_indices: Launched block's indices with 4 dimensions [batch_idx, kv_head_idx, chunk_idx, q_tile_idx] in decode append attention backend
        self.buffer["block_indices"] = paddle.full(
            [self.batch_size * self.kv_num_head * max_num_chunk * q_tile_num, 4], 0, dtype="int32"
        )
        # num_blocks: Number of Launched blocks in decode append attention backend, researched by config_for_attention op
        self.buffer["num_blocks"] = paddle.full([1], 0, dtype="int32")
        # chunk_size: Chunk size for split kv cache in decode append attention backend, researched by config_for_attention op
        self.buffer["chunk_size"] = paddle.full([1], 0, dtype="int32")
        # tmp_workspace: Workspace tensor for temporary store the result before merging in decode append attention backend
        self.buffer["tmp_workspace"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head * self.head_dim],
            0,
            dtype=self.dtype,
        )
        # tmp_m: Tmp_m tensor for temporary store the max value before merging in decode append attention backend
        self.buffer["tmp_m"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head], 0, dtype="float32"
        )
        # tmp_d: Tmp_d tensor for temporary store the exponential sum before merging in decode append attention backend
        self.buffer["tmp_d"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head], 0, dtype="float32"
        )

    def apply_qk_norm(self, head_dim, dtype, q, k):
        bs, q_num_head, seq_len, head_dim = q.shape
        _, kv_num_head, _, _ = k.shape

        q = q.reshape([-1, head_dim])
        k = k.reshape([-1, head_dim])
        q = fused_rms_norm(q.astype("float32"), self.q_norm_weight, None, self.rms_norm_eps)[0].astype(dtype)
        k = fused_rms_norm(k.astype("float32"), self.k_norm_weight, None, self.rms_norm_eps)[0].astype(dtype)
        q = q.reshape([-1, q_num_head, seq_len, head_dim])
        k = k.reshape([-1, kv_num_head, seq_len, head_dim])
        return q, k

    def naive_attention(self, pre_k, pre_v):
        q, k = self.rope._apply_rope(self.rotary_embs, self.q, self.k, self.cache_len)
        if self.use_qk_norm:
            q, k = self.apply_qk_norm(self.head_dim, self.dtype, q, k)

        out_ref = naive_attention_impl(
            q,
            k,
            self.v,
            pre_k,
            pre_v,
            self.attn_mask,
            self.softmax_scale,
            sinks=self.sinks,
        )
        out_ref = remove_padding(self.seq_lens_this_time, self.cu_seqlens_q, out_ref, self.token_num)
        return q, k, self.v, out_ref

    def append_attention(self):
        # buffer
        max_num_block_dec = self.batch_size * (self.max_model_len * self.group_size + 16 - 1) // 16
        decoder_batch_ids = paddle.full([max_num_block_dec], 0, dtype="int32")
        decoder_tile_ids_per_batch = paddle.full([max_num_block_dec], 0, dtype="int32")
        decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
        decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")

        max_num_block = self.batch_size * (self.max_model_len * self.group_size + 64 - 1) // 64
        encoder_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
        encoder_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
        encoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()

        kv_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
        kv_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
        kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        max_len_tensor_cpu = paddle.full([6], 0, dtype="int32").cpu()

        get_block_shape_and_split_kv_block(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_cpu,
            decoder_num_blocks_device,
            decoder_chunk_size_device,
            max_len_tensor_cpu,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            64,
            16,
            self.group_size,
            self.block_size,
        )
        qkv = copy.deepcopy(self.qkv)
        cache_k = copy.deepcopy(self.cache_k)
        cache_v = copy.deepcopy(self.cache_v)
        _ = append_attention(
            qkv,
            cache_k,
            cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_cpu,
            max_len_tensor_cpu,
            self.rotary_embs,
            None,  # attn_mask
            None,  # qkv_bias
            None,  # qkv_out_scales
            self.cache_k_scale,  # cache_k_quant_scales
            self.cache_v_scale,  # cache_v_quant_scales
            self.cache_k_out_scale,  # cache_k_dequant_scales
            self.cache_v_out_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # linear_shift
            None,  # linear_smooth
            self.mask_offset,
            None,  # kv_signal_data
            self.q_norm_weight,
            self.k_norm_weight,
            self.sinks,
            self.rms_norm_eps,
            "bf16",
            self.cache_quant_type,
            False,  # use_neox_rotary_style
            self.rope_3d,
            self.max_model_len,
            self.quant_max_bound,  # quant_max_bound
            self.quant_min_bound,  # quant_min_bound
            -1,
            64,
            16,
            32768,
            1024,
            self.max_tokens_per_batch,
            self.causal,
            self.max_tokens_per_batch > 1,
            self.sliding_window,
        )

    def decode_attention(self):
        paddle.disable_static()

        config_for_attention(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.buffer["block_indices"],
            self.buffer["num_blocks"],
            self.buffer["chunk_size"],
            self.buffer["max_len_tensor_cpu"],
            self.cache_quant_type,
            self.group_size,
            self.kv_num_head,
            self.max_tokens_per_batch,
        )
        # print(f"num_blocks: {self.buffer['num_blocks']}")
        decoder_write_cache_with_rope(
            self.qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.block_tables,
            self.buffer["max_len_tensor_cpu"],
            self.rotary_embs,  # rotary_embs
            None,  # qkv_bias
            self.cache_k_scale,  # cache_k_quant_scales
            self.cache_v_scale,  # cache_v_quant_scales
            self.cache_k_out_scale,  # cache_k_dequant_scales
            self.cache_v_out_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # kv_signal_data
            self.q_norm_weight,  # q_norm_weight
            self.k_norm_weight,  # k_norm_weight
            self.rms_norm_eps,
            self.cache_quant_type,
            False,  # use_neox_rotary_style
            self.rope_3d,
            self.max_model_len,
            self.quant_max_bound,  # quant_max_bound
            self.quant_min_bound,  # quant_min_bound
            self.max_tokens_per_batch > 1,  # speculate_decoder
        )

        out = decode_append_attention(
            self.qkv,
            self.cache_k,
            self.cache_v,
            self.buffer["tmp_workspace"],
            self.buffer["tmp_m"],
            self.buffer["tmp_d"],
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.block_tables,
            self.buffer["block_indices"],
            self.buffer["num_blocks"],
            self.buffer["chunk_size"],
            self.buffer["max_len_tensor_cpu"],  # rope_emb
            None,  # attn_mask
            self.cache_k_scale,  # cache_k_quant_scales
            self.cache_v_scale,  # cache_v_quant_scales
            self.cache_k_out_scale,  # cache_k_dequant_scales
            self.cache_v_out_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            self.mask_offset,  # mask_offset
            self.sinks,  # sinks
            self.cache_quant_type,
            self.max_model_len,
            self.quant_max_bound,  # quant_max_bound
            self.quant_min_bound,  # quant_min_bound
            self.max_tokens_per_batch,  # speculate_max_draft_token_num
            self.causal,  # causal
            self.sliding_window,
        )
        return self.qkv, out

    def prefill(self):
        # init seq_len
        seq_lens_encoder = copy.deepcopy(self.seq_lens_decoder)
        seq_lens_decoder = paddle.zeros([self.batch_size], dtype="int32")
        seq_lens_this_time = seq_lens_encoder
        token_num = seq_lens_this_time.sum().item()
        qkv_np = np.random.random([token_num, (self.q_num_head + 2 * self.kv_num_head) * self.head_dim]) - 0.5
        qkv = paddle.to_tensor(qkv_np, dtype=self.dtype)

        (
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, seq_lens_this_time)
        # buffer
        decode_max_tile_size = self.batch_size * (self.max_model_len * self.group_size + 16 - 1) // 16
        decoder_batch_ids = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        decoder_tile_ids_per_batch = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
        decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
        max_num_block = self.batch_size * (self.max_model_len * self.group_size + 64 - 1) // 64
        encoder_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
        encoder_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
        encoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()

        kv_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
        kv_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
        kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        max_len_tensor_cpu = paddle.full([6], 0, dtype="int32").cpu()
        get_block_shape_and_split_kv_block(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_cpu,
            decoder_num_blocks_device,
            decoder_chunk_size_device,
            max_len_tensor_cpu,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            64,
            16,
            self.group_size,
            self.block_size,
        )
        (
            cu_seqlens_k,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            kv_token_num_cpu,
        ) = pre_cache_len_concat(
            seq_lens_decoder,
            seq_lens_this_time,
            max_len_tensor_cpu[2],
            self.block_size,
        )
        q, k, v, _ = gqa_rope_write_cache(
            qkv,
            self.cache_k,
            self.cache_v,
            cu_seqlens_q,
            cu_seqlens_k,
            self.rotary_embs,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            batch_id_per_token,
            self.block_tables,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            pre_cache_batch_ids,
            pre_cache_tile_ids_per_batch,
            pre_cache_num_blocks_cpu,
            self.q_norm_weight,
            self.k_norm_weight,
            self.cache_k_scale,  # cache_k_quant_scales
            self.cache_v_scale,  # cache_v_quant_scales
            self.cache_k_out_scale,  # cache_k_dequant_scales
            self.cache_v_out_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # kv_signal_data
            kv_token_num_cpu[0].item(),
            self.max_model_len,
            self.rms_norm_eps,
            False,  # use_neox_rotary_style
            self.cache_quant_type,
            self.rope_3d,
        )

        k = k.reshape([self.batch_size, -1, self.kv_num_head, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([self.batch_size, -1, self.kv_num_head, self.head_dim]).transpose([0, 2, 1, 3])
        return k, v

    def test_all(self):
        pre_k, pre_v = self.prefill()

        q_ref, k_ref, v_ref, out_ref = self.naive_attention(pre_k, pre_v)
        qkv_out, out = self.decode_attention()

        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=1e-03,
            atol=2e-03,
        )

    # profiler
    def profile(self):
        pre_k, pre_v = self.prefill()
        paddle.device.synchronize()
        self.append_attention()
        paddle.device.synchronize()
        qkv_out, out = self.decode_attention()
        paddle.device.synchronize()


class TestDecodeAppendAttentionMultiBatch(TestDecodeAppendAttention):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 60
        self.max_tokens_per_batch = 2
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()


class TestDecodeAppendAttentionSpeculate(TestDecodeAppendAttention):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 6
        self.max_tokens_per_batch = 2
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()


class TestDecodeAppendAttentionMultiHead(TestDecodeAppendAttention):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 16
        self.kv_num_head = 2
        self.batch_size = 6
        self.max_tokens_per_batch = 2
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()


class TestDecodeAppendAttentionMultiSpeculate(TestDecodeAppendAttention):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 6
        self.max_tokens_per_batch = 4
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()


class TestDecodeAppendAttentionQKNorm(TestDecodeAppendAttention):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestDecodeAppendAttention"
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 6
        self.max_tokens_per_batch = 2
        self.cache_len = 500
        self.seq_len_dec = None
        self.seq_lens_this_time = None
        self.max_model_len = 131072
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_3d = False
        self.q_hid_dim = self.q_num_head * self.head_dim
        self.kv_hid_dim = self.kv_num_head * self.head_dim
        self.block_size = 64
        self.use_neox_rotary_style = False
        self.softmax_scale = self.head_dim**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.cache_quant_type = "cache_fp8"
        self.use_qk_norm = True
        self.use_mask_offset = False
        self.mask_matrix = False
        self.use_sinks = False
        self.causal = False
        self.use_dynamic_quant = False
        self.quant_min_bound = -448.0
        self.quant_max_bound = 448.0
        self.init_tensor()


if __name__ == "__main__":
    unittest.main()
