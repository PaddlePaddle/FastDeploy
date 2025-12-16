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

import time
import unittest

import numpy as np
import paddle
from paddle.incubate.nn.functional import fused_rms_norm

paddle.seed(10)


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
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def _apply_rope(self, rotary_emb, q, k, v=None, causal=False):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        # sin, cos = paddle.chunk(rp, 2, axis=-1)
        seq, head_dim = q.shape[2], q.shape[3]
        cos, sin = paddle.chunk(rotary_emb, 2, axis=0)
        cos = paddle.squeeze(cos, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        sin = paddle.squeeze(sin, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]

        if self.use_neox_rotary_style:
            sin_pos = sin
            cos_pos = cos
            # NeoX Stype：前后半部分分块旋转
            rotate_half_q = paddle.reshape(
                paddle.stack(
                    [
                        -q[:, :, :, q.shape[-1] // 2 :],
                        q[:, :, :, : q.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(q),
            )
            rotate_half_k = paddle.reshape(
                paddle.stack(
                    [
                        -k[:, :, :, k.shape[-1] // 2 :],
                        k[:, :, :, : k.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(k),
            )
        else:
            # import pdb;pdb.set_trace()
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


def create_attn_mask(
    mask_type,
    batch_size,
    seq_lens,
    pre_cache_length=0,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        # [batch_size, 1, max_seq_len, max_seq_len + pre_cache_length],
        [batch_size, 1, max_seq_len, max_seq_len],
        dtype=mask_type,
    )
    mask[:, :, :, :pre_cache_length] = 1
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type)) - 1
        ) * 1e4
    return mask


def block_cache_to_naive_cache(cache_k, cache_v, bsz, block_tables, cache_seq_len):
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_k.dtype)
    out_cache_v = paddle.zeros(shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_v.dtype)
    for i in range(bsz):
        for j in range(cache_seq_len):
            out_cache_k[i, :, j, :] = cache_k[block_tables[i, j // blocksize], :, j % blocksize, :]
            out_cache_v[i, :, j, :] = cache_v[block_tables[i, j // blocksize], :, j % blocksize, :]
    return out_cache_k, out_cache_v


def naive_attention_impl(
    query,
    key,
    value,
    cache_k=None,
    cache_v=None,
    pre_cache_k=None,
    pre_cache_v=None,
    mask=None,
    scale=1.0,
    cache_k_dequant_scales=None,
    cache_v_dequant_scales=None,
    use_cachekv_int8="None",
    q_norm_weight=None,
    k_norm_weight=None,
):
    batch = query.shape[0]
    heads = query.shape[1]
    seq_len = query.shape[2]
    head_dim = query.shape[3]
    kv_head = key.shape[1]

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, heads // kv_head, 1, 1])
    key = key.reshape([batch, heads, seq_len, head_dim])

    if cache_k is not None:
        cache_k = cache_k.reshape([batch, kv_head, 1, -1, head_dim])
        cache_k = paddle.tile(cache_k, [1, 1, heads // kv_head, 1, 1])
        cache_k = cache_k.reshape([batch, heads, -1, head_dim])
        key = paddle.concat([cache_k, key], axis=2)

    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, heads // kv_head, 1, 1])
    value = value.reshape([batch, heads, seq_len, head_dim])

    if cache_v is not None:
        cache_v = cache_v.reshape([batch, kv_head, 1, -1, head_dim])
        cache_v = paddle.tile(cache_v, [1, 1, heads // kv_head, 1, 1])
        cache_v = cache_v.reshape([batch, heads, -1, head_dim])
        value = paddle.concat([cache_v, value], axis=2)

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    if mask is not None:
        attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(paddle.cast(softmax_result, dtype=value.dtype), value)
    return result


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time, dtype="int32")
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k


def remove_padding(seq_lens, cu_seq_lens, inputs, token_num):
    bsz, num_head, seq_len, dim_head = inputs.shape
    output = paddle.zeros(shape=[token_num, num_head * dim_head], dtype=inputs.dtype)
    inputs = inputs.transpose([0, 2, 1, 3]).reshape([bsz, seq_len, -1])
    for i in range(bsz):
        seq_len_now = seq_lens[i]
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        output[start_idx:end_idx, :] = inputs[i, :seq_len_now, :]
    return output


def get_qkv_and_qkv_concat_tensor(bs, q_num_head, kv_num_head, seq_len, dim_head, place, dtype):
    query = np.random.random([bs, q_num_head, seq_len, dim_head]) / 10
    q = paddle.to_tensor(query, place=place, dtype=dtype, stop_gradient=False)
    key = np.random.random([bs, kv_num_head, seq_len, dim_head]) / 10
    k = paddle.to_tensor(key, place=place, dtype=dtype, stop_gradient=False)
    value = np.random.random([bs, kv_num_head, seq_len, dim_head]) / 10
    v = paddle.to_tensor(value, place=place, dtype=dtype, stop_gradient=False)
    token_num = bs * seq_len

    qkv = paddle.concat(
        [
            q.transpose([0, 2, 1, 3]).reshape([token_num, q_num_head * dim_head]),
            k.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * dim_head]),
            v.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * dim_head]),
        ],
        axis=1,
    ).reshape([token_num, -1])
    return q, k, v, qkv


def apply_qk_norm(head_dim, dtype, q, k):
    q_norm_weight = np.random.random([head_dim]) / 10
    k_norm_weight = np.random.random([head_dim]) / 10
    q_norm_weight_tensor = paddle.to_tensor(q_norm_weight, dtype="float32")
    k_norm_weight_tensor = paddle.to_tensor(k_norm_weight, dtype="float32")
    print("q:", q.shape)
    print("k:", k.shape)
    bs, q_num_head, seq_len, dim_head = q.shape
    _, kv_num_head, _, _ = k.shape

    q = q.reshape([-1, head_dim])
    k = k.reshape([-1, head_dim])
    print("q:", q)
    q = fused_rms_norm(q.astype("float32"), q_norm_weight_tensor, None, 1e-5)[0].astype(dtype)
    print("q after norm:", q)
    k = fused_rms_norm(k.astype("float32"), k_norm_weight_tensor, None, 1e-5)[0].astype(dtype)
    q = q.reshape([-1, q_num_head, seq_len, dim_head])
    k = k.reshape([-1, kv_num_head, seq_len, dim_head])
    return q, k, q_norm_weight_tensor, k_norm_weight_tensor


def split_query_by_phase(
    query,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    q_dim,
    k_dim,
    v_dim,
):
    """
    将 query 拆分为 encoder 和 decoder 的 Q/K/V。
    """

    batch = seq_lens_encoder.shape[0]
    max_seq = query.shape[0] // batch

    # 还原 query 为 [batch, seq, dim]
    total_dim = q_dim + k_dim + v_dim
    query = paddle.reshape(query, [batch, max_seq, total_dim])

    # 计算 mask，表示该 batch 是否是 encoder/decoder
    is_encoder = (seq_lens_encoder > 0).astype("bool").reshape([-1])  # [batch]
    is_decoder = (seq_lens_decoder > 0).astype("bool").reshape([-1])  # [batch]

    # 准备输出列表
    enc_qs, enc_ks, enc_vs = [], [], []
    dec_qs, dec_ks, dec_vs = [], [], []

    for i in range(batch):
        real_len = int(seq_lens_this_time[i])  # 当前 batch 的有效长度
        cur_query = query[i, :real_len, :]  # [seq_i, q+k+v]

        q, k, v = paddle.split(cur_query, [q_dim, k_dim, v_dim], axis=-1)

        if is_encoder[i]:
            enc_qs.append(q)
            enc_ks.append(k)
            enc_vs.append(v)
        elif is_decoder[i]:
            dec_qs.append(q)
            dec_ks.append(k)
            dec_vs.append(v)

    if enc_qs:
        enc_q = paddle.concat(enc_qs, axis=0)
        enc_k = paddle.concat(enc_ks, axis=0)
        enc_v = paddle.concat(enc_vs, axis=0)
    else:
        enc_q = enc_k = enc_v = paddle.zeros([0, q_dim], dtype=query.dtype)

    if dec_qs:
        dec_q = paddle.concat(dec_qs, axis=0)
        dec_k = paddle.concat(dec_ks, axis=0)
        dec_v = paddle.concat(dec_vs, axis=0)
    else:
        dec_q = dec_k = dec_v = paddle.zeros([0, q_dim], dtype=query.dtype)

    return (enc_q, enc_k, enc_v), (dec_q, dec_k, dec_v)


class TestAppendGroupQueryAttnWithRope(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestAppendGroupQueryAttnWithRope"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.q_num_head = 12
        self.kv_num_head = 2
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 128
        self.q_hid_dim = self.q_num_head * self.dim_head
        self.kv_hid_dim = self.kv_num_head * self.dim_head
        self.blocksize = 64
        self.use_neox_rotary_style = False
        # max_seq_len = self.seq_len + self.max_dec_len
        self.max_seq_len = self.seq_len + self.max_dec_len
        self.softmax_scale = self.dim_head**-0.5
        self.rope_theta = 10000
        self.dtype = "float16"
        self.use_qk_norm = True
        self.use_mask_offset = False
        self.init_tensor()

    def init_tensor(self):
        self.block_num_per_seq = (self.seq_len + self.max_dec_len + self.blocksize - 1) // self.blocksize
        self.rope = RopeEmbedding(self.use_neox_rotary_style)
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))

        self.seq_lens_enc = [
            self.seq_len,
        ] * self.batch_size
        self.seq_lens_dec = [
            0,
        ] * self.batch_size
        self.max_enc_len_this_time = max(self.seq_lens_enc)
        self.max_dec_len_this_time = max(self.seq_lens_dec)
        self.seq_lens_encoder = paddle.to_tensor(
            self.seq_lens_enc,
            "int32",
        )
        self.seq_lens_decoder = paddle.to_tensor(
            self.seq_lens_dec,
            "int32",
        )
        self.max_enc_len_this_time = paddle.to_tensor([self.max_enc_len_this_time], "int32", place=paddle.CPUPlace())
        self.max_dec_len_this_time = paddle.to_tensor([self.max_dec_len_this_time], "int32", place=paddle.CPUPlace())
        self.seq_lens_this_time = self.seq_lens_encoder

        decode_max_tile_size = 1024 * self.batch_size * np.ceil((2 * 10) / 12)
        self.decoder_batch_ids = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.decoder_tile_ids_per_batch = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
        self.decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        self.decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
        self.max_len_tensor_cpu = paddle.full([8], 0, dtype="int32").cpu()
        self.encoder_batch_ids = paddle.full([self.batch_size], 0, dtype="int32")
        self.encoder_tile_ids_per_batch = paddle.full([self.batch_size], 0, dtype="int32")
        self.encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        self.kv_batch_ids = paddle.full([self.batch_size], 0, dtype="int32")
        self.kv_tile_ids_per_batch = paddle.full([self.batch_size], 0, dtype="int32")
        self.kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

        self.cache_shape = (
            self.max_block_num,
            self.kv_num_head,
            self.blocksize,
            self.dim_head,
        )

        self.scale = 1.0 / np.sqrt(self.dim_head)
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(shape=(self.batch_size, self.block_num_per_seq), dtype="int32")
        for i in range(self.batch_size):
            need_block_num = (self.seq_len + self.max_dec_len + self.blocksize - 1) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, self.seq_len, self.seq_lens_this_time)
        self.token_num = self.padding_offset.shape[0]
        self.mask_offset = None
        if self.use_mask_offset:
            self.mask_offset = paddle.full(self.batch_size * self.seq_len * 2, 0, "int32")
            for i in range(self.batch_size):
                for j in range(self.seq_len):
                    self.mask_offset[i * self.seq_len * 2 + j * 2] = 0
                    self.mask_offset[i * self.seq_len * 2 + j * 2 + 1] = j + 1

    def cmp_append_attention(self, naive_cache_k=None, naive_cache_v=None, attn_mask=None):
        paddle.disable_static()
        self.token_num = self.seq_len * self.batch_size
        q, k, v, qkv = get_qkv_and_qkv_concat_tensor(
            self.batch_size,
            self.q_num_head,
            self.kv_num_head,
            self.seq_len,
            self.dim_head,
            self.place,
            self.dtype,
        )

        q, k = self.rope._apply_rope(self.rope_emb, q, k, causal=True)
        if self.use_qk_norm:
            q, k, q_norm_weight, k_norm_weight = apply_qk_norm(self.dim_head, self.dtype, q, k)
        else:
            q_norm_weight = None
            k_norm_weight = None
        out_ = naive_attention_impl(
            q,
            k,
            v,
            naive_cache_k,
            naive_cache_v,
            None,
            None,
            attn_mask,
            self.scale,
        )
        out_ = remove_padding(self.seq_lens_this_time, self.cu_seqlens_q, out_, self.token_num)
        speculate_max_draft_token_num = 1
        from fastdeploy.model_executor.layers.attention.ops import (
            append_attention_with_output,
            get_block_shape_and_split_kv_block,
        )

        get_block_shape_and_split_kv_block(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.decoder_batch_ids,
            self.decoder_tile_ids_per_batch,
            self.decoder_num_blocks_cpu,
            self.decoder_num_blocks_device,
            self.decoder_chunk_size_device,
            self.max_len_tensor_cpu,
            self.encoder_batch_ids,
            self.encoder_tile_ids_per_batch,
            self.encoder_num_blocks_x_cpu,
            self.kv_batch_ids,
            self.kv_tile_ids_per_batch,
            self.kv_num_blocks_x_cpu,
            64,
            12,
            (self.q_num_head + 2 * self.kv_num_head) // self.kv_num_head,
            self.blocksize,
        )

        # Warm up
        WARM_UP = 1
        RUN_TIME = 2
        out = paddle.zeros((qkv.shape[0], self.q_hid_dim), dtype=q.dtype).to(q.place)
        for i in range(WARM_UP + RUN_TIME):
            if i == WARM_UP:
                paddle.device.synchronize()
                start_time = time.time()
            append_attention_with_output(
                qkv,
                self.cache_k,
                self.cache_v,
                self.seq_lens_encoder,
                self.seq_lens_decoder,
                self.seq_lens_this_time,
                self.padding_offset,
                self.cum_offset,
                self.block_tables,
                self.encoder_batch_ids,
                self.encoder_tile_ids_per_batch,
                self.encoder_num_blocks_x_cpu,
                self.kv_batch_ids,
                self.kv_tile_ids_per_batch,
                self.kv_num_blocks_x_cpu,
                self.decoder_batch_ids,
                self.decoder_tile_ids_per_batch,
                self.decoder_num_blocks_cpu,
                self.max_len_tensor_cpu,
                out,
                self.rope_emb,  # rope_emb
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # linear_shift
                None,  # linear_smooth
                self.mask_offset,  # mask_offset
                None,  # kv_signal_data
                q_norm_weight,  # q_norm_weight
                k_norm_weight,  # k_norm_weight
                None,  # sinks
                1e-6,
                "fp16",
                "none",  # cache_quant_type
                self.use_neox_rotary_style,
                False,
                self.max_seq_len,
                0.0,  # quant_min_bound
                0.0,  # quant_max_bound
                -1,  # out_linear_in_scale
                64,  # encoder_block_shape_q
                16,  # decoder_block_shape_q
                32768,  # max_partition_size
                32768,  # encoder_max_partition_size
                speculate_max_draft_token_num + 1,  # speculate_max_draft_token_num
                True,  # causal
                False,  # speculate_decoder
                -1,
            )
        paddle.device.synchronize()
        end_time = time.time()
        print(f"[append-attn ut]  cost_time:{(end_time - start_time) / RUN_TIME * 1000}ms")
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )
        np.testing.assert_allclose(
            out.numpy(),
            out_.numpy(),
            rtol=1e-02,
            atol=1e-02,
        )

    def test_all(self):
        tmp_position_ids = paddle.arange(self.seq_len + self.max_dec_len).reshape((1, -1))
        # appendattn 传的是最大maxseq
        if self.use_neox_rotary_style:
            self.rope_emb = self.rope.get_neox_style_position_embedding(tmp_position_ids, self.dim_head)
        else:
            self.rope_emb = self.rope.get_rotary_position_embedding(tmp_position_ids, self.dim_head)
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        # encoder
        # self.seq_lens_encoder,self.seq_lens_decoder,self.max_enc_len_this_time,self.max_dec_len_this_time=get_encoder_decoder_len(self.batch_size,self.seq_len)
        self.seq_lens_this_time = self.seq_lens_encoder
        if self.use_mask_offset:
            print("encoder mask_offset: ", self.mask_offset)
        self.cmp_append_attention(attn_mask=self.attention_mask)
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            self.cache_k,
            self.cache_v,
            self.batch_size,
            self.block_tables,
            self.seq_len,
        )
        # decoder
        self.seq_lens_decoder[:] = self.seq_lens_encoder
        self.seq_lens_encoder[:] = 0
        self.seq_lens_this_time[:] = 1
        self.seq_lens_enc = [
            0,
        ] * self.batch_size
        self.seq_lens_dec = [
            self.seq_len,
        ] * self.batch_size
        self.max_enc_len_this_time = max(self.seq_lens_enc)
        self.max_dec_len_this_time = max(self.seq_lens_dec)
        self.max_enc_len_this_time = paddle.to_tensor([self.max_enc_len_this_time], "int32", place=paddle.CPUPlace())
        self.max_dec_len_this_time = paddle.to_tensor([self.max_dec_len_this_time], "int32", place=paddle.CPUPlace())

        self.seq_len = 1
        (
            self.padding_offset,
            self.cum_offset,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, 1, self.seq_lens_this_time)
        if self.use_mask_offset:
            self.mask_offset = paddle.full(self.batch_size * 2, 0, "int32")
            for i in range(self.batch_size):
                self.mask_offset[i * 2] = 0
                self.mask_offset[i * 2 + 1] = self.seq_lens_dec[i] + 1
            print("decoder mask_offset: ", self.mask_offset)
        self.cmp_append_attention(naive_cache_k, naive_cache_v, None)


class TestAppendGroupQueryAttnWithNeoXRope(TestAppendGroupQueryAttnWithRope):
    def setUp(self):
        paddle.disable_static()
        self.name = "TestAppendGroupQueryAttnWithRope"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.q_num_head = 12
        self.kv_num_head = 2
        self.seq_len = 64
        self.max_dec_len = 64
        self.dim_head = 128
        self.q_hid_dim = self.q_num_head * self.dim_head
        self.kv_hid_dim = self.kv_num_head * self.dim_head
        self.blocksize = 64
        self.use_neox_rotary_style = True
        # max_seq_len = self.seq_len + self.max_dec_len
        self.max_seq_len = self.seq_len + self.max_dec_len
        self.softmax_scale = self.dim_head**-0.5
        self.rope_theta = 10000
        self.dtype = "float16"
        self.use_qk_norm = False
        self.use_mask_offset = True
        self.init_tensor()


if __name__ == "__main__":
    unittest.main()
