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

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention as append_attention_op,
)
from fastdeploy.model_executor.layers.attention.ops import (
    config_for_attention,
    decode_unified_attention,
    decoder_write_cache_with_rope,
    get_block_shape_and_split_kv_block,
)

seed = 1000

random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


class RopeEmbedding:
    def __init__(self, use_neox_rotary_style=False):
        self.use_neox_rotary_style = use_neox_rotary_style
        self.base = 10000

    def get_rotary_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, head_dim // 2))
        emb = paddle.unsqueeze(emb, 2)
        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def _apply_rope(self, rotary_emb, q, k, start_pos=0):
        seq, head_dim = q.shape[2], q.shape[3]
        cos, sin = paddle.chunk(rotary_emb, 2, axis=0)
        cos = cos[:, :, start_pos : start_pos + seq, ...]
        sin = sin[:, :, start_pos : start_pos + seq, ...]
        cos = paddle.squeeze(cos, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        sin = paddle.squeeze(sin, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]

        sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), [1, 1, seq, head_dim])
        cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), [1, 1, seq, head_dim])
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


def naive_attention_impl(query, key, value, cache_k=None, cache_v=None, mask=None, scale=1.0):
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


def block_cache_to_naive_cache(cache_k, cache_v, bsz, block_tables, cache_seq_len):
    """Read K/V from paged cache and return as [batch, num_head, seq_len, dim_head]."""
    _, num_head, blocksize, dim_head = cache_k.shape
    out_cache_k = paddle.zeros(shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_k.dtype)
    out_cache_v = paddle.zeros(shape=[bsz, num_head, cache_seq_len, dim_head], dtype=cache_v.dtype)
    for i in range(bsz):
        for j in range(cache_seq_len):
            out_cache_k[i, :, j, :] = cache_k[block_tables[i, j // blocksize], :, j % blocksize, :]
            out_cache_v[i, :, j, :] = cache_v[block_tables[i, j // blocksize], :, j % blocksize, :]
    return out_cache_k, out_cache_v


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


class TestDecodeUnifiedAttentionC16(unittest.TestCase):
    """Base test class for decode append attention with cache_quant_type='none' (fp16/bf16 KV cache).

    Uses append_attention for prefill (verified correct by test_append_attention_c16.py)
    and then tests decode_unified_attention (new split ops) against the same naive reference.

    Subclasses override setUp to vary batch_size, max_tokens_per_batch, dtype, etc.
    """

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 1
        self.max_tokens_per_batch = 1
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head

        # Use small seq_len for fast testing; can increase later
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()

    def init_tensor(self):
        self.rope = RopeEmbedding(self.use_neox_rotary_style)
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

        # cache
        self.cache_shape = (
            self.max_block_num,
            self.kv_num_head,
            self.block_size,
            self.head_dim,
        )
        self.cache_k = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
        self.cache_v = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)

        # Encoder phase: prefill with seq_len tokens
        self.enc_q, self.enc_k, self.enc_v, self.enc_qkv = get_qkv_and_qkv_concat_tensor(
            self.batch_size,
            self.q_num_head,
            self.kv_num_head,
            self.seq_len,
            self.head_dim,
            self.place,
            self.dtype,
        )

        # Decoder phase: max_tokens_per_batch decode tokens
        self.dec_q, self.dec_k, self.dec_v, self.dec_qkv = get_qkv_and_qkv_concat_tensor(
            self.batch_size,
            self.q_num_head,
            self.kv_num_head,
            self.max_tokens_per_batch,
            self.head_dim,
            self.place,
            self.dtype,
        )

    def _get_block_shape_buffers(self, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time):
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
        return {
            "decoder_batch_ids": decoder_batch_ids,
            "decoder_tile_ids_per_batch": decoder_tile_ids_per_batch,
            "decoder_num_blocks_cpu": decoder_num_blocks_cpu,
            "encoder_batch_ids": encoder_batch_ids,
            "encoder_tile_ids_per_batch": encoder_tile_ids_per_batch,
            "encoder_num_blocks_cpu": encoder_num_blocks_cpu,
            "kv_batch_ids": kv_batch_ids,
            "kv_tile_ids_per_batch": kv_tile_ids_per_batch,
            "kv_num_blocks_x_cpu": kv_num_blocks_x_cpu,
            "max_len_tensor_cpu": max_len_tensor_cpu,
        }

    def run_append_attention(
        self,
        qkv,
        cache_k,
        cache_v,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
    ):
        """Run append_attention op."""
        buffers = self._get_block_shape_buffers(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time)

        qkv_copy = copy.deepcopy(qkv)
        cache_k_copy = copy.deepcopy(cache_k)
        cache_v_copy = copy.deepcopy(cache_v)

        out = append_attention_op(
            qkv_copy,
            cache_k_copy,
            cache_v_copy,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            self.block_tables,
            buffers["encoder_batch_ids"],
            buffers["encoder_tile_ids_per_batch"],
            buffers["encoder_num_blocks_cpu"],
            buffers["kv_batch_ids"],
            buffers["kv_tile_ids_per_batch"],
            buffers["kv_num_blocks_x_cpu"],
            buffers["decoder_batch_ids"],
            buffers["decoder_tile_ids_per_batch"],
            buffers["decoder_num_blocks_cpu"],
            buffers["max_len_tensor_cpu"],
            self.rotary_embs,
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
            None,  # mask_offset
            None,  # kv_signal_data
            None,  # q_norm_weight
            None,  # k_norm_weight
            None,  # sinks
            self.rms_norm_eps,
            "bf16",
            self.cache_quant_type,
            self.use_neox_rotary_style,
            self.rope_3d,
            self.max_model_len,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            -1,
            64,
            16,
            1024,
            self.max_model_len,
            self.max_tokens_per_batch,  # speculate_max_draft_token_num
            self.causal,
            self.max_tokens_per_batch > 1,  # speculate_decoder
        )
        return out, cache_k_copy, cache_v_copy

    def _build_decode_buffer(self):
        """Build buffer for new split decode ops."""
        buffer = {}
        min_chunk_size = 512
        max_num_chunk = (self.max_model_len + min_chunk_size - 1) // min_chunk_size
        q_tile_size = 16
        q_tile_num = (self.max_tokens_per_batch * self.group_size + q_tile_size - 1) // q_tile_size
        buffer["max_len_tensor_cpu"] = paddle.full([6], 0, dtype="int32").cpu()
        buffer["block_indices"] = paddle.full(
            [self.batch_size * self.kv_num_head * max_num_chunk * q_tile_num, 4], 0, dtype="int32"
        )
        buffer["num_blocks"] = paddle.full([1], 0, dtype="int32")
        buffer["chunk_size"] = paddle.full([1], 0, dtype="int32")
        buffer["tmp_workspace"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head * self.head_dim],
            0,
            dtype=self.dtype,
        )
        buffer["tmp_m"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head], 0, dtype="float32"
        )
        buffer["tmp_d"] = paddle.full(
            [self.batch_size * self.max_tokens_per_batch, max_num_chunk, self.q_num_head], 0, dtype="float32"
        )
        return buffer

    def _run_decode_unified_attention(
        self,
        cache_k,
        cache_v,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
    ):
        """Run config_for_attention + decoder_write_cache_with_rope + decode_unified_attention."""
        buffer = self._build_decode_buffer()

        config_for_attention(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            buffer["block_indices"],
            buffer["num_blocks"],
            buffer["chunk_size"],
            buffer["max_len_tensor_cpu"],
            self.cache_quant_type,
            self.group_size,
            self.kv_num_head,
            self.max_tokens_per_batch,
        )

        dec_cache_k = copy.deepcopy(cache_k)
        dec_cache_v = copy.deepcopy(cache_v)
        dec_qkv = copy.deepcopy(self.dec_qkv)

        decoder_write_cache_with_rope(
            dec_qkv,
            dec_cache_k,
            dec_cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            self.block_tables,
            buffer["max_len_tensor_cpu"],
            self.rotary_embs,
            None,  # qkv_bias
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # kv_signal_data
            None,  # q_norm_weight
            None,  # k_norm_weight
            self.rms_norm_eps,
            self.cache_quant_type,
            self.use_neox_rotary_style,
            self.rope_3d,
            self.max_model_len,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            self.max_tokens_per_batch > 1,  # speculate_decoder
        )

        out = decode_unified_attention(
            dec_qkv,
            dec_cache_k,
            dec_cache_v,
            buffer["tmp_workspace"],
            buffer["tmp_m"],
            buffer["tmp_d"],
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            self.block_tables,
            buffer["block_indices"],
            buffer["num_blocks"],
            buffer["chunk_size"],
            buffer["max_len_tensor_cpu"],
            None,  # attn_mask
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # mask_offset
            None,  # sinks
            paddle.empty([dec_qkv.shape[0], self.q_num_head * self.head_dim], dtype=dec_qkv.dtype),  # fmha_out
            self.cache_quant_type,
            self.max_model_len,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            self.max_tokens_per_batch,  # speculate_max_draft_token_num
            self.causal,  # causal
        )
        return out, dec_cache_k, dec_cache_v

    def do_prefill_with_append_attention(self):
        """Prefill using append_attention. Returns cache_k, cache_v after prefill."""
        seq_lens_encoder = paddle.to_tensor([self.seq_len] * self.batch_size, "int32")
        seq_lens_decoder = paddle.to_tensor([0] * self.batch_size, "int32")
        seq_lens_this_time = copy.deepcopy(seq_lens_encoder)

        batch_id_per_token, cu_seqlens_q, _ = get_padding_offset(self.batch_size, seq_lens_this_time)

        _, cache_k, cache_v = self.run_append_attention(
            self.enc_qkv,
            self.cache_k,
            self.cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
        )
        return cache_k, cache_v

    def compute_naive_decode_ref(self, cache_k, cache_v):
        """Compute naive reference for decode step using cache from paged cache."""
        # Read K/V from paged cache
        naive_cache_k, naive_cache_v = block_cache_to_naive_cache(
            cache_k, cache_v, self.batch_size, self.block_tables, self.seq_len
        )

        # Only use the first decode token (seq_lens_this_time=1 per batch)
        dec_q = self.dec_q[:, :, :1, :]
        dec_k = self.dec_k[:, :, :1, :]
        dec_v = self.dec_v[:, :, :1, :]

        # Apply RoPE to decode Q/K at position seq_len
        dec_q_rope, dec_k_rope = self.rope._apply_rope(self.rotary_embs, dec_q, dec_k, start_pos=self.seq_len)

        # Compute naive attention
        out_ref = naive_attention_impl(
            dec_q_rope,
            dec_k_rope,
            dec_v,
            cache_k=naive_cache_k,
            cache_v=naive_cache_v,
            scale=self.softmax_scale,
        )

        dec_seq_lens_this_time = paddle.to_tensor([1] * self.batch_size, "int32")
        dec_token_num = self.batch_size
        _, dec_cu_seqlens_q, _ = get_padding_offset(self.batch_size, dec_seq_lens_this_time)
        out_ref = remove_padding(dec_seq_lens_this_time, dec_cu_seqlens_q, out_ref, dec_token_num)
        return out_ref

    def test_naive_vs_append_attention_decode(self):
        """Test: prefill with append_attention, then decode with append_attention. Compare to naive."""
        # Step 1: Prefill
        cache_k, cache_v = self.do_prefill_with_append_attention()

        # Step 2: Naive reference for decode
        out_ref = self.compute_naive_decode_ref(cache_k, cache_v)

        # Step 3: Decode with append_attention
        # seq_lens_this_time must match qkv rows: batch_size * max_tokens_per_batch
        dec_seq_lens_encoder = paddle.to_tensor([0] * self.batch_size, "int32")
        dec_seq_lens_decoder = paddle.to_tensor([self.seq_len] * self.batch_size, "int32")
        dec_seq_lens_this_time = paddle.to_tensor([self.max_tokens_per_batch] * self.batch_size, "int32")

        dec_batch_id_per_token, dec_cu_seqlens_q, _ = get_padding_offset(self.batch_size, dec_seq_lens_this_time)

        out_dec, _, _ = self.run_append_attention(
            self.dec_qkv,
            cache_k,
            cache_v,
            dec_seq_lens_encoder,
            dec_seq_lens_decoder,
            dec_seq_lens_this_time,
            dec_batch_id_per_token,
            dec_cu_seqlens_q,
        )

        out_ref_f = out_ref.astype("float32").numpy()
        out_dec_f = out_dec.astype("float32").numpy()

        # Truncate to actual token count (output may be padded to max_tokens_per_batch)
        dec_token_num = self.batch_size
        out_dec_f = out_dec_f[:dec_token_num]

        np.testing.assert_allclose(
            out_dec_f,
            out_ref_f,
            rtol=1e-02,
            atol=1e-02,
            err_msg="append_attention decode output doesn't match naive reference",
        )

    def test_naive_vs_decode_unified_attention(self):
        """Test: prefill with append_attention, then decode with new split decode ops."""
        # Step 1: Prefill
        cache_k, cache_v = self.do_prefill_with_append_attention()

        # Step 2: Naive reference for decode
        out_ref = self.compute_naive_decode_ref(cache_k, cache_v)

        # Step 3: Decode with new split ops
        # seq_lens_this_time must match qkv rows: batch_size * max_tokens_per_batch
        dec_seq_lens_encoder = paddle.to_tensor([0] * self.batch_size, "int32")
        dec_seq_lens_decoder = paddle.to_tensor([self.seq_len] * self.batch_size, "int32")
        dec_seq_lens_this_time = paddle.to_tensor([self.max_tokens_per_batch] * self.batch_size, "int32")

        dec_batch_id_per_token, dec_cu_seqlens_q, _ = get_padding_offset(self.batch_size, dec_seq_lens_this_time)

        out, _, _ = self._run_decode_unified_attention(
            cache_k,
            cache_v,
            dec_seq_lens_encoder,
            dec_seq_lens_decoder,
            dec_seq_lens_this_time,
            dec_batch_id_per_token,
            dec_cu_seqlens_q,
        )

        out_ref_f = out_ref.astype("float32").numpy()
        out_decode_f = out.astype("float32").numpy()

        # Truncate to actual token count (output may be padded to max_tokens_per_batch)
        dec_token_num = self.batch_size
        out_decode_f = out_decode_f[:dec_token_num]

        np.testing.assert_allclose(
            out_decode_f,
            out_ref_f,
            rtol=1e-02,
            atol=1e-02,
            err_msg="decode_unified_attention output doesn't match naive reference",
        )

    def test_append_vs_decode_unified_attention(self):
        """Test: append_attention decode vs new split decode ops should produce same result."""
        # Step 1: Prefill
        cache_k, cache_v = self.do_prefill_with_append_attention()

        # Step 2: Decode with append_attention
        # seq_lens_this_time must match qkv rows: batch_size * max_tokens_per_batch
        dec_seq_lens_encoder = paddle.to_tensor([0] * self.batch_size, "int32")
        dec_seq_lens_decoder = paddle.to_tensor([self.seq_len] * self.batch_size, "int32")
        dec_seq_lens_this_time = paddle.to_tensor([self.max_tokens_per_batch] * self.batch_size, "int32")
        dec_batch_id_per_token, dec_cu_seqlens_q, _ = get_padding_offset(self.batch_size, dec_seq_lens_this_time)

        out_append, _, _ = self.run_append_attention(
            self.dec_qkv,
            copy.deepcopy(cache_k),
            copy.deepcopy(cache_v),
            dec_seq_lens_encoder,
            dec_seq_lens_decoder,
            dec_seq_lens_this_time,
            dec_batch_id_per_token,
            dec_cu_seqlens_q,
        )

        # Step 3: Decode with new split ops
        out_decode, _, _ = self._run_decode_unified_attention(
            cache_k,
            cache_v,
            dec_seq_lens_encoder,
            dec_seq_lens_decoder,
            dec_seq_lens_this_time,
            dec_batch_id_per_token,
            dec_cu_seqlens_q,
        )

        out_append_f = out_append.astype("float32").numpy()
        out_decode_f = out_decode.astype("float32").numpy()

        # Truncate to actual token count (output may be padded to max_tokens_per_batch)
        dec_token_num = self.batch_size
        out_append_f = out_append_f[:dec_token_num]
        out_decode_f = out_decode_f[:dec_token_num]

        np.testing.assert_allclose(
            out_decode_f,
            out_append_f,
            rtol=1e-02,
            atol=1e-02,
            err_msg="decode_unified_attention doesn't match append_attention decode",
        )


class TestDecodeUnifiedAttentionC16Speculate(TestDecodeUnifiedAttentionC16):
    """Test with speculate decode: max_tokens_per_batch=2.

    When max_tokens_per_batch > 1, naive ref only computes 1 token while ops
    compute multiple tokens. So naive comparison tests are skipped; only
    append_attention vs decode_unified_attention comparison is kept.
    """

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 1
        self.max_tokens_per_batch = 2
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()

    def test_naive_vs_append_attention_decode(self):
        """Skip: naive ref only computes 1 token, but ops compute max_tokens_per_batch tokens."""
        pass

    def test_naive_vs_decode_unified_attention(self):
        """Skip: naive ref only computes 1 token, but ops compute max_tokens_per_batch tokens."""
        pass


class TestDecodeUnifiedAttentionC16MultiBatch(TestDecodeUnifiedAttentionC16):
    """Test with multiple batches."""

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 4
        self.max_tokens_per_batch = 1
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()


class TestDecodeUnifiedAttentionC16MultiHead(TestDecodeUnifiedAttentionC16):
    """Test with multiple KV heads (GQA)."""

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 16
        self.kv_num_head = 2
        self.batch_size = 2
        self.max_tokens_per_batch = 1
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()


class TestDecodeUnifiedAttentionC16FP16(TestDecodeUnifiedAttentionC16):
    """Test with float16 dtype."""

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 1
        self.max_tokens_per_batch = 1
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "float16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()


class TestDecodeUnifiedAttentionC16NoCausal(TestDecodeUnifiedAttentionC16):
    """Test with causal=False."""

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 1
        self.max_tokens_per_batch = 1
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = False
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()


class TestDecodeUnifiedAttentionC16MultiBatchSpeculate(TestDecodeUnifiedAttentionC16):
    """Test with multi-batch + speculate decode.

    When max_tokens_per_batch > 1, the naive reference only computes 1 token
    while ops compute multiple tokens. So we only compare append_attention vs
    decode_unified_attention (both should produce same result), and skip the
    naive comparison tests.
    """

    def setUp(self):
        paddle.disable_static()
        self.place = paddle.CUDAPlace(0)
        self.q_num_head = 14
        self.kv_num_head = 1
        self.batch_size = 4
        self.max_tokens_per_batch = 2
        self.head_dim = 128
        self.block_size = 64
        self.dtype = "bfloat16"
        self.cache_quant_type = "none"
        self.use_neox_rotary_style = False
        self.rope_3d = False
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = 1e-6
        self.causal = True
        self.group_size = self.q_num_head // self.kv_num_head
        self.seq_len = 6400
        self.max_model_len = self.seq_len + 128
        self.init_tensor()

    def test_naive_vs_append_attention_decode(self):
        """Skip: naive ref only computes 1 token, but ops compute max_tokens_per_batch tokens."""
        pass

    def test_naive_vs_decode_unified_attention(self):
        """Skip: naive ref only computes 1 token, but ops compute max_tokens_per_batch tokens."""
        pass


if __name__ == "__main__":
    unittest.main()
