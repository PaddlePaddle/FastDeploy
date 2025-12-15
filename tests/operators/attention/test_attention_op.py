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
import math
import random
import time
import unittest

import numpy as np
import paddle
from paddle.incubate.nn.functional import fused_rms_norm

import fastdeploy

seed = 1000

random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
)
from fastdeploy.model_executor.ops.gpu import (
    decoder_write_cache_with_rope,
    decode_append_attention
)

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


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    token_num = paddle.sum(seq_lens_this_time)
    batch_id_per_token = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_seq_len = 0
    index = 0
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i].item()
        for j in range(seq_len_now):
            batch_id_per_token[index] = i
            index += 1
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return batch_id_per_token, cu_seqlens_q, cu_seqlens_k
    

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


class TestAppendGroupQueryAttnWithRope(unittest.TestCase):
    def setUp(self):
        self.name = "TestAppendGroupQueryAttnWithRope"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.q_num_head = 16
        self.kv_num_head = 2
        self.seq_len = 64
        self.max_dec_len = 32
        self.dim_head = 128
        self.q_hid_dim = self.q_num_head * self.dim_head
        self.kv_hid_dim = self.kv_num_head * self.dim_head
        self.blocksize = 64
        self.use_neox_rotary_style = False
        # max_seq_len = self.seq_len + self.max_dec_len
        self.max_seq_len = self.seq_len + self.max_dec_len
        self.softmax_scale = self.dim_head**-0.5
        self.rope_theta = 10000
        self.sliding_window = 128
        self.dtype = "bfloat16"
        self.use_qk_norm = True
        self.use_mask_offset = False
        self.use_sinks = True
        self.use_yarn = False
        self.use_dynamic_quant = False
        self.init_tensor()

    def init_tensor(self):
        self.block_num_per_seq = (self.kv_seq_len + self.max_dec_len + self.blocksize - 1) // self.blocksize
        self.rope = RopeEmbedding(self.use_neox_rotary_style)
        self.max_block_num = self.block_num_per_seq * self.batch_size
        self.free_list = list(range(self.max_block_num - 1, -1, -1))

        self.seq_lens_enc = [
            0,
        ] * self.batch_size
        self.seq_lens_dec = [
            self.kv_seq_len,
        ] * self.batch_size
        self.seq_lens_this = [
            self.seq_len,
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
        self.seq_lens_this_time = paddle.to_tensor(
            self.seq_lens_this,
            "int32",
        )
        self.max_enc_len_this_time = paddle.to_tensor([self.max_enc_len_this_time], "int32", place=paddle.CPUPlace())
        self.max_dec_len_this_time = paddle.to_tensor([self.max_dec_len_this_time], "int32", place=paddle.CPUPlace())
        

        decode_max_tile_size = 1024 * self.batch_size * np.ceil((2 * 10) / 16)
        self.decoder_batch_ids = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.decoder_tile_ids_per_batch = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
        self.decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        self.decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
        self.max_len_tensor_cpu = paddle.full([8], 0, dtype="int32").cpu()

        self.encoder_batch_ids = paddle.full([self.batch_size], 0, dtype="int32")
        self.encoder_tile_ids_per_batch = paddle.full([self.batch_size], 0, dtype="int32")
        self.encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        self.kv_batch_ids = paddle.full([self.batch_size * ((self.max_seq_len + self.blocksize - 1) // self.blocksize)], 0, dtype="int32")
        self.kv_tile_ids_per_batch = paddle.full([self.batch_size * ((self.max_seq_len + self.blocksize - 1) // self.blocksize)], 0, dtype="int32")
        self.kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

        self.cache_shape = (
            self.max_block_num,
            self.kv_num_head,
            self.blocksize,
            self.dim_head,
        )

        self.scale = 1.0 / np.sqrt(self.dim_head)
        if self.use_dynamic_quant:
            self.cache_scale_shape = (
                self.max_block_num,
                self.kv_num_head,
                self.blocksize,
            )
            self.cache_k = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_v = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_k_T = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
            self.cache_v_T = paddle.zeros(shape=self.cache_shape, dtype=self.dtype)
            self.key_cache_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.value_cache_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
        else:
            self.cache_scale_shape = (
                self.kv_num_head,
            )
            self.cache_k = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.cache_v = paddle.zeros(shape=self.cache_shape, dtype="uint8")
            self.key_cache_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.value_cache_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.key_cache_dequant_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
            self.value_cache_dequant_scale = paddle.zeros(shape=self.cache_scale_shape, dtype=self.dtype)
        self.block_tables = paddle.zeros(shape=(self.batch_size, self.block_num_per_seq), dtype="int32")
        for i in range(self.batch_size):
            need_block_num = (self.kv_seq_len + self.max_dec_len + self.blocksize - 1) // self.blocksize
            for j in range(need_block_num):
                self.block_tables[i, j] = self.free_list.pop()
        (
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
        ) = get_padding_offset(self.batch_size, self.max_seq_len, self.seq_lens_this_time)
        self.token_num = self.batch_id_per_token.shape[0]
        self.mask_offset = None
        if self.use_mask_offset:
            self.mask_offset = paddle.full(self.batch_size * self.seq_len * 2, 0, "int32")
            for i in range(self.batch_size):
                for j in range(self.seq_len):
                    self.mask_offset[i * self.seq_len * 2 + j * 2] = 0
                    self.mask_offset[i * self.seq_len * 2 + j * 2 + 1] = j + 1
        

    def test(self, max_partition_size=1024, decode_block_shape_q=16):
        tmp_position_ids = paddle.arange(self.max_seq_len).reshape((1, -1))
        # appendattn 传的是最大maxseq
        self.rope_emb = self.rope.get_rotary_position_embedding(tmp_position_ids, self.dim_head)
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
            sliding_window=self.sliding_window,
        )
        # encoder
        # self.seq_lens_encoder,self.seq_lens_decoder,self.max_enc_len_this_time,self.max_dec_len_this_time=get_encoder_decoder_len(self.batch_size,self.seq_len)
        if self.use_mask_offset:
            print("encoder mask_offset: ", self.mask_offset)

        # decoder
        self.max_enc_len_this_time = max(self.seq_lens_enc)
        self.max_dec_len_this_time = max(self.seq_lens_dec)
        if self.use_mask_offset:
            self.mask_offset = paddle.full(self.batch_size * 2, 0, "int32")
            for i in range(self.batch_size):
                self.mask_offset[i * 4] = 0
                self.mask_offset[i * 4 + 1] = self.seq_lens_dec[i] + 1
                self.mask_offset[i * 4 + 2] = 0
                self.mask_offset[i * 4 + 3] = self.seq_lens_dec[i] + 2
            print("decoder mask_offset: ", self.mask_offset)
        # print("use_dynamic_quant: ", self.use_dynamic_quant)
        self.token_num = self.seq_len * self.batch_size

        qkv = paddle.rand([self.token_num, (self.q_num_head + 2 * self.kv_num_head) * self.dim_head], dtype=self.dtype)
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
            decode_block_shape_q,
            self.q_num_head // self.kv_num_head,
            self.blocksize,
        )

        paddle.device.synchronize()
        # breakpoint()
        qkv_out = decoder_write_cache_with_rope(
            qkv,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.block_tables,
            self.kv_batch_ids,
            self.kv_tile_ids_per_batch,
            self.kv_num_blocks_x_cpu,
            self.max_len_tensor_cpu,
            self.rope_emb,  # rope_emb
            None,  # qkv_bias
            self.key_cache_scale,  # cache_k_quant_scales
            self.value_cache_scale,  # cache_v_quant_scales
            self.key_cache_dequant_scale,  # cache_k_dequant_scales
            self.value_cache_dequant_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # kv_signal_data
            None,  # q_norm_weight
            None,  # k_norm_weight
            1e-6,
            "cache_fp8",
            self.use_neox_rotary_style,
            False,
            self.max_seq_len,
            448,  # quant_min_bound
            -448,  # quant_max_bound
            self.draft_token > 0,  # speculate_decoder
        )
        paddle.device.synchronize()
        # breakpoint()
        out = decode_append_attention(
            qkv_out,
            self.cache_k,
            self.cache_v,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.seq_lens_this_time,
            self.batch_id_per_token,
            self.cu_seqlens_q,
            self.block_tables,
            self.decoder_batch_ids,
            self.decoder_tile_ids_per_batch,
            self.decoder_num_blocks_cpu,
            self.max_len_tensor_cpu,
            None,  # attn_mask
            self.key_cache_scale,  # cache_k_quant_scales
            self.value_cache_scale,  # cache_v_quant_scales
            self.key_cache_dequant_scale,  # cache_k_dequant_scales
            self.value_cache_dequant_scale,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            self.mask_offset,  # mask_offset
            None,  # sinks
            "cache_fp8",
            self.max_seq_len,
            448,  # quant_min_bound
            -448,  # quant_max_bound
            decode_block_shape_q,  # decoder_block_shape_q
            max_partition_size,  # max_partition_size
            self.seq_len,  # speculate_max_draft_token_num
            True,  # causal
            self.draft_token > 0,  # speculate_decoder
            self.sliding_window,
        )



class TestAppendGroupQueryAttnWithRopeDyCfp8(TestAppendGroupQueryAttnWithRope):
    def setUp(self):
        self.name = "TestAppendGroupQueryAttnWithRopeDyCfp8"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 32
        self.q_num_head = 14
        self.kv_num_head = 1
        self.kv_seq_len = 12800
        self.draft_token = 1
        self.seq_len = 1 + self.draft_token

        self.max_dec_len = 10
        self.dim_head = 128
        self.q_hid_dim = self.q_num_head * self.dim_head
        self.kv_hid_dim = self.kv_num_head * self.dim_head
        self.blocksize = 64
        self.use_neox_rotary_style = False
        # max_seq_len = self.seq_len + self.max_dec_len
        self.max_seq_len = self.kv_seq_len + self.seq_len
        self.softmax_scale = self.dim_head**-0.5
        self.rope_theta = 10000
        self.sliding_window = 0
        self.dtype = "bfloat16"
        self.use_qk_norm = False
        self.use_mask_offset = False
        self.use_sinks = False
        self.use_yarn = False
        self.use_dynamic_quant = False
        self.init_tensor()


if __name__ == "__main__":
    tester = TestAppendGroupQueryAttnWithRopeDyCfp8()
    tester.setUp()
    for partition_size in [256, 512, 1024, 2048, 4096, 8192]:
        paddle.device.synchronize()
        tester.test(partition_size, 32)
    tester.test(32768, 32)
    # tester.test(1024, 32)
