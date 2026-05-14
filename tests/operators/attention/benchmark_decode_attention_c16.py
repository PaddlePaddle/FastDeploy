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

"""Benchmark script comparing append_attention vs decode_unified_attention (C16) performance.

Each case runs append_attention once and decode_attention once, prints elapsed time.
Supports --op flag to run only one op (for ncu profiling).

Usage:
    python benchmark_decode_attention.py                    # run all cases, both ops
    python benchmark_decode_attention.py --op append        # run only append_attention
    python benchmark_decode_attention.py --op decode        # run only decode_attention
    python benchmark_decode_attention.py --case 0           # run only case index 0
"""

import argparse
import copy
import time

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


def get_padding_offset(bsz, seq_lens_this_time):
    token_num = int(paddle.sum(seq_lens_this_time).item())
    seq_lens_list = seq_lens_this_time.numpy().tolist()
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    batch_id_per_token = np.zeros(token_num, dtype="int32")
    offset = 0
    for i in range(bsz):
        sl = int(seq_lens_list[i])
        batch_id_per_token[offset : offset + sl] = i
        offset += sl
        cu_seqlens_q[i + 1] = offset
        cu_seqlens_k[i + 1] = offset
    return paddle.to_tensor(batch_id_per_token, dtype="int32"), cu_seqlens_q, cu_seqlens_k


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


def build_block_tables(batch_size, max_model_len, block_size):
    block_num_per_seq = (max_model_len + block_size - 1) // block_size
    max_block_num = block_num_per_seq * batch_size
    # Assign each batch a contiguous range of blocks (descending order to match test)
    block_ids = np.arange(max_block_num - 1, -1, -1, dtype="int32").reshape(batch_size, block_num_per_seq)
    block_tables = paddle.to_tensor(block_ids, dtype="int32")
    return block_tables, max_block_num


def build_append_attention_buffers(batch_size, max_model_len, group_size, block_size):
    max_num_block_dec = batch_size * (max_model_len * group_size + 16 - 1) // 16
    decoder_batch_ids = paddle.full([max_num_block_dec], 0, dtype="int32")
    decoder_tile_ids_per_batch = paddle.full([max_num_block_dec], 0, dtype="int32")
    decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
    decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
    decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
    max_num_block = batch_size * (max_model_len * group_size + 64 - 1) // 64
    encoder_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
    encoder_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
    encoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
    kv_batch_ids = paddle.full([max_num_block], 0, dtype="int32")
    kv_tile_ids_per_batch = paddle.full([max_num_block], 0, dtype="int32")
    kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
    max_len_tensor_cpu = paddle.full([6], 0, dtype="int32").cpu()
    return {
        "decoder_batch_ids": decoder_batch_ids,
        "decoder_tile_ids_per_batch": decoder_tile_ids_per_batch,
        "decoder_num_blocks_cpu": decoder_num_blocks_cpu,
        "decoder_num_blocks_device": decoder_num_blocks_device,
        "decoder_chunk_size_device": decoder_chunk_size_device,
        "encoder_batch_ids": encoder_batch_ids,
        "encoder_tile_ids_per_batch": encoder_tile_ids_per_batch,
        "encoder_num_blocks_cpu": encoder_num_blocks_cpu,
        "kv_batch_ids": kv_batch_ids,
        "kv_tile_ids_per_batch": kv_tile_ids_per_batch,
        "kv_num_blocks_x_cpu": kv_num_blocks_x_cpu,
        "max_len_tensor_cpu": max_len_tensor_cpu,
    }


def build_decode_attention_buffers(
    batch_size, max_model_len, kv_num_head, q_num_head, head_dim, max_tokens_per_batch, group_size, dtype
):
    buffer = {}
    min_chunk_size = 128
    max_num_chunk = (max_model_len + min_chunk_size - 1) // min_chunk_size
    q_tile_size = 16
    q_tile_num = (max_tokens_per_batch * group_size + q_tile_size - 1) // q_tile_size
    buffer["max_len_tensor_cpu"] = paddle.full([6], 0, dtype="int32").cpu()
    buffer["block_indices"] = paddle.full([batch_size * kv_num_head * max_num_chunk * q_tile_num, 4], 0, dtype="int32")
    buffer["num_blocks"] = paddle.full([1], 0, dtype="int32")
    buffer["chunk_size"] = paddle.full([1], 0, dtype="int32")
    buffer["tmp_workspace"] = paddle.full(
        [batch_size * max_tokens_per_batch, max_num_chunk, q_num_head * head_dim],
        0,
        dtype=dtype,
    )
    buffer["tmp_m"] = paddle.full([batch_size * max_tokens_per_batch, max_num_chunk, q_num_head], 0, dtype="float32")
    buffer["tmp_d"] = paddle.full([batch_size * max_tokens_per_batch, max_num_chunk, q_num_head], 0, dtype="float32")
    return buffer


class BenchmarkCase:
    def __init__(
        self,
        name,
        batch_size,
        q_num_head,
        kv_num_head,
        head_dim,
        seq_len,
        max_model_len,
        dtype="bfloat16",
        max_tokens_per_batch=1,
        block_size=64,
        causal=True,
    ):
        self.name = name
        self.batch_size = batch_size
        self.q_num_head = q_num_head
        self.kv_num_head = kv_num_head
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.max_tokens_per_batch = max_tokens_per_batch
        self.block_size = block_size
        self.causal = causal
        self.group_size = q_num_head // kv_num_head
        self.cache_quant_type = "none"

    def short_name(self):
        return self.name


CASES = [
    # BenchmarkCase(
    #     "bs1_seq64",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=64,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq512",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=512,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq2048",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2048,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq4096",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4096,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq8192",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8192,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq16k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16383,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq32k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32767,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq64k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65535,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq128k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=131071,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq64",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=64,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq512",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=512,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq2048",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2048,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq4096",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4096,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq8192",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8192,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq16k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16383,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq32k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32767,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq64k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65535,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs16_seq128k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=131071,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq64",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=64,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq512",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=512,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq2048",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2048,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq4096",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4096,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq8192",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8192,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq16k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16383,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq32k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32767,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq64k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65535,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs128_seq128k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=131071,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq64",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=64,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq512",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=512,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq2048",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2048,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq4096",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4096,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq8192",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8192,
    #     max_model_len=22528,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq16k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16383,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq32k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32767,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs256_seq64k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65535,
    #     max_model_len=131072,
    #     max_tokens_per_batch=1,
    # ),
    # BenchmarkCase(
    #     "bs1_seq64",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=62,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq512",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=510,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq2048",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2046,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq4096",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4094,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq8192",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8190,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq16k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16382,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq32k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32766,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq64k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65534,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs1_seq128k",
    #     batch_size=1,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=131070,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq64",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=62,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq512",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=510,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq2048",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2046,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq4096",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4094,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq8192",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8190,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq16k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16383,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq32k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32766,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq64k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65534,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs16_seq128k",
    #     batch_size=16,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=131070,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq64",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=62,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq512",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=510,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq2048",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2046,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq4096",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4096,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq8192",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8190,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq16k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16382,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs128_seq32k",
    #     batch_size=128,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32767,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    BenchmarkCase(
        "bs128_seq64k",
        batch_size=128,
        q_num_head=12,
        kv_num_head=1,
        head_dim=128,
        seq_len=65534,
        max_model_len=131072,
        max_tokens_per_batch=2,
    ),
    BenchmarkCase(
        "bs128_seq128k",
        batch_size=128,
        q_num_head=12,
        kv_num_head=1,
        head_dim=128,
        seq_len=131070,
        max_model_len=131072,
        max_tokens_per_batch=2,
    ),
    # BenchmarkCase(
    #     "bs256_seq64",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=62,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq512",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=510,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq2048",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=2046,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq4096",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=4094,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq8192",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=8190,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq16k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=16382,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq32k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=32766,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
    # BenchmarkCase(
    #     "bs256_seq64k",
    #     batch_size=256,
    #     q_num_head=12,
    #     kv_num_head=1,
    #     head_dim=128,
    #     seq_len=65534,
    #     max_model_len=131072,
    #     max_tokens_per_batch=2,
    # ),
]


def make_cache(case, block_tables):
    """Allocate zero-filled KV cache (no prefill, saves memory for long sequences)."""
    max_block_num = block_tables.shape[0] * block_tables.shape[1]
    cache_shape = (max_block_num, case.kv_num_head, case.block_size, case.head_dim)
    cache_k = paddle.zeros(shape=cache_shape, dtype=case.dtype)
    cache_v = paddle.zeros(shape=cache_shape, dtype=case.dtype)
    return cache_k, cache_v


def do_prefill(case, block_tables, rotary_embs, place):
    """Run prefill and return cache_k, cache_v after prefill."""
    max_block_num = block_tables.shape[0] * block_tables.shape[1]
    cache_shape = (max_block_num, case.kv_num_head, case.block_size, case.head_dim)
    cache_k = paddle.zeros(shape=cache_shape, dtype=case.dtype)
    cache_v = paddle.zeros(shape=cache_shape, dtype=case.dtype)

    _, _, _, enc_qkv = get_qkv_and_qkv_concat_tensor(
        case.batch_size,
        case.q_num_head,
        case.kv_num_head,
        case.seq_len,
        case.head_dim,
        place,
        case.dtype,
    )

    enc_seq_lens_encoder = paddle.to_tensor([case.seq_len] * case.batch_size, "int32")
    enc_seq_lens_decoder = paddle.to_tensor([0] * case.batch_size, "int32")
    enc_seq_lens_this_time = copy.deepcopy(enc_seq_lens_encoder)
    enc_batch_id_per_token, enc_cu_seqlens_q, _ = get_padding_offset(case.batch_size, enc_seq_lens_this_time)

    buffers = build_append_attention_buffers(case.batch_size, case.max_model_len, case.group_size, case.block_size)
    get_block_shape_and_split_kv_block(
        enc_seq_lens_encoder,
        enc_seq_lens_decoder,
        enc_seq_lens_this_time,
        buffers["decoder_batch_ids"],
        buffers["decoder_tile_ids_per_batch"],
        buffers["decoder_num_blocks_cpu"],
        buffers["decoder_num_blocks_device"],
        buffers["decoder_chunk_size_device"],
        buffers["max_len_tensor_cpu"],
        buffers["encoder_batch_ids"],
        buffers["encoder_tile_ids_per_batch"],
        buffers["encoder_num_blocks_cpu"],
        buffers["kv_batch_ids"],
        buffers["kv_tile_ids_per_batch"],
        buffers["kv_num_blocks_x_cpu"],
        64,
        16,
        case.group_size,
        case.block_size,
    )

    append_attention_op(
        enc_qkv,
        cache_k,
        cache_v,
        enc_seq_lens_encoder,
        enc_seq_lens_decoder,
        enc_seq_lens_this_time,
        enc_batch_id_per_token,
        enc_cu_seqlens_q,
        block_tables,
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
        rotary_embs,
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
        None,
        None,
        None,
        None,
        1e-6,
        "bf16",
        case.cache_quant_type,
        False,
        False,
        case.max_model_len,
        0.0,
        0.0,
        -1,
        64,
        16,
        1024,
        case.max_model_len,
        case.max_tokens_per_batch,
        case.causal,
        case.max_tokens_per_batch > 1,
    )
    return cache_k, cache_v


def get_decode_inputs(case, place):
    """Return decode qkv and seq_lens tensors."""
    _, _, _, dec_qkv = get_qkv_and_qkv_concat_tensor(
        case.batch_size,
        case.q_num_head,
        case.kv_num_head,
        case.max_tokens_per_batch,
        case.head_dim,
        place,
        case.dtype,
    )
    dec_seq_lens_encoder = paddle.to_tensor([0] * case.batch_size, "int32")
    dec_seq_lens_decoder = paddle.to_tensor([case.seq_len] * case.batch_size, "int32")
    dec_seq_lens_this_time = paddle.to_tensor([case.max_tokens_per_batch] * case.batch_size, "int32")
    dec_batch_id_per_token, dec_cu_seqlens_q, _ = get_padding_offset(case.batch_size, dec_seq_lens_this_time)
    return (
        dec_qkv,
        dec_seq_lens_encoder,
        dec_seq_lens_decoder,
        dec_seq_lens_this_time,
        dec_batch_id_per_token,
        dec_cu_seqlens_q,
    )


def run_append_attention(
    case,
    cache_k,
    cache_v,
    dec_qkv,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    batch_id_per_token,
    cu_seqlens_q,
    block_tables,
    rotary_embs,
):
    buffers = build_append_attention_buffers(case.batch_size, case.max_model_len, case.group_size, case.block_size)
    get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        buffers["decoder_batch_ids"],
        buffers["decoder_tile_ids_per_batch"],
        buffers["decoder_num_blocks_cpu"],
        buffers["decoder_num_blocks_device"],
        buffers["decoder_chunk_size_device"],
        buffers["max_len_tensor_cpu"],
        buffers["encoder_batch_ids"],
        buffers["encoder_tile_ids_per_batch"],
        buffers["encoder_num_blocks_cpu"],
        buffers["kv_batch_ids"],
        buffers["kv_tile_ids_per_batch"],
        buffers["kv_num_blocks_x_cpu"],
        64,
        16,
        case.group_size,
        case.block_size,
    )
    out = append_attention_op(
        dec_qkv,
        cache_k,
        cache_v,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
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
        rotary_embs,
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
        None,
        None,
        None,
        None,
        1e-6,
        "bf16",
        case.cache_quant_type,
        False,
        False,
        case.max_model_len,
        0.0,
        0.0,
        -1,
        64,
        16,
        1024,
        case.max_model_len,
        case.max_tokens_per_batch,
        case.causal,
        case.max_tokens_per_batch > 1,
    )
    return out


def run_decode_attention(
    case,
    cache_k,
    cache_v,
    dec_qkv,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    batch_id_per_token,
    cu_seqlens_q,
    block_tables,
    rotary_embs,
):
    buffer = build_decode_attention_buffers(
        case.batch_size,
        case.max_model_len,
        case.kv_num_head,
        case.q_num_head,
        case.head_dim,
        case.max_tokens_per_batch,
        case.group_size,
        case.dtype,
    )
    config_for_attention(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        buffer["block_indices"],
        buffer["num_blocks"],
        buffer["chunk_size"],
        buffer["max_len_tensor_cpu"],
        case.cache_quant_type,
        case.group_size,
        case.kv_num_head,
        case.max_tokens_per_batch,
    )
    dec_cache_k = cache_k
    dec_cache_v = cache_v
    dec_qkv_copy = dec_qkv
    decoder_write_cache_with_rope(
        dec_qkv_copy,
        dec_cache_k,
        dec_cache_v,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        buffer["max_len_tensor_cpu"],
        rotary_embs,
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
        1e-6,
        case.cache_quant_type,
        False,
        False,
        case.max_model_len,
        0.0,
        0.0,
        case.max_tokens_per_batch > 1,
    )
    out = decode_unified_attention(
        dec_qkv_copy,
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
        block_tables,
        buffer["block_indices"],
        buffer["num_blocks"],
        buffer["chunk_size"],
        buffer["max_len_tensor_cpu"],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        paddle.empty([dec_qkv_copy.shape[0], case.q_num_head * case.head_dim], dtype=dec_qkv_copy.dtype),  # fmha_out
        case.cache_quant_type,
        case.max_model_len,
        0.0,
        0.0,
        case.max_tokens_per_batch,
        case.causal,
    )
    return out


# Cache rope embeddings keyed by max_model_len to avoid recomputation
_rope_cache = {}


def _get_rotary_embs(max_model_len, head_dim):
    key = (max_model_len, head_dim)
    if key not in _rope_cache:
        rope = RopeEmbedding()
        tmp_position_ids = paddle.arange(max_model_len).reshape((1, -1))
        _rope_cache[key] = rope.get_rotary_position_embedding(tmp_position_ids, head_dim)
    return _rope_cache[key]


def benchmark_case(case, op="both"):
    """Run a single case: prefill per op to minimize peak memory."""
    paddle.disable_static()
    place = paddle.CUDAPlace(0)

    rotary_embs = _get_rotary_embs(case.max_model_len, case.head_dim)
    block_tables, max_block_num = build_block_tables(case.batch_size, case.max_model_len, case.block_size)
    dec_qkv, dec_sle, dec_sld, dec_slt, dec_bid, dec_csq = get_decode_inputs(case, place)

    results = {}

    if op in ("both", "append"):
        cache_k, cache_v = make_cache(case, block_tables)
        paddle.device.cuda.synchronize()
        t0 = time.perf_counter()
        run_append_attention(
            case,
            cache_k,
            cache_v,
            dec_qkv,
            dec_sle,
            dec_sld,
            dec_slt,
            dec_bid,
            dec_csq,
            block_tables,
            rotary_embs,
        )
        paddle.device.cuda.synchronize()
        results["append"] = (time.perf_counter() - t0) * 1000
        del cache_k, cache_v
        paddle.device.cuda.empty_cache()

    if op in ("both", "decode"):
        cache_k, cache_v = make_cache(case, block_tables)
        paddle.device.cuda.synchronize()
        t0 = time.perf_counter()
        run_decode_attention(
            case,
            cache_k,
            cache_v,
            dec_qkv,
            dec_sle,
            dec_sld,
            dec_slt,
            dec_bid,
            dec_csq,
            block_tables,
            rotary_embs,
        )
        paddle.device.cuda.synchronize()
        results["decode"] = (time.perf_counter() - t0) * 1000
        del cache_k, cache_v
        paddle.device.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark append_attention vs decode_unified_attention")
    parser.add_argument(
        "--op", choices=["both", "append", "decode"], default="both", help="Which op to run (default: both)"
    )
    parser.add_argument(
        "--case", type=int, default=-1, help="Run only case by index (0-based). Default: run all cases."
    )
    args = parser.parse_args()

    cases = CASES if args.case < 0 else [CASES[args.case]]

    if args.op == "both":
        print(f"{'Case':<25} {'append_attn (ms)':>18} {'decode_attn (ms)':>18} {'Ratio':>10}")
    elif args.op == "append":
        print(f"{'Case':<25} {'append_attn (ms)':>18}")
    else:
        print(f"{'Case':<25} {'decode_attn (ms)':>18}")
    print("-" * 75)

    for case in cases:
        results = benchmark_case(case, op=args.op)
        if args.op == "both":
            a, d = results["append"], results["decode"]
            ratio = a / d if d > 0 else float("inf")
            print(f"{case.short_name():<25} {a:>18.3f} {d:>18.3f} {ratio:>10.2f}x")
        elif args.op == "append":
            print(f"{case.short_name():<25} {results['append']:>18.3f}")
        else:
            print(f"{case.short_name():<25} {results['decode']:>18.3f}")


if __name__ == "__main__":
    main()
