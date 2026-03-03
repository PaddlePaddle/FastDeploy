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

"""
Tests for the c16 warp1_4 decoder kernel determinism fix.

The fix addresses two bugs in multiquery_attention_c16_impl.cuh:
  1. Outer: warp1_4 dispatcher lacked force_no_partition check (lines 1164-1175)
  2. Inner: nosplit kernel used runtime num_chunks_this_seq instead of
     compile-time partition_kv, causing nullptr writes (lines 545, 748, 772, 812)

This test exercises the c16 warp1_4 decoder path by:
  - dim_head=128, blocksize=64, cache_quant_type="none" -> c16 path
  - decoder_block_shape_q=16 -> NUM_WARP_Q=1 (warp1_4)
  - seq_lens_encoder=0, seq_lens_decoder>0 -> decoder mode
  - FD_DETERMINISTIC_MODE=1 -> force nosplit kernel
  - Small max_partition_size for decoder to ensure num_chunks > 1

Run:
  python -m pytest tests/deterministic/test_c16_warp1_4_determinism.py -v
"""

import copy
import os
import unittest

import numpy as np
import paddle

os.environ["FD_DETERMINISTIC_MODE"] = "1"

from fastdeploy.model_executor.layers.attention.ops import (  # noqa: E402
    append_attention,
    get_block_shape_and_split_kv_block,
)

SEED = 42


def make_rope_emb(max_seq_len, dim_head, base=10000):
    pos = paddle.arange(max_seq_len).reshape((1, -1))
    inv_freq = base ** (-paddle.arange(0, dim_head, 2, dtype="float32") / dim_head)
    freqs = paddle.einsum("ij,k->ijk", pos.cast("float32"), inv_freq)
    emb = freqs.reshape((1, max_seq_len, dim_head // 2)).unsqueeze(2)
    rope_emb = paddle.zeros((2, 1, max_seq_len, 1, dim_head // 2), dtype="float32")
    rope_emb[0] = paddle.cos(emb)
    rope_emb[1] = paddle.sin(emb)
    return rope_emb


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time, dtype="int32")
    cum_offsets = paddle.zeros(shape=(bsz + 1,), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = int(paddle.sum(seq_lens_this_time))
    batch_id_per_token = paddle.zeros(shape=(token_num,), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1,), dtype="int32")
    for i in range(bsz):
        sn = int(seq_lens_this_time[i])
        co = int(cum_offsets[i])
        for j in range(sn):
            batch_id_per_token[i * max_seq_len - co + j] = i
        cu_seqlens_q[i + 1] = (i + 1) * max_seq_len - int(cum_offsets[i + 1])
    return batch_id_per_token, cu_seqlens_q


def naive_attention_impl(query, key, value, cache_k, cache_v, scale):
    """Reference: Q @ K^T * scale -> softmax -> @ V, with GQA expansion."""
    batch, heads, seq_len, head_dim = query.shape
    kv_head = key.shape[1]
    g = heads // kv_head

    key = key.reshape([batch, kv_head, 1, seq_len, head_dim])
    key = paddle.tile(key, [1, 1, g, 1, 1]).reshape([batch, heads, seq_len, head_dim])
    value = value.reshape([batch, kv_head, 1, seq_len, head_dim])
    value = paddle.tile(value, [1, 1, g, 1, 1]).reshape([batch, heads, seq_len, head_dim])

    if cache_k is not None:
        ck = cache_k.reshape([batch, kv_head, 1, -1, head_dim])
        ck = paddle.tile(ck, [1, 1, g, 1, 1]).reshape([batch, heads, -1, head_dim])
        key = paddle.concat([ck, key], axis=2)
    if cache_v is not None:
        cv = cache_v.reshape([batch, kv_head, 1, -1, head_dim])
        cv = paddle.tile(cv, [1, 1, g, 1, 1]).reshape([batch, heads, -1, head_dim])
        value = paddle.concat([cv, value], axis=2)

    qk = paddle.matmul(query, key, transpose_y=True) * scale
    attn = paddle.nn.functional.softmax(qk, -1)
    return paddle.matmul(attn.cast(value.dtype), value)


def block_cache_to_naive(cache_k, cache_v, bsz, block_tables, seq_len):
    _, num_head, blocksize, dim_head = cache_k.shape
    ok = paddle.zeros([bsz, num_head, seq_len, dim_head], dtype=cache_k.dtype)
    ov = paddle.zeros([bsz, num_head, seq_len, dim_head], dtype=cache_v.dtype)
    for i in range(bsz):
        for j in range(seq_len):
            ok[i, :, j, :] = cache_k[block_tables[i, j // blocksize], :, j % blocksize, :]
            ov[i, :, j, :] = cache_v[block_tables[i, j // blocksize], :, j % blocksize, :]
    return ok, ov


def run_c16_warp14_decoder_test(
    batch_size,
    q_num_head,
    kv_num_head,
    dim_head,
    blocksize,
    prefill_seq_len,
    max_dec_len,
    dtype,
    decoder_max_partition_size,
    num_decode_runs,
):
    """
    Run encoder prefill + N decoder runs.
    Returns (list of decoder output numpy arrays, naive reference numpy array).
    """
    np.random.seed(SEED)
    paddle.seed(SEED)

    max_seq_len = prefill_seq_len + max_dec_len
    block_per_seq = (max_seq_len + blocksize - 1) // blocksize
    max_block_num = block_per_seq * batch_size
    scale = 1.0 / np.sqrt(dim_head)
    group_size = (q_num_head + 2 * kv_num_head) // kv_num_head

    rope_emb = make_rope_emb(max_seq_len, dim_head)

    # Block tables
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.zeros((batch_size, block_per_seq), dtype="int32")
    for i in range(batch_size):
        for j in range(block_per_seq):
            block_tables[i, j] = free_list.pop()

    cache_k = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head), dtype=dtype)
    cache_v = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head), dtype=dtype)

    # Tile metadata
    dtile = int(1024 * batch_size * np.ceil((2 * 10) / 12))
    dec_batch_ids = paddle.full([dtile], 0, dtype="int32")
    dec_tile_ids = paddle.full([dtile], 0, dtype="int32")
    dec_nblocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
    dec_nblocks_dev = paddle.full([1], 0, dtype="int32")
    dec_chunk_dev = paddle.full([1], decoder_max_partition_size, dtype="int32")
    max_len_cpu = paddle.full([8], 0, dtype="int32").cpu()
    enc_batch_ids = paddle.full([batch_size], 0, dtype="int32")
    enc_tile_ids = paddle.full([batch_size], 0, dtype="int32")
    enc_nblocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
    kv_batch_ids = paddle.full([batch_size], 0, dtype="int32")
    kv_tile_ids = paddle.full([batch_size], 0, dtype="int32")
    kv_nblocks_cpu = paddle.full([1], 0, dtype="int32").cpu()

    # ===== Encoder phase =====
    seq_enc = paddle.full([batch_size], prefill_seq_len, dtype="int32")
    seq_dec = paddle.full([batch_size], 0, dtype="int32")
    seq_this = copy.deepcopy(seq_enc)
    bid_enc, cu_enc = get_padding_offset(batch_size, prefill_seq_len, seq_this)
    token_num = batch_size * prefill_seq_len

    q_np = np.random.random([batch_size, q_num_head, prefill_seq_len, dim_head]).astype("float32") / 10
    k_np = np.random.random([batch_size, kv_num_head, prefill_seq_len, dim_head]).astype("float32") / 10
    v_np = np.random.random([batch_size, kv_num_head, prefill_seq_len, dim_head]).astype("float32") / 10

    q = paddle.to_tensor(q_np, dtype=dtype)
    k = paddle.to_tensor(k_np, dtype=dtype)
    v = paddle.to_tensor(v_np, dtype=dtype)
    qkv = paddle.concat(
        [
            q.transpose([0, 2, 1, 3]).reshape([token_num, q_num_head * dim_head]),
            k.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * dim_head]),
            v.transpose([0, 2, 1, 3]).reshape([token_num, kv_num_head * dim_head]),
        ],
        axis=1,
    )

    # Use large partition size for encoder to avoid issues in prefill
    encoder_partition_size = 32768

    get_block_shape_and_split_kv_block(
        seq_enc,
        seq_dec,
        seq_this,
        dec_batch_ids,
        dec_tile_ids,
        dec_nblocks_cpu,
        dec_nblocks_dev,
        dec_chunk_dev,
        max_len_cpu,
        enc_batch_ids,
        enc_tile_ids,
        enc_nblocks_cpu,
        kv_batch_ids,
        kv_tile_ids,
        kv_nblocks_cpu,
        encoder_partition_size,
        12,
        group_size,
        blocksize,
    )

    append_attention(
        qkv,
        cache_k,
        cache_v,
        seq_enc,
        seq_dec,
        seq_this,
        bid_enc,
        cu_enc,
        block_tables,
        enc_batch_ids,
        enc_tile_ids,
        enc_nblocks_cpu,
        kv_batch_ids,
        kv_tile_ids,
        kv_nblocks_cpu,
        dec_batch_ids,
        dec_tile_ids,
        dec_nblocks_cpu,
        max_len_cpu,
        rope_emb,
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
        "fp16",
        "none",
        False,
        False,
        max_seq_len,
        0.0,
        0.0,
        -1,
        64,  # encoder_block_shape_q
        16,  # decoder_block_shape_q
        encoder_partition_size,
        encoder_partition_size,
        2,
        True,
        False,
        0,
    )
    paddle.device.synchronize()

    # Extract naive KV cache for reference
    naive_ck, naive_cv = block_cache_to_naive(
        cache_k,
        cache_v,
        batch_size,
        block_tables,
        prefill_seq_len,
    )

    # ===== Decoder phase =====
    seq_enc_d = paddle.full([batch_size], 0, dtype="int32")
    seq_dec_d = paddle.full([batch_size], prefill_seq_len, dtype="int32")
    seq_this_d = paddle.full([batch_size], 1, dtype="int32")
    bid_dec, cu_dec = get_padding_offset(batch_size, 1, seq_this_d)

    dq_np = np.random.random([batch_size, q_num_head, 1, dim_head]).astype("float32") / 10
    dk_np = np.random.random([batch_size, kv_num_head, 1, dim_head]).astype("float32") / 10
    dv_np = np.random.random([batch_size, kv_num_head, 1, dim_head]).astype("float32") / 10
    dq = paddle.to_tensor(dq_np, dtype=dtype)
    dk = paddle.to_tensor(dk_np, dtype=dtype)
    dv = paddle.to_tensor(dv_np, dtype=dtype)
    dec_qkv = paddle.concat(
        [
            dq.transpose([0, 2, 1, 3]).reshape([batch_size, q_num_head * dim_head]),
            dk.transpose([0, 2, 1, 3]).reshape([batch_size, kv_num_head * dim_head]),
            dv.transpose([0, 2, 1, 3]).reshape([batch_size, kv_num_head * dim_head]),
        ],
        axis=1,
    )

    results = []
    for _ in range(num_decode_runs):
        cache_k_c = cache_k.clone()
        cache_v_c = cache_v.clone()
        qkv_c = dec_qkv.clone()

        get_block_shape_and_split_kv_block(
            seq_enc_d,
            seq_dec_d,
            seq_this_d,
            dec_batch_ids,
            dec_tile_ids,
            dec_nblocks_cpu,
            dec_nblocks_dev,
            dec_chunk_dev,
            max_len_cpu,
            enc_batch_ids,
            enc_tile_ids,
            enc_nblocks_cpu,
            kv_batch_ids,
            kv_tile_ids,
            kv_nblocks_cpu,
            decoder_max_partition_size,
            12,
            group_size,
            blocksize,
        )

        out = append_attention(
            qkv_c,
            cache_k_c,
            cache_v_c,
            seq_enc_d,
            seq_dec_d,
            seq_this_d,
            bid_dec,
            cu_dec,
            block_tables,
            enc_batch_ids,
            enc_tile_ids,
            enc_nblocks_cpu,
            kv_batch_ids,
            kv_tile_ids,
            kv_nblocks_cpu,
            dec_batch_ids,
            dec_tile_ids,
            dec_nblocks_cpu,
            max_len_cpu,
            rope_emb,
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
            "fp16",
            "none",
            False,
            False,
            max_seq_len,
            0.0,
            0.0,
            -1,
            64,  # encoder_block_shape_q
            16,  # decoder_block_shape_q
            decoder_max_partition_size,
            encoder_partition_size,
            2,
            True,
            False,
            0,
        )
        paddle.device.synchronize()
        results.append(out.numpy().copy())

    # Naive reference
    ref = naive_attention_impl(dq, dk, dv, naive_ck, naive_cv, scale)
    ref_np = ref.transpose([0, 2, 1, 3]).reshape([batch_size, q_num_head * dim_head]).numpy()

    return results, ref_np


class TestC16Warp14Determinism(unittest.TestCase):
    """
    Test the c16 warp1_4 decoder kernel under FD_DETERMINISTIC_MODE=1.

    Verifies:
      1. Correctness: output matches naive attention reference (rtol/atol=1e-2)
      2. Determinism: repeated runs with identical input -> bitwise-identical output
    """

    def test_short_kv_nosplit(self):
        """num_chunks=1 (short KV): basic nosplit path, partition_kv=false template."""
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=64,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=32768,
            num_decode_runs=5,
        )
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_allclose(results[0], ref, rtol=1e-2, atol=1e-2)

    def test_long_kv_multi_chunk(self):
        """
        num_chunks=4 (prefill=256, partition=64): the exact scenario the fix addresses.
        partition_kv=true template but grid_chunks=1 (deterministic).
        """
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=256,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=5,
        )
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_allclose(results[0], ref, rtol=1e-2, atol=1e-2)

    def test_multi_batch(self):
        """Multiple batches with multi-chunk decoder."""
        results, ref = run_c16_warp14_decoder_test(
            batch_size=4,
            q_num_head=8,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=256,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=3,
        )
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_allclose(results[0], ref, rtol=1e-2, atol=1e-2)

    def test_float16(self):
        """Float16 dtype with multi-chunk decoder."""
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=256,
            max_dec_len=32,
            dtype="float16",
            decoder_max_partition_size=64,
            num_decode_runs=3,
        )
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_allclose(results[0], ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
