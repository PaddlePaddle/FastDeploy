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
Test suite for the c16 warp1_4 decoder attention kernel determinism.

Background:
  The c16 warp1_4 decoder kernel (multiquery_attention_c16_impl.cuh) had two
  bugs that caused nondeterministic outputs under FD_DETERMINISTIC_MODE:
    1. The warp1_4 dispatcher (lines 1164-1175) lacked the force_no_partition
       check, so it still launched the multi-chunk (split) kernel even when
       deterministic mode requested the single-chunk (nosplit) path.
    2. The nosplit kernel template read runtime num_chunks_this_seq instead of
       compile-time partition_kv, causing out-of-bounds nullptr writes to the
       partial output buffer (lines 545, 748, 772, 812).

How the c16 warp1_4 path is triggered:
  - dim_head=128, blocksize=64, cache_quant_type="none"  -> selects c16 kernel
  - decoder_block_shape_q=16  -> NUM_WARP_Q=1, i.e. warp1_4 configuration
  - seq_lens_encoder=0, seq_lens_decoder>0               -> decoder mode
  - FD_DETERMINISTIC_MODE=1                               -> forces nosplit path
  - Small decoder_max_partition_size (e.g. 64) with long prefill (e.g. 256)
    ensures num_chunks > 1, which is the scenario that exposed the bugs.

Test items:
  1. test_short_kv_nosplit
     - Short KV (num_chunks=1): basic nosplit path with partition_kv=false.
     - Verifies both correctness (vs naive reference) and determinism (10 runs).

  2. test_long_kv_multi_chunk
     - Long KV (num_chunks=4, prefill=256, partition=64): the exact scenario
       the fix addresses. partition_kv=true template but grid_chunks=1.
     - Verifies correctness and determinism.

  3. test_multi_batch
     - Multiple batches (batch_size=4) with multi-chunk decoder.
     - Ensures the fix works across batch elements, not just single-batch.

  4. test_float16
     - Float16 dtype with multi-chunk decoder.
     - Ensures the fix is dtype-agnostic (not only bfloat16).

  5. test_unaligned_seq_len
     - prefill_seq_len not divisible by blocksize (100 % 64 != 0).
     - Catches off-by-one bugs in block/chunk boundary calculations.

  6. test_mha_no_gqa
     - MHA config: q_num_head == kv_num_head (no GQA grouping).
     - Ensures the fix is not GQA-specific.

  7. test_nosplit_vs_split_consistency
     - Cross-path check: deterministic nosplit vs non-deterministic split.
     - Both paths should produce numerically close results (rtol/atol=1e-2)
       and both should match the naive attention reference.

  8. test_partition_boundary
     - Edge case: prefill_seq_len equals partition_size (boundary condition).
     - Tests chunk calculation when num_chunks is exactly an integer.

  9. test_empty_kv
     - Edge case: decoder-only mode (no prefill, empty KV cache).
     - Tests scenario with no encoder prefill phase.

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

ENCODER_BLOCK_SHAPE_Q = 64
DECODER_BLOCK_SHAPE_Q = 16


def _assert_deterministic_and_correct(results, ref):
    """
    Helper to verify determinism and correctness.

    Args:
        results: List of output arrays from repeated runs
        ref: Reference output array from naive attention
    """
    # Verify determinism: all runs should produce identical output
    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Determinism failure: run 0 vs run {i}",
        )
    # Verify correctness: output should match naive reference
    np.testing.assert_allclose(results[0], ref, rtol=1e-2, atol=1e-2)


def make_rope_emb(max_seq_len, dim_head, base=10000):
    pos = paddle.arange(max_seq_len).reshape((1, -1))
    inv_freq = base ** (-paddle.arange(0, dim_head, 2, dtype="float32") / dim_head)
    freqs = paddle.einsum("ij,k->ijk", pos.cast("float32"), inv_freq)
    emb = freqs.reshape((1, max_seq_len, dim_head // 2)).unsqueeze(2)
    rope_emb = paddle.zeros((2, 1, max_seq_len, 1, dim_head // 2), dtype="float32")
    rope_emb[0] = paddle.cos(emb)
    rope_emb[1] = paddle.sin(emb)
    return rope_emb


def apply_rope(x, rope_emb, positions):
    """
    Apply rotary position embedding (non-neox interleaved style).

    x: (batch, heads, seq_len, dim_head)
    rope_emb: (2, 1, max_seq_len, 1, dim_head//2)
    positions: list of int, one position per seq index
    """
    dim_head = x.shape[-1]
    half = dim_head // 2
    x_f32 = x.cast("float32")
    out = x_f32.clone()

    for seq_idx, pos in enumerate(positions):
        cos_p = rope_emb[0, 0, pos, 0, :]  # (dim_head//2,)
        sin_p = rope_emb[1, 0, pos, 0, :]

        x_slice = x_f32[:, :, seq_idx, :]  # (batch, heads, dim_head)
        x_pairs = x_slice.reshape(list(x_slice.shape[:-1]) + [half, 2])
        x0 = x_pairs[..., 0]  # (batch, heads, half)
        x1 = x_pairs[..., 1]

        out0 = x0 * cos_p - x1 * sin_p
        out1 = x0 * sin_p + x1 * cos_p

        out[:, :, seq_idx, :] = paddle.stack([out0, out1], axis=-1).reshape(x_slice.shape)

    return out.cast(x.dtype)


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
    group_size = q_num_head // kv_num_head
    compute_type = "bf16" if dtype == "bfloat16" else "fp16"

    rope_emb = make_rope_emb(max_seq_len, dim_head)

    # Block tables
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.zeros((batch_size, block_per_seq), dtype="int32")
    for i in range(batch_size):
        for j in range(block_per_seq):
            block_tables[i, j] = free_list.pop()

    cache_k = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head), dtype=dtype)
    cache_v = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head), dtype=dtype)

    # Tile metadata buffers, sized per allocate_launch_related_buffer() formula
    gqa_ratio = q_num_head // kv_num_head
    decode_tile_size = int(1024 * batch_size * np.ceil((2 * gqa_ratio) / DECODER_BLOCK_SHAPE_Q))
    encode_tile_size = max(batch_size, batch_size * (max_seq_len * gqa_ratio // ENCODER_BLOCK_SHAPE_Q))
    kv_tile_size = max(batch_size, batch_size * (max_seq_len // blocksize))

    dec_batch_ids = paddle.full([decode_tile_size], 0, dtype="int32")
    dec_tile_ids = paddle.full([decode_tile_size], 0, dtype="int32")
    dec_nblocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
    dec_nblocks_dev = paddle.full([1], 0, dtype="int32")
    dec_chunk_dev = paddle.full([1], decoder_max_partition_size, dtype="int32")
    max_len_cpu = paddle.full([8], 0, dtype="int32").cpu()
    enc_batch_ids = paddle.full([encode_tile_size], 0, dtype="int32")
    enc_tile_ids = paddle.full([encode_tile_size], 0, dtype="int32")
    enc_nblocks_cpu = paddle.full([1], 0, dtype="int32").cpu()
    kv_batch_ids = paddle.full([kv_tile_size], 0, dtype="int32")
    kv_tile_ids = paddle.full([kv_tile_size], 0, dtype="int32")
    kv_nblocks_cpu = paddle.full([1], 0, dtype="int32").cpu()

    decoder_step_token_num = 2
    max_num_chunk = (max_seq_len + decoder_max_partition_size - 1) // decoder_max_partition_size
    tmp_workspace = paddle.full(
        [batch_size * decoder_step_token_num, max_num_chunk, q_num_head * dim_head],
        0,
        dtype=paddle.get_default_dtype(),
    )
    tmp_m = paddle.full([batch_size * decoder_step_token_num, max_num_chunk, q_num_head], 0, dtype="float32")
    tmp_d = paddle.full([batch_size * decoder_step_token_num, max_num_chunk, q_num_head], 0, dtype="float32")

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
        ENCODER_BLOCK_SHAPE_Q,
        DECODER_BLOCK_SHAPE_Q,
        group_size,
        blocksize,
    )

    append_attention(
        qkv,
        cache_k,
        cache_v,
        tmp_workspace,
        tmp_m,
        tmp_d,
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
        compute_type,
        "none",
        False,
        False,
        max_seq_len,
        0.0,
        0.0,
        -1,
        ENCODER_BLOCK_SHAPE_Q,
        DECODER_BLOCK_SHAPE_Q,
        decoder_max_partition_size,
        encoder_partition_size,
        2,
        True,
        False,
        0,
    )
    paddle.device.synchronize()

    # Extract naive KV cache for reference (already has RoPE applied by kernel)
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

    # Warmup: first decoder call on multi-chunk path may return zeros
    # due to kernel JIT compilation. Run once and discard.
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
        ENCODER_BLOCK_SHAPE_Q,
        DECODER_BLOCK_SHAPE_Q,
        group_size,
        blocksize,
    )
    append_attention(
        dec_qkv.clone(),
        cache_k.clone(),
        cache_v.clone(),
        tmp_workspace,
        tmp_m,
        tmp_d,
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
        compute_type,
        "none",
        False,
        False,
        max_seq_len,
        0.0,
        0.0,
        -1,
        ENCODER_BLOCK_SHAPE_Q,
        DECODER_BLOCK_SHAPE_Q,
        decoder_max_partition_size,
        encoder_partition_size,
        2,
        True,
        False,
        0,
    )
    paddle.device.synchronize()

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
            ENCODER_BLOCK_SHAPE_Q,
            DECODER_BLOCK_SHAPE_Q,
            group_size,
            blocksize,
        )

        out = append_attention(
            qkv_c,
            cache_k_c,
            cache_v_c,
            tmp_workspace,
            tmp_m,
            tmp_d,
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
            compute_type,
            "none",
            False,
            False,
            max_seq_len,
            0.0,
            0.0,
            -1,
            ENCODER_BLOCK_SHAPE_Q,
            DECODER_BLOCK_SHAPE_Q,
            decoder_max_partition_size,
            encoder_partition_size,
            2,
            True,
            False,
            0,
        )
        paddle.device.synchronize()
        results.append(out.numpy().copy())

    # Naive reference: apply RoPE to decoder Q/K at position prefill_seq_len
    # (cached K/V already have RoPE applied by the kernel during encoder phase)
    dq_rope = apply_rope(dq, rope_emb, [prefill_seq_len])
    dk_rope = apply_rope(dk, rope_emb, [prefill_seq_len])
    ref = naive_attention_impl(dq_rope, dk_rope, dv, naive_ck, naive_cv, scale)
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
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

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
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

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
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

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
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

    def test_unaligned_seq_len(self):
        """prefill_seq_len not divisible by blocksize (100 % 64 != 0)."""
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=100,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

    def test_mha_no_gqa(self):
        """MHA: q_num_head == kv_num_head (no GQA grouping)."""
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=8,
            kv_num_head=8,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=128,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

    def test_nosplit_vs_split_consistency(self):
        """
        Cross-path check: force_no_partition (deterministic) vs split (partitioned)
        should produce numerically close results.

        The two paths differ only in floating-point accumulation order:
          - nosplit: single chunk, sequential accumulation
          - split: multi-chunk parallel, then merge
        Difference should be within low-precision rounding (rtol/atol=1e-2 for bf16).

        Note: envs.FD_DETERMINISTIC_MODE is lazily evaluated via __getattr__,
        so runtime changes to os.environ["FD_DETERMINISTIC_MODE"] take effect.
        """
        # Run once in deterministic mode (already set at module level)
        det_results, det_ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=256,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=1,
        )

        # Run once in non-deterministic mode (split/partitioned path)
        os.environ["FD_DETERMINISTIC_MODE"] = "0"
        try:
            split_results, split_ref = run_c16_warp14_decoder_test(
                batch_size=1,
                q_num_head=16,
                kv_num_head=2,
                dim_head=128,
                blocksize=64,
                prefill_seq_len=256,
                max_dec_len=32,
                dtype="bfloat16",
                decoder_max_partition_size=64,
                num_decode_runs=1,
            )
        finally:
            os.environ["FD_DETERMINISTIC_MODE"] = "1"

        # Both paths should match the naive reference
        np.testing.assert_allclose(
            det_results[0], det_ref, rtol=1e-2, atol=1e-2, err_msg="Deterministic path vs naive reference"
        )
        np.testing.assert_allclose(
            split_results[0], split_ref, rtol=1e-2, atol=1e-2, err_msg="Split path vs naive reference"
        )

        # Cross-path: nosplit vs split should be close
        np.testing.assert_allclose(
            det_results[0],
            split_results[0],
            rtol=1e-2,
            atol=1e-2,
            err_msg="Deterministic (nosplit) vs split path divergence",
        )

    def test_partition_boundary(self):
        """
        Edge case: prefill_seq_len equals partition_size (boundary condition).

        This tests the scenario where num_chunks = prefill_seq_len / partition_size
        is exactly an integer, which is a boundary condition for chunk calculation.
        """
        results, ref = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=128,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=128,
            num_decode_runs=10,
        )
        _assert_deterministic_and_correct(results, ref)

    def test_empty_kv(self):
        """
        Edge case: decoder-only mode (no prefill, empty KV cache).

        This tests the scenario where there's no encoder prefill phase,
        only decoder with empty KV cache.
        """
        results, _ = run_c16_warp14_decoder_test(
            batch_size=1,
            q_num_head=16,
            kv_num_head=2,
            dim_head=128,
            blocksize=64,
            prefill_seq_len=0,
            max_dec_len=32,
            dtype="bfloat16",
            decoder_max_partition_size=64,
            num_decode_runs=10,
        )
        # For empty KV, we only check determinism as naive reference may not be valid
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0],
                results[i],
                err_msg=f"Determinism failure: run 0 vs run {i}",
            )


if __name__ == "__main__":
    unittest.main()
