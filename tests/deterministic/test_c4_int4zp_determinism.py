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
Test suite for the c4 (cache_int4_zp) decoder attention kernel determinism.

Background:
  The c4 kernel (multiquery_attention_c4_impl.cuh) uses force_no_partition
  (line 1129) to ensure deterministic floating-point accumulation when
  FD_DETERMINISTIC_MODE=1. This test verifies that the nosplit path
  is correctly dispatched for the INT4 with zero-point quantized KV cache variant.

How the c4 INT4-ZP path is triggered:
  - dim_head=128, blocksize=64, cache_quant_type="cache_int4_zp" -> selects c4 kernel
  - Cache shape uses dim_head // 2 (INT4 packs two values per byte)
  - decoder_block_shape_q=16 -> decoder mode
  - FD_DETERMINISTIC_MODE=1 -> forces nosplit path
  - Small decoder_max_partition_size with long prefill ensures num_chunks > 1

Test items:
  1. test_short_kv_nosplit
     - Short KV (num_chunks=1): basic nosplit path.
     - Verifies determinism (10 runs bitwise identical).

  2. test_long_kv_multi_chunk
     - Long KV (num_chunks>1): the scenario force_no_partition addresses.
     - Verifies determinism.

  3. test_multi_batch
     - Multiple batches (batch_size=4) with multi-chunk decoder.

  4. test_float16
     - Float16 dtype with multi-chunk decoder.

Run:
  python -m pytest tests/deterministic/test_c4_int4zp_determinism.py -v
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


def _assert_deterministic(results):
    """Verify all runs produce bitwise-identical output."""
    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Determinism failure: run 0 vs run {i}",
        )


def run_c4_int4zp_decoder_test(
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
    Run encoder prefill + N decoder runs with cache_int4_zp quantized KV cache.
    Returns list of decoder output numpy arrays.
    """
    np.random.seed(SEED)
    paddle.seed(SEED)

    max_seq_len = prefill_seq_len + max_dec_len
    block_per_seq = (max_seq_len + blocksize - 1) // blocksize
    max_block_num = block_per_seq * batch_size
    group_size = q_num_head // kv_num_head
    compute_type = "bf16" if dtype == "bfloat16" else "fp16"

    rope_emb = make_rope_emb(max_seq_len, dim_head)

    # Block tables
    free_list = list(range(max_block_num - 1, -1, -1))
    block_tables = paddle.zeros((batch_size, block_per_seq), dtype="int32")
    for i in range(batch_size):
        for j in range(block_per_seq):
            block_tables[i, j] = free_list.pop()

    # INT4 quantized KV cache: uint8 dtype, last dim is dim_head // 2 (two int4 values per byte)
    cache_k = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head // 2), dtype="uint8")
    cache_v = paddle.zeros((max_block_num, kv_num_head, blocksize, dim_head // 2), dtype="uint8")

    # Scale and zero-point tensors: channel-wise, shape [kv_num_head * dim_head]
    scale_shape = [kv_num_head * dim_head]
    k_quant_scale = paddle.ones(scale_shape, dtype=dtype)
    v_quant_scale = paddle.ones(scale_shape, dtype=dtype)
    k_dequant_scale = paddle.ones(scale_shape, dtype=dtype)
    v_dequant_scale = paddle.ones(scale_shape, dtype=dtype)
    cache_k_zp = paddle.zeros(scale_shape, dtype=dtype)
    cache_v_zp = paddle.zeros(scale_shape, dtype=dtype)

    # Tile metadata buffers
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
        None,  # attn_mask
        None,  # qkv_bias
        None,  # qkv_scale
        k_quant_scale,
        v_quant_scale,
        k_dequant_scale,
        v_dequant_scale,
        cache_k_zp,
        cache_v_zp,
        None,  # linear_shift
        None,  # linear_smooth
        None,  # mask_offset
        None,  # kv_signal_data
        None,  # q_norm_weight
        None,  # k_norm_weight
        None,  # sinks
        1e-6,
        compute_type,
        "cache_int4_zp",
        False,  # use_neox_rotary_style
        False,  # rope_3d
        max_seq_len,
        7.0,  # quant_max_bound
        -7.0,  # quant_min_bound
        -1,  # out_linear_in_scale
        ENCODER_BLOCK_SHAPE_Q,
        DECODER_BLOCK_SHAPE_Q,
        encoder_partition_size,
        encoder_partition_size,
        2,  # speculate_max_draft_token_num
        True,  # causal
        False,  # speculate_decoder
        0,  # sliding_window
    )
    paddle.device.synchronize()

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

    # Warmup: first decoder call may return zeros due to kernel JIT compilation
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
        k_quant_scale,
        v_quant_scale,
        k_dequant_scale,
        v_dequant_scale,
        cache_k_zp,
        cache_v_zp,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1e-6,
        compute_type,
        "cache_int4_zp",
        False,
        False,
        max_seq_len,
        7.0,
        -7.0,
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
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            cache_k_zp,
            cache_v_zp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1e-6,
            compute_type,
            "cache_int4_zp",
            False,
            False,
            max_seq_len,
            7.0,
            -7.0,
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

    return results


class TestC4Int4ZPDeterminism(unittest.TestCase):
    """
    Test the c4 INT4-ZP decoder kernel under FD_DETERMINISTIC_MODE=1.

    Verifies determinism: repeated runs with identical input -> bitwise-identical output.
    No naive reference comparison (quantization makes exact reference complex).
    """

    def test_short_kv_nosplit(self):
        """num_chunks=1 (short KV): basic nosplit path."""
        results = run_c4_int4zp_decoder_test(
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
        _assert_deterministic(results)

    def test_long_kv_multi_chunk(self):
        """num_chunks>1: force_no_partition forces nosplit for determinism."""
        results = run_c4_int4zp_decoder_test(
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
        _assert_deterministic(results)

    def test_multi_batch(self):
        """Multiple batches with multi-chunk decoder."""
        results = run_c4_int4zp_decoder_test(
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
        _assert_deterministic(results)

    def test_float16(self):
        """Float16 dtype with multi-chunk decoder."""
        results = run_c4_int4zp_decoder_test(
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
        _assert_deterministic(results)


if __name__ == "__main__":
    unittest.main()
