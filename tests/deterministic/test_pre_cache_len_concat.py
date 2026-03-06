"""
Tests for pre_cache_len_concat CUDA custom op.

Covers ACTION-4 scenarios (A1-A8):
  A1: Pure prefill, no prefix (prefix=0, extend=512)
  A2: Prefix hit (prefix=384, extend=128)
  A3: Multi-sequence unequal prefix (bs=3)
  A4: Mixed batch (prefill + decode)
  A5: Prefix exactly divisible by block_size
  A6: Prefix not divisible by block_size
  A7: Large batch (bs=32, random prefix/extend, mixed prefill+decode)
  A8: All-decode batch

Plus: boundary tests, structural assertions, error path tests.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_pre_cache_len_concat.py -v -s
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.ops.pre_cache_len_concat import (
    pre_cache_len_concat,
)

# ---------------------------------------------------------------------------
# Python reference implementation (matches CUDA kernel logic exactly)
# ---------------------------------------------------------------------------


def ref_pre_cache_len_concat(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_size):
    """
    Python reference matching the CUDA kernel logic in pre_cache_len_concat.cu.

    Key logic from CUDA:
      for each bid:
        cache_len = seq_lens_decoder[bid] if seq_lens_encoder[bid] > 0 else 0
        loop_times = div_up(cache_len, block_size)
        for tile_id in range(loop_times):
            batch_ids.append(bid), tile_ids.append(tile_id)
        total_tokens += cache_len + seq_lens_this_time[bid]
        cu_seqlens_k[bid+1] = total_tokens
    """
    bsz = len(seq_lens_this_time)
    cu_seqlens_k = np.zeros(bsz + 1, dtype=np.int32)
    batch_ids = []
    tile_ids_per_batch = []
    total_tokens = 0
    gridx = 0

    for bid in range(bsz):
        enc_len = int(seq_lens_encoder[bid])
        cache_len = int(seq_lens_decoder[bid]) if enc_len > 0 else 0
        q_len = int(seq_lens_this_time[bid])
        loop_times = (cache_len + block_size - 1) // block_size  # div_up
        for tile_id in range(loop_times):
            batch_ids.append(bid)
            tile_ids_per_batch.append(tile_id)
        gridx += loop_times
        total_tokens += cache_len + q_len
        cu_seqlens_k[bid + 1] = total_tokens

    return (
        cu_seqlens_k,
        np.array(batch_ids, dtype=np.int32),
        np.array(tile_ids_per_batch, dtype=np.int32),
        np.array([gridx], dtype=np.int32),
        np.array([total_tokens], dtype=np.int32),
    )


def _run_and_compare(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size):
    """Run CUDA op, compare against reference, verify structural invariants."""
    enc_t = paddle.to_tensor(seq_lens_encoder, dtype="int32")
    dec_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
    stt_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

    outputs = pre_cache_len_concat(enc_t, dec_t, stt_t, max_dec_len, block_size)
    cu_seqlens_k, batch_ids, tile_ids, num_blocks_cpu, kv_token_num_cpu = [o.numpy() for o in outputs]

    ref = ref_pre_cache_len_concat(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_size)
    ref_cu, ref_batch_ids, ref_tile_ids, ref_num_blocks, ref_kv_token_num = ref

    # cu_seqlens_k: exact match
    np.testing.assert_array_equal(cu_seqlens_k, ref_cu, err_msg="cu_seqlens_k mismatch")
    # cu_seqlens_k: must be monotonically non-decreasing and start at 0
    assert cu_seqlens_k[0] == 0, f"cu_seqlens_k[0] should be 0, got {cu_seqlens_k[0]}"
    assert np.all(np.diff(cu_seqlens_k) >= 0), "cu_seqlens_k must be monotonically non-decreasing"

    # batch_ids / tile_ids (only first num_blocks entries are meaningful)
    n = int(ref_num_blocks[0])
    np.testing.assert_array_equal(batch_ids[:n], ref_batch_ids, err_msg="batch_ids mismatch")
    np.testing.assert_array_equal(tile_ids[:n], ref_tile_ids, err_msg="tile_ids_per_batch mismatch")

    # scalar outputs
    assert num_blocks_cpu[0] == ref_num_blocks[0], f"num_blocks mismatch: {num_blocks_cpu[0]} vs {ref_num_blocks[0]}"
    assert (
        kv_token_num_cpu[0] == ref_kv_token_num[0]
    ), f"kv_token_num mismatch: {kv_token_num_cpu[0]} vs {ref_kv_token_num[0]}"

    # num_blocks must equal the actual number of valid batch_ids entries
    assert num_blocks_cpu[0] == len(
        ref_batch_ids
    ), f"num_blocks ({num_blocks_cpu[0]}) != len(batch_ids) ({len(ref_batch_ids)})"


# ---------------------------------------------------------------------------
# Test class — normal path (A1-A8) + boundary + error path
# ---------------------------------------------------------------------------


class TestPreCacheLenConcat:

    # A1: Pure prefill, no prefix (prefix=0, extend=512)
    def test_a1_pure_prefill_no_prefix(self):
        """bs=1, encoder=512 (prefill), decoder=0 (no prefix cache), this_time=512."""
        _run_and_compare(
            seq_lens_encoder=np.array([512], dtype=np.int32),
            seq_lens_decoder=np.array([0], dtype=np.int32),
            seq_lens_this_time=np.array([512], dtype=np.int32),
            max_dec_len=0,
            block_size=64,
        )

    # A2: Prefix hit (prefix=384, extend=128)
    def test_a2_prefix_hit(self):
        """bs=1, encoder=128 (extend), decoder=384 (prefix cached), this_time=128."""
        _run_and_compare(
            seq_lens_encoder=np.array([128], dtype=np.int32),
            seq_lens_decoder=np.array([384], dtype=np.int32),
            seq_lens_this_time=np.array([128], dtype=np.int32),
            max_dec_len=384,
            block_size=64,
        )

    # A3: Multi-sequence unequal prefix (bs=3)
    def test_a3_multi_seq_unequal_prefix(self):
        """bs=3 with different prefix lengths."""
        _run_and_compare(
            seq_lens_encoder=np.array([256, 128, 64], dtype=np.int32),
            seq_lens_decoder=np.array([0, 256, 128], dtype=np.int32),
            seq_lens_this_time=np.array([256, 128, 64], dtype=np.int32),
            max_dec_len=256,
            block_size=64,
        )

    # A4: Mixed batch (prefill + decode)
    def test_a4_mixed_batch(self):
        """bs=4: 2 prefill + 2 decode. Decode sequences have encoder=0."""
        _run_and_compare(
            seq_lens_encoder=np.array([512, 128, 0, 0], dtype=np.int32),
            seq_lens_decoder=np.array([0, 384, 100, 200], dtype=np.int32),
            seq_lens_this_time=np.array([512, 128, 1, 1], dtype=np.int32),
            max_dec_len=384,
            block_size=64,
        )

    # A5: Prefix exactly divisible by block_size
    def test_a5_prefix_exact_block(self):
        """prefix=256, block_size=64 -> 256/64 = 4 blocks exactly."""
        _run_and_compare(
            seq_lens_encoder=np.array([128], dtype=np.int32),
            seq_lens_decoder=np.array([256], dtype=np.int32),
            seq_lens_this_time=np.array([128], dtype=np.int32),
            max_dec_len=256,
            block_size=64,
        )

    # A6: Prefix not divisible by block_size
    def test_a6_prefix_not_aligned(self):
        """prefix=300, block_size=64 -> ceil(300/64) = 5 blocks."""
        _run_and_compare(
            seq_lens_encoder=np.array([100], dtype=np.int32),
            seq_lens_decoder=np.array([300], dtype=np.int32),
            seq_lens_this_time=np.array([100], dtype=np.int32),
            max_dec_len=300,
            block_size=64,
        )

    # A7: Large batch (bs=32, random prefix/extend, mixed prefill+decode)
    def test_a7_large_batch(self):
        """bs=32, random lengths, ~25% are decode-only (encoder=0)."""
        rng = np.random.RandomState(42)
        bs = 32
        block_size = 64

        prefixes = rng.randint(0, 512, size=bs).astype(np.int32)
        extends = rng.randint(1, 256, size=bs).astype(np.int32)

        # Make ~25% of sequences decode-only (encoder=0)
        decode_mask = rng.rand(bs) < 0.25
        encoder_lens = extends.copy()
        encoder_lens[decode_mask] = 0
        this_time = extends.copy()
        this_time[decode_mask] = 1  # decode sequences produce 1 token

        max_dec = int(np.max(prefixes))

        _run_and_compare(
            seq_lens_encoder=encoder_lens,
            seq_lens_decoder=prefixes,
            seq_lens_this_time=this_time,
            max_dec_len=max_dec,
            block_size=block_size,
        )

    # A8: All-decode batch (seq_lens_encoder all zero)
    def test_a8_all_decode(self):
        """bs=4, all decode (encoder=0). cache_len should be 0 for all."""
        _run_and_compare(
            seq_lens_encoder=np.array([0, 0, 0, 0], dtype=np.int32),
            seq_lens_decoder=np.array([100, 200, 50, 300], dtype=np.int32),
            seq_lens_this_time=np.array([1, 1, 1, 1], dtype=np.int32),
            max_dec_len=300,
            block_size=64,
        )


class TestPreCacheLenConcatBoundary:
    """Boundary value tests."""

    def test_single_token_prefill(self):
        """bs=1, single token (extend=1, prefix=0)."""
        _run_and_compare(
            seq_lens_encoder=np.array([1], dtype=np.int32),
            seq_lens_decoder=np.array([0], dtype=np.int32),
            seq_lens_this_time=np.array([1], dtype=np.int32),
            max_dec_len=0,
            block_size=64,
        )

    def test_prefix_equals_one_block(self):
        """prefix == block_size exactly (1 tile)."""
        _run_and_compare(
            seq_lens_encoder=np.array([64], dtype=np.int32),
            seq_lens_decoder=np.array([64], dtype=np.int32),
            seq_lens_this_time=np.array([64], dtype=np.int32),
            max_dec_len=64,
            block_size=64,
        )

    def test_prefix_one_token(self):
        """prefix=1 (smallest non-zero cache_len, 1 tile)."""
        _run_and_compare(
            seq_lens_encoder=np.array([32], dtype=np.int32),
            seq_lens_decoder=np.array([1], dtype=np.int32),
            seq_lens_this_time=np.array([32], dtype=np.int32),
            max_dec_len=1,
            block_size=64,
        )

    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_various_block_sizes(self, block_size):
        """Same data with different block sizes."""
        _run_and_compare(
            seq_lens_encoder=np.array([100, 50], dtype=np.int32),
            seq_lens_decoder=np.array([200, 100], dtype=np.int32),
            seq_lens_this_time=np.array([100, 50], dtype=np.int32),
            max_dec_len=200,
            block_size=block_size,
        )

    def test_bs1_decode_only(self):
        """bs=1, decode-only: should produce zero tiles, kv_token_num = this_time."""
        _run_and_compare(
            seq_lens_encoder=np.array([0], dtype=np.int32),
            seq_lens_decoder=np.array([500], dtype=np.int32),
            seq_lens_this_time=np.array([1], dtype=np.int32),
            max_dec_len=500,
            block_size=64,
        )

    def test_interleaved_prefill_decode(self):
        """Prefill and decode interleaved in the batch (not grouped)."""
        _run_and_compare(
            seq_lens_encoder=np.array([128, 0, 64, 0, 256], dtype=np.int32),
            seq_lens_decoder=np.array([64, 300, 128, 100, 0], dtype=np.int32),
            seq_lens_this_time=np.array([128, 1, 64, 1, 256], dtype=np.int32),
            max_dec_len=300,
            block_size=64,
        )


class TestPreCacheLenConcatError:
    """Error path tests — verify op rejects invalid inputs gracefully."""

    def test_wrong_dtype_float(self):
        """Float inputs should raise (kernel expects int32)."""
        enc = paddle.to_tensor([512.0], dtype="float32")
        dec = paddle.to_tensor([0.0], dtype="float32")
        stt = paddle.to_tensor([512.0], dtype="float32")
        with pytest.raises(Exception):
            pre_cache_len_concat(enc, dec, stt, 0, 64)

    def test_empty_batch(self):
        """bs=0 (empty tensors) should not crash."""
        enc = paddle.to_tensor([], dtype="int32")
        dec = paddle.to_tensor([], dtype="int32")
        stt = paddle.to_tensor([], dtype="int32")
        outputs = pre_cache_len_concat(enc, dec, stt, 0, 64)
        cu_seqlens_k = outputs[0].numpy()
        # cu_seqlens_k should be [0] for bs=0
        assert cu_seqlens_k[0] == 0
        assert outputs[3].numpy()[0] == 0  # num_blocks = 0
        assert outputs[4].numpy()[0] == 0  # kv_token_num = 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
