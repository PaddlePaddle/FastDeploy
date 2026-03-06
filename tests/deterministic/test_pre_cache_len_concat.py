"""
Tests for pre_cache_len_concat CUDA custom op.

Covers ACTION-4 scenarios (A1-A8):
  A1: Pure prefill, no prefix (prefix=0, extend=512)
  A2: Prefix hit (prefix=384, extend=128)
  A3: Multi-sequence unequal prefix (bs=3)
  A4: Mixed batch (prefill + decode)
  A5: Prefix exactly divisible by block_size
  A6: Prefix not divisible by block_size
  A7: Large batch (bs=32, random prefix/extend)
  A8: All-decode batch

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_pre_cache_len_concat.py -v -s
"""

import numpy as np
import paddle
import pytest

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
    """Run CUDA op and compare against reference."""
    from fastdeploy.model_executor.layers.attention.ops.pre_cache_len_concat import (
        pre_cache_len_concat,
    )

    enc_t = paddle.to_tensor(seq_lens_encoder, dtype="int32")
    dec_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
    stt_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

    outputs = pre_cache_len_concat(enc_t, dec_t, stt_t, max_dec_len, block_size)
    cu_seqlens_k, batch_ids, tile_ids, num_blocks_cpu, kv_token_num_cpu = [o.numpy() for o in outputs]

    ref = ref_pre_cache_len_concat(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_size)
    ref_cu, ref_batch_ids, ref_tile_ids, ref_num_blocks, ref_kv_token_num = ref

    # cu_seqlens_k
    np.testing.assert_array_equal(cu_seqlens_k, ref_cu, err_msg="cu_seqlens_k mismatch")
    # batch_ids (only meaningful entries)
    n = len(ref_batch_ids)
    np.testing.assert_array_equal(batch_ids[:n], ref_batch_ids, err_msg="batch_ids mismatch")
    # tile_ids
    np.testing.assert_array_equal(tile_ids[:n], ref_tile_ids, err_msg="tile_ids_per_batch mismatch")
    # num_blocks
    assert num_blocks_cpu[0] == ref_num_blocks[0], f"num_blocks mismatch: {num_blocks_cpu[0]} vs {ref_num_blocks[0]}"
    # kv_token_num
    assert (
        kv_token_num_cpu[0] == ref_kv_token_num[0]
    ), f"kv_token_num mismatch: {kv_token_num_cpu[0]} vs {ref_kv_token_num[0]}"


# ---------------------------------------------------------------------------
# Test class
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
        """prefix=256, block_size=64 → 256/64 = 4 blocks exactly."""
        _run_and_compare(
            seq_lens_encoder=np.array([128], dtype=np.int32),
            seq_lens_decoder=np.array([256], dtype=np.int32),
            seq_lens_this_time=np.array([128], dtype=np.int32),
            max_dec_len=256,
            block_size=64,
        )

    # A6: Prefix not divisible by block_size
    def test_a6_prefix_not_aligned(self):
        """prefix=300, block_size=64 → ceil(300/64) = 5 blocks."""
        _run_and_compare(
            seq_lens_encoder=np.array([100], dtype=np.int32),
            seq_lens_decoder=np.array([300], dtype=np.int32),
            seq_lens_this_time=np.array([100], dtype=np.int32),
            max_dec_len=300,
            block_size=64,
        )

    # A7: Large batch (bs=32, random prefix/extend)
    def test_a7_large_batch(self):
        """bs=32, random prefix and extend lengths."""
        rng = np.random.RandomState(42)
        bs = 32
        block_size = 64
        prefixes = rng.randint(0, 512, size=bs).astype(np.int32)
        extends = rng.randint(1, 256, size=bs).astype(np.int32)
        max_dec = int(np.max(prefixes))

        _run_and_compare(
            seq_lens_encoder=extends,
            seq_lens_decoder=prefixes,
            seq_lens_this_time=extends,
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

    # Additional edge case: single token prefill
    def test_single_token(self):
        """bs=1, single token (extend=1, prefix=0)."""
        _run_and_compare(
            seq_lens_encoder=np.array([1], dtype=np.int32),
            seq_lens_decoder=np.array([0], dtype=np.int32),
            seq_lens_this_time=np.array([1], dtype=np.int32),
            max_dec_len=0,
            block_size=64,
        )

    # Additional: different block sizes
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_various_block_sizes(self, block_size):
        """Test with various block sizes."""
        _run_and_compare(
            seq_lens_encoder=np.array([100, 50], dtype=np.int32),
            seq_lens_decoder=np.array([200, 100], dtype=np.int32),
            seq_lens_this_time=np.array([100, 50], dtype=np.int32),
            max_dec_len=200,
            block_size=block_size,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
