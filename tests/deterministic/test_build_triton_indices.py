"""
Tests for CUDA-Graph-compatible Triton index building kernels.

Compares the new Triton-based implementations (no .item() calls) against
the reference Python for-loop implementations to verify correctness.
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    _scatter_extend_kv_indices_kernel,
    build_kv_indices_from_block_tables,
    build_kv_indices_from_block_tables_ref,
    pre_cache_len_concat_ref,
    pre_cache_len_concat_triton,
)

# ---------------------------------------------------------------------------
# Test: build_kv_indices_from_block_tables (Triton) vs _ref (Python for-loop)
# ---------------------------------------------------------------------------


class TestBuildKvIndicesFromBlockTables:
    """Compare Triton kernel vs reference Python loop for building KV indices."""

    @staticmethod
    def _make_block_tables(bs, max_blocks_per_seq):
        """Create random block tables with unique physical block IDs."""
        # Use unique block IDs (0..bs*max_blocks-1) to avoid collisions
        total = bs * max_blocks_per_seq
        ids = np.random.permutation(total).reshape(bs, max_blocks_per_seq)
        return paddle.to_tensor(ids, dtype="int32")

    @pytest.mark.parametrize("block_size", [16, 64])
    @pytest.mark.parametrize(
        "bs, seq_lens_list",
        [
            (1, [10]),  # single sequence
            (1, [64]),  # exactly one block
            (3, [10, 20, 30]),  # multiple sequences
            (4, [0, 15, 0, 7]),  # sequences with zero length
            (2, [128, 256]),  # multi-block sequences
            (1, [1]),  # single token
            (5, [1, 1, 1, 1, 1]),  # all single-token (decode)
        ],
    )
    def test_matches_ref(self, block_size, bs, seq_lens_list):
        """Triton kernel output must exactly match the reference implementation."""
        max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens_list)
        max_blocks_per_seq = max(max_blocks_per_seq, 1)
        block_tables = self._make_block_tables(bs, max_blocks_per_seq)
        seq_lens = paddle.to_tensor(seq_lens_list, dtype="int32")
        total_kv_len = sum(seq_lens_list)

        # Reference
        indptr_ref, indices_ref = build_kv_indices_from_block_tables_ref(block_tables, seq_lens, block_size, bs)

        # Triton (with pre-computed total_kv_len — the CUDA Graph path)
        indptr_new, indices_new = build_kv_indices_from_block_tables(
            block_tables, seq_lens, block_size, bs, total_kv_len=total_kv_len
        )

        np.testing.assert_array_equal(indptr_new.numpy(), indptr_ref.numpy(), err_msg="kv_indptr mismatch")
        if total_kv_len > 0:
            np.testing.assert_array_equal(
                indices_new[:total_kv_len].numpy(),
                indices_ref[:total_kv_len].numpy(),
                err_msg="kv_indices mismatch",
            )

    @pytest.mark.parametrize("block_size", [16, 64])
    def test_auto_total_kv_len(self, block_size):
        """When total_kv_len is None, the function falls back to .item() (non-graph path)."""
        bs = 3
        seq_lens_list = [10, 20, 30]
        max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens_list)
        block_tables = self._make_block_tables(bs, max_blocks_per_seq)
        seq_lens = paddle.to_tensor(seq_lens_list, dtype="int32")

        indptr_ref, indices_ref = build_kv_indices_from_block_tables_ref(block_tables, seq_lens, block_size, bs)
        indptr_new, indices_new = build_kv_indices_from_block_tables(
            block_tables, seq_lens, block_size, bs, total_kv_len=None
        )

        total = sum(seq_lens_list)
        np.testing.assert_array_equal(indptr_new.numpy(), indptr_ref.numpy())
        np.testing.assert_array_equal(indices_new[:total].numpy(), indices_ref[:total].numpy())


# ---------------------------------------------------------------------------
# Test: _scatter_extend_kv_indices_kernel
# ---------------------------------------------------------------------------


class TestScatterExtendKvIndices:
    """Verify the Triton scatter kernel against a Python reference."""

    @staticmethod
    def _scatter_ref(all_kv_indices, all_kv_indptr, prefix_lens, extend_start_loc, extend_seq_lens, bs):
        """Python reference for the scatter operation."""
        total_extend = int(paddle.sum(extend_seq_lens).item())
        out = paddle.empty([max(total_extend, 1)], dtype="int32")
        for s in range(bs):
            plen = int(prefix_lens[s].item())
            elen = int(extend_seq_lens[s].item())
            if elen == 0:
                continue
            src_start = int(all_kv_indptr[s].item()) + plen
            dst_start = int(extend_start_loc[s].item())
            out[dst_start : dst_start + elen] = all_kv_indices[src_start : src_start + elen]
        return out

    @pytest.mark.parametrize(
        "bs, prefix_list, extend_list",
        [
            (1, [10], [5]),  # single seq
            (3, [10, 20, 30], [5, 3, 8]),  # multi seq
            (4, [0, 15, 0, 7], [3, 2, 0, 1]),  # mixed zero/non-zero
            (2, [100, 200], [1, 1]),  # decode-like (extend=1)
            (5, [0, 0, 0, 0, 0], [10, 20, 30, 40, 50]),  # all prefill, no prefix
        ],
    )
    def test_matches_ref(self, bs, prefix_list, extend_list):
        """Triton scatter kernel output must exactly match Python reference."""
        prefix_lens = paddle.to_tensor(prefix_list, dtype="int32")
        extend_seq_lens = paddle.to_tensor(extend_list, dtype="int32")
        total_seq_lens = prefix_lens + extend_seq_lens

        # Build all_kv_indptr and all_kv_indices (fake monotonic indices)
        all_kv_indptr = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(total_seq_lens).astype("int32"),
            ]
        )
        total_all = int(paddle.sum(total_seq_lens).item())
        all_kv_indices = paddle.arange(total_all, dtype="int32")  # 0, 1, 2, ...

        extend_start_loc = (
            paddle.concat(
                [
                    paddle.zeros([1], dtype="int32"),
                    paddle.cumsum(extend_seq_lens[:-1]).astype("int32"),
                ]
            )
            if bs > 1
            else paddle.zeros([1], dtype="int32")
        )

        total_extend = sum(extend_list)

        # Reference
        ref = self._scatter_ref(all_kv_indices, all_kv_indptr, prefix_lens, extend_start_loc, extend_seq_lens, bs)

        # Triton
        out = paddle.empty([max(total_extend, 1)], dtype="int32")
        if bs > 0 and total_extend > 0:
            _scatter_extend_kv_indices_kernel[(bs,)](
                all_kv_indices,
                all_kv_indptr,
                prefix_lens,
                extend_start_loc,
                extend_seq_lens,
                out,
                BLOCK=128,
            )

        if total_extend > 0:
            np.testing.assert_array_equal(
                out[:total_extend].numpy(),
                ref[:total_extend].numpy(),
                err_msg="scatter extend kv indices mismatch",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])


# ---------------------------------------------------------------------------
# Test: pre_cache_len_concat_triton vs pre_cache_len_concat_ref
# ---------------------------------------------------------------------------


class TestPreCacheLenConcat:
    """Compare Triton GPU-only pre_cache_len_concat vs Python reference."""

    @pytest.mark.parametrize("block_size", [16, 64, 128])
    @pytest.mark.parametrize(
        "bsz, enc_list, dec_list, qlen_list",
        [
            # Pure decode: all enc=0, so cache_len=0 for all
            (3, [0, 0, 0], [50, 100, 200], [1, 1, 1]),
            # Pure prefill: enc>0, cache_len = dec (chunked prefill)
            (2, [10, 20], [0, 0], [10, 20]),
            # Mixed: some prefill, some decode
            (4, [10, 0, 5, 0], [32, 100, 64, 200], [10, 1, 5, 1]),
            # Single batch
            (1, [1], [128], [1]),
            # All zero enc/dec (edge case)
            (3, [0, 0, 0], [0, 0, 0], [5, 3, 7]),
            # Large cache_len spanning many blocks
            (2, [1, 1], [512, 1024], [32, 64]),
            # Single token decode
            (5, [0, 0, 0, 0, 0], [10, 20, 30, 40, 50], [1, 1, 1, 1, 1]),
            # Mixed zero and non-zero enc
            (4, [5, 0, 0, 10], [100, 0, 50, 200], [5, 1, 1, 10]),
        ],
    )
    def test_matches_ref(self, block_size, bsz, enc_list, dec_list, qlen_list):
        """Triton pre_cache_len_concat must exactly match the reference."""
        seq_lens_encoder = paddle.to_tensor(enc_list, dtype="int32")
        seq_lens_decoder = paddle.to_tensor(dec_list, dtype="int32")
        seq_lens_this_time = paddle.to_tensor(qlen_list, dtype="int32")

        max_dec = max(dec_list) if dec_list else 0
        max_tile_per_bs = max((max_dec + block_size - 1) // block_size, 1)

        # Reference
        cu_ref, batch_ids_ref, tile_ids_ref = pre_cache_len_concat_ref(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            bsz,
            block_size,
            max_tile_per_bs,
        )

        # Triton
        cu_tri, batch_ids_tri, tile_ids_tri = pre_cache_len_concat_triton(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            bsz,
            block_size,
            max_tile_per_bs,
        )

        # cu_seqlens_k must match exactly
        np.testing.assert_array_equal(cu_tri.numpy(), cu_ref.numpy(), err_msg="cu_seqlens_k mismatch")

        # Compute total gridx from reference to know valid range of batch_ids/tile_ids
        gridx = 0
        for bid in range(bsz):
            enc = enc_list[bid]
            dec = dec_list[bid]
            cache_len = dec if enc > 0 else 0
            gridx += (cache_len + block_size - 1) // block_size

        if gridx > 0:
            np.testing.assert_array_equal(
                batch_ids_tri[:gridx].numpy(),
                batch_ids_ref[:gridx].numpy(),
                err_msg="batch_ids mismatch",
            )
            np.testing.assert_array_equal(
                tile_ids_tri[:gridx].numpy(),
                tile_ids_ref[:gridx].numpy(),
                err_msg="tile_ids mismatch",
            )

    def test_empty_batch(self):
        """Edge case: bsz=0 should produce cu_seqlens_k=[0]."""
        cu, _, _ = pre_cache_len_concat_triton(
            paddle.to_tensor([], dtype="int32"),
            paddle.to_tensor([], dtype="int32"),
            paddle.to_tensor([], dtype="int32"),
            0,
            64,
            1,
        )
        assert cu.shape == [1]
        assert int(cu[0].item()) == 0
