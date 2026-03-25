"""Unit tests for apply_top_k_top_p_triton."""

import os
import sys

import numpy as np
import paddle
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from fastdeploy.model_executor.layers.sample.ops.top_k_top_p_triton import (
    apply_top_k_top_p_triton,
)


@pytest.fixture(autouse=True)
def _use_gpu():
    paddle.set_device("gpu:0")


def _make_logits(batch_size: int, vocab_size: int, seed: int = 42) -> paddle.Tensor:
    """Create deterministic random float32 logits on GPU."""
    np.random.seed(seed)
    return paddle.to_tensor(np.random.randn(batch_size, vocab_size).astype("float32"))


# ---------------------------------------------------------------------------
# Reference implementation (CPU / NumPy) for correctness comparison
# ---------------------------------------------------------------------------


def _ref_top_k_top_p(
    logits_np: np.ndarray,
    k: np.ndarray | None,
    p: np.ndarray | None,
) -> np.ndarray:
    """
    Pure-NumPy reference: top-k first, then top-p on remaining tokens.
    Returns masked logits (masked positions set to -inf).
    """
    B, V = logits_np.shape
    out = logits_np.copy()

    for i in range(B):
        row = out[i]
        # --- top-k ---
        if k is not None:
            ki = int(k[i])
            if ki < V:
                threshold = np.partition(row, -ki)[-ki]
                row[row < threshold] = -np.inf
                # Handle duplicates at threshold: keep exactly ki
                kept = np.sum(row > -np.inf)
                if kept > ki:
                    at_thresh = np.where(row == threshold)[0]
                    excess = kept - ki
                    row[at_thresh[:excess]] = -np.inf

        # --- top-p on surviving tokens ---
        if p is not None:
            pi = float(p[i])
            alive = row > -np.inf
            if alive.sum() > 0 and pi < 1.0:
                alive_logits = row[alive]
                probs = np.exp(alive_logits - alive_logits.max())
                probs /= probs.sum()
                sorted_idx = np.argsort(-probs)
                cum = np.cumsum(probs[sorted_idx])
                # Keep tokens until cumulative prob >= p, always keep at least 1
                cutoff = np.searchsorted(cum, pi, side="left") + 1
                if cutoff < len(sorted_idx):
                    remove_local = sorted_idx[cutoff:]
                    alive_positions = np.where(alive)[0]
                    row[alive_positions[remove_local]] = -np.inf

        out[i] = row
    return out


# ---------------------------------------------------------------------------
# Top-K precision tests
# ---------------------------------------------------------------------------


class TestTopKPrecision:
    def test_exact_top_k_values(self):
        """Kept values must exactly equal the original top-k values (bitwise)."""
        B, V, K = 4, 1024, 10
        logits = _make_logits(B, V, seed=0)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        for i in range(B):
            orig_topk = paddle.topk(original[i], K)
            kept_vals = logits[i][logits[i] > float("-inf")]
            kept_sorted = paddle.sort(kept_vals, descending=True)
            orig_sorted = paddle.sort(orig_topk[0], descending=True)
            np.testing.assert_array_equal(
                kept_sorted.numpy()[:K],
                orig_sorted.numpy(),
                err_msg=f"row {i}: kept values differ from original top-k",
            )

    def test_exact_top_k_indices(self):
        """Kept positions must correspond to the original top-k indices."""
        B, V, K = 4, 512, 8
        logits = _make_logits(B, V, seed=1)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        for i in range(B):
            orig_topk_idx = set(paddle.topk(original[i], K)[1].numpy().tolist())
            kept_idx = set(np.where(logits[i].numpy() > -np.inf)[0].tolist())
            assert kept_idx.issubset(
                orig_topk_idx
            ), f"row {i}: kept indices {kept_idx - orig_topk_idx} not in original top-k"

    def test_masked_positions_are_neg_inf(self):
        """Masked positions must be exactly -inf, not just a large negative."""
        B, V, K = 2, 256, 5
        logits = _make_logits(B, V, seed=2)
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        for i in range(B):
            row = logits[i].numpy()
            non_kept = row[np.isneginf(row)]
            assert len(non_kept) >= V - K, f"row {i}: some masked values are not -inf"

    def test_per_row_different_k(self):
        """Each row can have a different k value."""
        B, V = 4, 256
        ks = [1, 5, 50, 256]
        logits = _make_logits(B, V, seed=3)
        k = paddle.to_tensor(ks, dtype="int32")
        apply_top_k_top_p_triton(logits, k=k, p=None)
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert non_masked <= ks[i], f"row {i}: expected <= {ks[i]}, got {non_masked}"
            assert non_masked > 0

    def test_k_equals_1(self):
        """k=1 should keep only the argmax."""
        B, V = 4, 512
        logits = _make_logits(B, V, seed=4)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=paddle.full([B], 1, dtype="int32"), p=None)
        for i in range(B):
            kept = logits[i][logits[i] > float("-inf")]
            assert kept.shape[0] == 1, f"row {i}: expected 1 kept, got {kept.shape[0]}"
            assert kept[0].item() == original[i].max().item()

    def test_k_equals_vocab(self):
        """k=vocab_size should be a no-op (all tokens kept)."""
        B, V = 2, 128
        logits = _make_logits(B, V, seed=5)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=paddle.full([B], V, dtype="int32"), p=None)
        np.testing.assert_array_equal(logits.numpy(), original.numpy())

    def test_duplicate_logit_values(self):
        """When multiple tokens share the same logit at the k-boundary, count is still <= k."""
        B, V, K = 2, 64, 5
        logits = paddle.zeros([B, V], dtype="float32")
        # Set top-5 to distinct values, rest share the 5th value
        for i in range(B):
            logits[i, :K] = paddle.to_tensor([10.0, 9.0, 8.0, 7.0, 6.0], dtype="float32")
            logits[i, K:] = 6.0  # duplicates of the boundary value
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert non_masked <= K, f"row {i}: expected <= {K}, got {non_masked}"


# ---------------------------------------------------------------------------
# Top-P precision tests
# ---------------------------------------------------------------------------


class TestTopPPrecision:
    def test_top_p_cumulative_probability(self):
        """Kept tokens' probabilities should sum to >= p."""
        B, V = 4, 256
        p_val = 0.9
        logits = _make_logits(B, V, seed=10)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=None, p=paddle.full([B], p_val, dtype="float32"))
        for i in range(B):
            # Compute original probs via softmax
            orig_probs = paddle.nn.functional.softmax(original[i], axis=-1).numpy()
            kept_mask = logits[i].numpy() > -np.inf
            kept_prob = orig_probs[kept_mask].sum()
            assert kept_prob >= p_val - 0.05, f"row {i}: kept prob {kept_prob:.4f} < p={p_val}"

    def test_top_p_minimality(self):
        """Removing any one kept token (except the largest) should drop cumsum below p."""
        B, V = 4, 512
        p_val = 0.8
        logits = _make_logits(B, V, seed=11)
        original = logits.clone()
        apply_top_k_top_p_triton(logits, k=None, p=paddle.full([B], p_val, dtype="float32"))
        for i in range(B):
            orig_probs = paddle.nn.functional.softmax(original[i], axis=-1).numpy()
            kept_mask = logits[i].numpy() > -np.inf
            kept_probs = orig_probs[kept_mask]
            total = kept_probs.sum()
            if len(kept_probs) <= 1:
                continue
            # The smallest kept token should be necessary (or near-necessary)
            smallest_kept = kept_probs.min()
            # Removing the smallest should bring total close to or below p
            assert total - smallest_kept <= p_val + 0.05, (
                f"row {i}: removing smallest kept still leaves " f"{total - smallest_kept:.4f} > p={p_val}+tolerance"
            )

    def test_top_p_various_values(self):
        """Test multiple p values: kept count should increase with p."""
        B, V = 1, 512
        logits_base = _make_logits(B, V, seed=12)
        counts = []
        for p_val in [0.1, 0.5, 0.9, 1.0]:
            logits = logits_base.clone()
            apply_top_k_top_p_triton(logits, k=None, p=paddle.full([B], p_val, dtype="float32"))
            cnt = (logits[0] > float("-inf")).sum().item()
            counts.append(cnt)
        # Monotonically non-decreasing
        for j in range(len(counts) - 1):
            assert counts[j] <= counts[j + 1], f"kept count not monotonic: p values -> counts {counts}"
        # p=1.0 should keep all
        assert counts[-1] == V

    def test_top_p_very_small(self):
        """Very small p should keep very few tokens (often just 1)."""
        B, V = 4, 1024
        logits = _make_logits(B, V, seed=13)
        apply_top_k_top_p_triton(logits, k=None, p=paddle.full([B], 0.01, dtype="float32"))
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert non_masked >= 1
            assert non_masked <= 20, f"row {i}: p=0.01 kept {non_masked} tokens"


# ---------------------------------------------------------------------------
# Combined top-k + top-p precision tests
# ---------------------------------------------------------------------------


class TestCombinedPrecision:
    def test_combined_vs_sequential(self):
        """Triton combined result should match top-k-then-top-p applied independently."""
        B, V, K = 4, 512, 50
        p_val = 0.9
        logits = _make_logits(B, V, seed=20)
        original_np = logits.numpy().copy()

        # --- Reference: top-k first, then top-p ---
        ref = _ref_top_k_top_p(
            original_np,
            k=np.full(B, K, dtype=np.int32),
            p=np.full(B, p_val, dtype=np.float32),
        )

        # --- Triton ---
        apply_top_k_top_p_triton(
            logits,
            k=paddle.full([B], K, dtype="int32"),
            p=paddle.full([B], p_val, dtype="float32"),
        )
        triton_np = logits.numpy()

        for i in range(B):
            ref_kept = set(np.where(ref[i] > -np.inf)[0])
            tri_kept = set(np.where(triton_np[i] > -np.inf)[0])
            # Allow small difference due to softmax precision
            sym_diff = ref_kept.symmetric_difference(tri_kept)
            assert len(sym_diff) <= 3, f"row {i}: ref vs triton kept sets differ by {len(sym_diff)} tokens: {sym_diff}"

    def test_combined_top_p_further_reduces_top_k(self):
        """With small p, combined should keep fewer tokens than top-k alone."""
        B, V, K = 4, 256, 50
        logits1 = _make_logits(B, V, seed=21)
        logits2 = logits1.clone()

        # top-k only
        apply_top_k_top_p_triton(logits1, k=paddle.full([B], K, dtype="int32"), p=None)
        # top-k + top-p
        apply_top_k_top_p_triton(
            logits2,
            k=paddle.full([B], K, dtype="int32"),
            p=paddle.full([B], 0.5, dtype="float32"),
        )
        for i in range(B):
            cnt_k = (logits1[i] > float("-inf")).sum().item()
            cnt_kp = (logits2[i] > float("-inf")).sum().item()
            assert cnt_kp <= cnt_k, f"row {i}: combined ({cnt_kp}) should be <= top-k only ({cnt_k})"

    def test_combined_per_row_mixed_params(self):
        """Different k and p per row."""
        B, V = 4, 512
        ks = [5, 20, 100, 512]
        ps = [0.3, 0.5, 0.9, 1.0]
        logits = _make_logits(B, V, seed=22)
        k = paddle.to_tensor(ks, dtype="int32")
        p = paddle.to_tensor(ps, dtype="float32")
        apply_top_k_top_p_triton(logits, k=k, p=p)
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert non_masked <= ks[i], f"row {i}: expected <= {ks[i]}, got {non_masked}"
            assert non_masked >= 1

    def test_kept_values_unchanged(self):
        """Kept (non-masked) logit values must be bitwise identical to original."""
        B, V, K = 4, 256, 20
        logits = _make_logits(B, V, seed=23)
        original = logits.clone()
        apply_top_k_top_p_triton(
            logits,
            k=paddle.full([B], K, dtype="int32"),
            p=paddle.full([B], 0.8, dtype="float32"),
        )
        for i in range(B):
            kept_mask = logits[i].numpy() > -np.inf
            np.testing.assert_array_equal(
                logits[i].numpy()[kept_mask],
                original[i].numpy()[kept_mask],
                err_msg=f"row {i}: kept logit values were modified",
            )


# ---------------------------------------------------------------------------
# Large vocab / batch stress tests
# ---------------------------------------------------------------------------


class TestLargeScale:
    def test_large_vocab(self):
        """Test with a realistic vocab size (32000)."""
        B, V, K = 2, 32000, 50
        logits = _make_logits(B, V, seed=30)
        original = logits.clone()
        apply_top_k_top_p_triton(
            logits,
            k=paddle.full([B], K, dtype="int32"),
            p=paddle.full([B], 0.9, dtype="float32"),
        )
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert 1 <= non_masked <= K
            # Verify kept values are unchanged
            kept_mask = logits[i].numpy() > -np.inf
            np.testing.assert_array_equal(
                logits[i].numpy()[kept_mask],
                original[i].numpy()[kept_mask],
            )

    def test_large_batch(self):
        """Test with a large batch (128 rows)."""
        B, V, K = 128, 1024, 10
        logits = _make_logits(B, V, seed=31)
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        for i in range(B):
            non_masked = (logits[i] > float("-inf")).sum().item()
            assert non_masked <= K
            assert non_masked >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_batch(self):
        """Empty batch should return immediately."""
        logits = paddle.empty([0, 128], dtype="float32")
        out = apply_top_k_top_p_triton(logits, k=None, p=None)
        assert out.shape == [0, 128]

    def test_no_filtering(self):
        """Both k and p as None should be a no-op."""
        B, V = 2, 64
        logits = _make_logits(B, V, seed=40)
        original = logits.clone()
        out = apply_top_k_top_p_triton(logits, k=None, p=None)
        np.testing.assert_array_equal(out.numpy(), original.numpy())

    def test_single_row(self):
        """Batch size 1 should work correctly."""
        B, V, K = 1, 256, 3
        logits = _make_logits(B, V, seed=41)
        apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None)
        assert (logits[0] > float("-inf")).sum().item() <= K

    def test_inplace_returns_same_tensor(self):
        """Return value should be the same tensor object (in-place)."""
        B, V = 2, 64
        logits = _make_logits(B, V, seed=42)
        out = apply_top_k_top_p_triton(logits, k=paddle.full([B], 5, dtype="int32"), p=None)
        assert out.data_ptr() == logits.data_ptr()


class TestReturnMask:
    """Tests for return_mask=True — mask output from the Triton kernel."""

    def test_mask_shape_and_dtype(self):
        """Mask should be [B, V] bool."""
        B, V, K = 4, 256, 10
        logits = _make_logits(B, V, seed=50)
        _, mask = apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None, return_mask=True)
        assert mask.shape == [B, V]
        assert mask.dtype == paddle.bool

    def test_mask_matches_logits(self):
        """Mask True positions should match logits != -inf exactly."""
        B, V, K = 8, 512, 5
        logits = _make_logits(B, V, seed=51)
        out, mask = apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None, return_mask=True)
        expected = out != float("-inf")
        np.testing.assert_array_equal(mask.numpy(), expected.numpy())

    def test_mask_top_p(self):
        """Mask with top-p should match logits != -inf."""
        B, V = 4, 256
        logits = _make_logits(B, V, seed=52)
        p = paddle.full([B], 0.9, dtype="float32")
        out, mask = apply_top_k_top_p_triton(logits, k=None, p=p, return_mask=True)
        expected = out != float("-inf")
        np.testing.assert_array_equal(mask.numpy(), expected.numpy())

    def test_mask_combined(self):
        """Mask with combined top-k + top-p should match logits != -inf."""
        B, V = 4, 256
        logits = _make_logits(B, V, seed=53)
        k = paddle.to_tensor([5, 10, 20, 50], dtype="int32")
        p = paddle.to_tensor([0.8, 0.9, 0.95, 1.0], dtype="float32")
        out, mask = apply_top_k_top_p_triton(logits, k=k, p=p, return_mask=True)
        expected = out != float("-inf")
        np.testing.assert_array_equal(mask.numpy(), expected.numpy())

    def test_mask_no_filtering(self):
        """When no filtering is needed, mask should be all True."""
        B, V = 2, 64
        logits = _make_logits(B, V, seed=54)
        out, mask = apply_top_k_top_p_triton(logits, k=None, p=None, return_mask=True)
        assert mask.all().item()

    def test_mask_count_matches_top_k(self):
        """Number of True values per row should be <= k."""
        B, V, K = 4, 512, 8
        logits = _make_logits(B, V, seed=55)
        _, mask = apply_top_k_top_p_triton(logits, k=paddle.full([B], K, dtype="int32"), p=None, return_mask=True)
        for i in range(B):
            assert mask[i].astype("int32").sum().item() <= K
