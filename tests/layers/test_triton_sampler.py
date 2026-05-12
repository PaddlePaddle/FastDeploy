"""
Unit tests for the triton sampling path introduced in commit 16e692f.

Covers:
  - _apply_triton_top_k_top_p: top-k/top-p masking on logits before softmax
  - _random_sample: Gumbel-max stochastic sampling from probabilities
"""

import sys
import types

import paddle
import pytest

import fastdeploy  # noqa: F401

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda *args, **kwargs: None

# Stub triton for unit isolation (same pattern as test_sampler.py).
if "triton" not in sys.modules:
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_lang_stub = types.ModuleType("triton.language")
    triton_lang_stub.constexpr = int
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub

# Must import after stubs are in place.
from fastdeploy.model_executor.layers.sample.sampler import (
    _apply_triton_top_k_top_p,
    _random_sample,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_triton_kernel(monkeypatch):
    """Patch apply_top_k_top_p_triton to avoid real GPU/Triton execution."""
    import fastdeploy.model_executor.layers.sample.sampler as sampler_mod

    def _fake_apply_top_k_top_p_triton(logits, k=None, p=None, return_mask=False):
        """CPU reference: mask logits to -inf for tokens outside top-k and top-p."""
        result = logits.clone()
        batch_size = result.shape[0]
        mask_out = paddle.ones(result.shape, dtype=paddle.bool)

        for i in range(batch_size):
            row = result[i]
            # Top-k filtering
            if k is not None:
                ki = int(k[i].item()) if k.ndim > 0 else int(k.item())
                if ki > 0 and ki < row.shape[0]:
                    topk_vals = paddle.topk(row, ki)
                    threshold = topk_vals.values[-1]
                    keep = row >= threshold
                    result[i] = paddle.where(keep, row, paddle.full_like(row, float("-inf")))
                    mask_out[i] = keep
            # Top-p filtering
            if p is not None:
                pi = float(p[i].item()) if p.ndim > 0 else float(p.item())
                if pi < 1.0:
                    sorted_idx = paddle.argsort(result[i], descending=True)
                    sorted_vals = result[i][sorted_idx]
                    probs = paddle.nn.functional.softmax(sorted_vals, axis=-1)
                    cumsum = paddle.cumsum(probs, axis=-1)
                    keep_sorted = (cumsum - probs) <= pi
                    # Also keep tokens that tie with the boundary
                    k_count = keep_sorted.sum().item()
                    if k_count > 0:
                        boundary = probs[int(k_count) - 1].item()
                        keep_sorted = keep_sorted | (probs >= boundary - 1e-9)
                    # Map back to original order
                    unsorted_keep = paddle.zeros_like(keep_sorted)
                    unsorted_keep[sorted_idx] = keep_sorted
                    result[i] = paddle.where(unsorted_keep, result[i], paddle.full_like(result[i], float("-inf")))
                    mask_out[i] = mask_out[i] & unsorted_keep

        if return_mask:
            return result, mask_out
        return result

    monkeypatch.setattr(
        sampler_mod,
        "apply_top_k_top_p_triton",
        _fake_apply_top_k_top_p_triton,
    )


@pytest.fixture(autouse=True)
def _patch_seeded_gumbel(monkeypatch):
    """Patch seeded_gumbel_noise to use paddle.uniform on CPU."""
    import fastdeploy.model_executor.layers.sample.sampler as sampler_mod

    def _fake_seeded_gumbel_noise(probs, seeds):
        u = paddle.uniform(probs.shape, dtype=probs.dtype, min=0.0, max=1.0)
        return -paddle.log(u.clip(min=1e-10))

    monkeypatch.setattr(
        sampler_mod,
        "seeded_gumbel_noise",
        _fake_seeded_gumbel_noise,
    )


# ---------------------------------------------------------------------------
# Tests for _apply_triton_top_k_top_p
# ---------------------------------------------------------------------------


class TestApplyTritonTopKTopP:
    """Tests for _apply_triton_top_k_top_p."""

    def test_returns_logits_unchanged_when_both_none(self):
        """When top_p and top_k are both None, return logits unchanged."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=None, top_k=None)
        assert paddle.equal_all(result, logits)

    def test_top_p_only_masks_low_prob_tokens(self):
        """top_p < 1.0 should mask tokens outside the nucleus."""
        logits = paddle.to_tensor([[1.0, 2.0, 5.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.7]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p)
        # The lowest logit (1.0) should be masked to -inf
        assert result[0, 0].item() == float("-inf")
        # Highest logit should be retained
        assert result[0, 2].item() != float("-inf")

    def test_top_p_1_keeps_all(self):
        """top_p=1.0 should keep all tokens."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p)
        # No tokens should be masked
        assert (result > float("-inf")).all().item()

    def test_top_k_only_masks_beyond_k(self):
        """top_k should keep only the k highest logits."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0]], dtype="float32")
        top_k = paddle.to_tensor([[2]], dtype="int64")
        top_k_list = [2]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        # Top 2 logits are 5.0 (idx 1) and 4.0 (idx 4)
        assert result[0, 1].item() != float("-inf")
        assert result[0, 4].item() != float("-inf")
        # Lower ones should be masked
        assert result[0, 0].item() == float("-inf")
        assert result[0, 2].item() == float("-inf")
        assert result[0, 3].item() == float("-inf")

    def test_top_k_disabled_when_all_zero(self):
        """top_k_list with all zeros should disable top-k filtering."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0]], dtype="float32")
        top_k = paddle.to_tensor([[0]], dtype="int64")
        top_k_list = [0]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        # No masking should occur (top_p=1.0, top_k disabled)
        assert paddle.equal_all(result, logits)

    def test_top_k_disabled_when_list_none(self):
        """top_k_list=None should disable top-k filtering."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=None, top_k_list=None)
        assert paddle.equal_all(result, logits)

    def test_combined_top_k_top_p(self):
        """Combined top-k + top-p should apply both filters."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.5]], dtype="float32")
        top_k = paddle.to_tensor([[3]], dtype="int64")
        top_k_list = [3]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        # After top_k=3: keep indices 1(5.0), 4(4.0), 2(3.0)
        # After top_p=0.5 on those 3: likely keep only the highest (5.0)
        assert result[0, 1].item() != float("-inf")
        # Some tokens must be masked
        assert (result == float("-inf")).any().item()

    def test_batch_slicing(self):
        """top_p and top_k should be sliced to batch_size."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        # Extra entries beyond batch size
        top_p = paddle.to_tensor([[0.5], [0.9], [0.1]], dtype="float32")
        top_k = paddle.to_tensor([[2], [1], [3]], dtype="int64")
        top_k_list = [2, 1, 3]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        assert result.shape == [2, 3]

    def test_return_mask_false(self):
        """return_mask=False should return only logits."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.9]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, return_mask=False)
        assert isinstance(result, paddle.Tensor)

    def test_return_mask_true(self):
        """return_mask=True should return (logits, mask) tuple."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.5]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, return_mask=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        logits_out, mask = result
        assert logits_out.shape == [1, 3]
        assert mask.shape == [1, 3]
        assert mask.dtype == paddle.bool

    def test_output_dtype_is_float32(self):
        """Output logits should always be float32."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float16")
        top_p = paddle.to_tensor([[0.9]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p)
        assert result.dtype == paddle.float32

    def test_mixed_top_k_in_batch(self):
        """Batch where some rows have top_k>0 and others top_k=0."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0], [0.9]], dtype="float32")
        top_k = paddle.to_tensor([[2], [0]], dtype="int64")
        top_k_list = [2, 0]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        # Row 0: top_k=2 active, should mask lowest (1.0)
        assert result[0, 0].item() == float("-inf")
        # Row 1: top_k=0 disabled, top_p=0.9 keeps most
        assert result[1, 2].item() != float("-inf")  # highest logit


# ---------------------------------------------------------------------------
# Tests for _random_sample
# ---------------------------------------------------------------------------


class TestRandomSample:
    """Tests for _random_sample."""

    def test_output_shape(self):
        """Output should have shape [batch_size, 1]."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], dtype="float32")
        result = _random_sample(probs)
        assert result.shape == [2, 1]

    def test_output_dtype(self):
        """Output token ids should be int64."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        result = _random_sample(probs)
        assert result.dtype == paddle.int64

    def test_tokens_within_vocab_range(self):
        """Sampled token ids must be valid indices into the vocab."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        result = _random_sample(probs)
        assert 0 <= result[0, 0].item() < 3

    def test_without_seed(self):
        """Without seed, uses paddle.uniform path and still produces valid samples."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], dtype="float32")
        result = _random_sample(probs, topp_seed=None)
        assert result.shape == [2, 1]
        for i in range(2):
            assert 0 <= result[i, 0].item() < 3

    def test_with_seed(self):
        """With seed, uses seeded_gumbel_noise (patched) path."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        seed = paddle.to_tensor([[42]], dtype="int64")
        result = _random_sample(probs, topp_seed=seed)
        assert result.shape == [1, 1]
        assert 0 <= result[0, 0].item() < 3

    def test_with_seed_sliced_to_batch_size(self):
        """Seed tensor is sliced to probs.shape[0] before use."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        # Seed tensor bigger than batch — should slice correctly
        seed = paddle.to_tensor([[10], [20], [30]], dtype="int64")
        result = _random_sample(probs, topp_seed=seed)
        assert result.shape == [1, 1]

    def test_greedy_with_peak_distribution(self):
        """Deterministic distribution should always sample the argmax."""
        probs = paddle.zeros([1, 10], dtype="float32")
        probs[0, 5] = 1.0
        result = _random_sample(probs)
        assert result[0, 0].item() == 5

    def test_batch_multiple_requests(self):
        """Multiple requests in a batch should each get a valid sample."""
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 1.0]], dtype="float32")
        result = _random_sample(probs)
        assert result.shape == [2, 1]
        assert 0 <= result[0, 0].item() < 3
        assert result[1, 0].item() == 2


if __name__ == "__main__":
    pytest.main([__file__])
