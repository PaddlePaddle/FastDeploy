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
Correctness tests for tritonmoe_preprocess
==========================================

Tests the fastdeploy wrapper:
    tritonmoe_preprocess(topk_ids, num_experts, block_size)
      -> (sorted_token_ids, expert_ids, num_tokens_post_padded)

The verification approach mirrors FlagTree/python/tutorials/tle/02-moe_align_block_size.py:
  - Use paddle.bincount as an independent reference (no second kernel to cross-compare).
  - Validate three dimensions:
      1. num_tokens_post_padded  – total token count after per-expert block alignment
      2. expert_ids              – each block is mapped to the correct expert
      3. sorted_token_ids        – every token is routed to the right expert's slot,
                                   and padding slots carry sentinel values >= num_tokens
"""

import unittest

import numpy as np
import paddle

# ---------------------------------------------------------------------------
# Import guard – skip entire module when CUDA is unavailable or
# fastdeploy is not installed (e.g. CPU-only CI environments).
# ---------------------------------------------------------------------------
try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

    _AVAILABLE = paddle.device.is_compiled_with_cuda()
except Exception:
    _AVAILABLE = False

DEVICE = "gpu"

# 仅对小规模 case 打印详细 tensor，超过此阈值只打印统计摘要
_PRINT_TENSOR_NUMEL_LIMIT = 64


def _fmt_tensor(t: paddle.Tensor, name: str) -> str:
    t_cpu = t.cpu()
    if t_cpu.numel() <= _PRINT_TENSOR_NUMEL_LIMIT:
        return f"{name}{list(t_cpu.shape)} = {t_cpu.tolist()}"
    return (
        f"{name}{list(t_cpu.shape)} | "
        f"min={int(t_cpu.min())} max={int(t_cpu.max())} "
        f"mean={float(t_cpu.cast('float32').mean()):.2f} numel={t_cpu.numel()}"
    )


# ---------------------------------------------------------------------------
# Reference helpers (CPU, independent of the kernel under test)
# ---------------------------------------------------------------------------


def _ref_counts_and_cumsum(topk_ids_flat: paddle.Tensor, num_experts: int, block_size: int):
    """
    Compute per-expert token counts and the cumulative sum of block-aligned counts.

    Returns:
        counts  : int32 tensor of shape (num_experts,)
        cumsum  : int32 tensor of shape (num_experts,)  – cumulative aligned counts
    """
    # Only consider valid expert ids [0, num_experts); ignore -1 (EP filtered)
    valid_mask = (topk_ids_flat >= 0) & (topk_ids_flat < num_experts)
    valid_ids = topk_ids_flat[valid_mask]
    counts = paddle.bincount(valid_ids.cast("int64"), minlength=num_experts).cast("int32")
    aligned = ((counts + block_size - 1) // block_size) * block_size
    cumsum = paddle.cumsum(aligned, axis=0).cast("int32")
    return counts, cumsum


# ---------------------------------------------------------------------------
# Core verification logic (shared across all test cases)
# ---------------------------------------------------------------------------


def _verify(topk_ids: paddle.Tensor, block_size: int, num_experts: int, label: str = ""):
    """
    Run tritonmoe_preprocess and verify all three output tensors.
    topk_ids may be 1-D or 2-D; dtype int32 or int64.
    Prints inputs, golden references, kernel outputs, and per-check comparison.
    """
    tag = f"[{label}] " if label else ""
    sep = "=" * 70

    sorted_token_ids, expert_ids, num_tokens_post_pad = tritonmoe_preprocess(topk_ids, num_experts, block_size)

    topk_ids_flat = topk_ids.flatten().cast("int64").cpu()
    num_tokens = topk_ids_flat.numel()

    counts, cumsum = _ref_counts_and_cumsum(topk_ids_flat, num_experts, block_size)
    aligned = ((counts + block_size - 1) // block_size) * block_size
    valid_length = int(cumsum[-1].item())
    num_blocks = valid_length // block_size

    expected_expert_ids = paddle.repeat_interleave(
        paddle.arange(num_experts, dtype="int32"),  # CPU
        (aligned // block_size).cast("int32"),
    )

    np.testing.assert_array_equal(
        num_tokens_post_pad.cpu().numpy(),
        cumsum[-1:].cpu().numpy(),
    )

    # ------------------------------------------------------------------ #
    # Check 2: expert_ids – each block maps to the expected expert       #
    # ------------------------------------------------------------------ #
    got_eids = expert_ids[:num_blocks].cpu()
    want_eids = expected_expert_ids.cpu()
    np.testing.assert_array_equal(
        got_eids.numpy(),
        want_eids.numpy(),
    )

    # ------------------------------------------------------------------ #
    # Check 3: sorted_token_ids – routing correctness per expert         #
    # ------------------------------------------------------------------ #

    start = 0
    for expert_id in range(num_experts):
        end = int(cumsum[expert_id].item())
        tokens = sorted_token_ids[start:end].cpu()
        valid_tokens = tokens[tokens < num_tokens]
        # padding_tokens = tokens[tokens >= num_tokens]

        want_count = int(counts[expert_id].item())
        got_count = valid_tokens.numel()
        count_ok = got_count == want_count

        assert count_ok, f"expert {expert_id}: expected {want_count} valid tokens, got {got_count}"
        if counts[expert_id] > 0:
            np.testing.assert_array_equal(
                topk_ids_flat[valid_tokens.cast("int64")].numpy(),
                paddle.full_like(valid_tokens, expert_id).numpy(),
            )
        start = end

    # padding 区域哨兵检查
    if valid_length < sorted_token_ids.numel():
        padding_region = sorted_token_ids[valid_length:].cpu()
        sentinel_ok = paddle.all(padding_region >= num_tokens).item()

        assert sentinel_ok, "padding slots beyond valid_length contain non-sentinel values"

    print(f"\n{tag}ALL CHECKS PASSED")
    print(sep)


# ---------------------------------------------------------------------------
# Original unittest-based tests (kept for backward compatibility)
# ---------------------------------------------------------------------------


class TestTritonMOEPreprocess(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_op(self, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M):
        """Convert numpy to Paddle Tensor and run operator"""
        topk_ids = paddle.to_tensor(topk_ids_np, dtype="int64")
        sorted_ids, expert_ids, num_tokens_post_pad = tritonmoe_preprocess(topk_ids, num_experts, GEMM_BLOCK_SIZE_M)
        return sorted_ids.numpy(), expert_ids.numpy(), num_tokens_post_pad.numpy()

    def _check_output_shapes(
        self, sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
    ):
        """Check output shapes and dtypes"""
        if topk_ids_np.size < num_experts + 1:
            expected_max_num_tokens_padded = topk_ids_np.size * GEMM_BLOCK_SIZE_M
        else:
            expected_max_num_tokens_padded = topk_ids_np.size + (num_experts + 1) * (GEMM_BLOCK_SIZE_M - 1)

        self.assertEqual(sorted_ids.shape[0], expected_max_num_tokens_padded)

        expected_max_num_m_blocks = (expected_max_num_tokens_padded + GEMM_BLOCK_SIZE_M - 1) // GEMM_BLOCK_SIZE_M
        self.assertEqual(expert_ids.shape[0], expected_max_num_m_blocks)

        self.assertEqual(num_tokens_post_pad.shape[0], 1)
        self.assertTrue(sorted_ids.dtype == np.int32)
        self.assertTrue(expert_ids.dtype == np.int32)
        self.assertTrue(num_tokens_post_pad.dtype == np.int32)

    def _check_output_values_basic(self, sorted_ids, expert_ids, num_tokens_post_pad):
        """Check expected values for the fixed example"""
        expected_sorted_ids = np.array(
            [
                8,
                12,
                16,
                16,
                4,
                9,
                15,
                16,
                5,
                10,
                14,
                16,
                6,
                11,
                13,
                16,
                3,
                7,
                16,
                16,
                2,
                16,
                16,
                16,
                1,
                16,
                16,
                16,
                0,
                16,
                16,
                16,
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(sorted_ids[: len(expected_sorted_ids)], expected_sorted_ids)

        expected_expert_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        np.testing.assert_array_equal(expert_ids[: len(expected_expert_ids)], expected_expert_ids)

        self.assertTrue(num_tokens_post_pad[0] % 4 == 0)

    def test_basic_case(self):
        """Basic fixed example test"""
        num_experts = 8
        GEMM_BLOCK_SIZE_M = 4
        topk_ids_np = np.array([[7, 6, 5, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int64)

        sorted_ids, expert_ids, num_tokens_post_pad = self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)
        self._check_output_shapes(
            sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
        )
        self._check_output_values_basic(sorted_ids, expert_ids, num_tokens_post_pad)


# ---------------------------------------------------------------------------
# Correctness tests (ported from test_moe_align_block_size.py)
# ---------------------------------------------------------------------------


class TestTritonMoePreprocessBasic(unittest.TestCase):
    """Basic / small cases – easy to reason about manually."""

    def setUp(self):
        if not _AVAILABLE:
            self.skipTest("CUDA or fastdeploy not available")

    def test_docstring_example(self):
        """Reproduce the example from the function docstring."""
        topk_ids = paddle.to_tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], dtype="int64")
        _verify(topk_ids, block_size=4, num_experts=5, label="docstring_example")

    def test_single_token_single_expert(self):
        """Minimal input: one token assigned to one expert."""
        topk_ids = paddle.to_tensor([[0]], dtype="int64")
        _verify(topk_ids, block_size=16, num_experts=8, label="single_token_single_expert")

    def test_all_tokens_same_expert(self):
        """All tokens go to expert 0 – only one expert's slot is used."""
        topk_ids = paddle.zeros((64, 1), dtype="int64")
        _verify(topk_ids, block_size=16, num_experts=8, label="all_tokens_same_expert")

    def test_uniform_1d(self):
        """1-D topk_ids (top_k=1 squeezed) with uniform distribution."""
        paddle.seed(42)
        topk_ids = paddle.randint(0, 8, (128,), dtype="int64")
        _verify(topk_ids, block_size=16, num_experts=8, label="uniform_1d")

    def test_topk_equals_num_experts(self):
        """Every token selects all experts (top_k == num_experts)."""
        num_experts = 4
        topk_ids = paddle.arange(num_experts, dtype="int64").unsqueeze(0).expand((8, num_experts))
        _verify(topk_ids, block_size=4, num_experts=num_experts, label="topk_equals_num_experts")

    def test_num_tokens_less_than_num_experts(self):
        """Fewer tokens than experts – exercises the small-input branch."""
        topk_ids = paddle.to_tensor([[0], [3]], dtype="int64")
        _verify(topk_ids, block_size=16, num_experts=64, label="num_tokens_less_than_num_experts")

    def test_exact_block_boundary(self):
        """Token count per expert is exactly block_size – no padding needed."""
        block_size = 16
        num_experts = 4
        topk_ids = paddle.concat([paddle.full((block_size,), e, dtype="int64") for e in range(num_experts)])
        _verify(topk_ids, block_size=block_size, num_experts=num_experts, label="exact_block_boundary")

    def test_block_size_1(self):
        """block_size=1 means no padding is ever added."""
        paddle.seed(0)
        topk_ids = paddle.randint(0, 16, (64,), dtype="int64")
        _verify(topk_ids, block_size=1, num_experts=16, label="block_size_1")


class TestTritonMoePreprocessEdgeCases(unittest.TestCase):
    """Edge / boundary cases."""

    def setUp(self):
        if not _AVAILABLE:
            self.skipTest("CUDA or fastdeploy not available")

    def test_empty_topk_ids(self):
        """Zero-token input should not crash; num_tokens_post_pad == 0."""
        topk_ids = paddle.empty((0,), dtype="int64").cuda()
        sorted_ids, expert_ids_out, num_post = tritonmoe_preprocess(topk_ids, 8, 16)
        got = int(num_post.item())

        self.assertEqual(got, 0)

    def test_one_expert(self):
        """Single expert: all tokens must end up in expert 0's bucket."""
        paddle.seed(1)
        topk_ids = paddle.zeros((32,), dtype="int64")
        _verify(topk_ids, block_size=8, num_experts=1, label="one_expert")

    def test_large_block_size(self):
        """block_size larger than total tokens."""
        topk_ids = paddle.randint(0, 4, (8,), dtype="int64")
        _verify(topk_ids, block_size=128, num_experts=4, label="large_block_size")

    def test_int64_dtype(self):
        """topk_ids in int64 – the kernel should handle dtype conversion."""
        paddle.seed(7)
        topk_ids = paddle.randint(0, 8, (64, 2), dtype="int64")
        _verify(topk_ids, block_size=16, num_experts=8, label="int64_dtype")


class TestTritonMoePreprocessRealistic(unittest.TestCase):
    """Larger, more realistic MoE shapes."""

    def setUp(self):
        if not _AVAILABLE:
            self.skipTest("CUDA or fastdeploy not available")

    def _run_uniform_distribution(self, num_tokens, num_experts, block_size):
        """Uniform random token-to-expert assignment across common MoE shapes."""
        paddle.seed(0)
        topk_ids = paddle.randint(0, num_experts, (num_tokens,), dtype="int64")
        _verify(
            topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            label=f"uniform_T{num_tokens}_E{num_experts}_B{block_size}",
        )

    def test_uniform_distribution(self):
        """Uniform random token-to-expert assignment across common MoE shapes."""
        for num_tokens, num_experts, block_size in [
            (256, 8, 16),
            (1024, 16, 16),
            (4096, 64, 16),
            (8192, 64, 32),
            (8192, 128, 64),
            (16384, 128, 128),
            (16384, 256, 128),
            (16384, 512, 256),
            (32768, 512, 256),
            (32768, 512, 64),
            (163840, 1024, 256),
        ]:
            with self.subTest(num_tokens=num_tokens, num_experts=num_experts, block_size=block_size):
                self._run_uniform_distribution(num_tokens, num_experts, block_size)

    def _run_topk_2d(self, num_tokens, top_k, num_experts, block_size):
        """2-D topk_ids as produced by the router (shape [num_tokens, top_k])."""
        paddle.seed(0)
        topk_ids = paddle.randint(0, num_experts, (num_tokens, top_k), dtype="int64")
        _verify(
            topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            label=f"topk2d_T{num_tokens}_K{top_k}_E{num_experts}_B{block_size}",
        )

    def test_topk_2d(self):
        """2-D topk_ids as produced by the router (shape [num_tokens, top_k])."""
        for num_tokens, top_k, num_experts, block_size in [
            (512, 2, 8, 16),
            (1024, 4, 16, 16),
            (2048, 8, 64, 16),
        ]:
            with self.subTest(num_tokens=num_tokens, top_k=top_k, num_experts=num_experts, block_size=block_size):
                self._run_topk_2d(num_tokens, top_k, num_experts, block_size)

    def _run_zipf_distribution(self, alpha):
        """Skewed (Zipf) token distribution – simulates real MoE load imbalance."""
        num_tokens, num_experts, block_size = 8192, 64, 16
        ranks = paddle.arange(1, num_experts + 1, dtype="float32")
        probs = 1.0 / ranks**alpha
        probs = probs / probs.sum()
        paddle.seed(0)
        topk_ids = paddle.multinomial(probs, num_tokens, replacement=True).cast("int64")
        _verify(topk_ids, block_size=block_size, num_experts=num_experts, label=f"zipf_alpha{alpha}")

    def test_zipf_distribution(self):
        """Skewed (Zipf) token distribution – simulates real MoE load imbalance."""
        for alpha in [0.5, 1.2, 2.0]:
            with self.subTest(alpha=alpha):
                self._run_zipf_distribution(alpha)

    def test_deterministic_with_fixed_seed(self):
        """Same seed must produce the same outputs (kernel is deterministic)."""
        num_tokens, num_experts, block_size = 4096, 64, 16

        paddle.seed(99)
        topk_ids = paddle.randint(0, num_experts, (num_tokens,), dtype="int64").cuda()
        s1, e1, n1 = tritonmoe_preprocess(topk_ids, num_experts, block_size)

        paddle.seed(99)
        topk_ids2 = paddle.randint(0, num_experts, (num_tokens,), dtype="int64").cuda()
        s2, e2, n2 = tritonmoe_preprocess(topk_ids2, num_experts, block_size)

        valid = int(n1.item())

        np.testing.assert_array_equal(n1.numpy(), n2.numpy())
        np.testing.assert_array_equal(e1[: valid // block_size].numpy(), e2[: valid // block_size].numpy())
        np.testing.assert_array_equal(paddle.sort(s1[:valid]).numpy(), paddle.sort(s2[:valid]).numpy())


# ---------------------------------------------------------------------------
# Direct-run entry point  (python test_tritonmoe_preprocess.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _AVAILABLE:
        print("SKIP: CUDA or fastdeploy not available.")
    else:
        basic = TestTritonMoePreprocessBasic()
        basic.test_docstring_example()
        basic.test_single_token_single_expert()
        basic.test_all_tokens_same_expert()
        basic.test_uniform_1d()
        basic.test_topk_equals_num_experts()
        basic.test_num_tokens_less_than_num_experts()
        basic.test_exact_block_boundary()
        basic.test_block_size_1()

        edge = TestTritonMoePreprocessEdgeCases()
        edge.test_empty_topk_ids()
        edge.test_one_expert()
        edge.test_large_block_size()
        edge.test_int64_dtype()

        real = TestTritonMoePreprocessRealistic()
        for num_tokens, num_experts, block_size in [
            (256, 8, 16),
            (1024, 16, 16),
            (4096, 64, 16),
            (8192, 64, 32),
            (8192, 128, 64),
            (16384, 256, 128),
        ]:
            real._run_uniform_distribution(num_tokens, num_experts, block_size)
        for num_tokens, top_k, num_experts, block_size in [
            (512, 2, 8, 16),
            (1024, 4, 16, 16),
            (2048, 8, 64, 16),
        ]:
            real._run_topk_2d(num_tokens, top_k, num_experts, block_size)
        for alpha in [0.5, 1.2, 2.0]:
            real._run_zipf_distribution(alpha)
        real.test_deterministic_with_fixed_seed()

        print("\n*** All direct-run tests passed ***")
