# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Unit tests for the `grouped_topk` custom CUDA op (low-level interface).

grouped_topk fuses sigmoid into the kernel and accepts raw logits directly,
unlike noaux_tc which requires Python-side sigmoid preprocessing.

Algorithm:
  1. scores           = sigmoid(gating_output)   [fused inside kernel]
  2. scores_with_bias = scores + e_score_correction_bias
  3. group_scores     = sum of top-2 biased expert scores per group
  4. Select top-topk_group groups
  5. Within selected groups select top-topk experts by biased score
  6. Gather unbiased sigmoid scores for selected experts as topk_values
  7. Optionally renormalize and scale by routed_scaling_factor

Model configs covered:
  DeepSeek-V3 / R1   num_experts=256, n_group=8, topk_group=4, topk=8,  renorm=True,  scale=2.5
  GLM-4.5-Air        num_experts=128, n_group=1, topk_group=1, topk=8,  renorm=True,  scale=1.0
  Qwen3-30B-A3B      num_experts=128, n_group=4, topk_group=2, topk=8,  renorm=False, scale=1.0
  Kimi-K2            num_experts=384, n_group=8, topk_group=2, topk=8,  renorm=False, scale=1.0
"""

import unittest

import numpy as np
import paddle

try:
    from fastdeploy.model_executor.ops.gpu import grouped_topk

    _GROUPED_TOPK_AVAILABLE = True
except Exception:
    _GROUPED_TOPK_AVAILABLE = False


def native_grouped_topk(
    gating_output: paddle.Tensor,
    e_score_correction_bias: paddle.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    """Pure-Python reference that mirrors the grouped_topk kernel semantics.

    Args:
        gating_output: raw logits, shape [num_tokens, num_experts]
        e_score_correction_bias: bias added to sigmoid scores, shape [1, num_experts] or [num_experts]
        n_group: number of expert groups
        topk_group: number of groups selected per token
        topk: number of experts selected per token
        renormalize: whether to L1-normalise the selected weights
        routed_scaling_factor: multiplicative scale applied after renorm

    Returns:
        (scores_out, topk_values, topk_indices)
          scores_out   – sparse score tensor, shape [num_tokens, num_experts]
          topk_values  – weights for selected experts, shape [num_tokens, topk]
          topk_indices – expert indices, shape [num_tokens, topk] (int64)
    """
    num_tokens, num_experts = gating_output.shape
    experts_per_group = num_experts // n_group

    scores = paddle.nn.functional.sigmoid(gating_output)
    scores_with_bias = scores + e_score_correction_bias

    # Step 1: group scores = sum of top-2 biased scores per group
    biased = scores_with_bias.reshape([num_tokens, n_group, experts_per_group])
    group_scores = biased.topk(min(2, experts_per_group), axis=-1)[0].sum(axis=-1)

    # Step 2: select top-topk_group groups
    group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]
    group_mask = paddle.zeros_like(group_scores)
    group_mask.put_along_axis_(group_idx, paddle.ones_like(group_idx, dtype=group_mask.dtype), axis=-1)
    score_mask = (
        group_mask.unsqueeze(-1).expand([num_tokens, n_group, experts_per_group]).reshape([num_tokens, num_experts])
    )

    # Step 3: select top-topk experts within selected groups (biased score)
    tmp_scores = scores_with_bias.masked_fill(~score_mask.cast(paddle.bool), float("-inf"))
    topk_indices = paddle.topk(tmp_scores, topk, axis=-1)[1]

    # Step 4: gather unbiased sigmoid scores
    topk_values = paddle.take_along_axis(scores, topk_indices, axis=1)

    # Step 5: renormalize + scale
    if renormalize:
        topk_values = topk_values / (topk_values.sum(axis=-1, keepdim=True) + 1e-20)
    if routed_scaling_factor != 1.0:
        topk_values = topk_values * routed_scaling_factor

    scores_out = paddle.zeros_like(scores)
    scores_out.put_along_axis_(topk_indices, topk_values, axis=1)

    return scores_out, topk_values, topk_indices.cast(paddle.int64)


@unittest.skipUnless(_GROUPED_TOPK_AVAILABLE, "grouped_topk custom op not available (not compiled)")
class TestGroupedTopkOp(unittest.TestCase):
    """Tests for the grouped_topk custom CUDA op."""

    ATOL = 1e-3
    RTOL = 1e-3

    def setUp(self):
        paddle.seed(42)

    # ------------------------------------------------------------------
    # Parametrised helper
    # ------------------------------------------------------------------
    def _run_case(
        self,
        num_tokens: int,
        num_experts: int,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
        input_dtype=paddle.float32,
        bias_scale: float = 0.1,
        seed: int = 42,
    ):
        paddle.seed(seed)
        gating = paddle.randn([num_tokens, num_experts], dtype=input_dtype)
        bias = (paddle.rand([1, num_experts], dtype=paddle.float32) - 0.5) * bias_scale

        # Reference always runs in fp32
        gating_fp32 = gating.cast(paddle.float32) if input_dtype != paddle.float32 else gating
        ref_scores, ref_tv, ref_ti = native_grouped_topk(
            gating_fp32.clone(),
            bias.clone(),
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )

        op_scores, op_tv, op_ti = grouped_topk(
            gating.clone(),
            bias.clone(),
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )

        label = (
            f"T={num_tokens}, E={num_experts}, n_group={n_group}, "
            f"topk_group={topk_group}, topk={topk}, "
            f"renorm={renormalize}, scale={routed_scaling_factor}, dtype={input_dtype}"
        )

        self.assertEqual(op_tv.shape, [num_tokens, topk], f"[{label}] topk_values shape")
        self.assertEqual(op_ti.shape, [num_tokens, topk], f"[{label}] topk_indices shape")
        self.assertEqual(op_ti.dtype, paddle.int64, f"[{label}] topk_indices dtype")
        self.assertEqual(op_tv.dtype, paddle.float32, f"[{label}] topk_values dtype")

        # Compare set-level index match (position order not guaranteed)
        ref_sorted = paddle.sort(ref_ti, axis=-1)
        op_sorted = paddle.sort(op_ti, axis=-1)
        if not paddle.equal_all(ref_sorted, op_sorted).item():
            n_diff = (ref_sorted != op_sorted).sum().item()
            self.fail(f"[{label}] topk_indices set mismatch: {n_diff} positions differ")

        # Align values by expert index before comparing
        ref_ord = paddle.argsort(ref_ti, axis=-1)
        op_ord = paddle.argsort(op_ti, axis=-1)
        ref_tv_s = paddle.take_along_axis(ref_tv, ref_ord, axis=-1)
        op_tv_s = paddle.take_along_axis(op_tv, op_ord, axis=-1)
        if not paddle.allclose(op_tv_s, ref_tv_s, atol=self.ATOL, rtol=self.RTOL).item():
            max_diff = (op_tv_s - ref_tv_s).abs().max().item()
            self.fail(f"[{label}] topk_values max_diff={max_diff:.2e}")

    # ------------------------------------------------------------------
    # GLM-4.5-Air: n_experts=128, n_group=1, topk_group=1, topk=8, renorm=True
    # ------------------------------------------------------------------
    def test_glm45air_T1(self):
        self._run_case(1, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T32(self):
        self._run_case(32, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T128(self):
        self._run_case(128, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T512(self):
        self._run_case(512, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T1024(self):
        self._run_case(1024, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T4096(self):
        self._run_case(4096, 128, 1, 1, 8, True, 1.0)

    def test_glm45air_T8192(self):
        self._run_case(8192, 128, 1, 1, 8, True, 1.0)

    # ------------------------------------------------------------------
    # DeepSeek-V3 / R1: n_experts=256, n_group=8, topk_group=4, topk=8,
    #                   renorm=True, scale=2.5
    # ------------------------------------------------------------------
    def test_deepseek_v3_T1(self):
        self._run_case(1, 256, 8, 4, 8, True, 2.5)

    def test_deepseek_v3_T32(self):
        self._run_case(32, 256, 8, 4, 8, True, 2.5)

    def test_deepseek_v3_T128(self):
        self._run_case(128, 256, 8, 4, 8, True, 2.5)

    def test_deepseek_v3_T512(self):
        self._run_case(512, 256, 8, 4, 8, True, 2.5)

    def test_deepseek_v3_T4096(self):
        self._run_case(4096, 256, 8, 4, 8, True, 2.5)

    def test_deepseek_v3_T8192(self):
        self._run_case(8192, 256, 8, 4, 8, True, 2.5)

    # ------------------------------------------------------------------
    # Qwen3-30B-A3B: n_experts=128, n_group=4, topk_group=2, topk=8,
    #                renorm=False
    # ------------------------------------------------------------------
    def test_qwen3_30b_T1(self):
        self._run_case(1, 128, 4, 2, 8, False, 1.0)

    def test_qwen3_30b_T128(self):
        self._run_case(128, 128, 4, 2, 8, False, 1.0)

    def test_qwen3_30b_T512(self):
        self._run_case(512, 128, 4, 2, 8, False, 1.0)

    def test_qwen3_30b_T4096(self):
        self._run_case(4096, 128, 4, 2, 8, False, 1.0)

    # ------------------------------------------------------------------
    # Kimi-K2: n_experts=384, n_group=8, topk_group=2, topk=8, renorm=False
    # ------------------------------------------------------------------
    def test_kimi_k2_T1(self):
        self._run_case(1, 384, 8, 2, 8, False, 1.0)

    def test_kimi_k2_T128(self):
        self._run_case(128, 384, 8, 2, 8, False, 1.0)

    def test_kimi_k2_T512(self):
        self._run_case(512, 384, 8, 2, 8, False, 1.0)

    def test_kimi_k2_T4096(self):
        self._run_case(4096, 384, 8, 2, 8, False, 1.0)

    # ------------------------------------------------------------------
    # bfloat16 input path: kernel should cast internally to fp32
    # ------------------------------------------------------------------
    def test_bf16_input_glm45air(self):
        self._run_case(128, 128, 1, 1, 8, True, 1.0, input_dtype=paddle.bfloat16)

    def test_bf16_input_deepseek_v3(self):
        self._run_case(128, 256, 8, 4, 8, True, 2.5, input_dtype=paddle.bfloat16)

    def test_bf16_input_qwen3_30b(self):
        self._run_case(128, 128, 4, 2, 8, False, 1.0, input_dtype=paddle.bfloat16)

    # ------------------------------------------------------------------
    # Output shape and dtype sanity
    # ------------------------------------------------------------------
    def test_output_shapes(self):
        """Verify output shapes for various (T, E, topk) combinations."""
        cases = [
            (1, 128, 1, 1, 8),
            (32, 256, 8, 4, 8),
            (64, 384, 8, 2, 8),
        ]
        for T, E, ng, tkg, topk in cases:
            gating = paddle.randn([T, E], dtype=paddle.float32)
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            _, tv, ti = grouped_topk(gating, bias, ng, tkg, topk, True, 1.0)
            self.assertEqual(tv.shape, [T, topk], f"T={T},E={E}: topk_values shape")
            self.assertEqual(ti.shape, [T, topk], f"T={T},E={E}: topk_indices shape")

    def test_output_dtype_is_float32(self):
        """topk_values must always be float32 regardless of input dtype."""
        for dtype in [paddle.float32, paddle.bfloat16]:
            gating = paddle.randn([16, 128], dtype=dtype)
            bias = paddle.zeros([1, 128], dtype=paddle.float32)
            _, tv, ti = grouped_topk(gating, bias, 1, 1, 8, True, 1.0)
            self.assertEqual(tv.dtype, paddle.float32, f"dtype={dtype}: topk_values not float32")
            self.assertEqual(ti.dtype, paddle.int64, f"dtype={dtype}: topk_indices not int64")

    # ------------------------------------------------------------------
    # Correctness invariants
    # ------------------------------------------------------------------
    def test_topk_indices_in_valid_range(self):
        """All selected expert indices must lie in [0, num_experts)."""
        for E, ng, tkg, topk in [(128, 1, 1, 8), (256, 8, 4, 8), (384, 8, 2, 8)]:
            gating = paddle.randn([64, E], dtype=paddle.float32)
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            _, _, ti = grouped_topk(gating, bias, ng, tkg, topk, True, 1.0)
            self.assertTrue((ti >= 0).all().item(), f"E={E}: negative index found")
            self.assertTrue((ti < E).all().item(), f"E={E}: index >= num_experts")

    def test_no_duplicate_experts_per_token(self):
        """Each token must select exactly topk distinct experts."""
        for E, ng, tkg, topk in [(128, 1, 1, 8), (256, 8, 4, 8)]:
            gating = paddle.randn([32, E], dtype=paddle.float32)
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            _, _, ti = grouped_topk(gating, bias, ng, tkg, topk, True, 1.0)
            for row in ti.numpy():
                self.assertEqual(len(set(row.tolist())), topk, f"E={E}: duplicate expert indices in row {row}")

    def test_topk_values_non_negative(self):
        """Sigmoid output is in (0,1); routing weights must be >= 0."""
        gating = paddle.randn([64, 128], dtype=paddle.float32)
        bias = paddle.zeros([1, 128], dtype=paddle.float32)
        _, tv, _ = grouped_topk(gating, bias, 1, 1, 8, True, 1.0)
        self.assertTrue((tv >= 0).all().item(), "topk_values contains negative weights")

    def test_renormalized_weights_sum_to_one(self):
        """With renormalize=True and scale=1.0, per-token weights sum ≈ 1."""
        num_tokens = 64
        gating = paddle.randn([num_tokens, 128], dtype=paddle.float32)
        bias = paddle.zeros([1, 128], dtype=paddle.float32)
        _, tv, _ = grouped_topk(gating, bias, 1, 1, 8, True, 1.0)
        row_sums = tv.sum(axis=-1).numpy()
        np.testing.assert_allclose(
            row_sums,
            np.ones(num_tokens, dtype=np.float32),
            atol=1e-3,
            err_msg="Renormalized weights do not sum to 1 per token",
        )

    def test_scaled_weights_sum(self):
        """With renormalize=True and scale=2.5, per-token weights sum ≈ 2.5."""
        num_tokens, scale = 64, 2.5
        gating = paddle.randn([num_tokens, 256], dtype=paddle.float32)
        bias = paddle.zeros([1, 256], dtype=paddle.float32)
        _, tv, _ = grouped_topk(gating, bias, 8, 4, 8, True, scale)
        row_sums = tv.sum(axis=-1).numpy()
        np.testing.assert_allclose(
            row_sums,
            np.full(num_tokens, scale, dtype=np.float32),
            atol=1e-2,
            err_msg=f"Scaled weights do not sum to {scale} per token",
        )

    def test_no_renorm_weights_are_raw_sigmoid(self):
        """With renormalize=False, topk_values must equal sigmoid(logits) at selected positions."""
        num_tokens, E = 32, 128
        gating = paddle.randn([num_tokens, E], dtype=paddle.float32)
        bias = paddle.zeros([1, E], dtype=paddle.float32)
        _, tv, ti = grouped_topk(gating, bias, 1, 1, 8, False, 1.0)
        expected = paddle.take_along_axis(paddle.nn.functional.sigmoid(gating), ti, axis=1)
        np.testing.assert_allclose(
            tv.numpy(),
            expected.numpy(),
            atol=1e-4,
            err_msg="Without renorm, topk_values should equal sigmoid(gating) at selected positions",
        )

    def test_deterministic(self):
        """Two identical calls must produce bit-for-bit identical outputs."""
        gating = paddle.randn([32, 256], dtype=paddle.float32)
        bias = (paddle.rand([1, 256], dtype=paddle.float32) - 0.5) * 0.1
        _, tv1, ti1 = grouped_topk(gating.clone(), bias.clone(), 8, 4, 8, True, 2.5)
        _, tv2, ti2 = grouped_topk(gating.clone(), bias.clone(), 8, 4, 8, True, 2.5)
        self.assertTrue(
            paddle.allclose(tv1, tv2, atol=0.0, rtol=0.0).item(),
            "topk_values not deterministic across two identical calls",
        )
        self.assertTrue(
            paddle.equal_all(ti1, ti2).item(),
            "topk_indices not deterministic across two identical calls",
        )

    def test_zero_bias(self):
        """All-zero bias: biased == unbiased; reference and op must agree."""
        for E, ng, tkg, topk in [(128, 1, 1, 8), (256, 8, 4, 8)]:
            paddle.seed(16)
            gating = paddle.randn([32, E], dtype=paddle.float32)
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            _, ref_tv, ref_ti = native_grouped_topk(gating.clone(), bias, ng, tkg, topk, True, 1.0)
            _, op_tv, op_ti = grouped_topk(gating.clone(), bias, ng, tkg, topk, True, 1.0)
            ref_s = paddle.sort(ref_ti, axis=-1)
            op_s = paddle.sort(op_ti, axis=-1)
            self.assertTrue(
                paddle.equal_all(ref_s, op_s).item(),
                f"E={E}/zero_bias: topk_indices set mismatch",
            )

    def test_large_bias_steers_routing(self):
        """Large positive bias on first half of experts must dominate selection."""
        E, topk = 128, 8
        paddle.seed(17)
        gating = paddle.randn([64, E], dtype=paddle.float32)
        bias = paddle.concat(
            [
                paddle.full([1, E // 2], 2.0, dtype=paddle.float32),
                paddle.full([1, E // 2], -2.0, dtype=paddle.float32),
            ],
            axis=1,
        )
        _, _, ti = grouped_topk(gating, bias, 1, 1, topk, True, 1.0)
        self.assertTrue(
            (ti < E // 2).all().item(),
            "Large positive bias on experts [0, E/2) did not steer all selections there",
        )

    def test_extreme_logits_no_nan_inf(self):
        """Very large logits must not produce NaN or Inf in outputs."""
        for E, ng, tkg, topk in [(128, 1, 1, 8), (256, 8, 4, 8)]:
            paddle.seed(18)
            gating = paddle.randn([8, E], dtype=paddle.float32) * 50.0
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            _, tv, _ = grouped_topk(gating, bias, ng, tkg, topk, False, 1.0)
            self.assertFalse(paddle.isnan(tv).any().item(), f"E={E}: NaN in topk_values")
            self.assertFalse(paddle.isinf(tv).any().item(), f"E={E}: Inf in topk_values")

    def test_single_expert_selected(self):
        """topk=1: each token selects exactly one expert; weight == 1.0 with renorm."""
        num_tokens = 16
        gating = paddle.randn([num_tokens, 128], dtype=paddle.float32)
        bias = paddle.zeros([1, 128], dtype=paddle.float32)
        _, tv, ti = grouped_topk(gating, bias, 1, 1, 1, True, 1.0)
        self.assertEqual(tv.shape, [num_tokens, 1])
        self.assertEqual(ti.shape, [num_tokens, 1])
        np.testing.assert_allclose(
            tv.numpy(),
            np.ones((num_tokens, 1), dtype=np.float32),
            atol=1e-5,
            err_msg="With topk=1 and renorm=True, each weight should be 1.0",
        )

    def test_sparse_scores_consistency(self):
        """Sparse scores tensor: non-zero at selected positions must equal topk_values; zero elsewhere."""
        for E, ng, tkg, topk in [(128, 1, 1, 8), (256, 8, 4, 8)]:
            gating = paddle.randn([16, E], dtype=paddle.float32)
            bias = paddle.zeros([1, E], dtype=paddle.float32)
            s, tv, ti = grouped_topk(gating, bias, ng, tkg, topk, True, 1.0)
            gathered = paddle.take_along_axis(s, ti, axis=1)
            np.testing.assert_allclose(
                gathered.numpy(),
                tv.numpy(),
                atol=1e-6,
                err_msg=f"E={E}: sparse scores at topk positions != topk_values",
            )
            nonzero_count = (s != 0).sum(axis=-1)
            self.assertTrue(
                (nonzero_count == topk).all().item(),
                f"E={E}: non-zero count per token != topk",
            )

    def test_irregular_token_counts(self):
        """Non-power-of-2 token counts must produce correct shapes and values."""
        irregular_T = [3, 7, 15, 33, 65, 127, 129, 257, 511, 513, 900]
        for T in irregular_T:
            gating = paddle.randn([T, 128], dtype=paddle.float32)
            bias = (paddle.rand([1, 128], dtype=paddle.float32) - 0.5) * 0.1
            _, ref_tv, ref_ti = native_grouped_topk(gating.clone(), bias.clone(), 1, 1, 8, True, 1.0)
            _, op_tv, op_ti = grouped_topk(gating.clone(), bias.clone(), 1, 1, 8, True, 1.0)
            self.assertEqual(op_tv.shape, [T, 8], f"T={T}: topk_values shape mismatch")
            self.assertEqual(op_ti.shape, [T, 8], f"T={T}: topk_indices shape mismatch")
            ref_s = paddle.sort(ref_ti, axis=-1)
            op_s = paddle.sort(op_ti, axis=-1)
            if not paddle.equal_all(ref_s, op_s).item():
                n_diff = (ref_s != op_s).sum().item()
                self.fail(f"T={T}: topk_indices mismatch, {n_diff} positions differ")


if __name__ == "__main__":
    unittest.main()
