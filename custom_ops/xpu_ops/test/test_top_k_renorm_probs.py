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
Unit tests for top_k_renorm_probs XPU custom op.

Scenarios covered:
1. Normal top_k=2, vocab_size=4 (standard filtering)
2. top_k=0  -> no filtering, all probs kept, sum=1
3. top_k=1  -> only max value kept, normalizer=1/max
4. top_k >= vocab_size -> equivalent to no filtering
5. fp32 / fp16 / bf16 dtype
"""

import unittest

import numpy as np
import paddle


def top_k_renorm_probs_ref(probs_np, top_k_np):
    """CPU reference implementation of ternary search top_k renorm."""
    batch_size, vocab_size = probs_np.shape
    out = np.zeros_like(probs_np, dtype=np.float32)
    for i in range(batch_size):
        row = probs_np[i].astype(np.float32)
        k = int(top_k_np[i])
        if k == 0 or k >= vocab_size:
            out[i] = row
            continue
        # Phase 1: max
        max_val = float(np.max(row))
        # Phase 2: ternary search
        low, high = 0.0, float(max_val)
        sum_low = 1.0
        for _ in range(200):  # enough iterations
            p0 = (high + 2.0 * low) / 3.0
            p1 = (2.0 * high + low) / 3.0
            sg0 = float(np.sum(row[row > p0]))
            sg1 = float(np.sum(row[row > p1]))
            cg0 = int(np.sum(row > p0))
            cg1 = int(np.sum(row > p1))
            above_low = row[row > low]
            below_high = row[row <= high]
            min_gt_low = float(np.min(above_low)) if len(above_low) > 0 else high
            max_le_high = float(np.max(below_high)) if len(below_high) > 0 else low
            if cg1 >= k:
                low = p1
                sum_low = sg1
            elif cg0 >= k:
                low = p0
                high = min(p1, max_le_high)
                sum_low = sg0
            else:
                high = min(p0, max_le_high)
            if min_gt_low == max_le_high:
                break
        normalizer = 1.0 / max(sum_low, 1e-8)
        pivot = low
        out[i] = np.where(row > pivot, row * normalizer, 0.0)
    return out


class TestTopKRenormProbs(unittest.TestCase):
    """Tests for top_k_renorm_probs XPU custom op."""

    def setUp(self):
        try:
            from fastdeploy.model_executor.ops.xpu import top_k_renorm_probs

            self.op = top_k_renorm_probs
        except ImportError as e:
            self.skipTest(f"top_k_renorm_probs not available: {e}")
        np.random.seed(42)

    def _run(self, probs_np, top_k_np, dtype=paddle.float32, atol=1e-5, rtol=1e-4):
        """Helper: run op and compare against reference."""
        ref = top_k_renorm_probs_ref(probs_np, top_k_np).astype(np.float32)
        probs_pd = paddle.to_tensor(probs_np, dtype=dtype)
        top_k_pd = paddle.to_tensor(top_k_np, dtype=paddle.int64)
        out_pd = self.op(probs_pd, top_k_pd)
        got = out_pd.cast(paddle.float32).numpy()
        max_diff = float(np.max(np.abs(got - ref)))
        if not np.allclose(got, ref, atol=atol, rtol=rtol):
            self.fail(
                f"top_k_renorm_probs mismatch (dtype={dtype})\n" f"  ref={ref}\n  got={got}\n  max_diff={max_diff}"
            )

    # ------------------------------------------------------------------
    # 1. Normal: top_k=2, vocab_size=4
    # ------------------------------------------------------------------
    def test_normal_top_k2(self):
        """top_k=2 keeps 2 largest values and renormalises."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([2], dtype=np.int64)
        self._run(probs, top_k)

    def test_normal_top_k2_batch(self):
        """top_k=2 with batch_size=3."""
        probs = np.array(
            [[0.05, 0.45, 0.35, 0.15], [0.25, 0.25, 0.25, 0.25], [0.6, 0.1, 0.2, 0.1]],
            dtype=np.float32,
        )
        top_k = np.array([2, 2, 2], dtype=np.int64)
        # verify output sums ≈ 1.0 for each row
        ref = top_k_renorm_probs_ref(probs, top_k)
        for i in range(probs.shape[0]):
            s = float(np.sum(ref[i]))
            self.assertAlmostEqual(s, 1.0, places=5, msg=f"Row {i} sum={s} != 1.0 (ref={ref[i]})")
        self._run(probs, top_k, atol=1e-4)

    # ------------------------------------------------------------------
    # 2. top_k=0 / top_k=-1 -> no filtering
    # ------------------------------------------------------------------
    def test_top_k_zero(self):
        """top_k=0 means no filtering; output == input."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([0], dtype=np.int64)
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        np.testing.assert_allclose(out, probs, atol=1e-6, err_msg=f"top_k=0: expected copy of input.\n  got={out}")

    def test_top_k_minus_one(self):
        """top_k=-1 is 'disable' per sampling_params.py; must behave same as top_k=0.

        GPU converts -1 (int64) -> uint32 -> 0xFFFFFFFF >> vocab_size -> no filter.
        XPU must match: output == input.
        Critical for mixed-batch [-1, 50, ...] correctness.
        """
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([-1], dtype=np.int64)
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        np.testing.assert_allclose(
            out, probs, atol=1e-6, err_msg=f"top_k=-1: expected copy of input (no filter).\n  got={out}"
        )

    def test_mixed_batch_neg1_and_positive(self):
        """Mixed batch [-1, 2]: row0 no filter, row1 top_k=2.

        This is the exact scenario that exposed the bug.
        """
        probs = np.array(
            [[0.1, 0.4, 0.3, 0.2], [0.1, 0.4, 0.3, 0.2]],  # top_k=-1 → copy unchanged  # top_k=2  → keep top-2, renorm
            dtype=np.float32,
        )
        top_k = np.array([-1, 2], dtype=np.int64)
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        # Row 0: no filter
        np.testing.assert_allclose(
            out[0], probs[0], atol=1e-6, err_msg=f"mixed batch row0 (top_k=-1): expected no filter.\n  got={out[0]}"
        )
        # Row 1: top_k=2, two non-zero values, sum≈1
        n_nonzero = int(np.sum(out[1] > 0))
        self.assertEqual(
            n_nonzero, 2, msg=f"mixed batch row1 (top_k=2): expected 2 non-zero, got {n_nonzero}: {out[1]}"
        )
        self.assertAlmostEqual(float(out[1].sum()), 1.0, places=5)

    # ------------------------------------------------------------------
    # 3. top_k=1 -> only the maximum value is kept
    # ------------------------------------------------------------------
    def test_top_k_one(self):
        """top_k=1: only the largest prob survives, normalised to 1.0."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([1], dtype=np.int64)
        self._run(probs, top_k)
        # independently check that exactly one position is non-zero
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        n_nonzero = int(np.sum(out > 0))
        self.assertEqual(n_nonzero, 1, msg=f"top_k=1 should give 1 non-zero, got {n_nonzero}: {out}")
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    # ------------------------------------------------------------------
    # 4. top_k >= vocab_size -> equivalent to no filtering
    # ------------------------------------------------------------------
    def test_top_k_ge_vocab_size(self):
        """top_k >= vocab_size: output == input."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([10], dtype=np.int64)  # 10 > 4
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        np.testing.assert_allclose(
            out, probs, atol=1e-6, err_msg=f"top_k>=vocab: expected copy of input.\n  got={out}"
        )

    # ------------------------------------------------------------------
    # 5. dtype tests
    # ------------------------------------------------------------------
    def test_fp16(self):
        """float16 dtype."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([2], dtype=np.int64)
        self._run(probs, top_k, dtype=paddle.float16, atol=1e-2, rtol=1e-2)

    def test_bf16(self):
        """bfloat16 dtype."""
        probs = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
        top_k = np.array([2], dtype=np.int64)
        self._run(probs, top_k, dtype=paddle.bfloat16, atol=1e-2, rtol=1e-2)

    # ------------------------------------------------------------------
    # 6. Larger vocab sanity check
    # ------------------------------------------------------------------
    def test_large_vocab(self):
        """Larger vocab_size with random probs."""
        vocab_size = 32000
        batch_size = 2
        probs = np.random.rand(batch_size, vocab_size).astype(np.float32)
        # Normalise so each row sums to 1
        probs /= probs.sum(axis=-1, keepdims=True)
        top_k = np.array([50, 100], dtype=np.int64)
        self._run(probs, top_k, atol=1e-4, rtol=1e-3)
        # Check output sums ≈ 1
        probs_pd = paddle.to_tensor(probs, dtype=paddle.float32)
        top_k_pd = paddle.to_tensor(top_k, dtype=paddle.int64)
        out = self.op(probs_pd, top_k_pd).numpy()
        for i in range(batch_size):
            s = float(out[i].sum())
            self.assertAlmostEqual(s, 1.0, places=3, msg=f"Large vocab row {i} sum={s} != 1.0")


if __name__ == "__main__":
    unittest.main()
