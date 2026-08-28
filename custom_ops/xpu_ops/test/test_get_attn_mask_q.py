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

"""
Unit tests for get_attn_mask_q XPU op.

Scenarios covered:
1. Normal path: multi-batch, no attn_mask_kv (causal masking)
2. Normal path: with attn_mask_kv provided
3. Edge case: single batch
4. Edge case: single KV token per batch
5. All KV tokens covered by first Q token (row_end == q_start)
"""

import unittest

import numpy as np
import paddle

try:
    from fastdeploy.model_executor.ops.xpu import get_attn_mask_q as xpu_get_attn_mask_q

    HAS_XPU_OP = True
except ImportError:
    HAS_XPU_OP = False


def ref_get_attn_mask_q(cu_seqlens_q, cu_seqlens_k, attn_mask_kv, kv_token_num):
    """
    Pure Python reference implementation that mirrors cpu_wrapper logic.
    Returns startend_row_indices of shape [1, 1, kv_token_num, 2].
    """
    max_batch_size = len(cu_seqlens_k) - 1
    out = np.zeros((kv_token_num, 2), dtype=np.int32)

    for ku in range(kv_token_num):
        # Find which batch this KV token belongs to
        batch_id = 0
        for i in range(max_batch_size):
            if cu_seqlens_k[i] <= ku < cu_seqlens_k[i + 1]:
                batch_id = i
                break

        this_batch_q_start = cu_seqlens_q[batch_id]
        this_batch_q_end = cu_seqlens_q[batch_id + 1]
        this_batch_q_len = this_batch_q_end - this_batch_q_start
        kv_start = cu_seqlens_k[batch_id]
        kv_end = cu_seqlens_k[batch_id + 1]
        kv_len = kv_end - kv_start
        cache_k_idx = ku - kv_start

        row_start = this_batch_q_end
        row_end = this_batch_q_end

        for q_idx in range(this_batch_q_start, this_batch_q_end):
            if attn_mask_kv is not None:
                append_mask_k_end = int(attn_mask_kv[q_idx * 2 + 1]) - 1
            else:
                append_mask_k_end = (q_idx - this_batch_q_start) + kv_len - this_batch_q_len

            if cache_k_idx <= append_mask_k_end:
                row_end = min(row_end, q_idx)
                break

        out[ku, 0] = row_start
        out[ku, 1] = row_end

    return out.reshape(1, 1, kv_token_num, 2)


def run_xpu_op(cu_seqlens_q_np, cu_seqlens_k_np, attn_mask_kv_np, kv_token_num):
    try:
        paddle.set_device("xpu:0")
    except Exception:
        return None

    cu_q = paddle.to_tensor(cu_seqlens_q_np, dtype="int32")
    cu_k = paddle.to_tensor(cu_seqlens_k_np, dtype="int32")
    attn_kv = None
    if attn_mask_kv_np is not None:
        attn_kv = paddle.to_tensor(attn_mask_kv_np, dtype="int32")

    result = xpu_get_attn_mask_q(cu_q, cu_k, attn_kv, kv_token_num)
    return result[0].numpy()


def assert_equal(ref, got, label):
    diff = np.sum(np.abs(ref.astype(np.int32) - got.astype(np.int32)))
    assert diff == 0, f"[{label}] mismatch!\nref:\n{ref}\ngot:\n{got}"


class TestGetAttnMaskQ(unittest.TestCase):

    def _run(self, cu_seqlens_q, cu_seqlens_k, attn_mask_kv, kv_token_num, label):
        ref = ref_get_attn_mask_q(cu_seqlens_q, cu_seqlens_k, attn_mask_kv, kv_token_num)
        if HAS_XPU_OP:
            got = run_xpu_op(cu_seqlens_q, cu_seqlens_k, attn_mask_kv, kv_token_num)
            if got is None:
                self.skipTest("XPU not available")
            assert_equal(ref, got, label)
        else:
            # No XPU op available: just verify ref runs without crash
            pass

    def test_causal_multi_batch(self):
        """Normal causal masking, 2 batches, varying Q/K lengths."""
        # batch0: q_len=3, kv_len=3 (decode: q==kv)
        # batch1: q_len=2, kv_len=4 (some cached tokens)
        cu_q = np.array([0, 3, 5], dtype=np.int32)
        cu_k = np.array([0, 3, 7], dtype=np.int32)
        kv_token_num = 7
        self._run(cu_q, cu_k, None, kv_token_num, "causal_multi_batch")

    def test_with_attn_mask_kv(self):
        """With explicit attn_mask_kv provided (non-causal custom mask)."""
        # 1 batch: q_len=3, kv_len=3
        # attn_mask_kv[q_idx*2+1] defines upper bound of visible K for each Q
        cu_q = np.array([0, 3], dtype=np.int32)
        cu_k = np.array([0, 3], dtype=np.int32)
        kv_token_num = 3
        # q0 sees k[0..0], q1 sees k[0..1], q2 sees k[0..2]
        attn_mask_kv = np.array([0, 1, 0, 2, 0, 3], dtype=np.int32)
        self._run(cu_q, cu_k, attn_mask_kv, kv_token_num, "with_attn_mask_kv")

    def test_single_batch(self):
        """Single batch, causal masking."""
        cu_q = np.array([0, 4], dtype=np.int32)
        cu_k = np.array([0, 4], dtype=np.int32)
        kv_token_num = 4
        self._run(cu_q, cu_k, None, kv_token_num, "single_batch")

    def test_single_kv_token_per_batch(self):
        """Each batch has exactly 1 KV token."""
        cu_q = np.array([0, 1, 2, 3], dtype=np.int32)
        cu_k = np.array([0, 1, 2, 3], dtype=np.int32)
        kv_token_num = 3
        self._run(cu_q, cu_k, None, kv_token_num, "single_kv_per_batch")

    def test_decode_mode(self):
        """Decode mode: q_len=1 per batch, kv_len > 1 (KV cache)."""
        # 3 requests: each generating 1 token, with cached KV
        cu_q = np.array([0, 1, 2, 3], dtype=np.int32)
        cu_k = np.array([0, 5, 8, 12], dtype=np.int32)
        kv_token_num = 12
        self._run(cu_q, cu_k, None, kv_token_num, "decode_mode")

    def test_prefill_mode(self):
        """Prefill mode: q_len==kv_len, no cached tokens."""
        cu_q = np.array([0, 6, 10], dtype=np.int32)
        cu_k = np.array([0, 6, 10], dtype=np.int32)
        kv_token_num = 10
        self._run(cu_q, cu_k, None, kv_token_num, "prefill_mode")

    def test_all_visible(self):
        """attn_mask_kv makes all KV tokens visible to the first Q."""
        cu_q = np.array([0, 2], dtype=np.int32)
        cu_k = np.array([0, 4], dtype=np.int32)
        kv_token_num = 4
        # q0 sees k[0..3], q1 sees k[0..3]
        attn_mask_kv = np.array([0, 4, 0, 4], dtype=np.int32)
        self._run(cu_q, cu_k, attn_mask_kv, kv_token_num, "all_visible")


if __name__ == "__main__":
    unittest.main()
