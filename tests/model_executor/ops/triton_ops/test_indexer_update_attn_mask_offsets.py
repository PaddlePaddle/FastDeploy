"""
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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.triton_ops.indexer_update_attn_mask_offsets import (
    update_indexer_attn_mask_offsets,
)


def ref_update_attn_mask_offsets(seq_lens_this_time, seq_lens_encoder, cu_seqlens_k):
    """Python reference implementation aligned with the original loop semantics in deepseek_v3.py.
    Returns attn_mask_offsets: [num_tokens * 2], even positions = start, odd positions = end

    Note: cu_seqlens_k in Indexer is the cumulative length on the Q side (cumsum of seq_lens_this_time),
    so num_tokens should be sum(seq_lens_this_time) rather than cu_seqlens_k[-1].
    """
    num_tokens = int(sum(int(s.numpy()) for s in seq_lens_this_time))
    result = np.zeros(num_tokens * 2, dtype=np.int32)

    bsz = len(seq_lens_this_time)
    for i in range(bsz):
        if int(seq_lens_encoder[i].numpy()) > 0:
            token_start_k = int(cu_seqlens_k[i].numpy())
            token_end_k = int(cu_seqlens_k[i + 1].numpy())
            for t in range(token_start_k, token_end_k):
                result[t * 2] = token_start_k  # start: k start offset of current batch
                result[t * 2 + 1] = t + 1  # end: global token index + 1
    return result


def make_cu_seqlens(seq_lens):
    """Build cu_seqlens (prefix sum) from a list of sequence lengths."""
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return paddle.to_tensor(cu, dtype=paddle.int32)


class TestIndexerUpdateAttnMaskOffsets(unittest.TestCase):

    def _run_and_compare(self, seq_lens_this_time_list, seq_lens_encoder_list, k_lens_list):
        """Build inputs, run the Triton kernel and reference implementation, then compare results."""
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_list, dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder_list, dtype=paddle.int32)
        cu_seqlens_k = make_cu_seqlens(k_lens_list)

        num_tokens = int(sum(seq_lens_this_time_list))
        ids_remove_padding = paddle.zeros([num_tokens], dtype=paddle.int32)

        triton_out = update_indexer_attn_mask_offsets(
            ids_remove_padding,
            seq_lens_this_time,
            seq_lens_encoder,
            cu_seqlens_k,
        ).numpy()

        ref_out = ref_update_attn_mask_offsets(seq_lens_this_time, seq_lens_encoder, cu_seqlens_k)

        np.testing.assert_array_equal(
            triton_out,
            ref_out,
            err_msg=f"Mismatch!\ntriton: {triton_out}\nref:    {ref_out}",
        )

    # ------------------------------------------------------------------
    # Basic functional test cases
    # ------------------------------------------------------------------

    def test_single_prefill_seq(self):
        """Single prefill sequence with 4 tokens."""
        self._run_and_compare(
            seq_lens_this_time_list=[4],
            seq_lens_encoder_list=[4],
            k_lens_list=[4],
        )

    def test_single_token_prefill(self):
        """Edge case: prefill sequence with only 1 token."""
        self._run_and_compare(
            seq_lens_this_time_list=[1],
            seq_lens_encoder_list=[1],
            k_lens_list=[1],
        )

    def test_single_decode_seq(self):
        """Single decode sequence; all outputs should be 0 (decode path does not write offsets)."""
        self._run_and_compare(
            seq_lens_this_time_list=[1],
            seq_lens_encoder_list=[0],
            k_lens_list=[1],
        )

    # ------------------------------------------------------------------
    # Multi-batch test cases
    # ------------------------------------------------------------------

    def test_multi_prefill_batch(self):
        """Multiple prefill sequences with different lengths."""
        self._run_and_compare(
            seq_lens_this_time_list=[3, 5, 2],
            seq_lens_encoder_list=[3, 5, 2],
            k_lens_list=[3, 5, 2],
        )

    def test_all_decode_batch(self):
        """All-decode batch; all even/odd positions should be 0.
        For decode requests, k_len = seq_lens_this_time (Q-side length), not KV cache history length.
        """
        self._run_and_compare(
            seq_lens_this_time_list=[1, 1, 1],
            seq_lens_encoder_list=[0, 0, 0],
            k_lens_list=[1, 1, 1],
        )

    def test_mixed_prefill_decode_batch(self):
        """Mixed batch: index 0 is prefill, index 1 is decode, index 2 is prefill.
        For decode requests, k_len = seq_lens_this_time = 1 (Q-side length).
        """
        self._run_and_compare(
            seq_lens_this_time_list=[4, 1, 3],
            seq_lens_encoder_list=[4, 0, 3],
            k_lens_list=[4, 1, 3],
        )

    # ------------------------------------------------------------------
    # Numerical correctness verification
    # ------------------------------------------------------------------

    def test_prefill_ks_ke_values(self):
        """Exactly verify the start/end values for prefill tokens.

        Scenario: bsz=1, seq=[0,1,2], k_start=0
        Expected:
            token 0: start=0, end=1
            token 1: start=0, end=2
            token 2: start=0, end=3
        """
        seq_lens_this_time = paddle.to_tensor([3], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([3], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 3], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([3], dtype=paddle.int32)

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # [ks0, ke0, ks1, ke1, ks2, ke2]
        expected = np.array([0, 1, 0, 2, 0, 3], dtype=np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_prefill_with_nonzero_k_start(self):
        """Verify that start offset propagates correctly when k_start is non-zero.

        Scenario: bsz=2, index 0 is decode (q_len=1), index 1 is prefill (q_len=3)
        cu_seqlens_k = cumsum(seq_lens_this_time) = [0, 1, 4]
        k_start=1 for index 1, global token indices = 1, 2, 3
        Expected:
            token 0 (decode): start=0, end=0
            token 1: start=1, end=2
            token 2: start=1, end=3
            token 3: start=1, end=4
        """
        seq_lens_this_time = paddle.to_tensor([1, 3], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([0, 3], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 1, 4], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([4], dtype=paddle.int32)  # 1+3 tokens

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # token 0 (decode): [0, 0]
        # token 1,2,3 (prefill): start=1; end=2,3,4
        expected = np.array([0, 0, 1, 2, 1, 3, 1, 4], dtype=np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_decode_tokens_remain_zero(self):
        """Decode token positions must remain 0 and must not be overwritten by the kernel."""
        seq_lens_this_time = paddle.to_tensor([1, 4], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([0, 4], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 1, 5], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([5], dtype=paddle.int32)  # 1+4 tokens

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # Decode request at index 0: start/end of token 0 should both be 0
        self.assertEqual(out[0], 0, "decode token start should be 0")
        self.assertEqual(out[1], 0, "decode token end should be 0")

    # ------------------------------------------------------------------
    # Large sequence stress test cases
    # ------------------------------------------------------------------

    def test_large_seq_len(self):
        """Long sequence to verify correctness of the BLOCK_M tiled loop logic."""
        seq_len = 512
        self._run_and_compare(
            seq_lens_this_time_list=[seq_len],
            seq_lens_encoder_list=[seq_len],
            k_lens_list=[seq_len],
        )

    def test_large_batch(self):
        """Large batch to verify correctness of multi-program parallel results."""
        bsz = 32
        seq_lens = [8] * bsz
        self._run_and_compare(
            seq_lens_this_time_list=seq_lens,
            seq_lens_encoder_list=seq_lens,
            k_lens_list=seq_lens,
        )

    def test_large_mixed_batch(self):
        """Large-scale mixed batch with alternating prefill/decode.
        For decode requests, k_len = seq_lens_this_time = 1.
        """
        bsz = 20
        seq_lens_this_time = [6 if i % 2 == 0 else 1 for i in range(bsz)]
        seq_lens_encoder = [6 if i % 2 == 0 else 0 for i in range(bsz)]
        k_lens = seq_lens_this_time  # cu_seqlens_k = cumsum(seq_lens_this_time)
        self._run_and_compare(seq_lens_this_time, seq_lens_encoder, k_lens)


if __name__ == "__main__":
    unittest.main()
