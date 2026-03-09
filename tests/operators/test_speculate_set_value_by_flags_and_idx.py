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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_set_value_by_flags_and_idx

paddle.seed(42)
np.random.seed(42)


def speculate_set_value_by_flags_and_idx_ref(
    token_ids_all,
    prompt_lens,
    accept_tokens,
    accept_num,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
):
    """
    NumPy reference implementation of speculate_set_value_by_flags_and_idx.

    This op is used in speculative decoding to write accepted draft tokens
    back into the token_ids_all buffer. It modifies three tensors inplace:
      - token_ids_all: accepted tokens are written at the position
        prompt_lens[i] + step_idx[i] (counting backwards for multiple tokens)
      - accept_num: zeroed for stopped sequences
      - seq_lens_decoder: zeroed for stopped sequences
    """
    result_token_ids = token_ids_all.copy()
    result_accept_num = accept_num.copy()
    result_seq_lens_decoder = seq_lens_decoder.copy()
    bs = seq_lens_this_time.shape[0]

    for i in range(bs):
        if stop_flags[i]:
            # Stopped sequences: zero out accept_num and seq_lens_decoder
            result_accept_num[i] = 0
            result_seq_lens_decoder[i] = 0
        else:
            seq_len_dec = seq_lens_decoder[i]
            seq_len_enc = seq_lens_encoder[i]
            # Skip if both encoder and decoder lengths are zero (already stopped)
            if seq_len_dec == 0 and seq_len_enc == 0:
                continue
            if step_idx[i] > 0:
                prompt_len = int(prompt_lens[i, 0] if prompt_lens.ndim == 2 else prompt_lens[i])
                for j in range(int(accept_num[i])):
                    result_token_ids[i, prompt_len + int(step_idx[i]) - j] = accept_tokens[
                        i, int(accept_num[i]) - 1 - j
                    ]

    return result_token_ids, result_accept_num, result_seq_lens_decoder


def _build_inputs(
    batch_size=16,
    max_model_len=128,
    max_draft_tokens=4,
    stop_ratio=0.2,
    zero_len_ratio=0.1,
):
    """Create standardized random test inputs."""
    token_ids_all = np.full((batch_size, max_model_len), -1, dtype="int64")
    prompt_lens = np.random.randint(0, max_model_len // 4, size=(batch_size, 1)).astype("int64")
    accept_tokens = np.random.randint(100, 50000, size=(batch_size, max_draft_tokens)).astype("int64")
    accept_num = np.random.randint(0, max_draft_tokens + 1, size=(batch_size,)).astype("int32")
    stop_flags = np.random.choice([True, False], size=(batch_size,), p=[stop_ratio, 1 - stop_ratio])
    seq_lens_this_time = np.ones((batch_size,), dtype="int32")
    seq_lens_encoder = np.random.randint(0, 5, size=(batch_size,)).astype("int32")
    seq_lens_decoder = np.random.randint(0, 5, size=(batch_size,)).astype("int32")

    # Ensure step_idx is valid: prompt_lens[i] + step_idx[i] + accept_num[i] < max_model_len
    step_idx = np.zeros((batch_size,), dtype="int64")
    for i in range(batch_size):
        prompt_len = int(prompt_lens[i, 0])
        max_step = max_model_len - prompt_len - max_draft_tokens - 1
        if max_step > max_draft_tokens:
            step_idx[i] = np.random.randint(max_draft_tokens, max_step)
        else:
            step_idx[i] = max_draft_tokens

    # Inject some zero-length entries
    zero_count = int(batch_size * zero_len_ratio)
    zero_indices = np.random.choice(batch_size, size=zero_count, replace=False)
    seq_lens_encoder[zero_indices] = 0
    seq_lens_decoder[zero_indices] = 0

    return (
        token_ids_all,
        prompt_lens,
        accept_tokens,
        accept_num,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
    )


class TestSpeculateSetValueByFlagsAndIdx(unittest.TestCase):
    """Unit tests for the speculate_set_value_by_flags_and_idx custom op."""

    def setUp(self):
        paddle.set_device("gpu")
        paddle.seed(42)
        np.random.seed(42)

    # -- Numerical Correctness (Category A) --

    def test_correctness_random(self):
        """Compare GPU op against NumPy reference with random inputs."""
        inputs_np = _build_inputs(batch_size=32, max_model_len=256, max_draft_tokens=4)
        (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        ) = inputs_np

        # NumPy reference
        ref_token_ids, ref_accept_num, ref_seq_lens_dec = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        # GPU op (inplace — modifies tensors directly)
        token_ids_all = paddle.to_tensor(token_ids_all_np)
        prompt_lens = paddle.to_tensor(prompt_lens_np)
        accept_tokens = paddle.to_tensor(accept_tokens_np)
        accept_num = paddle.to_tensor(accept_num_np)
        stop_flags = paddle.to_tensor(stop_flags_np)
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_np)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder_np)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder_np)
        step_idx = paddle.to_tensor(step_idx_np)

        speculate_set_value_by_flags_and_idx(
            token_ids_all,
            prompt_lens,
            accept_tokens,
            accept_num,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
        )

        np.testing.assert_array_equal(
            token_ids_all.numpy(),
            ref_token_ids,
            err_msg="token_ids_all mismatch after speculate_set_value_by_flags_and_idx",
        )
        np.testing.assert_array_equal(
            accept_num.numpy(),
            ref_accept_num,
            err_msg="accept_num mismatch (should be zeroed for stopped sequences)",
        )
        np.testing.assert_array_equal(
            seq_lens_decoder.numpy(),
            ref_seq_lens_dec,
            err_msg="seq_lens_decoder mismatch (should be zeroed for stopped sequences)",
        )

    def test_correctness_large_batch(self):
        """Correctness with a larger batch to stress multi-thread paths."""
        inputs_np = _build_inputs(batch_size=256, max_model_len=8192, max_draft_tokens=6)
        ref_token_ids, ref_accept_num, ref_seq_lens_dec = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        np.testing.assert_array_equal(tensors[0].numpy(), ref_token_ids)
        np.testing.assert_array_equal(tensors[3].numpy(), ref_accept_num)
        np.testing.assert_array_equal(tensors[7].numpy(), ref_seq_lens_dec)

    # -- Edge Cases (Category D) --

    def test_all_stopped(self):
        """All sequences have stop_flags=True."""
        bs, max_len, max_draft = 4, 32, 3
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.random.randint(1, 100, size=(bs, max_draft)).astype("int64")
        accept_num_np = np.array([2, 1, 3, 2], dtype="int32")
        stop_flags_np = np.array([True, True, True, True])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.ones(bs, dtype="int32")
        seq_lens_decoder_np = np.array([2, 3, 1, 4], dtype="int32")
        step_idx_np = np.array([5, 6, 7, 8], dtype="int64")

        inputs_np = (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        )

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        # token_ids_all should be unchanged (all stopped, no writes)
        np.testing.assert_array_equal(tensors[0].numpy(), token_ids_all_np)
        # accept_num should all be zero
        np.testing.assert_array_equal(tensors[3].numpy(), np.zeros(bs, dtype="int32"))
        # seq_lens_decoder should all be zero
        np.testing.assert_array_equal(tensors[7].numpy(), np.zeros(bs, dtype="int32"))

    def test_none_stopped(self):
        """No sequences stopped, all should write tokens."""
        bs, max_len = 4, 64
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
                [100, 110, 120],
            ],
            dtype="int64",
        )
        accept_num_np = np.array([2, 3, 1, 2], dtype="int32")
        stop_flags_np = np.array([False, False, False, False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.zeros(bs, dtype="int32")
        seq_lens_decoder_np = np.array([2, 3, 1, 2], dtype="int32")
        step_idx_np = np.array([10, 15, 20, 25], dtype="int64")

        inputs_np = (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        )
        ref_token_ids, ref_accept_num, ref_seq_lens_dec = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        np.testing.assert_array_equal(tensors[0].numpy(), ref_token_ids)
        # accept_num unchanged (no stopped sequences)
        np.testing.assert_array_equal(tensors[3].numpy(), accept_num_np)

    def test_both_lens_zero_skip(self):
        """When seq_lens_encoder=0 AND seq_lens_decoder=0, the op should skip."""
        bs, max_len = 2, 32
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.array([[99, 88], [77, 66]], dtype="int64")
        accept_num_np = np.array([2, 1], dtype="int32")
        stop_flags_np = np.array([False, False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.zeros(bs, dtype="int32")
        seq_lens_decoder_np = np.zeros(bs, dtype="int32")
        step_idx_np = np.array([5, 10], dtype="int64")

        tensors = [
            paddle.to_tensor(x)
            for x in (
                token_ids_all_np,
                prompt_lens_np,
                accept_tokens_np,
                accept_num_np,
                stop_flags_np,
                seq_lens_this_time_np,
                seq_lens_encoder_np,
                seq_lens_decoder_np,
                step_idx_np,
            )
        ]
        speculate_set_value_by_flags_and_idx(*tensors)

        # token_ids_all should be unchanged
        np.testing.assert_array_equal(tensors[0].numpy(), token_ids_all_np)

    def test_step_idx_zero(self):
        """When step_idx=0, the kernel condition step_idx > 0 is false, no write."""
        bs, max_len = 2, 32
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.array([[42, 43], [44, 45]], dtype="int64")
        accept_num_np = np.array([1, 2], dtype="int32")
        stop_flags_np = np.array([False, False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.array([1, 1], dtype="int32")
        seq_lens_decoder_np = np.array([1, 1], dtype="int32")
        step_idx_np = np.array([0, 0], dtype="int64")

        tensors = [
            paddle.to_tensor(x)
            for x in (
                token_ids_all_np,
                prompt_lens_np,
                accept_tokens_np,
                accept_num_np,
                stop_flags_np,
                seq_lens_this_time_np,
                seq_lens_encoder_np,
                seq_lens_decoder_np,
                step_idx_np,
            )
        ]
        speculate_set_value_by_flags_and_idx(*tensors)

        # token_ids_all should be unchanged (step_idx not > 0)
        np.testing.assert_array_equal(tensors[0].numpy(), token_ids_all_np)

    def test_accept_num_zero(self):
        """When accept_num=0, no tokens are written even with valid step_idx."""
        bs, max_len, max_draft = 2, 32, 3
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.random.randint(1, 100, size=(bs, max_draft)).astype("int64")
        accept_num_np = np.array([0, 0], dtype="int32")
        stop_flags_np = np.array([False, False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.array([1, 1], dtype="int32")
        seq_lens_decoder_np = np.array([2, 2], dtype="int32")
        step_idx_np = np.array([10, 15], dtype="int64")

        tensors = [
            paddle.to_tensor(x)
            for x in (
                token_ids_all_np,
                prompt_lens_np,
                accept_tokens_np,
                accept_num_np,
                stop_flags_np,
                seq_lens_this_time_np,
                seq_lens_encoder_np,
                seq_lens_decoder_np,
                step_idx_np,
            )
        ]
        speculate_set_value_by_flags_and_idx(*tensors)

        # token_ids_all should be unchanged (zero tokens to accept)
        np.testing.assert_array_equal(tensors[0].numpy(), token_ids_all_np)

    def test_with_prompt_offset(self):
        """Verify correct offset when prompt_lens > 0."""
        bs, max_len = 2, 64
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.array([[10], [20]], dtype="int64")
        accept_tokens_np = np.array([[500, 600, 700], [800, 900, 1000]], dtype="int64")
        accept_num_np = np.array([2, 3], dtype="int32")
        stop_flags_np = np.array([False, False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.zeros(bs, dtype="int32")
        seq_lens_decoder_np = np.array([2, 3], dtype="int32")
        step_idx_np = np.array([5, 10], dtype="int64")

        inputs_np = (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        )
        ref_token_ids, ref_accept_num, ref_seq_lens_dec = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        np.testing.assert_array_equal(tensors[0].numpy(), ref_token_ids)

        # Verify that tokens were written at prompt_lens + step_idx offsets
        # Batch 0: prompt_len=10, step_idx=5, accept_num=2
        #   writes at positions 15 and 14: tokens 600, 500
        self.assertEqual(ref_token_ids[0, 15], 600)
        self.assertEqual(ref_token_ids[0, 14], 500)
        # Batch 1: prompt_len=20, step_idx=10, accept_num=3
        #   writes at positions 30, 29, 28: tokens 1000, 900, 800
        self.assertEqual(ref_token_ids[1, 30], 1000)
        self.assertEqual(ref_token_ids[1, 29], 900)
        self.assertEqual(ref_token_ids[1, 28], 800)

    def test_mixed_stopped_and_active(self):
        """Mix of stopped and active sequences in one batch."""
        bs, max_len = 4, 64
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.zeros((bs, 1), dtype="int64")
        accept_tokens_np = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype="int64")
        accept_num_np = np.array([1, 2, 1, 2], dtype="int32")
        stop_flags_np = np.array([False, True, False, True])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.array([1, 1, 0, 1], dtype="int32")
        seq_lens_decoder_np = np.array([2, 3, 1, 4], dtype="int32")
        step_idx_np = np.array([5, 6, 7, 8], dtype="int64")

        inputs_np = (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        )
        ref_token_ids, ref_accept_num, ref_seq_lens_dec = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        np.testing.assert_array_equal(tensors[0].numpy(), ref_token_ids)
        np.testing.assert_array_equal(tensors[3].numpy(), ref_accept_num)
        np.testing.assert_array_equal(tensors[7].numpy(), ref_seq_lens_dec)

        # Verify stopped sequences had accept_num and seq_lens_decoder zeroed
        self.assertEqual(tensors[3].numpy()[1], 0)  # batch 1 stopped
        self.assertEqual(tensors[3].numpy()[3], 0)  # batch 3 stopped
        self.assertEqual(tensors[7].numpy()[1], 0)
        self.assertEqual(tensors[7].numpy()[3], 0)

    # -- Determinism (Category E) --

    def test_determinism(self):
        """Same inputs produce identical outputs across two runs."""
        np.random.seed(123)
        inputs_np = _build_inputs(batch_size=16, max_model_len=128, max_draft_tokens=4)

        # Run 1
        tensors_1 = [paddle.to_tensor(x.copy()) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors_1)

        # Run 2
        tensors_2 = [paddle.to_tensor(x.copy()) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors_2)

        np.testing.assert_array_equal(tensors_1[0].numpy(), tensors_2[0].numpy())
        np.testing.assert_array_equal(tensors_1[3].numpy(), tensors_2[3].numpy())
        np.testing.assert_array_equal(tensors_1[7].numpy(), tensors_2[7].numpy())

    # -- Single Element (Category D) --

    def test_batch_size_one(self):
        """Op handles batch_size=1 correctly."""
        bs, max_len = 1, 32
        token_ids_all_np = np.full((bs, max_len), -1, dtype="int64")
        prompt_lens_np = np.array([[5]], dtype="int64")
        accept_tokens_np = np.array([[111, 222, 333]], dtype="int64")
        accept_num_np = np.array([3], dtype="int32")
        stop_flags_np = np.array([False])
        seq_lens_this_time_np = np.ones(bs, dtype="int32")
        seq_lens_encoder_np = np.array([0], dtype="int32")
        seq_lens_decoder_np = np.array([3], dtype="int32")
        step_idx_np = np.array([10], dtype="int64")

        inputs_np = (
            token_ids_all_np,
            prompt_lens_np,
            accept_tokens_np,
            accept_num_np,
            stop_flags_np,
            seq_lens_this_time_np,
            seq_lens_encoder_np,
            seq_lens_decoder_np,
            step_idx_np,
        )
        ref_token_ids, _, _ = speculate_set_value_by_flags_and_idx_ref(*inputs_np)

        tensors = [paddle.to_tensor(x) for x in inputs_np]
        speculate_set_value_by_flags_and_idx(*tensors)

        np.testing.assert_array_equal(tensors[0].numpy(), ref_token_ids)

        # prompt_len=5, step_idx=10, accept_num=3
        # Writes at positions: 15, 14, 13 -> tokens 333, 222, 111
        self.assertEqual(tensors[0].numpy()[0, 15], 333)
        self.assertEqual(tensors[0].numpy()[0, 14], 222)
        self.assertEqual(tensors[0].numpy()[0, 13], 111)


if __name__ == "__main__":
    unittest.main()
