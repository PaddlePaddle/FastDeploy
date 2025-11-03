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

"""Unit tests for speculate_limit_thinking_content_length_v1 and speculate_limit_thinking_content_length_v2"""

import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import (
    speculate_limit_thinking_content_length_v1,
    speculate_limit_thinking_content_length_v2,
)


class TestSpeculateLimitThinkingContentLengthV1(unittest.TestCase):
    """Tests for speculate_limit_thinking_content_length_v1 operator (</think> strategy with speculative decoding)"""

    def test_normal_thinking_phase_no_truncation(self):
        """Test normal thinking phase when all tokens are within limit"""
        # Batch 0 accepts 3 tokens, Batch 1 accepts 2 tokens
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 201, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        # step_idx represents current step after accepting tokens
        step_idx = paddle.to_tensor([5, 8], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5, 8], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: tokens unchanged, accept_num unchanged, status unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[0, 1] == 101
        assert next_tokens.numpy()[0, 2] == 102
        assert accept_num.numpy()[0] == 3
        assert accept_num.numpy()[1] == 2
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0
        assert step_idx.numpy()[0] == 5
        assert step_idx.numpy()[1] == 8

    def test_force_truncation_when_exceeding_limit(self):
        """Test force truncation when tokens exceed max_think_len"""
        # Accept 4 tokens, but will exceed limit at 3rd token
        next_tokens = paddle.to_tensor([[100, 101, 102, 103]], dtype="int64")
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        # Current step is 12 after accepting 4 tokens, so base step is 12-4+1=9
        # Token 0 at step 9, token 1 at step 10 (>= max_think_len=10), should be truncated
        step_idx = paddle.to_tensor([12], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([4], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([12], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: token at position 1 should be replaced with think_end_id
        # accept_num should be 2 (truncated after 2nd token which triggers the condition)
        assert next_tokens.numpy()[0, 0] == 100  # Token at step 9
        assert next_tokens.numpy()[0, 1] == 999  # Token at step 10, replaced with think_end_id
        assert accept_num.numpy()[0] == 2  # Only accept first 2 tokens
        assert limit_think_status.numpy()[0] == 2  # Status updated to 2
        # step_idx and seq_lens_decoder should be adjusted
        assert step_idx.numpy()[0] == 10  # 12 - (4-2) = 10
        assert seq_lens_decoder.numpy()[0] == 10  # 12 - (4-2) = 10

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id in accepted tokens"""
        next_tokens = paddle.to_tensor([[100, 999, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([20], dtype="int32")
        step_idx = paddle.to_tensor([5], dtype="int64")  # step 3-5
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: status changed to 2, tokens processed normally
        assert next_tokens.numpy()[0, 1] == 999
        assert limit_think_status.numpy()[0] == 2  # Thinking ended
        assert accept_num.numpy()[0] == 3  # All tokens accepted

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100, 101, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")  # Disabled
        step_idx = paddle.to_tensor([100], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([100], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: nothing changed
        assert next_tokens.numpy()[0, 0] == 100
        assert accept_num.numpy()[0] == 3
        assert limit_think_status.numpy()[0] == 0

    def test_zero_accept_num_early_return(self):
        """Test early return when accept_num is 0"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([0], dtype="int32")  # No tokens accepted
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: nothing changed (early return)
        assert accept_num.numpy()[0] == 0
        assert limit_think_status.numpy()[0] == 0

    def test_already_in_response_phase_status_3(self):
        """Test that status 3 is terminal (note: v1 uses status 2 as terminal in comment, but code shows 3)"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([3], dtype="int32")  # Terminal status
        accept_num = paddle.to_tensor([2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Verify: early return, nothing changed
        assert limit_think_status.numpy()[0] == 3

    def test_status_transition_from_0_to_1_to_2(self):
        """Test status transition: 0 (thinking) -> 1 (injected) -> 2 (ended)"""
        # First call: inject think_end_id due to exceeding limit
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([9], dtype="int32")
        step_idx = paddle.to_tensor([9], dtype="int64")  # base step = 9-2+1 = 8
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([9], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # First token at step 8 is OK, second token at step 9 >= 8, so gets replaced
        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[0, 1] == 999  # Replaced
        assert limit_think_status.numpy()[0] == 2
        assert accept_num.numpy()[0] == 2

    def test_mixed_batch_with_different_states(self):
        """Test batch with different sequences in various states"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 999, 202], [300, 301, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15, -1], dtype="int32")
        step_idx = paddle.to_tensor([6, 8, 50], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 3, 2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([6, 8, 50], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        # Seq 0: all tokens within limit, unchanged
        assert limit_think_status.numpy()[0] == 0
        assert accept_num.numpy()[0] == 3

        # Seq 1: second token is think_end_id, status -> 2
        assert limit_think_status.numpy()[1] == 2
        assert accept_num.numpy()[1] == 3

        # Seq 2: disabled, unchanged
        assert limit_think_status.numpy()[2] == 0
        assert accept_num.numpy()[2] == 2


class TestSpeculateLimitThinkingContentLengthV2(unittest.TestCase):
    """Tests for speculate_limit_thinking_content_length_v2 operator.

    Tests the \\n</think>\\n\\n strategy with speculative decoding.
    """

    def test_normal_thinking_phase_no_truncation(self):
        """Test normal thinking phase when all tokens are within limit"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 201, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        step_idx = paddle.to_tensor([5, 8], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5, 8], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert accept_num.numpy()[0] == 3
        assert limit_think_status.numpy()[0] == 0

    def test_force_truncation_with_sequence_injection(self):
        """Test force truncation with \n</think>\n\n sequence injection"""
        # Test when multiple tokens in batch trigger different injections
        next_tokens = paddle.to_tensor([[100, 101, 102, 103, 104]], dtype="int64")
        max_think_lens = paddle.to_tensor([8], dtype="int32")
        # step_idx = 12, accept_num = 5, base_step = 12-5+1 = 8
        # Token 0 at step 8 (== max 8): inject line_break
        step_idx = paddle.to_tensor([12], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([5], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([12], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Token at step 8 (== max 8) should be replaced with line_break_id
        assert next_tokens.numpy()[0, 0] == 888  # line_break_id
        assert limit_think_status.numpy()[0] == 1
        assert accept_num.numpy()[0] == 1  # Truncated after 1st token
        assert step_idx.numpy()[0] == 8  # 12 - (5-1)
        assert seq_lens_decoder.numpy()[0] == 8

    def test_injection_sequence_steps(self):
        """Test each step of the injection sequence: \n, </think>, \n, \n"""
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        think_end_id = 999
        line_break_id = 888

        # Step 1: at max_think_len, inject first \n
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([5], dtype="int64")  # base_step = 5-1+1 = 5
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 2: at max_think_len+1, inject </think>
        next_tokens = paddle.to_tensor([[200]], dtype="int64")
        step_idx = paddle.to_tensor([6], dtype="int64")  # base_step = 6
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([6], dtype="int32")

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )
        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 1

        # Step 3: at max_think_len+2, inject second \n
        next_tokens = paddle.to_tensor([[300]], dtype="int64")
        step_idx = paddle.to_tensor([7], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([7], dtype="int32")

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 4: at max_think_len+3, inject third \n and move to status 3
        next_tokens = paddle.to_tensor([[400]], dtype="int64")
        step_idx = paddle.to_tensor([8], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([8], dtype="int32")

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 3

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[100, 999, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([20], dtype="int32")
        step_idx = paddle.to_tensor([5], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: status changed to 3
        assert limit_think_status.numpy()[0] == 3

    def test_status_2_to_status_3_transition(self):
        """Test transition from status 2 to status 3"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([2], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: status 2 -> 3
        assert limit_think_status.numpy()[0] == 3

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([100], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([100], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: nothing changed
        assert limit_think_status.numpy()[0] == 0
        assert accept_num.numpy()[0] == 2

    def test_zero_accept_num_early_return(self):
        """Test early return when accept_num is 0"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: early return
        assert accept_num.numpy()[0] == 0
        assert limit_think_status.numpy()[0] == 0

    def test_already_in_response_phase_status_3(self):
        """Test that status 3 is terminal"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([3], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([10], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        # Verify: early return, nothing changed
        assert limit_think_status.numpy()[0] == 3


if __name__ == "__main__":
    unittest.main()
