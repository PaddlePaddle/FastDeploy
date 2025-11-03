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

"""Unit tests for limit_thinking_content_length_v1 and limit_thinking_content_length_v2"""

import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import (
    limit_thinking_content_length_v1,
    limit_thinking_content_length_v2,
)


class TestLimitThinkingContentLengthV1(unittest.TestCase):
    """Tests for limit_thinking_content_length_v1 operator (</think> strategy)"""

    def test_normal_thinking_phase_no_limit_reached(self):
        """Test normal thinking phase when step < max_think_len"""
        next_tokens = paddle.to_tensor([[100], [200]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        step_idx = paddle.to_tensor([[5], [8]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: tokens unchanged, status unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[1, 0] == 200
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0

    def test_force_truncation_when_max_think_len_exceeded(self):
        """Test force truncation when step >= max_think_len"""
        next_tokens = paddle.to_tensor([[100], [200]], dtype="int64")
        max_think_lens = paddle.to_tensor([5, 8], dtype="int32")
        step_idx = paddle.to_tensor([[5], [10]], dtype="int64")  # Both exceed or equal limit
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: tokens replaced with think_end_id, status changed to 2
        assert next_tokens.numpy()[0, 0] == 999  # Replaced
        assert next_tokens.numpy()[1, 0] == 999  # Replaced
        assert limit_think_status.numpy()[0] == 2  # Status updated
        assert limit_think_status.numpy()[1] == 2  # Status updated

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")  # Model generated think_end_id
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        step_idx = paddle.to_tensor([[3]], dtype="int64")  # Still within limit
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: token unchanged (already think_end_id), status changed to 2
        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 2  # Move to response phase

    def test_status_1_to_status_2_transition(self):
        """Test transition from status 1 (injected) to status 2 (confirmed)"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")  # think_end_id from previous injection
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([[6]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")  # Status is 1
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: status changed to 2
        assert limit_think_status.numpy()[0] == 2

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")  # Disabled
        step_idx = paddle.to_tensor([[100]], dtype="int64")  # Would exceed limit if enabled
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: nothing changed
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

    def test_already_in_response_phase_status_2(self):
        """Test that status 2 (response phase) is terminal"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([[10]], dtype="int64")
        limit_think_status = paddle.to_tensor([2], dtype="int32")  # Already in response phase
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify: nothing changed
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 2

    def test_mixed_batch(self):
        """Test batch with different sequences in different states"""
        next_tokens = paddle.to_tensor([[100], [200], [999], [300]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 5, 8, -1], dtype="int32")
        step_idx = paddle.to_tensor([[3], [5], [4], [100]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        # Run operator
        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        # Verify each sequence
        # Seq 0: step 3 < max 10, status 0, token unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

        # Seq 1: step 5 >= max 5, force inject think_end_id, status -> 2
        assert next_tokens.numpy()[1, 0] == 999
        assert limit_think_status.numpy()[1] == 2

        # Seq 2: step 4 < max 8, but token is think_end_id, status -> 2
        assert next_tokens.numpy()[2, 0] == 999
        assert limit_think_status.numpy()[2] == 2

        # Seq 3: disabled (max -1), unchanged
        assert next_tokens.numpy()[3, 0] == 300
        assert limit_think_status.numpy()[3] == 0


class TestLimitThinkingContentLengthV2(unittest.TestCase):
    """Tests for limit_thinking_content_length_v2 operator (\n</think>\n\n strategy)"""

    def test_normal_thinking_phase_no_limit_reached(self):
        """Test normal thinking phase when step < max_think_len"""
        next_tokens = paddle.to_tensor([[100], [200]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        step_idx = paddle.to_tensor([[5], [8]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Verify: tokens unchanged, status unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[1, 0] == 200
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0

    def test_force_truncation_sequence_injection(self):
        """Test force truncation with \n</think>\n\n sequence injection"""
        # Test step == max_think_len (inject first \n)
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([[5]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888  # line_break_id
        assert limit_think_status.numpy()[0] == 1

        # Test step == max_think_len + 1 (inject </think>)
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[6]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 999  # think_end_id
        assert limit_think_status.numpy()[0] == 1

        # Test step == max_think_len + 2 (inject second \n)
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[7]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888  # line_break_id
        assert limit_think_status.numpy()[0] == 1

        # Test step == max_think_len + 3 (inject third \n and finish)
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[8]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888  # line_break_id
        assert limit_think_status.numpy()[0] == 3  # Move to status 3

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        step_idx = paddle.to_tensor([[3]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Verify: status changed to 3 (response phase)
        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 3

    def test_status_2_to_status_3_transition(self):
        """Test transition from status 2 (replacement done) to status 3 (thinking ended)"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([[9]], dtype="int64")
        limit_think_status = paddle.to_tensor([2], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Verify: status changed to 3
        assert limit_think_status.numpy()[0] == 3

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([[100]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Verify: nothing changed
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

    def test_already_in_response_phase_status_3(self):
        """Test that status 3 (response phase) is terminal"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([[10]], dtype="int64")
        limit_think_status = paddle.to_tensor([3], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Verify: nothing changed
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 3

    def test_mixed_batch_various_states(self):
        """Test batch with sequences in different states"""
        next_tokens = paddle.to_tensor([[100], [200], [999], [300], [400]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 5, 8, -1, 6], dtype="int32")
        step_idx = paddle.to_tensor([[3], [5], [4], [100], [9]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0, 0, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False, False, False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        # Run operator
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        # Seq 0: step 3 < max 10, status 0, unchanged
        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

        # Seq 1: step 5 == max 5, inject line_break_id, status -> 1
        assert next_tokens.numpy()[1, 0] == 888
        assert limit_think_status.numpy()[1] == 1

        # Seq 2: token is think_end_id, status 0 -> 3
        assert next_tokens.numpy()[2, 0] == 999
        assert limit_think_status.numpy()[2] == 3

        # Seq 3: disabled, unchanged
        assert next_tokens.numpy()[3, 0] == 300
        assert limit_think_status.numpy()[3] == 0

        # Seq 4: status 2 (replacement done), transition to 3
        assert limit_think_status.numpy()[4] == 3


if __name__ == "__main__":
    unittest.main()
