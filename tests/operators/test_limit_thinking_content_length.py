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

"""Unit tests for limit_thinking_content_length and speculate_limit_thinking_content_length operators (v1/v2/v3)"""

import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import (
    limit_thinking_content_length_v1,
    limit_thinking_content_length_v2,
    limit_thinking_content_length_v3,
    speculate_limit_thinking_content_length_v1,
    speculate_limit_thinking_content_length_v2,
    speculate_limit_thinking_content_length_v3,
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

        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[1, 0] == 200
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0

    def test_force_truncation_when_max_think_len_exceeded(self):
        """Test force truncation when step >= max_think_len"""
        next_tokens = paddle.to_tensor([[100], [200]], dtype="int64")
        max_think_lens = paddle.to_tensor([5, 8], dtype="int32")
        step_idx = paddle.to_tensor([[5], [10]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        assert next_tokens.numpy()[0, 0] == 999
        assert next_tokens.numpy()[1, 0] == 999
        assert limit_think_status.numpy()[0] == 2
        assert limit_think_status.numpy()[1] == 2

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        step_idx = paddle.to_tensor([[3]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 2

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([[100]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

    def test_mixed_batch(self):
        """Test batch with different sequences in different states"""
        next_tokens = paddle.to_tensor([[100], [200], [999], [300]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 5, 8, -1], dtype="int32")
        step_idx = paddle.to_tensor([[3], [5], [4], [100]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v1(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, eos_token_ids, think_end_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0
        assert next_tokens.numpy()[1, 0] == 999
        assert limit_think_status.numpy()[1] == 2
        assert next_tokens.numpy()[2, 0] == 999
        assert limit_think_status.numpy()[2] == 2
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

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[1, 0] == 200
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0

    def test_force_truncation_sequence_injection(self):
        """Test force truncation with \n</think>\n\n sequence injection"""
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        think_end_id = 999
        line_break_id = 888
        stop_flags = paddle.to_tensor([False], dtype="bool")

        # Step 1: at max_think_len, inject first \n
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[5]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 2: at max_think_len+1, inject </think>
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[6]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 1

        # Step 3: at max_think_len+2, inject second \n
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[7]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 4: at max_think_len+3, inject third \n and finish
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([[8]], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 3

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        step_idx = paddle.to_tensor([[3]], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        assert next_tokens.numpy()[0, 0] == 999
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

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0

    def test_mixed_batch_various_states(self):
        """Test batch with sequences in different states"""
        next_tokens = paddle.to_tensor([[100], [200], [999], [300], [400]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 5, 8, -1, 6], dtype="int32")
        step_idx = paddle.to_tensor([[3], [5], [4], [100], [9]], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0, 0, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False, False, False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, stop_flags, think_end_id, line_break_id
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert limit_think_status.numpy()[0] == 0
        assert next_tokens.numpy()[1, 0] == 888
        assert limit_think_status.numpy()[1] == 1
        assert next_tokens.numpy()[2, 0] == 999
        assert limit_think_status.numpy()[2] == 3
        assert next_tokens.numpy()[3, 0] == 300
        assert limit_think_status.numpy()[3] == 0
        assert limit_think_status.numpy()[4] == 3


class TestLimitThinkingContentLengthV3(unittest.TestCase):
    """Tests for limit_thinking_content_length_v3 operator (inject sequence + reply limit strategy)"""

    def test_normal_thinking_phase_no_limit_reached(self):
        """Test normal thinking phase when step < max_think_len"""
        next_tokens = paddle.to_tensor([[100], [200]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1, -1], dtype="int32")
        step_idx = paddle.to_tensor([[5], [8]], dtype="int64")
        limit_status = paddle.to_tensor([0, 0], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[1, 0] == 200
        assert limit_status.numpy()[0] == 0
        assert limit_status.numpy()[1] == 0

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[999]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([[3]], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert next_tokens.numpy()[0, 0] == 999
        assert limit_status.numpy()[0] == 1  # done_status = 1 when inject_len == 0

    def test_force_truncation_with_inject_tokens(self):
        """Test force truncation with inject token sequence"""
        inject_token_ids = paddle.to_tensor([888, 999, 888, 888], dtype="int64")  # \n</think>\n\n
        think_end_id = 999

        # At max_think_len, status 0 -> 1, inject first token
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([[5]], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")

        limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert next_tokens.numpy()[0, 0] == 888  # inject_token_ids[0]
        assert limit_status.numpy()[0] == 2  # status advanced to 2

    def test_disabled_both_limits(self):
        """Test that both limits disabled (-1) causes early return"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([[100]], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert limit_status.numpy()[0] == 0


class TestSpeculateLimitThinkingContentLengthV1(unittest.TestCase):
    """Tests for speculate_limit_thinking_content_length_v1 operator (</think> strategy with speculative decoding)"""

    def test_normal_thinking_phase_no_truncation(self):
        """Test normal thinking phase when all tokens are within limit"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 201, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        step_idx = paddle.to_tensor([5, 8], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[0, 1] == 101
        assert next_tokens.numpy()[0, 2] == 102
        assert accept_num.numpy()[0] == 3
        assert accept_num.numpy()[1] == 2
        assert limit_think_status.numpy()[0] == 0
        assert limit_think_status.numpy()[1] == 0

    def test_force_truncation_when_exceeding_limit(self):
        """Test force truncation when tokens exceed max_think_len"""
        next_tokens = paddle.to_tensor([[100, 101, 102, 103]], dtype="int64")
        max_think_lens = paddle.to_tensor([10], dtype="int32")
        step_idx = paddle.to_tensor([12], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([4], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert next_tokens.numpy()[0, 1] == 999
        assert accept_num.numpy()[0] == 2
        assert limit_think_status.numpy()[0] == 2
        assert step_idx.numpy()[0] == 10

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id in accepted tokens"""
        next_tokens = paddle.to_tensor([[100, 999, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([20], dtype="int32")
        step_idx = paddle.to_tensor([5], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert next_tokens.numpy()[0, 1] == 999
        assert limit_think_status.numpy()[0] == 2
        assert accept_num.numpy()[0] == 3

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100, 101, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([100], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert accept_num.numpy()[0] == 3
        assert limit_think_status.numpy()[0] == 0

    def test_zero_accept_num_early_return(self):
        """Test early return when accept_num is 0"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert accept_num.numpy()[0] == 0
        assert limit_think_status.numpy()[0] == 0

    def test_mixed_batch_with_different_states(self):
        """Test batch with different sequences in various states"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 999, 202], [300, 301, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15, -1], dtype="int32")
        step_idx = paddle.to_tensor([6, 8, 50], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 3, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2], [2]], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v1(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            think_end_id,
        )

        assert limit_think_status.numpy()[0] == 0
        assert accept_num.numpy()[0] == 3
        assert limit_think_status.numpy()[1] == 2
        assert accept_num.numpy()[1] == 3
        assert limit_think_status.numpy()[2] == 0
        assert accept_num.numpy()[2] == 2


class TestSpeculateLimitThinkingContentLengthV2(unittest.TestCase):
    """Tests for speculate_limit_thinking_content_length_v2 operator (\n</think>\n\n strategy with speculative decoding)"""

    def test_normal_thinking_phase_no_truncation(self):
        """Test normal thinking phase when all tokens are within limit"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 201, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        step_idx = paddle.to_tensor([5, 8], dtype="int64")
        limit_think_status = paddle.to_tensor([0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert accept_num.numpy()[0] == 3
        assert limit_think_status.numpy()[0] == 0

    def test_force_truncation_with_sequence_injection(self):
        """Test force truncation with \n</think>\n\n sequence injection"""
        next_tokens = paddle.to_tensor([[100, 101, 102, 103, 104]], dtype="int64")
        max_think_lens = paddle.to_tensor([8], dtype="int32")
        step_idx = paddle.to_tensor([12], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([5], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1
        assert accept_num.numpy()[0] == 1
        assert step_idx.numpy()[0] == 8

    def test_injection_sequence_steps(self):
        """Test each step of the injection sequence: \n, </think>, \n, \n"""
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        think_end_id = 999
        line_break_id = 888

        # Step 1: at max_think_len, inject first \n
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        step_idx = paddle.to_tensor([5], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        speculate_limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, accept_num, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 2: at max_think_len+1, inject </think>
        next_tokens = paddle.to_tensor([[200]], dtype="int64")
        step_idx = paddle.to_tensor([6], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        speculate_limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, accept_num, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 999
        assert limit_think_status.numpy()[0] == 1

        # Step 3: at max_think_len+2, inject second \n
        next_tokens = paddle.to_tensor([[300]], dtype="int64")
        step_idx = paddle.to_tensor([7], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        speculate_limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, accept_num, stop_flags, think_end_id, line_break_id
        )
        assert next_tokens.numpy()[0, 0] == 888
        assert limit_think_status.numpy()[0] == 1

        # Step 4: at max_think_len+3, inject third \n and move to status 3
        next_tokens = paddle.to_tensor([[400]], dtype="int64")
        step_idx = paddle.to_tensor([8], dtype="int64")
        limit_think_status = paddle.to_tensor([1], dtype="int32")
        accept_num = paddle.to_tensor([1], dtype="int32")
        speculate_limit_thinking_content_length_v2(
            next_tokens, max_think_lens, step_idx, limit_think_status, accept_num, stop_flags, think_end_id, line_break_id
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
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        assert limit_think_status.numpy()[0] == 3

    def test_disabled_feature_negative_max_think_len(self):
        """Test that negative max_think_len disables the feature"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([100], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([2], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        assert limit_think_status.numpy()[0] == 0
        assert accept_num.numpy()[0] == 2

    def test_zero_accept_num_early_return(self):
        """Test early return when accept_num is 0"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_think_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        think_end_id = 999
        line_break_id = 888

        speculate_limit_thinking_content_length_v2(
            next_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            stop_flags,
            think_end_id,
            line_break_id,
        )

        assert accept_num.numpy()[0] == 0
        assert limit_think_status.numpy()[0] == 0


class TestSpeculateLimitThinkingContentLengthV3(unittest.TestCase):
    """Tests for speculate_limit_thinking_content_length_v3 operator (inject sequence + reply limit with speculative decoding)"""

    def test_normal_thinking_phase_no_truncation(self):
        """Test normal thinking phase when all tokens are within limit"""
        next_tokens = paddle.to_tensor([[100, 101, 102], [200, 201, 0]], dtype="int64")
        max_think_lens = paddle.to_tensor([10, 15], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1, -1], dtype="int32")
        step_idx = paddle.to_tensor([5, 8], dtype="int64")
        limit_status = paddle.to_tensor([0, 0], dtype="int32")
        accept_num = paddle.to_tensor([3, 2], dtype="int32")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert next_tokens.numpy()[0, 0] == 100
        assert accept_num.numpy()[0] == 3
        assert limit_status.numpy()[0] == 0

    def test_model_naturally_generates_think_end_id(self):
        """Test when model naturally generates think_end_id"""
        next_tokens = paddle.to_tensor([[100, 999, 102]], dtype="int64")
        max_think_lens = paddle.to_tensor([20], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([5], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([3], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert limit_status.numpy()[0] == 1  # done_status = 1 when inject_len == 0

    def test_disabled_both_limits_early_return(self):
        """Test early return when both max_think_len and max_reply_len are negative"""
        next_tokens = paddle.to_tensor([[100, 101]], dtype="int64")
        max_think_lens = paddle.to_tensor([-1], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([100], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([2], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert limit_status.numpy()[0] == 0
        assert accept_num.numpy()[0] == 2

    def test_zero_accept_num_early_return(self):
        """Test early return when accept_num is 0"""
        next_tokens = paddle.to_tensor([[100]], dtype="int64")
        max_think_lens = paddle.to_tensor([5], dtype="int32")
        max_reply_lens = paddle.to_tensor([-1], dtype="int32")
        step_idx = paddle.to_tensor([10], dtype="int64")
        limit_status = paddle.to_tensor([0], dtype="int32")
        accept_num = paddle.to_tensor([0], dtype="int32")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        eos_token_ids = paddle.to_tensor([[2]], dtype="int64")
        inject_token_ids = paddle.to_tensor([], dtype="int64")
        think_end_id = 999

        speculate_limit_thinking_content_length_v3(
            next_tokens,
            max_think_lens,
            max_reply_lens,
            step_idx,
            limit_status,
            accept_num,
            stop_flags,
            eos_token_ids,
            inject_token_ids,
            think_end_id,
            False,
        )

        assert accept_num.numpy()[0] == 0
        assert limit_status.numpy()[0] == 0


if __name__ == "__main__":
    unittest.main()

