"""
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

import queue

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.pre_and_post_process import async_generate_output
from fastdeploy.worker.output import DecodeMode, LogprobsTensors, ModelRunnerOutput


class TestAsyncGenerateOutput:
    """Test cases for async_generate_output function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.async_output_queue = queue.Queue()
        self.batch_size = 2
        self.seq_len = 3

    def _create_sampled_tokens(self, dtype=paddle.int64):
        """Create sampled tokens tensor."""
        return paddle.randint(low=0, high=1000, shape=(self.batch_size, self.seq_len), dtype=dtype)

    def _create_accept_token_nums(self, values=None):
        """Create accept token nums array."""
        if values is None:
            values = [self.seq_len, self.seq_len]
        return np.array(values, dtype=np.int32)

    def _create_logprobs_tensors(self, num_positions=2, num_tokens_per_position=5):
        """Create LogprobsTensors instance."""
        return LogprobsTensors.empty_cpu(num_positions, num_tokens_per_position)

    def test_async_generate_output_basic_target_mode(self):
        """Test basic functionality with TARGET decode mode."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.decode_mode == DecodeMode.TARGET
        assert output.sampled_token_ids is not None
        assert output.sampled_token_ids.shape[0] == self.batch_size * self.seq_len  # Flattened
        assert np.array_equal(output.cu_num_generated_tokens, [0, self.seq_len, self.seq_len * 2])

    def test_async_generate_output_basic_draft_mode(self):
        """Test basic functionality with DRAFT decode mode."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.DRAFT,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.decode_mode == DecodeMode.DRAFT
        assert output.sampled_token_ids is not None
        # In DRAFT mode, tokens are not flattened
        assert output.sampled_token_ids.shape == (self.batch_size, self.seq_len)

    def test_async_generate_output_with_prompt_logprobs(self):
        """Test with prompt_logprobs_list provided."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()
        prompt_logprobs_list = [
            self._create_logprobs_tensors(),
            self._create_logprobs_tensors(),
        ]

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            prompt_logprobs_list=prompt_logprobs_list,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.prompt_logprobs is not None
        assert len(output.prompt_logprobs) == len(prompt_logprobs_list)
        assert all(pl is not None for pl in output.prompt_logprobs)

    def test_async_generate_output_with_logprobs_tensors(self):
        """Test with logprobs_tensors provided."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()
        logprobs_tensors = self._create_logprobs_tensors()

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            logprobs_tensors=logprobs_tensors,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.logprobs is not None
        assert output.logprobs.logprob_token_ids is not None

    def test_async_generate_output_with_mixed_logprobs(self):
        """Test with both prompt_logprobs and logprobs_tensors."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()
        prompt_logprobs_list = [
            self._create_logprobs_tensors(),
            None,  # Test None handling
        ]
        logprobs_tensors = self._create_logprobs_tensors()

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            prompt_logprobs_list=prompt_logprobs_list,
            logprobs_tensors=logprobs_tensors,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.prompt_logprobs is not None
        assert len(output.prompt_logprobs) == len(prompt_logprobs_list)
        assert output.prompt_logprobs[0] is not None  # First element should be cloned
        assert output.prompt_logprobs[1] is None  # Second element should remain None
        assert output.logprobs is not None

    def test_async_generate_output_partial_accept_tokens(self):
        """Test with partial accept token nums (some sequences with zero tokens)."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums([2, 0])  # One sequence has zero tokens

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        # Should only include tokens from the first sequence
        assert output.sampled_token_ids.shape[0] == 2
        assert np.array_equal(output.cu_num_generated_tokens, [0, 2, 2])

    def test_async_generate_output_all_zero_accept_tokens(self):
        """Test when all accept token nums are zero."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums([0, 0])

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        # Should have empty sampled_token_ids
        assert output.sampled_token_ids.shape[0] == 0
        assert np.array_equal(output.cu_num_generated_tokens, [0, 0, 0])

    def test_async_generate_output_single_batch(self):
        """Test with single batch item."""
        # Arrange
        self.batch_size = 1
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums([3])

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.sampled_token_ids.shape[0] == 3
        assert np.array_equal(output.cu_num_generated_tokens, [0, 3])

    def test_async_generate_output_large_batch(self):
        """Test with larger batch size."""
        # Arrange
        self.batch_size = 10
        self.seq_len = 5
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums([5] * 10)

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.sampled_token_ids.shape[0] == 50  # 10 * 5
        assert len(output.cu_num_generated_tokens) == 11  # [0, 5, 10, ..., 50]

    def test_async_generate_output_varying_accept_tokens(self):
        """Test with varying accept token nums."""
        # Arrange
        sampled_tokens = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype=paddle.int64)
        accept_token_nums = np.array([2, 1], dtype=np.int32)

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        # Should have tokens: [1, 2] from first sequence and [4] from second
        assert output.sampled_token_ids.shape[0] == 3
        assert np.array_equal(output.sampled_token_ids, [1, 2, 4])
        assert np.array_equal(output.cu_num_generated_tokens, [0, 2, 3])

    def test_async_generate_output_none_queue_raises_error(self):
        """Test that None queue raises assertion error."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()

        # Act & Assert
        with pytest.raises(AssertionError, match="async_output_queue must not be None"):
            async_generate_output(
                async_output_queue=None,
                sampled_tokens=sampled_tokens,
                accept_token_nums=accept_token_nums,
                decode_mode=DecodeMode.TARGET,
            )

    def test_async_generate_output_empty_prompt_logprobs(self):
        """Test with empty prompt_logprobs_list."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()
        prompt_logprobs_list = []  # Empty list

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            prompt_logprobs_list=prompt_logprobs_list,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        assert output.prompt_logprobs == []  # Should preserve empty list

    def test_async_generate_output_cloning_behavior(self):
        """Test that prompt_logprobs and logprobs_tensors are properly cloned."""
        # Arrange
        sampled_tokens = self._create_sampled_tokens()
        accept_token_nums = self._create_accept_token_nums()

        # Create original tensors
        original_logprobs = self._create_logprobs_tensors()
        original_prompt_logprobs = [self._create_logprobs_tensors()]

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            prompt_logprobs_list=original_prompt_logprobs,
            logprobs_tensors=original_logprobs,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()

        # Verify cloning - output objects should be different from originals
        assert output.logprobs is not original_logprobs
        assert output.prompt_logprobs[0] is not original_prompt_logprobs[0]

    def test_async_generate_output_with_float32_tokens(self):
        """Test with float32 tensor (edge case)."""
        # Arrange
        sampled_tokens = paddle.randn((self.batch_size, self.seq_len), dtype=paddle.float32)
        accept_token_nums = self._create_accept_token_nums()

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens,
            accept_token_nums=accept_token_nums,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert not self.async_output_queue.empty()
        output = self.async_output_queue.get()
        assert isinstance(output, ModelRunnerOutput)
        # Should still work even with float32, converted to numpy

    def test_async_generate_output_queue_ordering(self):
        """Test that multiple calls maintain proper queue ordering."""
        # Arrange
        sampled_tokens1 = paddle.to_tensor([[1, 2]], dtype=paddle.int64)
        accept_token_nums1 = np.array([2], dtype=np.int32)

        sampled_tokens2 = paddle.to_tensor([[3, 4]], dtype=paddle.int64)
        accept_token_nums2 = np.array([2], dtype=np.int32)

        # Act
        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens1,
            accept_token_nums=accept_token_nums1,
            decode_mode=DecodeMode.TARGET,
        )

        async_generate_output(
            async_output_queue=self.async_output_queue,
            sampled_tokens=sampled_tokens2,
            accept_token_nums=accept_token_nums2,
            decode_mode=DecodeMode.TARGET,
        )

        # Assert
        assert self.async_output_queue.qsize() == 2

        # First output should have tokens [1, 2]
        output1 = self.async_output_queue.get()
        assert np.array_equal(output1.sampled_token_ids, [1, 2])

        # Second output should have tokens [3, 4]
        output2 = self.async_output_queue.get()
        assert np.array_equal(output2.sampled_token_ids, [3, 4])


if __name__ == "__main__":
    pytest.main([__file__])
