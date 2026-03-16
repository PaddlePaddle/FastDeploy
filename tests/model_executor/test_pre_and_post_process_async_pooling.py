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

from __future__ import annotations

import queue
import sys
import traceback

import numpy as np
import paddle

from fastdeploy.model_executor.pre_and_post_process import async_pooling_output
from fastdeploy.worker.output import DecodeMode, ModelRunnerOutput


class TestAsyncPoolingOutput:
    """Test cases for async_pooling_output function."""

    def test_async_pooling_output_normal_case(self):
        """Test normal case with valid paddle tensors."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [
            paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            paddle.to_tensor([[5.0, 6.0]], dtype="float32"),
            paddle.to_tensor([[7.0, 8.0, 9.0]], dtype="float32"),
        ]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        assert not async_output_queue.empty()
        result = async_output_queue.get()

        assert isinstance(result, ModelRunnerOutput)
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 3

        # Check first tensor
        assert result.pooler_output[0] is not None
        assert result.pooler_output[0].shape == [2, 2]
        np.testing.assert_array_equal(
            result.pooler_output[0].numpy(), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        assert result.pooler_output[0].place.is_cpu_place()
        assert result.pooler_output[0].dtype == paddle.float32

        # Check second tensor
        assert result.pooler_output[1] is not None
        assert result.pooler_output[1].shape == [1, 2]
        np.testing.assert_array_equal(result.pooler_output[1].numpy(), np.array([[5.0, 6.0]], dtype=np.float32))
        assert result.pooler_output[1].place.is_cpu_place()
        assert result.pooler_output[1].dtype == paddle.float32

        # Check third tensor
        assert result.pooler_output[2] is not None
        assert result.pooler_output[2].shape == [1, 3]
        np.testing.assert_array_equal(result.pooler_output[2].numpy(), np.array([[7.0, 8.0, 9.0]], dtype=np.float32))
        assert result.pooler_output[2].place.is_cpu_place()
        assert result.pooler_output[2].dtype == paddle.float32

    def test_async_pooling_output_with_bfloat16_conversion(self):
        """Test conversion from bfloat16 to float32."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [
            paddle.to_tensor([[1.0, 2.0]], dtype="bfloat16"),
            paddle.to_tensor([[3.0, 4.0]], dtype="float32"),
        ]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 2

        # Check bfloat16 tensor conversion
        assert result.pooler_output[0] is not None
        assert result.pooler_output[0].dtype == paddle.float32
        assert result.pooler_output[0].place.is_cpu_place()

        # Check float32 tensor remains float32
        assert result.pooler_output[1] is not None
        assert result.pooler_output[1].dtype == paddle.float32
        assert result.pooler_output[1].place.is_cpu_place()

    def test_async_pooling_output_with_none_values(self):
        """Test handling of None values in the input list."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [
            paddle.to_tensor([[1.0, 2.0]], dtype="float32"),
            None,
            paddle.to_tensor([[3.0, 4.0]], dtype="float32"),
            None,
        ]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 4

        # Check first tensor
        assert result.pooler_output[0] is not None
        assert result.pooler_output[0].shape == [1, 2]

        # Check None values are preserved
        assert result.pooler_output[1] is None

        # Check third tensor
        assert result.pooler_output[2] is not None
        assert result.pooler_output[2].shape == [1, 2]

        # Check None values are preserved
        assert result.pooler_output[3] is None

    def test_async_pooling_output_with_empty_list(self):
        """Test handling of empty input list."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = []

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 0
        assert isinstance(result.pooler_output, list)

    def test_async_pooling_output_with_gpu_tensors(self):
        """Test handling of GPU tensors (conversion to CPU)."""
        # Skip if no GPU available
        if not paddle.device.is_compiled_with_cuda():
            print("○ SKIPPED (GPU not available)")
            return

        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [
            paddle.to_tensor([[1.0, 2.0]], dtype="float32"),
            paddle.to_tensor([[3.0, 4.0]], dtype="float32"),
        ]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 2

        # Both tensors should be on CPU
        for tensor in result.pooler_output:
            if tensor is not None:
                assert tensor.place.is_cpu_place()

    def test_async_pooling_output_tensor_cloning(self):
        """Test that tensors are properly cloned (not references)."""
        # Arrange
        async_output_queue = queue.Queue()
        original_tensor = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        pooler_output_list = [original_tensor]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Modify original tensor
        original_tensor[0, 0] = 99.0

        # Assert
        result = async_output_queue.get()
        output_tensor = result.pooler_output[0]

        # Output tensor should not be affected by modification to original
        assert output_tensor[0, 0].item() == 1.0  # Original value, not 99.0
        assert original_tensor[0, 0].item() == 99.0  # Modified value

    def test_async_pooling_output_multiple_dtypes(self):
        """Test handling of different tensor dtypes."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [
            paddle.to_tensor([[1, 2]], dtype="int32"),
            paddle.to_tensor([[3.0, 4.0]], dtype="float64"),
            paddle.to_tensor([[5.0, 6.0]], dtype="bfloat16"),
        ]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 3

        # All tensors should be converted to float32 (or stay as original dtype for non-bfloat16)
        # Check that bfloat16 was converted to float32
        assert result.pooler_output[2].dtype == paddle.float32

        # int32 and float64 should remain as their original dtypes
        assert result.pooler_output[0].dtype == paddle.int32
        assert result.pooler_output[1].dtype == paddle.float64

    def test_async_pooling_output_queue_behavior(self):
        """Test that the function properly uses the queue."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [paddle.to_tensor([[1.0]], dtype="float32")]

        # Act - call function multiple times
        async_pooling_output(async_output_queue, pooler_output_list)
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        assert async_output_queue.qsize() == 2

        # Check first result
        result1 = async_output_queue.get()
        assert result1.pooler_output is not None
        assert len(result1.pooler_output) == 1

        # Check second result
        result2 = async_output_queue.get()
        assert result2.pooler_output is not None
        assert len(result2.pooler_output) == 1

    def test_async_pooling_output_with_large_tensors(self):
        """Test handling of large tensors."""
        # Arrange
        async_output_queue = queue.Queue()
        large_tensor = paddle.randn([100, 100], dtype="float32")
        pooler_output_list = [large_tensor]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert result.pooler_output is not None
        assert len(result.pooler_output) == 1
        assert result.pooler_output[0].shape == [100, 100]
        assert result.pooler_output[0].dtype == paddle.float32
        assert result.pooler_output[0].place.is_cpu_place()

    def test_async_pooling_output_model_runner_output_structure(self):
        """Test that the output is a properly structured ModelRunnerOutput."""
        # Arrange
        async_output_queue = queue.Queue()
        pooler_output_list = [paddle.to_tensor([[1.0, 2.0]], dtype="float32")]

        # Act
        async_pooling_output(async_output_queue, pooler_output_list)

        # Assert
        result = async_output_queue.get()
        assert isinstance(result, ModelRunnerOutput)
        assert result.pooler_output is not None
        # decode_mode should have default value DecodeMode.TARGET
        assert result.decode_mode == DecodeMode.TARGET
        assert result.sampled_token_ids is None
        assert result.logprobs is None
        assert result.prompt_logprobs is None
        assert result.cu_num_generated_tokens == []


def run_all_tests():
    """Run all test cases and report results."""
    test_instance = TestAsyncPoolingOutput()
    test_methods = [
        method
        for method in dir(test_instance)
        if method.startswith("test_") and callable(getattr(test_instance, method))
    ]

    print(f"Running {len(test_methods)} test cases for async_pooling_output function...")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for method_name in test_methods:
        print(f"Running {method_name}...", end=" ")
        method = getattr(test_instance, method_name)

        try:
            method()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower():
                print("○ SKIPPED")
                skipped += 1
            else:
                print("✗ FAILED")
                print(f"  Error: {e}")
                traceback.print_exc()
                failed += 1

    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("🎉 All tests completed successfully!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
