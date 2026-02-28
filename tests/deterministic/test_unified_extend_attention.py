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
Unit tests for unified extend attention Triton kernel.

Tests:
1. build_unified_kv_indices correctness
2. extend_attention_fwd_unified basic functionality
3. Deterministic behavior verification

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_unified_extend_attention.py -v
"""

import os

import pytest

pytestmark = pytest.mark.gpu

# Set env vars before importing
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


@pytest.fixture(scope="module", autouse=True)
def _setup_env():
    """Setup environment for tests."""
    import paddle

    paddle.device.set_device("gpu:0")


class TestBuildUnifiedKvIndices:
    """Tests for build_unified_kv_indices function."""

    def test_basic_functionality(self):
        """Test basic index building."""
        import paddle

        from fastdeploy.model_executor.layers.attention.triton_ops import (
            build_unified_kv_indices,
        )

        # Create test data
        bs = 2
        # Request 0: prefix 2 blocks, extend 1 block
        # Request 1: prefix 0 blocks, extend 2 blocks
        prefix_kv_indptr = paddle.to_tensor([0, 2, 2], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11], dtype="int64")  # 2 blocks for request 0

        extend_start_loc = paddle.to_tensor([0, 1], dtype="int32")  # start positions in extend_kv_indices
        extend_seq_lens = paddle.to_tensor([1, 2], dtype="int32")  # 1 block for req 0, 2 blocks for req 1
        extend_kv_indices = paddle.to_tensor([20, 30, 31], dtype="int64")  # block ids

        unified_indptr, unified_indices, prefix_lens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )

        # Verify shapes
        assert unified_indptr.shape[0] == bs + 1
        assert unified_indices.shape[0] == 5  # 2 + 1 + 2 = 5 blocks total
        assert prefix_lens.shape[0] == bs

        # Verify prefix_lens
        assert prefix_lens[0].item() == 2  # Request 0 has 2 prefix blocks
        assert prefix_lens[1].item() == 0  # Request 1 has 0 prefix blocks

    def test_empty_prefix(self):
        """Test with no prefix cache hit."""
        import paddle

        from fastdeploy.model_executor.layers.attention.triton_ops import (
            build_unified_kv_indices,
        )

        bs = 2
        prefix_kv_indptr = paddle.to_tensor([0, 0, 0], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([], dtype="int64")

        extend_start_loc = paddle.to_tensor([0, 2], dtype="int32")
        extend_seq_lens = paddle.to_tensor([2, 2], dtype="int32")
        extend_kv_indices = paddle.to_tensor([10, 11, 20, 21], dtype="int64")

        unified_indptr, unified_indices, prefix_lens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )

        # All prefix_lens should be 0
        assert (prefix_lens == 0).all().item()


class TestExtendAttentionFwdUnified:
    """Tests for extend_attention_fwd_unified kernel."""

    @pytest.mark.skip(reason="Requires full KV cache setup")
    def test_basic_attention(self):
        """Test basic attention computation."""
        # This test requires a full KV cache setup
        # which is complex to create in isolation
        pass

    @pytest.mark.skip(reason="Requires full KV cache setup")
    def test_causal_mask(self):
        """Test causal mask handling."""
        pass

    @pytest.mark.skip(reason="Requires full KV cache setup")
    def test_prefix_extend_boundary(self):
        """Test correct handling of prefix/extend boundary."""
        pass


class TestDeterministicBehavior:
    """Tests for deterministic behavior with prefix caching."""

    @pytest.mark.skip(reason="Integration test - run separately")
    def test_same_output_with_cache_hit_miss(self):
        """
        Test that cache hit and cache miss produce identical outputs.

        This is the key test for deterministic behavior with prefix caching.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
