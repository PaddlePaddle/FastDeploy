from unittest.mock import Mock

import numpy as np
import paddle
import pytest

from fastdeploy.worker.input_batch import recover_batch_index_for_sampler_output


def _make_sampler_output(batch_size, with_sampling_mask=True):
    """Create a minimal mock SamplerOutput for testing reorder logic."""
    so = Mock()
    so.sampled_token_ids = paddle.arange(batch_size, dtype="int64").unsqueeze(1)
    so.logprobs_tensors = Mock()
    so.logprobs_tensors.logprob_token_ids = paddle.arange(batch_size, dtype="int64").unsqueeze(1)
    so.logprobs_tensors.logprobs = paddle.arange(batch_size, dtype="float32").unsqueeze(1)
    so.logprobs_tensors.selected_token_ranks = paddle.zeros([batch_size, 1], dtype="int64")
    so.token_num_per_batch = None
    so.cu_batch_token_offset = None
    so.logits = None

    if with_sampling_mask:
        so.sampling_mask = [np.array([i * 10, i * 10 + 1, i * 10 + 2]) for i in range(batch_size)]
    else:
        so.sampling_mask = None

    return so


class TestRecoverBatchIndexSamplingMask:
    """Test sampling_mask reordering in recover_batch_index_for_sampler_output."""

    def test_no_sampling_mask_no_error(self):
        """SamplerOutput without sampling_mask should not raise."""
        so = _make_sampler_output(batch_size=4, with_sampling_mask=False)
        index_to_batch_id = {0: 2, 1: 0, 2: 3, 3: 1}

        recover_batch_index_for_sampler_output(so, index_to_batch_id, enable_pd_reorder=True)

        assert so.sampling_mask is None

    def test_sampling_mask_reorder_matches_token_ids(self):
        """After reorder, sampling_mask[i] should correspond to sampled_token_ids[i]."""
        batch_size = 4
        so = _make_sampler_output(batch_size=batch_size, with_sampling_mask=True)

        original_masks = [m.copy() for m in so.sampling_mask]

        # index_to_batch_id = {0:2, 1:0, 2:3, 3:1}
        # src_order = [k for k,v in sorted(..., key=v)] = [1, 3, 0, 2]
        # result[i] = src[src_order[i]]
        index_to_batch_id = {0: 2, 1: 0, 2: 3, 3: 1}

        recover_batch_index_for_sampler_output(so, index_to_batch_id, enable_pd_reorder=True)

        reordered_token_ids = so.sampled_token_ids.numpy().flatten()
        for i in range(batch_size):
            token_id = int(reordered_token_ids[i])
            expected_mask = original_masks[token_id]
            np.testing.assert_array_equal(
                so.sampling_mask[i],
                expected_mask,
                err_msg=f"Position {i}: sampling_mask doesn't match sampled_token_ids",
            )

    def test_identity_reorder_is_noop(self):
        """When index_to_batch_id is identity, function returns early without changes."""
        batch_size = 3
        so = _make_sampler_output(batch_size=batch_size, with_sampling_mask=True)
        original_masks = [m.copy() for m in so.sampling_mask]

        index_to_batch_id = {0: 0, 1: 1, 2: 2}

        recover_batch_index_for_sampler_output(so, index_to_batch_id, enable_pd_reorder=True)

        for i in range(batch_size):
            np.testing.assert_array_equal(so.sampling_mask[i], original_masks[i])

    def test_pd_reorder_disabled_is_noop(self):
        """When enable_pd_reorder=False, nothing is reordered."""
        batch_size = 3
        so = _make_sampler_output(batch_size=batch_size, with_sampling_mask=True)
        original_masks = [m.copy() for m in so.sampling_mask]
        original_token_ids = so.sampled_token_ids.clone()

        index_to_batch_id = {0: 2, 1: 0, 2: 1}

        recover_batch_index_for_sampler_output(so, index_to_batch_id, enable_pd_reorder=False)

        assert paddle.equal_all(so.sampled_token_ids, original_token_ids)
        for i in range(batch_size):
            np.testing.assert_array_equal(so.sampling_mask[i], original_masks[i])

    def test_sampling_mask_longer_than_sort_len(self):
        """Tail elements beyond sort_len are preserved in place."""
        so = _make_sampler_output(batch_size=5, with_sampling_mask=True)
        original_masks = [m.copy() for m in so.sampling_mask]

        # Only reorder first 3 positions; positions 3,4 should stay put
        index_to_batch_id = {0: 1, 1: 2, 2: 0}

        recover_batch_index_for_sampler_output(so, index_to_batch_id, enable_pd_reorder=True)

        # src_order = [2, 0, 1]
        np.testing.assert_array_equal(so.sampling_mask[0], original_masks[2])
        np.testing.assert_array_equal(so.sampling_mask[1], original_masks[0])
        np.testing.assert_array_equal(so.sampling_mask[2], original_masks[1])
        np.testing.assert_array_equal(so.sampling_mask[3], original_masks[3])
        np.testing.assert_array_equal(so.sampling_mask[4], original_masks[4])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
