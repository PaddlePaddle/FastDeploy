"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from fastdeploy.model_executor.ops.gpu import top_p_candidates


def top_p_candidates_dynamic_top_p(probs, top_p_per_bid, candidates_len, output_padding_offset, max_seq_len):
    """
    Simulate TopPCandidates, supporting dynamic selection of Top-P values based on bid.

    Args:
        probs: numpy.ndarray, shape [token_num, vocab_size]
               Probability distribution over the vocabulary for each token.
        top_p_per_bid: list or numpy.ndarray, shape [num_bid]
               Top-P values for each logical block (bid), e.g., [0.7, 0.9, 0.5].
        candidates_len: int
               Maximum number of candidate tokens to return for each token.
        output_padding_offset: numpy.ndarray, shape [token_num]
               Offset for each token, used to compute the original token ID (ori_token_id).
        max_seq_len: int
               Used to compute bid = ori_token_id // max_seq_len.

    Returns:
        verify_tokens: List[List[int]], list of candidate token IDs for each token.
        verify_scores: List[List[float]], list of candidate token probability scores for each token.
        actual_candidate_lens: List[int], actual number of candidate tokens returned for each token.
        ori_token_ids: List[int], original token ID for each token.
        bid_list: List[int], bid for each token.
    """
    token_num, vocab_size = probs.shape
    verify_tokens = []
    verify_scores = []
    actual_candidate_lens = []
    ori_token_ids = []
    bid_list = []

    top_p_per_bid = np.array(top_p_per_bid)

    num_bid = len(top_p_per_bid)

    for token_id in range(token_num):
        # --- Compute ori_token_id and bid ---
        offset = output_padding_offset[token_id]
        ori_token_id = token_id + offset
        bid = ori_token_id // max_seq_len

        # If the bid is out of the range of top_p_per_bid, you can choose to clamp it to [0, num_bid - 1]
        if bid < 0:
            bid = 0
        if bid >= num_bid:
            bid = (
                num_bid - 1
            )  # Or you could raise an error or exception; here, we simply handle it by using the last bid.

        token_top_p = top_p_per_bid[bid]  # Dynamically retrieve the top_p value for the given bid.

        ori_token_ids.append(ori_token_id)
        bid_list.append(bid)

        # The probability distribution of the current token.
        token_probs = probs[token_id, :]
        # Sort by probability in descending order.
        sorted_indices = np.argsort(token_probs)[::-1]
        sorted_probs = token_probs[sorted_indices]

        accumulated_prob = 0.0
        selected_indices = []
        selected_probs = []

        for sort_idx, (prob, token_idx) in enumerate(zip(sorted_probs, sorted_indices)):
            if sort_idx >= candidates_len:
                break  # Return at most candidates_len.

            accumulated_prob += prob
            selected_indices.append(int(token_idx))
            selected_probs.append(float(prob))

            if accumulated_prob >= token_top_p:
                break  # The cumulative probability satisfies the Top-P criterion.

        # If the Top-P threshold is not met, return the tokens that have already been selected.
        actual_len = len(selected_indices)
        actual_candidate_lens.append(actual_len)
        # Pad the insufficient token_id with 0.
        padded_token_ids = selected_indices + [0] * (candidates_len - actual_len)
        # Pad the insufficient score with 0.0.
        padded_scores = selected_probs + [0.0] * (candidates_len - actual_len)

        verify_tokens.append(padded_token_ids)
        verify_scores.append(padded_scores)

    return verify_scores, verify_tokens, actual_candidate_lens, ori_token_ids, bid_list


def top_p_candidates_ref(probs, top_p, output_padding_offset, candidates_len, max_seq_len):
    ret = top_p_candidates_dynamic_top_p(probs, top_p, candidates_len, output_padding_offset, max_seq_len)
    return [ret[0], ret[1], ret[2]]


class TestTopPCandidates(unittest.TestCase):
    def test_top_p_candidates(self):
        paddle.seed(42)
        token_num = 5
        vocab_size = 100
        candidates_len = 5
        max_seq_len = 120
        probs = paddle.randn([token_num, vocab_size])
        top_p = paddle.randn([token_num])
        bs = 5
        tokens = [1] * bs
        output_padding_offset = []
        opo_offset = 0
        for bid in range(bs):
            ts = tokens[bid]
            for i in range(ts):
                output_padding_offset.append(opo_offset)
            opo_offset += max_seq_len - ts
        output_padding_offset = paddle.to_tensor(output_padding_offset).astype(paddle.int32)
        ret1 = top_p_candidates(probs, top_p, output_padding_offset, candidates_len, max_seq_len)
        ret2 = top_p_candidates_ref(probs, top_p, output_padding_offset, candidates_len, max_seq_len)
        np.testing.assert_allclose(ret1[0].numpy(), ret2[0])
        np.testing.assert_allclose(ret1[1].numpy(), ret2[1])
        np.testing.assert_allclose(ret1[2].numpy(), ret2[2])


if __name__ == "__main__":
    unittest.main()
