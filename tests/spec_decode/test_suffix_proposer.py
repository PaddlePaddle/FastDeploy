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
from utils import FakeModelConfig, get_default_test_fd_config

from fastdeploy.config import SpeculativeConfig
from fastdeploy.spec_decode.suffix import SuffixProposer


class TestSuffixProposer(unittest.TestCase):
    def setUp(self):
        self.fd_config = get_default_test_fd_config()
        self.fd_config.model_config = FakeModelConfig()
        self.fd_config.model_config.max_model_len = 2048
        self.fd_config.speculative_config = SpeculativeConfig({})
        self.fd_config.speculative_config.method = "suffix"
        self.fd_config.speculative_config.num_speculative_tokens = 4
        self.fd_config.speculative_config.suffix_decoding_max_tree_depth = 64
        self.fd_config.speculative_config.suffix_decoding_max_cached_requests = 4
        self.fd_config.speculative_config.suffix_decoding_max_spec_factor = 1.0
        self.fd_config.speculative_config.suffix_decoding_min_token_prob = 0.1
        self.fd_config.scheduler_config.max_num_seqs = 4

        bsz = self.fd_config.scheduler_config.max_num_seqs
        max_draft_tokens = self.fd_config.speculative_config.num_speculative_tokens
        self.share_inputs = {
            "stop_flags": paddle.full([bsz, 1], fill_value=False, dtype="bool"),
            "is_block_step": paddle.full([bsz], fill_value=False, dtype="bool"),
            "accept_tokens": paddle.zeros([bsz, max_draft_tokens], dtype="int64"),
            "accept_num": paddle.zeros([bsz], dtype="int32"),
            "seq_lens_this_time": paddle.zeros([bsz, 1], dtype="int32"),
            "seq_lens_encoder": paddle.zeros([bsz, 1], dtype="int32"),
            "seq_lens_decoder": paddle.zeros([bsz, 1], dtype="int32"),
            "draft_tokens": paddle.zeros([bsz, max_draft_tokens], dtype="int64"),
        }

    def test_start_and_stop_request(self):
        proposer = SuffixProposer(self.fd_config)

        idx = 0
        req_id = "req-001"
        prompt_token_ids = [1, 2, 3, 4]
        proposer.start_request(idx, req_id, prompt_token_ids)

        refs_context_tokens = np.full(
            (self.fd_config.scheduler_config.max_num_seqs, self.fd_config.model_config.max_model_len),
            -1,
            dtype=np.int32,
        )
        refs_req_id_to_idx = {}
        refs_idx_to_req_id = {}
        refs_context_tokens[idx, : len(prompt_token_ids)] = prompt_token_ids
        refs_req_id_to_idx[req_id] = idx
        refs_idx_to_req_id[idx] = req_id

        self.assertIsNotNone(proposer.suffix_cache)
        np.testing.assert_array_equal(proposer.context_tokens, refs_context_tokens)
        np.testing.assert_array_equal(proposer.req_id_to_idx, refs_req_id_to_idx)
        np.testing.assert_array_equal(proposer.idx_to_req_id, refs_idx_to_req_id)

        idx = 1
        req_id = "req-002"
        prompt_token_ids = [5, 6, 7, 8]
        proposer.start_request(idx, req_id, prompt_token_ids)

        refs_context_tokens[idx, : len(prompt_token_ids)] = prompt_token_ids
        refs_req_id_to_idx[req_id] = idx
        refs_idx_to_req_id[idx] = req_id

        np.testing.assert_array_equal(proposer.context_tokens, refs_context_tokens)
        np.testing.assert_array_equal(proposer.req_id_to_idx, refs_req_id_to_idx)
        np.testing.assert_array_equal(proposer.idx_to_req_id, refs_idx_to_req_id)

        proposer.stop_request("req-001")

        refs_req_id_to_idx.pop("req-001")
        refs_idx_to_req_id.pop(0)

        np.testing.assert_array_equal(proposer.context_tokens, refs_context_tokens)
        np.testing.assert_array_equal(proposer.req_id_to_idx, refs_req_id_to_idx)
        np.testing.assert_array_equal(proposer.idx_to_req_id, refs_idx_to_req_id)

    def test_propose(self):

        self.share_inputs["accept_tokens"][:, :2] = 42
        self.share_inputs["accept_num"][:] = 2
        self.share_inputs["seq_lens_this_time"][:, :] = 2
        self.share_inputs["seq_lens_encoder"][:, :] = 0
        self.share_inputs["seq_lens_decoder"][:, :] = 100
        self.share_inputs["draft_tokens"][:, :2] = 42
        self.share_inputs["draft_tokens"][:, 2:] = 53
        print(self.share_inputs)

        proposer = SuffixProposer(self.fd_config)
        ids = [0, 1, 2, 3]
        req_ids = ["req-001", "req-002", "req-003", "req-004"]
        prompt_token_ids_list = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        for idx, req_id, prompt_token_ids in zip(ids, req_ids, prompt_token_ids_list):
            proposer.start_request(idx, req_id, prompt_token_ids)

        proposer.run(self.share_inputs)

        refs_draft_tokens = np.array(
            [
                [42, 42, -1, -1],
                [42, 42, -1, -1],
                [42, 42, -1, -1],
                [42, 42, -1, -1],
            ],
            dtype=np.int64,
        )
        refs_seq_lens_this_time = np.array([[2], [2], [2], [2]], dtype=np.int32)

        np.testing.assert_array_equal(self.share_inputs["draft_tokens"].numpy(), refs_draft_tokens)
        np.testing.assert_array_equal(self.share_inputs["seq_lens_this_time"].numpy(), refs_seq_lens_this_time)


if __name__ == "__main__":
    unittest.main()
