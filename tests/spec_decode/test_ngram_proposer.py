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
from fastdeploy.spec_decode.ngram import NgramProposer


class TestNgramProposer(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available")
        paddle.set_device("gpu")
        try:
            import fastdeploy.model_executor.ops.gpu as _gpu_ops

            if getattr(_gpu_ops, "ngram_match", None) is None:
                raise ImportError("ngram_match op not compiled")
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import ngram_match op: {e}")

        fd_config = get_default_test_fd_config()
        fd_config.model_config = FakeModelConfig()
        fd_config.model_config.max_model_len = 256
        fd_config.speculative_config = SpeculativeConfig({})
        fd_config.speculative_config.method = "ngram"
        fd_config.speculative_config.num_speculative_tokens = 5
        fd_config.speculative_config.max_ngram_size = 3
        fd_config.speculative_config.min_ngram_size = 1
        fd_config.scheduler_config.max_num_seqs = 2
        self.fd_config = fd_config

        bsz = fd_config.scheduler_config.max_num_seqs
        max_draft = fd_config.speculative_config.num_speculative_tokens
        max_model_len = fd_config.model_config.max_model_len
        self.bsz = bsz
        self.max_draft = max_draft
        self.max_model_len = max_model_len

        self.share_inputs = {
            "token_ids_all": paddle.zeros([bsz, max_model_len], dtype="int64"),
            "prompt_lens": paddle.zeros([bsz, 1], dtype="int64"),
            "step_idx": paddle.zeros([bsz, 1], dtype="int64"),
            "actual_draft_token_num": paddle.full([bsz, 1], fill_value=max_draft, dtype="int32"),
            "draft_tokens": paddle.zeros([bsz, max_draft + 1], dtype="int64"),
            "seq_lens_this_time": paddle.ones([bsz], dtype="int32"),
            "seq_lens_encoder": paddle.zeros([bsz], dtype="int32"),
            "seq_lens_decoder": paddle.ones([bsz], dtype="int32"),
            "max_dec_len": paddle.full([bsz, 1], fill_value=200, dtype="int64"),
        }

    # Init / config binding
    def test_init_config_binding(self):
        """max_ngram_size and max_draft_token_num are correctly read from fd_config."""
        proposer = NgramProposer(self.fd_config)
        self.assertEqual(proposer.max_ngram_size, 3)
        self.assertEqual(proposer.max_draft_token_num, 5)

    # No-proposal scenarios
    def test_run_no_proposal_step_idx_zero(self):
        """step_idx=0 means no tokens generated; kernel cannot form any ngram pattern."""
        proposer = NgramProposer(self.fd_config)
        self.share_inputs["step_idx"][:] = 0
        proposer.run(self.share_inputs)
        paddle.device.synchronize()

        slt = self.share_inputs["seq_lens_this_time"].numpy()
        np.testing.assert_array_equal(slt, [1, 1], err_msg="seq_lens_this_time should remain 1 when step_idx=0")

    # No-proposal scenarios
    def test_run_no_proposal_tokens_not_in_prompt(self):
        """Generated tokens never appear in the prompt → no match, no draft proposals."""
        proposer = NgramProposer(self.fd_config)

        prompt_len = 6
        prompt = [1, 2, 3, 4, 5, 6]  # unique tokens
        generated = [100, 200, 300]  # tokens absent from prompt

        token_ids_all_np = np.zeros((self.bsz, self.max_model_len), dtype=np.int64)
        for b in range(self.bsz):
            token_ids_all_np[b, :prompt_len] = prompt
            token_ids_all_np[b, prompt_len : prompt_len + len(generated)] = generated

        self.share_inputs["token_ids_all"] = paddle.to_tensor(token_ids_all_np, place=paddle.CUDAPlace(0))
        self.share_inputs["prompt_lens"] = paddle.full([self.bsz, 1], fill_value=prompt_len, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([self.bsz, 1], fill_value=len(generated), dtype="int64")

        proposer.run(self.share_inputs)
        paddle.device.synchronize()

        slt = self.share_inputs["seq_lens_this_time"].numpy()
        np.testing.assert_array_equal(
            slt, [1, 1], err_msg="No match expected when generated tokens absent from prompt"
        )

    # Successful proposal
    def test_run_with_match_produces_draft_tokens(self):
        """
        When the last ngram_size generated tokens reappear in the prompt,
        the tokens following that match position become draft proposals.

        Setup (max_ngram_size=3, step_idx=3):
            prompt    = [10, 20, 30, 40, 50, 10, 20, 30]  (prompt_len=8)
            generated = [40, 50, 10]                        (step_idx=3)

        Pattern = generated[step_idx - 3 : step_idx] = [40, 50, 10]
        Matches prompt at position 3 → proposals = prompt[6:8] = [20, 30]
        Expected: seq_lens_this_time = 3, draft_tokens[:, 1:3] = [[20, 30], [20, 30]]
        """
        proposer = NgramProposer(self.fd_config)

        prompt_len = 8
        prompt = [10, 20, 30, 40, 50, 10, 20, 30]
        generated = [40, 50, 10]

        token_ids_all_np = np.zeros((self.bsz, self.max_model_len), dtype=np.int64)
        for b in range(self.bsz):
            token_ids_all_np[b, :prompt_len] = prompt
            token_ids_all_np[b, prompt_len : prompt_len + len(generated)] = generated

        self.share_inputs["token_ids_all"] = paddle.to_tensor(token_ids_all_np, place=paddle.CUDAPlace(0))
        self.share_inputs["prompt_lens"] = paddle.full([self.bsz, 1], fill_value=prompt_len, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([self.bsz, 1], fill_value=len(generated), dtype="int64")

        proposer.run(self.share_inputs)
        paddle.device.synchronize()

        slt = self.share_inputs["seq_lens_this_time"].numpy()
        dt = self.share_inputs["draft_tokens"].numpy()

        # 1 base token + 2 draft tokens = seq_len 3
        np.testing.assert_array_equal(slt, [3, 3], err_msg="seq_lens_this_time mismatch")
        # Draft slots 1 and 2 should be [20, 30] for every batch item
        np.testing.assert_array_equal(dt[:, 1:3], [[20, 30], [20, 30]], err_msg="draft_tokens mismatch")

    # Successful proposal
    def test_run_with_match_respects_max_dec_len(self):
        """Draft count is clipped when remaining budget (max_dec_len - step_idx) is exhausted."""
        proposer = NgramProposer(self.fd_config)

        prompt_len = 8
        prompt = [10, 20, 30, 40, 50, 10, 20, 30]
        generated = [40, 50, 10]

        token_ids_all_np = np.zeros((self.bsz, self.max_model_len), dtype=np.int64)
        for b in range(self.bsz):
            token_ids_all_np[b, :prompt_len] = prompt
            token_ids_all_np[b, prompt_len : prompt_len + len(generated)] = generated

        self.share_inputs["token_ids_all"] = paddle.to_tensor(token_ids_all_np, place=paddle.CUDAPlace(0))
        self.share_inputs["prompt_lens"] = paddle.full([self.bsz, 1], fill_value=prompt_len, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([self.bsz, 1], fill_value=len(generated), dtype="int64")
        # remaining = max_dec_len - step_idx - 1 = 4 - 3 - 1 = 0 → no draft tokens
        self.share_inputs["max_dec_len"] = paddle.full([self.bsz, 1], fill_value=4, dtype="int64")

        proposer.run(self.share_inputs)
        paddle.device.synchronize()

        slt = self.share_inputs["seq_lens_this_time"].numpy()
        np.testing.assert_array_equal(slt, [1, 1], err_msg="No drafts expected when max_dec_len budget exhausted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
