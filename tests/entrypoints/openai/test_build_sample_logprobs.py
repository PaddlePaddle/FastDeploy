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
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.worker.output import Logprob, LogprobsLists


def get_patch_path(cls, method="__init__"):
    return f"{cls.__module__}.{cls.__qualname__}.{method}"


class TestBuildSampleLogprobs(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by creating an instance of the LLM class using Mock.
        """
        patch_llm = get_patch_path(LLM)
        with patch(patch_llm, return_value=None):
            self.llm = LLM()
            # mock d data_processor
            self.llm.llm_engine = MagicMock()
            self.llm.llm_engine.data_processor.process_logprob_response.side_effect = (
                lambda ids, **kwargs: f"token_{ids[0]}"
            )

    def test_build_sample_logprobs_basic(self):
        """
        Test case for building sample logprobs when `topk_logprobs` is valid.
        """
        logprob_token_ids = [[100, 101, 102]]
        logprobs = [[-0.1, -0.5, -1.0]]
        sampled_token_ranks = [0]

        logprobs_lists = LogprobsLists(
            logprob_token_ids=logprob_token_ids, logprobs=logprobs, sampled_token_ranks=sampled_token_ranks
        )

        result = self.llm._build_sample_logprobs(logprobs_lists, topk_logprobs=2)

        expected = [
            {
                101: Logprob(logprob=-0.5, rank=1, decoded_token="token_101"),
                102: Logprob(logprob=-1.0, rank=2, decoded_token="token_102"),
            }
        ]

        self.assertEqual(result, expected)

    def test_build_sample_logprobs_empty_input(self):
        """
        Test case where `logprob_token_ids` is empty.
        """
        logprobs_lists = MagicMock(spec=LogprobsLists)
        logprobs_lists.logprob_token_ids = []
        result = self.llm._build_sample_logprobs(logprobs_lists, topk_logprobs=2)
        self.assertIsNone(result)

    def test_build_sample_logprobs_invalid_topk(self):
        """
        Test case where `topk` value exceeds length of first element in `logprob_token_ids`.
        """
        logprobs_lists = MagicMock(spec=LogprobsLists)
        logprobs_lists.logprob_token_ids = [[100]]
        result = self.llm._build_sample_logprobs(logprobs_lists, topk_logprobs=2)
        self.assertIsNone(result)

    def test_decode_token(self):
        """
        Test case for decoding a single token ID.
        """
        token_id = 123
        decoded = self.llm._decode_token(token_id)
        self.assertEqual(decoded, "token_123")


if __name__ == "__main__":
    unittest.main()
