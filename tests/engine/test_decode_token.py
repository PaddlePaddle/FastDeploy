"""
Author:
Date: 2026-03-31 10:40:18
LastEditors:
LastEditTime: 2026-04-01 11:00:47
FilePath: /fastdeploy/test_decode_token.py
"""

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

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fastdeploy.engine.common_engine import EngineService
from fastdeploy.utils import envs


def _make_mock_ids2tokens(decode_status, undecoded_tokens=None):
    """
    Create a mock ids2tokens that simulates incremental decoding behavior.

    Simulates the non-HF path of DataProcessor.ids2tokens:
    - decode_status[task_id] = [prefix_offset, read_offset, cumulative_token_ids, cumulative_text]
    - Returns (delta_text, previous_token_ids, previous_texts)

    Args:
        decode_status: shared dict for tracking decode state
        undecoded_tokens: set of token IDs that produce no visible text (delta_text=""),
                          simulating tokens that cannot be decoded incrementally.
                          read_offset will NOT advance for these tokens.
    """
    undecoded_tokens = undecoded_tokens or set()

    # Simple token->char mapping for testing
    token_map = {
        1000: "你",
        1001: "好",
    }

    def mock_ids2tokens(token_ids, task_id):
        if task_id not in decode_status:
            decode_status[task_id] = [0, 0, [], ""]

        previous_token_ids = list(decode_status[task_id][2])
        previous_texts = decode_status[task_id][3]

        # Append new tokens to cumulative list
        decode_status[task_id][2] += token_ids

        # Check if all new tokens are "undecoded" (produce no visible text)
        all_undecoded = all(tid in undecoded_tokens for tid in token_ids) if token_ids else True

        if all_undecoded and token_ids:
            # These tokens can't be decoded yet - don't advance read_offset
            delta_text = ""
        else:
            # Normal decoding
            delta_text = ""
            for tid in token_ids:
                delta_text += token_map.get(tid, f"<{tid}>")

            if token_ids:
                # Only advance offsets when there are actual tokens
                cum_len = len(decode_status[task_id][2])
                decode_status[task_id][0] = max(0, cum_len - 1)  # prefix_offset
                decode_status[task_id][1] = cum_len  # read_offset
                decode_status[task_id][3] += delta_text

        return delta_text, previous_token_ids, previous_texts

    return mock_ids2tokens


class TestDecodeToken(unittest.TestCase):
    """Test case for _decode_token method with mocked tokenizer"""

    def setUp(self):
        self.req_id = "test_req_123"
        self._setup_engine()

    def _setup_engine(self, undecoded_tokens=None):
        self.decode_status = {}

        self.data_processor = MagicMock()
        self.data_processor.decode_status = self.decode_status
        self.data_processor.ids2tokens = _make_mock_ids2tokens(self.decode_status, undecoded_tokens)

        self.engine = MagicMock(spec=EngineService)
        self.engine.data_processor = self.data_processor
        self.engine._decode_token = EngineService._decode_token.__get__(self.engine, EngineService)

        # Common init for decode_status
        self.decode_status[self.req_id] = [0, 0, [], ""]

    def _assert_cleaned_up(self):
        self.assertNotIn(self.req_id, self.data_processor.decode_status)

    def test_empty_end(self):
        """Empty token_ids with is_end=True should return empty and cleanup"""
        with patch.object(envs, "FD_ENABLE_RETURN_TEXT", True):
            delta_text, returned_tokens = self.engine._decode_token([], self.req_id, is_end=True)
            self.assertEqual(delta_text, "")
            self.assertEqual(returned_tokens, [])
            self._assert_cleaned_up()

    def test_incremental_decoding_and_cleanup(self):
        """Tokens added in multiple steps should decode correctly and cleanup at end"""
        with patch.object(envs, "FD_ENABLE_RETURN_TEXT", True):
            for token_id in [1000, 1001]:  # "你", "好"
                delta_text, _ = self.engine._decode_token([token_id], self.req_id, is_end=False)
                self.assertTrue(len(delta_text) > 0)

            delta_text, _ = self.engine._decode_token([], self.req_id, is_end=True)
            self._assert_cleaned_up()

    def test_undecoded_tokens_on_end(self):
        """Test that tokens which produce no visible text during streaming
        are force-decoded when is_end=True"""
        # Re-setup with 109584 as an undecoded token (produces no delta_text during streaming)
        self._setup_engine(undecoded_tokens={109584})
        self.decode_status[self.req_id] = [0, 0, [], ""]

        with patch.object(envs, "FD_ENABLE_RETURN_TEXT", True), patch.dict(os.environ, {"DEBUG_DECODE": "1"}):
            all_delta = ""

            delta_text, _ = self.engine._decode_token([109584], self.req_id, is_end=False)
            all_delta += delta_text

            # Now end the stream - force decode should recover any remaining text
            delta_end, _ = self.engine._decode_token([109584], self.req_id, is_end=False)
            all_delta += delta_end
            delta_end, _ = self.engine._decode_token([109584], self.req_id, is_end=False)
            all_delta += delta_end
            delta_end, token_ids = self.engine._decode_token([], self.req_id, is_end=True)
            all_delta += delta_end

            # The full text must be recovered either during streaming or at end
            self.assertEqual(token_ids, [109584, 109584, 109584])
            self._assert_cleaned_up()


if __name__ == "__main__":
    unittest.main()
