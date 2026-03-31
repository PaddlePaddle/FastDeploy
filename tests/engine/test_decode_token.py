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
from fastdeploy.input.text_processor import DataProcessor
from fastdeploy.utils import envs

MODEL_PATH = os.getenv("MODEL_PATH", "") + "/ERNIE-4.5-0.3B-Paddle"


class TestDecodeToken(unittest.TestCase):
    """Test case for _decode_token method with real tokenizer"""

    def setUp(self):
        self.req_id = "test_req_123"

        self.data_processor_obj = DataProcessor(MODEL_PATH)

        self.data_processor = MagicMock()
        self.data_processor.tokenizer = self.data_processor_obj.tokenizer
        self.data_processor.decode_status = self.data_processor_obj.decode_status
        self.data_processor.ids2tokens = self.data_processor_obj.ids2tokens

        self.engine = MagicMock(spec=EngineService)
        self.engine.data_processor = self.data_processor
        self.engine._decode_token = EngineService._decode_token.__get__(self.engine, EngineService)

        # Common init for decode_status
        self.data_processor.decode_status[self.req_id] = [0, 0, [], ""]

    def _tokenize(self, text):
        return self.data_processor_obj.tokenizer(text, add_special_tokens=False)["input_ids"]

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
            for char in ["你", "好"]:
                tokens = self._tokenize(char)
                delta_text, _ = self.engine._decode_token(tokens, self.req_id, is_end=False)
                self.assertTrue(len(delta_text) > 0)

            delta_text, _ = self.engine._decode_token([], self.req_id, is_end=True)
            self._assert_cleaned_up()

    def test_undecoded_tokens_on_end(self):
        """Test that tokens which produce no visible text during streaming
        are force-decoded when is_end=True"""
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
