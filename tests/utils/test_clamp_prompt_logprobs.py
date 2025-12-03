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

from fastdeploy.utils import clamp_prompt_logprobs
from fastdeploy.worker.output import Logprob


class TestClampPromptLogprobs(unittest.TestCase):
    def test_none_input(self):
        """Test case when input is None"""
        result = clamp_prompt_logprobs(None)
        self.assertIsNone(result)

    def test_empty_list(self):
        """Test empty list input"""
        result = clamp_prompt_logprobs([])
        self.assertEqual(result, [])

    def test_normal_logprobs(self):
        """Test normal logprobs values (without -inf)"""
        logprob_dict = {
            1: Logprob(logprob=-2.5, rank=1, decoded_token="hello"),
            2: Logprob(logprob=-1.0, rank=2, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # Original values should remain unchanged
        self.assertEqual(result[0][1].logprob, -2.5)
        self.assertEqual(result[0][2].logprob, -1.0)

    def test_negative_inf_logprobs_raises_error(self):
        """Test that logprobs containing -inf raises AttributeError"""
        logprob_dict = {
            1: Logprob(logprob=float("-inf"), rank=1, decoded_token="hello"),
            2: Logprob(logprob=-1.0, rank=2, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict]

        # Since Logprob is a NamedTuple, its fields cannot be modified, should raise AttributeError
        with self.assertRaises(AttributeError) as context:
            clamp_prompt_logprobs(prompt_logprobs)

        self.assertIn("can't set attribute", str(context.exception))

    def test_multiple_negative_inf_raises_error(self):
        """Test that multiple -inf logprobs values raise AttributeError"""
        logprob_dict = {
            1: Logprob(logprob=float("-inf"), rank=1, decoded_token="hello"),
            2: Logprob(logprob=float("-inf"), rank=2, decoded_token="world"),
            3: Logprob(logprob=-0.5, rank=3, decoded_token="test"),
        }
        prompt_logprobs = [logprob_dict]

        # Since Logprob is a NamedTuple, its fields cannot be modified, should raise AttributeError
        with self.assertRaises(AttributeError):
            clamp_prompt_logprobs(prompt_logprobs)

    def test_none_dict_in_list(self):
        """Test case when list contains None"""
        prompt_logprobs = [None]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # None should be skipped
        self.assertIsNone(result[0])

    def test_multiple_dicts_normal_values(self):
        """Test multiple dictionaries case (without -inf)"""
        logprob_dict1 = {
            1: Logprob(logprob=-2.0, rank=1, decoded_token="hello"),
        }
        logprob_dict2 = {
            2: Logprob(logprob=-2.0, rank=1, decoded_token="world"),
        }
        prompt_logprobs = [logprob_dict1, logprob_dict2]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # Should return normally, values remain unchanged
        self.assertEqual(result[0][1].logprob, -2.0)
        self.assertEqual(result[1][2].logprob, -2.0)

    def test_mixed_values_without_inf(self):
        """Test mixed values case (without -inf)"""
        logprob_dict = {
            1: Logprob(logprob=-9999.0, rank=1, decoded_token="hello"),
            2: Logprob(logprob=-9999.0, rank=2, decoded_token="world"),
            3: Logprob(logprob=0.0, rank=3, decoded_token="test"),
            4: Logprob(logprob=-1.5, rank=4, decoded_token="again"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # All values should remain unchanged
        self.assertEqual(result[0][1].logprob, -9999.0)
        self.assertEqual(result[0][2].logprob, -9999.0)
        self.assertEqual(result[0][3].logprob, 0.0)
        self.assertEqual(result[0][4].logprob, -1.5)

    def test_return_same_object(self):
        """Test that function returns the same object (in-place modification attempt)"""
        logprob_dict = {
            1: Logprob(logprob=-2.0, rank=1, decoded_token="hello"),
        }
        prompt_logprobs = [logprob_dict]

        result = clamp_prompt_logprobs(prompt_logprobs)

        # Should return the same object (function attempts in-place modification)
        self.assertIs(result, prompt_logprobs)
        self.assertIs(result[0], prompt_logprobs[0])


if __name__ == "__main__":
    unittest.main()
