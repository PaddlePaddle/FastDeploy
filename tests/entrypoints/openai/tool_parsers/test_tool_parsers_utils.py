"""
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

import unittest

from partial_json_parser.core.options import Allow

from fastdeploy.entrypoints.openai.tool_parsers import utils


class TestPartialJsonUtils(unittest.TestCase):
    """Unit test suite for partial JSON utility functions."""

    def test_find_common_prefix(self):
        """Test common prefix detection between two strings."""
        string1 = '{"fruit": "ap"}'
        string2 = '{"fruit": "apple"}'
        self.assertEqual(utils.find_common_prefix(string1, string2), '{"fruit": "ap')

    def test_find_common_suffix(self):
        """Test common suffix detection between two strings."""
        string1 = '{"fruit": "ap"}'
        string2 = '{"fruit": "apple"}'
        self.assertEqual(utils.find_common_suffix(string1, string2), '"}')

    def test_extract_intermediate_diff(self):
        """Test extraction of intermediate difference between current and old strings."""
        old_string = '{"fruit": "ap"}'
        current_string = '{"fruit": "apple"}'
        self.assertEqual(utils.extract_intermediate_diff(current_string, old_string), "ple")

    def test_find_all_indices(self):
        """Test finding all occurrence indices of a substring in a string."""
        target_string = "banana"
        substring = "an"
        self.assertEqual(utils.find_all_indices(target_string, substring), [1, 3])

    def test_partial_json_loads_complete(self):
        """Test partial_json_loads with a complete JSON string."""
        input_json = '{"a": 1, "b": 2}'
        parse_flags = Allow.ALL
        parsed_obj, parsed_length = utils.partial_json_loads(input_json, parse_flags)
        self.assertEqual(parsed_obj, {"a": 1, "b": 2})
        self.assertEqual(parsed_length, len(input_json))

    def test_is_complete_json(self):
        """Test JSON completeness check."""
        self.assertTrue(utils.is_complete_json('{"a": 1}'))
        self.assertFalse(utils.is_complete_json('{"a": 1'))

    def test_consume_space(self):
        """Test whitespace consumption from the start of a string."""
        input_string = "   \t\nabc"
        # 3 spaces + 1 tab + 1 newline = 5 whitespace characters
        first_non_whitespace_idx = utils.consume_space(0, input_string)
        self.assertEqual(first_non_whitespace_idx, 5)


if __name__ == "__main__":
    unittest.main()
