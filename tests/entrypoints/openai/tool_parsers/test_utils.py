"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import json
import unittest
from json import JSONDecodeError
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.tool_parsers.utils import (
    consume_space,
    extract_intermediate_diff,
    find_all_indices,
    find_common_prefix,
    find_common_suffix,
    is_complete_json,
    partial_json_loads,
)


class TestToolParsersUtils(unittest.TestCase):
    """Test case for tool parsers utility functions"""

    def test_find_common_prefix_identical_strings(self):
        """Test finding common prefix with identical strings"""
        s1 = "hello world"
        s2 = "hello world"
        result = find_common_prefix(s1, s2)
        self.assertEqual(result, "hello world")

    def test_find_common_prefix_different_strings(self):
        """Test finding common prefix with different strings"""
        s1 = "hello world"
        s2 = "hello universe"
        result = find_common_prefix(s1, s2)
        self.assertEqual(result, "hello ")

    def test_find_common_prefix_no_common_prefix(self):
        """Test finding common prefix with no common prefix"""
        s1 = "apple"
        s2 = "banana"
        result = find_common_prefix(s1, s2)
        self.assertEqual(result, "")

    def test_find_common_prefix_empty_strings(self):
        """Test finding common prefix with empty strings"""
        result = find_common_prefix("", "hello")
        self.assertEqual(result, "")
        
        result = find_common_prefix("hello", "")
        self.assertEqual(result, "")
        
        result = find_common_prefix("", "")
        self.assertEqual(result, "")

    def test_find_common_prefix_json_example(self):
        """Test finding common prefix with JSON example from docstring"""
        s1 = '{"fruit": "ap"}'
        s2 = '{"fruit": "apple"}'
        result = find_common_prefix(s1, s2)
        self.assertEqual(result, '{"fruit": "ap')

    def test_find_common_suffix_identical_strings(self):
        """Test finding common suffix with identical strings"""
        s1 = "hello world"
        s2 = "hello world"
        result = find_common_suffix(s1, s2)
        # Should stop at alphanumeric characters
        self.assertEqual(result, " ")

    def test_find_common_suffix_json_example(self):
        """Test finding common suffix with JSON example from docstring"""
        s1 = '{"fruit": "ap"}'
        s2 = '{"fruit": "apple"}'
        result = find_common_suffix(s1, s2)
        self.assertEqual(result, '"}')

    def test_find_common_suffix_no_common_suffix(self):
        """Test finding common suffix with no common suffix"""
        s1 = "apple123"
        s2 = "banana456"
        result = find_common_suffix(s1, s2)
        self.assertEqual(result, "")

    def test_find_common_suffix_punctuation_only(self):
        """Test finding common suffix stops at alphanumeric characters"""
        s1 = "test123!@#"
        s2 = "another123!@#"
        result = find_common_suffix(s1, s2)
        self.assertEqual(result, "!@#")

    def test_find_common_suffix_empty_strings(self):
        """Test finding common suffix with empty strings"""
        result = find_common_suffix("", "hello")
        self.assertEqual(result, "")
        
        result = find_common_suffix("hello", "")
        self.assertEqual(result, "")

    def test_extract_intermediate_diff_basic(self):
        """Test extracting intermediate diff with basic example"""
        curr = '{"fruit": "apple"}'
        old = '{"fruit": "ap"}'
        result = extract_intermediate_diff(curr, old)
        self.assertEqual(result, "ple")

    def test_extract_intermediate_diff_no_change(self):
        """Test extracting intermediate diff with no change"""
        curr = '{"fruit": "apple"}'
        old = '{"fruit": "apple"}'
        result = extract_intermediate_diff(curr, old)
        self.assertEqual(result, "")

    def test_extract_intermediate_diff_addition_only(self):
        """Test extracting intermediate diff with only addition"""
        curr = "hello world"
        old = "hello"
        result = extract_intermediate_diff(curr, old)
        self.assertEqual(result, " world")

    def test_extract_intermediate_diff_complex_json(self):
        """Test extracting intermediate diff with complex JSON"""
        curr = '{"name": "John", "age": 30}'
        old = '{"name": "John"}'
        result = extract_intermediate_diff(curr, old)
        self.assertEqual(result, ', "age": 30')

    def test_find_all_indices_single_occurrence(self):
        """Test finding all indices with single occurrence"""
        string = "hello world"
        substring = "world"
        result = find_all_indices(string, substring)
        self.assertEqual(result, [6])

    def test_find_all_indices_multiple_occurrences(self):
        """Test finding all indices with multiple occurrences"""
        string = "hello hello hello"
        substring = "hello"
        result = find_all_indices(string, substring)
        self.assertEqual(result, [0, 6, 12])

    def test_find_all_indices_no_occurrences(self):
        """Test finding all indices with no occurrences"""
        string = "hello world"
        substring = "python"
        result = find_all_indices(string, substring)
        self.assertEqual(result, [])

    def test_find_all_indices_overlapping(self):
        """Test finding all indices with overlapping substrings"""
        string = "aaaa"
        substring = "aa"
        result = find_all_indices(string, substring)
        self.assertEqual(result, [0, 1, 2])

    def test_find_all_indices_empty_substring(self):
        """Test finding all indices with empty substring"""
        string = "hello"
        substring = ""
        result = find_all_indices(string, substring)
        # Empty string should be found at every position
        self.assertEqual(result, [0, 1, 2, 3, 4, 5])

    def test_find_all_indices_empty_string(self):
        """Test finding all indices in empty string"""
        string = ""
        substring = "hello"
        result = find_all_indices(string, substring)
        self.assertEqual(result, [])

    @patch('fastdeploy.entrypoints.openai.tool_parsers.utils.partial_json_parser')
    def test_partial_json_loads_success(self, mock_partial_json_parser):
        """Test partial_json_loads with successful parsing"""
        mock_partial_json_parser.loads.return_value = {"key": "value"}
        
        # Mock Allow class
        mock_flags = MagicMock()
        
        result = partial_json_loads('{"key": "value"}', mock_flags)
        
        self.assertEqual(result, ({"key": "value"}, 15))
        mock_partial_json_parser.loads.assert_called_once_with('{"key": "value"}', mock_flags)

    @patch('fastdeploy.entrypoints.openai.tool_parsers.utils.partial_json_parser')
    @patch('fastdeploy.entrypoints.openai.tool_parsers.utils.JSONDecoder')
    def test_partial_json_loads_extra_data(self, mock_json_decoder, mock_partial_json_parser):
        """Test partial_json_loads with extra data error"""
        # Set up the mock to raise JSONDecodeError with "Extra data" message
        error = JSONDecodeError("Extra data", '{"key": "value"} extra', 15)
        error.msg = "Extra data: line 1 column 16 (char 15)"
        mock_partial_json_parser.loads.side_effect = error
        
        # Mock the JSONDecoder
        mock_decoder_instance = MagicMock()
        mock_decoder_instance.raw_decode.return_value = ({"key": "value"}, 15)
        mock_json_decoder.return_value = mock_decoder_instance
        
        mock_flags = MagicMock()
        result = partial_json_loads('{"key": "value"} extra', mock_flags)
        
        self.assertEqual(result, ({"key": "value"}, 15))
        mock_decoder_instance.raw_decode.assert_called_once_with('{"key": "value"} extra')

    @patch('fastdeploy.entrypoints.openai.tool_parsers.utils.partial_json_parser')
    def test_partial_json_loads_other_error(self, mock_partial_json_parser):
        """Test partial_json_loads with other JSON error"""
        error = JSONDecodeError("Invalid JSON", '{"key": invalid}', 7)
        error.msg = "Expecting value: line 1 column 8 (char 7)"
        mock_partial_json_parser.loads.side_effect = error
        
        mock_flags = MagicMock()
        
        with self.assertRaises(JSONDecodeError):
            partial_json_loads('{"key": invalid}', mock_flags)

    def test_is_complete_json_valid(self):
        """Test is_complete_json with valid JSON"""
        self.assertTrue(is_complete_json('{"key": "value"}'))
        self.assertTrue(is_complete_json('[]'))
        self.assertTrue(is_complete_json('null'))
        self.assertTrue(is_complete_json('true'))
        self.assertTrue(is_complete_json('123'))
        self.assertTrue(is_complete_json('"string"'))

    def test_is_complete_json_invalid(self):
        """Test is_complete_json with invalid JSON"""
        self.assertFalse(is_complete_json('{"key": "value"'))  # Missing closing brace
        self.assertFalse(is_complete_json('{"key": }'))        # Missing value
        self.assertFalse(is_complete_json('[1, 2,'))          # Incomplete array
        self.assertFalse(is_complete_json('invalid'))         # Not JSON
        self.assertFalse(is_complete_json(''))                # Empty string

    def test_consume_space_no_spaces(self):
        """Test consume_space with no spaces"""
        result = consume_space(0, "hello")
        self.assertEqual(result, 0)

    def test_consume_space_leading_spaces(self):
        """Test consume_space with leading spaces"""
        result = consume_space(0, "   hello")
        self.assertEqual(result, 3)

    def test_consume_space_middle_position(self):
        """Test consume_space starting from middle position"""
        result = consume_space(5, "hello   world")
        self.assertEqual(result, 8)

    def test_consume_space_mixed_whitespace(self):
        """Test consume_space with mixed whitespace"""
        result = consume_space(0, " \t\n\r hello")
        self.assertEqual(result, 5)

    def test_consume_space_all_spaces(self):
        """Test consume_space with all spaces"""
        result = consume_space(0, "   ")
        self.assertEqual(result, 3)

    def test_consume_space_end_of_string(self):
        """Test consume_space at end of string"""
        result = consume_space(5, "hello")
        self.assertEqual(result, 5)

    def test_consume_space_beyond_string(self):
        """Test consume_space beyond string length"""
        result = consume_space(10, "hello")
        self.assertEqual(result, 10)

    def test_consume_space_empty_string(self):
        """Test consume_space with empty string"""
        result = consume_space(0, "")
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()