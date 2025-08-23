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

import json
import unittest
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


class TestToolParserUtils(unittest.TestCase):
    """Unit tests for tool parser utility functions"""

    def test_find_common_prefix(self):
        """Test find_common_prefix function"""
        # Test with common prefix
        result = find_common_prefix('{"fruit": "ap"}', '{"fruit": "apple"}')
        self.assertEqual(result, '{"fruit": "ap')

        # Test with no common prefix
        result = find_common_prefix("hello", "world")
        self.assertEqual(result, "")

        # Test with identical strings
        result = find_common_prefix("test", "test")
        self.assertEqual(result, "test")

        # Test with empty strings
        result = find_common_prefix("", "test")
        self.assertEqual(result, "")

        # Test with one empty string
        result = find_common_prefix("test", "")
        self.assertEqual(result, "")

    def test_find_common_suffix(self):
        """Test find_common_suffix function"""
        # Test with common suffix
        result = find_common_suffix('{"fruit": "ap"}', '{"fruit": "apple"}')
        self.assertEqual(result, '"}')

        # Test with no common suffix
        result = find_common_suffix("hello", "world")
        self.assertEqual(result, "")

        # Test with alphanumeric characters (should stop)
        result = find_common_suffix("test123", "best123")
        self.assertEqual(result, "")

        # Test with non-alphanumeric suffix
        result = find_common_suffix("test!!!", "best!!!")
        self.assertEqual(result, "!!!")

        # Test with identical strings
        result = find_common_suffix("}]", "}]")
        self.assertEqual(result, "}]")

    def test_extract_intermediate_diff(self):
        """Test extract_intermediate_diff function"""
        # Test basic case
        result = extract_intermediate_diff('{"fruit": "apple"}', '{"fruit": "ap"}')
        self.assertEqual(result, "ple")

        # Test with no difference
        result = extract_intermediate_diff("same", "same")
        self.assertEqual(result, "")

        # Test with complex JSON
        curr = '{"name": "John", "age": 30}'
        old = '{"name": "John", "age":'
        result = extract_intermediate_diff(curr, old)
        self.assertEqual(result, " 30")

    def test_find_all_indices(self):
        """Test find_all_indices function"""
        # Test multiple occurrences
        result = find_all_indices("hello world hello", "hello")
        self.assertEqual(result, [0, 12])

        # Test single occurrence
        result = find_all_indices("find me", "me")
        self.assertEqual(result, [5])

        # Test no occurrences
        result = find_all_indices("nothing here", "xyz")
        self.assertEqual(result, [])

        # Test overlapping patterns
        result = find_all_indices("aaaa", "aa")
        self.assertEqual(result, [0, 1, 2])

        # Test empty string
        result = find_all_indices("", "test")
        self.assertEqual(result, [])

    @patch("partial_json_parser.loads")
    def test_partial_json_loads_success(self, mock_loads):
        """Test partial_json_loads successful parsing"""
        mock_loads.return_value = {"key": "value"}

        result = partial_json_loads('{"key": "value"}', None)
        self.assertEqual(result, ({"key": "value"}, 15))
        mock_loads.assert_called_once()

    @patch("partial_json_parser.loads")
    def test_partial_json_loads_extra_data(self, mock_loads):
        """Test partial_json_loads with extra data error"""
        from json import JSONDecodeError

        # Mock partial_json_parser to raise JSONDecodeError with "Extra data"
        mock_loads.side_effect = JSONDecodeError("Extra data", '{"key": "value"} extra', 15)

        # This should fall back to JSONDecoder.raw_decode
        result = partial_json_loads('{"key": "value"} extra', None)
        self.assertEqual(result[0], {"key": "value"})
        self.assertEqual(result[1], 15)

    @patch("partial_json_parser.loads")
    def test_partial_json_loads_other_error(self, mock_loads):
        """Test partial_json_loads with other JSONDecodeError"""
        from json import JSONDecodeError

        # Mock to raise a different JSONDecodeError
        mock_loads.side_effect = JSONDecodeError("Invalid JSON", '{"key":}', 7)

        with self.assertRaises(JSONDecodeError):
            partial_json_loads('{"key":}', None)

    def test_is_complete_json_valid(self):
        """Test is_complete_json with valid JSON"""
        self.assertTrue(is_complete_json('{"key": "value"}'))
        self.assertTrue(is_complete_json("[]"))
        self.assertTrue(is_complete_json("null"))
        self.assertTrue(is_complete_json("true"))
        self.assertTrue(is_complete_json("123"))

    def test_is_complete_json_invalid(self):
        """Test is_complete_json with invalid JSON"""
        self.assertFalse(is_complete_json('{"key": "value"'))
        self.assertFalse(is_complete_json('{"key":}'))
        self.assertFalse(is_complete_json("[1,2,"))
        self.assertFalse(is_complete_json(""))

    def test_consume_space(self):
        """Test consume_space function"""
        # Test with leading spaces
        result = consume_space(0, "   hello")
        self.assertEqual(result, 3)

        # Test with tabs and newlines
        result = consume_space(0, "\t\n  test")
        self.assertEqual(result, 4)

        # Test starting from middle
        result = consume_space(2, "ab   def")
        self.assertEqual(result, 5)

        # Test no spaces
        result = consume_space(0, "nospaceshere")
        self.assertEqual(result, 0)

        # Test at end of string
        result = consume_space(5, "hello")
        self.assertEqual(result, 5)

        # Test empty string
        result = consume_space(0, "")
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
