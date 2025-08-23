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
from json import JSONDecodeError
from unittest.mock import patch


class TestStandaloneUtils(unittest.TestCase):
    """Standalone tests for utility functions without dependencies"""

    def test_find_common_prefix(self):
        """Test find_common_prefix function (copied from tool_parsers/utils.py)"""

        def find_common_prefix(s1: str, s2: str) -> str:
            """Find common prefix between two strings"""
            prefix = ""
            min_length = min(len(s1), len(s2))
            for i in range(0, min_length):
                if s1[i] == s2[i]:
                    prefix += s1[i]
                else:
                    break
            return prefix

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

    def test_find_common_suffix(self):
        """Test find_common_suffix function (copied from tool_parsers/utils.py)"""

        def find_common_suffix(s1: str, s2: str) -> str:
            """Find common suffix between two strings, stopping at alphanumeric"""
            suffix = ""
            min_length = min(len(s1), len(s2))
            for i in range(1, min_length + 1):
                if s1[-i] == s2[-i] and not s1[-i].isalnum():
                    suffix = s1[-i] + suffix
                else:
                    break
            return suffix

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

    def test_extract_intermediate_diff(self):
        """Test extract_intermediate_diff function"""

        def find_common_prefix(s1: str, s2: str) -> str:
            prefix = ""
            min_length = min(len(s1), len(s2))
            for i in range(0, min_length):
                if s1[i] == s2[i]:
                    prefix += s1[i]
                else:
                    break
            return prefix

        def find_common_suffix(s1: str, s2: str) -> str:
            suffix = ""
            min_length = min(len(s1), len(s2))
            for i in range(1, min_length + 1):
                if s1[-i] == s2[-i] and not s1[-i].isalnum():
                    suffix = s1[-i] + suffix
                else:
                    break
            return suffix

        def extract_intermediate_diff(curr: str, old: str) -> str:
            """Extract difference between two strings with common prefix/suffix"""
            suffix = find_common_suffix(curr, old)
            old = old[::-1].replace(suffix[::-1], "", 1)[::-1]
            prefix = find_common_prefix(curr, old)
            diff = curr
            if len(suffix):
                diff = diff[::-1].replace(suffix[::-1], "", 1)[::-1]
            if len(prefix):
                diff = diff.replace(prefix, "", 1)
            return diff

        # Test basic case
        result = extract_intermediate_diff('{"fruit": "apple"}', '{"fruit": "ap"}')
        self.assertEqual(result, "ple")

        # Test with no difference
        result = extract_intermediate_diff("same", "same")
        self.assertEqual(result, "")

    def test_find_all_indices(self):
        """Test find_all_indices function"""

        def find_all_indices(string: str, substring: str) -> list[int]:
            """Find all starting indices of substring in string"""
            indices = []
            index = -1
            while True:
                index = string.find(substring, index + 1)
                if index == -1:
                    break
                indices.append(index)
            return indices

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

    def test_is_complete_json(self):
        """Test is_complete_json function"""

        def is_complete_json(input_str: str) -> bool:
            """Check if string is complete valid JSON"""
            try:
                json.loads(input_str)
                return True
            except JSONDecodeError:
                return False

        # Test valid JSON
        self.assertTrue(is_complete_json('{"key": "value"}'))
        self.assertTrue(is_complete_json("[]"))
        self.assertTrue(is_complete_json("null"))
        self.assertTrue(is_complete_json("true"))
        self.assertTrue(is_complete_json("123"))

        # Test invalid JSON
        self.assertFalse(is_complete_json('{"key": "value"'))
        self.assertFalse(is_complete_json('{"key":}'))
        self.assertFalse(is_complete_json("[1,2,"))
        self.assertFalse(is_complete_json(""))

    def test_consume_space(self):
        """Test consume_space function"""

        def consume_space(i: int, s: str) -> int:
            """Consume whitespace starting from index i"""
            while i < len(s) and s[i].isspace():
                i += 1
            return i

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

    def test_partial_json_loads_simulation(self):
        """Test partial JSON loading simulation"""

        def partial_json_loads_mock(input_str: str) -> tuple:
            """Mock partial JSON loader"""
            try:
                # Try standard JSON first
                obj = json.loads(input_str)
                return (obj, len(input_str))
            except JSONDecodeError as e:
                if "Extra data" in str(e):
                    # Simulate raw_decode behavior
                    decoder = json.JSONDecoder()
                    try:
                        return decoder.raw_decode(input_str)
                    except JSONDecodeError:
                        raise
                else:
                    raise

        # Test complete JSON
        result = partial_json_loads_mock('{"key": "value"}')
        self.assertEqual(result[0], {"key": "value"})
        self.assertEqual(result[1], 15)

        # Test JSON with extra data - raw_decode returns position after valid JSON
        input_with_extra = '{"key": "value"} extra'
        result = partial_json_loads_mock(input_with_extra)
        self.assertEqual(result[0], {"key": "value"})
        # raw_decode returns the end position of valid JSON, which includes the space
        self.assertEqual(result[1], 16)

    def test_random_tool_call_id_generation(self):
        """Test tool call ID generation pattern"""

        import uuid

        def random_tool_call_id() -> str:
            """Generate a random tool call ID"""
            return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"

        # Test ID generation
        tool_id = random_tool_call_id()
        self.assertTrue(tool_id.startswith("chatcmpl-tool-"))
        self.assertEqual(len(tool_id), len("chatcmpl-tool-") + 32)  # UUID hex is 32 chars

        # Test uniqueness
        tool_id1 = random_tool_call_id()
        tool_id2 = random_tool_call_id()
        self.assertNotEqual(tool_id1, tool_id2)

    def test_message_role_validation(self):
        """Test message role validation patterns"""

        def validate_message_role(role: str, valid_roles: list) -> bool:
            """Validate if role is in valid roles"""
            return role in valid_roles

        valid_roles = ["user", "assistant", "system", "tool"]

        # Test valid roles
        self.assertTrue(validate_message_role("user", valid_roles))
        self.assertTrue(validate_message_role("assistant", valid_roles))
        self.assertTrue(validate_message_role("system", valid_roles))
        self.assertTrue(validate_message_role("tool", valid_roles))

        # Test invalid role
        self.assertFalse(validate_message_role("invalid", valid_roles))
        self.assertFalse(validate_message_role("", valid_roles))

    def test_token_encoding_patterns(self):
        """Test token encoding patterns from serving modules"""

        def simulate_token_processing(prompt_text: str, mock_tokenizer):
            """Simulate token processing"""
            # Mock tokenization
            tokens = prompt_text.split()  # Simple word splitting for test
            token_ids = [hash(token) % 1000 for token in tokens]  # Mock token IDs
            return tokens, token_ids

        class MockTokenizer:
            def tokenize(self, text):
                return text.split()

            def convert_tokens_to_ids(self, tokens):
                return [hash(token) % 1000 for token in tokens]

        mock_tokenizer = MockTokenizer()
        tokens, token_ids = simulate_token_processing("Hello world", mock_tokenizer)

        self.assertEqual(tokens, ["Hello", "world"])
        self.assertEqual(len(token_ids), 2)
        self.assertIsInstance(token_ids[0], int)
        self.assertIsInstance(token_ids[1], int)

    def test_error_response_formatting(self):
        """Test error response formatting patterns"""

        def format_error_response(error_msg: str, error_code: int = 400) -> dict:
            """Format error response"""
            return {"error": {"message": error_msg, "code": error_code, "type": "invalid_request_error"}}

        # Test error formatting
        error_resp = format_error_response("Invalid input", 400)
        self.assertEqual(error_resp["error"]["message"], "Invalid input")
        self.assertEqual(error_resp["error"]["code"], 400)
        self.assertEqual(error_resp["error"]["type"], "invalid_request_error")

        # Test with different code
        error_resp2 = format_error_response("Server error", 500)
        self.assertEqual(error_resp2["error"]["code"], 500)


if __name__ == "__main__":
    unittest.main()
