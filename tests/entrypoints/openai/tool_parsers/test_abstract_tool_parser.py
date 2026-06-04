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

from fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser


class _DummyTokenizer:
    def get_vocab(self):
        return {}


class _PairedTagParser(ToolParser):
    """A concrete parser declaring paired sentinel tokens for testing."""

    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"


class _NoSentinelParser(ToolParser):
    """A parser that did not opt in to prefix detection."""


class TestDetectToolPrefix(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _DummyTokenizer()
        self.parser = _PairedTagParser(self.tokenizer)

    def test_initial_state(self):
        self.assertEqual(self.parser._tool_prefix, "")
        self.assertFalse(self.parser._tool_prefix_computed)
        self.assertFalse(self.parser._tool_prefix_injected_to_delta)

    def test_empty_prompt_returns_empty(self):
        self.assertEqual(self.parser.detect_tool_prefix(""), "")

    def test_no_start_token_returns_empty(self):
        self.assertEqual(
            self.parser.detect_tool_prefix("user: hello\nassistant: hi"),
            "",
        )

    def test_parser_without_sentinel_returns_empty(self):
        parser = _NoSentinelParser(self.tokenizer)
        self.assertEqual(
            parser.detect_tool_prefix("anything <tool_call> here"),
            "",
        )

    def test_trailing_start_token_only(self):
        prompt = "user: q\n<tool_call>"
        self.assertEqual(self.parser.detect_tool_prefix(prompt), "<tool_call>")

    def test_trailing_start_with_invoke_prefix(self):
        prompt = "history\n<tool_call><invoke name="
        self.assertEqual(
            self.parser.detect_tool_prefix(prompt),
            "<tool_call><invoke name=",
        )

    def test_history_closed_tool_call_no_injection(self):
        prompt = "<tool_call>{...}</tool_call>\nuser: next"
        self.assertEqual(self.parser.detect_tool_prefix(prompt), "")

    def test_history_closed_plus_new_injected_prefix(self):
        prompt = "<tool_call>{a:1}</tool_call>\n<tool_call><invoke name="
        self.assertEqual(
            self.parser.detect_tool_prefix(prompt),
            "<tool_call><invoke name=",
        )

    def test_multiple_closed_history_no_injection(self):
        prompt = "<tool_call>{a:1}</tool_call>\n" "<tool_call>{b:2}</tool_call>\n" "assistant: done"
        self.assertEqual(self.parser.detect_tool_prefix(prompt), "")

    def test_trailing_whitespace_after_start(self):
        prompt = "history\n<tool_call>   "
        self.assertEqual(
            self.parser.detect_tool_prefix(prompt),
            "<tool_call>   ",
        )


if __name__ == "__main__":
    unittest.main()
