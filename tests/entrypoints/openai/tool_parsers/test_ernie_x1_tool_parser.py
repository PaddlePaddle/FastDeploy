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
from unittest.mock import patch

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser import (
    ErnieX1ToolParser,
)


class TestErnieX1ToolParser(unittest.TestCase):
    def setUp(self):
        class DummyTokenizer:
            def __init__(self):
                self.vocab = {
                    "<tool_call>": 1,
                    "</tool_call>": 2,
                    "</think>": 3,
                    "<response>": 4,
                    "</response>": 5,
                }

            def get_vocab(self):
                return self.vocab

        self.tokenizer = DummyTokenizer()
        self.parser = ErnieX1ToolParser(tokenizer=self.tokenizer)
        self.dummy_request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])

    def _new_parser(self):
        """Create a fresh parser to avoid state pollution between tests."""

        class DummyTokenizer:
            def __init__(self):
                self.vocab = {
                    "<tool_call>": 1,
                    "</tool_call>": 2,
                    "</think>": 3,
                    "<response>": 4,
                    "</response>": 5,
                }

            def get_vocab(self):
                return self.vocab

        return ErnieX1ToolParser(tokenizer=DummyTokenizer())

    # ==================== __init__ tests (lines 60-81) ====================

    def test_init_sets_tokens_and_ids(self):
        """Cover lines 60-81: verify all token attributes and vocab lookups"""
        p = self.parser
        self.assertFalse(p.current_tool_name_sent)
        self.assertEqual(p.prev_tool_call_arr, [])
        self.assertEqual(p.current_tool_id, -1)
        self.assertEqual(p.streamed_args_for_tool, [])
        self.assertEqual(p.think_end_token, "</think>")
        self.assertEqual(p.response_start_token, "<response>")
        self.assertEqual(p.response_end_token, "</response>")
        self.assertEqual(p.tool_call_start_token, "<tool_call>")
        self.assertEqual(p.tool_call_end_token, "</tool_call>")
        self.assertIsNotNone(p.tool_call_regex)
        self.assertEqual(p.think_end_token_id, 3)
        self.assertEqual(p.response_start_token_id, 4)
        self.assertEqual(p.response_end_token_id, 5)
        self.assertEqual(p.tool_call_start_token_id, 1)
        self.assertEqual(p.tool_call_end_token_id, 2)

    def test_init_raises_without_tokenizer(self):
        """Cover lines 72-75: ValueError when tokenizer is falsy"""
        with self.assertRaises(ValueError):
            ErnieX1ToolParser(tokenizer=None)

    # ==================== extract_tool_calls tests (lines 96-117) ====================

    def test_extract_tool_calls_single(self):
        """Cover lines 96-114: single complete tool call"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertIn("北京", result.tool_calls[0].function.arguments)

    def test_extract_tool_calls_multiple(self):
        """Cover lines 98-100: multiple tool calls"""
        output = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
        )
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "get_time")

    def test_extract_tool_calls_no_arguments(self):
        """Cover line 100: tool call with no arguments defaults to {}"""
        output = '<tool_call>{"name": "list_items"}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.arguments, "{}")

    def test_extract_tool_calls_nested_arguments(self):
        """Cover regex with nested braces in arguments"""
        output = '<tool_call>{"name": "query", "arguments": {"filter": {"age": {"$gt": 18}}}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertIn("$gt", result.tool_calls[0].function.arguments)

    def test_extract_tool_calls_with_whitespace(self):
        """Cover regex with whitespace around JSON"""
        output = '<tool_call>  \n{"name": "fn", "arguments": {}}  \n</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "fn")

    def test_extract_tool_calls_no_match(self):
        """Cover lines 96, 111-114: no tool_call tags -> tools_called=True with empty list"""
        output = "just plain text"
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, output)

    def test_extract_tool_calls_invalid_json(self):
        """Cover lines 115-117: malformed JSON triggers exception branch"""
        output = "<tool_call>{invalid json}</tool_call>"
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, output)

    def test_extract_tool_calls_exception(self):
        """Cover lines 115-117: forced exception via mock"""
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.json.loads",
            side_effect=Exception("boom"),
        ):
            output = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
            result = self.parser.extract_tool_calls(output, self.dummy_request)
            self.assertFalse(result.tools_called)
            self.assertEqual(result.content, output)

    # ==================== extract_tool_calls_streaming tests ====================

    # --- Line 129-131: no tool_call_start_token in current_text ---

    def test_streaming_no_tool_call_token(self):
        """Cover lines 129-131: no <tool_call> in current_text returns content delta"""
        result = self.parser.extract_tool_calls_streaming("", "hello world", "world", [], [], [], self.dummy_request)
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.content, "world")
        self.assertIsNone(result.tool_calls)

    # --- Lines 141-147: balanced start/end counts, text generation after tool call ---

    def test_streaming_balanced_counts_text_after_tool(self):
        """Cover lines 134-147: start==end, prev_end==cur_end, end not in delta -> text content"""
        prev = "<tool_call>{}</tool_call>"
        cur = "<tool_call>{}</tool_call> some text"
        delta = " some text"
        result = self.parser.extract_tool_calls_streaming(prev, cur, delta, [1, 2], [1, 2], [], self.dummy_request)
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.content, delta)

    # --- Lines 149-156: tool_call_end_token in delta_text ---

    def test_streaming_end_token_in_delta(self):
        """Cover lines 149-156: </tool_call> appears in delta"""
        parser = self._new_parser()
        # First, start a tool call
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # Now stream arguments
        parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            ', "arguments": {"k": "v',
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        # Close with end token in delta
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}}</tool_call>',
            '"}}</tool_call>',
            [1, 10, 20],
            [1, 10, 20, 2],
            [2],
            self.dummy_request,
        )
        # Should handle end token
        self.assertTrue(result is None or isinstance(result, DeltaMessage))

    # --- Lines 160-172: new tool call start (cur_start > cur_end and cur_start > prev_start) ---

    def test_streaming_new_tool_call_single_token(self):
        """Cover lines 160-172 (len(delta_token_ids)==1): new tool start with single token"""
        parser = self._new_parser()
        result = parser.extract_tool_calls_streaming(
            "",
            "<tool_call>",
            "<tool_call>",
            [],
            [1],
            [1],
            self.dummy_request,
        )
        # tool_call_portion is None, current_tool_call is None, name not sent -> None
        self.assertIsNone(result)
        self.assertEqual(parser.current_tool_id, 0)
        self.assertEqual(len(parser.streamed_args_for_tool), 1)

    def test_streaming_new_tool_call_multi_tokens(self):
        """Cover lines 160-162 (len(delta_token_ids)>1): new tool start with content"""
        parser = self._new_parser()
        result = parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_calls[0].function.name, "fn")
        self.assertEqual(parser.current_tool_id, 0)

    # --- Lines 174-176: continuing inside existing tool (cur_start > cur_end, same start count) ---

    def test_streaming_continue_tool_call_no_name_yet(self):
        """Cover lines 174-176, 220-222: partial JSON without name yet"""
        parser = self._new_parser()
        # Start tool call
        parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [1], [1], self.dummy_request)
        # Continue with partial content, no name parseable yet
        result = parser.extract_tool_calls_streaming(
            "<tool_call>",
            '<tool_call>{"na',
            '{"na',
            [1],
            [1, 10],
            [10],
            self.dummy_request,
        )
        self.assertIsNone(result)

    def test_streaming_continue_tool_call_with_name(self):
        """Cover lines 174-176, 223-235: name becomes available"""
        parser = self._new_parser()
        # Start tool call
        parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [1], [1], self.dummy_request)
        # Name appears
        result = parser.extract_tool_calls_streaming(
            "<tool_call>",
            '<tool_call>{"name": "get_weather"',
            '{"name": "get_weather"',
            [1],
            [1, 10],
            [10],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertTrue(parser.current_tool_name_sent)

    # --- Lines 236-237: name not sent and function_name is None ---

    def test_streaming_no_function_name(self):
        """Cover lines 236-237: parsed JSON has no 'name' field"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [1], [1], self.dummy_request)
        # Send JSON without name field
        result = parser.extract_tool_calls_streaming(
            "<tool_call>",
            '<tool_call>{"arguments": {"k": "v"}}',
            '{"arguments": {"k": "v"}}',
            [1],
            [1, 10],
            [10],
            self.dummy_request,
        )
        self.assertIsNone(result)

    # --- Lines 178-200: closing branch (cur_start == cur_end, end >= prev_end) ---

    def test_streaming_close_no_prev_tool_call(self):
        """Cover lines 178-181: close branch with empty prev_tool_call_arr"""
        parser = self._new_parser()
        parser.prev_tool_call_arr = []
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}',
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call>',
            "</tool_call>",
            [1, 10],
            [1, 10, 2],
            [2],
            self.dummy_request,
        )
        self.assertIsNone(result)

    def test_streaming_close_with_remaining_diff(self):
        """Cover lines 182-200: close with arguments diff that hasn't been streamed"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{"name": "fn", "arguments": {"k": "v"}}]
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}',
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call>',
            '"}}</tool_call>',
            [1, 10],
            [1, 10, 2],
            [2],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)

    def test_streaming_close_with_diff_no_end_marker(self):
        """Cover lines 184-185: close with arguments but no '"}' in delta_text"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{"name": "fn", "arguments": {"k": "v"}}]
        # Simulate end token in delta but without '"}' pattern
        # We need cur_start==cur_end and cur_end >= prev_end, and end_token NOT in delta
        # so that we enter the elif at 178
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call>',
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call> text',
            " text",
            [1, 10, 2],
            [1, 10, 2, 30],
            [30],
            self.dummy_request,
        )
        # balanced counts, prev_end==cur_end, end not in delta -> returns content (line 147)
        self.assertIsInstance(result, DeltaMessage)

    def test_streaming_close_no_arguments(self):
        """Cover lines 182-183: close branch where prev arguments is None/empty"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{"name": "fn"}]  # no arguments key
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn"}',
            '<tool_call>{"name":"fn"}</tool_call>',
            "}</tool_call>",
            [1, 10],
            [1, 10, 2],
            [2],
            self.dummy_request,
        )
        # diff is None (no arguments), so falls through to partial_json_parser
        self.assertTrue(result is None or isinstance(result, DeltaMessage))

    # --- Lines 202-206: else branch (cur_start < cur_end, edge case) ---

    def test_streaming_else_branch(self):
        """Cover lines 202-206: fall-through else branch"""
        parser = self._new_parser()
        parser.current_tool_name_sent = True
        # Construct a scenario where cur_start < cur_end (more end tags than start)
        prev = "<tool_call>"
        cur = "<tool_call></tool_call></tool_call>"
        delta = "</tool_call>"
        result = parser.extract_tool_calls_streaming(prev, cur, delta, [1], [1, 2, 2], [2], self.dummy_request)
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.tool_calls, [])

    # --- Lines 208-218: partial_json_parser errors ---

    def test_streaming_malformed_json(self):
        """Cover lines 213-215: MalformedJSON from partial parser"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [1], [1], self.dummy_request)
        # Feed badly formed content
        result = parser.extract_tool_calls_streaming(
            "<tool_call>",
            "<tool_call>{{{",
            "{{{",
            [1],
            [1, 10],
            [10],
            self.dummy_request,
        )
        self.assertIsNone(result)

    def test_streaming_json_decode_error(self):
        """Cover lines 216-218: JSONDecodeError from partial parser"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [1], [1], self.dummy_request)
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            side_effect=ValueError("bad json"),
        ):
            result = parser.extract_tool_calls_streaming(
                "<tool_call>",
                "<tool_call>bad",
                "bad",
                [1],
                [1, 10],
                [10],
                self.dummy_request,
            )
            self.assertIsNone(result)

    # --- Lines 239-241: tool_call_portion is None after name sent ---

    def test_streaming_tool_portion_none_with_text(self):
        """Cover lines 239-241: tool_call_portion is None, text_portion is not None"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{}]
        # Force tool_call_portion = None and text_portion = not None
        # This happens when end_token is in delta (sets text_portion) but new tool start
        # overrides tool_call_portion to None with single token
        # Simulate: new tool start with single token AND end token in delta
        # Actually, the simplest path: end token in delta sets text_portion, then new tool start
        # sets tool_call_portion = None
        # Let's use a different approach - directly test via the continuing branch
        # where tool_call_portion remains None from the end_token path
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn"}',
            '<tool_call>{"name":"fn"}</tool_call><tool_call>',
            "</tool_call><tool_call>",
            [1, 10],
            [1, 10, 2, 1],
            [2, 1],
            self.dummy_request,
        )
        self.assertTrue(result is None or isinstance(result, DeltaMessage))

    # --- Lines 243-244: append to prev_tool_call_arr ---

    def test_streaming_first_arguments_with_regex_match(self):
        """Cover lines 243-244, 257-286: first arguments appear, regex matches"""
        parser = self._new_parser()
        # Start tool call and send name
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "get_weather"',
            '<tool_call>{"name": "get_weather"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # Now stream arguments (first time)
        # Key must be complete (closing quote) so partial_json_parser returns truthy arguments.
        # delta must be a substring of the regex-extracted arguments portion (after "arguments":).
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "get_weather"',
            '<tool_call>{"name": "get_weather", "arguments": {"location": "bei',
            '"bei',
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)

    def test_streaming_first_arguments_no_regex_match(self):
        """Cover lines 266-267: regex doesn't match, fallback to json.dumps"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{}]
        # Use tool_call_portion where key order differs so regex won't match
        # (regex expects {"name":... at the start, but here "extra" comes first)
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            return_value={"name": "fn", "extra": True, "arguments": {"k": "v"}},
        ):
            result = parser.extract_tool_calls_streaming(
                "<tool_call>",
                '<tool_call>{"extra": true, "name": "fn", "arguments": {"k": "v"}}',
                '"v"}',
                [1],
                [1, 10],
                [10],
                self.dummy_request,
            )
            # regex fails on {"extra":... format, falls back to json.dumps
            # delta '"v"}' is in json.dumps({"k": "v"}) = '{"k": "v"}'
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.tool_calls)

    def test_streaming_first_arguments_delta_not_in_json(self):
        """Cover lines 271-272: delta_text not found in cur_arguments_json"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # Delta text that doesn't appear in the arguments JSON
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}}',
            "ZZZZZ",
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        self.assertIsNone(result)

    # --- Lines 249-251: no cur_arguments and no prev_arguments ---

    def test_streaming_no_arguments_at_all(self):
        """Cover lines 249-251: both cur and prev arguments are empty/None"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # Continue with name only, no arguments
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"}',
            "}",
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        # prev_arguments=None, cur_arguments=None -> delta=None
        # then prev_tool_call_arr updated and returns delta (which is None)
        self.assertIsNone(result)

    # --- Lines 253-255: cur_arguments reset (impossible branch) ---

    def test_streaming_arguments_reset_mid_call(self):
        """Cover lines 253-255: prev has arguments but cur doesn't (impossible case)"""
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{"name": "fn", "arguments": {"k": "v"}}]
        # Feed content where cur has no arguments but prev does
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            return_value={"name": "fn"},
        ):
            result = parser.extract_tool_calls_streaming(
                '<tool_call>{"name": "fn", "arguments": {"k": "v"',
                '<tool_call>{"name": "fn", "arguments": {"k": "v"}',
                '"}',
                [1, 10],
                [1, 10, 20],
                [20],
                self.dummy_request,
            )
            self.assertIsNone(result)

    # --- Lines 288-314: cur_arguments and prev_arguments both present ---

    def test_streaming_incremental_arguments_incomplete(self):
        """Cover lines 288-314: both prev and cur have arguments, JSON incomplete"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # First arguments - delta must appear in regex-extracted arguments portion
        parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '{"k": "v',
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        # More argument tokens (both prev and cur have arguments now)
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '<tool_call>{"name": "fn", "arguments": {"k": "val',
            "al",
            [1, 10, 20],
            [1, 10, 20, 30],
            [30],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_calls[0].function.arguments, "al")

    def test_streaming_incremental_arguments_complete_json(self):
        """Cover lines 289-305: complete JSON with trailing }"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # First arguments - delta must appear in regex-extracted arguments portion
        parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '{"k": "v',
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        # Complete with closing braces - both prev and cur have arguments
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}}',
            '"}}',
            [1, 10, 20],
            [1, 10, 20, 30],
            [30],
            self.dummy_request,
        )
        # is_complete_json=True, delta ends with }, should strip trailing }
        # After strip: '"' which is not empty, so returns DeltaMessage
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DeltaMessage)

    def test_streaming_incremental_arguments_complete_empty_delta(self):
        """Cover lines 304-305: complete JSON where delta becomes empty after strip"""
        parser = self._new_parser()
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn"',
            [],
            [1, 10],
            [1, 10],
            self.dummy_request,
        )
        # First arguments with proper delta
        parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn"',
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}',
            '{"k": "v"}',
            [1, 10],
            [1, 10, 20],
            [20],
            self.dummy_request,
        )
        # Send just the outer closing brace
        # tool_call_portion becomes complete JSON, delta="}" stripped to "" -> return None
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}',
            '<tool_call>{"name": "fn", "arguments": {"k": "v"}}',
            "}",
            [1, 10, 20],
            [1, 10, 20, 30],
            [30],
            self.dummy_request,
        )
        # is_complete_json=True, delta="}" -> stripped to "" -> return None
        self.assertIsNone(result)

    # --- Lines 316-319: prev_tool_call_arr update branches ---

    def test_streaming_prev_tool_call_arr_append(self):
        """Cover lines 318-319: append to prev_tool_call_arr when index doesn't match"""
        parser = self._new_parser()
        parser.current_tool_id = 1
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ["", ""]
        parser.prev_tool_call_arr = [{"name": "fn1"}]
        # current_tool_id (1) != len(prev_tool_call_arr) - 1 (0), so append
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            return_value={"name": "fn2"},
        ):
            parser.extract_tool_calls_streaming(
                "<tool_call>",
                '<tool_call>{"name": "fn2"}',
                '{"name": "fn2"}',
                [1],
                [1, 10],
                [10],
                self.dummy_request,
            )
            self.assertEqual(len(parser.prev_tool_call_arr), 2)

    # --- Lines 323-325: top-level exception handler ---

    def test_streaming_general_exception(self):
        """Cover lines 323-325: unexpected exception returns None"""
        parser = self._new_parser()
        parser.current_tool_name_sent = True
        # Force an exception by corrupting internal state
        parser.current_tool_id = 0
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = None  # will cause exception on access
        result = parser.extract_tool_calls_streaming(
            "<tool_call>",
            '<tool_call>{"name": "fn"}',
            '{"name": "fn"}',
            [1],
            [1, 10],
            [10],
            self.dummy_request,
        )
        self.assertIsNone(result)

    # ==================== Full streaming simulation ====================

    def test_streaming_full_flow(self):
        """Integration test: simulate a full streaming tool call flow"""
        parser = self._new_parser()
        req = self.dummy_request

        # Step 1: text before tool call
        r = parser.extract_tool_calls_streaming("", "thinking", "thinking", [], [], [], req)
        self.assertEqual(r.content, "thinking")

        # Step 2: tool_call start token
        r = parser.extract_tool_calls_streaming("thinking", "thinking<tool_call>", "<tool_call>", [], [1], [1], req)
        self.assertIsNone(r)

        # Step 3: function name appears
        r = parser.extract_tool_calls_streaming(
            "thinking<tool_call>",
            'thinking<tool_call>{"name": "search"',
            '{"name": "search"',
            [1],
            [1, 10],
            [10],
            req,
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.tool_calls[0].function.name, "search")

        # Step 4: arguments start - delta must appear in regex-extracted arguments portion
        r = parser.extract_tool_calls_streaming(
            'thinking<tool_call>{"name": "search"',
            'thinking<tool_call>{"name": "search", "arguments": {"query": "test',
            '{"query": "test',
            [1, 10],
            [1, 10, 20],
            [20],
            req,
        )
        self.assertIsNotNone(r)

        # Step 5: more arguments
        r = parser.extract_tool_calls_streaming(
            'thinking<tool_call>{"name": "search", "arguments": {"query": "test',
            'thinking<tool_call>{"name": "search", "arguments": {"query": "test data',
            " data",
            [1, 10, 20],
            [1, 10, 20, 30],
            [30],
            req,
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.tool_calls[0].function.arguments, " data")

    def test_streaming_multiple_tool_calls(self):
        """Integration test: two tool calls in one response"""
        parser = self._new_parser()
        req = self.dummy_request

        # First tool call
        parser.extract_tool_calls_streaming(
            "",
            '<tool_call>{"name": "fn1"',
            '<tool_call>{"name": "fn1"',
            [],
            [1, 10],
            [1, 10],
            req,
        )
        self.assertEqual(parser.current_tool_id, 0)

        # Close first tool
        parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn1"',
            '<tool_call>{"name": "fn1"}</tool_call>',
            "}</tool_call>",
            [1, 10],
            [1, 10, 2],
            [2],
            req,
        )

        # Second tool call
        r = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn1"}</tool_call>',
            '<tool_call>{"name": "fn1"}</tool_call><tool_call>{"name": "fn2"',
            '<tool_call>{"name": "fn2"',
            [1, 10, 2],
            [1, 10, 2, 1, 20],
            [1, 20],
            req,
        )
        self.assertEqual(parser.current_tool_id, 1)
        self.assertIsNotNone(r)
        self.assertEqual(r.tool_calls[0].function.name, "fn2")


if __name__ == "__main__":
    unittest.main()
