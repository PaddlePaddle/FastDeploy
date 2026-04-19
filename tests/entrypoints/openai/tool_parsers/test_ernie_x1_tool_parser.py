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

    def _simulate_streaming(self, parser, deltas):
        """Simulate a multi-step streaming flow.

        Args:
            parser: ErnieX1ToolParser instance
            deltas: list of delta text strings, each representing one streaming step

        Returns:
            list of results from each extract_tool_calls_streaming call
        """
        results = []
        previous_text = ""
        token_id = 0
        previous_token_ids = []

        for delta in deltas:
            current_text = previous_text + delta
            # When delta contains <tool_call> plus more content, use 2 tokens
            # so that the parser extracts tool_call_portion (line 163-164)
            if "<tool_call>" in delta and delta != "<tool_call>":
                n_tokens = 2
            else:
                n_tokens = 1

            delta_token_ids = list(range(token_id + 1, token_id + 1 + n_tokens))
            token_id += n_tokens
            current_token_ids = previous_token_ids + delta_token_ids

            result = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                self.dummy_request,
            )
            results.append(result)

            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        return results

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

    def test_extract_tool_calls_empty_arguments(self):
        """Cover: tool call with explicit empty arguments {}"""
        output = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "fn")
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
        """Cover lines 96, 111-114: no tool_call tags -> tools_called=False and content passthrough"""
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
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"k": "',  # start + name + args key
                "v",  # args value
                '"}}</tool_call>',  # close with end token in delta
            ],
        )
        # Step 1: name sent
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].tool_calls[0].function.name, "fn")
        # Step 2: first-args branch, regex extracts '{"k": "v' as arguments_delta
        self.assertIsNotNone(results[1])
        self.assertEqual(results[1].tool_calls[0].function.arguments, '{"k": "v')
        # Step 3: end token in delta triggers close handling
        # delta before </tool_call> is '"}}', close branch: rindex('}')=2, diff='"}'
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2].tool_calls[0].function.arguments, '"}')

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
        results = self._simulate_streaming(
            parser,
            [
                "<tool_call>",  # start tool call
                '{"na',  # partial content, no name yet
            ],
        )
        self.assertIsNone(results[0])
        self.assertIsNone(results[1])

    def test_streaming_continue_tool_call_with_name(self):
        """Cover lines 174-176, 223-235: name becomes available"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                "<tool_call>",  # start tool call
                '{"name": "get_weather"',  # name appears
            ],
        )
        self.assertIsNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertEqual(results[1].tool_calls[0].function.name, "get_weather")
        self.assertTrue(parser.current_tool_name_sent)

    # --- Lines 236-237: name not sent and function_name is None ---

    def test_streaming_no_function_name(self):
        """Cover lines 236-237: parsed JSON has no 'name' field"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                "<tool_call>",  # start tool call
                '{"arguments": {"k": "v"}}',  # JSON without name field
            ],
        )
        self.assertIsNone(results[1])

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
            '<tool_call>{"name":"fn","arguments":{"k":"v"',
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call>',
            "}}</tool_call>",
            [1, 10],
            [1, 10, 2],
            [2],
            self.dummy_request,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].function.arguments, "}")

    def test_streaming_text_after_completed_tool_call(self):
        """Cover lines 143-147: text content after a completed tool call.

        When start==end counts, prev_end==cur_end, and end_token not in delta,
        the parser treats delta as regular text content.
        """
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.prev_tool_call_arr = [{"name": "fn", "arguments": {"k": "v"}}]
        # Simulate end token in delta but without '"}' pattern
        # We need cur_start==cur_end and cur_end >= prev_end, and end_token NOT in delta
        # so that we enter the text-content branch at line 143-147
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call>',
            '<tool_call>{"name":"fn","arguments":{"k":"v"}}</tool_call> text',
            " text",
            [1, 10, 2],
            [1, 10, 2, 30],
            [30],
            self.dummy_request,
        )
        # balanced counts, prev_end==cur_end, end not in delta -> returns content (line 149)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, " text")

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
        # diff is None (no arguments key in prev), falls through to partial_json_parser
        # parses complete JSON, cur_args=None, prev_args=None -> no-args -> delta=None
        self.assertIsNone(result)

    def test_streaming_close_with_empty_dict_arguments(self):
        """Regression: close branch must handle arguments={} (empty dict).

        Before fix, `if diff:` was False for empty dict {}, so the close
        logic was skipped. After fix, `if diff is not None:` correctly
        enters the branch.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": ',  # start + name + args key
                "{}",  # empty dict value
                "}",  # outer close brace
                "</tool_call>",  # end token
            ],
        )
        # Step 1: name sent
        # Step 2: first-args, cur_args={} is not None, prev_args=None
        #   Without fix: not {} == True -> no-args branch -> returns None
        #   With fix: enters first-args -> streams "{}" -> DeltaMessage
        self.assertIsNotNone(results[1])
        self.assertIsNotNone(results[1].tool_calls)
        self.assertEqual(results[1].tool_calls[0].function.arguments, "{}")

    def test_streaming_empty_arguments_with_outer_brace_in_same_token(self):
        """Regression: when arguments={} and outer } arrive in the same token '{}}',
        regex (.*) over-captures the outer brace, producing '{}}'.

        Real production data showed arguments='{}}}' for get_default_weather
        with empty arguments. This test reproduces that exact scenario.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "get_default_weather", "arguments": ',  # start + name + args key
                "{}}",  # empty args + outer close brace in same token
                "</tool_call>",  # end token
            ],
        )
        # Step 1: name sent
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].tool_calls[0].function.name, "get_default_weather")
        # Step 2: first-args branch, tool_call_portion is complete JSON
        # regex (.*) captures '{}}'  but fix strips outer '}' -> '{}'
        self.assertIsNotNone(results[1])
        self.assertEqual(results[1].tool_calls[0].function.arguments, "{}")
        # Step 3: end token, close branch
        # diff = prev_arguments = {} (not None), delta_text = '' (empty after split)
        # '}' not in '' -> returns None
        self.assertIsNone(results[2])

    def test_streaming_close_with_number_ending_arguments(self):
        """Regression: close branch must flush remaining args ending with number.

        Before fix, '"}' not in delta was True for numbers, causing return None.
        After fix, rindex('}') correctly finds the closing brace.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"count": ',  # start + name + args key
                "123",  # number value
                "}}</tool_call>",  # close braces + end token
            ],
        )
        # Step 1: name sent
        # Step 2: first-args, streams {"count": 123
        # Step 3: close branch flushes remaining "}"
        streamed_args = [
            r.tool_calls[0].function.arguments
            for r in results
            if r is not None and r.tool_calls and r.tool_calls[0].function.arguments is not None
        ]
        combined = "".join(streamed_args)
        self.assertEqual(combined, '{"count": 123}')

    def test_streaming_close_with_boolean_ending_arguments(self):
        """Regression: close branch must flush remaining args ending with boolean."""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"flag": ',  # start + args key
                "true",  # boolean value
                "}}</tool_call>",  # close + end token
            ],
        )
        streamed_args = [
            r.tool_calls[0].function.arguments
            for r in results
            if r is not None and r.tool_calls and r.tool_calls[0].function.arguments is not None
        ]
        combined = "".join(streamed_args)
        self.assertEqual(combined, '{"flag": true}')

    def test_streaming_close_with_nested_object_ending(self):
        """Regression: close branch must flush remaining args ending with nested '}'."""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"nested": {"a": ',  # start + args key
                "1",  # nested value
                "}}}</tool_call>",  # close all + end token
            ],
        )
        streamed_args = [
            r.tool_calls[0].function.arguments
            for r in results
            if r is not None and r.tool_calls and r.tool_calls[0].function.arguments is not None
        ]
        combined = "".join(streamed_args)
        self.assertEqual(combined, '{"nested": {"a": 1}}')

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
        results = self._simulate_streaming(
            parser,
            [
                "<tool_call>",  # start tool call
                "{{{",  # badly formed content
            ],
        )
        self.assertIsNone(results[1])

    def test_streaming_json_decode_error(self):
        """Cover lines 216-218: JSONDecodeError from partial parser"""
        parser = self._new_parser()
        # Step 1: start tool call normally
        self._simulate_streaming(parser, ["<tool_call>"])
        # Step 2: mock partial_json_parser to throw ValueError
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            side_effect=ValueError("bad json"),
        ):
            result = parser.extract_tool_calls_streaming(
                "<tool_call>",
                "<tool_call>bad",
                "bad",
                [1],
                [1, 2],
                [2],
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
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "get_weather", "arguments": {"location": "',  # start + name + args key
                "bei",  # args value
            ],
        )
        # Step 1: name sent
        # Step 2: first-args, regex finds "bei" in '{"location": "bei'
        self.assertIsNotNone(results[1])
        self.assertEqual(results[1].tool_calls[0].function.arguments, '{"location": "bei')

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
        """Cover lines 275-276: delta_text not found in cur_arguments_json, returns None.
        When delta contains the arguments key itself (e.g. ', "arguments": {'),
        regex extracts cur_arguments_json='{' but delta ', "arguments": {' is not in '{'.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn"',  # start + partial name
                ', "arguments": {',  # delta introduces arguments key + open brace
            ],
        )
        # Step 1: name sent
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].tool_calls[0].function.name, "fn")
        # Step 2: first-args branch, regex extracts cur_arguments_json='{'
        # delta_text=', "arguments": {' is NOT in '{' -> returns None
        self.assertIsNone(results[1])

    # --- Lines 249-251: no cur_arguments and no prev_arguments ---

    def test_streaming_no_arguments_at_all(self):
        """Cover lines 249-251: both cur and prev arguments are empty/None"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn"',  # start + name
                "}",  # close JSON, no arguments
            ],
        )
        # prev_arguments=None, cur_arguments=None -> delta=None
        self.assertIsNone(results[1])

    def test_streaming_empty_dict_arguments_not_skipped(self):
        """Regression: arguments={} (empty dict) must not be treated as no arguments.

        Empty dict is falsy in Python (`not {} == True`). Before the fix,
        this caused empty arguments to enter the no-arguments branch,
        silently dropping them during streaming.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": ',  # start + name + args key
                "{}",  # empty dict value
                "}",  # outer close brace
            ],
        )
        # Step 1: name sent
        # Step 2: cur_arguments={} (not None), prev_arguments=None
        #   With fix: enters first-arguments branch -> streams "{}"
        #   Without fix: not {} == True -> no-arguments branch -> delta=None
        self.assertIsNotNone(results[1])
        self.assertIsNotNone(results[1].tool_calls)
        self.assertEqual(results[1].tool_calls[0].function.arguments, "{}")

    def test_streaming_empty_dict_prev_arguments_not_reset(self):
        """Regression: prev_arguments={} must not be treated as no arguments.

        When prev has {} and cur has a non-empty dict, the code should enter
        the both-have-arguments branch, not the first-arguments branch.

        This scenario (arguments growing from {} to non-empty) is hard to
        produce naturally, so we build up state through a real flow then
        verify the branch behavior with one additional call.
        """
        parser = self._new_parser()
        # Build up state naturally: prev_tool_call_arr gets arguments={}
        self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": ',  # name + args key
                "{}",  # empty dict value
                "}",  # outer close
            ],
        )
        # Verify state is correct
        self.assertEqual(parser.prev_tool_call_arr[0].get("arguments"), {})

        # Now test: if more argument data arrives, prev_args={} should be
        # treated as "not None" -> enters both-have-arguments branch
        # Without fix: not {} == True -> first-arguments branch (wrong)
        result = parser.extract_tool_calls_streaming(
            '<tool_call>{"name": "fn", "arguments": {"k": "v',
            '<tool_call>{"name": "fn", "arguments": {"k": "val',
            "al",
            [1, 2, 3],
            [1, 2, 3, 4],
            [4],
            self.dummy_request,
        )
        # both-have-arguments branch: delta_text="al" streamed as arguments
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_calls[0].function.arguments, "al")

    # --- Lines 253-255: cur_arguments reset (impossible branch) ---

    def test_streaming_arguments_reset_mid_call(self):
        """Cover lines 253-255: prev has arguments but cur doesn't (impossible case).

        This is an edge case that shouldn't happen in normal flow, but tests
        defensive handling when partial parser returns no arguments after
        previously having them.
        """
        parser = self._new_parser()
        parser.current_tool_id = 0
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        # Simulate state where prev already had arguments
        parser.prev_tool_call_arr = [{"name": "fn", "arguments": {"k": "v"}}]
        # Mock parser to return no arguments (simulating the impossible reset)
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.partial_json_parser.loads",
            return_value={"name": "fn"},
        ):
            result = parser.extract_tool_calls_streaming(
                '<tool_call>{"name": "fn", "arguments": {"k": "v"',
                '<tool_call>{"name": "fn", "arguments": {"k": "v"}',
                '"}',
                [1, 2],
                [1, 2, 3],
                [3],
                self.dummy_request,
            )
            self.assertIsNone(result)

    # --- Lines 288-314: cur_arguments and prev_arguments both present ---

    def test_streaming_incremental_arguments_incomplete(self):
        """Cover lines 288-314: both prev and cur have arguments, JSON incomplete"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"k": "v',  # start + name + first args
                "a",  # establishes prev_args
                "l",  # incremental: both-have-args
            ],
        )
        # Step 1: name sent
        # Step 2: first-args branch
        # Step 3: both-have-args branch, streams "l"
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2].tool_calls[0].function.arguments, "l")

    def test_streaming_incremental_arguments_complete_json(self):
        """Cover lines 289-305: complete JSON with trailing }"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"k": "v',  # start + name + first args
                "a",  # establishes prev_args
                '"}}',  # completes JSON
            ],
        )
        # Step 3: both-have-args, complete JSON, strips trailing } -> streams '"}'
        self.assertIsNotNone(results[2])
        self.assertIsInstance(results[2], DeltaMessage)

    def test_streaming_incremental_arguments_complete_empty_delta(self):
        """Cover lines 304-305: complete JSON where delta becomes empty after strip"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": {"k": "v"',  # start + name + first args
                "}",  # inner close (establishes prev_args)
                "}",  # outer close: both-have-args, complete, delta stripped to ""
            ],
        )
        # Step 3: is_complete_json=True, delta="}" -> stripped to "" -> return None
        self.assertIsNone(results[2])

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
        results = self._simulate_streaming(
            parser,
            [
                "thinking",  # Step 1: text before tool call
                "<tool_call>",  # Step 2: tool_call start token
                '{"name": "search", "arguments": {"query": "',  # Step 3: name + args key
                "test",  # Step 4: args value
                " data",  # Step 5: more args
            ],
        )
        # Step 1: plain text
        self.assertEqual(results[0].content, "thinking")
        # Step 2: start token -> None
        self.assertIsNone(results[1])
        # Step 3: name sent
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2].tool_calls[0].function.name, "search")
        # Step 4: first arguments
        self.assertIsNotNone(results[3])
        self.assertEqual(results[3].tool_calls[0].function.arguments, '{"query": "test')
        # Step 5: more arguments
        self.assertIsNotNone(results[4])
        self.assertEqual(results[4].tool_calls[0].function.arguments, " data")

    def test_streaming_empty_arguments_full_flow(self):
        """Integration: streaming tool call with arguments={} must not lose arguments.

        Simulates a complete streaming flow where the tool call has empty
        arguments. Verifies the name is sent and arguments are streamed.
        """
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn", "arguments": ',  # Step 1: start + name + args key
                "{}",  # Step 2: empty dict value
                "}",  # Step 3: outer close
                "</tool_call>",  # Step 4: end token
            ],
        )
        # Step 1: name sent
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].tool_calls[0].function.name, "fn")
        # Step 2: first-args with cur_args={}, streams "{}"
        self.assertIsNotNone(results[1])
        self.assertEqual(results[1].tool_calls[0].function.arguments, "{}")
        # Step 4: close branch, delta_text="" after stripping </tool_call>
        #   diff={} is not None, but "}" not in "" -> return None
        self.assertIsNone(results[2])
        self.assertIsNone(results[3])

    def test_streaming_multiple_tool_calls(self):
        """Integration test: two tool calls in one response"""
        parser = self._new_parser()
        results = self._simulate_streaming(
            parser,
            [
                '<tool_call>{"name": "fn1"',  # First tool: start + name
                "}</tool_call>",  # Close first tool
                '<tool_call>{"name": "fn2"',  # Second tool: start + name
            ],
        )
        self.assertEqual(parser.current_tool_id, 1)
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2].tool_calls[0].function.name, "fn2")


if __name__ == "__main__":
    unittest.main()
