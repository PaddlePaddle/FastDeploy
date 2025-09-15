# Copyright (c) 2025 PaddlePaddle Authors.
# Licensed under the Apache License, Version 2.0

import unittest

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager
from fastdeploy.reasoning.ernie_x1_reasoning_parsers import ErnieX1ReasoningParser


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "</think>": 1001,
            "<tool_call>": 1002,
            "<response>": 1003,
            "</response>": 1004,
        }

    def get_vocab(self):
        return self.vocab


class TestErnieX1ReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.parser = ErnieX1ReasoningParser(self.tokenizer)

    def test_init_without_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            ErnieX1ReasoningParser(None)

    def test_init_without_vocab_token_raises(self):
        bad_tokenizer = MockTokenizer()
        del bad_tokenizer.vocab["</think>"]
        with self.assertRaises(RuntimeError):
            ErnieX1ReasoningParser(bad_tokenizer)

    def test_streaming_reasoning_normal_text(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="h",
            delta_text="h",
            previous_token_ids=[],
            current_token_ids=[10],
            delta_token_ids=[10],
        )
        self.assertEqual(msg.reasoning_content, "h")

    def test_streaming_reasoning_single_newline_then_text(self):
        # First add newline
        self.parser.extract_reasoning_content_streaming("", "\n", "\n", [], [11], [11])
        # Then next token (not </think>)
        msg = self.parser.extract_reasoning_content_streaming("\n", "\na", "a", [11], [11, 12], [12])
        self.assertEqual(msg.reasoning_content, "\na")

    def test_streaming_reasoning_multiple_newlines_then_text(self):
        # Two consecutive line breaks
        self.parser.extract_reasoning_content_streaming("", "\n", "\n", [], [11], [11])
        self.parser.extract_reasoning_content_streaming("\n", "\n\n", "\n", [11], [11, 11], [11])
        # Then comes a normal character
        msg = self.parser.extract_reasoning_content_streaming("\n\n", "\n\nx", "x", [], [], [30])
        self.assertEqual(msg.reasoning_content, "\n\nx")

    def test_streaming_drop_newline_before_think(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="\n</think>",
            delta_text="</think>",
            previous_token_ids=[],
            current_token_ids=[1001],
            delta_token_ids=[1001],
        )
        self.assertIsNone(msg)

    def test_streaming_response_handling(self):
        # Simulate reasoning ended
        prev_text = "abc</think>"
        current_text = prev_text + "<response>"
        msg = self.parser.extract_reasoning_content_streaming(prev_text, current_text, "<response>", [], [], [1003])
        self.assertIsNone(msg)

        # newline immediately after <response>
        prev_text = "abc</think><response>"
        msg = self.parser.extract_reasoning_content_streaming(prev_text, prev_text + "\n", "\n", [], [], [13])
        self.assertIsNone(msg)

        # real response content
        prev_text = "abc</think><response>"
        msg = self.parser.extract_reasoning_content_streaming(prev_text, prev_text + "hello", "hello", [], [], [20])
        self.assertEqual(msg.content, "hello")

        # buffered newline then text
        self.parser._pending_newlines = 1
        msg = self.parser.extract_reasoning_content_streaming(
            "abc</think><response>\n", "abc</think><response>\na", "a", [], [], [21]
        )
        self.assertEqual(msg.content, "\na")

        # multiple buffered newlines then text
        self.parser._pending_newlines = 2
        msg = self.parser.extract_reasoning_content_streaming(
            "abc</think><response>\n\n", "abc</think><response>\n\nb", "b", [], [], [22]
        )
        self.assertEqual(msg.content, "\n\nb")

        # end response tag
        msg = self.parser.extract_reasoning_content_streaming(
            "abc</think><response>hi", "abc</think><response>hi</response>", "</response>", [], [], [1004]
        )
        self.assertIsNone(msg)

    def test_streaming_tool_call_handling(self):
        prev_text = "abc</think>"
        current_text = prev_text + "<tool_call>"
        msg = self.parser.extract_reasoning_content_streaming(prev_text, current_text, "<tool_call>", [], [], [1002])
        self.assertIsNone(msg)

    def test_batch_extraction_reasoning_and_response(self):
        output = "thinking...\n</think>\n<response>\nfinal answer\n</response>"
        reasoning, response = self.parser.extract_reasoning_content(
            output, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(reasoning, "thinking...")
        self.assertEqual(response, "final answer")

    def test_batch_extraction_response_without_end(self):
        output = "abc</think>\n<response>hello world"
        reasoning, response = self.parser.extract_reasoning_content(
            output, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "hello world")

    def test_batch_extraction_tool_call(self):
        output = "abc</think>\n<tool_call>something</tool_call>"
        reasoning, response = self.parser.extract_reasoning_content(
            output, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "")

    def test_batch_extraction_only_reasoning(self):
        output = "just thinking..."
        reasoning, response = self.parser.extract_reasoning_content(
            output, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(reasoning, "just thinking...")
        self.assertEqual(response, "")


# 保留原有 ReasoningParserManager 测试
class TestReasoningParser(ReasoningParser):
    def is_reasoning_end(self, input_ids):
        return True

    def extract_content_ids(self, input_ids):
        return input_ids

    def extract_reasoning_content(self, model_output, request):
        return model_output, model_output

    def extract_reasoning_content_streaming(self, *args, **kwargs):
        return None


class TestReasoningParserManager(unittest.TestCase):
    def setUp(self):
        self.original_parsers = ReasoningParserManager.reasoning_parsers.copy()

    def tearDown(self):
        ReasoningParserManager.reasoning_parsers = self.original_parsers.copy()

    def test_register_and_get_parser(self):
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser", force=True)
        parser_cls = ReasoningParserManager.get_reasoning_parser("test_parser")
        self.assertIs(parser_cls, TestReasoningParser)

    def test_register_duplicate_without_force_raises(self):
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=True)
        with self.assertRaises(KeyError):
            ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=False)

    def test_register_non_subclass_raises(self):
        class NotParser:
            pass

        with self.assertRaises(TypeError):
            ReasoningParserManager.register_module(module=NotParser, name="not_parser")

    def test_get_unregistered_parser_raises(self):
        with self.assertRaises(KeyError):
            ReasoningParserManager.get_reasoning_parser("nonexistent_parser")


if __name__ == "__main__":
    unittest.main()
