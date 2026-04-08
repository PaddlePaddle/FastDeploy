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

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.reasoning.ernie_vl_reasoning_parsers import ErnieVLReasoningParser


class MockTokenizer:
    """Minimal tokenizer with vocab for testing."""

    def __init__(self):
        self.vocab = {
            "<think>": 100,
            "</think>": 101,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class TestErnieVLReasoningParser(unittest.TestCase):
    def setUp(self):
        self.parser = ErnieVLReasoningParser(MockTokenizer())
        self.request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "test message"}])
        self.tokenizer = MockTokenizer()

    def test_get_model_status(self):
        status = self.parser.get_model_status([1, 2, 100])
        self.assertEqual(status, "think_start")
        status = self.parser.get_model_status([1, 2, 101])
        self.assertEqual(status, "think_end")
        status = self.parser.get_model_status([1])
        self.assertEqual(status, "think_start")

    def test_streaming_thinking_content(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a</think>b",
            delta_text="a</think>b",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 101, 102],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")
        self.assertEqual(msg.content, "b")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="a</think>",
            current_text="a</think>b",
            delta_text="b",
            previous_token_ids=[1, 101],
            current_token_ids=[],
            delta_token_ids=[102],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "b")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
            model_status="think_end",
        )
        self.assertEqual(msg.content, "a")

    def test_none_streaming_thinking_content(self):
        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, "")
        self.assertEqual(content, "a")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a</think>b",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, "a")
        self.assertEqual(content, "b")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a",
            request={},
            model_status="think_end",
        )
        self.assertEqual(reasoning_content, "")
        self.assertEqual(content, "a")


if __name__ == "__main__":
    unittest.main()
