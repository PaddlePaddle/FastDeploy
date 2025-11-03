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

from pydantic import ValidationError

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


class TestChatCompletionRequest(unittest.TestCase):

    def test_required_messages(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest()

    def test_messages_accepts_list_of_any_and_int(self):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(req.messages[0]["role"], "user")

        req = ChatCompletionRequest(messages=[1, 2, 3])
        self.assertEqual(req.messages, [1, 2, 3])

    def test_default_values(self):
        req = ChatCompletionRequest(messages=[1])
        self.assertEqual(req.model, "default")
        self.assertFalse(req.logprobs)
        self.assertEqual(req.top_logprobs, 0)
        self.assertEqual(req.n, 1)
        self.assertEqual(req.stop, [])

    def test_boundary_values(self):
        valid_cases = [
            ("frequency_penalty", -2),
            ("frequency_penalty", 2),
            ("presence_penalty", -2),
            ("presence_penalty", 2),
            ("temperature", 0),
            ("top_p", 1),
            ("seed", 0),
            ("seed", 922337203685477580),
        ]
        for field, value in valid_cases:
            with self.subTest(field=field, value=value):
                req = ChatCompletionRequest(messages=[1], **{field: value})
                self.assertEqual(getattr(req, field), value)

    def test_invalid_boundary_values(self):
        invalid_cases = [
            ("frequency_penalty", -3),
            ("frequency_penalty", 3),
            ("presence_penalty", -3),
            ("presence_penalty", 3),
            ("temperature", -1),
            ("top_p", 1.1),
            ("seed", -1),
            ("seed", 922337203685477581),
        ]
        for field, value in invalid_cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest(messages=[1], **{field: value})

    def test_stop_field_accepts_str_or_list(self):
        req1 = ChatCompletionRequest(messages=[1], stop="end")
        self.assertEqual(req1.stop, "end")

        req2 = ChatCompletionRequest(messages=[1], stop=["a", "b"])
        self.assertEqual(req2.stop, ["a", "b"])

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(messages=[1], stop=123)

    def test_deprecated_max_tokens_field(self):
        req = ChatCompletionRequest(messages=[1], max_tokens=10)
        self.assertEqual(req.max_tokens, 10)

    def test_field_names_snapshot(self):
        expected_fields = set(ChatCompletionRequest.__fields__.keys())
        self.assertEqual(set(ChatCompletionRequest.__fields__.keys()), expected_fields)


class TestCompletionRequest(unittest.TestCase):

    def test_required_prompt(self):
        with self.assertRaises(ValidationError):
            CompletionRequest()

    def test_prompt_accepts_various_types(self):
        # str
        req = CompletionRequest(prompt="hello")
        self.assertEqual(req.prompt, "hello")

        # list of str
        req = CompletionRequest(prompt=["hello", "world"])
        self.assertEqual(req.prompt, ["hello", "world"])

        # list of int
        req = CompletionRequest(prompt=[1, 2, 3])
        self.assertEqual(req.prompt, [1, 2, 3])

        # list of list of int
        req = CompletionRequest(prompt=[[1, 2], [3, 4]])
        self.assertEqual(req.prompt, [[1, 2], [3, 4]])

    def test_default_values(self):
        req = CompletionRequest(prompt="test")
        self.assertEqual(req.model, "default")
        self.assertEqual(req.echo, False)
        self.assertEqual(req.temp_scaled_logprobs, False)
        self.assertEqual(req.top_p_normalized_logprobs, False)
        self.assertEqual(req.n, 1)
        self.assertEqual(req.stop, [])
        self.assertEqual(req.stream, False)

    def test_boundary_values(self):
        valid_cases = [
            ("frequency_penalty", -2),
            ("frequency_penalty", 2),
            ("presence_penalty", -2),
            ("presence_penalty", 2),
            ("temperature", 0),
            ("top_p", 0),
            ("top_p", 1),
            ("seed", 0),
            ("seed", 922337203685477580),
        ]
        for field, value in valid_cases:
            with self.subTest(field=field, value=value):
                req = CompletionRequest(prompt="hi", **{field: value})
                self.assertEqual(getattr(req, field), value)

    def test_invalid_boundary_values(self):
        invalid_cases = [
            ("frequency_penalty", -3),
            ("frequency_penalty", 3),
            ("presence_penalty", -3),
            ("presence_penalty", 3),
            ("temperature", -0.1),
            ("top_p", -0.1),
            ("top_p", 1.1),
            ("seed", -1),
            ("seed", 922337203685477581),
        ]
        for field, value in invalid_cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    CompletionRequest(prompt="hi", **{field: value})


if __name__ == "__main__":
    unittest.main()
