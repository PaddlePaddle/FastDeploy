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
from unittest.mock import MagicMock

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat


class TestOpenAIServingCompletion(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by creating an instance of the OpenAIServingChat class using Mock.
        """
        self.mock_engine = MagicMock()
        self.chat_completion_handler = OpenAIServingChat(
            self.mock_engine,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=10,
            chat_template=None,
        )

    def test_enable_thinking(self):
        request = ChatCompletionRequest(messages=[], chat_template_kwargs={})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, None)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"enable_thinking": True})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, True)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"enable_thinking": False})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, False)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"options": {"thinking_mode": "close"}})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, False)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"options": {"thinking_mode": "false"}})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, False)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"options": {"thinking_mode": "open"}})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, True)

        request = ChatCompletionRequest(messages=[], chat_template_kwargs={"options": {"thinking_mode": "123"}})
        enable_thinking = self.chat_completion_handler._get_thinking_status(request)
        self.assertEqual(enable_thinking, True)


if __name__ == "__main__":
    unittest.main()
