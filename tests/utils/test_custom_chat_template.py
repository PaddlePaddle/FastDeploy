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

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat


class TestLodChatTemplate(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """
        Set up the test environment by creating an instance of the LLM class using Mock.
        """
        self.input_chat_template = "unit test \n"
        self.mock_engine = MagicMock()
        self.tokenizer = MagicMock()

    def test_load_chat_template_non(self):
        result = load_chat_template(None)
        self.assertEqual(None, result)

    def test_load_chat_template_str(self):
        result = load_chat_template(self.input_chat_template)
        self.assertEqual(self.input_chat_template, result)

    def test_load_chat_template_path(self):
        with open("chat_template", "w", encoding="utf-8") as file:
            file.write(self.input_chat_template)
        file_path = os.path.join(os.getcwd(), "chat_template")
        result = load_chat_template(file_path)
        os.remove(file_path)
        self.assertEqual(self.input_chat_template, result)

    def test_load_chat_template_non_str_and_path(self):
        with self.assertRaises(ValueError):
            load_chat_template("unit test")

    def test_path_with_literal_true(self):
        with self.assertRaises(TypeError):
            load_chat_template(Path("./chat_template"), is_literal=True)

    def test_path_object_file_error(self):
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = OSError("File error")
            with self.assertRaises(OSError):
                load_chat_template(Path("./chat_template"))

    async def test_serving_chat(self):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "你好"}])
        self.chat_completion_handler = OpenAIServingChat(
            self.mock_engine,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=-1,
            chat_template=self.input_chat_template,
        )

        async def mock_chat_completion_full_generator(
            request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens_list
        ):
            return prompt_token_ids

        async def mock_format_and_add_data(current_req_dict):
            current_req_dict["text_after_process"] = "你好"
            return current_req_dict

        self.chat_completion_handler.chat_completion_full_generator = mock_chat_completion_full_generator
        self.chat_completion_handler.engine_client.format_and_add_data = mock_format_and_add_data
        self.chat_completion_handler.engine_client.semaphore = AsyncMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=None)
        self.chat_completion_handler.engine_client.semaphore.status = MagicMock(return_value="mock_status")
        chat_completiom = await self.chat_completion_handler.create_chat_completion(request)
        self.assertEqual(self.input_chat_template, chat_completiom["chat_template"])

    async def test_serving_chat_cus(self):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], chat_template="hello")
        self.chat_completion_handler = OpenAIServingChat(
            self.mock_engine,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=10,
            chat_template=self.input_chat_template,
        )

        async def mock_chat_completion_full_generator(
            request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens_list
        ):
            return prompt_token_ids

        async def mock_format_and_add_data(current_req_dict):
            current_req_dict["text_after_process"] = "你好"
            return current_req_dict

        self.chat_completion_handler.chat_completion_full_generator = mock_chat_completion_full_generator
        self.chat_completion_handler.engine_client.format_and_add_data = mock_format_and_add_data
        self.chat_completion_handler.engine_client.semaphore = AsyncMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=None)
        self.chat_completion_handler.engine_client.semaphore.status = MagicMock(return_value="mock_status")
        chat_completion = await self.chat_completion_handler.create_chat_completion(request)
        self.assertEqual("hello", chat_completion["chat_template"])

    @patch("fastdeploy.entrypoints.llm.LLM.__init__")
    def test_llm(self, mock_class):
        mock_class.return_value = None
        llm = LLM()
        llm.llm_engine = MagicMock()
        llm.default_sampling_params = MagicMock()

        def mock_run_engine(req_ids, **kwargs):
            return req_ids

        def mock_add_request(**kwargs):
            return kwargs.get("chat_template")

        llm._run_engine = mock_run_engine
        llm._add_request = mock_add_request
        result = llm.chat(["hello"], sampling_params=SamplingParams(1), chat_template="hello")
        self.assertEqual("hello", result)


if __name__ == "__main__":
    unittest.main()
