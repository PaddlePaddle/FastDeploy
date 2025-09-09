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

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
)


class YourClass:
    async def _1(self, a, b, c):
        if b["outputs"].get("send_idx", -1) == 0 and a.echo:
            if isinstance(a.prompt, list):
                text = a.prompt[c]
            else:
                text = a.prompt
            b["outputs"]["text"] = text + (b["outputs"]["text"] or "")


class TestCompletionEcho(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_engine = MagicMock()
        self.completion_handler = None
        self.mock_engine.data_processor.tokenizer.decode = lambda x: f"decoded_{x}"

    def test_single_str_prompt_non_streaming(self):
        """Testing echo prompt in non-streaming of a single str prompt"""
        self.completion_handler = OpenAIServingCompletion(
            self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30
        )

        request = CompletionRequest(prompt="test prompt", max_tokens=10, echo=True, logprobs=1)

        mock_output = {
            "outputs": {
                "text": " generated text",
                "token_ids": [1, 2, 3],
                "top_logprobs": {"token1": -0.1, "token2": -0.2},
                "finished": True,
            },
            "output_token_ids": 3,
        }
        self.mock_engine.generate.return_value = [mock_output]

        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=[mock_output],
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2]],
            completion_batched_token_ids=[[3, 4, 5]],
            text_after_process_list=["test prompt"],
        )

        self.assertEqual(response.choices[0].text, "test prompt generated text")

    def test_single_int_prompt_non_streaming(self):
        """Testing echo prompt in non-streaming of a single int prompt"""
        self.completion_handler = OpenAIServingCompletion(
            self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30
        )

        request = CompletionRequest(prompt=[1, 2, 3], max_tokens=10, echo=True, logprobs=1)

        mock_output = {
            "outputs": {
                "text": " generated text",
                "token_ids": [1, 2, 3],
                "top_logprobs": {"token1": -0.1, "token2": -0.2},
                "finished": True,
            },
            "output_token_ids": 3,
        }
        self.mock_engine.generate.return_value = [mock_output]

        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=[mock_output],
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2]],
            completion_batched_token_ids=[[3, 4, 5]],
            text_after_process_list=["test prompt"],
        )
        self.assertEqual(response.choices[0].text, "decoded_[1, 2, 3] generated text")

    async def test_single_str_prompt_streaming(self):
        """Testing echo prompts in streaming of a single str prompt"""
        request = CompletionRequest(echo=True, prompt=["Hello"])
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "Hello!")

    async def test_single_int_prompt_streaming(self):
        """Testing echoing prompts in streaming of a single int prompt"""
        request = CompletionRequest(prompt=[1, 2, 3], max_tokens=10, stream=True, echo=True)
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "decoded_[1, 2, 3]!")

    def test_multi_str_prompt_non_streaming(self):
        """Testing echo prompts in non-streaming of multiple str prompts"""
        self.completion_handler = OpenAIServingCompletion(
            self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30
        )

        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, echo=True)

        mock_outputs = [
            {
                "outputs": {"text": " response1", "token_ids": [1, 2], "top_logprobs": None, "finished": True},
                "output_token_ids": 2,
            },
            {
                "outputs": {"text": " response2", "token_ids": [3, 4], "top_logprobs": None, "finished": True},
                "output_token_ids": 2,
            },
        ]
        self.mock_engine.generate.return_value = mock_outputs

        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=mock_outputs,
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1], [2]],
            completion_batched_token_ids=[[1, 2], [3, 4]],
            text_after_process_list=["prompt1", "prompt2"],
        )

        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.choices[0].text, "prompt1 response1")
        self.assertEqual(response.choices[1].text, "prompt2 response2")

    def test_multi_int_prompt_non_streaming(self):
        """Testing echo prompts in non-streaming of multiple int prompts"""
        self.completion_handler = OpenAIServingCompletion(
            self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30
        )

        request = CompletionRequest(prompt=[[1, 2, 3], [4, 5, 6]], max_tokens=10, echo=True)

        mock_outputs = [
            {
                "outputs": {"text": " response1", "token_ids": [1, 2], "top_logprobs": None, "finished": True},
                "output_token_ids": 2,
            },
            {
                "outputs": {"text": " response2", "token_ids": [3, 4], "top_logprobs": None, "finished": True},
                "output_token_ids": 2,
            },
        ]
        self.mock_engine.generate.return_value = mock_outputs

        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=mock_outputs,
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1], [2]],
            completion_batched_token_ids=[[1, 2], [3, 4]],
            text_after_process_list=["prompt1", "prompt2"],
        )

        self.assertEqual(len(response.choices), 2)
        print("response.choices[0].text", response.choices[0].text)
        print("response.choices[1].text", response.choices[1].text)
        self.assertEqual(response.choices[0].text, "decoded_[1, 2, 3] response1")
        self.assertEqual(response.choices[1].text, "decoded_[4, 5, 6] response2")

    async def test_multi_str_prompt_streaming(self):
        """Testing echo prompts in streaming of multiple str prompts"""
        request = CompletionRequest(echo=True, prompt=["Hello", "World"])
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 1

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "World!")

    async def test_multi_int_prompt_streaming(self):
        """Testing echo prompts in streaming of multiple int prompts"""
        request = CompletionRequest(echo=True, prompt=[[1, 2, 3], [4, 5, 6]])
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 1

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "decoded_[4, 5, 6]!")

    async def test_send_idx_is_not_0(self):
        """Testing send_idx is not 0"""
        request = CompletionRequest(echo=True, prompt="Hello")
        res = {"outputs": {"send_idx": 1, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "!")

    async def test_echo_is_false(self):
        """Testing echo prompts when echo is False"""
        request = CompletionRequest(echo=False, prompt="Hello")
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "!")


if __name__ == "__main__":
    unittest.main()
