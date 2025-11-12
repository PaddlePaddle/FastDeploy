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


class TestCompletionEcho(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_engine = MagicMock()
        self.completion_handler = None
        self.mock_engine.data_processor.tokenizer.decode = lambda x: f"decoded_{x}"

    """Testing echo prompt in non-streaming of a single str prompt"""

    def test_single_str_prompt_non_streaming(self):
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
            prompt_tokens_list=["test prompt"],
            max_tokens_list=[100],
        )

        self.assertEqual(response.choices[0].text, "test prompt generated text")

    """Testing echo prompt in non-streaming of a single int prompt"""

    def test_single_int_prompt_non_streaming(self):
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
            prompt_tokens_list=["test prompt"],
            max_tokens_list=[100],
        )
        self.assertEqual(response.choices[0].text, "decoded_[1, 2, 3] generated text")

    """Testing echo prompts in non-streaming of multiple str prompts"""

    def test_multi_str_prompt_non_streaming(self):
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
            prompt_tokens_list=["prompt1", "prompt2"],
            max_tokens_list=[100, 100],
        )

        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.choices[0].text, "prompt1 response1")
        self.assertEqual(response.choices[1].text, "prompt2 response2")

    """Testing echo prompts in non-streaming of multiple int prompts"""

    def test_multi_int_prompt_non_streaming(self):
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
            prompt_tokens_list=["prompt1", "prompt2"],
            max_tokens_list=[100, 100],
        )

        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.choices[0].text, "decoded_[1, 2, 3] response1")
        self.assertEqual(response.choices[1].text, "decoded_[4, 5, 6] response2")

    """Testing echo prompts in streaming of a single str prompt"""

    async def test_single_str_prompt_streaming(self):
        request = CompletionRequest(prompt="test prompt", max_tokens=10, stream=True, echo=True)
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        res = await instance._process_echo_logic(request, idx, res["outputs"])
        self.assertEqual(res["text"], "test prompt!")

    """Testing echo prompts in streaming of a single int prompt"""

    async def test_single_int_prompt_streaming(self):
        request = CompletionRequest(prompt=[1, 2, 3], max_tokens=10, stream=True, echo=True)
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        res = await instance._process_echo_logic(request, idx, res["outputs"])
        self.assertEqual(res["text"], "decoded_[1, 2, 3]!")

    """Testing echo prompts in streaming of multi str prompt"""

    async def test_multi_str_prompt_streaming(self):
        request = CompletionRequest(prompt=["test prompt1", "test prompt2"], max_tokens=10, stream=True, echo=True)
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, models=None, pid=123, ips=None, max_waiting_time=30)
        res = await instance._process_echo_logic(request, idx, res["outputs"])
        self.assertEqual(res["text"], "test prompt1!")


if __name__ == "__main__":
    unittest.main()
