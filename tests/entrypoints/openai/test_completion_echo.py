import unittest
from unittest.mock import MagicMock, patch

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

    def test_single_prompt_non_streaming(self):
        """测试单prompt非流式响应"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

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

    async def test_echo_back_prompt_and_streaming(self):
        """测试_echo_back_prompt方法和流式响应的prompt拼接逻辑"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        request = CompletionRequest(prompt="test prompt", max_tokens=10, stream=True, echo=True)

        mock_response = {"outputs": {"text": "test output", "token_ids": [1, 2, 3], "finished": True}}

        with patch.object(self.completion_handler, "_echo_back_prompt") as mock_echo:

            def mock_echo_side_effect(req, res, idx):
                res["outputs"]["text"] = req.prompt + res["outputs"]["text"]

            mock_echo.side_effect = mock_echo_side_effect

            await self.completion_handler._echo_back_prompt(request, mock_response, 0)

            mock_echo.assert_called_once_with(request, mock_response, 0)

            self.assertEqual(mock_response["outputs"]["text"], "test prompttest output")
            self.assertEqual(request.prompt, "test prompt")

    def test_multi_prompt_non_streaming(self):
        """测试多prompt非流式响应"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

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

    async def test_multi_prompt_streaming(self):
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, stream=True, echo=True)

        mock_responses = [
            {"outputs": {"text": " response1", "token_ids": [1, 2], "finished": True}},
            {"outputs": {"text": " response2", "token_ids": [3, 4], "finished": True}},
        ]

        with patch.object(self.completion_handler, "_echo_back_prompt") as mock_echo:

            def mock_echo_side_effect(req, res, idx):
                res["outputs"]["text"] = req.prompt[idx] + res["outputs"]["text"]

            mock_echo.side_effect = mock_echo_side_effect

            await self.completion_handler._echo_back_prompt(request, mock_responses[0], 0)
            await self.completion_handler._echo_back_prompt(request, mock_responses[1], 1)

            self.assertEqual(mock_echo.call_count, 2)
            mock_echo.assert_any_call(request, mock_responses[0], 0)
            mock_echo.assert_any_call(request, mock_responses[1], 1)

            self.assertEqual(mock_responses[0]["outputs"]["text"], "prompt1 response1")
            self.assertEqual(mock_responses[1]["outputs"]["text"], "prompt2 response2")
            self.assertEqual(request.prompt, ["prompt1", "prompt2"])

    async def test_echo_back_prompt_and_streaming1(self):
        request = CompletionRequest(echo=True, prompt=["Hello", "World"])
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "Hello!")

    async def test_1_prompt_is_string_and_send_idx_is_0(self):
        request = CompletionRequest(echo=True, prompt="Hello")
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "Hello!")

    async def test_1_send_idx_is_not_0(self):
        request = CompletionRequest(echo=True, prompt="Hello")
        res = {"outputs": {"send_idx": 1, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "!")

    async def test_1_echo_is_false(self):
        """测试echo为False时，_echo_back_prompt不拼接prompt"""
        request = CompletionRequest(echo=False, prompt="Hello")
        res = {"outputs": {"send_idx": 0, "text": "!"}}
        idx = 0

        instance = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)
        await instance._echo_back_prompt(request, res, idx)
        self.assertEqual(res["outputs"]["text"], "!")


if __name__ == "__main__":
    unittest.main()
