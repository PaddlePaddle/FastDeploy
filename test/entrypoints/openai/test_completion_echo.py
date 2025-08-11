import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
)


class TestCompletionEcho(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 初始化测试环境
        self.mock_engine = MagicMock()
        self.completion_handler = None  # 将在测试中初始化

    def test_single_prompt_non_streaming(self):
        """测试单prompt非流式响应"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        # 准备测试数据
        request = CompletionRequest(prompt="test prompt", max_tokens=10, echo=True, logprobs=1)

        # 模拟engine返回
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

        # 调用测试方法
        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=[mock_output],
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2]],
            completion_batched_token_ids=[[3, 4, 5]],
        )

        # 验证结果
        self.assertEqual(response.choices[0].text, "test prompt generated text")

    async def test_echo_back_prompt_and_streaming(self):
        """测试_echo_back_prompt方法和流式响应的prompt拼接逻辑"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        # 准备测试数据
        request = CompletionRequest(prompt="test prompt", max_tokens=10, stream=True, echo=True)

        # 准备mock响应数据
        mock_response = {"outputs": {"text": "test output", "token_ids": [1, 2, 3], "finished": True}}

        # 使用patch来mock _echo_back_prompt方法
        with patch.object(self.completion_handler, "_echo_back_prompt") as mock_echo:
            # 设置mock方法直接修改response中的text
            def mock_echo_side_effect(req, res, idx):
                res["outputs"]["text"] = req.prompt + res["outputs"]["text"]

            mock_echo.side_effect = mock_echo_side_effect

            # 调用_echo_back_prompt方法
            await self.completion_handler._echo_back_prompt(request, mock_response, 0)

            # 验证方法被正确调用
            mock_echo.assert_called_once_with(request, mock_response, 0)

            # 验证结果
            self.assertEqual(mock_response["outputs"]["text"], "test prompttest output")
            self.assertEqual(request.prompt, "test prompt")

    def test_multi_prompt_non_streaming(self):
        """测试多prompt非流式响应"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        # 准备测试数据 (2个prompt)
        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, echo=True)

        # 模拟engine返回 (2个结果)
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

        # 调用测试方法
        response = self.completion_handler.request_output_to_completion_response(
            final_res_batch=mock_outputs,
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1], [2]],
            completion_batched_token_ids=[[1, 2], [3, 4]],
        )

        # 验证结果
        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.choices[0].text, "prompt1 response1")
        self.assertEqual(response.choices[1].text, "prompt2 response2")

    async def test_multi_prompt_streaming(self):
        """测试多prompt流式响应的_echo_back_prompt处理"""
        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None, max_waiting_time=30)

        # 准备测试数据 (2个prompt)
        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, stream=True, echo=True)

        # 准备mock响应数据 (2个prompt的响应)
        mock_responses = [
            {"outputs": {"text": " response1", "token_ids": [1, 2], "finished": True}},
            {"outputs": {"text": " response2", "token_ids": [3, 4], "finished": True}},
        ]

        # 使用patch来mock _echo_back_prompt方法
        with patch.object(self.completion_handler, "_echo_back_prompt") as mock_echo:
            # 设置mock方法根据索引修改对应的response中的text
            def mock_echo_side_effect(req, res, idx):
                res["outputs"]["text"] = req.prompt[idx] + res["outputs"]["text"]

            mock_echo.side_effect = mock_echo_side_effect

            # 调用_echo_back_prompt方法处理两个prompt
            await self.completion_handler._echo_back_prompt(request, mock_responses[0], 0)
            await self.completion_handler._echo_back_prompt(request, mock_responses[1], 1)

            # 验证方法被正确调用
            self.assertEqual(mock_echo.call_count, 2)
            mock_echo.assert_any_call(request, mock_responses[0], 0)
            mock_echo.assert_any_call(request, mock_responses[1], 1)

            # 验证结果
            self.assertEqual(mock_responses[0]["outputs"]["text"], "prompt1 response1")
            self.assertEqual(mock_responses[1]["outputs"]["text"], "prompt2 response2")
            self.assertEqual(request.prompt, ["prompt1", "prompt2"])


if __name__ == "__main__":
    unittest.main()
