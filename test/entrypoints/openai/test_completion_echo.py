import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
)


class TestCompletionEcho(unittest.TestCase):
    def setUp(self):
        # 初始化测试环境
        self.mock_engine = MagicMock()
        self.completion_handler = None  # 将在测试中初始化

    def test_single_prompt_non_streaming(self):
        """测试单prompt非流式响应"""

        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None)

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
            "output_token_ids": len([1, 2, 3]),  # 修正为token数量而不是token列表
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

    @patch("fastdeploy.entrypoints.openai.serving_completion.time.time")
    async def test_single_prompt_streaming(self, mock_time):
        """测试单prompt流式响应"""

        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None)
        mock_time.return_value = 12345

        # 准备测试数据
        request = CompletionRequest(prompt="test prompt", max_tokens=10, stream=True, echo=True)

        # 模拟engine返回的流式数据
        mock_responses = [
            {"outputs": {"text": " chunk1", "token_ids": [1], "send_idx": 0, "finished": False}},
            {"outputs": {"text": " chunk2", "token_ids": [2], "finished": True, "finish_reason": "stop"}},
        ]

        # 模拟异步generate方法
        async def mock_generate(*args, **kwargs):
            for resp in mock_responses:
                yield resp

        self.mock_engine.generate.side_effect = mock_generate

        # 调用测试方法并收集结果
        generator = self.completion_handler.completion_stream_generator(
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2]],
            num_choices=1,
        )

        results = []
        async for chunk in generator:
            results.append(chunk)

        # 验证结果
        self.assertEqual(len(results), 3)  # 2个数据块 + DONE
        self.assertIn('"text": "test prompt chunk1"', results[0])
        self.assertIn('"text": "test prompt chunk2"', results[1])

    def test_multi_prompt_non_streaming(self):
        """测试多prompt非流式响应"""

        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None)

        # 准备测试数据 (2个prompt)
        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, echo=True)

        # 模拟engine返回 (2个结果)
        mock_outputs = [
            {
                "outputs": {"text": " response1", "token_ids": [1, 2], "top_logprobs": None, "finished": True},
                "output_token_ids": len([1, 2]),  # 修正为token数量而不是token列表
            },
            {
                "outputs": {"text": " response2", "token_ids": [3, 4], "top_logprobs": None, "finished": True},
                "output_token_ids": len([3, 4]),  # 修正为token数量而不是token列表
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
            prompt_batched_token_ids=[[1], [2]],  # 2个prompt的token ids
            completion_batched_token_ids=[[1, 2], [3, 4]],  # 2个completion的token ids
        )

        # 验证结果
        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.choices[0].text, "prompt1 response1")
        self.assertEqual(response.choices[1].text, "prompt2 response2")

    @patch("fastdeploy.entrypoints.openai.serving_completion.time.time")
    async def test_multi_prompt_streaming(self, mock_time):
        """测试多prompt流式响应"""

        self.completion_handler = OpenAIServingCompletion(self.mock_engine, pid=123, ips=None)
        mock_time.return_value = 12345

        # 准备测试数据 (2个prompt)
        request = CompletionRequest(prompt=["prompt1", "prompt2"], max_tokens=10, stream=True, echo=True)

        # 模拟engine返回的流式数据 (2个prompt的响应交错)
        mock_responses = [
            # prompt1的第一个chunk
            {"outputs": {"index": 0, "send_idx": 0, "text": " chunk1", "token_ids": [1], "finished": False}},
            # prompt2的第一个chunk
            {"outputs": {"index": 1, "send_idx": 0, "text": " chunkA", "token_ids": [101], "finished": False}},
            # prompt1的最后一个chunk
            {"outputs": {"text": " chunk2", "token_ids": [2], "index": 0, "finished": True, "finish_reason": "stop"}},
            # prompt2的最后一个chunk
            {
                "outputs": {
                    "text": " chunkB",
                    "token_ids": [102],
                    "index": 1,
                    "finished": True,
                    "finish_reason": "length",
                }
            },
        ]

        # 模拟异步generate方法
        async def mock_generate(*args, **kwargs):
            for resp in mock_responses:
                yield resp

        self.mock_engine.generate.side_effect = mock_generate

        # 调用测试方法并收集结果
        generator = self.completion_handler.completion_stream_generator(
            request=request,
            request_id="test_id",
            created_time=12345,
            model_name="test_model",
            prompt_batched_token_ids=[[1], [101]],  # 2个prompt的token ids
            num_choices=2,
        )

        results = []
        async for chunk in generator:
            results.append(chunk)

        # 验证结果
        self.assertEqual(len(results), 5)  # 4个数据块 + DONE

        # 检查prompt1的响应
        self.assertIn('"text": "prompt1 chunk1"', results[0])

        # 检查prompt2的响应
        self.assertIn('"text": "prompt2 chunkA"', results[1])


if __name__ == "__main__":
    unittest.main()
