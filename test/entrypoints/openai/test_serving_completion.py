import unittest
from typing import List
from unittest.mock import Mock

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
    RequestOutput,
)


class TestOpenAIServingCompletion(unittest.TestCase):

    def test_calc_finish_reason_tool_calls(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie_x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie_x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips", 360)
        # 创建一个模拟的output，并设置finish_reason为"tool_call"
        output = {"tool_call": "tool_call"}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(None, 100, output, False)
        # 断言结果为"tool_calls"
        assert result == "tool_calls"

    def test_calc_finish_reason_stop(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie_x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie_x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips", 360)
        # 创建一个模拟的output，并设置finish_reason为其他值
        output = {"finish_reason": "other_reason"}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(None, 100, output, False)
        # 断言结果为"stop"
        assert result == "stop"

    def test_calc_finish_reason_length(self):
        # 创建一个模拟的engine_client
        engine_client = Mock()
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips", 360)
        # 创建一个模拟的output
        output = {}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(100, 100, output, False)
        # 断言结果为"length"
        assert result == "length"

    def test_request_output_to_completion_response(self):
        engine_client = Mock()
        # 创建一个OpenAIServingCompletion实例
        openai_serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips", 360)
        final_res_batch: List[RequestOutput] = [
            {
                "prompt": "Hello, world!",
                "outputs": {
                    "token_ids": [1, 2, 3],
                    "text": " world!",
                    "top_logprobs": {
                        "a": 0.1,
                        "b": 0.2,
                    },
                },
                "output_token_ids": 3,
            },
            {
                "prompt": "Hello, world!",
                "outputs": {
                    "token_ids": [4, 5, 6],
                    "text": " world!",
                    "top_logprobs": {
                        "a": 0.3,
                        "b": 0.4,
                    },
                },
                "output_token_ids": 3,
            },
        ]

        request: CompletionRequest = Mock()
        request_id = "test_request_id"
        created_time = 1655136000
        model_name = "test_model"
        prompt_batched_token_ids = [[1, 2, 3], [4, 5, 6]]
        completion_batched_token_ids = [[7, 8, 9], [10, 11, 12]]

        completion_response = openai_serving_completion.request_output_to_completion_response(
            final_res_batch=final_res_batch,
            request=request,
            request_id=request_id,
            created_time=created_time,
            model_name=model_name,
            prompt_batched_token_ids=prompt_batched_token_ids,
            completion_batched_token_ids=completion_batched_token_ids,
            text_after_process_list=["1", "1"],
        )

        assert completion_response.id == request_id
        assert completion_response.created == created_time
        assert completion_response.model == model_name
        assert len(completion_response.choices) == 2

        # 验证 choices 的 text 属性
        assert completion_response.choices[0].text == "Hello, world! world!"
        assert completion_response.choices[1].text == "Hello, world! world!"


if __name__ == "__main__":
    unittest.main()
