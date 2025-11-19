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
from typing import List
from unittest.mock import Mock

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
    RequestOutput,
)
from fastdeploy.utils import get_host_ip


class TestOpenAIServingCompletion(unittest.TestCase):

    def test_check_master_tp4_dp1(self):
        engine_client = Mock()
        engine_client.tensor_parallel_size = 4
        max_chips_per_node = 8
        if engine_client.tensor_parallel_size <= max_chips_per_node:
            engine_client.is_master = True
        else:
            engine_client.is_master = False
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)
        self.assertTrue(serving_completion._check_master())

    def test_check_master_tp4_dp4(self):
        engine_client = Mock()
        engine_client.tensor_parallel_size = 4
        max_chips_per_node = 8
        if engine_client.tensor_parallel_size <= max_chips_per_node:
            engine_client.is_master = True
        else:
            engine_client.is_master = False
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "0.0.0.0, {get_host_ip()}", 360)
        self.assertTrue(serving_completion._check_master())

    def test_check_master_tp16_dp1_slave(self):
        engine_client = Mock()
        engine_client.tensor_parallel_size = 16
        max_chips_per_node = 8
        if engine_client.tensor_parallel_size <= max_chips_per_node:
            engine_client.is_master = True
        else:
            engine_client.is_master = False
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", f"0.0.0.0, {get_host_ip()}", 360)
        self.assertFalse(serving_completion._check_master())

    def test_check_master_tp16_dp1_master(self):
        engine_client = Mock()
        engine_client.tensor_parallel_size = 16
        max_chips_per_node = 8
        if engine_client.tensor_parallel_size <= max_chips_per_node:
            engine_client.is_master = True
        else:
            engine_client.is_master = False
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", f"{get_host_ip()}, 0.0.0.0", 360)
        self.assertTrue(serving_completion._check_master())

    def test_calc_finish_reason_tool_calls(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie-x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie-x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)
        # 创建一个模拟的output，并设置finish_reason为"tool_call"
        output = {"tool_call": "tool_call"}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(None, 100, output, False)
        # 断言结果为"tool_calls"
        assert result == "tool_calls"

    def test_calc_finish_reason_stop(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie-x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie-x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)
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
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)
        # 创建一个模拟的output
        output = {}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(100, 100, output, False)
        # 断言结果为"length"
        assert result == "length"

    def test_request_output_to_completion_response(self):
        engine_client = Mock()
        # 创建一个OpenAIServingCompletion实例
        openai_serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)
        final_res_batch: List[RequestOutput] = [
            {
                "outputs": {
                    "token_ids": [1, 2, 3],
                    "text": " world!",
                    "top_logprobs": {
                        "a": 0.1,
                        "b": 0.2,
                    },
                    "reasoning_token_num": 10,
                },
                "output_token_ids": 3,
            },
            {
                "outputs": {
                    "token_ids": [4, 5, 6],
                    "text": " world!",
                    "top_logprobs": {
                        "a": 0.3,
                        "b": 0.4,
                    },
                    "reasoning_token_num": 20,
                },
                "output_token_ids": 3,
            },
        ]

        request: CompletionRequest = Mock()
        request.prompt = "Hello, world!"
        request.echo = True
        request.n = 2
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
            prompt_tokens_list=["1", "1"],
            max_tokens_list=[10, 10],
        )

        assert completion_response.id == request_id
        assert completion_response.created == created_time
        assert completion_response.model == model_name
        assert len(completion_response.choices) == 2

        # 验证 choices 的 text 属性
        assert completion_response.choices[0].text == "Hello, world! world!"
        assert completion_response.choices[1].text == "Hello, world! world!"

        assert completion_response.usage.completion_tokens_details.reasoning_tokens == 30


if __name__ == "__main__":
    unittest.main()
