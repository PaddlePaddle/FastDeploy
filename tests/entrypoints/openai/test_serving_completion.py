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
from unittest.mock import AsyncMock, Mock, patch

import paddle

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
    RequestOutput,
)
from fastdeploy.utils import get_host_ip
from fastdeploy.worker.output import Logprob, LogprobsTensors


class TestOpenAIServingCompletion(unittest.IsolatedAsyncioTestCase):

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
        output = {"tool_calls": "tool_call"}
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
                "metrics": {},
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
                "metrics": {},
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

    def setUp(self):
        """
        Set up the test environment by creating an instance of the OpenAIServingCompletion class using Mock.
        """
        self.mock_engine = Mock()
        self.serving_completion = OpenAIServingCompletion(
            self.mock_engine,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=10,
        )

    def test_build_prompt_logprobs_basic(self):
        """Test basic functionality of _build_prompt_logprobs"""
        # Create mock data
        num_prompt_tokens = 2
        num_logprobs = 3

        # Create tensors
        token_ids = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], dtype=paddle.float32)
        ranks = paddle.to_tensor([1, 2], dtype=paddle.int64)

        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        # Mock the data processor
        with patch.object(
            self.serving_completion.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["token1", "token2", "token3", "token4", "token5", "token6"]

            result = self.serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

            # Verify result structure (first element is None, then actual results)
            self.assertEqual(len(result), num_prompt_tokens + 1)
            self.assertIsNone(result[0])

            # Check first position (index 1 since index 0 is None)
            first_pos_result = result[1]
            self.assertEqual(len(first_pos_result), num_logprobs)

            # Check token IDs and logprobs for first position
            expected_tokens = [1, 2, 3]
            expected_logprobs = [float(logprobs[0][i]) for i in range(num_logprobs)]
            expected_ranks = [1, 1, 2]  # First token uses rank from ranks tensor, then topk ranks start from 1

            for i, token_id in enumerate(expected_tokens):
                self.assertIn(token_id, first_pos_result)
                self.assertIsInstance(first_pos_result[token_id], Logprob)
                self.assertEqual(first_pos_result[token_id].logprob, expected_logprobs[i])
                self.assertEqual(first_pos_result[token_id].rank, expected_ranks[i])
                self.assertEqual(first_pos_result[token_id].decoded_token, f"token{i+1}")

    def test_build_prompt_logprobs_with_all_logprobs(self):
        """Test _build_prompt_logprobs with num_prompt_logprobs=-1 (all logprobs)"""
        num_prompt_tokens = 1
        num_logprobs = 2

        token_ids = paddle.to_tensor([[10, 20]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-1.0, -2.0]], dtype=paddle.float32)
        ranks = paddle.to_tensor([0], dtype=paddle.int64)

        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        with patch.object(
            self.serving_completion.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["hello", "world"]

            result = self.serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, -1, True)

            self.assertEqual(len(result), num_prompt_tokens + 1)
            self.assertIsNone(result[0])
            first_pos_result = result[1]
            self.assertEqual(len(first_pos_result), num_logprobs)

            # Verify all logprobs are included when num_prompt_logprobs=-1
            for token_id in first_pos_result:
                self.assertIn(token_id, [10, 20])

    def test_build_prompt_logprobs_single_token(self):
        """Test _build_prompt_logprobs with single prompt token"""
        num_prompt_tokens = 1
        num_logprobs = 1

        token_ids = paddle.to_tensor([[100]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-0.5]], dtype=paddle.float32)
        ranks = paddle.to_tensor([1], dtype=paddle.int64)

        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        with patch.object(
            self.serving_completion.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.return_value = "single_token"

            result = self.serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

            self.assertEqual(len(result), num_prompt_tokens + 1)
            self.assertIsNone(result[0])
            first_pos_result = result[1]
            self.assertEqual(len(first_pos_result), num_logprobs)

            # Check the single token
            self.assertIn(100, first_pos_result)
            self.assertEqual(first_pos_result[100].logprob, -0.5)
            self.assertEqual(first_pos_result[100].rank, 1)
            self.assertEqual(first_pos_result[100].decoded_token, "single_token")

    def test_build_prompt_logprobs_multiple_positions(self):
        """Test _build_prompt_logprobs with multiple prompt positions"""
        num_prompt_tokens = 3
        num_logprobs = 2

        token_ids = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=paddle.float32)
        ranks = paddle.to_tensor([1, 2, 3], dtype=paddle.int64)

        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        with patch.object(
            self.serving_completion.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["t1", "t2", "t3", "t4", "t5", "t6"]

            result = self.serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

            self.assertEqual(len(result), num_prompt_tokens + 1)
            self.assertIsNone(result[0])

            # Check each position (index + 1 since index 0 is None)
            for pos in range(num_prompt_tokens):
                pos_result = result[pos + 1]
                self.assertEqual(len(pos_result), num_logprobs)

                # Verify token IDs and their properties
                expected_tokens = [int(token_ids[pos][0]), int(token_ids[pos][1])]
                expected_ranks = [
                    ranks[pos],
                    1,
                ]  # First token uses rank from ranks tensor, second token uses topk rank 1

                for i, token_id in enumerate(expected_tokens):
                    self.assertIn(token_id, pos_result)
                    self.assertEqual(pos_result[token_id].logprob, float(logprobs[pos][i]))
                    self.assertEqual(pos_result[token_id].rank, expected_ranks[i])
                    self.assertEqual(pos_result[token_id].decoded_token, f"t{pos*2 + i + 1}")

    def test_build_prompt_logprobs_empty_tensors(self):
        """Test _build_prompt_logprobs with empty tensors"""
        num_prompt_tokens = 0
        num_logprobs = 0

        token_ids = paddle.to_tensor([], dtype=paddle.int64).reshape([0, 0])
        logprobs = paddle.to_tensor([], dtype=paddle.float32).reshape([0, 0])
        ranks = paddle.to_tensor([], dtype=paddle.int64)

        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        result = self.serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

        self.assertEqual(len(result), num_prompt_tokens + 1)
        self.assertIsNone(result[0])

    def test_make_logprob_dict(self):
        """Test the static method _make_logprob_dict"""
        logprobs = [-0.1, -0.2, -0.3]
        logprob_token_ids = [1, 2, 3]
        decoded_tokens = ["token1", "token2", "token3"]
        rank = 1
        num_logprobs = 3

        result = OpenAIServingCompletion._make_logprob_dict(
            logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs
        )

        self.assertEqual(len(result), num_logprobs)

        # Check first token (sampled token)
        self.assertIn(1, result)
        self.assertEqual(result[1].logprob, -0.1)
        self.assertEqual(result[1].rank, rank)  # rank of sampled token
        self.assertEqual(result[1].decoded_token, "token1")

        # Check other tokens - topk ranks start from 1
        expected_ranks = [rank, 1, 2]  # First token uses rank, then topk ranks
        for i, token_id in enumerate(logprob_token_ids):
            self.assertIn(token_id, result)
            self.assertEqual(result[token_id].logprob, logprobs[i])
            self.assertEqual(result[token_id].rank, expected_ranks[i])
            self.assertEqual(result[token_id].decoded_token, decoded_tokens[i])

    def test_make_logprob_dict_with_negative_num_logprobs(self):
        """Test _make_logprob_dict with num_logprobs=-1"""
        logprobs = [-0.1, -0.2]
        logprob_token_ids = [1, 2]
        decoded_tokens = ["token1", "token2"]
        rank = 1
        num_logprobs = -1

        result = OpenAIServingCompletion._make_logprob_dict(
            logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs
        )

        # Should include all logprobs when num_logprobs=-1
        self.assertEqual(len(result), len(logprobs))

        # Expected ranks: first token uses rank, second token uses topk rank 1
        expected_ranks = [rank, 1]

        for i, token_id in enumerate(logprob_token_ids):
            self.assertIn(token_id, result)
            self.assertEqual(result[token_id].logprob, logprobs[i])
            self.assertEqual(result[token_id].rank, expected_ranks[i])
            self.assertEqual(result[token_id].decoded_token, decoded_tokens[i])

    def test_make_logprob_dict_with_limited_logprobs(self):
        """Test _make_logprob_dict with fewer logprobs than available"""
        logprobs = [-0.1, -0.2, -0.3, -0.4]
        logprob_token_ids = [1, 2, 3, 4]
        decoded_tokens = ["token1", "token2", "token3", "token4"]
        rank = 2
        num_logprobs = 2

        result = OpenAIServingCompletion._make_logprob_dict(
            logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs
        )

        # When num_logprobs=2, we get the sampled token + 1 topk token
        self.assertEqual(len(result), 3)

        # Check sampled token (first token)
        self.assertIn(1, result)
        self.assertEqual(result[1].logprob, -0.1)
        self.assertEqual(result[1].rank, rank)
        self.assertEqual(result[1].decoded_token, "token1")

        # Check top-k token (second token)
        self.assertIn(2, result)
        self.assertEqual(result[2].logprob, -0.2)
        self.assertEqual(result[2].rank, 1)  # topk rank starts from 1
        self.assertEqual(result[2].decoded_token, "token2")

    async def test_completion_stream_generator_with_prompt_logprobs(self):
        """Test completion_stream_generator with prompt_logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(side_effect=lambda x, **kwargs: f"token_{x}")
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data with prompt_logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "prompt_logprobs": LogprobsTensors(
                    logprob_token_ids=paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.int64),
                    logprobs=paddle.to_tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]], dtype=paddle.float32
                    ),
                    selected_token_ranks=paddle.to_tensor([1, 2, 3], dtype=paddle.int64),
                ),
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with prompt_logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = 3
        mock_request.logprobs = None
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_streaming_response_tokens = 1
        mock_request.max_tokens = None
        mock_request.stream_options = Mock()
        mock_request.stream_options.include_usage = False
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result_generator = serving_completion.completion_stream_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Collect results
        results = []
        async for result in result_generator:
            results.append(result)

        # Verify results
        self.assertTrue(len(results) > 0)
        # Check that the first response contains prompt_logprobs
        self.assertIn("prompt_logprobs", results[0])

    async def test_completion_stream_generator_with_logprobs(self):
        """Test completion_stream_generator with logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(side_effect=lambda x, **kwargs: f"token_{x}")
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data with logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": [
                        [[100]],  # logprob_token_ids (nested properly)
                        [[-0.1]],  # logprobs (nested properly)
                        [[1]],  # sampled_token_ranks (nested properly)
                    ],
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = None
        mock_request.logprobs = 3
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_streaming_response_tokens = 1
        mock_request.max_tokens = None
        mock_request.stream_options = Mock()
        mock_request.stream_options.include_usage = False
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result_generator = serving_completion.completion_stream_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Collect results
        results = []
        async for result in result_generator:
            results.append(result)

        # Verify results
        self.assertTrue(len(results) > 0)
        # Check that the response contains logprobs
        self.assertIn("logprobs", results[0])

    async def test_completion_stream_generator_with_both_logprobs(self):
        """Test completion_stream_generator with both prompt_logprobs and logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data with both prompt_logprobs and logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "prompt_logprobs": LogprobsTensors(
                    logprob_token_ids=paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.int64),
                    logprobs=paddle.to_tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]], dtype=paddle.float32
                    ),
                    selected_token_ranks=paddle.to_tensor([1, 2, 3], dtype=paddle.int64),
                ),
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": [
                        [[100]],  # logprob_token_ids (nested properly)
                        [[-0.1]],  # logprobs (nested properly)
                        [[1]],  # sampled_token_ranks (nested properly)
                    ],
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with both logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = 3
        mock_request.logprobs = 3
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_streaming_response_tokens = 1
        mock_request.max_tokens = None
        mock_request.stream_options = Mock()
        mock_request.stream_options.include_usage = False
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result_generator = serving_completion.completion_stream_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Collect results
        results = []
        async for result in result_generator:
            results.append(result)

        # Verify results
        self.assertTrue(len(results) > 0)
        # Check that the response contains both prompt_logprobs and logprobs
        self.assertIn("prompt_logprobs", results[0])
        self.assertIn("logprobs", results[0])

    async def test_completion_stream_generator_without_logprobs(self):
        """Test completion_stream_generator without logprobs enabled"""
        import json

        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data without logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string to match expected type
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                    "tool_calls": None,
                    "reasoning_content": "",
                    "skipped": False,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request without logprobs
        mock_request = Mock()
        mock_request.prompt_logprobs = None
        mock_request.logprobs = None
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_streaming_response_tokens = 1
        mock_request.max_tokens = None
        mock_request.stream_options = Mock()
        mock_request.stream_options.include_usage = False
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result_generator = serving_completion.completion_stream_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Collect results
        results = []
        async for result in result_generator:
            results.append(result)

        # Verify results
        self.assertTrue(len(results) > 0)

        # Parse all results to check for logprobs fields
        found_prompt_logprobs = False
        found_logprobs = False
        prompt_logprobs_null = False
        logprobs_null = False

        for result in results:
            # Skip [DONE] messages
            if result.strip() == "[DONE]":
                continue

            # Extract JSON part from SSE format (data: {...})
            if result.startswith("data: "):
                json_str = result[6:]  # Remove "data: " prefix
                # Skip [DONE] messages in data format
                if json_str.strip() == "[DONE]":
                    continue
                parsed_result = json.loads(json_str)
            else:
                # Skip [DONE] messages without data prefix
                if result.strip() == "[DONE]":
                    continue
                parsed_result = json.loads(result)

            choice = parsed_result["choices"][0]

            # Check for prompt_logprobs
            if "prompt_logprobs" in choice:
                found_prompt_logprobs = True
                if choice["prompt_logprobs"] is None:
                    prompt_logprobs_null = True

            # Check for logprobs
            if "logprobs" in choice:
                found_logprobs = True
                if choice["logprobs"] is None:
                    logprobs_null = True

        # Verify that both fields are found and null when not requested
        self.assertTrue(found_prompt_logprobs, "prompt_logprobs field should be present")
        self.assertTrue(found_logprobs, "logprobs field should be present")
        self.assertTrue(prompt_logprobs_null, "prompt_logprobs should be null when not requested")
        self.assertTrue(logprobs_null, "logprobs should be null when not requested")

    async def test_completion_full_generator_with_prompt_logprobs(self):
        """Test completion_full_generator with prompt_logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data with prompt_logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "prompt_logprobs": LogprobsTensors(
                    logprob_token_ids=paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.int64),
                    logprobs=paddle.to_tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]], dtype=paddle.float32
                    ),
                    selected_token_ranks=paddle.to_tensor([1, 2, 3], dtype=paddle.int64),
                ),
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with prompt_logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = 3
        mock_request.logprobs = None
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_tokens = None
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result = await serving_completion.completion_full_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Verify results
        self.assertIsNotNone(result)
        # Check that the response contains prompt_logprobs
        self.assertIsNotNone(result.choices[0].prompt_logprobs)
        self.assertEqual(len(result.choices[0].prompt_logprobs), 4)  # 3 prompt tokens + 1 None element
        self.assertIsNone(result.choices[0].prompt_logprobs[0])  # First element should be None

    async def test_completion_full_generator_with_logprobs(self):
        """Test completion_full_generator with logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()
        # Create mock response data with logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": [
                        [[100]],  # logprob_token_ids (nested properly)
                        [[-0.1]],  # logprobs (nested properly)
                        [[1]],  # sampled_token_ranks (nested properly)
                    ],
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = None
        mock_request.logprobs = 3
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_tokens = None
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result = await serving_completion.completion_full_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Verify results
        self.assertIsNotNone(result)
        # Check that the response contains logprobs
        self.assertIsNotNone(result.choices[0].logprobs)
        self.assertEqual(len(result.choices[0].logprobs.tokens), 1)  # 1 completion token

    async def test_completion_full_generator_with_both_logprobs(self):
        """Test completion_full_generator with both prompt_logprobs and logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data with both prompt_logprobs and logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "prompt_logprobs": LogprobsTensors(
                    logprob_token_ids=paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=paddle.int64),
                    logprobs=paddle.to_tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]], dtype=paddle.float32
                    ),
                    selected_token_ranks=paddle.to_tensor([1, 2, 3], dtype=paddle.int64),
                ),
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": [
                        [[100]],  # logprob_token_ids (properly nested)
                        [[-0.1]],  # logprobs (properly nested)
                        [[1]],  # sampled_token_ranks (properly nested)
                    ],
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request with both logprobs enabled
        mock_request = Mock()
        mock_request.prompt_logprobs = 3
        mock_request.logprobs = 3
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_tokens = None
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result = await serving_completion.completion_full_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Verify results
        self.assertIsNotNone(result)
        # Check that the response contains both prompt_logprobs and logprobs
        self.assertIsNotNone(result.choices[0].prompt_logprobs)
        self.assertIsNotNone(result.choices[0].logprobs)
        self.assertEqual(len(result.choices[0].prompt_logprobs), 4)  # 3 prompt tokens + 1 None element
        self.assertIsNone(result.choices[0].prompt_logprobs[0])  # First element should be None
        self.assertEqual(len(result.choices[0].logprobs.tokens), 1)  # 1 completion token

    async def test_completion_full_generator_without_logprobs(self):
        """Test completion_full_generator without logprobs enabled"""
        # Mock the engine client and its dependencies
        mock_engine_client = Mock()
        mock_engine_client.semaphore = Mock()
        mock_engine_client.semaphore.acquire = AsyncMock()
        mock_engine_client.semaphore.release = Mock()
        mock_engine_client.connection_manager = AsyncMock()
        mock_engine_client.data_processor = Mock()
        mock_engine_client.ori_vocab_size = 1000
        mock_engine_client.check_model_weight_status.return_value = False
        mock_engine_client.check_health.return_value = (True, "Healthy")

        # Mock the data_processor methods
        mock_engine_client.data_processor.process_logprob_response = Mock(
            side_effect=lambda x, **kwargs: f"token_{x[0] if isinstance(x, list) else x}"
        )
        mock_engine_client.data_processor.process_response_dict = Mock()

        # Mock connection manager get_connection method
        mock_dealer = Mock()
        mock_dealer.write = Mock()
        mock_response_queue = AsyncMock()

        # Create mock response data without logprobs
        mock_response_data = [
            {
                "request_id": "test_request_0",
                "error_code": 200,
                "metrics": {
                    "arrival_time": 1234567890,
                    "inference_start_time": 1234567890,
                    "first_token_time": 1234567890,
                },
                "outputs": {
                    "text": "Hello",
                    "token_ids": [100],
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": "1",  # Changed to string
                    "num_cache_tokens": 0,
                    "num_image_tokens": 0,
                    "reasoning_token_num": 0,
                },
                "finished": True,
            }
        ]

        mock_response_queue.get.return_value = mock_response_data
        mock_engine_client.connection_manager.get_connection.return_value = (mock_dealer, mock_response_queue)

        # Create serving completion instance
        serving_completion = OpenAIServingCompletion(mock_engine_client, None, "pid", None, 360)

        # Create mock request without logprobs
        mock_request = Mock()
        mock_request.prompt_logprobs = None
        mock_request.logprobs = None
        mock_request.include_draft_logprobs = False
        mock_request.return_token_ids = True
        mock_request.include_stop_str_in_output = False
        mock_request.max_tokens = None
        mock_request.n = 1
        mock_request.echo = False  # Disable echo to avoid the echo logic issue

        # Call the method
        result = await serving_completion.completion_full_generator(
            request=mock_request,
            num_choices=1,
            request_id="test_request",
            created_time=1234567890,
            model_name="test_model",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=["hello", "world"],
            max_tokens_list=[100],
        )

        # Verify results
        self.assertIsNotNone(result)
        # Check that the response contains null logprobs fields
        self.assertIsNone(result.choices[0].prompt_logprobs)
        self.assertIsNone(result.choices[0].logprobs)


if __name__ == "__main__":
    unittest.main()
