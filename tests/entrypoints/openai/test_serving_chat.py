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

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import paddle

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.worker.output import Logprob, LogprobsTensors


class TestOpenAIServingCompletion(unittest.IsolatedAsyncioTestCase):

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
            self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["token1", "token2", "token3", "token4", "token5", "token6"]

            result = self.chat_completion_handler._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

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
            self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["hello", "world"]

            result = self.chat_completion_handler._build_prompt_logprobs(prompt_logprobs_tensors, -1, True)

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
            self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.return_value = "single_token"

            result = self.chat_completion_handler._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

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
            self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
        ) as mock_decode:
            mock_decode.side_effect = ["t1", "t2", "t3", "t4", "t5", "t6"]

            result = self.chat_completion_handler._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

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

        result = self.chat_completion_handler._build_prompt_logprobs(prompt_logprobs_tensors, num_logprobs, True)

        self.assertEqual(len(result), num_prompt_tokens + 1)
        self.assertIsNone(result[0])

    def test_make_logprob_dict(self):
        """Test the static method _make_logprob_dict"""
        logprobs = [-0.1, -0.2, -0.3]
        logprob_token_ids = [1, 2, 3]
        decoded_tokens = ["token1", "token2", "token3"]
        rank = 1
        num_logprobs = 3

        result = OpenAIServingChat._make_logprob_dict(logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs)

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

        result = OpenAIServingChat._make_logprob_dict(logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs)

        # Should include all logprobs when num_logprobs=-1
        self.assertEqual(len(result), len(logprobs))

        # Expected ranks: first token uses rank, second token uses topk rank 1
        expected_ranks = [rank, 1]

        for i, token_id in enumerate(logprob_token_ids):
            self.assertIn(token_id, result)
            self.assertEqual(result[token_id].logprob, logprobs[i])
            self.assertEqual(result[token_id].rank, expected_ranks[i])
            self.assertEqual(result[token_id].decoded_token, decoded_tokens[i])

    def test_make_logprob_dict_partial_logprobs(self):
        """Test _make_logprob_dict with fewer logprobs than available"""
        logprobs = [-0.1, -0.2, -0.3, -0.4]
        logprob_token_ids = [1, 2, 3, 4]
        decoded_tokens = ["token1", "token2", "token3", "token4"]
        rank = 2
        num_logprobs = 2

        result = OpenAIServingChat._make_logprob_dict(logprobs, logprob_token_ids, decoded_tokens, rank, num_logprobs)

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

    async def test_chat_completion_stream_generator_with_prompt_logprobs(self):
        """Test chat_completion_stream_generator with prompt_logprobs enabled"""
        # Create mock request with prompt_logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], prompt_logprobs=3, logprobs=False, stream=True
        )

        request_id = "test_request_123"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with prompt_logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": LogprobsTensors(
                logprob_token_ids=paddle.to_tensor([[1, 2, 3, 4]], dtype=paddle.int64),
                logprobs=paddle.to_tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=paddle.float32),
                selected_token_ranks=paddle.to_tensor([1], dtype=paddle.int64),
            ),
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": None,
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.side_effect = ["Hello", "world", "test", "token"]

                # Execute the generator
                results = []
                async for chunk in self.chat_completion_handler.chat_completion_stream_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                ):
                    results.append(chunk)

                # Verify that prompt_logprobs are included in the response
                self.assertGreater(len(results), 0)

                # Check that the first chunk contains prompt_logprobs
                first_chunk_data = json.loads(results[0].replace("data: ", "").strip())
                self.assertIn("choices", first_chunk_data)
                self.assertEqual(len(first_chunk_data["choices"]), 1)

                choice = first_chunk_data["choices"][0]
                self.assertIn("prompt_logprobs", choice)
                self.assertIsNotNone(choice["prompt_logprobs"])

                # Verify prompt_logprobs structure
                prompt_logprobs = choice["prompt_logprobs"]
                self.assertIsInstance(prompt_logprobs, list)
                self.assertGreater(len(prompt_logprobs), 0)

    async def test_chat_completion_stream_generator_with_logprobs(self):
        """Test chat_completion_stream_generator with logprobs enabled"""
        # Create mock request with logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            prompt_logprobs=None,
            logprobs=True,
            top_logprobs=2,
            stream=True,
        )

        request_id = "test_request_456"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": None,
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": [
                    [[5, 6]],  # logprob_token_ids
                    [[-0.1, -0.2]],  # logprobs
                    [1],  # sampled_token_ranks
                ],
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        # Mock the data processor for logprob response
        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.return_value = "Hi"

                # Execute the generator
                results = []
                async for chunk in self.chat_completion_handler.chat_completion_stream_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                ):
                    results.append(chunk)

                # Verify that logprobs are included in the response
                self.assertGreater(len(results), 0)

                # Find chunks that contain logprobs
                logprobs_chunks = []
                for result in results:
                    if "logprobs" in result:
                        logprobs_chunks.append(result)

                self.assertGreater(len(logprobs_chunks), 0)

                # Check logprobs structure in response
                for chunk in logprobs_chunks:
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        choice = chunk_data["choices"][0]
                        if "logprobs" in choice:
                            self.assertIsNotNone(choice["logprobs"])

    async def test_chat_completion_stream_generator_with_both_logprobs(self):
        """Test chat_completion_stream_generator with both prompt_logprobs and logprobs enabled"""
        # Create mock request with both logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            prompt_logprobs=2,
            logprobs=True,
            top_logprobs=2,
            stream=True,
        )

        request_id = "test_request_789"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with both logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": LogprobsTensors(
                logprob_token_ids=paddle.to_tensor([[1, 2, 3]], dtype=paddle.int64),
                logprobs=paddle.to_tensor([[-0.1, -0.2, -0.3]], dtype=paddle.float32),
                selected_token_ranks=paddle.to_tensor([1], dtype=paddle.int64),
            ),
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": [
                    [[5, 6]],  # logprob_token_ids
                    [[-0.1, -0.2]],  # logprobs
                    [1],  # sampled_token_ranks
                ],
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.return_value = "Hi"

                # Execute the generator
                results = []
                async for chunk in self.chat_completion_handler.chat_completion_stream_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                ):
                    results.append(chunk)

                # Verify that both types of logprobs are included
                self.assertGreater(len(results), 0)

                # Check for prompt_logprobs
                first_chunk_data = json.loads(results[0].replace("data: ", "").strip())
                self.assertIn("choices", first_chunk_data)
                choice = first_chunk_data["choices"][0]
                self.assertIn("prompt_logprobs", choice)
                self.assertIsNotNone(choice["prompt_logprobs"])

                # Check for logprobs in subsequent chunks
                logprobs_found = False
                for result in results:
                    # Skip [DONE] message
                    if result.strip() == "data: [DONE]":
                        continue
                    chunk_data = json.loads(result.replace("data: ", "").strip())
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        choice = chunk_data["choices"][0]
                        if "logprobs" in choice and choice["logprobs"] is not None:
                            logprobs_found = True
                            break

                self.assertTrue(logprobs_found, "logprobs should be found in response chunks")

    async def test_chat_completion_stream_generator_without_logprobs(self):
        """Test chat_completion_stream_generator without logprobs enabled"""
        # Create mock request without logprobs
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], prompt_logprobs=None, logprobs=False, stream=True
        )

        request_id = "test_request_no_logprobs"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response without logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": None,
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": None,
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            # Execute the generator
            results = []
            async for chunk in self.chat_completion_handler.chat_completion_stream_generator(
                request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
            ):
                results.append(chunk)

                # Verify that logprobs are not included in the response
            self.assertGreater(len(results), 0)

            for result in results:
                # Skip [DONE] message
                if result.strip() == "data: [DONE]":
                    continue

                chunk_data = json.loads(result.replace("data: ", "").strip())
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    choice = chunk_data["choices"][0]
                    # prompt_logprobs should be None when not requested
                    self.assertIsNone(choice.get("prompt_logprobs"))
                    # logprobs should be None when not requested
                    self.assertIsNone(choice.get("logprobs"))

    async def test_chat_completion_full_generator_with_prompt_logprobs(self):
        """Test chat_completion_full_generator with prompt_logprobs enabled"""
        # Create mock request with prompt_logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], prompt_logprobs=3, logprobs=False, stream=False
        )

        request_id = "test_request_full_123"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with prompt_logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": LogprobsTensors(
                logprob_token_ids=paddle.to_tensor([[1, 2, 3, 4]], dtype=paddle.int64),
                logprobs=paddle.to_tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=paddle.float32),
                selected_token_ranks=paddle.to_tensor([1], dtype=paddle.int64),
            ),
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": None,
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.side_effect = ["Hello", "world", "test", "token"]

                # Execute the generator
                result = await self.chat_completion_handler.chat_completion_full_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                )

                # Verify that prompt_logprobs are included in the response
                self.assertIsNotNone(result)
                self.assertIn("choices", result.model_dump())
                self.assertGreater(len(result.choices), 0)

                choice = result.choices[0]
                self.assertIn("prompt_logprobs", choice.model_dump())
                self.assertIsNotNone(choice.prompt_logprobs)

                # Verify prompt_logprobs structure
                prompt_logprobs = choice.prompt_logprobs
                self.assertIsInstance(prompt_logprobs, list)
                self.assertGreater(len(prompt_logprobs), 0)

    async def test_chat_completion_full_generator_with_logprobs(self):
        """Test chat_completion_full_generator with logprobs enabled"""
        # Create mock request with logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            prompt_logprobs=None,
            logprobs=True,
            top_logprobs=2,
            stream=False,
        )

        request_id = "test_request_full_456"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": None,
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": [
                    [[5, 6]],  # logprob_token_ids
                    [[-0.1, -0.2]],  # logprobs
                    [1],  # sampled_token_ranks
                ],
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        # Mock the data processor for logprob response
        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.return_value = "Hi"

                # Execute the generator
                result = await self.chat_completion_handler.chat_completion_full_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                )

                # Verify that logprobs are included in the response
                self.assertIsNotNone(result)
                self.assertIn("choices", result.model_dump())
                self.assertGreater(len(result.choices), 0)

                choice = result.choices[0]
                self.assertIn("logprobs", choice.model_dump())
                self.assertIsNotNone(choice.logprobs)

    async def test_chat_completion_full_generator_with_both_logprobs(self):
        """Test chat_completion_full_generator with both prompt_logprobs and logprobs enabled"""
        # Create mock request with both logprobs enabled
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            prompt_logprobs=2,
            logprobs=True,
            top_logprobs=2,
            stream=False,
        )

        request_id = "test_request_full_789"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response with both logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": LogprobsTensors(
                logprob_token_ids=paddle.to_tensor([[1, 2, 3]], dtype=paddle.int64),
                logprobs=paddle.to_tensor([[-0.1, -0.2, -0.3]], dtype=paddle.float32),
                selected_token_ranks=paddle.to_tensor([1], dtype=paddle.int64),
            ),
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": [
                    [[5, 6]],  # logprob_token_ids
                    [[-0.1, -0.2]],  # logprobs
                    [1],  # sampled_token_ranks
                ],
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            with patch.object(
                self.chat_completion_handler.engine_client.data_processor, "process_logprob_response"
            ) as mock_decode:
                mock_decode.return_value = "Hi"

                # Execute the generator
                result = await self.chat_completion_handler.chat_completion_full_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                )

                # Verify that both types of logprobs are included
                self.assertIsNotNone(result)
                self.assertIn("choices", result.model_dump())
                self.assertGreater(len(result.choices), 0)

                choice = result.choices[0]

                # Check for prompt_logprobs
                self.assertIn("prompt_logprobs", choice.model_dump())
                self.assertIsNotNone(choice.prompt_logprobs)

                # Check for logprobs
                self.assertIn("logprobs", choice.model_dump())
                self.assertIsNotNone(choice.logprobs)

    async def test_chat_completion_full_generator_without_logprobs(self):
        """Test chat_completion_full_generator without logprobs enabled"""
        # Create mock request without logprobs
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], prompt_logprobs=None, logprobs=False, stream=False
        )

        request_id = "test_request_full_no_logprobs"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager and response queue
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Create mock response without logprobs data
        mock_response = {
            "request_id": f"{request_id}_0",
            "error_code": 200,
            "metrics": {
                "first_token_time": 1234567890,
                "inference_start_time": 1234567880,
                "arrival_time": 1234567890,
                "request_start_time": 1234567870,
            },
            "prompt_logprobs": None,
            "outputs": {
                "token_ids": [5],
                "text": "Hi",
                "top_logprobs": None,
                "draft_top_logprobs": None,
                "multipart": [{"type": "text", "text": "Hi"}],
            },
            "finished": True,
            "num_cached_tokens": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

        mock_response_queue.get.return_value = mock_response

        # Mock the connection manager
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator():
            yield mock_response

        mock_response_processor.process_response_chat.return_value = mock_async_generator()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            # Execute the generator
            result = await self.chat_completion_handler.chat_completion_full_generator(
                request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
            )

            # Verify that logprobs are not included in the response
            self.assertIsNotNone(result)
            self.assertIn("choices", result.model_dump())
            self.assertGreater(len(result.choices), 0)

            choice = result.choices[0]
            # prompt_logprobs should be None when not requested
            self.assertIsNone(choice.prompt_logprobs)
            # logprobs should be None when not requested
            self.assertIsNone(choice.logprobs)

    async def test_create_chat_completion_cancelled_error(self):
        """Test asyncio.CancelledError handling in create_chat_completion method"""
        # Create mock request
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}], stream=False)

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock format_and_add_data to raise CancelledError
        self.chat_completion_handler.engine_client.format_and_add_data = AsyncMock(
            side_effect=asyncio.CancelledError("Test cancellation during data formatting")
        )

        # Mock the abort method that should be called when CancelledError occurs
        self.chat_completion_handler.engine_client.abort = AsyncMock()

        # Execute and verify that CancelledError is handled properly
        # The CancelledError should be caught and handled, not re-raised
        try:
            await self.chat_completion_handler.create_chat_completion(request)
        except asyncio.CancelledError:
            # This should not happen as CancelledError should be caught and handled
            self.fail("CancelledError should be caught and handled, not re-raised")

        # Verify abort was called despite the cancellation
        self.chat_completion_handler.engine_client.abort.assert_called_once()

    async def test_chat_completion_stream_generator_cancelled_error(self):
        """Test asyncio.CancelledError handling in chat_completion_stream_generator method"""
        # Create mock request
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}], stream=True)

        request_id = "test_cancel_request"
        model_name = "test_model"
        prompt_token_ids = [1, 2, 3]
        prompt_tokens = "Hello world"

        # Mock the connection manager
        mock_dealer = MagicMock()
        mock_response_queue = AsyncMock()

        # Mock get_connection to return normally
        self.chat_completion_handler.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        # Mock the semaphore
        self.chat_completion_handler.engine_client.semaphore = MagicMock()
        self.chat_completion_handler.engine_client.semaphore.acquire = AsyncMock(return_value=True)
        self.chat_completion_handler.engine_client.semaphore.release = MagicMock()

        # Mock the model weight status check
        self.chat_completion_handler.engine_client.check_model_weight_status = Mock(return_value=False)

        # Mock the response processor to raise CancelledError during processing
        mock_response_processor = MagicMock()
        mock_response_processor.enable_multimodal_content.return_value = False

        async def mock_async_generator_with_cancel():
            # Simulate some normal response first
            yield {
                "request_id": f"{request_id}_0",
                "error_code": 200,
                "metrics": {
                    "first_token_time": 1234567890,
                    "inference_start_time": 1234567880,
                    "arrival_time": 1234567890,
                    "request_start_time": 1234567870,
                },
                "prompt_logprobs": None,
                "outputs": {
                    "token_ids": [5],
                    "text": "Hi",
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "multipart": [{"type": "text", "text": "Hi"}],
                },
                "finished": False,
                "num_cached_tokens": 0,
                "num_input_image_tokens": 0,
                "num_input_video_tokens": 0,
            }
            # Then raise CancelledError
            raise asyncio.CancelledError("Test cancellation during streaming")

        mock_response_processor.process_response_chat.return_value = mock_async_generator_with_cancel()

        # Mock the cleanup method
        self.chat_completion_handler.engine_client.connection_manager.cleanup_request = AsyncMock()

        # Mock the abort method that should be called when CancelledError occurs
        self.chat_completion_handler.engine_client.abort = AsyncMock()

        with patch(
            "fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor", return_value=mock_response_processor
        ):
            # Execute the generator and verify CancelledError handling
            # The CancelledError should be caught and handled, not re-raised
            chunks = []
            try:
                async for chunk in self.chat_completion_handler.chat_completion_stream_generator(
                    request, request_id, model_name, prompt_token_ids, prompt_tokens, max_tokens=100
                ):
                    chunks.append(chunk)
            except asyncio.CancelledError:
                # This should not happen as CancelledError should be caught and handled
                self.fail("CancelledError should be caught and handled, not re-raised")

            # Should have received at least one chunk before cancellation
            self.assertGreaterEqual(len(chunks), 1)
            self.assertIsNotNone(chunks[0])

            # Verify cleanup and abort were called despite the cancellation
            self.chat_completion_handler.engine_client.connection_manager.cleanup_request.assert_called_once()
            self.chat_completion_handler.engine_client.abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
