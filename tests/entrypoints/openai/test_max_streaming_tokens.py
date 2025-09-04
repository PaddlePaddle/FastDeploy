import json
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion


class TestMaxStreamingResponseTokens(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine_client = Mock()
        self.engine_client.connection_initialized = False
        self.engine_client.connection_manager = AsyncMock()
        self.engine_client.connection_manager.initialize = AsyncMock()
        self.engine_client.connection_manager.get_connection = AsyncMock()
        self.engine_client.connection_manager.cleanup_request = AsyncMock()
        self.engine_client.semaphore = Mock()
        self.engine_client.semaphore.acquire = AsyncMock()
        self.engine_client.semaphore.release = Mock()
        self.engine_client.data_processor = Mock()
        self.engine_client.is_master = True

        self.chat_serving = OpenAIServingChat(
            engine_client=self.engine_client,
            models=None,
            pid=123,
            ips=None,
            max_waiting_time=30,
            chat_template="default",
            enable_mm_output=False,
            tokenizer_base_url=None,
        )

        self.completion_serving = OpenAIServingCompletion(
            engine_client=self.engine_client, models=None, pid=123, ips=None, max_waiting_time=30
        )

    def test_metadata_parameter_setting(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            metadata={"max_streaming_response_tokens": 100},
        )

        max_tokens = (
            request.max_streaming_response_tokens
            if request.max_streaming_response_tokens is not None
            else (request.metadata or {}).get("max_streaming_response_tokens", 1)
        )

        self.assertEqual(max_tokens, 100)

    def test_default_value(self):
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        max_tokens = (
            request.max_streaming_response_tokens
            if request.max_streaming_response_tokens is not None
            else (request.metadata or {}).get("max_streaming_response_tokens", 1)
        )

        self.assertEqual(max_tokens, 1)

    def test_edge_case_zero_value(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            max_streaming_response_tokens=0,
        )

        max_streaming_response_tokens = (
            request.max_streaming_response_tokens
            if request.max_streaming_response_tokens is not None
            else (request.metadata or {}).get("max_streaming_response_tokens", 1)
        )
        max_streaming_response_tokens = max(1, max_streaming_response_tokens)

        self.assertEqual(max_streaming_response_tokens, 1)

    @patch("fastdeploy.entrypoints.openai.serving_chat.api_server_logger")
    @patch("fastdeploy.entrypoints.openai.serving_chat.ChatResponseProcessor")
    async def test_integration_with_chat_stream_generator(self, mock_processor_class, mock_logger):
        response_data = [
            {
                "outputs": {"token_ids": [1], "text": "a", "top_logprobs": None},
                "metrics": {"first_token_time": 0.1, "inference_start_time": 0.1},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [2], "text": "b", "top_logprobs": None},
                "metrics": {"arrival_time": 0.2, "first_token_time": None},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [3], "text": "c", "top_logprobs": None},
                "metrics": {"arrival_time": 0.3, "first_token_time": None},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [4], "text": "d", "top_logprobs": None},
                "metrics": {"arrival_time": 0.4, "first_token_time": None},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [5], "text": "e", "top_logprobs": None},
                "metrics": {"arrival_time": 0.5, "first_token_time": None},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [6], "text": "f", "top_logprobs": None},
                "metrics": {"arrival_time": 0.6, "first_token_time": None},
                "finished": False,
            },
            {
                "outputs": {"token_ids": [7], "text": "g", "top_logprobs": None},
                "metrics": {"arrival_time": 0.7, "first_token_time": None, "request_start_time": 0.1},
                "finished": True,
            },
        ]

        mock_response_queue = AsyncMock()
        mock_response_queue.get.side_effect = response_data

        mock_dealer = Mock()
        mock_dealer.write = Mock()

        # Mock the connection manager call
        self.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        mock_processor_instance = Mock()

        async def mock_process_response_chat_single(response, stream, enable_thinking, include_stop_str_in_output):
            yield response

        mock_processor_instance.process_response_chat = mock_process_response_chat_single
        mock_processor_instance.enable_multimodal_content = Mock(return_value=False)
        mock_processor_class.return_value = mock_processor_instance

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            max_streaming_response_tokens=3,
        )

        generator = self.chat_serving.chat_completion_stream_generator(
            request=request,
            request_id="test-request-id",
            model_name="test-model",
            prompt_token_ids=[1, 2, 3],
            text_after_process="Hello",
        )

        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0, "No chucks!")

        parsed_chunks = []
        for i, chunk_str in enumerate(chunks):
            if i == 0:
                continue
            if chunk_str.startswith("data: ") and chunk_str.endswith("\n\n"):
                json_part = chunk_str[6:-2]
                if json_part == "[DONE]":
                    parsed_chunks.append({"type": "done", "raw": chunk_str})
                    break
                try:
                    chunk_dict = json.loads(json_part)
                    parsed_chunks.append(chunk_dict)
                except json.JSONDecodeError as e:
                    self.fail(f"Cannot parser {i+1} chunck, JSON: {e}\n origin string: {repr(chunk_str)}")
            else:
                self.fail(f"{i+1} chunk is unexcepted 'data: JSON\\n\\n': {repr(chunk_str)}")
        for chunk_dict in parsed_chunks:
            choices_list = chunk_dict["choices"]
            if choices_list[-1].get("finish_reason") is not None:
                break
            else:
                self.assertEqual(len(choices_list), 3, f"Chunk {chunk_dict} should has three choices")

        found_done = any("[DONE]" in chunk for chunk in chunks)
        self.assertTrue(found_done, "Not Receive '[DONE]'")

    @patch("fastdeploy.entrypoints.openai.serving_completion.api_server_logger")
    async def test_integration_with_completion_stream_generator(self, mock_logger):
        response_data = [
            [
                {
                    "request_id": "test-request-id-0",
                    "outputs": {"token_ids": [1], "text": "a", "top_logprobs": None},
                    "metrics": {"first_token_time": 0.1, "inference_start_time": 0.1},
                    "finished": False,
                },
                {
                    "request_id": "test-request-id-0",
                    "outputs": {"token_ids": [2], "text": "b", "top_logprobs": None},
                    "metrics": {"arrival_time": 0.2, "first_token_time": None},
                    "finished": False,
                },
            ],
            [
                {
                    "request_id": "test-request-id-0",
                    "outputs": {"token_ids": [7], "text": "g", "top_logprobs": None},
                    "metrics": {"arrival_time": 0.7, "first_token_time": None, "request_start_time": 0.1},
                    "finished": True,
                }
            ],
        ]

        mock_response_queue = AsyncMock()
        mock_response_queue.get.side_effect = response_data

        mock_dealer = Mock()
        mock_dealer.write = Mock()

        # Mock the connection manager call
        self.engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        request = CompletionRequest(model="test-model", prompt="Hello", stream=True, max_streaming_response_tokens=3)

        generator = self.completion_serving.completion_stream_generator(
            request=request,
            num_choices=1,
            request_id="test-request-id",
            model_name="test-model",
            created_time=11,
            prompt_batched_token_ids=[[1, 2, 3]],
            text_after_process_list=["Hello"],
        )

        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0, "No chucks!")

        parsed_chunks = []
        for i, chunk_str in enumerate(chunks):
            if chunk_str.startswith("data: ") and chunk_str.endswith("\n\n"):
                json_part = chunk_str[6:-2]
                if json_part == "[DONE]":
                    break
                try:
                    chunk_dict = json.loads(json_part)
                    parsed_chunks.append(chunk_dict)
                except json.JSONDecodeError as e:
                    self.fail(f"Cannot parser {i+1} chunck, JSON: {e}\n origin string: {repr(chunk_str)}")
            else:
                self.fail(f"{i+1} chunk is unexcepted 'data: JSON\\n\\n': {repr(chunk_str)}")
        self.assertEqual(len(parsed_chunks), 1)
        for chunk_dict in parsed_chunks:
            choices_list = chunk_dict["choices"]
            self.assertEqual(len(choices_list), 3, f"Chunk {chunk_dict} should has three choices")
            self.assertEqual(
                choices_list[-1].get("finish_reason"), "stop", f"Chunk {chunk_dict} should has stop reason"
            )

        found_done = any("[DONE]" in chunk for chunk in chunks)
        self.assertTrue(found_done, "Not Receive '[DONE]'")


if __name__ == "__main__":
    unittest.main()
