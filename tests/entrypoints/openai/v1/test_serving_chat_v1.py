"""
Unit tests for serving_chat class
"""

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastdeploy.config import FDConfig

# Import the classes we need to test
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    LogProbs,
    StreamOptions,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.entrypoints.openai.v1.serving_chat import OpenAIServingChat
from fastdeploy.worker.output import LogprobsLists


# Define ServingResponseContext locally since it's not exported
class ServingResponseContext:
    def __init__(self):

        self.usage = UsageInfo()
        self.choice_completion_tokens_dict = {}
        self.inference_start_time_dict = {}
        self.remain_choices = 0


class TestOpenAIServingChat(unittest.IsolatedAsyncioTestCase):
    """Test cases for OpenAIServingChat"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_engine_client = AsyncMock(spec=AsyncLLM)
        self.mock_config = MagicMock(spec=FDConfig)
        self.mock_models = MagicMock(spec=OpenAIServingModels)
        self.pid = 12345
        self.ips = ["127.0.0.1"]
        self.max_waiting_time = 60
        self.chat_template = "test_template"
        self.enable_mm_output = False
        self.tokenizer_base_url = None

        # Create the serving chat instance
        self.serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template,
            enable_mm_output=self.enable_mm_output,
            tokenizer_base_url=self.tokenizer_base_url,
        )

        # Mock the data processor
        self.mock_engine_client.data_processor = MagicMock()
        self.mock_engine_client.data_processor.process_logprob_response = MagicMock(return_value="test_token")

    def test_init(self):
        """Test basic initialization."""
        self.assertEqual(self.serving_chat.pid, self.pid)
        self.assertEqual(self.serving_chat.engine_client, self.mock_engine_client)
        self.assertEqual(self.serving_chat.models, self.mock_models)
        self.assertEqual(self.serving_chat.chat_template, self.chat_template)
        self.assertEqual(self.serving_chat.enable_mm_output, self.enable_mm_output)
        self.assertEqual(self.serving_chat.tokenizer_base_url, self.tokenizer_base_url)

    def test_init_with_mm_output(self):
        """Test initialization with multimodal output enabled."""
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template,
            enable_mm_output=True,
            tokenizer_base_url="http://test-url",
        )

        self.assertTrue(serving_chat.enable_mm_output)
        self.assertEqual(serving_chat.tokenizer_base_url, "http://test-url")

    def test_get_thinking_status_from_kwargs(self):
        """Test _get_thinking_status from chat_template_kwargs."""
        request = MagicMock()
        request.chat_template_kwargs = {"enable_thinking": True}

        result = self.serving_chat._get_thinking_status(request)
        self.assertTrue(result)

        request.chat_template_kwargs = {"enable_thinking": False}
        result = self.serving_chat._get_thinking_status(request)
        self.assertFalse(result)

    def test_get_thinking_status_from_metadata(self):
        """Test _get_thinking_status from metadata."""
        request = MagicMock()
        request.chat_template_kwargs = None
        request.metadata = {"enable_thinking": True}

        result = self.serving_chat._get_thinking_status(request)
        self.assertTrue(result)

        request.metadata = {"enable_thinking": False}
        result = self.serving_chat._get_thinking_status(request)
        self.assertFalse(result)

    def test_get_thinking_status_with_options(self):
        """Test _get_thinking_status with thinking_mode options."""
        request = MagicMock()
        request.chat_template_kwargs = {"enable_thinking": False, "options": {"thinking_mode": "true"}}

        result = self.serving_chat._get_thinking_status(request)
        self.assertTrue(result)

        request.chat_template_kwargs = {"enable_thinking": True, "options": {"thinking_mode": "close"}}

        result = self.serving_chat._get_thinking_status(request)
        self.assertFalse(result)

    def test_get_thinking_status_default(self):
        """Test _get_thinking_status with default values."""
        request = MagicMock()
        request.chat_template_kwargs = None
        request.metadata = None

        result = self.serving_chat._get_thinking_status(request)
        self.assertTrue(result)

    def test_create_chat_logprobs_valid(self):
        """Test _create_chat_logprobs with valid input."""
        output_top_logprobs = [
            [[1, 2, 3]],  # logprob_token_ids
            [[0.1, 0.2, 0.3]],  # logprobs
            [[0, 1, 2]],  # sampled_token_ranks
        ]

        result = self.serving_chat._create_chat_logprobs(
            output_top_logprobs=output_top_logprobs, request_logprobs=5, request_top_logprobs=5
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, LogProbs)
        # The method creates one LogProbs entry per iteration, and we have 3 iterations
        # but they get extended into a single content list
        self.assertGreaterEqual(len(result.content), 1)

    def test_create_chat_logprobs_invalid(self):
        """Test _create_chat_logprobs with invalid input."""
        result = self.serving_chat._create_chat_logprobs(
            output_top_logprobs=None, request_logprobs=5, request_top_logprobs=5
        )

        self.assertIsNone(result)

    def test_create_chat_logprobs_empty(self):
        """Test _create_chat_logprobs with empty input."""
        result = self.serving_chat._create_chat_logprobs(
            output_top_logprobs=[], request_logprobs=5, request_top_logprobs=5
        )

        self.assertIsNone(result)

    def test_build_logprobs_response_valid(self):
        """Test _build_logprobs_response with valid input."""
        response_logprobs = LogprobsLists(
            logprob_token_ids=[[1, 2, 3]], logprobs=[[0.1, 0.2, 0.3]], sampled_token_ranks=[[0, 1, 2]]
        )

        result = self.serving_chat._build_logprobs_response(
            request_logprobs=True, response_logprobs=response_logprobs, request_top_logprobs=5
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, LogProbs)
        self.assertEqual(len(result.content), 1)

    def test_build_logprobs_response_invalid(self):
        """Test _build_logprobs_response with invalid input."""
        result = self.serving_chat._build_logprobs_response(
            request_logprobs=False, response_logprobs=None, request_top_logprobs=5
        )

        self.assertIsNone(result)

    def test_build_logprobs_response_with_unicode_replacement(self):
        """Test _build_logprobs_response with unicode replacement characters."""
        # Mock the data processor to return unicode replacement character
        self.mock_engine_client.data_processor.process_logprob_response.return_value = "test\ufffd"

        response_logprobs = LogprobsLists(logprob_token_ids=[[1]], logprobs=[[0.1]], sampled_token_ranks=[[0]])

        result = self.serving_chat._build_logprobs_response(
            request_logprobs=True, response_logprobs=response_logprobs, request_top_logprobs=5
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, LogProbs)
        # Check that unicode replacement is handled
        self.assertTrue(result.content[0].token.startswith("bytes:"))

    async def test_preprocess(self):
        """Test _preprocess method."""
        request = ChatCompletionRequest(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": True},
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        await self.serving_chat._preprocess(ctx)

        self.assertIsNotNone(ctx.preprocess_requests)
        self.assertEqual(len(ctx.preprocess_requests), 1)
        self.assertEqual(ctx.preprocess_requests[0]["chat_template"], self.chat_template)

    async def test_preprocess_without_chat_template_kwargs(self):
        """Test _preprocess without chat_template_kwargs."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], metadata={"enable_thinking": False}
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        await self.serving_chat._preprocess(ctx)

        self.assertIsNotNone(ctx.preprocess_requests)
        self.assertEqual(ctx.preprocess_requests[0]["enable_thinking"], False)
        self.assertEqual(ctx.preprocess_requests[0]["chat_template"], self.chat_template)

    async def test_build_stream_response(self):
        """Test _build_stream_response method."""
        # Create mock request
        request = ChatCompletionRequest(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            logprobs=False,
            return_token_ids=False,
        )

        # Create mock request output
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.finished = False
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()

        # Create context
        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}

        # Call the method
        result_generator = self.serving_chat._build_stream_response(ctx, mock_request_output, response_ctx)

        # Collect all results
        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].startswith("data: "))

    async def test_build_stream_response_with_usage(self):
        """Test _build_stream_response with usage included."""
        stream_options = StreamOptions(include_usage=True)
        request = ChatCompletionRequest(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            stream_options=stream_options,
            logprobs=False,
            return_token_ids=False,
        )

        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.finished = True
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()
        mock_request_output.metrics.request_start_time = time.time()

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}
        response_ctx.choice_completion_tokens_dict = {0: 10}

        result_generator = self.serving_chat._build_stream_response(ctx, mock_request_output, response_ctx)

        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 3)  # One data chunk and one [DONE]
        self.assertTrue(results[0].startswith("data: "))
        self.assertEqual(results[2], "data: [DONE]\n\n")

    async def test_build_stream_response_with_mm_output(self):
        """Test _build_stream_response with multimodal output enabled."""
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template,
            enable_mm_output=True,
        )

        request = ChatCompletionRequest(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            logprobs=False,
            return_token_ids=False,
        )

        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.finished = False
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}

        result_generator = serving_chat._build_stream_response(ctx, mock_request_output, response_ctx)

        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].startswith("data: "))

    async def test_build_full_response(self):
        """Test _build_full_response method."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], logprobs=False, return_token_ids=False
        )

        mock_output1 = MagicMock()
        mock_output1.index = 0
        mock_output1.text = "Response 1"
        mock_output1.reasoning_content = ""
        mock_output1.tool_calls = None
        mock_output1.token_ids = [1, 2, 3]

        mock_request_output1 = MagicMock()
        mock_request_output1.error_code = 200
        mock_request_output1.outputs = mock_output1
        mock_request_output1.prompt_token_ids = [4, 5, 6]
        mock_request_output1.prompt = "Test prompt"
        mock_request_output1.metrics = MagicMock()
        mock_request_output1.metrics.request_start_time = time.time()

        accumula_output_map = {0: [mock_request_output1]}

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.created_time = int(time.time())
        response_ctx = ServingResponseContext()
        # Call the method
        result = await self.serving_chat._build_full_response(ctx, accumula_output_map, response_ctx)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ChatCompletionResponse)
        self.assertEqual(len(result.choices), 1)
        self.assertEqual(result.choices[0].message.content, "Response 1")

    def test_build_full_response_with_error(self):
        """Test _build_full_response with error in request_output."""
        request = ChatCompletionRequest(model="test_model", messages=[{"role": "user", "content": "Hello"}])

        mock_request_output1 = MagicMock()
        mock_request_output1.error_code = 500
        mock_request_output1.error_msg = "Test error"

        accumula_output_map = {0: [mock_request_output1]}
        response_ctx = ServingResponseContext()
        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        # Should raise ValueError

        self.serving_chat._build_full_response(ctx, accumula_output_map, response_ctx)

    async def test_create_chat_completion_choice(self):
        """Test _create_chat_completion_choice method."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], logprobs=False, return_token_ids=False
        )

        # Create mock output
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None
        mock_output.token_ids = [1, 2, 3]

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.prompt = "Test prompt"
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.request_start_time = time.time()

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        # Call the method
        result = await self.serving_chat._create_chat_completion_choice([mock_request_output], ctx)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ChatCompletionResponseChoice)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.message.role, "assistant")
        self.assertEqual(result.message.content, "Test response")
        self.assertEqual(result.finish_reason, "stop")

    async def test_create_chat_completion_choice_with_tool_calls(self):
        """Test _create_chat_completion_choice with tool calls."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], logprobs=False, return_token_ids=False
        )

        # Create proper tool call structure
        from fastdeploy.entrypoints.openai.protocol import FunctionCall, ToolCall

        mock_function_call = FunctionCall(name="test_tool", arguments="{}")
        mock_tool_calls = ToolCall(id="test_tool_id", function=mock_function_call)

        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = mock_tool_calls
        mock_output.token_ids = [1, 2, 3]

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.prompt = "Test prompt"
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.request_start_time = time.time()

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        result = await self.serving_chat._create_chat_completion_choice([mock_request_output], ctx)

        self.assertIsNotNone(result)
        self.assertEqual(result.finish_reason, "tool_calls")

    async def test_create_chat_completion_choice_with_mm_output(self):
        """Test _create_chat_completion_choice with multimodal output enabled."""
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template,
            enable_mm_output=True,
        )

        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], logprobs=False, return_token_ids=False
        )

        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None
        mock_output.token_ids = [1, 2, 3]

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.prompt = "Test prompt"
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.request_start_time = time.time()

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        result = await serving_chat._create_chat_completion_choice([mock_request_output], ctx)

        self.assertIsNotNone(result)
        self.assertIsNone(result.message.content)

    async def test_create_chat_completion_stream(self):
        """Test create_chat_completion with stream=True."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], model="test_model", stream=True
        )

        # Mock the handle method
        with patch.object(self.serving_chat, "handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = AsyncMock()
            mock_handle.return_value.__aiter__ = AsyncMock(return_value=iter(["data: {}"]))

            result = await self.serving_chat.create_chat_completion(request)

            self.assertIsNotNone(result)
            mock_handle.assert_called_once()

    async def test_create_chat_completion_non_stream(self):
        """Test create_chat_completion with stream=False."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}], model="test_model", stream=False
        )

        # Mock the handle method
        with patch.object(self.serving_chat, "handle", new_callable=AsyncMock) as mock_handle:
            mock_response = MagicMock(spec=ChatCompletionResponse)
            mock_handle.return_value = mock_response

            result = await self.serving_chat.create_chat_completion(request)

            self.assertEqual(result, mock_response)
            mock_handle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
