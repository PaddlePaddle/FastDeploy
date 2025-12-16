"""
Unit tests for AsyncLLMOpenAIServingCompletion class
"""

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastdeploy.config import FDConfig

# Import the classes we need to test
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.entrypoints.openai.protocol import (
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.entrypoints.openai.v1.serving_completion import OpenAIServingCompletion
from fastdeploy.utils import ErrorType
from fastdeploy.worker.output import LogprobsLists


# Define ServingResponseContext locally since it's not exported
class ServingResponseContext:
    def __init__(self):
        from fastdeploy.entrypoints.openai.protocol import UsageInfo

        self.usage = UsageInfo()
        self.choice_completion_tokens_dict = {}
        self.inference_start_time_dict = {}


class TestOpenAIServingCompletion(unittest.IsolatedAsyncioTestCase):
    """Test cases for OpenAIServingCompletion"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_engine_client = AsyncMock(spec=AsyncLLM)
        self.mock_config = MagicMock(spec=FDConfig)
        self.mock_models = MagicMock(spec=OpenAIServingModels)
        self.pid = 12345
        self.ips = ["127.0.0.1"]
        self.max_waiting_time = 60

        # Create the serving completion instance
        self.serving_completion = OpenAIServingCompletion(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
        )

        # Mock the data processor
        self.mock_engine_client.data_processor = MagicMock()
        self.mock_engine_client.data_processor.process_logprob_response = MagicMock(return_value="test_token")

    def test_init(self):
        """Test basic initialization."""
        self.assertEqual(self.serving_completion.pid, self.pid)
        self.assertEqual(self.serving_completion.engine_client, self.mock_engine_client)
        self.assertEqual(self.serving_completion.models, self.mock_models)

    def test_create_completion_logprobs_valid(self):
        """Test _create_completion_logprobs with valid input."""
        output_top_logprobs = [
            [[1, 2, 3]],  # logprob_token_ids
            [[0.1, 0.2, 0.3]],  # logprobs
            [[0, 1, 2]],  # sampled_token_ranks
        ]

        result = self.serving_completion._create_completion_logprobs(
            output_top_logprobs=output_top_logprobs, request_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompletionLogprobs)
        self.assertEqual(len(result.tokens), 1)
        self.assertEqual(len(result.token_logprobs), 1)
        self.assertEqual(len(result.top_logprobs), 1)

    def test_create_completion_logprobs_invalid(self):
        """Test _create_completion_logprobs with invalid input."""
        result = self.serving_completion._create_completion_logprobs(
            output_top_logprobs=None, request_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNone(result)

    def test_create_completion_logprobs_empty(self):
        """Test _create_completion_logprobs with empty input."""
        result = self.serving_completion._create_completion_logprobs(
            output_top_logprobs=[], request_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNone(result)

    def test_build_logprobs_response_valid(self):
        """Test _build_logprobs_response with valid input."""
        response_logprobs = LogprobsLists(
            logprob_token_ids=[[1, 2, 3]], logprobs=[[0.1, 0.2, 0.3]], sampled_token_ranks=[[0, 1, 2]]
        )

        result = self.serving_completion._build_logprobs_response(
            response_logprobs=response_logprobs, request_top_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompletionLogprobs)
        self.assertEqual(len(result.tokens), 1)
        self.assertEqual(len(result.token_logprobs), 1)
        self.assertEqual(len(result.top_logprobs), 1)

    def test_build_logprobs_response_invalid(self):
        """Test _build_logprobs_response with invalid input."""
        result = self.serving_completion._build_logprobs_response(
            response_logprobs=None, request_top_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNone(result)

    async def test_build_stream_response(self):
        """Test _build_stream_response method."""
        # Create mock request output
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.token_ids = [1, 2, 3]
        mock_output.top_logprobs = None
        mock_output.tool_calls = None
        mock_output.reasoning_content = None

        mock_request_output = MagicMock()
        mock_request_output.error_code = 200
        mock_request_output.outputs = mock_output
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()
        mock_request_output.finished = False

        # Create mock request
        mock_request = CompletionRequest(
            model="test_model",
            prompt="Hello",
            stream_options=None,
            max_streaming_response_tokens=1,
            return_token_ids=True,
            logprobs=False,
            include_draft_logprobs=False,
        )

        # Create context
        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}

        # Call the method
        result_generator = self.serving_completion._build_stream_response(ctx, mock_request_output, response_ctx)

        # Collect all results
        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].startswith("data: "))

    async def test_build_full_response(self):
        """Test _build_full_response method."""
        # Create mock request outputs
        mock_output1 = MagicMock()
        mock_output1.index = 0
        mock_output1.text = "Response 1"
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.top_logprobs = None
        mock_output1.tool_calls = None
        mock_output1.reasoning_content = None

        mock_request_output1 = MagicMock()
        mock_request_output1.error_code = 200
        mock_request_output1.outputs = mock_output1
        mock_request_output1.request_id = "test_request_id_0"
        mock_request_output1.prompt_token_ids = [4, 5, 6]
        mock_request_output1.prompt = "Test prompt"

        accumula_output_map = {0: [mock_request_output1]}

        # Create mock request
        mock_request = CompletionRequest(
            model="test_model", prompt="Hello", max_tokens=None, logprobs=False, return_token_ids=True
        )

        # Create context
        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
        )
        ctx.created_time = int(time.time())
        res_ctx = ServingResponseContext()
        # Call the method
        result = await self.serving_completion._build_full_response(ctx, accumula_output_map, res_ctx)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompletionResponse)

    def test_build_completion_choice(self):
        """Test build_completion_choice method."""
        # Create mock output object (this is final_res.outputs)
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3]
        mock_output.text = "Test response"
        mock_output.reasoning_content = ""
        mock_output.tool_calls = None
        mock_output.top_logprobs = None
        mock_output.draft_top_logprobs = None

        # Create mock request output (this is final_res)
        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.error_code = 200
        mock_request_output.error_msg = None

        # Create real CompletionRequest object
        mock_request = CompletionRequest(
            model="test_model",
            prompt="Hello",
            max_tokens=None,
            logprobs=False,
            return_token_ids=True,
        )

        # Create context
        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
        )

        # Call the method
        result = self.serving_completion.build_completion_choice(0, mock_request_output, ctx)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompletionResponseChoice)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.text, "Test response")
        self.assertEqual(result.finish_reason, "stop")

    def test_calc_finish_reason_stop(self):
        """Test finish reason calculation for normal stop."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = None

        finish_reason = self.serving_completion._calc_finish_reason(mock_request_output, max_tokens=None, token_nums=5)

        self.assertEqual(finish_reason, "stop")

    def test_calc_finish_reason_tool_calls(self):
        """Test finish reason calculation for tool calls."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = {"name": "test_tool"}
        mock_request_output.error_msg = None

        finish_reason = self.serving_completion._calc_finish_reason(mock_request_output, max_tokens=None, token_nums=5)

        self.assertEqual(finish_reason, "tool_calls")

    def test_calc_finish_reason_length(self):
        """Test finish reason calculation for max tokens reached."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = None

        finish_reason = self.serving_completion._calc_finish_reason(mock_request_output, max_tokens=5, token_nums=5)

        self.assertEqual(finish_reason, "length")

    def test_calc_finish_reason_recover_stop(self):
        """Test finish reason calculation for recover stop."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = "Recover from error"

        finish_reason = self.serving_completion._calc_finish_reason(mock_request_output, max_tokens=None, token_nums=5)

        self.assertEqual(finish_reason, "recover_stop")

    def test_calc_usage(self):
        """Test usage calculation."""
        mock_output = MagicMock()
        mock_output.token_ids = [1, 2, 3]
        mock_output.send_idx = 0
        mock_output.reasoning_token_num = 1
        mock_output.decode_type = 0

        mock_request_output = MagicMock()
        mock_request_output.outputs = mock_output
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.num_cached_tokens = 0
        mock_request_output.num_input_image_tokens = 0
        mock_request_output.num_input_video_tokens = 0

        usage = self.serving_completion._calc_usage(mock_request_output)

        self.assertIsNotNone(usage)
        self.assertIsInstance(usage, UsageInfo)
        self.assertEqual(usage.prompt_tokens, 3)
        self.assertEqual(usage.completion_tokens, 3)
        self.assertEqual(usage.total_tokens, 6)

    async def test_create_completion_stream(self):
        """Test create_completion with stream=True."""
        request = CompletionRequest(prompt="Hello, world!", model="test_model", stream=True)

        # Mock the handle method
        with patch.object(self.serving_completion, "handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = AsyncMock()
            mock_handle.return_value.__aiter__ = AsyncMock(return_value=iter(["data: {}"]))

            result = await self.serving_completion.create_completion(request)

            self.assertIsNotNone(result)
            mock_handle.assert_called_once()

    async def test_create_completion_non_stream(self):
        """Test create_completion with stream=False."""
        request = CompletionRequest(prompt="Hello, world!", model="test_model", stream=False)

        # Mock the handle method
        with patch.object(self.serving_completion, "handle", new_callable=AsyncMock) as mock_handle:
            mock_response = MagicMock(spec=CompletionResponse)
            mock_handle.return_value = mock_response

            result = await self.serving_completion.create_completion(request)

            self.assertEqual(result, mock_response)
            mock_handle.assert_called_once()

    def test_create_completion_logprobs_with_unicode_replacement(self):
        """Test _create_completion_logprobs with unicode replacement characters."""
        # Mock the data processor to return unicode replacement character
        self.mock_engine_client.data_processor.process_logprob_response.return_value = "test\ufffd"

        output_top_logprobs = [[[1]], [[0.1]], [[0]]]  # logprob_token_ids  # logprobs  # sampled_token_ranks

        result = self.serving_completion._create_completion_logprobs(
            output_top_logprobs=output_top_logprobs, request_logprobs=5, prompt_text_offset=0
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompletionLogprobs)
        # Check that unicode replacement is handled
        self.assertTrue(result.tokens[0].startswith("bytes:"))

    async def test_build_stream_response_with_error(self):
        """Test _build_stream_response with error in request_output."""
        mock_request_output = MagicMock()
        mock_request_output.error_code = 500
        mock_request_output.error_msg = "Test error"

        mock_request = CompletionRequest(model="test_model", prompt="Hello", stream_options=None)

        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
            response_ctx=ServingResponseContext(),
            mm_hashes=[],
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()

        result_generator = self.serving_completion._build_stream_response(ctx, mock_request_output, response_ctx)

        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertIn("error", results[0])

    async def test_build_stream_response_with_logprobs(self):
        """Test _build_stream_response with logprobs enabled."""
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.token_ids = [1, 2, 3]
        mock_output.top_logprobs = [[[1]], [[0.1]], [[0]]]  # logprob_token_ids  # logprobs  # sampled_token_ranks
        mock_output.draft_top_logprobs = None
        mock_output.tool_calls = None
        mock_output.reasoning_content = None

        mock_request_output = MagicMock()
        mock_request_output.error_code = 200
        mock_request_output.outputs = mock_output
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()
        mock_request_output.finished = False

        mock_request = CompletionRequest(
            model="test_model",
            prompt="Hello",
            stream_options=None,
            max_streaming_response_tokens=1,
            return_token_ids=True,
            logprobs=True,
            include_draft_logprobs=False,
        )

        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
            response_ctx=ServingResponseContext(),
            mm_hashes=[],
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}

        result_generator = self.serving_completion._build_stream_response(ctx, mock_request_output, response_ctx)

        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].startswith("data: "))

    async def test_build_stream_response_with_draft_logprobs(self):
        """Test _build_stream_response with draft logprobs enabled."""
        mock_output = MagicMock()
        mock_output.index = 0
        mock_output.text = "Test response"
        mock_output.token_ids = [1, 2, 3]
        mock_output.top_logprobs = [[[1]], [[0.1]], [[0]]]  # logprob_token_ids  # logprobs  # sampled_token_ranks
        mock_output.draft_top_logprobs = [
            [[2]],  # logprob_token_ids
            [[0.2]],  # logprobs
            [[1]],  # sampled_token_ranks
        ]
        mock_output.tool_calls = None
        mock_output.reasoning_content = None

        mock_request_output = MagicMock()
        mock_request_output.error_code = 200
        mock_request_output.outputs = mock_output
        mock_request_output.metrics = MagicMock()
        mock_request_output.metrics.first_token_time = time.time()
        mock_request_output.metrics.inference_start_time = time.time()
        mock_request_output.finished = False

        mock_request = CompletionRequest(
            model="test_model",
            prompt="Hello",
            stream_options=None,
            max_streaming_response_tokens=1,
            return_token_ids=True,
            logprobs=True,
            include_draft_logprobs=True,
        )

        ctx = ServeContext[CompletionRequest](
            request=mock_request,
            model_name="test_model",
            request_id="test_request_id",
        )
        ctx.created_time = int(time.time())

        response_ctx = ServingResponseContext()
        response_ctx.inference_start_time_dict = {0: time.time()}

        result_generator = self.serving_completion._build_stream_response(ctx, mock_request_output, response_ctx)

        results = []
        async for result in result_generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].startswith("data: "))


class TestAsyncLLMOpenAIServingCompletionPreprocess(unittest.IsolatedAsyncioTestCase):
    """Test cases for AsyncLLMOpenAIServingCompletion._preprocess method"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock dependencies
        mock_engine_client = MagicMock(spec=AsyncLLM)
        mock_config = MagicMock()
        mock_models = MagicMock(spec=OpenAIServingModels)
        mock_pid = 12345
        mock_ips = ["127.0.0.1"]
        max_waiting_time = 60

        self.serving_completion = OpenAIServingCompletion(
            engine_client=mock_engine_client,
            config=mock_config,
            models=mock_models,
            pid=mock_pid,
            ips=mock_ips,
            max_waiting_time=max_waiting_time,
        )
        self.serving_completion.eoi_token_id = 999  # Mock EOI token ID

    async def test_preprocess_valid_prompt_string(self):
        """Test _preprocess with valid string prompt"""
        # Setup
        request = CompletionRequest(model="test_model", prompt="Hello world", max_tokens=100)
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)  # Should return None on success
        self.assertEqual(len(ctx.preprocess_requests), 1)
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], "Hello world")
        self.assertEqual(ctx.preprocess_requests[0]["request_id"], "test_request_id_0")
        self.assertIn("arrival_time", ctx.preprocess_requests[0])

    async def test_preprocess_valid_prompt_string_list(self):
        """Test _preprocess with valid list of string prompts"""
        # Setup
        request = CompletionRequest(model="test_model", prompt=["Hello", "World"], max_tokens=100)
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 2)
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], "Hello")
        self.assertEqual(ctx.preprocess_requests[0]["request_id"], "test_request_id_0")
        self.assertEqual(ctx.preprocess_requests[1]["prompt"], "World")
        self.assertEqual(ctx.preprocess_requests[1]["request_id"], "test_request_id_1")

    async def test_preprocess_valid_prompt_int_list(self):
        """Test _preprocess with prompt as list of ints"""
        # Setup
        request = CompletionRequest(model="test_model", prompt=[1, 2, 3, 4, 5], max_tokens=50)
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 1)
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], [1, 2, 3, 4, 5])

    async def test_preprocess_valid_prompt_nested_int_list(self):
        """Test _preprocess with prompt as nested list of ints"""
        # Setup
        request = CompletionRequest(model="test_model", prompt=[[1, 2, 3], [4, 5, 6]], max_tokens=50)
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 2)
        # When prompt is [[1,2,3], [4,5,6]], it should be treated as token IDs for batch inference
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], [1, 2, 3])
        self.assertEqual(ctx.preprocess_requests[1]["prompt"], [4, 5, 6])

    async def test_preprocess_valid_prompt_token_ids_int_list(self):
        """Test _preprocess with valid list of int token IDs"""
        # Setup
        request = CompletionRequest(
            model="test_model", prompt="dummy", prompt_token_ids=[1, 2, 3, 4, 5], max_tokens=50  # Required field
        )
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 1)
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], [1, 2, 3, 4, 5])
        self.assertIsNone(request.prompt_token_ids)  # Should be reset

    async def test_preprocess_valid_prompt_token_ids_nested_list(self):
        """Test _preprocess with valid nested list of token IDs (batch inference)"""
        # Setup
        request = CompletionRequest(
            model="test_model",
            prompt="dummy",  # Required field
            prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
            max_tokens=50,
        )
        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 2)
        # When prompt_token_ids is [[1,2,3], [4,5,6]], it should be treated as batch token IDs
        self.assertEqual(ctx.preprocess_requests[0]["prompt"], [1, 2, 3])
        self.assertEqual(ctx.preprocess_requests[1]["prompt"], [4, 5, 6])
        self.assertIsNone(request.prompt_token_ids)  # Should be reset

    async def test_preprocess_empty_prompt_token_ids(self):
        """Test _preprocess with empty prompt_token_ids list - should be caught by Pydantic validation"""
        # Since Pydantic validates that prompt_token_ids must not be empty,
        # we can't create a CompletionRequest with empty prompt_token_ids
        # This test verifies the behavior when empty list is bypassed
        request = CompletionRequest(model="test_model", prompt="dummy", max_tokens=50)
        # Manually set empty prompt_token_ids to bypass Pydantic validation
        request.prompt_token_ids = []

        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute and Assert
        await self.serving_completion._preprocess(ctx)

    async def test_preprocess_invalid_prompt_token_ids_type(self):
        """Test _preprocess with invalid prompt_token_ids type"""
        # Since Pydantic validates input, we need to bypass validation to test internal error handling
        request = CompletionRequest(model="test_model", prompt="dummy", max_tokens=50)
        # Manually set invalid prompt_token_ids to bypass Pydantic validation
        request.prompt_token_ids = ["invalid", "type"]

        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsInstance(result, ErrorResponse)
        self.assertIn("ValueError", result.error.message)
        self.assertEqual(result.error.type, ErrorType.INTERNAL_ERROR)

    async def test_preprocess_invalid_prompt_type(self):
        """Test _preprocess with invalid prompt type"""
        # Since Pydantic validates input, we need to bypass validation to test internal error handling
        request = CompletionRequest(model="test_model", prompt="dummy", max_tokens=50)
        # Manually set invalid prompt to bypass Pydantic validation
        request.prompt = {"invalid": "type"}

        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsInstance(result, ErrorResponse)
        self.assertIn("ValueError", result.error.message)
        self.assertEqual(result.error.type, ErrorType.INTERNAL_ERROR)

    async def test_preprocess_invalid_list_prompt_type(self):
        """Test _preprocess with list containing invalid types"""
        # Since Pydantic validates input, we need to bypass validation to test internal error handling
        request = CompletionRequest(model="test_model", prompt="dummy", max_tokens=50)
        # Manually set invalid prompt to bypass Pydantic validation
        request.prompt = ["valid", 123]  # Mixed types

        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsInstance(result, ErrorResponse)
        self.assertIn("ValueError", result.error.message)
        self.assertEqual(result.error.type, ErrorType.INTERNAL_ERROR)

    async def test_preprocess_with_complex_request(self):
        """Test _preprocess with complex request containing all parameters"""
        # Setup
        request = CompletionRequest(
            model="test_model",
            prompt="Complex test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=2,
            stop=["\n", "STOP"],
            stream=False,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        ctx = ServeContext(request=request, model_name="test_model", request_id="complex_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsNone(result)
        self.assertEqual(len(ctx.preprocess_requests), 1)
        preprocessed_request = ctx.preprocess_requests[0]

        # Verify key parameters are preserved
        self.assertEqual(preprocessed_request["prompt"], "Complex test prompt")
        self.assertEqual(preprocessed_request["max_tokens"], 100)
        self.assertEqual(preprocessed_request["temperature"], 0.7)
        self.assertEqual(preprocessed_request["top_p"], 0.9)
        self.assertEqual(preprocessed_request["n"], 2)
        self.assertEqual(preprocessed_request["stop"], ["\n", "STOP"])
        self.assertEqual(preprocessed_request["stream"], False)

    async def test_preprocess_request_id_generation(self):
        """Test _preprocess with different request ID patterns"""
        test_cases = [
            ("simple_id", ["Hello"], 1),
            ("id_with_underscore", ["Prompt1", "Prompt2"], 2),
            ("", ["Single prompt"], 1),  # Empty request ID
        ]

        for request_id, prompts, expected_requests in test_cases:
            with self.subTest(request_id=request_id, prompts=prompts):
                # Setup
                if len(prompts) == 1:
                    prompt = prompts[0]
                else:
                    prompt = prompts

                request = CompletionRequest(model="test_model", prompt=prompt, max_tokens=50)
                ctx = ServeContext(request=request, model_name="test_model", request_id=request_id)

                # Execute
                result = await self.serving_completion._preprocess(ctx)

                # Assert
                self.assertIsNone(result)
                self.assertEqual(len(ctx.preprocess_requests), expected_requests)

                for i in range(expected_requests):
                    expected_id = f"{request_id}_{i}" if request_id else f"_{i}"
                    self.assertEqual(ctx.preprocess_requests[i]["request_id"], expected_id)

    @patch("fastdeploy.entrypoints.openai.v1.serving_completion.api_server_logger")
    async def test_preprocess_exception_logging(self, mock_logger):
        """Test _preprocess logs exceptions properly"""
        # Setup - create a request that will cause an exception
        request = CompletionRequest(model="test_model", prompt="dummy", max_tokens=50)
        # Manually set invalid prompt to bypass Pydantic validation
        request.prompt = {"invalid": "type"}

        ctx = ServeContext(request=request, model_name="test_model", request_id="test_request_id")

        # Execute
        result = await self.serving_completion._preprocess(ctx)

        # Assert
        self.assertIsInstance(result, ErrorResponse)
        mock_logger.error.assert_called_once()
        error_log = mock_logger.error.call_args[0][0]
        self.assertIn("OpenAIServingCompletion create_completion", error_log)
        self.assertIn("ValueError", error_log)
        self.assertIn("Traceback", error_log)  # Changed from "traceback" to "Traceback"


if __name__ == "__main__":
    unittest.main()
