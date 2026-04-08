"""
Unit tests for AsyncLLMOpenAiServingBase class
"""

import unittest
import uuid
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from fastdeploy.config import FDConfig

# Import classes we need to test
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.entrypoints.openai.v1.serving_base import (
    OpenAiServingBase,
    ServingResponseContext,
)


# Create a concrete subclass for testing the abstract base class
class TestOpenAiServingBase(OpenAiServingBase):
    """Concrete implementation of OpenAiServingBase for testing"""

    async def _build_stream_response(
        self,
        ctx: ServeContext[ChatCompletionRequest],
        request_output: RequestOutput,
        response_ctx: ServingResponseContext,
    ) -> AsyncGenerator:
        """Mock implementation for testing"""
        yield f"data: {request_output}"

    def _build_full_response(
        self,
        ctx: ServeContext[ChatCompletionRequest | CompletionRequest],
        accumula_output_map: dict[int, RequestOutput],
        response_ctx: ServingResponseContext,
    ) -> Any:
        """Mock implementation for testing"""
        return {"response": "full_response", "outputs": accumula_output_map}


class TestOpenAiServingBaseClass(unittest.IsolatedAsyncioTestCase):
    """Test cases for AsyncLLMOpenAiServingBase"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_engine_client = AsyncMock(spec=AsyncLLM)
        self.mock_config = MagicMock(spec=FDConfig)
        self.mock_models = MagicMock(spec=OpenAIServingModels)
        self.pid = 12345
        self.ips = ["127.0.0.1", "192.168.1.100"]
        self.max_waiting_time = 60

        # Create a serving base instance
        self.serving_base = TestOpenAiServingBase(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
        )

    def test_init_with_list_ips(self):
        """Test initialization with list of IPs."""
        serving_base = TestOpenAiServingBase(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
        )

        self.assertEqual(serving_base.master_ip, self.ips[0])
        self.assertEqual(serving_base.engine_client, self.mock_engine_client)
        self.assertEqual(serving_base.models, self.mock_models)
        self.assertEqual(serving_base.pid, self.pid)
        self.assertEqual(serving_base.max_waiting_time, self.max_waiting_time)

    def test_init_with_string_ips(self):
        """Test initialization with comma-separated string IPs."""
        ip_string = "127.0.0.1,192.168.1.100"
        serving_base = TestOpenAiServingBase(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=ip_string,
            max_waiting_time=self.max_waiting_time,
        )

        self.assertEqual(serving_base.master_ip, "127.0.0.1")

    def test_init_with_none_ips(self):
        """Test initialization with None IPs."""
        serving_base = TestOpenAiServingBase(
            engine_client=self.mock_engine_client,
            config=self.mock_config,
            models=self.mock_models,
            pid=self.pid,
            ips=None,
            max_waiting_time=self.max_waiting_time,
        )

        self.assertEqual(serving_base.master_ip, "0.0.0.0")
        self.assertTrue(serving_base.is_master_ip)

    def test_check_master_true(self):
        """Test _check_master when is_master_ip is True."""
        with patch("fastdeploy.entrypoints.openai.v1.serving_base.get_host_ip", return_value="127.0.0.1"):
            serving_base = TestOpenAiServingBase(
                engine_client=self.mock_engine_client,
                config=self.mock_config,
                models=self.mock_models,
                pid=self.pid,
                ips=["127.0.0.1", "192.168.1.100"],
                max_waiting_time=self.max_waiting_time,
            )
            self.assertTrue(serving_base._check_master())

    def test_check_master_false(self):
        """Test _check_master when is_master_ip is False."""
        with patch("fastdeploy.entrypoints.openai.v1.serving_base.get_host_ip", return_value="192.168.1.200"):
            serving_base = TestOpenAiServingBase(
                engine_client=self.mock_engine_client,
                config=self.mock_config,
                models=self.mock_models,
                pid=self.pid,
                ips=self.ips,
                max_waiting_time=self.max_waiting_time,
            )
            self.assertFalse(serving_base._check_master())

    def test_generate_request_id_with_request_id(self):
        """Test _generate_request_id with existing request_id."""
        request = MagicMock()
        request.request_id = "chatcmpl-custom_id"

        result = self.serving_base._generate_request_id(request)
        self.assertEqual(result, "chatcmpl-custom_id")

    def test_generate_request_id_with_request_id_prefix(self):
        """Test _generate_request_id with request_id needing prefix."""
        request = MagicMock()
        request.request_id = "test123"
        request.user = None

        result = self.serving_base._generate_request_id(request)
        self.assertEqual(result, "chatcmpl-test123")

    def test_generate_request_id_with_user(self):
        """Test _generate_request_id with user parameter."""
        request = MagicMock()
        request.request_id = None
        request.user = "test_user"

        result = self.serving_base._generate_request_id(request)
        self.assertTrue(result.startswith("chatcmpl-test_user-"))
        # Extract UUID part and validate
        uuid_part = result[len("chatcmpl-test_user-") :]
        self.assertEqual(len(uuid_part.split("-")), 5)  # UUID format has 5 parts

    def test_generate_request_id_default(self):
        """Test _generate_request_id with default parameters."""
        request = MagicMock()
        request.request_id = None
        request.user = None

        result = self.serving_base._generate_request_id(request)
        self.assertTrue(result.startswith("chatcmpl-"))
        # Should be a valid UUID
        uuid_part = result.replace("chatcmpl-", "")
        uuid.UUID(uuid_part)  # This will raise ValueError if not a valid UUID

    async def test_preprocess(self):
        """Test _preprocess method."""
        request = ChatCompletionRequest(model="test_model", messages=[{"role": "user", "content": "Hello"}])

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        await self.serving_base._preprocess(ctx)

        self.assertIsNotNone(ctx.preprocess_requests)
        self.assertEqual(len(ctx.preprocess_requests), 1)
        self.assertIn("test_request_id_0", str(ctx.preprocess_requests[0]))

    async def test_prepare_generators(self):
        """Test _prepare_generators method."""
        request = ChatCompletionRequest(model="test_model", messages=[{"role": "user", "content": "Hello"}])

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.preprocess_requests = [{"test": "request_dict"}]

        # Mock the engine client generate method to return an async generator directly
        mock_request_output = MagicMock()

        async def mock_generator(*args, **kwargs):
            yield mock_request_output

        self.mock_engine_client.generate = mock_generator

        # Test the generator
        generator = self.serving_base._prepare_generators(ctx)
        results = []
        async for result in generator:
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], mock_request_output)

    async def test_prepare_generators_with_none_preprocess_requests(self):
        """Test _prepare_generators with None preprocess_requests"""
        ctx = MagicMock()
        ctx.preprocess_requests = None

        with self.assertRaises(ValueError) as context:
            async for _ in self.serving_base._prepare_generators(ctx):
                pass

        self.assertIn("preprocess_requests is None", str(context.exception))

    def test_build_response(self):
        """Test _build_response method."""
        ctx = MagicMock()
        ctx.request_id = "test_request_id"

        mock_request_output = MagicMock()

        result = self.serving_base._build_response(ctx, mock_request_output)

        self.assertEqual(result, mock_request_output)

    async def test_handle_stream_basic(self):
        """Test handle method with stream=True basic functionality."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        # Mock the handleStream method to return an async generator
        async def mock_handle_stream(ctx):
            yield "data: test_response"

        self.serving_base.handle_stream = mock_handle_stream

        # Test the handle method
        result = await self.serving_base.handle(ctx)

        # Verify it's an async generator
        self.assertTrue(hasattr(result, "__aiter__"))

        # Collect results
        results = []
        async for item in result:
            results.append(item)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "data: test_response")

    async def test_handle_non_stream_basic(self):
        """Test handle method with stream=False basic functionality."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=False
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )

        # Mock the handleNonStream method
        with patch.object(self.serving_base, "handle_non_stream", new_callable=AsyncMock) as mock_handle:
            mock_response = {"response": "test"}
            mock_handle.return_value = mock_response

            result = await self.serving_base.handle(ctx)

            self.assertEqual(result, mock_response)

    def test_calc_finish_reason_stop(self):
        """Test _calc_finish_reason with stop reason."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = None

        result = self.serving_base._calc_finish_reason(mock_request_output, None, 10)
        self.assertEqual(result, "stop")

    def test_calc_finish_reason_tool_calls(self):
        """Test _calc_finish_reason with tool_calls."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = {"name": "test_tool"}
        mock_request_output.error_msg = None

        result = self.serving_base._calc_finish_reason(mock_request_output, None, 10)
        self.assertEqual(result, "tool_calls")

    def test_calc_finish_reason_length(self):
        """Test _calc_finish_reason with length limit."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = None

        result = self.serving_base._calc_finish_reason(mock_request_output, 5, 10)
        self.assertEqual(result, "length")

    def test_calc_finish_reason_recover_stop(self):
        """Test _calc_finish_reason with recover stop."""
        mock_request_output = MagicMock()
        mock_request_output.outputs.tool_calls = None
        mock_request_output.error_msg = "Recover from error"

        result = self.serving_base._calc_finish_reason(mock_request_output, None, 10)
        self.assertEqual(result, "recover_stop")

    def test_calc_usage_with_send_idx_zero(self):
        """Test _calc_usage with send_idx = 0."""
        mock_request_output = MagicMock()
        mock_request_output.outputs = MagicMock()
        mock_request_output.outputs.token_ids = [1, 2, 3]
        mock_request_output.outputs.send_idx = 0
        mock_request_output.outputs.reasoning_token_num = 2
        mock_request_output.outputs.decode_type = 0
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.num_cached_tokens = 1
        mock_request_output.num_input_image_tokens = 2
        mock_request_output.num_input_video_tokens = 3

        result = self.serving_base._calc_usage(mock_request_output)

        self.assertIsInstance(result, UsageInfo)
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.total_tokens, 6)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 1)
        self.assertEqual(result.prompt_tokens_details.image_tokens, 2)
        self.assertEqual(result.prompt_tokens_details.video_tokens, 3)
        self.assertEqual(result.completion_tokens_details.reasoning_tokens, 2)
        self.assertEqual(result.completion_tokens_details.image_tokens, 0)

    def test_calc_usage_with_send_idx_non_zero(self):
        """Test _calc_usage with send_idx > 0."""
        mock_request_output = MagicMock()
        mock_request_output.outputs = MagicMock()
        mock_request_output.outputs.token_ids = [1, 2, 3]
        mock_request_output.outputs.send_idx = 1
        mock_request_output.outputs.reasoning_token_num = 2
        mock_request_output.outputs.decode_type = 1
        mock_request_output.prompt_token_ids = [4, 5, 6]
        mock_request_output.num_cached_tokens = 1
        mock_request_output.num_input_image_tokens = 2
        mock_request_output.num_input_video_tokens = 3

        result = self.serving_base._calc_usage(mock_request_output)

        self.assertEqual(result.prompt_tokens, 0)  # send_idx > 0
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.total_tokens, 3)
        self.assertEqual(result.completion_tokens_details.image_tokens, 3)  # decode_type = 1

    async def test_handle_stream_complex_scenario(self):
        """Test handleStream method with complex scenario including accumulation and eoi token."""
        # Mock a complex scenario with decode_type = 1 and eoi_token_id
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.preprocess_requests = [{"test": "request"}]

        # Create mock request outputs for complex scenario
        mock_output1 = MagicMock()
        mock_output1.outputs = MagicMock()
        mock_output1.outputs.index = 0
        mock_output1.outputs.decode_type = 1  # Will be accumulated
        mock_output1.outputs.token_ids = [1, 2]
        mock_output1.outputs.reasoning_token_num = 0
        mock_output1.finished = False

        mock_output2 = MagicMock()
        mock_output2.outputs = MagicMock()
        mock_output2.outputs.index = 0
        mock_output2.outputs.decode_type = 1  # Will trigger eoi processing
        mock_output2.outputs.token_ids = [3, 4, self.serving_base.eoi_token_id]
        mock_output2.outputs.reasoning_token_num = 0
        mock_output2.finished = True

        mock_output3 = MagicMock()
        mock_output3.outputs = MagicMock()
        mock_output3.outputs.index = 1
        mock_output3.outputs.decode_type = 0  # Normal output
        mock_output3.outputs.token_ids = [5, 6]
        mock_output3.outputs.reasoning_token_num = 0
        mock_output3.finished = True

        # Mock the _pipeline generator
        async def mock_pipeline(ctx):
            yield mock_output1
            yield mock_output2
            yield mock_output3

        self.serving_base._pipeline = mock_pipeline

        # Mock _build_stream_response
        async def mock_build_stream_response(ctx, request_output, response_ctx):
            yield f"data: {request_output.outputs.token_ids}"

        self.serving_base._build_stream_response = mock_build_stream_response

        # Test handleStream
        result = self.serving_base.handle_stream(ctx)

        # Verify it returns an async generator
        self.assertTrue(hasattr(result, "__aiter__"))

        # Collect and verify results
        results = []
        async for item in result:
            results.append(item)

        # Should have 2 stream responses (one from accumulated output + one from normal output)
        self.assertEqual(len(results), 1)

    async def test_handle_non_stream_error_scenario(self):
        """Test handleNonStream with error handling scenario."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=False
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.preprocess_requests = [{"test": "request"}]

        # Mock a request output with error
        mock_output = MagicMock()
        mock_output.outputs = MagicMock()
        mock_output.outputs.index = 0
        mock_output.outputs.decode_type = 0
        mock_output.outputs.token_ids = [1, 2]
        mock_output.finished = True
        mock_output.error_msg = "Test error"

        # Mock the _pipeline generator
        async def mock_pipeline(ctx):
            yield mock_output

        self.serving_base._pipeline = mock_pipeline

        # Mock _build_full_response to handle error
        async def mock_build_full_response(ctx, accumula_output_map, response_ctx):
            return {"error": "Test error response"}

        self.serving_base._build_full_response = mock_build_full_response

        # Test handleNonStream
        result = await self.serving_base.handle_non_stream(ctx)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["error"], "Test error response")

    async def test_handle_stream_multiple_choices(self):
        """Test handleStream with multiple choices (n parameter)."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=True, n=2
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.preprocess_requests = [{"test": "request"}]

        # Create mock outputs for multiple indices
        mock_outputs = []
        for i in range(4):  # 2 indices * 2 outputs each
            mock_output = MagicMock()
            mock_output.outputs = MagicMock()
            mock_output.outputs.index = i % 2
            mock_output.outputs.decode_type = 0
            mock_output.outputs.token_ids = [i + 1]
            mock_output.finished = i % 2 == 1  # Finish after every 2nd output
            mock_outputs.append(mock_output)

        async def mock_pipeline(ctx):
            for output in mock_outputs:
                yield output

        self.serving_base._pipeline = mock_pipeline

        async def mock_build_stream_response(ctx, request_output, response_ctx):
            yield f"data: choice_{request_output.outputs.index}"

        self.serving_base._build_stream_response = mock_build_stream_response

        result = self.serving_base.handle_stream(ctx)

        results = []
        async for item in result:
            results.append(item)

        # Should have 4 stream responses (2 choices * 2 outputs each)
        self.assertEqual(len(results), 4)

    async def test_handle_non_stream_accumulation_logic(self):
        """Test handleNonStream accumulation logic with different decode types."""
        request = ChatCompletionRequest(
            model="test_model", messages=[{"role": "user", "content": "Hello"}], stream=False
        )

        ctx = ServeContext[ChatCompletionRequest](
            request=request, model_name="test_model", request_id="test_request_id"
        )
        ctx.preprocess_requests = [{"test": "request"}]

        # Create mock outputs with different decode types for accumulation testing
        mock_output1 = MagicMock()
        mock_output1.outputs = MagicMock()
        mock_output1.outputs.index = 0
        mock_output1.outputs.decode_type = 0  # Normal type
        mock_output1.outputs.token_ids = [1, 2]

        mock_output2 = MagicMock()
        mock_output2.outputs = MagicMock()
        mock_output2.outputs.index = 0
        mock_output2.outputs.decode_type = 1  # Different type - should create new list entry
        mock_output2.outputs.token_ids = [3, 4]

        mock_output3 = MagicMock()
        mock_output3.outputs = MagicMock()
        mock_output3.outputs.index = 0
        mock_output3.outputs.decode_type = 1  # Same type - should be accumulated
        mock_output3.outputs.token_ids = [5, 6]

        mock_output1.finished = mock_output2.finished = mock_output3.finished = True

        async def mock_pipeline(ctx):
            yield mock_output1
            yield mock_output2
            yield mock_output3

        self.serving_base._pipeline = mock_pipeline

        async def mock_build_full_response(ctx, accumula_output_map, response_ctx):
            # Verify accumulation logic worked correctly
            self.assertEqual(len(accumula_output_map), 1)  # Only one index
            self.assertEqual(len(accumula_output_map[0]), 2)  # Two different decode type groups
            self.assertEqual(len(accumula_output_map[0][0].outputs.token_ids), 2)  # First group
            return {"status": "success"}

        self.serving_base._build_full_response = mock_build_full_response

        result = await self.serving_base.handle_non_stream(ctx)
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
