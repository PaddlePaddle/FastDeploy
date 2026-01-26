# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import paddle

import fastdeploy.envs as envs
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.utils import ErrorCode, ParameterError
from fastdeploy.worker.output import LogprobsLists, LogprobsTensors


class TestServingCompletion(unittest.IsolatedAsyncioTestCase):
    def test_init_with_ips_list(self):
        """__init__ should honor the first ip in the list."""
        engine_client = Mock()
        engine_client.is_master = True
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", ["1.2.3.4", "5.6.7.8"], 360)

        self.assertEqual(serving_completion.master_ip, "1.2.3.4")

    def test_build_logprobs_response_invalid_params(self):
        """_build_logprobs_response should short-circuit on invalid inputs."""
        _ = paddle.to_tensor([1], dtype=paddle.int64)
        engine_client = Mock()
        engine_client.data_processor = Mock()
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        self.assertIsNone(serving_completion._build_logprobs_response(None, request_top_logprobs=1))
        self.assertIsNone(serving_completion._build_logprobs_response(LogprobsLists([[1]], [[-0.1]], [0]), None))
        self.assertIsNone(
            serving_completion._build_logprobs_response(LogprobsLists([[1]], [[-0.1]], [0]), request_top_logprobs=-1)
        )

    def test_build_logprobs_response_replaces_invalid_token(self):
        """_build_logprobs_response should fallback to bytes when token decode is invalid."""
        # Mock data_processor to avoid real tokenizer dependency.
        engine_client = Mock()
        engine_client.data_processor = Mock()
        engine_client.data_processor.process_logprob_response = Mock(return_value="\ufffd")
        engine_client.data_processor.tokenizer = Mock()
        engine_client.data_processor.tokenizer.convert_ids_to_tokens = Mock(return_value="A")
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        response_logprobs = LogprobsLists([[65]], [[-0.1]], [0])
        result = serving_completion._build_logprobs_response(response_logprobs, request_top_logprobs=0)

        self.assertIsNotNone(result)
        self.assertEqual(result.tokens, ["bytes:\\x41"])
        self.assertEqual(result.token_logprobs, [-0.1])
        self.assertEqual(result.top_logprobs[0]["bytes:\\x41"], -0.1)

    def test_build_logprobs_response_handles_processor_error(self):
        """_build_logprobs_response should return None on processor errors."""
        engine_client = Mock()
        engine_client.data_processor = Mock()
        engine_client.data_processor.process_logprob_response = Mock(side_effect=RuntimeError("boom"))
        engine_client.data_processor.tokenizer = Mock()
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        response_logprobs = LogprobsLists([[1]], [[-0.1]], [0])
        result = serving_completion._build_logprobs_response(response_logprobs, request_top_logprobs=0)

        self.assertIsNone(result)
        engine_client.data_processor.process_logprob_response.assert_called_once()

    def test_build_prompt_logprobs_without_decode(self):
        """_build_prompt_logprobs should not decode tokens when disabled."""
        engine_client = Mock()
        engine_client.data_processor = Mock()
        engine_client.data_processor.process_logprob_response = Mock()
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        token_ids = paddle.to_tensor([[1, 2]], dtype=paddle.int64)
        logprobs = paddle.to_tensor([[-0.1, -0.2]], dtype=paddle.float32)
        ranks = paddle.to_tensor([1], dtype=paddle.int64)
        prompt_logprobs_tensors = LogprobsTensors(token_ids, logprobs, ranks)

        result = serving_completion._build_prompt_logprobs(prompt_logprobs_tensors, 2, False)

        self.assertIsNone(result[1][1].decoded_token)
        engine_client.data_processor.process_logprob_response.assert_not_called()

    def test_create_completion_logprobs_invalid_inputs(self):
        """_create_completion_logprobs should return None on invalid inputs."""
        engine_client = Mock()
        engine_client.data_processor = Mock()
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        self.assertIsNone(serving_completion._create_completion_logprobs(None, 3, 0))
        self.assertIsNone(serving_completion._create_completion_logprobs([], 3, 0))
        self.assertIsNone(serving_completion._create_completion_logprobs([[1], [2]], 3, 0))

    async def test_process_echo_logic_for_list_prompt_ids(self):
        """_process_echo_logic should prepend decoded prompt for list[int]."""
        # Mock tokenizer to avoid external dependency.
        engine_client = Mock()
        engine_client.data_processor = Mock()
        engine_client.data_processor.tokenizer = Mock()
        engine_client.data_processor.tokenizer.decode = Mock(return_value="decoded_prompt")
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        request = Mock()
        request.echo = True
        request.n = 1
        request.prompt = [1, 2, 3]
        res_outputs = {"send_idx": 0, "text": "X"}

        result = await serving_completion._process_echo_logic(request, 0, res_outputs)
        self.assertEqual(result["text"], "decoded_promptX")

    async def test_process_echo_logic_for_list_of_list_prompt_ids(self):
        """_process_echo_logic should decode list[list[int]] prompt by index."""
        # Mock tokenizer to avoid external dependency.
        engine_client = Mock()
        engine_client.data_processor = Mock()
        engine_client.data_processor.tokenizer = Mock()
        engine_client.data_processor.tokenizer.decode = Mock(return_value="decoded_nested")
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        request = Mock()
        request.echo = True
        request.n = 1
        request.prompt = [[7, 8], [9, 10]]
        res_outputs = {"send_idx": 0, "text": "Y"}

        result = await serving_completion._process_echo_logic(request, 1, res_outputs)
        self.assertEqual(result["text"], "decoded_nestedY")

    async def test_create_completion_rejects_non_master(self):
        """create_completion should reject requests on non-master nodes."""
        engine_client = Mock()
        engine_client.is_master = False
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "0.0.0.0", 360)

        request = Mock()
        request.prompt = "hi"
        request.prompt_token_ids = None
        request.stream = False
        request.n = 1
        request.request_id = None
        request.user = None
        request.trace_context = {}

        result = await serving_completion.create_completion(request)

        self.assertIsNotNone(result.error)
        self.assertIn("Only master node", result.error.message)

    async def test_create_completion_invalid_prompt_token_ids(self):
        """create_completion should return error for invalid prompt_token_ids types."""
        engine_client = Mock()
        engine_client.is_master = True
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        request = Mock()
        request.prompt = "hi"
        request.prompt_token_ids = ["bad"]
        request.stream = False
        request.n = 1
        request.request_id = "req"
        request.user = None
        request.trace_context = {}

        result = await serving_completion.create_completion(request)

        self.assertIsNotNone(result.error)
        self.assertIn("prompt_token_ids", result.error.message)

    async def test_create_completion_timeout_waiting_semaphore(self):
        """create_completion should return timeout error on semaphore wait."""
        engine_client = Mock()
        engine_client.is_master = True
        engine_client.semaphore = Mock()

        async def slow_acquire():
            await asyncio.sleep(0.05)

        engine_client.semaphore.acquire = AsyncMock(side_effect=slow_acquire)
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 0.01)

        request = Mock()
        request.prompt = "hi"
        request.prompt_token_ids = None
        request.stream = False
        request.n = 1
        request.request_id = None
        request.user = None
        request.trace_context = {}

        result = await serving_completion.create_completion(request)

        self.assertEqual(result.error.code, ErrorCode.TIMEOUT)

    async def test_create_completion_parameter_error(self):
        """create_completion should map ParameterError to invalid_request."""
        engine_client = Mock()
        engine_client.is_master = True
        engine_client.semaphore = Mock()
        engine_client.semaphore.acquire = AsyncMock()
        engine_client.semaphore.release = Mock()
        engine_client.format_and_add_data = AsyncMock(side_effect=ParameterError("max_tokens", "bad max tokens"))
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, -1)

        request = Mock()
        request.prompt = "hi"
        request.prompt_token_ids = None
        request.stream = False
        request.n = 1
        request.request_id = "req123"
        request.user = None
        request.trace_context = {}

        def to_dict_for_infer(request_id_idx, prompt):
            return {
                "prompt": prompt,
                "request_id": request_id_idx,
                "prompt_tokens": [1],
                "max_tokens": 2,
                "metrics": {},
            }

        request.to_dict_for_infer = to_dict_for_infer

        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            result = await serving_completion.create_completion(request)

        self.assertEqual(result.error.code, "400")
        self.assertEqual(result.error.param, "max_tokens")

    async def test_create_completion_stream_branch_prefixes_request_id(self):
        """create_completion should prefix request_id and return stream handler."""
        engine_client = Mock()
        engine_client.is_master = True
        engine_client.semaphore = Mock()
        engine_client.semaphore.acquire = AsyncMock()
        engine_client.format_and_add_data = AsyncMock(return_value=np.array([1, 2]))
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, -1)

        request = Mock()
        request.prompt = "unused"
        request.prompt_token_ids = [1, 2]
        request.stream = True
        request.n = 1
        request.request_id = "req123"
        request.user = None
        request.trace_context = {}

        captured = {}

        def to_dict_for_infer(request_id_idx, prompt):
            captured["rid"] = request_id_idx
            return {
                "prompt": prompt,
                "request_id": request_id_idx,
                "prompt_tokens": [1],
                "max_tokens": 2,
                "metrics": {},
            }

        request.to_dict_for_infer = to_dict_for_infer

        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            with patch.object(serving_completion, "completion_stream_generator", return_value="streamed"):
                result = await serving_completion.create_completion(request)

        self.assertEqual(result, "streamed")
        self.assertEqual(captured["rid"], "cmpl-req123_0")

    async def test_create_completion_generic_format_error(self):
        """create_completion should surface generic format errors."""
        engine_client = Mock()
        engine_client.is_master = True
        engine_client.semaphore = Mock()
        engine_client.semaphore.acquire = AsyncMock()
        engine_client.semaphore.release = Mock()
        engine_client.format_and_add_data = AsyncMock(side_effect=ValueError("bad format"))
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, -1)

        request = Mock()
        request.prompt = "ignored"
        request.prompt_token_ids = [[1, 2]]
        request.stream = False
        request.n = 1
        request.request_id = "req"
        request.user = None
        request.trace_context = {}

        def from_generic_request(_, request_id):
            return {
                "prompt": "ignored",
                "request_id": request_id,
                "prompt_tokens": [1],
                "max_tokens": 2,
                "metrics": {},
            }

        with patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", True):
            with patch(
                "fastdeploy.entrypoints.openai.serving_completion.Request.from_generic_request",
                side_effect=from_generic_request,
            ):
                result = await serving_completion.create_completion(request)

        self.assertEqual(result.error.code, ErrorCode.INVALID_VALUE)
        engine_client.semaphore.release.assert_called_once()

    async def test_call_process_response_dict_async_and_sync(self):
        """_call_process_response_dict should handle async and sync handlers."""
        engine_client_async = Mock()
        engine_client_async.data_processor = Mock()
        engine_client_async.data_processor.process_response_dict = AsyncMock()
        serving_completion_async = OpenAIServingCompletion(engine_client_async, None, "pid", None, 360)

        request = Mock()
        request.include_stop_str_in_output = True
        await serving_completion_async._call_process_response_dict({"a": 1}, request, stream=True)
        engine_client_async.data_processor.process_response_dict.assert_awaited_once_with(
            {"a": 1}, stream=True, include_stop_str_in_output=True
        )

        engine_client_sync = Mock()
        engine_client_sync.data_processor = Mock()
        engine_client_sync.data_processor.process_response_dict = Mock()
        serving_completion_sync = OpenAIServingCompletion(engine_client_sync, None, "pid", None, 360)
        await serving_completion_sync._call_process_response_dict({"b": 2}, request, stream=False)
        engine_client_sync.data_processor.process_response_dict.assert_called_once_with(
            {"b": 2}, stream=False, include_stop_str_in_output=True
        )

    def test_request_output_to_completion_response_without_echo(self):
        """request_output_to_completion_response should preserve token ids when echo=False."""
        engine_client = Mock()
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        final_res_batch = [
            {
                "outputs": {
                    "token_ids": [3, 4],
                    "text": "ok",
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "num_cache_tokens": 2,
                    "num_image_tokens": 1,
                    "reasoning_token_num": 5,
                },
                "output_token_ids": 2,
                "metrics": {},
            }
        ]

        request = Mock()
        request.prompt = "hi"
        request.echo = False
        request.n = None
        request.logprobs = None
        request.prompt_logprobs = None
        request.include_logprobs_decode_token = False
        request.return_token_ids = False

        response = serving_completion.request_output_to_completion_response(
            final_res_batch=final_res_batch,
            request=request,
            request_id="req",
            created_time=123,
            model_name="m",
            prompt_batched_token_ids=[[1, 2, 3]],
            completion_batched_token_ids=[[3, 4]],
            prompt_tokens_list=[["p1", "p2", "p3"]],
            max_tokens_list=[10],
        )

        self.assertEqual(response.choices[0].text, "ok")
        self.assertIsNone(response.choices[0].prompt_token_ids)
        self.assertIsNone(response.choices[0].completion_token_ids)
        self.assertEqual(response.usage.completion_tokens_details.image_tokens, 1)
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 5)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 2)
        self.assertEqual(response.usage.total_tokens, 3 + 3)

    async def test_completion_stream_generator_usage_and_tool_calls(self):
        """completion_stream_generator should emit usage chunk and tool_calls finish_reason."""
        engine_client = Mock()
        engine_client.semaphore = Mock()
        engine_client.semaphore.release = Mock()
        engine_client.connection_manager = AsyncMock()
        engine_client.data_processor = Mock()
        engine_client.ori_vocab_size = 1000
        engine_client.check_model_weight_status.return_value = False
        engine_client.check_health.return_value = (True, "Healthy")
        engine_client.data_processor.process_response_dict = Mock()

        dealer = Mock()
        dealer.write = Mock()
        response_queue = AsyncMock()
        response_queue.get.return_value = [
            {
                "request_id": "req_0",
                "error_code": 200,
                "metrics": {
                    "arrival_time": 1,
                    "inference_start_time": 1,
                    "first_token_time": 1,
                    "engine_recv_latest_token_time": 2,
                },
                "outputs": {
                    "text": "hi",
                    "token_ids": [10],
                    "top_logprobs": None,
                    "draft_top_logprobs": None,
                    "send_idx": 0,
                    "completion_tokens": 1,
                    "num_cache_tokens": 1,
                    "num_image_tokens": 2,
                    "reasoning_token_num": 3,
                    "tool_calls": [],
                    "reasoning_content": "",
                    "skipped": False,
                },
                "finished": True,
            }
        ]
        engine_client.connection_manager.get_connection.return_value = (dealer, response_queue)

        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", None, 360)

        request = Mock()
        request.prompt_logprobs = None
        request.logprobs = None
        request.include_draft_logprobs = False
        request.return_token_ids = False
        request.include_stop_str_in_output = False
        request.max_streaming_response_tokens = 1
        request.max_tokens = None
        request.stream_options = SimpleNamespace(include_usage=True)
        request.n = 1
        request.echo = False
        request.collect_metrics = False

        results = []
        async for item in serving_completion.completion_stream_generator(
            request=request,
            num_choices=1,
            request_id="req",
            created_time=123,
            model_name="m",
            prompt_batched_token_ids=[[1, 2, 3]],
            prompt_tokens_list=[["p1", "p2", "p3"]],
            max_tokens_list=[5],
        ):
            results.append(item)

        usage_payloads = [r for r in results if '"usage"' in r]
        self.assertTrue(usage_payloads)
        self.assertTrue(any('"finish_reason":"tool_calls"' in r for r in results))


_BASE_TEST_PATH = Path(__file__).resolve().parent / "openai" / "test_serving_completion.py"
_BASE_SPEC = importlib.util.spec_from_file_location("fd_openai_test_serving_completion", _BASE_TEST_PATH)
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MODULE)


class TestServingCompletionMirror(_BASE_MODULE.TestOpenAIServingCompletion):
    """Reuse existing OpenAI completion tests to align coverage with CI."""


if __name__ == "__main__":
    unittest.main()
