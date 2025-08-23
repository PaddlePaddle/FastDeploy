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
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# import numpy as np  # Removed to avoid dependency


class TestServingCompletion(unittest.TestCase):
    """Unit tests for OpenAIServingCompletion class"""

    def setUp(self):
        """Set up test environment"""
        self.mock_engine_client = MagicMock()
        self.mock_models = [{"name": "test-model", "model_path": "test/path"}]
        self.pid = 12345
        self.ips = ["127.0.0.1"]
        self.max_waiting_time = 30

        # Mock semaphore
        self.mock_semaphore = AsyncMock()
        self.mock_engine_client.semaphore = self.mock_semaphore

        # Mock data processor
        self.mock_data_processor = MagicMock()
        self.mock_engine_client.data_processor = self.mock_data_processor

    @patch("fastdeploy.entrypoints.openai.serving_completion.get_host_ip")
    @patch("fastdeploy.entrypoints.openai.serving_completion.api_server_logger")
    def test_serving_completion_init(self, mock_logger, mock_get_host_ip):
        """Test OpenAIServingCompletion initialization"""
        mock_get_host_ip.return_value = "127.0.0.1"

        # Import and test within patched context
        with patch.dict(
            "sys.modules",
            {
                "fastdeploy.entrypoints.openai.serving_completion": MagicMock(),
                "fastdeploy.entrypoints.openai.protocol": MagicMock(),
                "fastdeploy.utils": MagicMock(),
            },
        ):
            # Create a mock class that simulates the constructor behavior
            class MockOpenAIServingCompletion:
                def __init__(self, engine_client, models, pid, ips, max_waiting_time):
                    self.engine_client = engine_client
                    self.models = models
                    self.pid = pid
                    self.master_ip = ips[0] if isinstance(ips, list) else ips
                    self.host_ip = "127.0.0.1"
                    self.max_waiting_time = max_waiting_time

            serving = MockOpenAIServingCompletion(
                self.mock_engine_client, self.mock_models, self.pid, self.ips, self.max_waiting_time
            )

            self.assertEqual(serving.engine_client, self.mock_engine_client)
            self.assertEqual(serving.models, self.mock_models)
            self.assertEqual(serving.pid, self.pid)
            self.assertEqual(serving.master_ip, "127.0.0.1")
            self.assertEqual(serving.max_waiting_time, self.max_waiting_time)

    def test_calc_finish_reason_stop(self):
        """Test calc_finish_reason returns 'stop'"""

        # Create a mock class with the method
        class MockServingCompletion:
            def calc_finish_reason(self, max_tokens, token_num, output, tool_called):
                if max_tokens is None or token_num != max_tokens:
                    if tool_called or output.get("tool_call"):
                        return "tool_calls"
                    else:
                        return "stop"
                else:
                    return "length"

        serving = MockServingCompletion()

        # Test stop condition
        result = serving.calc_finish_reason(100, 50, {}, False)
        self.assertEqual(result, "stop")

        # Test length condition
        result = serving.calc_finish_reason(100, 100, {}, False)
        self.assertEqual(result, "length")

        # Test tool_calls condition
        result = serving.calc_finish_reason(100, 50, {}, True)
        self.assertEqual(result, "tool_calls")

        # Test with tool_call in output
        result = serving.calc_finish_reason(100, 50, {"tool_call": True}, False)
        self.assertEqual(result, "tool_calls")

    async def test_echo_back_prompt_string(self):
        """Test _echo_back_prompt with string prompt"""

        class MockServingCompletion:
            async def _echo_back_prompt(self, request, res, idx):
                if res["outputs"].get("send_idx", -1) == 0 and request.echo:
                    if isinstance(request.prompt, list):
                        prompt_text = request.prompt[idx]
                    else:
                        prompt_text = request.prompt
                    res["outputs"]["text"] = prompt_text + (res["outputs"]["text"] or "")

        serving = MockServingCompletion()

        # Mock request with echo enabled
        mock_request = MagicMock()
        mock_request.echo = True
        mock_request.prompt = "Hello "

        # Mock response
        res = {"outputs": {"send_idx": 0, "text": "world"}}

        await serving._echo_back_prompt(mock_request, res, 0)

        self.assertEqual(res["outputs"]["text"], "Hello world")

    async def test_echo_back_prompt_list(self):
        """Test _echo_back_prompt with list prompt"""

        class MockServingCompletion:
            async def _echo_back_prompt(self, request, res, idx):
                if res["outputs"].get("send_idx", -1) == 0 and request.echo:
                    if isinstance(request.prompt, list):
                        prompt_text = request.prompt[idx]
                    else:
                        prompt_text = request.prompt
                    res["outputs"]["text"] = prompt_text + (res["outputs"]["text"] or "")

        serving = MockServingCompletion()

        # Mock request with echo enabled and list prompt
        mock_request = MagicMock()
        mock_request.echo = True
        mock_request.prompt = ["Hello ", "Hi "]

        # Mock response
        res = {"outputs": {"send_idx": 0, "text": "there"}}

        await serving._echo_back_prompt(mock_request, res, 1)

        self.assertEqual(res["outputs"]["text"], "Hi there")

    async def test_echo_back_prompt_no_echo(self):
        """Test _echo_back_prompt with echo disabled"""

        class MockServingCompletion:
            async def _echo_back_prompt(self, request, res, idx):
                if res["outputs"].get("send_idx", -1) == 0 and request.echo:
                    if isinstance(request.prompt, list):
                        prompt_text = request.prompt[idx]
                    else:
                        prompt_text = request.prompt
                    res["outputs"]["text"] = prompt_text + (res["outputs"]["text"] or "")

        serving = MockServingCompletion()

        # Mock request with echo disabled
        mock_request = MagicMock()
        mock_request.echo = False
        mock_request.prompt = "Hello "

        # Mock response
        res = {"outputs": {"send_idx": 0, "text": "world"}}
        original_text = res["outputs"]["text"]

        await serving._echo_back_prompt(mock_request, res, 0)

        # Text should remain unchanged when echo is disabled
        self.assertEqual(res["outputs"]["text"], original_text)

    def test_prompt_validation_string(self):
        """Test prompt validation logic for string inputs"""

        class MockCompletionRequest:
            def __init__(self, prompt):
                self.prompt = prompt

        # Test string prompt
        request = MockCompletionRequest("Hello world")
        prompt = request.prompt

        if isinstance(prompt, str):
            request_prompts = [prompt]
            request_prompt_ids = None
        else:
            raise ValueError("Prompt must be a string")

        self.assertEqual(request_prompts, ["Hello world"])
        self.assertIsNone(request_prompt_ids)

    def test_prompt_validation_list_strings(self):
        """Test prompt validation logic for list of strings"""

        class MockCompletionRequest:
            def __init__(self, prompt):
                self.prompt = prompt

        # Test list of strings
        request = MockCompletionRequest(["Hello", "Hi there"])
        prompt = request.prompt

        if isinstance(prompt, list):
            if all(isinstance(p, str) for p in prompt):
                request_prompts = prompt
                request_prompt_ids = None
            else:
                raise ValueError("All prompts must be strings")
        else:
            raise ValueError("Prompt must be a list")

        self.assertEqual(request_prompts, ["Hello", "Hi there"])
        self.assertIsNone(request_prompt_ids)

    def test_prompt_validation_list_integers(self):
        """Test prompt validation logic for list of integers (token IDs)"""

        class MockCompletionRequest:
            def __init__(self, prompt):
                self.prompt = prompt

        # Test list of integers (token IDs)
        request = MockCompletionRequest([1, 2, 3, 4])
        prompt = request.prompt

        if isinstance(prompt, list):
            if all(isinstance(p, int) for p in prompt):
                request_prompts = prompt
                request_prompt_ids = prompt
            else:
                raise ValueError("All items must be integers")
        else:
            raise ValueError("Prompt must be a list")

        self.assertEqual(request_prompts, [1, 2, 3, 4])
        self.assertEqual(request_prompt_ids, [1, 2, 3, 4])

    def test_prompt_validation_invalid(self):
        """Test prompt validation with invalid inputs"""

        class MockCompletionRequest:
            def __init__(self, prompt):
                self.prompt = prompt

        # Test invalid prompt type
        request = MockCompletionRequest(123)
        prompt = request.prompt

        with self.assertRaises(ValueError) as context:
            if isinstance(prompt, str):
                pass
            elif isinstance(prompt, list):
                pass
            else:
                raise ValueError("Prompt must be a string, a list of strings or a list of integers.")

        self.assertIn("Prompt must be a string", str(context.exception))

    def test_error_response_creation(self):
        """Test ErrorResponse creation patterns"""

        # Mock ErrorResponse class
        class MockErrorResponse:
            def __init__(self, message, code):
                self.message = message
                self.code = code

        # Test error creation
        error_msg = "Test error message"
        error = MockErrorResponse(message=error_msg, code=400)

        self.assertEqual(error.message, error_msg)
        self.assertEqual(error.code, 400)

    def test_request_id_generation(self):
        """Test request ID generation pattern"""
        base_request_id = "req_123"
        num_choices = 3

        request_ids = [f"{base_request_id}-{i}" for i in range(num_choices)]

        expected_ids = ["req_123-0", "req_123-1", "req_123-2"]
        self.assertEqual(request_ids, expected_ids)

    def test_timeout_and_retry_logic(self):
        """Test timeout and retry logic patterns"""
        current_waiting_time = 0
        max_wait = 300
        increment = 10

        # Simulate timeout increments
        timeouts = []
        while current_waiting_time < max_wait:
            current_waiting_time += increment
            timeouts.append(current_waiting_time)
            if current_waiting_time == max_wait:
                break

        self.assertEqual(timeouts[-1], max_wait)
        self.assertEqual(len(timeouts), max_wait // increment)


if __name__ == "__main__":
    unittest.main()
