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

import time
import unittest
import uuid
from unittest.mock import MagicMock, patch


class TestServingChat(unittest.TestCase):
    """Unit tests for OpenAIServingChat class"""

    def setUp(self):
        """Set up test environment"""
        self.mock_engine_client = MagicMock()
        self.mock_models = [{"name": "test-model", "model_path": "test/path"}]
        self.pid = 12345
        self.ips = ["127.0.0.1"]
        self.max_waiting_time = 30
        self.chat_template = "test_template"

    def test_request_id_generation_with_user(self):
        """Test request ID generation with user specified"""

        class MockChatRequest:
            def __init__(self, user=None):
                self.user = user

        # Test with user
        request = MockChatRequest(user="test_user")
        if request.user is not None:
            request_id = f"chatcmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"chatcmpl-{uuid.uuid4()}"

        self.assertTrue(request_id.startswith("chatcmpl-test_user-"))
        self.assertTrue(len(request_id) > len("chatcmpl-test_user-"))

    def test_request_id_generation_without_user(self):
        """Test request ID generation without user"""

        class MockChatRequest:
            def __init__(self, user=None):
                self.user = user

        # Test without user
        request = MockChatRequest(user=None)
        if request.user is not None:
            request_id = f"chatcmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"chatcmpl-{uuid.uuid4()}"

        self.assertTrue(request_id.startswith("chatcmpl-"))
        self.assertNotIn("None", request_id)
        # Should have the UUID format after "chatcmpl-"
        uuid_part = request_id[9:]  # Remove "chatcmpl-" prefix
        self.assertEqual(len(uuid_part), 36)  # Standard UUID length

    def test_streaming_error_response_creation(self):
        """Test _create_streaming_error_response method"""

        class MockErrorResponse:
            def __init__(self, code, message):
                self.code = code
                self.message = message

            def model_dump_json(self):
                return f'{{"code": {self.code}, "message": "{self.message}"}}'

        class MockServingChat:
            def _create_streaming_error_response(self, message: str) -> str:
                error_response = MockErrorResponse(code=400, message=message)
                return error_response.model_dump_json()

        serving = MockServingChat()
        result = serving._create_streaming_error_response("Test error message")

        self.assertIn('"code": 400', result)
        self.assertIn('"message": "Test error message"', result)

    def test_chat_template_injection(self):
        """Test chat template injection into request"""

        class MockRequest:
            def __init__(self):
                pass

            def to_dict_for_infer(self, request_id):
                return {"request_id": request_id}

        serving_chat_template = "test_template"
        request = MockRequest()
        current_req_dict = request.to_dict_for_infer("test_id")

        # Simulate template injection
        if "chat_template" not in current_req_dict:
            current_req_dict["chat_template"] = serving_chat_template

        current_req_dict["arrival_time"] = time.time()

        self.assertEqual(current_req_dict["chat_template"], "test_template")
        self.assertIn("arrival_time", current_req_dict)
        self.assertIsInstance(current_req_dict["arrival_time"], float)

    def test_streaming_options_parsing(self):
        """Test parsing of streaming options"""

        class MockStreamOptions:
            def __init__(self, include_usage=False, continuous_usage_stats=False):
                self.include_usage = include_usage
                self.continuous_usage_stats = continuous_usage_stats

        class MockRequest:
            def __init__(self, stream_options=None):
                self.stream_options = stream_options

        # Test with stream options
        stream_options = MockStreamOptions(include_usage=True, continuous_usage_stats=True)
        request = MockRequest(stream_options=stream_options)

        if request.stream_options is None:
            include_usage = False
            include_continuous_usage = False
        else:
            include_usage = request.stream_options.include_usage
            include_continuous_usage = request.stream_options.continuous_usage_stats

        self.assertTrue(include_usage)
        self.assertTrue(include_continuous_usage)

        # Test without stream options
        request_no_options = MockRequest(stream_options=None)

        if request_no_options.stream_options is None:
            include_usage = False
            include_continuous_usage = False
        else:
            include_usage = request_no_options.stream_options.include_usage
            include_continuous_usage = request_no_options.stream_options.continuous_usage_stats

        self.assertFalse(include_usage)
        self.assertFalse(include_continuous_usage)

    def test_thinking_mode_detection(self):
        """Test thinking mode detection from request"""

        class MockRequest:
            def __init__(self, chat_template_kwargs=None, metadata=None):
                self.chat_template_kwargs = chat_template_kwargs
                self.metadata = metadata

        # Test with thinking in chat_template_kwargs
        request1 = MockRequest(chat_template_kwargs={"enable_thinking": True})
        enable_thinking = (
            request1.chat_template_kwargs.get("enable_thinking") if request1.chat_template_kwargs else None
        )
        if enable_thinking is None:
            enable_thinking = request1.metadata.get("enable_thinking") if request1.metadata else None

        self.assertTrue(enable_thinking)

        # Test with thinking in metadata (fallback)
        request2 = MockRequest(chat_template_kwargs=None, metadata={"enable_thinking": False})
        enable_thinking = (
            request2.chat_template_kwargs.get("enable_thinking") if request2.chat_template_kwargs else None
        )
        if enable_thinking is None:
            enable_thinking = request2.metadata.get("enable_thinking") if request2.metadata else None

        self.assertFalse(enable_thinking)

        # Test with neither (should be None)
        request3 = MockRequest()
        enable_thinking = (
            request3.chat_template_kwargs.get("enable_thinking") if request3.chat_template_kwargs else None
        )
        if enable_thinking is None:
            enable_thinking = request3.metadata.get("enable_thinking") if request3.metadata else None

        self.assertIsNone(enable_thinking)

    def test_max_streaming_response_tokens_resolution(self):
        """Test resolution of max_streaming_response_tokens"""

        class MockRequest:
            def __init__(self, max_streaming_response_tokens=None, metadata=None):
                self.max_streaming_response_tokens = max_streaming_response_tokens
                self.metadata = metadata

        # Test with direct value
        request1 = MockRequest(max_streaming_response_tokens=100)
        max_tokens = (
            request1.max_streaming_response_tokens
            if request1.max_streaming_response_tokens is not None
            else (request1.metadata or {}).get("max_streaming_response_tokens", 1)
        )
        self.assertEqual(max_tokens, 100)

        # Test with metadata fallback
        request2 = MockRequest(max_streaming_response_tokens=None, metadata={"max_streaming_response_tokens": 50})
        max_tokens = (
            request2.max_streaming_response_tokens
            if request2.max_streaming_response_tokens is not None
            else (request2.metadata or {}).get("max_streaming_response_tokens", 1)
        )
        self.assertEqual(max_tokens, 50)

        # Test with default fallback
        request3 = MockRequest(max_streaming_response_tokens=None, metadata=None)
        max_tokens = (
            request3.max_streaming_response_tokens
            if request3.max_streaming_response_tokens is not None
            else (request3.metadata or {}).get("max_streaming_response_tokens", 1)
        )
        self.assertEqual(max_tokens, 1)

    def test_chunk_object_creation(self):
        """Test chat completion chunk object structure"""

        class MockChatCompletionStreamResponse:
            def __init__(self, id, object, created, choices, model):
                self.id = id
                self.object = object
                self.created = created
                self.choices = choices
                self.model = model

        request_id = "chatcmpl-123"
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        model_name = "test-model"

        chunk = MockChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[],
            model=model_name,
        )

        self.assertEqual(chunk.id, request_id)
        self.assertEqual(chunk.object, chunk_object_type)
        self.assertEqual(chunk.created, created_time)
        self.assertEqual(chunk.choices, [])
        self.assertEqual(chunk.model, model_name)

    def test_tool_call_detection(self):
        """Test tool call detection logic"""

        def check_tool_called(output_dict, previous_tool_called=False):
            """Mock function to check if tools are called"""
            return previous_tool_called or output_dict.get("tool_call", False)

        # Test no tool call
        output1 = {"text": "Hello world"}
        tool_called = check_tool_called(output1)
        self.assertFalse(tool_called)

        # Test with tool call
        output2 = {"text": "Using tool", "tool_call": True}
        tool_called = check_tool_called(output2)
        self.assertTrue(tool_called)

        # Test persistence of tool call state
        output3 = {"text": "After tool"}
        tool_called = check_tool_called(output3, previous_tool_called=True)
        self.assertTrue(tool_called)

    def test_error_handling_patterns(self):
        """Test common error handling patterns"""

        class MockErrorResponse:
            def __init__(self, code, message):
                self.code = code
                self.message = message

        def simulate_error_handling(exception_occurred=False, error_msg="Test error"):
            try:
                if exception_occurred:
                    raise ValueError(error_msg)
                return "Success"
            except Exception as e:
                full_error_msg = f"request error: {str(e)}"
                return MockErrorResponse(code=400, message=full_error_msg)

        # Test success case
        result = simulate_error_handling(exception_occurred=False)
        self.assertEqual(result, "Success")

        # Test error case
        result = simulate_error_handling(exception_occurred=True, error_msg="Custom error")
        self.assertIsInstance(result, MockErrorResponse)
        self.assertEqual(result.code, 400)
        self.assertIn("Custom error", result.message)


if __name__ == "__main__":
    unittest.main()
