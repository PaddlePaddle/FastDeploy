"""
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
"""

import json
import time
import unittest
from unittest.mock import patch

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionLogprobs,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
    LogProbEntry,
    LogProbs,
    ModelInfo,
    ModelList,
    ModelPermission,
    PromptTokenUsageInfo,
    ResponseFormat,
    StreamOptions,
    ToolCall,
    UsageInfo,
)


class TestProtocolModels(unittest.TestCase):
    """Test case for protocol models"""

    def test_error_response(self):
        """Test ErrorResponse model"""
        error = ErrorResponse(message="Test error", code=400)
        self.assertEqual(error.object, "error")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.code, 400)

    def test_prompt_token_usage_info(self):
        """Test PromptTokenUsageInfo model"""
        usage = PromptTokenUsageInfo(cached_tokens=100)
        self.assertEqual(usage.cached_tokens, 100)
        
        # Test with None
        usage_none = PromptTokenUsageInfo()
        self.assertIsNone(usage_none.cached_tokens)

    def test_usage_info(self):
        """Test UsageInfo model"""
        usage = UsageInfo(
            prompt_tokens=50,
            total_tokens=100,
            completion_tokens=50
        )
        self.assertEqual(usage.prompt_tokens, 50)
        self.assertEqual(usage.total_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)

    def test_usage_info_defaults(self):
        """Test UsageInfo model with defaults"""
        usage = UsageInfo()
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertIsNone(usage.prompt_tokens_details)

    def test_model_permission(self):
        """Test ModelPermission model"""
        with patch('time.time', return_value=1234567890):
            permission = ModelPermission()
            self.assertTrue(permission.id.startswith("modelperm-"))
            self.assertEqual(permission.object, "model_permission")
            self.assertEqual(permission.created, 1234567890)
            self.assertFalse(permission.allow_create_engine)
            self.assertTrue(permission.allow_sampling)
            self.assertTrue(permission.allow_logprobs)
            self.assertEqual(permission.organization, "*")

    def test_model_info(self):
        """Test ModelInfo model"""
        with patch('time.time', return_value=1234567890):
            model = ModelInfo(id="test-model")
            self.assertEqual(model.id, "test-model")
            self.assertEqual(model.object, "model")
            self.assertEqual(model.created, 1234567890)
            self.assertEqual(model.owned_by, "FastDeploy")
            self.assertEqual(model.permission, [])

    def test_model_info_with_max_len(self):
        """Test ModelInfo model with max_model_len"""
        model = ModelInfo(id="test-model", max_model_len=4096)
        self.assertEqual(model.max_model_len, 4096)

    def test_model_list(self):
        """Test ModelList model"""
        model1 = ModelInfo(id="model1")
        model2 = ModelInfo(id="model2")
        model_list = ModelList(data=[model1, model2])
        
        self.assertEqual(model_list.object, "list")
        self.assertEqual(len(model_list.data), 2)
        self.assertEqual(model_list.data[0].id, "model1")
        self.assertEqual(model_list.data[1].id, "model2")

    def test_function_call(self):
        """Test FunctionCall model"""
        func_call = FunctionCall(
            name="test_function",
            arguments='{"param": "value"}'
        )
        self.assertEqual(func_call.name, "test_function")
        self.assertEqual(func_call.arguments, '{"param": "value"}')

    def test_tool_call(self):
        """Test ToolCall model"""
        func_call = FunctionCall(name="test_func", arguments="{}")
        tool_call = ToolCall(id="tool-123", function=func_call)
        
        self.assertEqual(tool_call.id, "tool-123")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "test_func")

    def test_delta_function_call(self):
        """Test DeltaFunctionCall model"""
        delta_func = DeltaFunctionCall(
            name="test_func",
            arguments='{"partial": true}'
        )
        self.assertEqual(delta_func.name, "test_func")
        self.assertEqual(delta_func.arguments, '{"partial": true}')

    def test_delta_tool_call(self):
        """Test DeltaToolCall model"""
        delta_func = DeltaFunctionCall(name="test_func")
        delta_tool = DeltaToolCall(
            id="tool-123",
            type="function",
            index=0,
            function=delta_func
        )
        self.assertEqual(delta_tool.id, "tool-123")
        self.assertEqual(delta_tool.type, "function")
        self.assertEqual(delta_tool.index, 0)
        self.assertEqual(delta_tool.function.name, "test_func")

    def test_extracted_tool_call_information(self):
        """Test ExtractedToolCallInformation model"""
        func_call = FunctionCall(name="test_func", arguments="{}")
        tool_call = ToolCall(id="tool-123", function=func_call)
        
        extracted = ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[tool_call],
            content="Test content"
        )
        self.assertTrue(extracted.tools_called)
        self.assertEqual(len(extracted.tool_calls), 1)
        self.assertEqual(extracted.content, "Test content")

    def test_extracted_tool_call_information_no_tools(self):
        """Test ExtractedToolCallInformation when no tools called"""
        extracted = ExtractedToolCallInformation(
            tools_called=False,
            content="Regular response"
        )
        self.assertFalse(extracted.tools_called)
        self.assertIsNone(extracted.tool_calls)
        self.assertEqual(extracted.content, "Regular response")

    def test_function_definition(self):
        """Test FunctionDefinition model"""
        func_def = FunctionDefinition(
            name="test_function",
            description="A test function",
            parameters={"type": "object", "properties": {}}
        )
        self.assertEqual(func_def.name, "test_function")
        self.assertEqual(func_def.description, "A test function")
        self.assertIsInstance(func_def.parameters, dict)

    def test_chat_message(self):
        """Test ChatMessage model"""
        message = ChatMessage(
            role="user",
            content="Hello, world!",
            reasoning_content="User greeting"
        )
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.reasoning_content, "User greeting")
        self.assertIsNone(message.tool_calls)

    def test_chat_completion_response_choice(self):
        """Test ChatCompletionResponseChoice model"""
        message = ChatMessage(role="assistant", content="Hello!")
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.message.content, "Hello!")
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.logprobs)

    def test_chat_completion_response(self):
        """Test ChatCompletionResponse model"""
        message = ChatMessage(role="assistant", content="Hello!")
        choice = ChatCompletionResponseChoice(index=0, message=message)
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with patch('time.time', return_value=1234567890):
            response = ChatCompletionResponse(
                id="chatcmpl-123",
                model="test-model",
                choices=[choice],
                usage=usage
            )
            self.assertEqual(response.id, "chatcmpl-123")
            self.assertEqual(response.object, "chat.completion")
            self.assertEqual(response.created, 1234567890)
            self.assertEqual(response.model, "test-model")
            self.assertEqual(len(response.choices), 1)

    def test_log_prob_entry(self):
        """Test LogProbEntry model"""
        entry = LogProbEntry(
            token="hello",
            logprob=-1.5,
            bytes=[72, 101, 108, 108, 111]
        )
        self.assertEqual(entry.token, "hello")
        self.assertEqual(entry.logprob, -1.5)
        self.assertEqual(entry.bytes, [72, 101, 108, 108, 111])

    def test_log_probs(self):
        """Test LogProbs model"""
        entry = LogProbEntry(token="hello", logprob=-1.5)
        logprobs = LogProbs(content=[entry])
        
        self.assertEqual(len(logprobs.content), 1)
        self.assertEqual(logprobs.content[0].token, "hello")
        self.assertIsNone(logprobs.refusal)

    def test_delta_message(self):
        """Test DeltaMessage model"""
        delta = DeltaMessage(
            role="assistant",
            content="Partial content",
            reasoning_content="Partial reasoning"
        )
        self.assertEqual(delta.role, "assistant")
        self.assertEqual(delta.content, "Partial content")
        self.assertEqual(delta.reasoning_content, "Partial reasoning")

    def test_completion_logprobs(self):
        """Test CompletionLogprobs model"""
        logprobs = CompletionLogprobs(
            tokens=["hello", "world"],
            token_logprobs=[-1.5, -2.0],
            top_logprobs=[{"hello": -1.5}, {"world": -2.0}],
            text_offset=[0, 5]
        )
        self.assertEqual(logprobs.tokens, ["hello", "world"])
        self.assertEqual(logprobs.token_logprobs, [-1.5, -2.0])
        self.assertEqual(len(logprobs.top_logprobs), 2)
        self.assertEqual(logprobs.text_offset, [0, 5])

    def test_completion_response_choice(self):
        """Test CompletionResponseChoice model"""
        choice = CompletionResponseChoice(
            index=0,
            text="Hello, world!",
            finish_reason="stop"
        )
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.text, "Hello, world!")
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.logprobs)

    def test_completion_response(self):
        """Test CompletionResponse model"""
        choice = CompletionResponseChoice(index=0, text="Hello!")
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with patch('time.time', return_value=1234567890):
            response = CompletionResponse(
                id="cmpl-123",
                model="test-model",
                choices=[choice],
                usage=usage
            )
            self.assertEqual(response.id, "cmpl-123")
            self.assertEqual(response.object, "text_completion")
            self.assertEqual(response.created, 1234567890)
            self.assertEqual(len(response.choices), 1)

    def test_stream_options(self):
        """Test StreamOptions model"""
        options = StreamOptions(
            include_usage=False,
            continuous_usage_stats=True
        )
        self.assertFalse(options.include_usage)
        self.assertTrue(options.continuous_usage_stats)

    def test_stream_options_defaults(self):
        """Test StreamOptions model with defaults"""
        options = StreamOptions()
        self.assertTrue(options.include_usage)
        self.assertFalse(options.continuous_usage_stats)

    def test_response_format(self):
        """Test ResponseFormat model"""
        response_format = ResponseFormat(type="json_object")
        self.assertEqual(response_format.type, "json_object")
        self.assertIsNone(response_format.json_schema)

    def test_response_format_with_schema(self):
        """Test ResponseFormat model with JSON schema"""
        from fastdeploy.entrypoints.openai.protocol import JsonSchemaResponseFormat
        
        schema = JsonSchemaResponseFormat(
            name="test_schema",
            description="Test schema",
            strict=True
        )
        response_format = ResponseFormat(
            type="json_schema",
            json_schema=schema
        )
        self.assertEqual(response_format.type, "json_schema")
        self.assertEqual(response_format.json_schema.name, "test_schema")

    def test_model_serialization(self):
        """Test that models can be serialized to JSON"""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        json_data = usage.model_dump()
        
        expected = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": None
        }
        self.assertEqual(json_data, expected)

    def test_model_deserialization(self):
        """Test that models can be created from JSON"""
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
        usage = UsageInfo(**data)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()