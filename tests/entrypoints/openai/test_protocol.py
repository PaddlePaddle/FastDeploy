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

import json
import time
import unittest
from unittest.mock import patch

from fastdeploy.entrypoints.openai.protocol import (
    ErrorResponse,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
    ModelInfo,
    ModelList,
    ModelPermission,
    PromptTokenUsageInfo,
    ToolCall,
    UsageInfo,
)


class TestProtocolModels(unittest.TestCase):
    """Unit tests for OpenAI protocol models"""

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
        prompt_details = PromptTokenUsageInfo(cached_tokens=50)
        usage = UsageInfo(
            prompt_tokens=100,
            total_tokens=150,
            completion_tokens=50,
            prompt_tokens_details=prompt_details,
        )
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 50)

    def test_model_permission(self):
        """Test ModelPermission model"""
        with patch("time.time", return_value=1234567890):
            permission = ModelPermission()
            self.assertEqual(permission.object, "model_permission")
            self.assertEqual(permission.created, 1234567890)
            self.assertFalse(permission.allow_create_engine)
            self.assertTrue(permission.allow_sampling)
            self.assertTrue(permission.allow_logprobs)
            self.assertFalse(permission.allow_search_indices)
            self.assertTrue(permission.allow_view)
            self.assertFalse(permission.allow_fine_tuning)
            self.assertEqual(permission.organization, "*")
            self.assertIsNone(permission.group)
            self.assertFalse(permission.is_blocking)

    def test_model_info(self):
        """Test ModelInfo model"""
        with patch("time.time", return_value=1234567890):
            model = ModelInfo(id="test-model", max_model_len=2048)
            self.assertEqual(model.id, "test-model")
            self.assertEqual(model.object, "model")
            self.assertEqual(model.created, 1234567890)
            self.assertEqual(model.owned_by, "FastDeploy")
            self.assertEqual(model.max_model_len, 2048)
            self.assertIsNone(model.root)
            self.assertIsNone(model.parent)
            self.assertEqual(model.permission, [])

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
            name="get_weather",
            arguments='{"location": "New York"}'
        )
        self.assertEqual(func_call.name, "get_weather")
        self.assertEqual(func_call.arguments, '{"location": "New York"}')

    def test_tool_call(self):
        """Test ToolCall model"""
        func_call = FunctionCall(name="test_func", arguments="{}")
        tool_call = ToolCall(id="call_123", function=func_call)
        
        self.assertEqual(tool_call.id, "call_123")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "test_func")
        self.assertEqual(tool_call.function.arguments, "{}")

    def test_extracted_tool_call_information(self):
        """Test ExtractedToolCallInformation model"""
        # Test without tool calls
        info = ExtractedToolCallInformation(tools_called=False)
        self.assertFalse(info.tools_called)
        self.assertIsNone(info.tool_calls)
        self.assertIsNone(info.content)

        # Test with tool calls
        func_call = FunctionCall(name="test", arguments="{}")
        tool_call = ToolCall(id="123", function=func_call)
        info_with_tools = ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[tool_call],
            content="Some content"
        )
        self.assertTrue(info_with_tools.tools_called)
        self.assertEqual(len(info_with_tools.tool_calls), 1)
        self.assertEqual(info_with_tools.content, "Some content")

    def test_function_definition(self):
        """Test FunctionDefinition model"""
        func_def = FunctionDefinition(
            name="calculate",
            description="Performs calculations",
            parameters={"type": "object", "properties": {"x": {"type": "number"}}}
        )
        self.assertEqual(func_def.name, "calculate")
        self.assertEqual(func_def.description, "Performs calculations")
        self.assertIsInstance(func_def.parameters, dict)
        self.assertEqual(func_def.parameters["type"], "object")

        # Test minimal function definition
        minimal_func = FunctionDefinition(name="simple_func")
        self.assertEqual(minimal_func.name, "simple_func")
        self.assertIsNone(minimal_func.description)
        self.assertIsNone(minimal_func.parameters)


if __name__ == "__main__":
    unittest.main()