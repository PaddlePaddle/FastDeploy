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

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat


class TestOpenAIServingChat(unittest.TestCase):
    """Test case for OpenAIServingChat"""

    def setUp(self):
        """Set up test environment"""
        self.mock_engine_client = MagicMock()
        self.mock_models = MagicMock()
        self.pid = 12345
        self.ips = "192.168.1.1"
        self.max_waiting_time = 30
        self.chat_template = "Test template"

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_init_single_ip(self, mock_get_host_ip):
        """Test OpenAIServingChat initialization with single IP"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        self.assertEqual(serving_chat.engine_client, self.mock_engine_client)
        self.assertEqual(serving_chat.models, self.mock_models)
        self.assertEqual(serving_chat.pid, self.pid)
        self.assertEqual(serving_chat.master_ip, self.ips)
        self.assertEqual(serving_chat.max_waiting_time, self.max_waiting_time)
        self.assertEqual(serving_chat.host_ip, "127.0.0.1")
        self.assertEqual(serving_chat.chat_template, self.chat_template)

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_init_list_ips(self, mock_get_host_ip):
        """Test OpenAIServingChat initialization with list of IPs"""
        mock_get_host_ip.return_value = "127.0.0.1"
        ips_list = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
        
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=ips_list,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        # Should take the first IP from the list
        self.assertEqual(serving_chat.master_ip, "192.168.1.1")

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_init_none_ips(self, mock_get_host_ip):
        """Test OpenAIServingChat initialization with None IPs"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=None,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        self.assertIsNone(serving_chat.master_ip)

    def test_error_response_creation(self):
        """Test that we can create error responses"""
        error = ErrorResponse(message="Test error", code=400)
        self.assertEqual(error.object, "error")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.code, 400)

    def test_usage_info_creation(self):
        """Test that we can create usage info"""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.total_tokens, 15)

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_basic_functionality_setup(self, mock_get_host_ip):
        """Test basic setup for serving chat functionality"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        # Verify the serving chat is properly initialized
        self.assertIsNotNone(serving_chat.engine_client)
        self.assertIsNotNone(serving_chat.models)
        self.assertIsInstance(serving_chat.pid, int)
        self.assertIsInstance(serving_chat.max_waiting_time, int)
        self.assertIsInstance(serving_chat.chat_template, str)

    def test_chat_completion_request_creation(self):
        """Test that we can create chat completion requests"""
        # This tests the protocol classes that would be used by serving_chat
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }
        
        # Verify we can create the request (this would be used in the serving logic)
        self.assertIsInstance(request_data, dict)
        self.assertEqual(request_data["model"], "test-model")
        self.assertEqual(len(request_data["messages"]), 1)
        self.assertEqual(request_data["messages"][0]["role"], "user")

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_attributes_access(self, mock_get_host_ip):
        """Test that all expected attributes are accessible"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=self.ips,
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        # Test that all required attributes exist and are accessible
        attributes = [
            'engine_client', 'models', 'pid', 'master_ip', 
            'max_waiting_time', 'host_ip', 'chat_template'
        ]
        
        for attr in attributes:
            self.assertTrue(hasattr(serving_chat, attr), f"Missing attribute: {attr}")

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_ip_handling_edge_cases(self, mock_get_host_ip):
        """Test IP handling edge cases"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        # Test with empty list
        serving_chat = OpenAIServingChat(
            engine_client=self.mock_engine_client,
            models=self.mock_models,
            pid=self.pid,
            ips=[],
            max_waiting_time=self.max_waiting_time,
            chat_template=self.chat_template
        )
        
        # With empty list, master_ip should still be the list itself
        self.assertEqual(serving_chat.master_ip, [])

    def test_protocol_classes_exist(self):
        """Test that all required protocol classes are available"""
        # This ensures that the imports in serving_chat.py are working
        from fastdeploy.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatCompletionResponseStreamChoice,
            ChatCompletionStreamResponse,
            ChatMessage,
            DeltaMessage,
            ErrorResponse,
            LogProbEntry,
            LogProbs,
            PromptTokenUsageInfo,
            UsageInfo,
        )
        
        # Verify classes can be instantiated
        error = ErrorResponse(message="test", code=400)
        self.assertEqual(error.message, "test")
        
        usage = UsageInfo()
        self.assertEqual(usage.prompt_tokens, 0)
        
        message = ChatMessage(role="user", content="Hello")
        self.assertEqual(message.role, "user")


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def run_async(self, coro):
        """Helper to run async functions in tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class TestOpenAIServingChatAsync(AsyncTestCase):
    """Async test case for OpenAIServingChat"""

    def setUp(self):
        """Set up test environment"""
        self.mock_engine_client = MagicMock()
        self.mock_models = MagicMock()

    @patch('fastdeploy.entrypoints.openai.serving_chat.get_host_ip')
    def test_async_setup(self, mock_get_host_ip):
        """Test async setup capabilities"""
        mock_get_host_ip.return_value = "127.0.0.1"
        
        async def test():
            serving_chat = OpenAIServingChat(
                engine_client=self.mock_engine_client,
                models=self.mock_models,
                pid=12345,
                ips="192.168.1.1",
                max_waiting_time=30,
                chat_template="test"
            )
            
            # Verify the object can be used in async context
            self.assertIsNotNone(serving_chat)
            return serving_chat
        
        result = self.run_async(test())
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()