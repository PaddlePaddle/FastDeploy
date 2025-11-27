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

import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.engine_client import EngineClient


class TestEngineClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 创建 EngineClient 实例的模拟对象
        with patch.object(EngineClient, "__init__", return_value=None) as mock_init:
            self.engine_client = EngineClient("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        self.engine_client.data_processor = MagicMock()
        self.engine_client.zmq_client = MagicMock()
        self.engine_client.max_model_len = 1024
        self.engine_client.enable_mm = False

    async def test_add_request(self):
        request = {
            "request_id": "test-request-id",
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1],
            "chat_template": "Hello",
            "max_tokens": 20,
            "tools": [1],
        }

        await self.engine_client.add_requests(request)
        assert "chat_template" in request["chat_template_kwargs"], "'chat_template' not found in 'chat_template_kwargs"
        # assert "tools" in request["chat_template_kwargs"], "'tools' not found in 'chat_template_kwargs'"
        assert request["chat_template_kwargs"]["chat_template"] == "Hello"
        assert request["tools"] == [1]
        # assert request["chat_template_kwargs"]["tools"] == [1]

    def test_valid_parameters(self):
        request = {
            "request_id": "test-request-id",
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1],
            "chat_template": "Hello",
            "max_tokens": 20,
            "tools": [1],
            "temperature": 0,
        }
        self.engine_client.valid_parameters(request)
        assert request["temperature"] == 1e-6


if __name__ == "__main__":
    unittest.main()
