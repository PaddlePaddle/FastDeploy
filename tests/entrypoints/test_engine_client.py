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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.engine_client import EngineClient


class DummyConfig(SimpleNamespace):
    def __getattr__(self, name):
        return None


class TestEngineClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        with (
            patch("fastdeploy.entrypoints.engine_client.IPCSignal"),
            patch("fastdeploy.entrypoints.engine_client.DealerConnectionManager"),
            patch("fastdeploy.entrypoints.engine_client.InputPreprocessor"),
            patch("fastdeploy.entrypoints.engine_client.FileLock"),
            patch("fastdeploy.entrypoints.engine_client.StatefulSemaphore"),
        ):
            self.engine_config = DummyConfig(
                model_config=DummyConfig(enable_mm=True, enable_logprob=True, max_model_len=1024),
                cache_config=DummyConfig(enable_prefix_caching=True, max_processor_cache=10),
                scheduler_config=DummyConfig(splitwise_role="mixed", max_num_seqs=128),
                parallel_config=DummyConfig(tensor_parallel_size=1),
                structured_outputs_config=DummyConfig(reasoning_parser="reasoning_parser"),
                eplb_config=DummyConfig(enable_eplb=True, eplb_max_tokens=1024),
            )
            self.engine_client = EngineClient(pid=10000, port=1234, fd_config=self.engine_config)
            self.engine_client.zmq_client = MagicMock()

    def test_engine_client_initialized_by_fd_config(self):
        for config_group_name, config_group in self.engine_config.__dict__.items():
            for config_name, config_value in config_group.__dict__.items():
                if hasattr(self.engine_client, config_name):
                    assert getattr(self.engine_client, config_name) == config_value

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
