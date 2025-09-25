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
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1],
            "chat_template": "Hello",
            "max_tokens": 20,
            "tools": [1],
        }

        await self.engine_client.add_requests(request)
        assert "chat_template" in request["chat_template_kwargs"], "'chat_template' not found in 'chat_template_kwargs"
        assert "tools" in request["chat_template_kwargs"], "'tools' not found in 'chat_template_kwargs'"
        assert request["chat_template_kwargs"]["chat_template"] == "Hello"
        assert request["chat_template_kwargs"]["tools"] == [1]


if __name__ == "__main__":
    unittest.main()
