import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.entrypoints.engine_client import EngineClient


@pytest.mark.asyncio
async def test_format_and_add_data():
    client = EngineClient.__new__(EngineClient)
    client.max_model_len = 100
    client.data_processor_pool = None
    client.add_requests = MagicMock()

    prompts = {"prompt_token_ids": [1, 2, 3], "prompt": "test prompt"}

    loop = asyncio.get_running_loop()

    async def mock_run_in_executor(executor, func, *args):
        return func(*args)

    with patch.object(loop, "run_in_executor", new=mock_run_in_executor):
        result = await client.format_and_add_data(prompts)

    client.add_requests.assert_called_once_with(prompts)

    assert result == prompts["prompt_token_ids"]

    assert "request_id" in prompts
    assert prompts["max_tokens"] == client.max_model_len - 1


@pytest.mark.asyncio
async def test_process_response_dict():
    engine_client = EngineClient.__new__(EngineClient)
    engine_client.data_processor_pool = None

    mock_process = MagicMock(return_value="processed_result")
    engine_client.data_processor = MagicMock()
    engine_client.data_processor.process_response_dict = mock_process

    loop = asyncio.get_running_loop()

    async def mock_run_in_executor(executor, func, *args):
        return func()

    with patch.object(loop, "run_in_executor", new=mock_run_in_executor):
        await engine_client.process_response_dict(
            response_dict={"key": "value"}, stream=True, enable_thinking=False, include_stop_str_in_output=True
        )

        engine_client.data_processor.process_response_dict.assert_called_once_with(
            response_dict={"key": "value"}, stream=True, enable_thinking=False, include_stop_str_in_output=True
        )
