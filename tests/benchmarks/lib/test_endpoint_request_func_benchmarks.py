"""
Test cases for endpoint_request_func.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastdeploy.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
    async_request_deepspeed_mii,
    async_request_eb_openai_chat_completions,
    async_request_eb_openai_completions,
    async_request_openai_audio,
    async_request_openai_completions,
    async_request_tgi,
    async_request_trt_llm,
)


@pytest.fixture
def mock_request_input():
    return RequestFuncInput(
        no=1,
        prompt="test prompt",
        history_QA=None,
        hyper_parameters={},
        api_url="http://test.com/completions",
        prompt_len=10,
        output_len=20,
        model="test-model",
        debug=True,
    )


@pytest.mark.asyncio
async def test_async_request_eb_openai_chat_completions(mock_request_input):
    """Test async_request_eb_openai_chat_completions with mock response"""
    # Create a mock response that will work with the async context manager
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    # Mock the streaming response
    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {"prompt_tokens_details": {"cached_tokens": 5}}}\n\n',
        b'data: {"choices": [{"delta": {"content": " World"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_eb_openai_chat_completions(mock_request_input)

    assert output.success is True
    assert "Hello World" in output.generated_text
    assert output.ttft > 0


@pytest.mark.asyncio
async def test_async_request_eb_openai_completions(mock_request_input):
    """Test async_request_eb_openai_completions with mock response"""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.reason = "OK"
    mock_response.__aenter__.return_value = mock_response

    chunks = [
        b'data: {"choices": [{"text": "Test"}]}\n\n',
        b'data: {"choices": [{"text": " response"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_eb_openai_completions(mock_request_input)

    assert output.success is True
    assert "Test response" in output.generated_text


@pytest.mark.asyncio
async def test_async_request_tgi(mock_request_input):
    """Test async_request_tgi with mock response"""
    mock_request_input.api_url = "http://test.com/generate_stream"

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    chunks = [b'data: {"generated_text": "TGI response", "arrival_time": 1234567890}\n\n', b"data: [DONE]\n\n"]
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_tgi(mock_request_input)

    assert output.success is False


@pytest.mark.asyncio
async def test_async_request_trt_llm(mock_request_input):
    """Test async_request_trt_llm with mock response"""
    mock_request_input.api_url = "http://test.com/generate_stream"

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    chunks = [b'data: {"text_output": "TRT LLM response"}\n\n', b"data: [DONE]\n\n"]
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_trt_llm(mock_request_input)

    assert output.success is False


@pytest.mark.asyncio
async def test_async_request_openai_completions(mock_request_input):
    """Test async_request_openai_completions with mock response"""
    mock_request_input.api_url = "http://test.com/completions"

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    chunks = [
        b'data: {"choices": [{"text": "OpenAI"}]}\n\n',
        b'data: {"choices": [{"text": " Completions"}]}\n\n',
        b'data: {"usage": {"completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_openai_completions(mock_request_input)

    assert output.success is True
    assert "OpenAI Completions" in output.generated_text
    assert output.output_tokens == 2


@pytest.mark.asyncio
async def test_async_request_deepspeed_mii(mock_request_input):
    """Test async_request_deepspeed_mii with mock response"""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response
    mock_response.json = AsyncMock(return_value={"choices": [{"text": "DeepSpeed MII response"}]})

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_deepspeed_mii(mock_request_input)

    assert output.success is True
    assert "DeepSpeed MII response" in output.generated_text


@pytest.mark.asyncio
async def test_async_request_openai_audio(mock_request_input):
    """Test async_request_openai_audio with mock response"""
    pytest.skip("Skipping audio test due to soundfile dependency")

    # 保留测试结构但不实际执行
    mock_request_input.multi_modal_content = {"audio": (b"test", 16000)}
    mock_request_input.api_url = "http://test.com/transcriptions"

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response

    chunks = [b'data: {"choices": [{"delta": {"content": "test"}}]}\n\n']
    mock_response.content.__aiter__.return_value = chunks

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        output = await async_request_openai_audio(mock_request_input)

    assert output.success is True


@pytest.mark.asyncio
async def test_async_request_functions_dict():
    """Test ASYNC_REQUEST_FUNCS contains all expected functions"""
    assert len(ASYNC_REQUEST_FUNCS) >= 8
    assert "tgi" in ASYNC_REQUEST_FUNCS
    assert "openai-chat" in ASYNC_REQUEST_FUNCS
    assert "openai" in ASYNC_REQUEST_FUNCS
    assert "tensorrt-llm" in ASYNC_REQUEST_FUNCS
    assert "deepspeed-mii" in ASYNC_REQUEST_FUNCS
    assert "openai-audio" in ASYNC_REQUEST_FUNCS


@pytest.mark.asyncio
async def test_openai_compatible_backends():
    """Test OPENAI_COMPATIBLE_BACKENDS contains expected backends"""
    assert len(OPENAI_COMPATIBLE_BACKENDS) >= 2
    assert "openai-chat" in OPENAI_COMPATIBLE_BACKENDS
    assert "vllm" in OPENAI_COMPATIBLE_BACKENDS


@pytest.mark.asyncio
async def test_request_func_output_defaults():
    """Test RequestFuncOutput default values"""
    output = RequestFuncOutput()
    assert output.no == 0
    assert output.generated_text == ""
    assert output.success is False
    assert output.latency == 0.0
