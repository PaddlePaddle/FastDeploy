from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fastdeploy.entrypoints.openai.response_processors import ChatResponseProcessor

AsyncTokenizerClient = AsyncMock


@pytest.fixture
def mock_data_processor():
    mock = MagicMock()
    mock.process_response_dict = MagicMock(return_value={"processed": True})
    return mock


@pytest_asyncio.fixture
async def processor_with_mm(mock_data_processor):
    p = ChatResponseProcessor(
        data_processor=mock_data_processor,
        enable_mm_output=True,
        eoi_token_id=101032,
        decoder_base_url="http://fake-decoder",
    )
    p.decoder_client.decode_image = AsyncMock(return_value={"http_url": "http://image.url/test.png"})
    return p


@pytest.mark.asyncio
async def test_process_response_chat_text_only(mock_data_processor):
    processor = ChatResponseProcessor(mock_data_processor)
    request_outputs = [{"outputs": {"text": "hello"}}]

    results = [r async for r in processor.process_response_chat(request_outputs, False, False, False)]

    mock_data_processor.process_response_dict.assert_called_once()
    assert results == [{"processed": True}]


@pytest.mark.asyncio
async def test_process_response_chat_mm_text_without_eoi(processor_with_mm):
    request_outputs = [{"request_id": "req1", "outputs": {"decode_type": 0, "token_ids": [1, 2], "text": "hi"}}]

    results = [r async for r in processor_with_mm.process_response_chat(request_outputs, False, False, False)]

    # text response should be wrapped as multipart
    assert results[0]["outputs"]["multipart"][0]["type"] == "text"
    assert results[0]["outputs"]["multipart"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_process_response_chat_mm_with_eoi(processor_with_mm):
    request_outputs = [
        {"request_id": "req2", "outputs": {"decode_type": 0, "token_ids": [101031], "text": "start"}},
        {"request_id": "req2", "outputs": {"decode_type": 1, "token_ids": [[11, 22]]}},
        {"request_id": "req2", "outputs": {"decode_type": 1, "token_ids": [[33, 44]]}},
        {"request_id": "req2", "outputs": {"decode_type": 0, "token_ids": [101032], "text": "done"}},
    ]

    results = [r async for r in processor_with_mm.process_response_chat(request_outputs, False, False, False)]

    # 第一个 yield 是 text
    text_part = results[0]["outputs"]["multipart"][0]
    assert text_part["type"] == "text"
    assert text_part["text"] == "start"

    # 第二个 yield 应该是 image
    image_part = results[1]["outputs"]["multipart"][0]
    assert results[1]["outputs"]["token_ids"] == [[11, 22], [33, 44]]
    assert image_part["type"] == "image"
    assert image_part["url"] == "http://image.url/test.png"

    # 第三个 yield 是 text
    text_part = results[2]["outputs"]["multipart"][0]
    assert text_part["type"] == "text"
    assert text_part["text"] == "done"


@pytest.mark.asyncio
async def test_process_response_chat_mm_buffer_accumulation(processor_with_mm):
    request_outputs = [{"request_id": "req3", "outputs": {"decode_type": 1, "token_ids": [[55, 66]]}}]

    results = [r async for r in processor_with_mm.process_response_chat(request_outputs, False, False, False)]

    # decode_type=1 不会 yield 任何输出，只会累积 buffer
    assert results == []
    assert processor_with_mm._mm_buffer == [[55, 66]]
