import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Mock the argument parsing before importing anything from api_server
with patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args:
    mock_parse_args.return_value = MagicMock(workers=1, max_concurrency=10, local_data_parallel_id=0)

    # Now import the module
    from fastdeploy.entrypoints.openai.api_server import wrap_streaming_generator


@pytest.fixture
def mock_api_server_imports():
    """Mock the imports that cause command line argument parsing"""
    # Return the already imported function
    return wrap_streaming_generator


def test_wrap_streaming_generator_normal_flow(mock_api_server_imports):
    """Test normal streaming generation flow"""
    wrap_streaming_generator = mock_api_server_imports

    mock_generator = AsyncMock()
    mock_generator.__aiter__.return_value = iter([b"chunk1", b"chunk2"])

    wrapped = wrap_streaming_generator(mock_generator)
    result = []

    async def collect_chunks():
        async for chunk in wrapped():
            result.append(chunk)

    import asyncio

    asyncio.run(collect_chunks())

    assert result == [b"chunk1", b"chunk2"]


def test_wrap_streaming_generator_exception_handling(mock_api_server_imports):
    """Test exception handling and resource release"""
    wrap_streaming_generator = mock_api_server_imports

    mock_generator = AsyncMock()
    mock_generator.__aiter__.side_effect = Exception("Test error")

    wrapped = wrap_streaming_generator(mock_generator)

    async def test_exception():
        with pytest.raises(Exception, match="Test error"):
            async for _ in wrapped():
                pass

    import asyncio

    asyncio.run(test_exception())


def test_wrap_streaming_generator_with_span(mock_api_server_imports):
    """Test span recording functionality"""
    wrap_streaming_generator = mock_api_server_imports

    mock_generator = AsyncMock()
    mock_generator.__aiter__.return_value = iter([b"chunk1", b"chunk2"])

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    from opentelemetry import trace

    async def test_with_span():
        with patch.object(trace, "get_current_span", return_value=mock_span):
            wrapped = wrap_streaming_generator(mock_generator)
            async for _ in wrapped():
                pass

        # Check that events were recorded with correct names and structure
        first_chunk_calls = [call for call in mock_span.add_event.call_args_list if call[0][0] == "first_chunk"]
        last_chunk_calls = [call for call in mock_span.add_event.call_args_list if call[0][0] == "last_chunk"]

        assert len(first_chunk_calls) > 0, "first_chunk event was not recorded"
        assert len(last_chunk_calls) > 0, "last_chunk event was not recorded"

        # Verify the structure of the events
        first_chunk_call = first_chunk_calls[0]
        last_chunk_call = last_chunk_calls[0]

        assert "time" in first_chunk_call[0][1], "first_chunk event missing time"
        assert "time" in last_chunk_call[0][1], "last_chunk event missing time"
        assert "processed_tokens" in last_chunk_call[0][1], "last_chunk event missing processed_tokens"

    import asyncio

    asyncio.run(test_with_span())


def test_wrap_streaming_generator_no_span(mock_api_server_imports):
    """Test behavior when no span is available"""
    wrap_streaming_generator = mock_api_server_imports

    mock_generator = AsyncMock()
    mock_generator.__aiter__.return_value = iter([b"chunk1", b"chunk2"])

    from opentelemetry import trace

    async def test_no_span():
        with patch.object(trace, "get_current_span", return_value=None):
            wrapped = wrap_streaming_generator(mock_generator)
            result = []
            async for chunk in wrapped():
                result.append(chunk)

        assert result == [b"chunk1", b"chunk2"]

    import asyncio

    asyncio.run(test_no_span())
