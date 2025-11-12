import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Mock the argument parser and model loading before importing api_server
with (
    patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
    patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
):

    mock_args = SimpleNamespace(
        workers=1,
        model="test-model",
        revision=None,
        chat_template=None,
        tool_parser_plugin=None,
        max_concurrency=100,  # Add required attribute
        max_num_seqs=100,
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_logprob=False,
        enable_early_stop=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_partial_prefills=0,
        max_long_partial_prefills=0,
        long_prefill_token_threshold=0,
        cache_transfer_protocol=None,
        scheduler_name=None,
        scheduler_host=None,
        scheduler_port=None,
        scheduler_db=None,
        scheduler_password=None,
        scheduler_topic=None,
        api_key=None,
    )
    mock_parse_args.return_value = mock_args
    mock_retrive_model.return_value = "test-model"  # Just return the model name without downloading

    from fastdeploy.entrypoints.openai.api_server import wrap_streaming_generator


@pytest.mark.asyncio
async def test_wrap_streaming_generator_normal_flow():
    """Test normal streaming generation flow"""

    async def mock_generator():
        yield "chunk1"
        yield "chunk2"

    wrapped = wrap_streaming_generator(mock_generator())

    chunks = []
    async for chunk in wrapped():
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_wrap_streaming_generator_exception_handling():
    """Test exception handling in wrapped generator"""

    async def mock_generator():
        yield "chunk1"
        raise ValueError("test error")

    wrapped = wrap_streaming_generator(mock_generator())

    chunks = []
    with pytest.raises(ValueError, match="test error"):
        async for chunk in wrapped():
            chunks.append(chunk)

    assert chunks == ["chunk1"]


@pytest.mark.asyncio
async def test_wrap_streaming_generator_semaphore_release():
    """Test semaphore is released after generation"""

    mock_semaphore = MagicMock()

    async def mock_generator():
        yield "chunk"

    # Patch the global connection_semaphore
    with patch("fastdeploy.entrypoints.openai.api_server.connection_semaphore", mock_semaphore):
        wrapped = wrap_streaming_generator(mock_generator())

        async for _ in wrapped():
            pass

        mock_semaphore.release.assert_called_once()


@pytest.mark.asyncio
async def test_wrap_streaming_generator_span_recording():
    """Test span recording functionality"""

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    async def mock_generator():
        yield "chunk1"
        yield "chunk2"

    # Patch trace.get_current_span
    with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
        wrapped = wrap_streaming_generator(mock_generator())

        chunks = []
        async for chunk in wrapped():
            chunks.append(chunk)

        # Verify span events were recorded
        mock_span.add_event.assert_any_call("first_chunk", {"time": pytest.approx(time.time(), abs=1)})
        mock_span.add_event.assert_any_call(
            "last_chunk", {"time": pytest.approx(time.time(), abs=1), "total_chunk": 2}
        )


@pytest.mark.asyncio
async def test_wrap_streaming_generator_no_span():
    """Test behavior when no span is active"""

    async def mock_generator():
        yield "chunk"

    # Patch trace.get_current_span to return None
    with patch("opentelemetry.trace.get_current_span", return_value=None):
        wrapped = wrap_streaming_generator(mock_generator())

        chunks = []
        async for chunk in wrapped():
            chunks.append(chunk)

        assert chunks == ["chunk"]
