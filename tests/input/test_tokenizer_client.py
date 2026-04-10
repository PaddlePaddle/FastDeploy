"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import httpx
import pytest
import respx

from fastdeploy.input.tokenizer_client import (
    AsyncTokenizerClient,
    ImageEncodeRequest,
    VideoEncodeRequest,
)


@pytest.mark.asyncio
@respx.mock
async def test_encode_image_success():
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url)

    # Mock 创建任务接口
    respx.post(f"{base_url}/image/encode").mock(
        return_value=httpx.Response(200, json={"code": 0, "task_tag": "task123"})
    )
    # Mock 轮询接口，返回完成状态
    mock_get_ret = {
        "state": "Finished",
        "result": {"feature_url": "bos://host:port/key", "feature_shape": [80, 45, 1563]},
    }
    respx.get(f"{base_url}/encode/get").mock(return_value=httpx.Response(200, json=mock_get_ret))

    request = ImageEncodeRequest(
        version="v1", req_id="req_img_001", is_gen=False, resolution=512, image_url="http://example.com/image.jpg"
    )

    result = await client.encode_image(request)
    assert result["feature_url"] == "bos://host:port/key"
    assert result["feature_shape"] == [80, 45, 1563]


@pytest.mark.asyncio
@respx.mock
async def test_encode_video_failure():
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url, max_wait=1)

    respx.post(f"{base_url}/video/encode").mock(
        return_value=httpx.Response(200, json={"code": 0, "task_tag": "task_vid_001"})
    )
    # 模拟轮询接口失败状态
    respx.get(f"{base_url}/encode/get").mock(
        return_value=httpx.Response(200, json={"state": "Error", "message": "Encode failed"})
    )

    request = VideoEncodeRequest(
        version="v1",
        req_id="req_vid_001",
        is_gen=True,
        resolution=720,
        video_url="http://example.com/video.mp4",
        start_ts=0.0,
        end_ts=10.0,
        frames=30,
        vit_merge=True,
    )

    with pytest.raises(RuntimeError, match="Encode failed"):
        await client.encode_video(request)


@pytest.mark.asyncio
@respx.mock
async def test_encode_timeout():
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url, max_wait=1, poll_interval=0.1)

    respx.post(f"{base_url}/image/encode").mock(
        return_value=httpx.Response(200, json={"code": 0, "task_tag": "task_timeout"})
    )
    # 模拟轮询接口一直返回等待状态，导致超时
    respx.get(f"{base_url}/encode/get").mock(return_value=httpx.Response(200, json={"status": "processing"}))

    request = ImageEncodeRequest(
        version="v1", req_id="req_img_timeout", is_gen=False, resolution=256, image_url="http://example.com/image.jpg"
    )

    with pytest.raises(TimeoutError):
        await client.encode_image(request)


@pytest.mark.asyncio
async def test_encode_invalid_type():
    """Test invalid encode type raises ValueError (line 130).
    NOTE: Public methods hardcode the type param, so we test the private method directly
    to verify the validation boundary."""
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url)

    request = ImageEncodeRequest(
        version="v1", req_id="req_invalid", is_gen=False, resolution=256, image_url="http://example.com/image.jpg"
    )

    with pytest.raises(ValueError, match="Invalid encode type"):
        await client._async_encode_request("invalid_type", request.model_dump())


@pytest.mark.asyncio
async def test_decode_invalid_type():
    """Test invalid decode type raises ValueError (line 186).
    NOTE: Public methods hardcode the type param, so we test the private method directly
    to verify the validation boundary."""
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url)

    with pytest.raises(ValueError, match="Invalid decode type"):
        await client._async_decode_request("invalid_type", {})


@pytest.mark.asyncio
@respx.mock
async def test_encode_network_error_continues_polling():
    """Test network error during polling is caught and logged (line 164)."""
    base_url = "http://testserver"
    client = AsyncTokenizerClient(base_url=base_url, max_wait=2, poll_interval=0.1)

    # Mock create task
    respx.post(f"{base_url}/image/encode").mock(
        return_value=httpx.Response(200, json={"code": 0, "task_tag": "task_network_error"})
    )

    # First poll fails with network error, second succeeds
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.RequestError("Network error")
        return httpx.Response(200, json={"state": "Finished", "result": {"key": "value"}})

    respx.get(f"{base_url}/encode/get").mock(side_effect=side_effect)

    request = ImageEncodeRequest(
        version="v1", req_id="req_network", is_gen=False, resolution=256, image_url="http://example.com/image.jpg"
    )

    result = await client.encode_image(request)
    assert result["key"] == "value"
