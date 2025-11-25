import httpx
import pytest
import respx

from fastdeploy.input.tokenzier_client import (
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
