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

import asyncio
import time
from typing import Any, Optional, Union

import httpx
from pydantic import BaseModel, HttpUrl

from fastdeploy.utils import data_processor_logger


class BaseEncodeRequest(BaseModel):
    version: str
    req_id: str
    is_gen: bool
    resolution: int


class ImageEncodeRequest(BaseEncodeRequest):
    image_url: Union[str, HttpUrl]


class VideoEncodeRequest(BaseEncodeRequest):
    video_url: Union[str, HttpUrl]
    start_ts: int
    end_ts: int
    frames: int
    vit_merge: bool


class ImageDecodeRequest(BaseModel):
    req_id: str
    data: list[Any]


class AsyncTokenizerClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        poll_interval: float = 0.5,
        max_wait: float = 60.0,
    ):
        """
        :param mode: 'local' 或 'remote'
        :param base_url: 远程服务地址
        :param timeout: 单次 HTTP 请求超时（秒）
        :param poll_interval: 查询结果的轮询间隔（秒）
        :param max_wait: 最大等待时间（秒）
        """
        self.base_url = base_url
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    async def encode_image(self, request: ImageEncodeRequest):
        return await self._async_encode_request("image", request.__dict__)

    async def encode_video(self, request: VideoEncodeRequest):
        return await self._async_encode_request("video", request.__dict__)

    async def decode_image(self, request: ImageDecodeRequest):
        return await self._async_decode_request("image", request.__dict__)

    async def log_request(self, request):
        data_processor_logger.debug(f">>> Request: {request.method} {request.url}")
        data_processor_logger.debug(f">>> Headers: {request.headers}")
        if request.content:
            data_processor_logger.debug(f">>> Content: {request.content.decode('utf-8')}")

    async def log_response(self, response):
        data_processor_logger.debug(f"<<< Response status: {response.status_code}")
        data_processor_logger.debug(f"<<< Headers: {response.headers}")

    async def _async_encode_request(self, type: str, request: dict):
        if not self.base_url:
            raise ValueError("Missing base_url")

        async with httpx.AsyncClient(
            timeout=self.timeout, event_hooks={"request": [self.log_request], "response": [self.log_response]}
        ) as client:
            req_id = request.get("req_id")
            try:
                url = None
                if type == "image":
                    url = f"{self.base_url}/image/encode"
                elif type == "video":
                    url = f"{self.base_url}/video/encode"
                else:
                    raise ValueError("Invalid type")

                resp = await client.post(url, json=request)
                resp.raise_for_status()
            except httpx.RequestError as e:
                raise RuntimeError(f"Failed to create tokenize task: {e}") from e

            task_info = resp.json()
            if task_info.get("code") != 0:
                raise RuntimeError(f"Tokenize task creation failed, {task_info.get('message')}")

            task_tag = task_info.get("task_tag")
            if not task_tag:
                raise RuntimeError("No task_tag returned from server")

            # 2. 轮询结果
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    r = await client.get(
                        f"{self.base_url}/encode/get", params={"task_tag": task_tag, "req_id": req_id}
                    )
                    r.raise_for_status()
                    data = r.json()

                    # 异步encode任务当前执行状态: Processing, Finished, Error
                    if data.get("state") == "Finished":
                        return data.get("result")
                    elif data.get("state") == "Error":
                        raise RuntimeError(f"Tokenize task failed: {data.get('message')}")

                except httpx.RequestError:
                    # 网络问题时继续轮询
                    pass

                # 超时检测
                if asyncio.get_event_loop().time() - start_time > self.max_wait:
                    raise TimeoutError(f"Tokenize task {task_tag} timed out after {self.max_wait}s")

                await asyncio.sleep(self.poll_interval)

    async def _async_decode_request(self, type: str, request: dict):
        if not self.base_url:
            raise ValueError("Missing base_url")

        async with httpx.AsyncClient(
            timeout=self.timeout, event_hooks={"request": [self.log_request], "response": [self.log_response]}
        ) as client:
            try:
                url = None
                if type == "image":
                    url = f"{self.base_url}/image/decode"
                else:
                    raise ValueError("Invalid type")

                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(url, json=request)
                        resp.raise_for_status()
                        if resp.json().get("code") != 0:
                            raise RuntimeError(f"Tokenize task creation failed, {resp.json().get('message')}")
                        return resp.json().get("result")
                    except Exception as e:
                        data_processor_logger.error(f"Attempt to decode_request {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            data_processor_logger.error(
                                f"Max retries of decode_request reached. Giving up. request is {request}"
                            )
                        time.sleep(10)
            except httpx.RequestError as e:
                raise RuntimeError(f"Failed to decode: {e}") from e
