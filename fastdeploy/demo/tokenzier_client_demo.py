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

from fastdeploy.input.tokenzier_client import (
    AsyncTokenizerClient,
    ImageDecodeRequest,
    ImageEncodeRequest,
    VideoEncodeRequest,
)


async def main():
    """
    测试AsyncTokenizerClient类
    """
    base_url = "http://example.com/"

    client = AsyncTokenizerClient(base_url=base_url)

    # # 测试图片编码请求
    image_encode_request = ImageEncodeRequest(
        version="v1", req_id="req_image_001", is_gen=False, resolution=512, image_url="http://example.com/image.jpg"
    )

    image_encode_ret = await client.encode_image(image_encode_request)
    print(f"Image encode result:{image_encode_ret}")

    # 测试视频编码请求
    video_encode_req = VideoEncodeRequest(
        version="v1",
        req_id="req_video_001",
        video_url="http://example.com/video.mp4",
        is_gen=False,
        resolution=1024,
        start_ts=0,
        end_ts=5,
        frames=1,
    )
    video_encode_result = await client.encode_video(video_encode_req)
    print(f"Video Encode Result:{video_encode_result}")
    # 测试图片解码请求
    with open("./image_decode_demo.json", "r", encoding="utf-8") as file:
        import json
        import time

        start_time = time.time()
        start_process_time = time.process_time()  # 记录开始时间
        json_data = json.load(file)
        image_decoding_request = ImageDecodeRequest(req_id="req_image_001", data=json_data.get("data"))
        # import pdb; pdb.set_trace()
        image_decode_result = await client.decode_image(image_decoding_request)
        print(f"Image decode result:{image_decode_result}")
        elapsed_time = time.time() - start_time
        elapsed_process_time = time.process_time() - start_process_time
        print(f"decode elapsed_time: {elapsed_time:.6f}s, elapsed_process_time: {elapsed_process_time:.6f}s")


if __name__ == "__main__":
    asyncio.run(main())
