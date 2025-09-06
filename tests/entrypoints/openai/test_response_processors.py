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

import unittest
from unittest.mock import AsyncMock, MagicMock

from fastdeploy.entrypoints.openai.response_processors import ChatResponseProcessor


class TestChatResponseProcessor(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_data_processor = MagicMock()
        self.mock_data_processor.process_response_dict = MagicMock(
            side_effect=lambda response_dict, **_: {"processed": True, "raw": response_dict}
        )

    async def asyncSetUp(self):
        self.processor_mm = ChatResponseProcessor(
            data_processor=self.mock_data_processor,
            enable_mm_output=True,
            eoi_token_id=101032,
            eos_token_id=2,
            decoder_base_url="http://fake-decoder",
        )
        self.processor_mm.decoder_client.decode_image = AsyncMock(
            return_value={"http_url": "http://image.url/test.png"}
        )

    async def test_text_only_mode(self):
        """不开启 multimodal 时，直接走 data_processor"""
        processor = ChatResponseProcessor(self.mock_data_processor)
        request_outputs = [{"outputs": {"text": "hello"}}]

        results = [
            r
            async for r in processor.process_response_chat(
                request_outputs, stream=False, enable_thinking=False, include_stop_str_in_output=False
            )
        ]

        self.mock_data_processor.process_response_dict.assert_called_once()
        self.assertEqual(results[0]["processed"], True)
        self.assertEqual(results[0]["raw"]["outputs"]["text"], "hello")

    async def test_streaming_text_and_image(self):
        """流式模式下：text → image → text"""
        request_outputs = [
            {"request_id": "req1", "outputs": {"decode_type": 0, "token_ids": [1], "text": "hi"}},
            {"request_id": "req1", "outputs": {"decode_type": 1, "token_ids": [[11, 22]]}},
            {"request_id": "req1", "outputs": {"decode_type": 0, "token_ids": [101032], "text": "done"}},
        ]

        results = [
            r
            async for r in self.processor_mm.process_response_chat(
                request_outputs, stream=True, enable_thinking=False, include_stop_str_in_output=False
            )
        ]

        # 第一个 yield：text
        text_part = results[0]["outputs"]["multipart"][0]
        self.assertEqual(text_part["type"], "text")
        self.assertEqual(text_part["text"], "hi")

        # 第二个 yield：image（token_ids 被拼起来了）
        image_part = results[1]["outputs"]["multipart"][0]
        self.assertEqual(image_part["type"], "image")
        self.assertEqual(image_part["url"], "http://image.url/test.png")
        self.assertEqual(results[1]["outputs"]["token_ids"], [[[11, 22]]])

        # 第三个 yield：text
        text_part = results[2]["outputs"]["multipart"][0]
        self.assertEqual(text_part["type"], "text")
        self.assertEqual(text_part["text"], "done")

    async def test_streaming_buffer_accumulation(self):
        """流式模式：decode_type=1 只累积 buffer，不 yield"""
        request_outputs = [{"request_id": "req2", "outputs": {"decode_type": 1, "token_ids": [[33, 44]]}}]

        results = [
            r
            async for r in self.processor_mm.process_response_chat(
                request_outputs, stream=True, enable_thinking=False, include_stop_str_in_output=False
            )
        ]

        self.assertEqual(results, [])
        self.assertEqual(self.processor_mm._mm_buffer, [[[33, 44]]])

    async def test_non_streaming_accumulate_and_emit(self):
        """非流式模式：等 eos_token_id 才输出 multipart（text+image）"""
        request_outputs = [
            {"request_id": "req3", "outputs": {"decode_type": 0, "token_ids": [10], "text": "hello"}},
            {"request_id": "req3", "outputs": {"decode_type": 1, "token_ids": [[55, 66]]}},
            {"request_id": "req3", "outputs": {"decode_type": 0, "token_ids": [2], "text": "bye"}},  # eos_token_id
        ]

        results = [
            r
            async for r in self.processor_mm.process_response_chat(
                request_outputs, stream=False, enable_thinking=False, include_stop_str_in_output=False
            )
        ]

        # 只在最后一个输出 yield
        self.assertEqual(len(results), 1)
        multipart = results[0]["outputs"]["multipart"]

        self.assertEqual(multipart[0]["type"], "text")
        self.assertEqual(multipart[0]["text"], "hello")

        self.assertEqual(multipart[1]["type"], "image")
        self.assertEqual(multipart[1]["url"], "http://image.url/test.png")

        self.assertEqual(multipart[2]["type"], "text")
        self.assertEqual(multipart[2]["text"], "bye")


if __name__ == "__main__":
    unittest.main()
