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
import asyncio
from typing import List
from unittest.mock import Mock

from fastdeploy.entrypoints.openai.serving_completion import (
    CompletionRequest,
    OpenAIServingCompletion,
    RequestOutput,
)


class TestCreateCompletion(unittest.TestCase):
    def test_create_(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie_x1"
        engine_client = Mock()
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)

    def test_request_prompt_handling(self):
        engine_client = Mock()
        engine_client.data_processor.tokenizer.decode = lambda x: f"decoded_{x}"
        
        serving_completion = OpenAIServingCompletion(engine_client, None, "pid", "ips", 360)
        
        # 测试整数列表输入
        request1 = CompletionRequest(prompt=[1, 2, 3], request_prompt_ids="ids1")
        asyncio.run(serving_completion.create_completion(request1))
        self.assertEqual(request1.prompt, ["decoded_[1, 2, 3]"])
        
        # 测试整数列表的列表输入
        request2 = CompletionRequest(prompt=[[1, 2], [3, 4]], request_prompt_ids="ids2")
        asyncio.run(serving_completion.create_completion(request2))
        self.assertEqual(request2.prompt, ["decoded_[1, 2]", "decoded_[3, 4]"])
        
        # 测试其他类型输入
        request3 = CompletionRequest(prompt="text prompt", request_prompt_ids="ids3")
        asyncio.run(serving_completion.create_completion(request3))
        self.assertEqual(request3.prompt, "text prompt")

if __name__ == "__main__":
    unittest.main()