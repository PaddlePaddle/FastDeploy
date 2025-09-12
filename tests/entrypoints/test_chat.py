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

import os
import unittest
import weakref

from fastdeploy.entrypoints.llm import LLM

MODEL_NAME = os.getenv("MODEL_PATH") + "/ERNIE-4.5-0.3B-Paddle"


class TestChat(unittest.TestCase):
    """Test case for chat functionality"""

    PROMPTS = [
        [{"content": "The color of tomato is ", "role": "user"}],
        [{"content": "The equation 2+3= ", "role": "user"}],
        [{"content": "The equation 4-1= ", "role": "user"}],
        [{"content": "PaddlePaddle is ", "role": "user"}],
    ]

    @classmethod
    def setUpClass(cls):
        try:
            llm = LLM(
                model=MODEL_NAME,
                max_num_batched_tokens=4096,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT")),
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT")),
            )
            cls.llm = weakref.proxy(llm)
        except Exception as e:
            print(f"Setting up LLM failed: {e}")
            raise unittest.SkipTest(f"LLM initialization failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        if hasattr(cls, "llm"):
            del cls.llm

    def test_chat(self):
        outputs = self.llm.chat(messages=self.PROMPTS, sampling_params=None)
        self.assertEqual(len(self.PROMPTS), len(outputs))


if __name__ == "__main__":
    unittest.main()
