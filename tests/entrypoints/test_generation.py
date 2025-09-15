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

import copy
import os
import unittest
import weakref

from fastdeploy.engine.request import RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

MODEL_NAME = os.getenv("MODEL_PATH") + "/ERNIE-4.5-0.3B-Paddle"


class TestGeneration(unittest.TestCase):
    """Test case for generation functionality"""

    TOKEN_IDS = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 2, 4, 6],
    ]

    PROMPTS = [
        "Hello, my name is",
        "The capital of China is",
        "The future of AI is",
        "人工智能是",
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

    def assert_outputs_equal(self, o1: list[RequestOutput], o2: list[RequestOutput]):
        self.assertEqual([o.outputs for o in o1], [o.outputs for o in o2])

    def test_consistency_single_prompt_tokens(self):
        """Test consistency between different prompt input formats"""
        sampling_params = SamplingParams(temperature=1.0, top_p=0.0)

        for prompt_token_ids in self.TOKEN_IDS:
            with self.subTest(prompt_token_ids=prompt_token_ids):
                output1 = self.llm.generate(prompts=prompt_token_ids, sampling_params=sampling_params)
                output2 = self.llm.generate(
                    {"prompt": "", "prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params
                )
                self.assert_outputs_equal(output1, output2)

    def test_api_consistency_multi_prompt_tokens(self):
        """Test consistency with multiple prompt tokens"""
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.0,
        )

        output1 = self.llm.generate(prompts=self.TOKEN_IDS, sampling_params=sampling_params)

        output2 = self.llm.generate(
            [{"prompt": "", "prompt_token_ids": p} for p in self.TOKEN_IDS],
            sampling_params=sampling_params,
        )

        self.assert_outputs_equal(output1, output2)

    def test_multiple_sampling_params(self):
        """Test multiple sampling parameters combinations"""
        sampling_params = [
            SamplingParams(temperature=0.01, top_p=0.95),
            SamplingParams(temperature=0.3, top_p=0.95),
            SamplingParams(temperature=0.7, top_p=0.95),
            SamplingParams(temperature=0.99, top_p=0.95),
        ]

        # Multiple SamplingParams should be matched with each prompt
        outputs = self.llm.generate(prompts=self.PROMPTS, sampling_params=sampling_params)
        self.assertEqual(len(self.PROMPTS), len(outputs))

        # Exception raised if size mismatch
        with self.assertRaises(ValueError):
            self.llm.generate(prompts=self.PROMPTS, sampling_params=sampling_params[:3])

        # Single SamplingParams should be applied to every prompt
        single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
        outputs = self.llm.generate(prompts=self.PROMPTS, sampling_params=single_sampling_params)
        self.assertEqual(len(self.PROMPTS), len(outputs))

        # sampling_params is None, default params should be applied
        outputs = self.llm.generate(prompts=self.PROMPTS, sampling_params=None)
        self.assertEqual(len(self.PROMPTS), len(outputs))

    def test_consistency_single_prompt_tokens_chat(self):
        """Test consistency between different prompt input formats"""
        sampling_params = SamplingParams(temperature=1.0, top_p=0.0)

        for prompt_token_ids in self.TOKEN_IDS:
            with self.subTest(prompt_token_ids=prompt_token_ids):
                output1 = self.llm.chat(messages=[prompt_token_ids], sampling_params=sampling_params)
                output2 = self.llm.chat(
                    [{"prompt": "", "prompt_token_ids": prompt_token_ids}], sampling_params=sampling_params
                )
                self.assert_outputs_equal(output1, output2)

    def test_multiple_sampling_params_chat(self):
        """Test multiple sampling parameters combinations"""
        sampling_params = [
            SamplingParams(temperature=0.01, top_p=0.95),
            SamplingParams(temperature=0.3, top_p=0.95),
            SamplingParams(temperature=0.7, top_p=0.95),
            SamplingParams(temperature=0.99, top_p=0.95),
        ]

        prompts = copy.copy(self.PROMPTS)
        # Multiple SamplingParams should be matched with each prompt
        outputs = self.llm.chat(messages=prompts, sampling_params=sampling_params)
        self.assertEqual(len(self.PROMPTS), len(outputs))

        prompts = copy.copy(self.PROMPTS)
        # Exception raised if size mismatch
        with self.assertRaises(ValueError):
            self.llm.chat(messages=prompts, sampling_params=sampling_params[:3])

        prompts = copy.copy(self.PROMPTS)
        # Single SamplingParams should be applied to every prompt
        single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
        outputs = self.llm.chat(messages=prompts, sampling_params=single_sampling_params)
        self.assertEqual(len(self.PROMPTS), len(outputs))

        prompts = copy.copy(self.PROMPTS)
        # sampling_params is None, default params should be applied
        outputs = self.llm.chat(messages=prompts, sampling_params=None)
        self.assertEqual(len(self.PROMPTS), len(outputs))


if __name__ == "__main__":
    unittest.main()
