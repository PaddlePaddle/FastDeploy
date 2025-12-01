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
from unittest.mock import MagicMock

import numpy as np
import paddle

from fastdeploy.engine.request import RequestOutput
from fastdeploy.output.token_processor import TokenProcessor


class TestProcessBatchDraftTokens(unittest.TestCase):

    def setUp(self):
        # 模拟 cfg
        cfg = MagicMock()
        cfg.speculative_config = MagicMock()
        cfg.speculative_config.method = "mtp"
        cfg.speculative_config.num_speculative_tokens = 3
        cfg.model_config = MagicMock()
        cfg.model_config.enable_logprob = True

        self.processor = TokenProcessor(
            cfg=cfg, cached_generated_tokens=MagicMock(), engine_worker_queue=MagicMock(), split_connector=MagicMock()
        )

        # mock resource_manager
        self.processor.resource_manager = MagicMock()
        self.processor.resource_manager.stop_flags = [False] * 512
        self.processor.resource_manager.tasks_list = [MagicMock()] * 512

        for task in self.processor.resource_manager.tasks_list:
            task.request_id = "test_request"
            task.eos_token_ids = [2]

    def test_process_batch_draft_tokens_normal_case(self):
        """测试正常情况下的target处理"""
        batch = 2
        accept_num = [3, 2]
        K = 20
        MAX_DRAFT_TOKENS = 6

        tokens = np.random.randint(100, 200, size=(batch, MAX_DRAFT_TOKENS, K + 1))
        scores = np.random.rand(batch, MAX_DRAFT_TOKENS, K + 1).astype(np.float32)
        ranks = np.random.randint(0, K, size=(batch, MAX_DRAFT_TOKENS))

        results = self.processor._process_batch_draft_tokens(
            mtype=4,
            batch=batch,
            accept_num=accept_num,
            tokens=paddle.to_tensor(tokens),
            scores=paddle.to_tensor(scores),
            ranks=paddle.to_tensor(ranks),
        )

        self.assertEqual(len(results), batch)
        for i, result in enumerate(results):
            self.assertIsInstance(result, RequestOutput)
            self.assertEqual(result.output_type, 4)
            self.assertEqual(result.outputs.index, i)
            self.assertEqual(len(result.outputs.draft_top_logprobs.logprob_token_ids), accept_num[i])
            self.assertEqual(len(result.outputs.draft_top_logprobs.logprobs), accept_num[i])
            self.assertEqual(len(result.outputs.draft_top_logprobs.sampled_token_ranks), accept_num[i])

    def test_process_batch_draft_tokens_with_stop_flag(self):
        """测试有停止标志的情况"""
        batch = 3
        self.processor.resource_manager.stop_flags[1] = True  # 第二个 request 停止

        accept_num = [3, 2, 1]
        K = 20
        MAX_DRAFT_TOKENS = 6

        tokens = np.random.randint(100, 200, size=(batch, MAX_DRAFT_TOKENS, K + 1))
        scores = np.random.rand(batch, MAX_DRAFT_TOKENS, K + 1).astype(np.float32)
        ranks = np.random.randint(0, K, size=(batch, MAX_DRAFT_TOKENS))

        results = self.processor._process_batch_draft_tokens(
            mtype=4,
            batch=batch,
            accept_num=accept_num,
            tokens=paddle.to_tensor(tokens),
            scores=paddle.to_tensor(scores),
            ranks=paddle.to_tensor(ranks),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].outputs.index, 0)
        self.assertEqual(results[1].outputs.index, 2)

    def test_process_batch_draft_tokens_empty_accept(self):
        """测试 accept_num 为 0 的情况"""
        batch = 2
        accept_num = [0, 0]

        K = 20
        MAX_DRAFT_TOKENS = 6
        tokens = np.random.randint(100, 200, size=(batch, MAX_DRAFT_TOKENS, K + 1))
        scores = np.random.rand(batch, MAX_DRAFT_TOKENS, K + 1).astype(np.float32)
        ranks = np.random.randint(0, K, size=(batch, MAX_DRAFT_TOKENS))

        results = self.processor._process_batch_draft_tokens(
            mtype=4,
            batch=batch,
            accept_num=accept_num,
            tokens=paddle.to_tensor(tokens),
            scores=paddle.to_tensor(scores),
            ranks=paddle.to_tensor(ranks),
        )

        self.assertEqual(len(results), batch)
        for result in results:
            self.assertIsNone(result.outputs.draft_top_logprobs)

    def test_process_batch_draft_tokens_different_k_values(self):
        """测试不同 K 值情况"""
        batch = 2
        accept_num = [3, 2]

        K = 5
        MAX_DRAFT_TOKENS = 6
        tokens = np.random.randint(100, 200, size=(batch, MAX_DRAFT_TOKENS, K + 1))
        scores = np.random.rand(batch, MAX_DRAFT_TOKENS, K + 1).astype(np.float32)
        ranks = np.random.randint(0, K, size=(batch, MAX_DRAFT_TOKENS))

        results = self.processor._process_batch_draft_tokens(
            mtype=4,
            batch=batch,
            accept_num=accept_num,
            tokens=paddle.to_tensor(tokens),
            scores=paddle.to_tensor(scores),
            ranks=paddle.to_tensor(ranks),
        )

        self.assertEqual(len(results), batch)
        for i, result in enumerate(results):
            self.assertEqual(len(result.outputs.draft_top_logprobs.logprob_token_ids[0]), K + 1)
            self.assertEqual(len(result.outputs.draft_top_logprobs.logprobs[0]), K + 1)


if __name__ == "__main__":
    unittest.main()
