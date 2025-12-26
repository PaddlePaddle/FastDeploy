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

import random
import unittest
from unittest.mock import Mock

import paddle

from fastdeploy.engine.request import Request
from fastdeploy.model_executor.logits_processor.builtin import LogitBiasLogitsProcessor


class TestLogitsProcessor(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 10
        self.max_num_seqs = 16
        self.dtype = "float32"
        self.share_inputs = {
            "stop_flags": paddle.tensor([True for _ in range(self.max_num_seqs)]),
            "logits_processors_args": [{} for _ in range(self.max_num_seqs)],
        }

    def create_request(self, **kwargs):
        """Create a mock request with specified logit bias"""
        request = Mock(spec=Request)
        for k, v in kwargs.items():
            setattr(request, k, v)
        return request

    def create_logits(self):
        return paddle.randn([self.get_batch_size(), self.vocab_size], dtype=self.dtype)

    def add_request(self, req):
        self.share_inputs["stop_flags"][req.idx] = False
        self.share_inputs["logits_processors_args"][req.idx]["logit_bias"] = req.logit_bias

    def del_request(self, req):
        self.share_inputs["stop_flags"][req.idx] = True
        self.share_inputs["logits_processors_args"][req.idx] = {}

    def get_batch_size(self):
        return self.max_num_seqs - sum(self.share_inputs["stop_flags"])

    def test_logit_bias_logit_processor(self):

        fd_config = Mock()
        fd_config.model_config.dtype = self.dtype
        logits_processor = LogitBiasLogitsProcessor(fd_config)

        print("Phase 1: Empty batch")
        logits = self.create_logits()
        logits_processor.update_state(self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        self.assertTrue(paddle.all(processed_logits == logits), "Logits should remain unchanged with empty batch")

        print("Phase 2: Add first request")
        request1 = self.create_request(
            request_id="req1", idx=0, logit_bias={random.randint(0, self.vocab_size - 1): random.random() - 0.5}
        )
        self.add_request(request1)
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        batch_id = 0
        for slot_id, flag in enumerate(self.share_inputs["stop_flags"]):
            if not flag:
                logit_bias = self.share_inputs["logits_processors_args"][slot_id].get("logit_bias", {})
                for token_id, bias in logit_bias.items():
                    expected_logits[batch_id, token_id] += bias
                batch_id += 1
        self.assertTrue(
            paddle.all(processed_logits == expected_logits),
            f"Logits should be modified with req1 biases\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("Phase 3: Add second request with multiple tokens to apply bias")
        request2 = self.create_request(
            request_id="req2",
            idx=1,
            logit_bias=dict(
                zip(random.choices(range(0, self.vocab_size), k=3), [random.random() - 0.5 for _ in range(3)])
            ),
        )
        self.add_request(request2)
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        batch_id = 0
        for slot_id, flag in enumerate(self.share_inputs["stop_flags"]):
            if not flag:
                logit_bias = self.share_inputs["logits_processors_args"][slot_id].get("logit_bias") or {}
                for token_id, bias in logit_bias.items():
                    expected_logits[batch_id, token_id] += bias
                batch_id += 1
        self.assertTrue(
            paddle.all(processed_logits == expected_logits),
            "Logits should be modified with req1 and req2 biases\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("Phase 4: Remove first request")
        self.del_request(request1)
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        batch_id = 0
        for slot_id, flag in enumerate(self.share_inputs["stop_flags"]):
            if not flag:
                logit_bias = self.share_inputs["logits_processors_args"][slot_id].get("logit_bias") or {}
                for token_id, bias in logit_bias.items():
                    expected_logits[batch_id, token_id] += bias
                batch_id += 1
        self.assertTrue(
            paddle.all(processed_logits == expected_logits),
            "Logits should only have biases from request2 after removal\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("Phase 5: Add third request with no logit bias")
        request3 = self.create_request(request_id="req3", idx=0, logit_bias=None)
        self.add_request(request3)
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        batch_id = 0
        for slot_id, flag in enumerate(self.share_inputs["stop_flags"]):
            if not flag:
                logit_bias = self.share_inputs["logits_processors_args"][slot_id].get("logit_bias") or {}
                for token_id, bias in logit_bias.items():
                    expected_logits[batch_id, token_id] += bias
                batch_id += 1
        self.assertTrue(
            paddle.all(processed_logits == expected_logits),
            "Logits should remain unchanged with request having no bias\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("All test phases completed successfully!")


if __name__ == "__main__":
    unittest.main()
