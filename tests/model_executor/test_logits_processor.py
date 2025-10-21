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
        self.batch = None
        self.share_inputs = self.create_share_inputs()

    def create_request(self, **kwargs):
        """Create a mock request with specified logit bias"""
        request = Mock(spec=Request)
        for k, v in kwargs.items():
            setattr(request, k, v)
        return request

    def create_share_inputs(self):
        """Create a share_inputs dict to mock inference context"""
        share_inputs = {}
        share_inputs["logit_bias"] = [None] * self.max_num_seqs
        return share_inputs

    def create_logits(self):
        return paddle.randn([len(self.batch) if self.batch is not None else 0, self.vocab_size])

    def add_request(self, request):
        # print(f"Adding new request: {request.__dict__}")
        if self.batch is None:
            self.batch = [request]
        else:
            self.batch.append(request)
        for req in self.batch:
            self.share_inputs["logit_bias"][req.idx] = req.logit_bias
        return

    def del_request(self, request_id):
        # print(f"Deleting request with id: {request_id}")
        del_idx = None
        for i, req in enumerate(self.batch.copy()):
            if req.request_id == request_id:
                self.batch.pop(i)
                del_idx = req.idx
                break
        self.share_inputs["logit_bias"][del_idx] = None

    def test_logit_bias_logit_processor(self):

        logits_processor = LogitBiasLogitsProcessor()

        print("Phase 1: Empty batch")
        logits = self.create_logits()
        logits_processor.update_state(self.batch, self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        self.assertTrue(
            paddle.allclose(processed_logits, logits, atol=1e-6), "Logits should remain unchanged with empty batch"
        )

        print("Phase 2: Add first request")
        request1 = self.create_request(
            request_id="req1", idx=0, logit_bias={random.randint(0, self.vocab_size - 1): random.random() - 0.5}
        )
        self.add_request(request1)
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.batch, self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        for i, req in enumerate(self.batch):
            if req.logit_bias is not None:
                for token_id, bias in req.logit_bias.items():
                    expected_logits[i, token_id] += bias
        self.assertTrue(
            paddle.allclose(processed_logits, expected_logits, atol=1e-6),
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
        logits_processor.update_state(self.batch, self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        for i, req in enumerate(self.batch):
            if req.logit_bias is not None:
                for token_id, bias in req.logit_bias.items():
                    expected_logits[i, token_id] += bias
        self.assertTrue(
            paddle.allclose(processed_logits, expected_logits, atol=1e-6),
            "Logits should be modified with req1 and req2 biases\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("Phase 4: Remove first request")
        self.del_request("req1")
        logits = self.create_logits()
        original_logits = logits.clone()
        expected_logits = logits.clone()
        logits_processor.update_state(self.batch, self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        for i, req in enumerate(self.batch):
            if req.logit_bias is not None:
                for token_id, bias in req.logit_bias.items():
                    expected_logits[i, token_id] += bias
        self.assertTrue(
            paddle.allclose(processed_logits, expected_logits, atol=1e-6),
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
        logits_processor.update_state(self.batch, self.share_inputs)
        processed_logits = logits_processor.apply(logits)
        for i, req in enumerate(self.batch):
            if req.logit_bias is not None:
                for token_id, bias in req.logit_bias.items():
                    expected_logits[i, token_id] += bias
        processed_logits = logits_processor.apply(logits)
        self.assertTrue(
            paddle.allclose(processed_logits, logits, atol=1e-6),
            "Logits should remain unchanged with request having no bias\n"
            f"original: {original_logits}\n"
            f"processed: {processed_logits}\n"
            f"expected: {expected_logits}\n"
            f"diff: {processed_logits-expected_logits}",
        )

        print("All test phases completed successfully!")


if __name__ == "__main__":
    unittest.main()
