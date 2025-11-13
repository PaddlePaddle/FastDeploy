# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import time
import unittest

import numpy as np
import paddle

from fastdeploy.engine.request import Request
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


# Mock classes and constants needed for the test
class MockConfig:

    class ModelConfig:
        enable_logprob = False
        max_logprobs = -1
        logprobs_mode = "raw_logprobs"

    class SchedulerConfig:
        max_num_seqs = 6

    class CacheConfig:
        enable_prefix_caching = False

    speculative_config = None
    model_config = ModelConfig()
    scheduler_config = SchedulerConfig()
    cache_config = CacheConfig()


class MockTask:
    def __init__(self):
        paddle.seed(0)
        self.request_id = "test_request_1"
        self.arrival_time = time.time()
        self.inference_start_time = time.time()
        self.schedule_start_time = time.time()
        self.preprocess_end_time = time.time() - 0.1
        self.preprocess_start_time = time.time() - 0.2
        self.eos_token_ids = [2]
        self.output_token_ids = []
        self.messages = "Test prompt"
        self.num_cached_tokens = 0
        self.disaggregate_info = None
        self.prefill_chunk_info = None
        self.prefill_chunk_num = 0
        self.pooling_params = None
        self.llm_engine_recv_req_timestamp = time.time()

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, "sampling_params") and hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        else:
            return default_value


class FakeModel:
    def __init__(self, vocab_size=128, hidden_size=128):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = paddle.rand([hidden_size, vocab_size], dtype="float32")

    def compute_logits(self, x):
        return paddle.matmul(x.astype("float32"), self.weight)


class TestGPUPromptLogprobs(unittest.TestCase):
    def setup_model_runner(self):
        """Helper method to setup GPUModelRunner with different configurations"""
        cfg = MockConfig()
        cfg.model_config.ori_vocab_size = 128
        cfg.model_config.vocab_size = 128
        cfg.model_config.hidden_size = 64

        model_runner = GPUModelRunner.__new__(GPUModelRunner)
        model_runner.fd_config = cfg
        model_runner.scheduler_config = cfg.scheduler_config
        model_runner.ori_vocab_size = cfg.model_config.ori_vocab_size
        model_runner.share_inputs = {}
        model_runner.share_inputs["cu_seqlens_q"] = paddle.to_tensor([0, 1, 2, 3], dtype="int32")
        model_runner.sampler = Sampler()

        model_runner.model = FakeModel(cfg.model_config.vocab_size, cfg.model_config.hidden_size)

        model_runner.in_progress_prompt_logprobs = {}

        return model_runner

    def test_prompt_logprobs(self):
        model_runner = self.setup_model_runner()

        req: Request = Request(
            prompt=None,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=None,
            arrival_time=None,
            request_id="asd1",
            prompt_token_ids=[1, 2, 3, 4],
            prompt_token_ids_len=4,
            prefill_start_index=0,
            prefill_end_index=4,
            sampling_params=SamplingParams(prompt_logprobs=-1),
        )
        req.idx = 0
        model_runner.prompt_logprobs_reqs = {req.request_id: req}

        hidden_states = paddle.rand(
            [len(req.prompt_token_ids) - 1, model_runner.fd_config.model_config.hidden_size], dtype="bfloat16"
        )
        ref_logits = model_runner.model.compute_logits(hidden_states)
        ref_raw_logprobs = model_runner.sampler.compute_logprobs(ref_logits)
        token_is = paddle.to_tensor(req.prompt_token_ids[1:], dtype="int64")

        ref_token_ids, ref_logprobs, ref_ranks = model_runner.sampler.gather_logprobs(
            ref_raw_logprobs, model_runner.fd_config.model_config.ori_vocab_size, token_is
        )
        prompt_logprobs = model_runner._get_prompt_logprobs_list(hidden_states)[0]
        np.testing.assert_allclose(ref_logprobs.numpy(), prompt_logprobs.logprobs.numpy(), rtol=1e-04, atol=1e-04)
        np.testing.assert_allclose(
            ref_token_ids.numpy(), prompt_logprobs.logprob_token_ids.numpy(), rtol=1e-04, atol=1e-04
        )
        np.testing.assert_allclose(
            ref_ranks.numpy(), prompt_logprobs.selected_token_ranks.numpy(), rtol=1e-04, atol=1e-04
        )


if __name__ == "__main__":
    unittest.main()
