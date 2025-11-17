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

import json
import os
import time
import unittest

import numpy as np
import paddle

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.engine.request import Request
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.scheduler import SchedulerConfig
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


def build_config_json() -> str:
    config_dict = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 7168,
        "moe_intermediate_size": 1,
        "moe_num_experts": 1,
        "moe_k": 1,
        "hidden_act": "silu",
        "num_attention_heads": 64,
        "dtype": "bfloat16",
    }

    tmp_dir = f"./tmpefef{paddle.distributed.get_rank()}"
    os.makedirs(tmp_dir, exist_ok=True)
    with open(f"./{tmp_dir}/config.json", "w") as f:
        json.dump(config_dict, f)
    model_name_or_path = os.path.join(os.getcwd(), tmp_dir)
    print("model_name_or_path", model_name_or_path)
    return model_name_or_path


def get_fd_config(batch_size: int):
    fd_config = FDConfig(
        model_config=ModelConfig(
            {
                "model": build_config_json(),
                "max_model_len": 2048,
            }
        ),
        parallel_config=ParallelConfig(
            {
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
                "expert_parallel_rank": 0,
                "data_parallel_size": 1,
            }
        ),
        # quant_config=BlockWiseFP8Config(weight_block_size=[128, 128]),
        scheduler_config=SchedulerConfig({"max_num_seqs": batch_size}),
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        ips="0.0.0.0",
    )
    return fd_config


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
        model_runner.sampler = Sampler(get_fd_config(batch_size=1))

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
