"""
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
"""

import json
import os

import paddle
import paddle.nn.functional as F

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.scheduler import SchedulerConfig


def _create_fake_logits(batch_size: int, vocab_size: int) -> paddle.Tensor:
    fake_logits = paddle.rand(shape=[batch_size, vocab_size], dtype="float32")
    return fake_logits


def _create_penalty_tensor(batch_size: int, penalty_value: float) -> paddle.Tensor:
    return paddle.full(shape=[batch_size, 1], fill_value=penalty_value, dtype="float32")


def _create_tokens_tensor(
    batch_size: int,
    max_seq_len: int,
) -> paddle.Tensor:
    pre_token_ids = paddle.full(shape=[batch_size, max_seq_len], fill_value=-1, dtype="int64")
    return pre_token_ids


def _create_default_sampling_metadata(
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    max_num_logprobs: int = None,
) -> SamplingMetadata:

    fake_sampling_metadata = SamplingMetadata(
        temperature=paddle.full(shape=[batch_size, 1], fill_value=0.9, dtype="float32"),
        top_p=paddle.full(shape=[batch_size, 1], fill_value=0.7, dtype="float32"),
        prompt_ids=paddle.full(shape=[batch_size, max_seq_len], fill_value=0, dtype="int64"),
        prompt_lens=paddle.full(shape=[batch_size, 1], fill_value=5, dtype="int64"),
        step_idx=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        pre_token_ids=_create_tokens_tensor(batch_size, max_seq_len),
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0),
        min_dec_lens=paddle.full(shape=[batch_size, 1], fill_value=min_seq_len, dtype="int64"),
        bad_words_token_ids=paddle.full(shape=[batch_size], fill_value=-1, dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size], fill_value=-2, dtype="int64"),
        min_p=paddle.randn([batch_size]),
        seed=paddle.to_tensor([[2025]]),
        logits_processors=None,
    )
    if max_num_logprobs is not None:
        fake_sampling_metadata.max_num_logprobs = max_num_logprobs
    return fake_sampling_metadata


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


def test_sampler():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024

    sampler = Sampler(get_fd_config(batch_size))
    logits = _create_fake_logits(batch_size, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len)
    next_tokens = sampler(logits, sampling_metadata)
    print(next_tokens)


def get_baseline_logprobs(logits, sampling_metadata, logprobs_mode, token_ids):
    if logprobs_mode == "raw_logprobs":
        logprobs = F.log_softmax(logits, axis=-1)
    elif logprobs_mode == "raw_logits":
        logprobs = logits.clone()
    elif logprobs_mode == "processed_logprobs":
        from fastdeploy.model_executor.layers.sample.ops import (
            apply_penalty_multi_scores,
        )

        for proc in sampling_metadata.logits_processors or []:
            logits = proc.apply(logits)

        logits = apply_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            sampling_metadata.prompt_ids,
            sampling_metadata.prompt_lens,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )
        logprobs = F.log_softmax(logits, axis=-1)
    else:
        from fastdeploy.model_executor.layers.sample.ops import (
            apply_penalty_multi_scores,
        )

        for proc in sampling_metadata.logits_processors or []:
            logits = proc.apply(logits)

        logits = apply_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            sampling_metadata.prompt_ids,
            sampling_metadata.prompt_lens,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )
        logprobs = logits
    token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)
    return token_logprobs


def test_sampler_logprobs():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    logprobs_mode_list = ["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"]
    logits = _create_fake_logits(batch_size, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len, max_num_logprobs=0)
    for logprobs_mode in logprobs_mode_list:
        fd_config = get_fd_config(batch_size)
        fd_config.model_config.logprobs_mode = logprobs_mode
        sampler = Sampler(logprobs_mode=logprobs_mode, fd_config=fd_config)
        assert sampler.logprobs_mode == logprobs_mode
        sampler_output = sampler(logits.clone(), sampling_metadata)
        baseline_logprobs = get_baseline_logprobs(
            logits.clone(), sampling_metadata, logprobs_mode=logprobs_mode, token_ids=sampler_output.sampled_token_ids
        )
        logprobs = sampler_output.logprobs_tensors.logprobs
        print(f"baseline_logprobs = {baseline_logprobs}")
        print(f"logprobs = {logprobs}")
        equal = paddle.allclose(baseline_logprobs, logprobs, atol=1e-03, rtol=1e-03).item()
        print(f"logprobs_mode: {logprobs_mode} equal={equal}")
        assert equal


if __name__ == "__main__":
    test_sampler()
    test_sampler_logprobs()
