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
from concurrent.futures import Future
from types import SimpleNamespace

import paddle
import paddle.nn.functional as F
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace(enable_torch_proxy=lambda **_: None)

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import (
    GuidedDecoding,
    Sampler,
    padding_sampling_params,
    top_p_normalize_probs_paddle,
)
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
        prompt_lens=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        step_idx=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        token_ids_all=_create_tokens_tensor(batch_size, max_seq_len),
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0),
        min_dec_lens=paddle.full(shape=[batch_size, 1], fill_value=min_seq_len, dtype="int64"),
        bad_words_token_ids=paddle.full(shape=[batch_size], fill_value=-1, dtype="int64"),
        bad_words_token_len=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
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
            sampling_metadata.token_ids_all,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.bad_words_token_len,
            sampling_metadata.prompt_lens,
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
            sampling_metadata.token_ids_all,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.bad_words_token_len,
            sampling_metadata.prompt_lens,
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


class _DummyProcessor:
    def __init__(self, terminated=False, enable_reasoning=True, accept_result=True):
        self.is_terminated = terminated
        self.enable_reasoning = enable_reasoning
        self.reasoning_ended = False
        self.accept_result = accept_result
        self.filled = []
        self.accepted = []

    def allocate_token_bitmask(self):
        return paddle.zeros([4, 8], dtype="int32")

    def fill_token_bitmask(self, token_bitmask, idx):
        token_bitmask[idx, idx] = 1
        self.filled.append(idx)

    def accept_token(self, token):
        self.accepted.append(token)
        return self.accept_result


def test_top_p_normalize_probs_and_padding_params():
    probs = paddle.to_tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype="float32")
    top_ps = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
    normalized = top_p_normalize_probs_paddle(probs, top_ps)
    assert paddle.allclose(normalized[0], paddle.to_tensor([0.5714286, 0.4285714, 0.0, 0.0]), atol=1e-5)
    assert paddle.allclose(normalized[1], probs[1])

    top_p = paddle.to_tensor([0.9, 0.8], dtype="float32")
    top_k = paddle.to_tensor([10, 20], dtype="int64")
    infer_seed = paddle.to_tensor([100, 200], dtype="int64")
    seq_lens_this_time = paddle.to_tensor([3, 2], dtype="int64")
    seq_lens_encoder = paddle.to_tensor([0, 1], dtype="int64")
    with pytest.raises(RuntimeError, match="gather"):
        padding_sampling_params(top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder)


def test_guided_decoding_update_apply_and_accept_paths(monkeypatch):
    gd = GuidedDecoding(SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3)))
    p0 = _DummyProcessor(terminated=True)
    p1 = _DummyProcessor(enable_reasoning=False)
    p2 = _DummyProcessor(accept_result=True)
    gd.logits_processors = [p0, p1, p2]
    gd._prefill_done_idxs = [False, False, True]

    done_future = Future()
    done_future.set_result(p1)
    gd.logits_processors[1] = done_future

    gd.update_vocab_mask(prefill_done_idxs=[0, 1])
    assert gd.logits_processors[0] is None
    assert gd._prefill_done_idxs[0] is False
    assert gd._prefill_done_idxs[1] is True

    gd._tokens_to_acc[2] = [7]
    gd.accept_tokens_from_prefill_node(2)
    assert gd._tokens_to_acc[2] is None
    assert gd.logits_processors[2].accepted[-1] == 7

    def _fake_apply_token_mask(logits, token_bitmask, indices, is_cuda_platform):
        assert token_bitmask is not None
        assert indices == [1, 2]
        return logits + 1.0

    monkeypatch.setattr(gd, "join_async_fillmask", lambda: None)
    import sys
    import types

    xgrammar_backend = types.SimpleNamespace(apply_token_mask=_fake_apply_token_mask)
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.guided_decoding.xgrammar_backend", xgrammar_backend)
    out = gd.apply_token_mask(paddle.zeros([3, 8], dtype="float32"))
    assert float(out.sum()) == pytest.approx(24.0)

    gd.reasoning_parser = SimpleNamespace(is_reasoning_end=lambda tokens: tokens[0] == 3)
    gd.logits_processors[1].reasoning_ended = False
    gd._accept_token(1, 3)
    assert gd.logits_processors[1].reasoning_ended is True

    gd._prefill_done_idxs = [False, True, True]
    gd.update_output_tokens(paddle.to_tensor([[0], [2], [-1]], dtype="int64"))
    assert gd.logits_processors[1] is not None
    assert gd.logits_processors[2] is None


test_sampler = pytest.mark.skip(reason="Requires full CUDA sampler runtime in CI image")(test_sampler)
test_sampler_logprobs = pytest.mark.skip(reason="Requires full CUDA sampler runtime in CI image")(
    test_sampler_logprobs
)


if __name__ == "__main__":
    test_sampler()
    test_sampler_logprobs()
