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

import sys
import types
from concurrent.futures import Future
from unittest.mock import Mock

import paddle
import paddle.nn.functional as F
import pytest

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import (
    GuidedDecoding,
    Sampler,
    padding_sampling_params,
    top_p_normalize_probs_paddle,
)
from fastdeploy.platforms import current_platform
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
        bad_words_token_len=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size], fill_value=-2, dtype="int64"),
        min_p=paddle.randn([batch_size]),
        seed=paddle.to_tensor([[2025]]),
        logits_processors=None,
    )
    if max_num_logprobs is not None:
        fake_sampling_metadata.max_num_logprobs = max_num_logprobs
    return fake_sampling_metadata


def get_fd_config(batch_size: int, logprobs_mode: str = "raw_logprobs"):
    model_config: Mock = Mock()
    model_config.logprobs_mode = logprobs_mode
    model_config.max_model_len = 2048
    fd_config = FDConfig(
        model_config=model_config,
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
        ips="0.0.0.0",
    )
    return fd_config


def _patch_sampler_ops(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_iluvatar", lambda: False)
    monkeypatch.setattr(current_platform, "is_gcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_dcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_maca", lambda: False)
    monkeypatch.setattr(current_platform, "is_intel_hpu", lambda: False)

    def _noop_apply_penalty(*args, **kwargs):
        return args[3]

    def _noop_min_p_sampling(probs, min_p_arr, min_p_arr_cpu):
        return probs

    def _safe_top_k_top_p_sampling(
        x,
        top_p,
        top_k=None,
        top_k_list=None,
        threshold=None,
        topp_seed=None,
        seed=-1,
        k=0,
        mode="truncated",
        order="top_k_first",
    ):
        ids = paddle.argmax(x, axis=-1).unsqueeze(-1)
        return None, ids

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_penalty_multi_scores",
        _noop_apply_penalty,
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.ops.apply_penalty_multi_scores",
        _noop_apply_penalty,
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.min_p_sampling",
        _noop_min_p_sampling,
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling",
        _safe_top_k_top_p_sampling,
    )


def test_sampler(monkeypatch):
    _patch_sampler_ops(monkeypatch)
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
            sampling_metadata.bad_words_token_len,
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
            sampling_metadata.bad_words_token_len,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )
        logprobs = logits
    token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)
    return token_logprobs


def test_sampler_logprobs(monkeypatch):
    _patch_sampler_ops(monkeypatch)
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    logprobs_mode_list = ["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"]
    logits = _create_fake_logits(batch_size, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len, max_num_logprobs=0)
    for logprobs_mode in logprobs_mode_list:
        fd_config = get_fd_config(batch_size, logprobs_mode)
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
    def __init__(self, enable_reasoning=True, reasoning_ended=False, accept_result=True, terminated=False):
        self.enable_reasoning = enable_reasoning
        self.reasoning_ended = reasoning_ended
        self.accept_result = accept_result
        self.is_terminated = terminated
        self.accepted_tokens = []
        self.fill_calls = 0

    def allocate_token_bitmask(self):
        return paddle.zeros([2], dtype="int32")

    def fill_token_bitmask(self, token_bitmask, idx):
        self.fill_calls += 1

    def accept_token(self, token):
        self.accepted_tokens.append(token)
        return self.accept_result


class _DummyReasoningParser:
    def __init__(self, end_token):
        self.end_token = end_token

    def is_reasoning_end(self, tokens):
        return tokens[0] == self.end_token


def test_guided_decoding_bitmask_and_accept(monkeypatch):
    _patch_sampler_ops(monkeypatch)
    guided_decoding = GuidedDecoding(get_fd_config(batch_size=2))
    processor = _DummyProcessor(enable_reasoning=True, reasoning_ended=False)
    future = Future()
    future.set_result(processor)
    guided_decoding.add_logits_processor(0, future, prefill_tokens=[])

    fake_backend = types.SimpleNamespace(
        apply_token_mask=lambda logits, token_bitmask, indices, is_cuda_platform: logits + 1.0
    )
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.guided_decoding.xgrammar_backend", fake_backend)

    logits = paddle.zeros([2, 4], dtype="float32")
    guided_decoding.update_vocab_mask(prefill_done_idxs=[0])
    guided_decoding.join_async_fillmask()
    assert processor.fill_calls == 1
    masked_logits = guided_decoding.apply_token_mask(logits, prefill_done_idxs=[0])
    assert paddle.allclose(masked_logits, logits + 1.0)


def test_guided_decoding_reasoning_and_reset(monkeypatch):
    _patch_sampler_ops(monkeypatch)
    guided_decoding = GuidedDecoding(get_fd_config(batch_size=1))
    processor = _DummyProcessor(enable_reasoning=False, reasoning_ended=False)
    guided_decoding.logits_processors[0] = processor
    guided_decoding._prefill_done_idxs[0] = True
    guided_decoding.apply_reasoning_parser(_DummyReasoningParser(end_token=7))

    guided_decoding.update_output_tokens(paddle.to_tensor([[7]], dtype="int64"))
    assert processor.reasoning_ended is True

    guided_decoding.update_output_tokens(paddle.to_tensor([[-1]], dtype="int64"))
    assert guided_decoding.logits_processors[0] is None


def test_top_p_normalize_and_padding():
    probs = paddle.to_tensor([[0.4, 0.3, 0.2, 0.1]], dtype="float32")
    top_p = paddle.to_tensor([[0.5]], dtype="float32")
    normalized = top_p_normalize_probs_paddle(probs, top_p)
    assert paddle.sum(normalized).item() == pytest.approx(1.0)
    assert normalized[0, 2].item() == pytest.approx(0.0)

    top_p_padding, top_k_padding = padding_sampling_params(
        top_p=paddle.to_tensor([[0.5], [0.9]], dtype="float32"),
        top_k=paddle.to_tensor([[1], [2]], dtype="int32"),
        seq_lens_this_time=paddle.to_tensor([[2], [1]], dtype="int32"),
        seq_lens_encoder=paddle.to_tensor([[0], [1]], dtype="int32"),
    )
    assert top_p_padding.shape[0] == 3
    assert top_k_padding[-1].item() == 2


def test_sampler_compute_logprobs_with_top_p_normalization(monkeypatch):
    _patch_sampler_ops(monkeypatch)
    fd_config = get_fd_config(batch_size=2)
    sampler = Sampler(fd_config)
    logits = paddle.to_tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype="float32")
    sampling_metadata = _create_default_sampling_metadata(batch_size=2, min_seq_len=1, max_seq_len=4)
    sampling_metadata.temperature = paddle.to_tensor([[2.0], [1.0]], dtype="float32")
    sampling_metadata.temp_scaled_logprobs_flag = True
    sampling_metadata.temp_scaled_logprobs = paddle.to_tensor([[True], [False]])
    sampling_metadata.top_p_normalized_logprobs_flag = True
    sampling_metadata.top_p_normalized_logprobs = paddle.to_tensor([[True], [True]])
    sampling_metadata.top_p = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
    sampling_metadata.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1], [1]], dtype="int32"),
        "seq_lens_encoder": paddle.to_tensor([[0], [0]], dtype="int32"),
        "seq_lens_decoder": paddle.to_tensor([[0], [0]], dtype="int32"),
    }

    logprobs = sampler.compute_logprobs(logits, sampling_metadata)
    scaled_logits = logits.clone()
    scaled_logits[0] = scaled_logits[0] / 2.0
    probs = F.softmax(scaled_logits, axis=-1)
    expected_top_p = top_p_normalize_probs_paddle(probs[:1], sampling_metadata.top_p[:1])
    expected_logprobs = paddle.log(expected_top_p)
    assert paddle.allclose(logprobs[:1], expected_logprobs, atol=1e-6)
    expected_row1 = F.log_softmax(logits[1:2], axis=-1)
    assert paddle.allclose(logprobs[1:2], expected_row1, atol=1e-6)


def test_sampler_gather_logprobs(monkeypatch):
    _patch_sampler_ops(monkeypatch)
    sampler = Sampler(get_fd_config(batch_size=2))
    logprobs = paddle.to_tensor([[0.0, -1.0, -2.0], [-0.1, -0.2, -0.3]], dtype="float32")
    token_ids = paddle.to_tensor([0, 2], dtype="int64")
    topk = sampler.gather_logprobs(logprobs, num_logprobs=2, token_ids=token_ids)
    assert tuple(topk.logprob_token_ids.shape) == (2, 3)
    assert topk.logprob_token_ids[:, 0].tolist() == token_ids.tolist()

    none_topk = sampler.gather_logprobs(logprobs, num_logprobs=0, token_ids=token_ids)
    assert tuple(none_topk.logprob_token_ids.shape) == (2, 1)
    assert none_topk.logprob_token_ids[:, 0].tolist() == token_ids.tolist()


if __name__ == "__main__":
    test_sampler()
    test_sampler_logprobs()
