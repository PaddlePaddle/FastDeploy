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
import sys
import types
from concurrent.futures import Future

import paddle
import paddle.nn.functional as F
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda *args, **kwargs: None)

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
    MTPSampler,
    Sampler,
    SpeculativeSampler,
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
        min_p=paddle.randn([batch_size], dtype="float32"),
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


class FakeLogitsProcessor:
    def __init__(self, accept_result=True, terminated=False):
        self.accept_result = accept_result
        self.is_terminated = terminated
        self.enable_reasoning = False
        self.reasoning_ended = False
        self.accepted_tokens = []
        self.fill_calls = []

    def allocate_token_bitmask(self):
        return paddle.zeros([4], dtype="int32")

    def fill_token_bitmask(self, token_bitmask, idx):
        self.fill_calls.append((idx, token_bitmask.shape[0]))

    def accept_token(self, token):
        self.accepted_tokens.append(token)
        return self.accept_result


class FakeFuture(Future):
    def __init__(self, result_value, done_value=False):
        super().__init__()
        self._result_value = result_value
        self._done_value = done_value

    def done(self):
        return self._done_value

    def result(self, timeout=None):
        return self._result_value


class FakeReasoningParser:
    def __init__(self, should_end=True):
        self.should_end = should_end

    def is_reasoning_end(self, tokens):
        return self.should_end


def _make_stubbed_sampler(logprobs_mode="processed_logprobs"):
    sampler = Sampler.__new__(Sampler)
    sampler.guided_decoding = types.SimpleNamespace(apply_token_mask=lambda logits, p_done_idxs: logits)
    sampler.logprobs_mode = logprobs_mode
    sampler.early_stopper = types.SimpleNamespace(process=lambda probs, next_tokens, stop_flags: None)
    return sampler


def _build_sampling_metadata_for_forward(batch_size, min_seq_len, max_seq_len):
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len, max_num_logprobs=2)
    sampling_metadata.top_k = paddle.full([batch_size, 1], 5, dtype="int64")
    sampling_metadata.top_k_list = [5 for _ in range(batch_size)]
    sampling_metadata.min_p = paddle.full([batch_size], 0.0, dtype="float32")
    sampling_metadata.min_p_list = [0.0 for _ in range(batch_size)]
    sampling_metadata.seed = paddle.full([batch_size, 1], 7, dtype="int64")
    sampling_metadata.enable_early_stop = True
    sampling_metadata.stop_flags = paddle.zeros([batch_size, 1], dtype="int32")
    return sampling_metadata


def _make_min_fd_config(max_num_seqs):
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_top_p_normalize_probs_paddle():
    probs = paddle.to_tensor([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]], dtype="float32")
    top_ps = paddle.to_tensor([[0.7], [0.5]], dtype="float32")
    normalized = top_p_normalize_probs_paddle(probs, top_ps)
    assert paddle.allclose(normalized.sum(axis=-1), paddle.ones([2], dtype="float32"))
    assert normalized[0, 2].item() == pytest.approx(0.0)
    assert normalized[1, 1].item() == pytest.approx(0.0)
    assert normalized[1, 2].item() == pytest.approx(0.0)


def test_padding_sampling_params_offsets(monkeypatch):
    top_p = paddle.to_tensor([[0.9], [0.8]], dtype="float32")
    top_k = paddle.to_tensor([[4], [3]], dtype="int64")
    infer_seed = paddle.to_tensor([[10], [20]], dtype="int64")
    seq_lens_this_time = paddle.to_tensor([[2], [1]], dtype="int64")
    seq_lens_encoder = paddle.to_tensor([[0], [1]], dtype="int64")

    original_gather = paddle.gather

    def _safe_gather(x, index, axis=0, name=None):
        if x.dtype == paddle.bool:
            x = x.astype("int32")
        return original_gather(x, index, axis=axis, name=name)

    monkeypatch.setattr(paddle, "gather", _safe_gather)
    original_where = paddle.where

    def _safe_where(condition, x=None, y=None, name=None):
        if condition.dtype != paddle.bool:
            condition = condition.astype("bool")
        return original_where(condition, x, y, name=name)

    monkeypatch.setattr(paddle, "where", _safe_where)
    top_p_padding, top_k_padding, topp_seed = padding_sampling_params(
        top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder
    )
    assert top_p_padding.shape[0] == 3
    assert top_k_padding.shape[0] == 3
    seed_values = topp_seed[:, 0].numpy().tolist()
    assert seed_values[1] - seed_values[0] == 4


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


def test_guided_decoding_update_and_mask(monkeypatch):
    guided = GuidedDecoding(_make_min_fd_config(1))
    guided.fill_bitmask_parallel_batch_size = 1
    processor = FakeLogitsProcessor()
    future = Future()
    future.set_result(processor)
    guided.add_logits_processor(0, future=future, prefill_tokens=[])
    guided.update_vocab_mask(prefill_done_idxs=[0])
    guided.join_async_fillmask()
    guided._tokens_to_acc[0] = [2, 3]
    guided.accept_tokens_from_prefill_node(0)
    assert processor.accepted_tokens == [2, 3]
    assert processor.fill_calls

    def _apply_token_mask(logits, token_bitmask, indices, is_cuda_platform):
        assert indices == [0]
        return logits + 1.0

    stub_backend = types.SimpleNamespace(apply_token_mask=_apply_token_mask)
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.guided_decoding.xgrammar_backend", stub_backend)
    logits = paddle.zeros([1, 4], dtype="float32")
    guided._prefill_done_idxs[0] = True
    guided.logits_processors[0] = processor
    masked_logits = guided.apply_token_mask(logits)
    assert masked_logits[0, 0].item() == pytest.approx(1.0)


def test_guided_decoding_reasoning_and_reset():
    guided = GuidedDecoding(_make_min_fd_config(2))
    processor = FakeLogitsProcessor()
    guided.logits_processors[0] = processor
    guided._prefill_done_idxs[0] = True
    guided.apply_reasoning_parser(FakeReasoningParser(should_end=True))
    next_tokens = paddle.to_tensor([[5], [6]], dtype="int64")
    guided.update_output_tokens(next_tokens)
    assert processor.reasoning_ended is True
    guided.apply_reasoning_parser(None)
    guided.logits_processors[1] = FakeLogitsProcessor(accept_result=False)
    guided._prefill_done_idxs[1] = True
    guided.update_output_tokens(paddle.to_tensor([[5], [7]], dtype="int64"))
    assert guided.logits_processors[1] is None
    guided.logits_processors[0] = FakeLogitsProcessor()
    guided._prefill_done_idxs[0] = True
    guided.update_output_tokens(paddle.to_tensor([[-1]], dtype="int64"))
    assert guided.logits_processors[0] is None


def test_guided_decoding_update_output_tokens_empty():
    guided = GuidedDecoding(_make_min_fd_config(0))
    guided.update_output_tokens(paddle.to_tensor([[1]], dtype="int64"))
    assert guided.logits_processors == []


def test_guided_decoding_update_output_tokens_short_batch():
    guided = GuidedDecoding(_make_min_fd_config(2))
    guided.logits_processors[0] = FakeLogitsProcessor()
    guided.logits_processors[1] = FakeLogitsProcessor()
    guided._prefill_done_idxs[0] = True
    guided._prefill_done_idxs[1] = True
    guided.update_output_tokens(paddle.to_tensor([[1]], dtype="int64"))
    assert guided.logits_processors[1] is not None


def test_sampler_compute_and_gather_logprobs():
    sampler = Sampler.__new__(Sampler)
    logits = paddle.to_tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], dtype="float32")
    sampling_metadata = _create_default_sampling_metadata(batch_size=2, min_seq_len=1, max_seq_len=3)
    sampling_metadata.temp_scaled_logprobs = paddle.to_tensor([[1], [0]], dtype="bool")
    sampling_metadata.temp_scaled_logprobs_flag = True
    sampling_metadata.top_p_normalized_logprobs = paddle.to_tensor([[1], [0]], dtype="bool")
    sampling_metadata.top_p_normalized_logprobs_flag = True
    sampling_metadata.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1], [1]], dtype="int64"),
        "seq_lens_encoder": paddle.to_tensor([[0], [0]], dtype="int64"),
        "seq_lens_decoder": paddle.to_tensor([[0], [0]], dtype="int64"),
    }
    sampling_metadata.top_p = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
    sampling_metadata.temperature = paddle.to_tensor([[2.0], [1.0]], dtype="float32")
    logprobs = sampler.compute_logprobs(logits, sampling_metadata)
    expected_probs = top_p_normalize_probs_paddle(F.softmax(logits / 2.0, axis=-1), sampling_metadata.top_p)
    assert logprobs[0].exp().numpy().sum() == pytest.approx(1.0)
    assert logprobs[0].exp()[2].item() <= expected_probs[0, 2].item() + 1e-6
    token_ids = paddle.to_tensor([2, 0], dtype="int64")
    gathered = sampler.gather_logprobs(logprobs, num_logprobs=2, token_ids=token_ids)
    assert gathered.logprob_token_ids.shape[1] == 3
    assert gathered.selected_token_ranks.shape[0] == 2
    gathered_zero = sampler.gather_logprobs(logprobs, num_logprobs=0, token_ids=token_ids)
    assert gathered_zero.logprob_token_ids.shape[1] == 1


def test_sampler_init_and_hooks(monkeypatch):
    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits"),
        scheduler_config=types.SimpleNamespace(max_num_seqs=1),
        early_stop_config=None,
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_iluvatar", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_gcu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_dcu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_maca", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_intel_hpu", lambda: False)
    sampler = Sampler(fd_config=fd_config)
    sampler.apply_logits_processor(0, future=None, prefill_tokens=[])
    sampler.pre_process([])
    sampler.post_process(paddle.to_tensor([[1]], dtype="int64"))
    assert sampler.logprobs_mode == "raw_logits"


def test_sampler_forward_cuda(monkeypatch):
    sampler = _make_stubbed_sampler(logprobs_mode="processed_logprobs")
    logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
    sampling_metadata = _build_sampling_metadata_for_forward(batch_size=1, min_seq_len=1, max_seq_len=3)
    sampling_metadata.logits_processors = []

    def _apply_penalty(*args, **kwargs):
        return args[3] + 0.5

    def _min_p_sampling(probs, min_p, min_p_list):
        return probs

    def _top_k_top_p_sampling(probs, top_p, top_k, top_k_list, topp_seed=None):
        return None, paddle.to_tensor([[2]], dtype="int64")

    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.apply_penalty_multi_scores", _apply_penalty)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.min_p_sampling", _min_p_sampling)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling", _top_k_top_p_sampling)

    output = sampler.forward_cuda(logits, sampling_metadata)
    assert output.sampled_token_ids.numpy().tolist() == [[2]]
    assert output.logprobs_tensors is not None


def test_sampler_forward_intel_hpu(monkeypatch):
    sampler = Sampler.__new__(Sampler)

    def _fused_sampler(*args, **kwargs):
        return None, paddle.to_tensor([[1], [2]], dtype="int64")

    stub_hpu = types.SimpleNamespace(fused_sampler=_fused_sampler)
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.ops.intel_hpu", stub_hpu)
    sampling_metadata = _create_default_sampling_metadata(batch_size=2, min_seq_len=1, max_seq_len=3)
    logits = paddle.ones([2, 4], dtype="float16")
    batch_ids = paddle.to_tensor([0, 1], dtype="int64")
    result = sampler.forward_intel_hpu(logits, sampling_metadata, batch_ids, max_batch=3, rank=0, local_rank=0)
    assert result.shape[0] == 3


def test_speculative_sampler_init_and_hooks(monkeypatch):
    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits", think_end_id=1, line_break_id=2),
        speculative_config=types.SimpleNamespace(
            verify_window=2,
            max_candidate_len=4,
            benchmark_mode=False,
            enf_gen_phase_tag=False,
        ),
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    sampler = SpeculativeSampler(fd_config)
    sampler.pre_process([])
    sampler.set_reasoning_parser(None)
    sampler.post_process(paddle.to_tensor([[1]], dtype="int64"))
    sampler.apply_logits_processor(0)
    assert sampler.logprobs_mode == "raw_logits"


def test_mtp_sampler_forward_cuda(monkeypatch):
    sampler = MTPSampler.__new__(MTPSampler)
    sampler.logprobs_mode = "raw_logits"
    sampler.enable_draft_logprob = True
    sampling_metadata = _create_default_sampling_metadata(
        batch_size=2, min_seq_len=1, max_seq_len=3, max_num_logprobs=1
    )
    share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1], [1]], dtype="int64"),
        "seq_lens_encoder": paddle.to_tensor([[0], [1]], dtype="int64"),
        "batch_token_num": paddle.to_tensor([[1], [1]], dtype="int64"),
        "substep": 0,
        "draft_logits": paddle.ones([2, 4], dtype="float32"),
        "accept_tokens": paddle.zeros([2, 2], dtype="int64"),
        "cu_next_token_offset": paddle.zeros([2], dtype="int64"),
        "cu_batch_token_offset": paddle.zeros([2], dtype="int64"),
        "output_padding_offset": paddle.zeros([2, 1], dtype="int64"),
        "output_cum_offsets": paddle.zeros([2, 1], dtype="int64"),
        "batch_id_per_token_output": paddle.zeros([2], dtype="int64"),
        "cu_seqlens_q_output": paddle.zeros([3], dtype="int32"),
    }
    sampling_metadata.share_inputs = share_inputs

    def _apply_speculative_penalty(*args, **kwargs):
        return args[1]

    def _speculate_insert_first_token(token_ids, accept_tokens, next_tokens, *args, **kwargs):
        token_ids[:] = next_tokens.flatten()

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_speculative_penalty_multi_scores",
        _apply_speculative_penalty,
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.speculate_insert_first_token",
        _speculate_insert_first_token,
    )
    next_tokens, output = sampler.forward_cuda(
        paddle.ones([2, 4], dtype="float32"), sampling_metadata, max_model_len=8, share_inputs=share_inputs
    )
    assert next_tokens.shape[0] == 2
    assert output.logprobs_tensors is not None


def test_mtp_sampler_init_and_compute_logprobs(monkeypatch):
    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits"),
        speculative_config=types.SimpleNamespace(enable_draft_logprob=True),
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    sampler = MTPSampler(fd_config)
    sampler.pre_process([])
    sampler.set_reasoning_parser(None)
    sampler.post_process(paddle.to_tensor([[1]], dtype="int64"))
    sampler.apply_logits_processor(0)


if __name__ == "__main__":
    pytest.main([__file__])
