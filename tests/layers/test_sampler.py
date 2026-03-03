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
from dataclasses import dataclass, field
from pathlib import Path

import paddle
import paddle.nn.functional as F
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda *args, **kwargs: None)


if "triton" not in sys.modules:
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_lang_stub = types.ModuleType("triton.language")
    triton_lang_stub.constexpr = int
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub

# Avoid importing fastdeploy/__init__.py during unit tests. The package __init__
# pulls in optional runtime dependencies that are not required by sampler tests.
if "fastdeploy" not in sys.modules:
    fastdeploy_pkg = types.ModuleType("fastdeploy")
    fastdeploy_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "fastdeploy")]
    sys.modules["fastdeploy"] = fastdeploy_pkg

from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import (
    GuidedDecoding,
    MTPSampler,
    Sampler,
    SpeculativeSampler,
    padding_sampling_params,
    top_p_normalize_probs_paddle,
)


@pytest.fixture(autouse=True)
def _disable_triton_cuda_path(monkeypatch):
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.logprobs.current_platform.is_cuda", lambda: False)


@pytest.fixture
def common_ops_mocks(monkeypatch):
    """Common lightweight op mocks for sampler-family tests."""

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_penalty_multi_scores", lambda *a, **k: a[1]
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_speculative_penalty_multi_scores",
        lambda *a, **k: a[2],
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.min_p_sampling", lambda probs, *a, **k: probs)
    return monkeypatch


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
        min_p=paddle.randn([batch_size], dtype="float32"),
        seed=paddle.to_tensor([[2025]]),
        logits_processors=None,
    )
    if max_num_logprobs is not None:
        fake_sampling_metadata.max_num_logprobs = max_num_logprobs
    return fake_sampling_metadata


@dataclass
class FakeLogitsProcessor:
    accept_result: bool = True
    is_terminated: bool = False
    enable_reasoning: bool = False
    reasoning_ended: bool = False
    accepted_tokens: list = field(default_factory=list)
    fill_calls: list = field(default_factory=list)

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


def test_guided_decoding_add_logits_processor_variants():
    guided = GuidedDecoding(_make_min_fd_config(2))
    processor = FakeLogitsProcessor()
    future_done = Future()
    future_done.set_result(processor)
    guided.add_logits_processor(0, future=future_done, prefill_tokens=[1])
    assert processor.accepted_tokens == [1]
    guided.add_logits_processor(1, future=None)
    assert guided.logits_processors[1] is None
    async_future = FakeFuture(processor, done_value=False)
    guided.add_logits_processor(1, future=async_future, prefill_tokens=[4])
    assert guided._tokens_to_acc[1] == [4]
    processor.enable_reasoning = True
    guided.apply_reasoning_parser(FakeReasoningParser())
    assert guided.should_fill_bitmask(0) is True
    processor.enable_reasoning = False
    processor.reasoning_ended = False
    assert guided.should_fill_bitmask(0) is False


def test_guided_decoding_apply_token_mask_future_wait(monkeypatch):
    guided = GuidedDecoding(_make_min_fd_config(1))
    processor = FakeLogitsProcessor()
    future = FakeFuture(processor, done_value=False)
    guided.logits_processors[0] = future
    guided._prefill_done_idxs[0] = True
    guided._tokens_to_acc[0] = [9]

    def _apply_token_mask(logits, token_bitmask, indices, is_cuda_platform):
        return logits

    stub_backend = types.SimpleNamespace(apply_token_mask=_apply_token_mask)
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.guided_decoding.xgrammar_backend", stub_backend)
    guided.token_bitmask = processor.allocate_token_bitmask()
    logits = paddle.zeros([1, 4], dtype="float32")
    guided.apply_token_mask(logits)
    assert processor.accepted_tokens == [9]


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


def test_sampler_compute_logprobs_without_metadata():
    sampler = Sampler.__new__(Sampler)
    logits = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
    logprobs = sampler.compute_logprobs(logits, sampling_metadata=None)
    assert paddle.allclose(logprobs, F.log_softmax(logits, axis=-1))


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


@pytest.mark.parametrize(
    "logprobs_mode,next_token,use_processor",
    [
        ("processed_logprobs", 2, False),
        ("raw_logprobs", 1, True),
        ("processed_logits", 1, True),
    ],
)
def test_sampler_forward_cuda_variants(common_ops_mocks, monkeypatch, logprobs_mode, next_token, use_processor):
    sampler = _make_stubbed_sampler(logprobs_mode=logprobs_mode)
    logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
    sampling_metadata = _build_sampling_metadata_for_forward(batch_size=1, min_seq_len=1, max_seq_len=3)

    if use_processor:
        sampling_metadata.logits_processors = [types.SimpleNamespace(apply=lambda tensor: tensor + 0.2)]
    else:
        sampling_metadata.logits_processors = []

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling",
        lambda probs, top_p, top_k, top_k_list, topp_seed=None: (
            None,
            paddle.to_tensor([[next_token]], dtype="int64"),
        ),
    )

    output = sampler.forward_cuda(logits, sampling_metadata)
    assert output.sampled_token_ids.numpy().tolist() == [[next_token]]
    assert output.logprobs_tensors is not None
    assert output.logits.shape[-1] == 3


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


def test_sampler_forward_intel_hpu_full_batch(monkeypatch):
    sampler = Sampler.__new__(Sampler)

    def _fused_sampler(*args, **kwargs):
        return None, paddle.to_tensor([[1], [2], [3]], dtype="int64")

    stub_hpu = types.SimpleNamespace(fused_sampler=_fused_sampler)
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.ops.intel_hpu", stub_hpu)
    sampling_metadata = _create_default_sampling_metadata(batch_size=3, min_seq_len=1, max_seq_len=3)
    logits = paddle.ones([3, 4], dtype="float32")
    batch_ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    result = sampler.forward_intel_hpu(logits, sampling_metadata, batch_ids, max_batch=3, rank=0, local_rank=0)
    assert result.shape[0] == 3


def _make_speculative_sampler():
    sampler = SpeculativeSampler.__new__(SpeculativeSampler)
    sampler.logprobs_mode = "raw_logprobs"
    sampler.speculative_verify_window = 2
    sampler.speculative_max_candidate_len = 3
    sampler.speculative_benchmark_mode = False
    sampler.think_end_id = 1
    sampler.line_break_id = 2
    sampler.enf_gen_phase_tag = True
    return sampler


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


def test_speculative_sampler_compute_and_gather_logprobs():
    sampler = _make_speculative_sampler()
    logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
    sampling_metadata = _create_default_sampling_metadata(batch_size=1, min_seq_len=1, max_seq_len=3)
    sampling_metadata.temp_scaled_logprobs = paddle.to_tensor([[1]], dtype="bool")
    sampling_metadata.top_p_normalized_logprobs = paddle.to_tensor([[1]], dtype="bool")
    sampling_metadata.temp_scaled_logprobs_flag = True
    sampling_metadata.top_p_normalized_logprobs_flag = True
    sampling_metadata.top_p = paddle.to_tensor([[0.6]], dtype="float32")
    sampling_metadata.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "accept_num": paddle.to_tensor([1], dtype="int64"),
    }
    logprobs = sampler.compute_logprobs(logits, sampling_metadata)
    gathered = sampler.gather_logprobs(logprobs, num_logprobs=0, token_ids=paddle.to_tensor([1], dtype="int64"))
    assert gathered.logprob_token_ids.shape[1] == 1


def test_mtp_sampler_forward_xpu(common_ops_mocks, monkeypatch):
    sampler = MTPSampler.__new__(MTPSampler)
    sampler.logprobs_mode = "raw_logits"
    sampler.enable_draft_logprob = False
    sampling_metadata = _create_default_sampling_metadata(batch_size=1, min_seq_len=1, max_seq_len=3)
    sampling_metadata.top_k = paddle.full([1, 1], 2, dtype="int64")
    sampling_metadata.top_k_list = [2]
    share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "seq_lens_encoder": paddle.to_tensor([[0]], dtype="int64"),
        "batch_token_num": paddle.to_tensor([[1]], dtype="int64"),
        "output_padding_offset": paddle.zeros([1, 1], dtype="int64"),
        "output_cum_offsets": paddle.zeros([1, 1], dtype="int64"),
    }
    sampling_metadata.share_inputs = share_inputs

    def _top_k_top_p_sampling(probs, top_p, top_k, top_k_list):
        return None, paddle.to_tensor([[1]], dtype="int64")

    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling", _top_k_top_p_sampling)
    next_tokens, output = sampler.forward_xpu(
        paddle.ones([1, 4], dtype="float32"), sampling_metadata, max_model_len=8, share_inputs=share_inputs
    )
    assert next_tokens.shape[0] == 1
    assert output.logprobs_tensors is None


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

    sampling_metadata = _create_default_sampling_metadata(batch_size=1, min_seq_len=1, max_seq_len=3)
    sampling_metadata.top_p_normalized_logprobs = paddle.to_tensor([[1]], dtype="bool")
    sampling_metadata.top_p_normalized_logprobs_flag = True
    sampling_metadata.temp_scaled_logprobs = paddle.to_tensor([[1]], dtype="bool")
    sampling_metadata.temp_scaled_logprobs_flag = True
    sampling_metadata.top_p = paddle.to_tensor([[0.9]], dtype="float32")
    sampling_metadata.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "batch_token_num": paddle.to_tensor([[1]], dtype="int64"),
    }
    logprobs = sampler.compute_logprobs(paddle.to_tensor([[1.0, 2.0]], dtype="float32"), sampling_metadata)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.logprobs.current_platform.is_cuda", lambda: False)
    gathered = sampler.gather_logprobs(logprobs, num_logprobs=0, token_ids=paddle.to_tensor([1], dtype="int64"))
    assert gathered.logprob_token_ids.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__])
