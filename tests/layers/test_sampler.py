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
from unittest.mock import MagicMock

import paddle
import pytest

import fastdeploy  # noqa: F401

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda *args, **kwargs: None)

# Optional runtime deps are intentionally stubbed for unit isolation.
if "triton" not in sys.modules:
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_lang_stub = types.ModuleType("triton.language")
    triton_lang_stub.constexpr = int
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub

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
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.batched_count_greater_than", lambda x, y: (x >= y).sum(-1)
    )
    # Also patch batched_count_greater_than in logprobs module itself, because
    # build_output_logprobs -> gather_logprobs calls it from logprobs scope.
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.logprobs.batched_count_greater_than", lambda x, y: (x >= y).sum(-1)
    )


@pytest.fixture
def mock_ops(monkeypatch):
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_penalty_multi_scores", lambda *a, **k: a[1]
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_speculative_penalty_multi_scores",
        lambda *a, **k: a[2],
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.min_p_sampling", lambda probs, *a, **k: probs)
    return monkeypatch


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


def _create_metadata(batch_size=1, min_seq_len=1, max_seq_len=3, max_num_logprobs=None, **overrides):
    m = SamplingMetadata(
        temperature=paddle.full(shape=[batch_size, 1], fill_value=0.9, dtype="float32"),
        top_p=paddle.full(shape=[batch_size, 1], fill_value=0.7, dtype="float32"),
        prompt_lens=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        step_idx=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        token_ids_all=paddle.full(shape=[batch_size, max_seq_len], fill_value=-1, dtype="int64"),
        frequency_penalties=paddle.full(shape=[batch_size, 1], fill_value=0.0, dtype="float32"),
        presence_penalties=paddle.full(shape=[batch_size, 1], fill_value=0.0, dtype="float32"),
        repetition_penalties=paddle.full(shape=[batch_size, 1], fill_value=1.0, dtype="float32"),
        min_dec_lens=paddle.full(shape=[batch_size, 1], fill_value=min_seq_len, dtype="int64"),
        bad_words_token_ids=paddle.full(shape=[batch_size], fill_value=-1, dtype="int64"),
        bad_words_token_len=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size], fill_value=-2, dtype="int64"),
        min_p=paddle.zeros([batch_size], dtype="float32"),
        seed=paddle.full([batch_size, 1], 7, dtype="int64"),
        logits_processors=None,
    )
    m.max_num_logprobs = max_num_logprobs
    m.top_k = paddle.full([batch_size, 1], 5, dtype="int64")
    m.top_k_list = [5 for _ in range(batch_size)]
    m.min_p_list = [0.0 for _ in range(batch_size)]
    m.enable_early_stop = True
    m.stop_flags = paddle.zeros([batch_size, 1], dtype="int32")
    m.share_inputs = {
        "seq_lens_this_time": paddle.ones([batch_size, 1], dtype="int64"),
        "seq_lens_encoder": paddle.zeros([batch_size, 1], dtype="int64"),
        "seq_lens_decoder": paddle.zeros([batch_size, 1], dtype="int64"),
    }
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def _make_stubbed_sampler(mode="processed_logprobs"):
    s = Sampler.__new__(Sampler)
    s.guided_decoding = types.SimpleNamespace(apply_token_mask=lambda logits, p_done_idxs: logits)
    s.logprobs_mode = mode
    s.early_stopper = types.SimpleNamespace(process=lambda probs, next_tokens, stop_flags: None)
    return s


def _min_fd_config(max_num_seqs):
    return types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs))


def test_top_p_and_padding_sampling_params(monkeypatch):
    probs = paddle.to_tensor([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]], dtype="float32")
    top_ps = paddle.to_tensor([[0.7], [0.5]], dtype="float32")
    normalized = top_p_normalize_probs_paddle(probs, top_ps)
    assert paddle.allclose(normalized.sum(axis=-1), paddle.ones([2], dtype="float32"))

    top_p = paddle.to_tensor([[0.9], [0.8]], dtype="float32")
    top_k = paddle.to_tensor([[4], [3]], dtype="int64")
    infer_seed = paddle.to_tensor([[10], [20]], dtype="int64")
    seq_lens_this_time = paddle.to_tensor([[2], [1]], dtype="int64")
    seq_lens_encoder = paddle.to_tensor([[0], [1]], dtype="int64")

    original_gather, original_where = paddle.gather, paddle.where
    monkeypatch.setattr(
        paddle,
        "gather",
        lambda x, index, axis=0, name=None: original_gather(
            x.astype("int32") if x.dtype == paddle.bool else x, index, axis=axis, name=name
        ),
    )
    monkeypatch.setattr(
        paddle,
        "where",
        lambda cond, x=None, y=None, name=None: original_where(
            cond.astype("bool") if cond.dtype != paddle.bool else cond, x, y, name=name
        ),
    )

    top_p_padding, top_k_padding, topp_seed = padding_sampling_params(
        top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder
    )
    assert top_p_padding.shape[0] == 3 and top_k_padding.shape[0] == 3
    assert topp_seed[1, 0].item() - topp_seed[0, 0].item() == 4


@pytest.mark.parametrize("use_future,with_reasoning", [(False, False), (True, False), (False, True)])
def test_guided_decoding_core(monkeypatch, use_future, with_reasoning):
    guided = GuidedDecoding(_min_fd_config(1))
    processor = FakeLogitsProcessor()

    if use_future:
        guided.logits_processors[0] = FakeFuture(processor, done_value=False)
        guided._tokens_to_acc[0] = [9]
        guided._prefill_done_idxs[0] = True
    else:
        fut = Future()
        fut.set_result(processor)
        guided.add_logits_processor(0, future=fut, prefill_tokens=[])
        guided._tokens_to_acc[0] = [2]
        guided.update_vocab_mask(prefill_done_idxs=[0])
        guided.join_async_fillmask()

    if with_reasoning:
        processor.enable_reasoning = True
        guided.apply_reasoning_parser(FakeReasoningParser(True))

    monkeypatch.setitem(
        sys.modules,
        "fastdeploy.model_executor.guided_decoding.xgrammar_backend",
        types.SimpleNamespace(apply_token_mask=lambda logits, token_bitmask, indices, is_cuda_platform: logits + 1.0),
    )

    guided.token_bitmask = processor.allocate_token_bitmask()
    out = guided.apply_token_mask(paddle.zeros([1, 4], dtype="float32"))
    assert out.shape == [1, 4]
    assert processor.accepted_tokens in ([2], [9])


def test_sampler_compute_and_gather_logprobs():
    sampler = Sampler.__new__(Sampler)
    logits = paddle.to_tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], dtype="float32")
    m = _create_metadata(batch_size=2, max_num_logprobs=2)
    m.temp_scaled_logprobs = paddle.to_tensor([[1], [0]], dtype="bool")
    m.temp_scaled_logprobs_flag = True
    m.top_p_normalized_logprobs = paddle.to_tensor([[1], [0]], dtype="bool")
    m.top_p_normalized_logprobs_flag = True
    m.top_p = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
    m.temperature = paddle.to_tensor([[2.0], [1.0]], dtype="float32")
    logprobs = sampler.compute_logprobs(logits, m)
    gathered = sampler.gather_logprobs(logprobs, num_logprobs=2, token_ids=paddle.to_tensor([2, 0], dtype="int64"))
    assert gathered.logprob_token_ids.shape == [2, 3]


@pytest.mark.parametrize(
    "mode,next_token,use_processor",
    [("processed_logprobs", 2, False), ("raw_logprobs", 1, True), ("processed_logits", 1, True)],
)
def test_sampler_forward_cuda_variants(mock_ops, monkeypatch, mode, next_token, use_processor):
    sampler = _make_stubbed_sampler(mode)
    m = _create_metadata(batch_size=1, max_num_logprobs=2)
    m.logits_processors = [types.SimpleNamespace(apply=lambda t: t + 0.2)] if use_processor else []

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling",
        lambda probs, top_p, top_k, top_k_list, topp_seed=None: (
            None,
            paddle.to_tensor([[next_token]], dtype="int64"),
        ),
    )
    output = sampler.forward_cuda(paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32"), m)
    assert output.sampled_token_ids.numpy().tolist() == [[next_token]]
    assert output.logprobs_tensors is not None


def test_sampler_init_and_intel_hpu(monkeypatch):
    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits"),
        early_stop_config=types.SimpleNamespace(early_stop_strategy="none", enable_early_stop=False),
        scheduler_config=types.SimpleNamespace(max_num_seqs=1),
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_iluvatar", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_gcu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_dcu", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_maca", lambda: False)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_intel_hpu", lambda: False)
    sampler = Sampler(fd_config=fd_config)
    assert sampler.logprobs_mode == "raw_logits"

    monkeypatch.setitem(
        sys.modules,
        "fastdeploy.model_executor.ops.intel_hpu",
        types.SimpleNamespace(fused_sampler=lambda *a, **k: (None, paddle.to_tensor([[1], [2]], dtype="int64"))),
    )
    m = _create_metadata(batch_size=2)
    out = sampler.forward_intel_hpu(
        paddle.ones([2, 4], dtype="float16"), m, paddle.to_tensor([0, 1], dtype="int64"), 3, 0, 0
    )
    assert out.shape[0] == 3


def test_speculative_sampler_basic(monkeypatch):
    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits", think_end_id=1, line_break_id=2),
        speculative_config=types.SimpleNamespace(
            method="ngram",
            verify_window=2,
            max_candidate_len=4,
            benchmark_mode=False,
            enf_gen_phase_tag=False,
            verify_strategy="topp",
            accept_policy="normal",
        ),
        parallel_config=types.SimpleNamespace(prefill_one_step_stop=False),
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    sampler = SpeculativeSampler(fd_config)
    sampler.pre_process([])
    sampler.post_process(paddle.to_tensor([[1]], dtype="int64"))

    logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
    m = _create_metadata(batch_size=1)
    m.temp_scaled_logprobs = paddle.to_tensor([[1]], dtype="bool")
    m.top_p_normalized_logprobs = paddle.to_tensor([[1]], dtype="bool")
    m.temp_scaled_logprobs_flag = True
    m.top_p_normalized_logprobs_flag = True
    m.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "accept_num": paddle.to_tensor([1], dtype="int64"),
    }
    gathered = sampler.gather_logprobs(sampler.compute_logprobs(logits, m), 0, paddle.to_tensor([1], dtype="int64"))
    assert gathered.logprob_token_ids.shape[1] == 1


def test_mtp_sampler_xpu_and_compute(mock_ops, monkeypatch):
    sampler = MTPSampler.__new__(MTPSampler)
    sampler.logprobs_mode = "raw_logits"
    sampler.enable_draft_logprob = False
    m = _create_metadata(batch_size=1)
    m.top_k = paddle.full([1, 1], 2, dtype="int64")
    m.top_k_list = [2]
    m.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "seq_lens_encoder": paddle.to_tensor([[0]], dtype="int64"),
        "batch_token_num": paddle.to_tensor([[1]], dtype="int64"),
        "output_padding_offset": paddle.zeros([1, 1], dtype="int64"),
        "output_cum_offsets": paddle.zeros([1, 1], dtype="int64"),
        "batch_id_per_token_output": paddle.to_tensor([0], dtype="int32"),
        "cu_seqlens_q_output": paddle.to_tensor([0, 1], dtype="int32"),
    }
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling",
        lambda *a, **k: (None, paddle.to_tensor([[1]], dtype="int64")),
    )
    next_tokens, output = sampler.forward_xpu(
        paddle.ones([1, 4], dtype="float32"), m, max_model_len=8, share_inputs=m.share_inputs
    )
    assert next_tokens.shape[0] == 1 and output.logprobs_tensors is None

    fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(logprobs_mode="raw_logits"),
        speculative_config=types.SimpleNamespace(enable_draft_logprob=True),
    )
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu", lambda: False)
    mtp = MTPSampler(fd_config)
    m2 = _create_metadata(batch_size=1)
    m2.top_p_normalized_logprobs = paddle.to_tensor([[1]], dtype="bool")
    m2.top_p_normalized_logprobs_flag = True
    m2.temp_scaled_logprobs = paddle.to_tensor([[1]], dtype="bool")
    m2.temp_scaled_logprobs_flag = True
    m2.share_inputs = {
        "seq_lens_this_time": paddle.to_tensor([[1]], dtype="int64"),
        "batch_token_num": paddle.to_tensor([[1]], dtype="int64"),
    }
    gathered = mtp.gather_logprobs(
        mtp.compute_logprobs(paddle.to_tensor([[1.0, 2.0]], dtype="float32"), m2),
        0,
        paddle.to_tensor([1], dtype="int64"),
    )
    assert gathered.logprob_token_ids.shape[1] == 1


def test_sampler_deterministic_log_mode_calls_diagnostic(mock_ops, monkeypatch):
    """When FD_DETERMINISTIC_LOG_MODE is True, sampler calls _record_logits_diagnostic."""
    import fastdeploy.envs as envs

    # Mock the diagnostic function
    mock_diagnostic = MagicMock()
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler._record_logits_diagnostic",
        mock_diagnostic,
    )

    # Enable deterministic log mode
    monkeypatch.setattr(envs, "FD_DETERMINISTIC_LOG_MODE", True)

    sampler = _make_stubbed_sampler()
    m = _create_metadata(batch_size=1, max_num_logprobs=None)
    m.logits_processors = []

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.top_k_top_p_sampling",
        lambda probs, top_p, top_k, top_k_list, topp_seed=None: (
            None,
            paddle.to_tensor([[1]], dtype="int64"),
        ),
    )

    logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
    sampler.forward_cuda(logits, m)

    # Should be called twice: once for raw_logits, once for post_penalty_logits
    assert mock_diagnostic.call_count == 2
    call_tags = [call.kwargs.get("tag", "") for call in mock_diagnostic.call_args_list]
    assert "raw_logits" in call_tags
    assert "post_penalty_logits" in call_tags


def test_top_k_top_p_sampling_resets_cuda_rng_in_deterministic_mode(monkeypatch):
    """When FD_DETERMINISTIC_MODE is True, top_k_top_p_sampling resets CUDA RNG."""
    import sys

    import fastdeploy.envs as envs

    # Enable deterministic mode
    monkeypatch.setattr(envs, "FD_DETERMINISTIC_MODE", True)

    # Get the module directly from sys.modules
    sampling_mod = sys.modules["fastdeploy.model_executor.layers.sample.ops.top_k_top_p_sampling"]

    # Mock the reset function to track if it's called
    mock_reset = MagicMock()
    monkeypatch.setattr(sampling_mod, "_reset_cuda_generator_for_determinism", mock_reset)

    # Mock the sampling to avoid actual GPU ops
    monkeypatch.setattr(
        sampling_mod.paddle.tensor,
        "top_p_sampling",
        lambda *a, **k: (None, paddle.to_tensor([[1]], dtype="int64")),
    )

    probs = paddle.to_tensor([[0.5, 0.3, 0.2]], dtype="float32")
    top_p = paddle.to_tensor([[0.9]], dtype="float32")

    sampling_mod.top_k_top_p_sampling(probs, top_p)

    # Verify reset function was called
    mock_reset.assert_called_once()


def mixed_mock(probs, *a, **k):
    ids = paddle.argmax(probs, axis=-1, keepdim=True)
    # 1 left non_zero token after renorm → greedy, or → renturn 99
    non_zero_count = (probs > 0).sum(axis=-1, keepdim=True)
    sampled = non_zero_count > 1
    ids = paddle.where(sampled, paddle.to_tensor([[99]], dtype="int64"), ids)
    return None, ids


def test_top_k_1_returns_argmax(monkeypatch):
    """top_k=1 should produce argmax results regardless of FD_DETERMINISTIC_MODE."""
    import sys

    import fastdeploy.envs as envs

    sampling_mod = sys.modules["fastdeploy.model_executor.layers.sample.ops.top_k_top_p_sampling"]

    # Enable deterministic mode and force "base" sampling class
    monkeypatch.setattr(envs, "FD_DETERMINISTIC_MODE", True)
    monkeypatch.setattr(envs, "FD_SAMPLING_CLASS", "base")

    # Probs with clear argmax: row0 -> col2, row1 -> col0
    probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], dtype="float32")
    top_p = paddle.to_tensor([[0.9], [0.9]], dtype="float32")
    expected = paddle.argmax(probs, axis=-1, keepdim=True)

    # --- All-greedy: both rows top_k=1 ---
    top_k = paddle.to_tensor([[1], [1]], dtype="int64")
    top_k_list = [1, 1]

    _, ids = sampling_mod.top_k_top_p_sampling(probs, top_p, top_k, top_k_list)
    assert paddle.equal_all(ids, expected), f"all-greedy: {ids.numpy()} != {expected.numpy()}"

    # --- Mixed batch: row0 greedy, row1 sampled ---
    top_k_mixed = paddle.to_tensor([[1], [50]], dtype="int64")
    top_k_list_mixed = [1, 50]

    # Mock the base sampling to return a fixed token for all rows
    monkeypatch.setattr(
        sampling_mod.paddle.tensor,
        "top_p_sampling",
        mixed_mock,
    )

    _, ids_mixed = sampling_mod.top_k_top_p_sampling(probs, top_p, top_k_mixed, top_k_list_mixed)
    # Row 0 (greedy) must be argmax=2, row 1 keeps sampled value=99
    assert ids_mixed[0, 0].item() == 2, f"mixed row0: expected 2, got {ids_mixed[0, 0].item()}"
    assert ids_mixed[1, 0].item() == 99, f"mixed row1: expected 99, got {ids_mixed[1, 0].item()}"


def test_top_k_1_returns_argmax_without_deterministic_mode(monkeypatch):
    """top_k=1 should trigger argmax even when FD_DETERMINISTIC_MODE is False."""
    import sys

    import fastdeploy.envs as envs

    sampling_mod = sys.modules["fastdeploy.model_executor.layers.sample.ops.top_k_top_p_sampling"]

    # Disable deterministic mode, force "base" sampling class
    monkeypatch.setattr(envs, "FD_DETERMINISTIC_MODE", False)
    monkeypatch.setattr(envs, "FD_SAMPLING_CLASS", "base")

    probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], dtype="float32")
    top_p = paddle.to_tensor([[0.9], [0.9]], dtype="float32")
    expected = paddle.argmax(probs, axis=-1, keepdim=True)

    # --- All-greedy ---
    top_k = paddle.to_tensor([[1], [1]], dtype="int64")
    top_k_list = [1, 1]

    _, ids = sampling_mod.top_k_top_p_sampling(probs, top_p, top_k, top_k_list)
    assert paddle.equal_all(ids, expected), f"all-greedy: {ids.numpy()} != {expected.numpy()}"

    # --- Mixed batch: row0 greedy, row1 sampled ---
    top_k_mixed = paddle.to_tensor([[1], [50]], dtype="int64")
    top_k_list_mixed = [1, 50]

    monkeypatch.setattr(
        sampling_mod.paddle.tensor,
        "top_p_sampling",
        mixed_mock,
    )

    _, ids_mixed = sampling_mod.top_k_top_p_sampling(probs, top_p, top_k_mixed, top_k_list_mixed)
    assert ids_mixed[0, 0].item() == 2, f"mixed row0: expected 2, got {ids_mixed[0, 0].item()}"
    assert ids_mixed[1, 0].item() == 99, f"mixed row1: expected 99, got {ids_mixed[1, 0].item()}"


if __name__ == "__main__":
    pytest.main([__file__])
