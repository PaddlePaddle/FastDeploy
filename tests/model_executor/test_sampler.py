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

from __future__ import annotations

import importlib
import sys
import types
from concurrent.futures import Future

import paddle
import pytest

from fastdeploy.model_executor.layers.sample import sampler as sampler_module
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.worker.output import LogprobsTensors


class _FakePlatform:
    def __init__(self, kind: str = "cuda"):
        self.kind = kind

    def is_cuda(self):  # noqa: D401 - simple boolean helpers
        return self.kind == "cuda"

    def is_xpu(self):
        return self.kind == "xpu"

    def is_iluvatar(self):
        return False

    def is_gcu(self):
        return False

    def is_dcu(self):
        return False

    def is_maca(self):
        return False

    def is_intel_hpu(self):
        return self.kind == "intel_hpu"


class _RecordingProcessor(sampler_module.LogitsProcessorBase):
    def __init__(self, enable_reasoning: bool = False):
        super().__init__(enable_reasoning=enable_reasoning)
        self.accepted_tokens: list[int] = []
        self.token_bitmask = None
        self.mask_fills: list[int] = []
        self.applied = 0
        self.terminated = False

    def allocate_token_bitmask(self):
        return paddle.zeros([2, 3], dtype="float32")

    def fill_token_bitmask(self, token_bitmask, idx):
        token_bitmask[idx, :] = idx + 1
        self.mask_fills.append(idx)

    def apply_token_mask(self, logits, token_bitmask, indices=None):
        self.applied += 1
        masked = logits.clone()
        if indices:
            for idx in indices:
                masked[idx] = 0
        return masked

    def accept_token(self, token):
        self.accepted_tokens.append(int(token))
        return True

    def is_terminated(self):
        return self.terminated

    def reset(self):
        self.accepted_tokens.clear()

    def copy(self):  # pragma: no cover - unused helper
        return _RecordingProcessor()


class _AddOffsetProcessor:
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def apply(self, logits):
        return logits + self.delta


class _FakeEarlyStopper:
    def __init__(self):
        self.initialized_with = None
        self.process_calls: list[tuple] = []

    def initialize(self, max_num_seqs, config):
        self.initialized_with = (max_num_seqs, config)

    def process(self, probs, next_tokens, stop_flags):
        self.process_calls.append((probs, next_tokens, stop_flags))


def _tensor(value, dtype="float32"):
    return paddle.to_tensor(value, dtype=dtype)


def _build_sampling_metadata(**overrides) -> SamplingMetadata:
    base = dict(
        temperature=_tensor([[1.0], [0.5]]),
        pre_token_ids=_tensor([[0, 1], [1, 2]], dtype="int64"),
        eos_token_ids=_tensor([2, 3], dtype="int64"),
        frequency_penalties=_tensor([[0.0], [0.0]]),
        presence_penalties=_tensor([[0.0], [0.0]]),
        repetition_penalties=_tensor([[1.0], [1.1]]),
        min_dec_lens=_tensor([[0], [0]], dtype="int64"),
        bad_words_token_ids=_tensor([[0], [0]], dtype="int64"),
        step_idx=_tensor([[0], [0]], dtype="int64"),
        top_p=_tensor([[0.9], [0.8]]),
        top_k=_tensor([[2], [2]], dtype="int64"),
        top_k_list=[2, 2],
        min_p=_tensor([[0.2], [0.3]]),
        min_p_list=[0.2, 0.3],
        seed=_tensor([[123]], dtype="int64"),
        max_num_logprobs=2,
        enable_early_stop=True,
        stop_flags=_tensor([[0], [0]], dtype="int64"),
        share_inputs=dict(
            seq_lens_this_time=_tensor([[1], [1]], dtype="int64"),
            seq_lens_encoder=_tensor([[0], [0]], dtype="int64"),
            seq_lens_decoder=_tensor([[0], [0]], dtype="int64"),
            output_padding_offset=_tensor([[0], [0]], dtype="int64"),
            output_cum_offsets=_tensor([[0], [0]], dtype="int64"),
            accept_tokens=_tensor([[0, 0]], dtype="int64"),
            accept_num=_tensor([1], dtype="int64"),
            step_idx=_tensor([[0], [0]], dtype="int64"),
            stop_flags=_tensor([[0], [0]], dtype="int64"),
            draft_tokens=_tensor([[0, 0]], dtype="int64"),
            max_dec_len=_tensor([[8], [8]], dtype="int64"),
            is_block_step=_tensor([[0], [0]], dtype="int64"),
        ),
    )
    base.update(overrides)
    return SamplingMetadata(**base)


@pytest.fixture(autouse=True)
def sampler_op_stubs(monkeypatch):
    """Patch heavy CUDA helpers with deterministic numpy-free fallbacks."""

    records = {}

    def _apply_penalty(*args, **kwargs):
        records["apply_penalty"] = args[0]
        logits = args[3]
        return logits + 0.1

    def _apply_speculative_penalty(*args, **kwargs):
        records["apply_speculative"] = True
        return args[1]

    def _min_p_sampling(probs, *_, **__):
        records.setdefault("min_p_calls", 0)
        records["min_p_calls"] += 1
        return probs

    def _top_k_top_p_sampling(probs, *_, **__):
        next_tokens = paddle.argmax(probs, axis=-1, keepdim=True).astype("int64")
        return probs, next_tokens

    def _speculate_insert_first_token(token_ids, *_args):
        token_ids.set_value(paddle.arange(token_ids.shape[0], dtype="int64"))

    def _speculate_get_target_logits(target_logits, logits, *_args):
        target_logits.set_value(logits[: target_logits.shape[0]])

    monkeypatch.setattr(sampler_module, "apply_penalty_multi_scores", _apply_penalty)
    monkeypatch.setattr(sampler_module, "apply_speculative_penalty_multi_scores", _apply_speculative_penalty)
    monkeypatch.setattr(sampler_module, "min_p_sampling", _min_p_sampling)
    monkeypatch.setattr(sampler_module, "top_k_top_p_sampling", _top_k_top_p_sampling)
    monkeypatch.setattr(sampler_module, "speculate_insert_first_token", _speculate_insert_first_token)
    monkeypatch.setattr(sampler_module, "speculate_get_target_logits", _speculate_get_target_logits)

    def _ensure_module(mod_path: str):
        try:
            module = importlib.import_module(mod_path)
            created = False
        except ImportError:
            module = types.ModuleType(mod_path)
            sys.modules[mod_path] = module
            created = True
        return module, created

    _MISSING = object()
    gpu_mod, created_gpu = _ensure_module("fastdeploy.model_executor.ops.gpu")
    hpu_mod, created_hpu = _ensure_module("fastdeploy.model_executor.ops.intel_hpu")
    original_gpu_attrs = {}
    original_hpu_attrs = {}

    def _set_attr(module, name, value, store):
        store[name] = getattr(module, name, _MISSING)
        setattr(module, name, value)

    def _speculate_verify(*_args, **_kwargs):
        records["speculate_verify"] = True

    def _top_p_candidates(probs, *_args, **_kwargs):
        batch = probs.shape[0]
        verify_scores = paddle.ones([batch, 1], dtype=probs.dtype)
        verify_tokens = paddle.zeros([batch, 1], dtype="int64")
        actual_len = paddle.to_tensor([1], dtype="int32")
        return verify_scores, verify_tokens, actual_len

    if not hasattr(gpu_mod, "speculate_verify"):
        _set_attr(gpu_mod, "speculate_verify", _speculate_verify, original_gpu_attrs)
    if not hasattr(gpu_mod, "top_p_candidates"):
        _set_attr(gpu_mod, "top_p_candidates", _top_p_candidates, original_gpu_attrs)

    def _fused_sampler(*args):
        pre_token_ids = args[0]
        tokens = paddle.arange(pre_token_ids.shape[0], dtype="int64").reshape([-1, 1])
        return None, tokens

    if not hasattr(hpu_mod, "fused_sampler"):
        _set_attr(hpu_mod, "fused_sampler", _fused_sampler, original_hpu_attrs)
    if not hasattr(hpu_mod, "get_token_penalty_multi_scores"):
        _set_attr(hpu_mod, "get_token_penalty_multi_scores", lambda *a, **k: a[3], original_hpu_attrs)

    yield records

    for name, previous in original_gpu_attrs.items():
        if previous is _MISSING:
            delattr(gpu_mod, name)
        else:
            setattr(gpu_mod, name, previous)
    for name, previous in original_hpu_attrs.items():
        if previous is _MISSING:
            delattr(hpu_mod, name)
        else:
            setattr(hpu_mod, name, previous)

    if created_gpu:
        sys.modules.pop("fastdeploy.model_executor.ops.gpu", None)
    if created_hpu:
        sys.modules.pop("fastdeploy.model_executor.ops.intel_hpu", None)


@pytest.fixture
def set_platform(monkeypatch):
    def _set(kind="cuda"):
        platform = _FakePlatform(kind)
        monkeypatch.setattr(sampler_module, "current_platform", platform)
        return platform

    return _set


@pytest.fixture
def fd_config_factory():
    class _Config:
        def __init__(self):
            self.model_config = types.SimpleNamespace(logprobs_mode="raw_logprobs")
            self.early_stop_config = None
            self.scheduler_config = types.SimpleNamespace(max_num_seqs=4)

    def _factory(**overrides):
        cfg = _Config()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    return _factory


def test_top_p_normalize_and_padding_params():
    probs = paddle.to_tensor([[0.6, 0.3, 0.1]], dtype="float32")
    normalized = sampler_module.top_p_normalize_probs_paddle(probs, paddle.to_tensor([0.5], dtype="float32"))
    assert pytest.approx(float(normalized.sum().item()), rel=1e-6) == 1.0

    top_p = paddle.to_tensor([[0.8], [0.6]], dtype="float32")
    top_k = paddle.to_tensor([[2], [3]], dtype="int64")
    seq_lens_this_time = paddle.to_tensor([[1], [2]], dtype="int64")
    seq_lens_encoder = paddle.to_tensor([[0], [1]], dtype="int64")
    infer_seed = paddle.to_tensor([[123], [456]], dtype="int64")
    padded_p, padded_k, padded_seed = sampler_module.padding_sampling_params(
        top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder
    )
    assert list(padded_p.shape) == [2, 1]
    assert list(padded_k.shape) == [2, 1]
    assert list(padded_seed.shape) == [padded_p.shape[0], 1]


def test_guided_decoding_tracks_processors_and_reasoning(fd_config_factory):
    guided = sampler_module.GuidedDecoding(fd_config_factory())
    reasoning_calls = []

    class _Reasoning:
        def is_reasoning_end(self, tokens):
            reasoning_calls.append(tokens)
            return tokens == [9]

    guided.apply_reasoning_parser(_Reasoning())
    mask_proc = _RecordingProcessor(enable_reasoning=False)
    mask_future = Future()
    mask_future.set_result(mask_proc)
    guided.add_logits_processor(0, mask_future, prefill_tokens=[1, 2])

    reason_proc = _RecordingProcessor(enable_reasoning=True)
    reason_future = Future()
    reason_future.set_result(reason_proc)
    guided.add_logits_processor(1, reason_future, prefill_tokens=[3])
    guided.pre_process()
    logits = paddle.ones([2, 3], dtype="float32")
    masked = guided.apply_token_mask(logits)
    assert getattr(mask_proc, "reasoning_ended", False) is False
    assert mask_proc.accepted_tokens == []
    assert reason_proc.accepted_tokens == [3]
    assert bool(paddle.allclose(masked, logits))

    next_tokens = paddle.to_tensor([[4], [9]], dtype="int64")
    guided.update_output_tokens(next_tokens)
    assert reasoning_calls[:2] == [[1], [2]]

    guided.add_logits_processor(0, None)
    assert guided.logits_processors[0] is None
    with pytest.raises(IndexError):
        guided._accept_token(99, 0)


def test_guided_decoding_prefill_and_bitmask(monkeypatch, fd_config_factory):
    guided = sampler_module.GuidedDecoding(fd_config_factory())

    class _Reasoning:
        def is_reasoning_end(self, tokens):
            return tokens == [2]

    guided.apply_reasoning_parser(_Reasoning())
    async_future = Future()
    guided.add_logits_processor(0, async_future, prefill_tokens=[1, 2])

    ready_future = Future()
    ready_future.set_result(_RecordingProcessor(enable_reasoning=True))
    guided.add_logits_processor(1, ready_future)
    guided.pre_process(prefill_done_idxs=[1])

    async_processor = _RecordingProcessor(enable_reasoning=True)
    async_future.set_result(async_processor)
    guided._tokens_to_acc[0] = None

    backend_mod = types.ModuleType("fastdeploy.model_executor.guided_decoding.xgrammar_backend")

    def _apply_token_mask(logits, token_bitmask, indices=None, is_cuda_platform=None):
        for idx in indices or []:
            token_bitmask[idx] += 1
        return logits

    backend_mod.apply_token_mask = _apply_token_mask
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.guided_decoding.xgrammar_backend", backend_mod)

    logits = paddle.ones([2, 3], dtype="float32")
    masked = guided.apply_token_mask(logits)

    assert guided.token_bitmask is not None
    assert async_processor.accepted_tokens == []
    assert paddle.allclose(masked, logits)


def test_sampler_compute_logprobs_handles_metadata(set_platform, fd_config_factory):
    set_platform("cuda")
    sampler = sampler_module.Sampler(fd_config=fd_config_factory())
    logits = paddle.zeros([2, 3], dtype="float32")
    assert list(sampler.compute_logprobs(logits).shape) == [2, 3]

    metadata = _build_sampling_metadata(
        temp_scaled_logprobs=_tensor([[True], [False]], dtype="bool"),
        top_p_normalized_logprobs=_tensor([[True], [False]], dtype="bool"),
    )
    result = sampler.compute_logprobs(logits, sampling_metadata=metadata)
    assert list(result.shape) == [2, 3]


def test_sampler_compute_logprobs_with_scaling(set_platform, fd_config_factory):
    set_platform("cuda")
    sampler = sampler_module.Sampler(fd_config=fd_config_factory())
    logits = paddle.ones([2, 3], dtype="float32")
    share_inputs = dict(
        seq_lens_this_time=_tensor([[1], [0]], dtype="int64"),
        seq_lens_encoder=_tensor([[1], [0]], dtype="int64"),
        seq_lens_decoder=_tensor([[1], [0]], dtype="int64"),
    )
    metadata = _build_sampling_metadata(
        temperature=_tensor([[2.0], [1.0]]),
        temp_scaled_logprobs=_tensor([[1], [0]], dtype="bool"),
        temp_scaled_logprobs_flag=True,
        top_p_normalized_logprobs=_tensor([[1], [0]], dtype="bool"),
        top_p_normalized_logprobs_flag=True,
        share_inputs=share_inputs,
        top_p=_tensor([[0.5], [1.0]]),
    )
    logprobs = sampler.compute_logprobs(logits, sampling_metadata=metadata)
    assert logprobs.shape == logits.shape


def test_sampler_gather_logprobs_variants(set_platform, fd_config_factory):
    set_platform("cuda")
    sampler = sampler_module.Sampler(fd_config=fd_config_factory())
    logprobs = paddle.to_tensor([[0.1, 0.2, 0.3]], dtype="float32")
    token_ids = paddle.to_tensor([2], dtype="int64")
    tensors = sampler.gather_logprobs(logprobs, 2, token_ids)
    assert isinstance(tensors, LogprobsTensors)
    tensors_zero = sampler.gather_logprobs(logprobs, 0, token_ids)
    assert list(tensors_zero.logprob_token_ids.shape) == [1, 1]


def test_sampler_forward_cuda_variants(monkeypatch, set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    cfg.early_stop_config = types.SimpleNamespace(enable_early_stop=True, strategy="mock")
    cfg.scheduler_config = types.SimpleNamespace(max_num_seqs=4)

    monkeypatch.setattr(
        sampler_module,
        "get_early_stopper_cls_from_stragegy",
        lambda *_: _FakeEarlyStopper,
    )
    sampler = sampler_module.Sampler(fd_config=cfg)
    sampler.guided_decoding = types.SimpleNamespace(apply_token_mask=lambda logits, *_: logits)

    metadata = _build_sampling_metadata()
    logits = paddle.randn([2, 4], dtype="float32")

    output = sampler.forward_cuda(logits, metadata)
    assert output.sampled_token_ids.shape[0] == metadata.share_inputs["seq_lens_this_time"].shape[0]
    assert sampler.early_stopper.process_calls

    sampler.logprobs_mode = "processed_logprobs"
    sampler.forward_cuda(logits, metadata)

    sampler.logprobs_mode = "processed_logits"
    metadata.max_num_logprobs = None
    sampler.forward_cuda(logits, metadata)


def test_sampler_forward_intel_hpu_path(set_platform, fd_config_factory):
    set_platform("intel_hpu")
    sampler = sampler_module.Sampler(fd_config=fd_config_factory())
    metadata = _build_sampling_metadata()
    batch_ids = paddle.to_tensor([0], dtype="int64")
    logits = paddle.randn([2, 3])
    tokens = sampler.forward_intel_hpu(logits, metadata, batch_ids, max_batch=2, rank=0, local_rank=0)
    assert tokens.shape[0] == 2


def test_sampler_raises_for_unsupported_platform(set_platform, fd_config_factory):
    set_platform("unknown")
    with pytest.raises(NotImplementedError):
        sampler_module.Sampler(fd_config=fd_config_factory())


def test_speculative_sampler_compute_and_forward(set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    cfg.model_config.logprobs_mode = "raw_logprobs"
    cfg.speculative_config = types.SimpleNamespace(verify_window=2, max_candidate_len=2, benchmark_mode=False)
    sampler = sampler_module.SpeculativeSampler(cfg)

    share_inputs = {
        "seq_lens_this_time": _tensor([[1]], dtype="int64"),
        "seq_lens_encoder": _tensor([[0]], dtype="int64"),
        "seq_lens_decoder": _tensor([[0]], dtype="int64"),
        "output_padding_offset": _tensor([[0]], dtype="int64"),
        "output_cum_offsets": _tensor([[0]], dtype="int64"),
        "accept_tokens": _tensor([[0, 0]], dtype="int64"),
        "accept_num": _tensor([1], dtype="int64"),
        "step_idx": _tensor([[0]], dtype="int64"),
        "stop_flags": _tensor([[False]], dtype="bool"),
        "draft_tokens": _tensor([[0, 0]], dtype="int64"),
        "max_dec_len": _tensor([[8]], dtype="int64"),
        "is_block_step": _tensor([[False]], dtype="bool"),
        "actual_draft_token_num": _tensor([[1]], dtype="int64"),
    }

    metadata = _build_sampling_metadata(share_inputs=share_inputs)
    logits = paddle.randn([1, 3], dtype="float32")

    logprobs = sampler.compute_logprobs(logits, metadata)
    assert logprobs.shape == logits.shape

    output = sampler.forward_cuda(logits, metadata, max_model_len=8, share_inputs=share_inputs)
    assert output.token_num_per_batch.equal(share_inputs["accept_num"])


def test_speculative_sampler_noop_hooks(set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    cfg.speculative_config = types.SimpleNamespace(verify_window=2, max_candidate_len=2, benchmark_mode=False)
    sampler = sampler_module.SpeculativeSampler(cfg)
    sampler.pre_process()
    sampler.set_reasoning_parser(None)
    sampler.post_process(paddle.zeros([1, 1]))
    sampler.apply_logits_processor(0, None)


def test_mtp_sampler_compute_and_forward(set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    mtp = sampler_module.MTPSampler(cfg)

    share_inputs = {
        "seq_lens_this_time": _tensor([[1]], dtype="int64"),
        "seq_lens_encoder": _tensor([[0]], dtype="int64"),
        "seq_lens_decoder": _tensor([[0]], dtype="int64"),
        "batch_token_num": _tensor([1], dtype="int64"),
        "draft_logits": paddle.randn([1, 3], dtype="float32"),
        "substep": 0,
        "accept_tokens": _tensor([[0, 0]], dtype="int64"),
        "cu_next_token_offset": _tensor([0], dtype="int64"),
        "cu_batch_token_offset": _tensor([0, 1], dtype="int32"),
        "output_padding_offset": _tensor([[0]], dtype="int64"),
        "output_cum_offsets": _tensor([[0]], dtype="int64"),
    }
    metadata = _build_sampling_metadata(share_inputs=share_inputs)
    logits = paddle.randn([1, 3], dtype="float32")
    mtp.compute_logprobs(logits, metadata)
    mtp.forward_cuda(logits, metadata, max_model_len=4, share_inputs=share_inputs)


def test_mtp_sampler_noop_hooks(set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    mtp = sampler_module.MTPSampler(cfg)
    mtp.pre_process()
    mtp.apply_logits_processor(0)
    mtp.set_reasoning_parser(None)
    mtp.post_process(paddle.zeros([1, 1]))


def test_mtp_sampler_prefill_and_reasoning(set_platform, fd_config_factory):
    set_platform("cuda")
    cfg = fd_config_factory()
    mtp = sampler_module.MTPSampler(cfg)
    mtp.set_reasoning_parser(lambda *_: None)
    processor = _AddOffsetProcessor(0.5)
    mtp.apply_logits_processor(0, processor)
    logits = paddle.zeros([1, 3], dtype="float32")
    share_inputs = {
        "seq_lens_this_time": _tensor([[1]], dtype="int64"),
        "seq_lens_encoder": _tensor([[0]], dtype="int64"),
        "seq_lens_decoder": _tensor([[0]], dtype="int64"),
        "batch_token_num": _tensor([1], dtype="int64"),
        "draft_logits": paddle.randn([1, 3], dtype="float32"),
        "substep": 0,
        "accept_tokens": _tensor([[0, 0]], dtype="int64"),
        "cu_next_token_offset": _tensor([0], dtype="int64"),
        "cu_batch_token_offset": _tensor([0, 1], dtype="int32"),
        "output_padding_offset": _tensor([[0]], dtype="int64"),
        "output_cum_offsets": _tensor([[0]], dtype="int64"),
    }
    metadata = _build_sampling_metadata(
        share_inputs=share_inputs,
        top_p_normalized_logprobs=_tensor([[1]], dtype="int64"),
        top_p=_tensor([[0.5]], dtype="float32"),
        top_k=_tensor([[2]], dtype="int64"),
        top_k_list=[2],
        temperature=_tensor([[1.0]], dtype="float32"),
    )
    logprobs = mtp.compute_logprobs(logits, metadata)
    assert logprobs.shape == logits.shape
    mtp.post_process(paddle.zeros([1, 1]))
