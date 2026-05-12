"""
Unit tests for the triton sampling path introduced in commit 16e692f.

Covers:
  - _apply_triton_top_k_top_p / apply_top_k_top_p_triton Python wrapper
  - _random_sample / seeded_gumbel_noise Python wrapper
  - Sampler.forward_cuda triton branch (FD_SAMPLING_CLASS="triton")
  - SpeculativeSampler triton branches
"""

import sys
import types

import paddle
import pytest

import fastdeploy  # noqa: F401

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda *args, **kwargs: None

# Stub triton for unit isolation (same pattern as test_sampler.py).
if "triton" not in sys.modules:
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_stub.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    triton_lang_stub = types.ModuleType("triton.language")
    triton_lang_stub.constexpr = int
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub

from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata

# Must import after stubs are in place.
from fastdeploy.model_executor.layers.sample.sampler import (
    Sampler,
    SpeculativeSampler,
    _apply_triton_top_k_top_p,
    _random_sample,
)
from fastdeploy.spec_decode import VerifyStrategy

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_gpu_deps(monkeypatch):
    """Patch only GPU-specific calls so Python wrapper code can execute on CPU."""
    import fastdeploy.model_executor.layers.sample.ops.top_k_top_p_triton as triton_mod

    # Patch the kernel launch inside apply_top_k_top_p_triton: replace
    # _topk_topp_kernel so it becomes a no-op (logits left unchanged →
    # equivalent to "keep all" when no real GPU masking happens).
    # This lets the Python wrapper (lines 830-936) run for coverage.
    def _fake_kernel_call(grid, kwargs):
        pass

    monkeypatch.setattr(triton_mod._topk_topp_kernel, "__call__", _fake_kernel_call)

    # Patch paddle.device.cuda.get_device_properties used inside
    # apply_top_k_top_p_triton to avoid "no CUDA device" error.
    fake_props = types.SimpleNamespace(multi_processor_count=1)
    monkeypatch.setattr(
        paddle.device.cuda,
        "get_device_properties",
        lambda idx: fake_props,
    )

    # Patch _seeded_gumbel_kernel similarly so seeded_gumbel_noise (lines
    # 960-981) runs its Python logic without real GPU.
    def _fake_gumbel_kernel_call(grid, kwargs):
        pass

    monkeypatch.setattr(triton_mod._seeded_gumbel_kernel, "__call__", _fake_gumbel_kernel_call)

    # Patch batched_count_greater_than (used in gather_logprobs).
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.batched_count_greater_than",
        lambda x, y: (x >= y).sum(-1),
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.logprobs.batched_count_greater_than",
        lambda x, y: (x >= y).sum(-1),
    )

    # Patch current_platform so is_cuda() returns True (needed for
    # build_sampling_params import).
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.current_platform.is_xpu",
        lambda: False,
    )


@pytest.fixture
def mock_ops(monkeypatch):
    """Patch heavy GPU ops that are not the focus of triton tests."""
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_penalty_multi_scores",
        lambda *a, **k: a[1],
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.apply_speculative_penalty_multi_scores",
        lambda *a, **k: a[2],
    )
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.sample.sampler.min_p_sampling",
        lambda probs, *a, **k: probs,
    )
    return monkeypatch


@pytest.fixture
def triton_mode(monkeypatch):
    """Set FD_SAMPLING_CLASS to triton for the duration of the test."""
    import fastdeploy.envs as envs

    monkeypatch.setattr(envs, "FD_SAMPLING_CLASS", "triton")


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


# ---------------------------------------------------------------------------
# Tests for _apply_triton_top_k_top_p (direct call)
# ---------------------------------------------------------------------------


class TestApplyTritonTopKTopP:
    """Tests for _apply_triton_top_k_top_p."""

    def test_returns_logits_unchanged_when_both_none(self):
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=None, top_k=None)
        assert paddle.equal_all(result, logits)

    def test_top_p_only_no_error(self):
        """top_p filtering runs through apply_top_k_top_p_triton wrapper."""
        logits = paddle.to_tensor([[1.0, 2.0, 5.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.7]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p)
        assert result.shape == [1, 3]

    def test_top_k_disabled_when_list_none(self):
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[1.0]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=None, top_k_list=None)
        assert result.shape == [1, 3]

    def test_return_mask_false(self):
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.9]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, return_mask=False)
        assert isinstance(result, paddle.Tensor)

    def test_return_mask_true(self):
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.5]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, return_mask=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        logits_out, mask = result
        assert logits_out.shape == [1, 3]
        assert mask.shape == [1, 3]
        assert mask.dtype == paddle.bool

    def test_output_dtype_is_float32(self):
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float16")
        top_p = paddle.to_tensor([[0.9]], dtype="float32")
        result = _apply_triton_top_k_top_p(logits, top_p=top_p)
        assert result.dtype == paddle.float32

    def test_combined_top_k_top_p(self):
        logits = paddle.to_tensor([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype="float32")
        top_p = paddle.to_tensor([[0.5]], dtype="float32")
        top_k = paddle.to_tensor([[3]], dtype="int64")
        top_k_list = [3]
        result = _apply_triton_top_k_top_p(logits, top_p=top_p, top_k=top_k, top_k_list=top_k_list)
        assert result.shape == [1, 5]


# ---------------------------------------------------------------------------
# Tests for _random_sample (direct call)
# ---------------------------------------------------------------------------


class TestRandomSample:
    """Tests for _random_sample."""

    def test_output_shape_and_dtype(self):
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], dtype="float32")
        result = _random_sample(probs)
        assert result.shape == [2, 1]
        assert result.dtype == paddle.int64

    def test_without_seed(self):
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        result = _random_sample(probs, topp_seed=None)
        assert 0 <= result[0, 0].item() < 3

    def test_with_seed(self):
        probs = paddle.to_tensor([[0.1, 0.2, 0.7]], dtype="float32")
        seed = paddle.to_tensor([[42]], dtype="int64")
        result = _random_sample(probs, topp_seed=seed)
        assert result.shape == [1, 1]

    def test_greedy_with_peak_distribution(self):
        probs = paddle.zeros([1, 10], dtype="float32")
        probs[0, 5] = 1.0
        result = _random_sample(probs)
        assert result[0, 0].item() == 5

    def test_batch_multiple_requests(self):
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 1.0]], dtype="float32")
        result = _random_sample(probs)
        assert result.shape == [2, 1]
        assert 0 <= result[0, 0].item() < 3
        assert result[1, 0].item() == 2


# ---------------------------------------------------------------------------
# Tests for Sampler.forward_cuda with triton path
# ---------------------------------------------------------------------------


class TestSamplerTritonPath:
    """Test Sampler.forward_cuda with FD_SAMPLING_CLASS=triton."""

    def test_forward_cuda_triton_path(self, mock_ops, triton_mode):
        """Sampler.forward_cuda should call _apply_triton_top_k_top_p and _random_sample."""
        sampler = _make_stubbed_sampler("processed_logprobs")
        m = _create_metadata(batch_size=1, max_num_logprobs=2)

        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        output = sampler.forward_cuda(logits, m)
        assert output.sampled_token_ids.shape == [1, 1]
        assert output.logprobs_tensors is not None


# ---------------------------------------------------------------------------
# Tests for SpeculativeSampler triton branches
# ---------------------------------------------------------------------------


def _make_spec_sampler(verify_strategy=VerifyStrategy.TARGET_MATCH, spec_method=None):
    """Create a SpeculativeSampler with stubbed internals."""
    s = SpeculativeSampler.__new__(SpeculativeSampler)
    s.verify_strategy = verify_strategy
    s.spec_method = spec_method  # None → NAIVE path
    s.enf_gen_phase_tag = False
    s.config_accept_all = False
    s.config_reject_all = False
    s.speculative_benchmark_mode = False
    s.speculative_max_candidate_len = 1
    s.speculative_verify_window = 2
    s.think_end_id = 1
    s.line_break_id = 2
    s.logprobs_mode = "processed_logprobs"
    return s


def _spec_share_inputs(batch_size=1):
    return {
        "seq_lens_this_time": paddle.ones([batch_size, 1], dtype="int64"),
        "seq_lens_encoder": paddle.zeros([batch_size, 1], dtype="int64"),
        "cu_seqlens_q_output": paddle.to_tensor([0] + [1] * batch_size, dtype="int32"),
        "batch_id_per_token_output": paddle.zeros([batch_size], dtype="int32"),
        "accept_tokens": paddle.zeros([batch_size, 1], dtype="int64"),
        "accept_num": paddle.zeros([batch_size], dtype="int32"),
        "draft_tokens": paddle.zeros([batch_size, 1], dtype="int64"),
        "stop_flags": paddle.zeros([batch_size, 1], dtype="int32"),
        "is_block_step": paddle.zeros([batch_size], dtype="int32"),
        "reasoning_status": paddle.zeros([batch_size, 1], dtype="int32"),
        "max_dec_len": paddle.full([batch_size, 1], 1024, dtype="int64"),
        "step_idx": paddle.zeros([batch_size, 1], dtype="int64"),
    }


class TestSpeculativeSamplerTritonPath:
    """Test SpeculativeSampler triton branches (lines 916, 1016-1017, 1120-1132)."""

    def test_verify_and_sample_target_match_triton(self, mock_ops, triton_mode, monkeypatch):
        """_verify_and_sample with TARGET_MATCH + triton → calls _random_sample (line 916)."""
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.sample.sampler.build_sampling_params",
            lambda *a, **k: (
                paddle.to_tensor([[0.9]], dtype="float32"),
                paddle.to_tensor([[5]], dtype="int64"),
                paddle.to_tensor([[7]], dtype="int64"),
            ),
        )
        # verify_draft_tokens is lazily imported inside _verify_and_sample
        import fastdeploy.model_executor.ops.gpu as gpu_ops

        monkeypatch.setattr(gpu_ops, "verify_draft_tokens", lambda *a, **k: None)
        monkeypatch.setattr(gpu_ops, "top_p_candidates", lambda *a, **k: (None, None, None))

        sampler = _make_spec_sampler(verify_strategy=VerifyStrategy.TARGET_MATCH, spec_method="ngram")
        m = _create_metadata(batch_size=1)
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        probs = paddle.nn.functional.softmax(logits, axis=-1)

        out = sampler._verify_and_sample(
            logits,
            probs,
            m,
            max_model_len=8,
            share_inputs=_spec_share_inputs(),
            token_num_output_cpu=1,
            increment_value=1,
        )
        assert out.sampled_token_ids is not None

    def test_normal_sample_triton(self, mock_ops, triton_mode, monkeypatch):
        """_normal_sample with triton → calls _random_sample (line 1016-1017)."""
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.sample.sampler.naive_update_model_status",
            lambda *a, **k: None,
        )

        sampler = _make_spec_sampler(spec_method=None)  # None → NAIVE
        m = _create_metadata(batch_size=1)
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        probs = paddle.nn.functional.softmax(logits, axis=-1)

        out = sampler._normal_sample(logits, probs, m, share_inputs=_spec_share_inputs())
        assert out.sampled_token_ids is not None

    def test_forward_cuda_triton_logit_mask(self, mock_ops, triton_mode, monkeypatch):
        """SpeculativeSampler.forward_cuda with triton → masks logits (lines 1120-1132)."""
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.sample.sampler.build_sampling_params",
            lambda *a, **k: (
                paddle.to_tensor([[0.9]], dtype="float32"),
                paddle.to_tensor([[5]], dtype="int64"),
                paddle.to_tensor([[7]], dtype="int64"),
            ),
        )
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.sample.sampler.naive_update_model_status",
            lambda *a, **k: None,
        )

        sampler = _make_spec_sampler(spec_method=None)  # NAIVE → _normal_sample
        m = _create_metadata(batch_size=1)
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")

        out = sampler.forward_cuda(
            logits,
            m,
            max_model_len=8,
            share_inputs=_spec_share_inputs(),
            token_num_output_cpu=1,
            increment_value=1,
        )
        assert out.sampled_token_ids is not None


# ---------------------------------------------------------------------------
# Tests for triton Python wrapper functions (top_k_top_p_triton.py coverage)
# ---------------------------------------------------------------------------


class TestTritonWrapperFunctions:
    """Cover the Python wrapper functions in top_k_top_p_triton.py."""

    def test_reset_buffer_cache(self, monkeypatch):
        """reset_buffer_cache should run without error."""
        from fastdeploy.model_executor.layers.sample.ops.top_k_top_p_triton import (
            reset_buffer_cache,
        )

        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.sample.ops.top_k_top_p_triton.paddle.accelerator",
            types.SimpleNamespace(empty_cache=lambda: None),
            raising=False,
        )
        reset_buffer_cache()


if __name__ == "__main__":
    pytest.main([__file__])
