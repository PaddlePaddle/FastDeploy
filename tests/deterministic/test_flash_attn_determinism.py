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
Unit tests for flash_attn_func deterministic mode.

Verifies that flash_attn_func passes correct deterministic parameters
(e.g. num_splits=1 for FA3) when FD_DETERMINISTIC_MODE=1.

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_flash_attn_determinism.py -v
"""

import importlib
import os

import numpy as np
import paddle
import pytest

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_HEADS = 8
KV_NUM_HEADS = 8
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    return prop.major * 10 + prop.minor


def _reload_flash_attn_backend():
    """Reload flash_attn_backend so env-var changes take effect."""
    import fastdeploy.model_executor.layers.attention.flash_attn_backend as mod

    importlib.reload(mod)
    return mod


def _make_tensors(seq_lens, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
    """Create Q/K/V tensors and cu_seqlens for a batch of sequences."""
    total_tokens = sum(seq_lens)
    q = paddle.randn([total_tokens, num_heads, head_dim], dtype="bfloat16")
    k = paddle.randn([total_tokens, num_heads, head_dim], dtype="bfloat16")
    v = paddle.randn([total_tokens, num_heads, head_dim], dtype="bfloat16")
    cu_seqlens = paddle.to_tensor(np.array([0] + list(np.cumsum(seq_lens))), dtype="int32")
    max_seqlen = max(seq_lens)
    return q, k, v, cu_seqlens, max_seqlen


def _call_flash_attn_func(mod, q, k, v, cu_seqlens, max_seqlen, version=None):
    """Call flash_attn_func and return the output tensor."""
    result = mod.flash_attn_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        num_heads=NUM_HEADS,
        kv_num_heads=KV_NUM_HEADS,
        head_dim=HEAD_DIM,
        version=version,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


def _run_determinism_check(mod, seq_lens, runs, version, test_name):
    """Run flash_attn_func multiple times and verify deterministic output."""
    q, k, v, cu_seqlens, max_seqlen = _make_tensors(seq_lens)

    outputs = []
    for _ in range(runs):
        out = _call_flash_attn_func(mod, q, k, v, cu_seqlens, max_seqlen, version=version)
        outputs.append(out.numpy())

    for i in range(1, runs):
        assert np.array_equal(outputs[0], outputs[i]), f"{test_name}: run {i} differs from run 0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env():
    """Save and restore determinism-related env vars around every test."""
    keys = ["FD_DETERMINISTIC_MODE", "FD_DETERMINISTIC_DEBUG"]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def _deterministic_mode_enabled():
    """Enable deterministic mode and return reloaded module."""
    os.environ["FD_DETERMINISTIC_MODE"] = "1"
    return _reload_flash_attn_backend()


@pytest.fixture
def _nondeterministic_mode_enabled():
    """Disable deterministic mode and return reloaded module."""
    os.environ["FD_DETERMINISTIC_MODE"] = "0"
    return _reload_flash_attn_backend()


# ---------------------------------------------------------------------------
# Tests: _is_deterministic_mode
# ---------------------------------------------------------------------------


class TestIsDeterministicMode:
    """Test the _is_deterministic_mode helper."""

    def test_enabled(self):
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        mod = _reload_flash_attn_backend()
        assert mod._is_deterministic_mode() is True

    def test_disabled(self):
        os.environ["FD_DETERMINISTIC_MODE"] = "0"
        mod = _reload_flash_attn_backend()
        assert mod._is_deterministic_mode() is False

    def test_unset_defaults_false(self):
        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        mod = _reload_flash_attn_backend()
        assert mod._is_deterministic_mode() is False


# ---------------------------------------------------------------------------
# Tests: FA3 determinism (requires SM89+, <SM100)
# ---------------------------------------------------------------------------


class TestFA3Determinism:
    """Test FA3 deterministic behavior with num_splits control."""

    @pytest.fixture(autouse=True)
    def _require_fa3(self):
        sm = _get_sm_version()
        if sm < 89 or sm >= 100:
            pytest.skip(f"FA3 requires SM89-99, current SM={sm}")
        paddle.set_flags({"FLAGS_flash_attn_version": 3})

    def test_deterministic_produces_identical_output(self, _deterministic_mode_enabled):
        """num_splits=1 (deterministic) gives bitwise identical results."""
        _run_determinism_check(_deterministic_mode_enabled, [64, 128, 256], 5, 3, "FA3 deterministic")

    def test_long_sequence_determinism(self, _deterministic_mode_enabled):
        """Long sequences (>1024 tokens) remain deterministic with FA3."""
        _run_determinism_check(_deterministic_mode_enabled, [2048], 3, 3, "FA3 long seq")

    def test_mixed_batch_determinism(self, _deterministic_mode_enabled):
        """Mixed batch with varying sequence lengths stays deterministic."""
        _run_determinism_check(_deterministic_mode_enabled, [16, 512, 1024, 64], 3, 3, "FA3 mixed batch")

    def test_nondeterministic_mode_also_works(self, _nondeterministic_mode_enabled):
        """FD_DETERMINISTIC_MODE=0 still works (num_splits=1 is always used)."""
        q, k, v, cu_seqlens, max_seqlen = _make_tensors([256])
        out = _call_flash_attn_func(_nondeterministic_mode_enabled, q, k, v, cu_seqlens, max_seqlen, version=3)
        assert out.shape[0] == 256
        assert out.shape[1] == NUM_HEADS
        assert out.shape[2] == HEAD_DIM


# ---------------------------------------------------------------------------
# Tests: FA2 determinism
# ---------------------------------------------------------------------------


class TestFA2Determinism:
    """Test FA2 deterministic behavior (inherently deterministic forward)."""

    @pytest.fixture(autouse=True)
    def _set_fa2(self):
        paddle.set_flags({"FLAGS_flash_attn_version": 2})

    def test_fa2_deterministic(self, _deterministic_mode_enabled):
        """FA2 forward is inherently deterministic (no split-KV)."""
        _run_determinism_check(_deterministic_mode_enabled, [128, 256], 5, 2, "FA2 deterministic")

    def test_fa2_long_sequence(self, _deterministic_mode_enabled):
        """FA2 with long sequence remains deterministic."""
        _run_determinism_check(_deterministic_mode_enabled, [2048], 3, 2, "FA2 long seq")
