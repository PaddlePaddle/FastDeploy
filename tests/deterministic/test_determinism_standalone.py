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
Determinism unit tests (lightweight, no model loading required)

Test scenarios:
1. SamplingParams seed behavior in deterministic / non-deterministic mode
2. Environment variable handling (FD_DETERMINISTIC_MODE, SPLIT_KV_SIZE, LOG_MODE)
3. Token allocation alignment logic (_get_num_new_tokens)
4. Cross-mode behavior validation

Usage:
    pytest tests/deterministic/test_determinism_standalone.py -v
"""

import importlib
import os
from dataclasses import dataclass
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_sp():
    """Reload envs + sampling_params so env-var changes take effect."""
    import fastdeploy.engine.sampling_params as sp_module
    import fastdeploy.envs as envs_module

    importlib.reload(envs_module)
    importlib.reload(sp_module)
    return sp_module, envs_module


@dataclass
class _FakeRequest:
    """Minimal stand-in for a scheduler request object."""

    need_prefill_tokens: int
    num_computed_tokens: int
    request_id: str = "fake-0"
    prompt_token_ids: Optional[list] = None
    multimodal_inputs: Optional[dict] = None
    with_image: bool = False


def _align_tokens(current_pos, remaining, budget, split_kv_size):
    """
    Pure-function replica of the alignment logic in
    ResourceManagerV1._get_num_new_tokens (deterministic branch).

    Returns the number of new tokens to allocate.
    """
    if remaining < split_kv_size:
        # Final chunk - no alignment needed
        return min(remaining, budget)

    # Next split_kv_size boundary from current_pos
    next_boundary = ((current_pos + split_kv_size - 1) // split_kv_size) * split_kv_size
    tokens_to_boundary = next_boundary - current_pos

    if budget < tokens_to_boundary:
        return 0  # defer

    aligned_end = ((current_pos + budget) // split_kv_size) * split_kv_size
    num_new = aligned_end - current_pos
    return min(num_new, budget, remaining)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env():
    """Save and restore determinism-related env vars around every test."""
    keys = [
        "FD_DETERMINISTIC_MODE",
        "FD_DETERMINISTIC_SPLIT_KV_SIZE",
        "FD_DETERMINISTIC_LOG_MODE",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _set_env(key, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


# ===================== SamplingParams seed tests =====================


class TestSamplingParamsSeed:
    """Verify seed assignment in SamplingParams under different modes."""

    def test_non_deterministic_uses_random_seed(self):
        """Without FD_DETERMINISTIC_MODE, each SamplingParams gets a random seed."""
        _set_env("FD_DETERMINISTIC_MODE", None)
        sp_mod, _ = _reload_sp()

        seeds = {sp_mod.SamplingParams().seed for _ in range(10)}
        assert len(seeds) > 1, "Non-deterministic mode should produce different random seeds"

    def test_deterministic_uses_fixed_seed(self):
        """With FD_DETERMINISTIC_MODE=1, default seed is always 42."""
        _set_env("FD_DETERMINISTIC_MODE", "1")
        sp_mod, _ = _reload_sp()

        seeds = {sp_mod.SamplingParams().seed for _ in range(10)}
        assert seeds == {42}, f"Deterministic mode should always use seed=42, got {seeds}"

    def test_explicit_seed_overrides_mode(self):
        """User-supplied seed takes precedence over deterministic default."""
        _set_env("FD_DETERMINISTIC_MODE", "1")
        sp_mod, _ = _reload_sp()

        assert sp_mod.SamplingParams(seed=123).seed == 123

    def test_seed_zero_is_valid(self):
        """seed=0 must not be confused with 'unset'."""
        _set_env("FD_DETERMINISTIC_MODE", "1")
        sp_mod, _ = _reload_sp()

        assert sp_mod.SamplingParams(seed=0).seed == 0

    def test_seed_max_value(self):
        """Upper-bound seed accepted by _verify_args."""
        _set_env("FD_DETERMINISTIC_MODE", "1")
        sp_mod, _ = _reload_sp()

        max_seed = 922337203685477580
        assert sp_mod.SamplingParams(seed=max_seed).seed == max_seed

    def test_explicit_seed_works_in_both_modes(self):
        """Same explicit seed yields same value regardless of mode."""
        explicit_seed = 12345
        for mode in ("0", "1"):
            _set_env("FD_DETERMINISTIC_MODE", mode)
            sp_mod, _ = _reload_sp()
            assert sp_mod.SamplingParams(seed=explicit_seed).seed == explicit_seed


# ===================== Environment variable tests =====================


class TestDeterminismEnvVars:
    """Verify env-var parsing in fastdeploy.envs."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, False),
            ("0", False),
            ("1", True),
        ],
    )
    def test_deterministic_mode(self, raw, expected):
        _set_env("FD_DETERMINISTIC_MODE", raw)
        _, envs_mod = _reload_sp()
        assert envs_mod.FD_DETERMINISTIC_MODE is expected

    def test_split_kv_size_default(self):
        _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)
        _, envs_mod = _reload_sp()
        assert envs_mod.FD_DETERMINISTIC_SPLIT_KV_SIZE == 16

    def test_split_kv_size_custom(self):
        _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "32")
        _, envs_mod = _reload_sp()
        assert envs_mod.FD_DETERMINISTIC_SPLIT_KV_SIZE == 32

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, False),
            ("1", True),
        ],
    )
    def test_log_mode(self, raw, expected):
        _set_env("FD_DETERMINISTIC_LOG_MODE", raw)
        _, envs_mod = _reload_sp()
        assert envs_mod.FD_DETERMINISTIC_LOG_MODE is expected


# ===================== Token alignment logic tests =====================


class TestTokenAlignment:
    """
    Verify the deterministic token-alignment algorithm.

    The alignment logic ensures chunk boundaries fall on split_kv_size
    multiples so that attention computation is batch-invariant.
    """

    @pytest.mark.parametrize(
        "cur,remaining,budget,kv,expected",
        [
            # --- basic cases (cur=0) ---
            (0, 100, 5, 16, 0),  # budget < kv_size, defer
            (0, 100, 16, 16, 16),  # budget == kv_size
            (0, 100, 32, 16, 32),  # budget == 2*kv_size
            (0, 100, 50, 16, 48),  # round-down to 48
            # --- non-zero current_pos ---
            (10, 90, 20, 16, 6),  # next boundary=16, then end=16, alloc=6
            (8, 92, 20, 16, 8),  # next boundary=16, aligned_end=16, alloc=8
            (16, 84, 32, 16, 32),  # already on boundary
            (15, 85, 1, 16, 1),  # exactly 1 token to next boundary
            (17, 83, 2, 16, 0),  # 15 tokens to boundary=32, budget=2 => defer
            # --- final-chunk (remaining < kv_size) ---
            (96, 4, 10, 16, 4),  # final chunk, no alignment
            (96, 4, 2, 16, 2),  # final chunk, budget < remaining
            # --- large kv_size ---
            (0, 200, 100, 64, 64),  # kv=64, 100//64*64=64
            (0, 200, 128, 64, 128),  # kv=64, 128//64*64=128
        ],
    )
    def test_align_tokens(self, cur, remaining, budget, kv, expected):
        result = _align_tokens(cur, remaining, budget, kv)
        assert result == expected, (
            f"align_tokens(cur={cur}, remaining={remaining}, budget={budget}, kv={kv}): "
            f"expected {expected}, got {result}"
        )

    def test_alignment_vs_non_deterministic(self):
        """Deterministic mode allocates fewer tokens due to alignment."""
        budget, kv = 50, 16
        det_result = _align_tokens(0, 100, budget, kv)  # 48
        non_det_result = min(100, budget)  # 50
        assert det_result < non_det_result
        assert det_result == 48
        assert non_det_result == 50

    def test_result_always_on_boundary_or_final_allocation(self):
        """After allocation, (current_pos + result) sits on a kv boundary
        unless this allocation exhausts all remaining tokens."""
        kv = 16
        for cur in range(0, 80, 7):
            for remaining in [5, 10, 30, 60, 100]:
                for budget in [1, 8, 16, 32, 64]:
                    result = _align_tokens(cur, remaining, budget, kv)
                    if result == 0:
                        continue
                    end = cur + result
                    is_final = result == remaining
                    if remaining >= kv and not is_final:
                        assert end % kv == 0, (
                            f"cur={cur} remaining={remaining} budget={budget}: " f"end={end} is not aligned to {kv}"
                        )


# ===================== Cross-mode behavior validation =====================


class TestCrossModeBehavior:
    """Prove that mode switch actually changes observable behavior."""

    def test_deterministic_mode_consistent_seeds(self):
        _set_env("FD_DETERMINISTIC_MODE", "1")
        sp_mod, _ = _reload_sp()
        seeds = [sp_mod.SamplingParams().seed for _ in range(10)]
        assert len(set(seeds)) == 1 and seeds[0] == 42

    def test_non_deterministic_mode_varied_seeds(self):
        _set_env("FD_DETERMINISTIC_MODE", "0")
        sp_mod, _ = _reload_sp()
        seeds = [sp_mod.SamplingParams().seed for _ in range(10)]
        assert len(set(seeds)) > 1


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
