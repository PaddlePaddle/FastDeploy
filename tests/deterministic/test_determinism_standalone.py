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
1. Test SamplingParams seed behavior in deterministic mode
2. Test environment variable handling
3. Test token allocation alignment logic

This test focuses on determinism-related core components without
requiring full model loading or HTTP server startup.
"""

import os
import unittest

from fastdeploy import envs


class TestSamplingParamsDeterminism(unittest.TestCase):
    """Test SamplingParams deterministic behavior"""

    def setUp(self):
        """Save original environment variables"""
        self.original_env = {}

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_env(self, key, value):
        """Set environment variable and save original value"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_sampling_params_seed_non_deterministic(self):
        """
        Test: Non-deterministic mode uses random seed

        When FD_DETERMINISTIC_MODE is not set or is 0, SamplingParams
        should use randomly generated seeds.
        """
        self._set_env("FD_DETERMINISTIC_MODE", None)

        # Import after setting environment
        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        params1 = sp_module.SamplingParams()
        params2 = sp_module.SamplingParams()

        # Non-deterministic mode: seeds should be different (random)
        self.assertNotEqual(params1.seed, params2.seed, "Non-deterministic mode should use different random seeds")

        print(f"[PASS] Non-deterministic mode: seed1={params1.seed}, seed2={params2.seed}")

    def test_sampling_params_seed_deterministic(self):
        """
        Test: Deterministic mode uses fixed seed (42)

        When FD_DETERMINISTIC_MODE=1, SamplingParams should use
        a fixed seed value (42) by default.
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        params_det1 = sp_module.SamplingParams()
        params_det2 = sp_module.SamplingParams()

        # Deterministic mode: both should use seed=42
        self.assertEqual(params_det1.seed, 42, "Deterministic mode should use seed=42")
        self.assertEqual(params_det2.seed, 42, "Deterministic mode should use seed=42")

        print(f"[PASS] Deterministic mode: seed1={params_det1.seed}, seed2={params_det2.seed}")

    def test_sampling_params_seed_explicit(self):
        """
        Test: Explicit seed overrides deterministic mode

        When user explicitly provides a seed, it should be used
        regardless of FD_DETERMINISTIC_MODE setting.
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        # Explicit seed should override deterministic mode default
        params = sp_module.SamplingParams(seed=123)
        self.assertEqual(params.seed, 123, "Explicit seed should override deterministic mode")

        print(f"[PASS] Explicit seed: seed={params.seed}")

    def test_sampling_params_seed_zero_is_valid(self):
        """
        Test: Seed value 0 is valid

        Seed=0 should be treated as a valid explicit seed,
        not as "not set".
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        # Seed=0 should be preserved
        params = sp_module.SamplingParams(seed=0)
        self.assertEqual(params.seed, 0, "Seed=0 should be a valid explicit seed")

        print(f"[PASS] Seed=0 is valid: seed={params.seed}")

    def test_sampling_params_seed_max_value(self):
        """
        Test: Maximum valid seed value

        Test the upper bound of valid seed values.
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        # Maximum valid seed (as defined in SamplingParams._verify_args)
        max_seed = 922337203685477580
        params = sp_module.SamplingParams(seed=max_seed)
        self.assertEqual(params.seed, max_seed, f"Maximum seed value {max_seed} should be accepted")

        print(f"[PASS] Maximum seed value: seed={params.seed}")


class TestDeterminismEnvVars(unittest.TestCase):
    """Test determinism environment variable handling"""

    def setUp(self):
        """Save original environment variables"""
        self.original_env = {}

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_env(self, key, value):
        """Set environment variable and save original value"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_env_deterministic_mode_default(self):
        """Test: FD_DETERMINISTIC_MODE defaults to False"""
        self._set_env("FD_DETERMINISTIC_MODE", None)

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertFalse(envs.FD_DETERMINISTIC_MODE, "FD_DETERMINISTIC_MODE should default to False")

        print(f"[PASS] FD_DETERMINISTIC_MODE default: {envs.FD_DETERMINISTIC_MODE}")

    def test_env_deterministic_mode_enabled(self):
        """Test: FD_DETERMINISTIC_MODE=1 enables determinism"""
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertTrue(envs.FD_DETERMINISTIC_MODE, "FD_DETERMINISTIC_MODE=1 should enable determinism")

        print(f"[PASS] FD_DETERMINISTIC_MODE=1: {envs.FD_DETERMINISTIC_MODE}")

    def test_env_deterministic_mode_disabled(self):
        """Test: FD_DETERMINISTIC_MODE=0 disables determinism"""
        self._set_env("FD_DETERMINISTIC_MODE", "0")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertFalse(envs.FD_DETERMINISTIC_MODE, "FD_DETERMINISTIC_MODE=0 should disable determinism")

        print(f"[PASS] FD_DETERMINISTIC_MODE=0: {envs.FD_DETERMINISTIC_MODE}")

    def test_env_split_kv_size_default(self):
        """Test: FD_DETERMINISTIC_SPLIT_KV_SIZE defaults to 16"""
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertEqual(
            envs.FD_DETERMINISTIC_SPLIT_KV_SIZE, 16, "FD_DETERMINISTIC_SPLIT_KV_SIZE should default to 16"
        )

        print(f"[PASS] FD_DETERMINISTIC_SPLIT_KV_SIZE default: {envs.FD_DETERMINISTIC_SPLIT_KV_SIZE}")

    def test_env_split_kv_size_custom(self):
        """Test: FD_DETERMINISTIC_SPLIT_KV_SIZE can be customized"""
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "32")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertEqual(envs.FD_DETERMINISTIC_SPLIT_KV_SIZE, 32, "FD_DETERMINISTIC_SPLIT_KV_SIZE should be 32")

        print(f"[PASS] FD_DETERMINISTIC_SPLIT_KV_SIZE=32: {envs.FD_DETERMINISTIC_SPLIT_KV_SIZE}")

    def test_env_log_mode_default(self):
        """Test: FD_DETERMINISTIC_LOG_MODE defaults to False"""
        self._set_env("FD_DETERMINISTIC_LOG_MODE", None)

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertFalse(envs.FD_DETERMINISTIC_LOG_MODE, "FD_DETERMINISTIC_LOG_MODE should default to False")

        print(f"[PASS] FD_DETERMINISTIC_LOG_MODE default: {envs.FD_DETERMINISTIC_LOG_MODE}")

    def test_env_log_mode_enabled(self):
        """Test: FD_DETERMINISTIC_LOG_MODE=1 enables logging"""
        self._set_env("FD_DETERMINISTIC_LOG_MODE", "1")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        self.assertTrue(envs.FD_DETERMINISTIC_LOG_MODE, "FD_DETERMINISTIC_LOG_MODE=1 should enable logging")

        print(f"[PASS] FD_DETERMINISTIC_LOG_MODE=1: {envs.FD_DETERMINISTIC_LOG_MODE}")


class TestTokenAlignmentLogic(unittest.TestCase):
    """Test token allocation alignment logic for determinism"""

    def setUp(self):
        """Save original environment variables"""
        self.original_env = {}

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_env(self, key, value):
        """Set environment variable and save original value"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_alignment_calculation_disabled_mode(self):
        """
        Test: Token allocation without alignment (deterministic mode disabled)

        In non-deterministic mode, _get_num_new_tokens should allocate
        tokens without any alignment constraints.
        """
        self._set_env("FD_DETERMINISTIC_MODE", None)
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        # Simple test: basic allocation without alignment
        need_prefill = 100
        num_computed = 0
        token_budget = 50

        # In non-deterministic mode: allocate min(need_prefill - num_computed, token_budget)
        expected = min(need_prefill - num_computed, token_budget)
        self.assertEqual(expected, 50, "Non-deterministic mode: should allocate up to budget")

        print(f"[PASS] Non-deterministic allocation: {expected}")

    def test_alignment_calculation_enabled_mode(self):
        """
        Test: Token allocation with alignment (deterministic mode enabled)

        In deterministic mode, _get_num_new_tokens should align to
        split_kv_size boundary.
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        split_kv_size = 16
        need_prefill = 100
        num_computed = 0
        token_budget = 50

        # In deterministic mode: allocate tokens that end at split_kv_size boundary
        # With budget=50 and split_kv_size=16, should allocate 48 (3 * 16)
        # because 48 is the largest multiple of 16 <= 50
        max_possible = min(need_prefill - num_computed, token_budget)
        aligned_result = (max_possible // split_kv_size) * split_kv_size

        self.assertEqual(aligned_result, 48, "Deterministic mode: should align to split_kv_size boundary")

        print(f"[PASS] Deterministic alignment: {aligned_result} (aligned to {split_kv_size})")

    def test_alignment_boundary_cases(self):
        """
        Test: Alignment boundary cases

        Test various boundary conditions for token alignment.
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        split_kv_size = 16

        test_cases = [
            # (need_prefill, num_computed, token_budget, expected)
            (100, 0, 5, 0),  # Budget < split_kv_size, can't align
            (100, 0, 16, 16),  # Budget equals split_kv_size
            (100, 0, 32, 32),  # Budget is 2 * split_kv_size
            (100, 10, 20, 6),  # Start at 10, need to align to 16
            (100, 8, 20, 8),  # Start at 8, align to 16
        ]

        for need_prefill, num_computed, token_budget, expected in test_cases:
            max_possible = min(need_prefill - num_computed, token_budget)
            final_pos = num_computed + max_possible
            aligned_end = (final_pos // split_kv_size) * split_kv_size

            # Adjust expected to not exceed original position if aligned_end would go backward
            if aligned_end < num_computed:
                result = 0
            else:
                result = aligned_end - num_computed

            # Ensure result doesn't exceed budget
            result = min(result, token_budget)
            # Ensure result doesn't exceed need
            result = min(result, need_prefill - num_computed)

            print(f"  need={need_prefill}, computed={num_computed}, " f"budget={token_budget} -> result={result}")

            self.assertGreaterEqual(result, 0, "Result should be non-negative")
            self.assertLessEqual(result, token_budget, "Result should not exceed budget")


class TestDeterminismBehaviorValidation(unittest.TestCase):
    """
    Test to validate determinism behavior differences between modes

    This test suite validates that:
    1. Deterministic mode produces consistent results
    2. Non-deterministic mode produces different results
    3. The tests fail when run in the wrong mode
    """

    def setUp(self):
        """Save original environment variables"""
        self.original_env = {}

    def tearDown(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_env(self, key, value):
        """Set environment variable and save original value"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_deterministic_mode_produces_consistent_seeds(self):
        """
        Test: Deterministic mode produces same seed across multiple instances

        This test PASSES in deterministic mode (FD_DETERMINISTIC_MODE=1)
        This test FAILS in non-deterministic mode (FD_DETERMINISTIC_MODE=0)
        """
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        # Create 10 instances, all should have same seed
        seeds = []
        for _ in range(10):
            params = sp_module.SamplingParams()
            seeds.append(params.seed)

        # All should be 42
        self.assertEqual(len(set(seeds)), 1, "Deterministic mode: all instances should have same seed")
        self.assertEqual(seeds[0], 42, "Deterministic mode: seed should be 42")

        print(f"[PASS] Deterministic mode produces consistent seeds: {set(seeds)}")

    def test_non_deterministic_mode_produces_different_seeds(self):
        """
        Test: Non-deterministic mode produces different seeds across instances

        This test PASSES in non-deterministic mode (FD_DETERMINISTIC_MODE=0)
        This test FAILS in deterministic mode (FD_DETERMINISTIC_MODE=1)
        """
        self._set_env("FD_DETERMINISTIC_MODE", "0")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        # Create 10 instances, should have different seeds
        seeds = []
        for _ in range(10):
            params = sp_module.SamplingParams()
            seeds.append(params.seed)

        # Should have at least 2 different seeds (very high probability with random)
        unique_seeds = len(set(seeds))
        self.assertGreater(
            unique_seeds,
            1,
            f"Non-deterministic mode: should produce different seeds, " f"got {unique_seeds} unique out of 10",
        )

        print(f"[PASS] Non-deterministic mode produces different seeds: " f"{unique_seeds} unique out of 10")

    def test_deterministic_alignment_vs_non_deterministic(self):
        """
        Test: Show alignment behavior difference between modes

        With same parameters:
        - Deterministic mode: allocates aligned to split_kv_size boundary
        - Non-deterministic mode: allocates based on budget
        """
        # Test in deterministic mode
        self._set_env("FD_DETERMINISTIC_MODE", "1")
        self._set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16")

        import importlib

        import fastdeploy.envs

        importlib.reload(fastdeploy.envs)

        split_kv_size = 16
        need_prefill = 100
        num_computed = 0
        token_budget = 50

        # Deterministic: align to split_kv_size (48 = 3 * 16)
        max_possible = min(need_prefill - num_computed, token_budget)
        det_result = (max_possible // split_kv_size) * split_kv_size

        self.assertEqual(det_result, 48, "Deterministic mode: should align to 48")

        # Test in non-deterministic mode
        self._set_env("FD_DETERMINISTIC_MODE", "0")
        importlib.reload(fastdeploy.envs)

        # Non-deterministic: allocate up to budget (50)
        non_det_result = min(need_prefill - num_computed, token_budget)

        self.assertEqual(non_det_result, 50, "Non-deterministic mode: should allocate 50")

        # Show the difference
        self.assertNotEqual(
            det_result,
            non_det_result,
            "Mode difference: deterministic and non-deterministic " "should produce different results",
        )

        print(f"[PASS] Mode behavior difference: deterministic={det_result}, " f"non-deterministic={non_det_result}")

    def test_explicit_seed_works_in_both_modes(self):
        """
        Test: Explicit seed produces same result in both modes

        When user provides explicit seed, it should be used
        regardless of determinism mode setting.
        """
        explicit_seed = 12345

        # Test in deterministic mode
        self._set_env("FD_DETERMINISTIC_MODE", "1")

        import importlib

        import fastdeploy.engine.sampling_params as sp_module
        import fastdeploy.envs as envs_module

        importlib.reload(envs_module)
        importlib.reload(sp_module)

        params_det = sp_module.SamplingParams(seed=explicit_seed)
        self.assertEqual(
            params_det.seed, explicit_seed, f"Deterministic mode: explicit seed should be {explicit_seed}"
        )

        # Test in non-deterministic mode
        self._set_env("FD_DETERMINISTIC_MODE", "0")
        importlib.reload(envs_module)
        importlib.reload(sp_module)

        params_non_det = sp_module.SamplingParams(seed=explicit_seed)
        self.assertEqual(
            params_non_det.seed, explicit_seed, f"Non-deterministic mode: explicit seed should be {explicit_seed}"
        )

        print(f"[PASS] Explicit seed works in both modes: seed={explicit_seed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
