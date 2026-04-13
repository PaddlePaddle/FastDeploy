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
Chunked Prefill Determinism Tests

Test _get_num_new_tokens alignment behavior in ResourceManagerV1:
1. Deterministic disabled (no alignment)
2. Deterministic enabled (split_kv_size boundary alignment)
3. Boundary cases
4. Continuous chunk consistency
5. Multimodal inputs (image / video / audio)
6. Real batch scheduling scenarios
7. Corner cases (empty request, invalid state, large split, dynamic switch, etc.)
"""

import os
import unittest

from fastdeploy.engine.request import Request
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1

# ---------------------------------------------------------------------------
# Minimal config stubs -- only fields accessed by ResourceManagerV1.__init__
# and _get_num_new_tokens are kept.
# ---------------------------------------------------------------------------


class ModelConfig:
    def __init__(self):
        self.enable_mm = False
        self.causal = True


class CacheConfig:
    def __init__(self):
        self.block_size = 16
        self.enable_prefix_caching = False
        self.kvcache_storage_backend = None
        self.write_policy = None
        self.num_cpu_blocks = 0
        self.total_block_num = 10000
        self.prefill_kvcache_block_num = 10000
        self.max_encoder_cache = 0
        self.max_processor_cache = 0
        self.bytes_per_token_per_layer = 32 * 32 * 128 * 2


class ParallelConfig:
    def __init__(self):
        self.local_engine_worker_queue_port = None
        self.tensor_parallel_size = 1


class SpeculativeConfig:
    def __init__(self):
        self.method = None
        self.num_speculative_tokens = 0
        self.model_type = None


class StubConfig:
    """Assembles the minimal sub-configs needed by ResourceManagerV1."""

    def __init__(self):
        self.model_config = ModelConfig()
        self.cache_config = CacheConfig()
        self.parallel_config = ParallelConfig()
        self.speculative_config = SpeculativeConfig()
        self.enable_mm_runtime = self.model_config.enable_mm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_request(request_id, prompt_token_ids, num_computed_tokens=0, multimodal_inputs=None):
    """Create a real Request object for testing."""
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        prompt_token_ids_len=len(prompt_token_ids),
        num_computed_tokens=num_computed_tokens,
        multimodal_inputs=multimodal_inputs,
    )


def _build_mm_inputs(prompt_len, text_len, modal_id, extra=None):
    """Build a multimodal_inputs dict for a single-modality request."""
    mm_len = prompt_len - text_len
    patch_idx_val = modal_id  # 1=image, 2=video, 3=audio
    inputs = {
        "image_patch_id": prompt_len + 1,
        "image_end_id": prompt_len + 2,
        "video_patch_id": prompt_len + 3,
        "video_end_id": prompt_len + 4,
        "audio_patch_id": prompt_len + 5,
        "audio_end_id": prompt_len + 6,
        "patch_idx": [0] * text_len + [patch_idx_val] * mm_len,
        "patch_map": [
            {"modal_id": 0, "end_idx": text_len, "image_num": 0, "video_num": 0},
            {
                "modal_id": modal_id,
                "end_idx": prompt_len,
                "image_num": 1 if modal_id == 1 else 0,
                "video_num": 1 if modal_id == 2 else 0,
            },
        ],
        "tts": False,
    }
    if extra:
        inputs.update(extra)
    return inputs


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestChunkedPrefillDeterminism(unittest.TestCase):
    """Test _get_num_new_tokens alignment in deterministic mode."""

    def setUp(self):
        self._saved_env = {}
        for key in ("FD_DETERMINISTIC_MODE", "FD_DETERMINISTIC_SPLIT_KV_SIZE"):
            self._saved_env[key] = os.environ.get(key)
        self.config = StubConfig()
        self.rm = self._create_resource_manager(self.config)

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # -- env helpers --

    def _enable_deterministic(self, split_kv_size=16):
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

    def _disable_deterministic(self):
        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def _create_resource_manager(self, config):
        return ResourceManagerV1(
            max_num_seqs=32,
            config=config,
            tensor_parallel_size=1,
            splitwise_role="mixed",
            local_data_parallel_id=0,
        )

    def _create_mm_resource_manager(self):
        config = StubConfig()
        config.model_config.enable_mm = True
        config.enable_mm_runtime = config.model_config.enable_mm
        return self._create_resource_manager(config)

    # ==================== 1. Deterministic disabled ====================

    def test_get_num_new_tokens_deterministic_disabled(self):
        """No alignment when deterministic mode is off; budget=0 returns 0."""
        self._disable_deterministic()

        test_cases = [
            # (prompt_tokens, num_computed, token_budget, expected)
            (list(range(100)), 0, 50, 50),
            (list(range(100)), 50, 30, 30),
            (list(range(100)), 90, 20, 10),
            (list(range(32)), 0, 15, 15),
            # budget=0 -> 0
            (list(range(100)), 0, 0, 0),
        ]
        for prompt_ids, num_computed, budget, expected in test_cases:
            with self.subTest(prompt_len=len(prompt_ids), computed=num_computed, budget=budget):
                req = _create_request("req", prompt_ids, num_computed)
                result = self.rm._get_num_new_tokens(req, budget)
                self.assertEqual(result, expected)

    # ==================== 2. Deterministic enabled alignment ====================

    def test_get_num_new_tokens_deterministic_enabled_alignment(self):
        """Results must align to split_kv_size boundary."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)

        test_cases = [
            # (prompt_tokens, num_computed, token_budget, expected)
            (list(range(100)), 0, 20, 16),
            (list(range(100)), 0, 32, 32),
            (list(range(100)), 0, 40, 32),
            (list(range(100)), 0, 50, 48),
            (list(range(100)), 8, 20, 8),
            (list(range(100)), 8, 30, 24),
            (list(range(100)), 16, 20, 16),
            (list(range(100)), 16, 25, 16),
        ]
        for prompt_ids, num_computed, budget, expected in test_cases:
            with self.subTest(computed=num_computed, budget=budget):
                req = _create_request("req", prompt_ids, num_computed)
                result = self.rm._get_num_new_tokens(req, budget)
                self.assertEqual(result, expected)
                # Verify alignment
                if result > 0:
                    final_pos = num_computed + result
                    self.assertEqual(final_pos % split_kv_size, 0)

    # ==================== 3. Boundary cases ====================

    def test_get_num_new_tokens_boundary_cases(self):
        """Boundary conditions including large budget."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)

        test_cases = [
            (list(range(100)), 0, 5, "budget < split_kv_size, start at 0"),
            (list(range(100)), 0, 1, "budget = 1, start at 0"),
            (list(range(100)), 10, 5, "budget < split_kv_size, start at 10"),
            (list(range(100)), 15, 5, "budget < split_kv_size, near boundary"),
            (list(range(16)), 0, 16, "exactly split_kv_size tokens needed"),
            (list(range(16)), 0, 32, "budget > needed"),
            # Very large budget (overflow guard)
            (list(range(100)), 0, 1000000, "very large budget"),
        ]
        for prompt_ids, num_computed, budget, desc in test_cases:
            with self.subTest(desc=desc):
                req = _create_request("req", prompt_ids, num_computed)
                result = self.rm._get_num_new_tokens(req, budget)
                max_possible = min(len(prompt_ids) - num_computed, budget)
                self.assertGreaterEqual(result, 0)
                self.assertLessEqual(result, max_possible)

    # ==================== 4. Chunk consistency ====================

    def test_get_num_new_tokens_consistency_across_chunks(self):
        """All chunk boundaries must align to split_kv_size."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)

        prompt_ids = list(range(112))
        budget = 50
        num_computed = 0
        chunk_sizes = []

        while num_computed < len(prompt_ids):
            req = _create_request("req", prompt_ids, num_computed)
            result = self.rm._get_num_new_tokens(req, budget)
            if result == 0:
                break
            chunk_sizes.append(result)
            num_computed += result

        # Every intermediate boundary must be aligned; final position may equal seq length
        position = 0
        for chunk_size in chunk_sizes:
            position += chunk_size
            is_ok = (position % split_kv_size == 0) or (position == len(prompt_ids))
            self.assertTrue(is_ok, f"position {position} not aligned to {split_kv_size}")

        self.assertEqual(num_computed, len(prompt_ids))

    # ==================== 5. Multimodal (parameterized) ====================

    _MULTIMODAL_CASES = [
        {"name": "image", "prompt_len": 150, "text_len": 50, "modal_id": 1, "budget": 60, "extra": {}},
        {
            "name": "video",
            "prompt_len": 200,
            "text_len": 80,
            "modal_id": 2,
            "budget": 50,
            "extra": {"can_split_idx_list": [96, 112, 128, 144, 160, 176, 192]},
        },
        {"name": "audio", "prompt_len": 120, "text_len": 60, "modal_id": 3, "budget": 40, "extra": {}},
    ]

    def test_multimodal_input_single_modality(self):
        """Token allocation for image / video / audio multimodal requests."""
        self._enable_deterministic(16)
        rm = self._create_mm_resource_manager()

        for case in self._MULTIMODAL_CASES:
            with self.subTest(modality=case["name"]):
                prompt_ids = list(range(case["prompt_len"]))
                mm_inputs = _build_mm_inputs(case["prompt_len"], case["text_len"], case["modal_id"], case["extra"])
                req = _create_request(f"mm_{case['name']}", prompt_ids, 0, mm_inputs)
                result = rm._get_num_new_tokens(req, case["budget"])
                self.assertGreaterEqual(result, 0)
                self.assertLessEqual(result, case["budget"])

    # ==================== 6. Real batch scheduling ====================

    def test_real_batch_scheduling_concurrent_requests(self):
        """Multiple requests competing for budget, all must respect alignment."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)
        budget = 50

        batch = [
            ("req1", list(range(27)), 0),
            ("req2", list(range(63)), 0),
            ("req3", list(range(128)), 0),
            ("req4", list(range(60)), 10),
            ("req5", list(range(47)), 7),
        ]
        for rid, prompt_ids, computed in batch:
            with self.subTest(request=rid):
                req = _create_request(rid, prompt_ids, computed)
                result = self.rm._get_num_new_tokens(req, budget)
                final_pos = computed + result
                max_possible = min(len(prompt_ids) - computed, budget)
                self.assertLessEqual(result, max_possible)
                if result > 0:
                    is_ok = (final_pos % split_kv_size == 0) or (final_pos == len(prompt_ids))
                    self.assertTrue(is_ok, f"{rid}: final_pos={final_pos} not aligned")

    def test_real_batch_scheduling_continuous_prefill(self):
        """Continuous prefill: all chunks fully consume a 47-token prompt."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)

        prompt_ids = list(range(47))
        budget = 50
        num_computed = 0
        iterations = 0

        while num_computed < len(prompt_ids) and iterations < 10:
            req = _create_request("cont", prompt_ids, num_computed)
            result = self.rm._get_num_new_tokens(req, budget)
            self.assertGreater(result, 0, f"stuck at {num_computed}")
            final_pos = num_computed + result
            is_ok = (final_pos % split_kv_size == 0) or (final_pos == len(prompt_ids))
            self.assertTrue(is_ok, f"chunk ending at {final_pos} not aligned")
            num_computed += result
            iterations += 1

        self.assertEqual(num_computed, len(prompt_ids))

    def test_real_batch_scheduling_with_multimodal_requests(self):
        """Mixed batch: text-only + image requests."""
        self._enable_deterministic(16)
        rm = self._create_mm_resource_manager()
        budget = 30

        # Text-only request
        req_text = _create_request("text_only", list(range(100)), 0)
        r1 = rm._get_num_new_tokens(req_text, budget)
        self.assertGreaterEqual(r1, 0)
        self.assertLessEqual(r1, budget)

        # Image request
        mm_inputs = _build_mm_inputs(80, 40, modal_id=1)
        req_img = _create_request("with_image", list(range(80)), 0, mm_inputs)
        r2 = rm._get_num_new_tokens(req_img, budget)
        self.assertGreaterEqual(r2, 0)
        self.assertLessEqual(r2, budget)

    # ==================== 7. Corner cases ====================

    def test_corner_case_invalid_request_states(self):
        """Empty prompt, completed prefill, and num_computed > need_prefill must assert."""
        self._enable_deterministic(16)

        # Empty prompt
        with self.subTest(case="empty prompt"):
            with self.assertRaises(AssertionError):
                self.rm._get_num_new_tokens(_create_request("e", [], 0), 50)

        # Already completed
        with self.subTest(case="completed prefill"):
            with self.assertRaises(AssertionError):
                self.rm._get_num_new_tokens(_create_request("c", list(range(100)), 100), 50)

        # Inconsistent state
        with self.subTest(case="num_computed > need_prefill"):
            with self.assertRaises(AssertionError):
                self.rm._get_num_new_tokens(_create_request("i", list(range(50)), 100), 50)

        # Zero budget (legitimate, returns 0)
        with self.subTest(case="zero budget"):
            result = self.rm._get_num_new_tokens(_create_request("z", list(range(100)), 0), 0)
            self.assertEqual(result, 0)

    def test_corner_case_minimum_split_size(self):
        """split_kv_size=1: every position is aligned, so max allocation is allowed."""
        self._enable_deterministic(1)

        for prompt_ids, computed, budget, expected in [
            (list(range(100)), 0, 20, 20),
            (list(range(100)), 10, 15, 15),
            (list(range(100)), 50, 10, 10),
        ]:
            with self.subTest(computed=computed, budget=budget):
                req = _create_request("min", prompt_ids, computed)
                result = self.rm._get_num_new_tokens(req, budget)
                self.assertEqual(result, expected)

    def test_corner_case_large_split_size(self):
        """split_kv_size >> budget or sequence length."""
        test_cases = [
            # (split_kv_size, prompt_ids, num_computed, budget, description)
            (128, list(range(100)), 0, 10, "split >> budget: budget=10"),
            (128, list(range(100)), 0, 1, "split >> budget: budget=1"),
            (128, list(range(100)), 64, 20, "split >> budget: near boundary"),
            (256, list(range(50)), 0, 100, "split >> seq_len"),
        ]
        for split_kv_size, prompt_ids, computed, budget, desc in test_cases:
            with self.subTest(desc=desc):
                self._enable_deterministic(split_kv_size)
                req = _create_request("lg", prompt_ids, computed)
                result = self.rm._get_num_new_tokens(req, budget)
                max_possible = min(len(prompt_ids) - computed, budget)
                self.assertGreaterEqual(result, 0)
                self.assertLessEqual(result, max_possible)

    def test_corner_case_dynamic_config_switch(self):
        """Switching from non-deterministic to deterministic mid-stream."""
        # Phase 1: non-deterministic
        self._disable_deterministic()
        req1 = _create_request("sw1", list(range(100)), 0)
        result1 = self.rm._get_num_new_tokens(req1, 30)

        # Phase 2: enable deterministic, continue from result1
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)
        req2 = _create_request("sw2", list(range(100)), result1)
        result2 = self.rm._get_num_new_tokens(req2, 30)

        if result2 > 0:
            final_pos = result1 + result2
            is_aligned = (final_pos % split_kv_size == 0) or (final_pos == 100)
            self.assertTrue(is_aligned, f"final_pos={final_pos} not aligned after switch")

    def test_deterministic_return_zero_budget_below_boundary(self):
        """Returns 0 when budget cannot reach the next alignment boundary."""
        split_kv_size = 16
        self._enable_deterministic(split_kv_size)

        test_cases = [
            # (prompt_ids, num_computed, budget)
            # pos=10, next_boundary=16, need 6, budget=5
            (list(range(100)), 10, 5),
            # pos=1, next_boundary=16, need 15, budget=3
            (list(range(100)), 1, 3),
            # pos=17, next_boundary=32, need 15, budget=14
            (list(range(100)), 17, 14),
            # budget=0 (deterministic)
            (list(range(100)), 0, 0),
        ]
        for prompt_ids, computed, budget in test_cases:
            with self.subTest(computed=computed, budget=budget):
                req = _create_request("det0", prompt_ids, computed)
                result = self.rm._get_num_new_tokens(req, budget)
                self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
