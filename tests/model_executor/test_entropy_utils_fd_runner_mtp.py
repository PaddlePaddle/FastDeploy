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
Unit tests for entropy_utils under FD runner (EB5_ENABLE_FD_RUNNER=1) + MTP scenario.

This test can be run standalone:
    EB5_ENABLE_FD_RUNNER=1 python tests/model_executor/test_entropy_utils_fd_runner_mtp.py

Key differences from ernie5_runner path:
- speculate_calculate_logits_entropy receives logits of shape [sum(seq_lens_this_time), vocab]
  which includes ALL positions (accepted + rejected). The code must use accepted_idx to
  extract the correct rows.
- calculate_logits_entropy receives logits of shape [max_bsz, vocab], one row per slot.
"""

import importlib
import logging
import os
import sys
import types
import unittest

# Force FD runner path before importing the module
os.environ["EB5_ENABLE_FD_RUNNER"] = "1"

import paddle

# Mock fastdeploy.utils to avoid importing the full fastdeploy package
_mock_fastdeploy = types.ModuleType("fastdeploy")
_mock_fastdeploy.__path__ = []
_mock_utils = types.ModuleType("fastdeploy.utils")
_mock_utils.data_processor_logger = logging.getLogger("test_entropy_fd_runner")
_mock_fastdeploy.utils = _mock_utils

sys.modules["fastdeploy"] = _mock_fastdeploy
sys.modules["fastdeploy.utils"] = _mock_utils

# Now import the module under test via importlib to ensure it picks up our mocks
_entropy_spec = importlib.util.spec_from_file_location(
    "fastdeploy.model_executor.entropy_utils",
    os.path.join(os.path.dirname(__file__), "../../fastdeploy/model_executor/entropy_utils.py"),
)
_entropy_module = importlib.util.module_from_spec(_entropy_spec)
sys.modules["fastdeploy.model_executor"] = types.ModuleType("fastdeploy.model_executor")
sys.modules["fastdeploy.model_executor.entropy_utils"] = _entropy_module
_entropy_spec.loader.exec_module(_entropy_module)

_USE_FD_RUNNER = _entropy_module._USE_FD_RUNNER
get_entropy = _entropy_module.get_entropy
calculate_logits_entropy = _entropy_module.calculate_logits_entropy
speculate_calculate_logits_entropy = _entropy_module.speculate_calculate_logits_entropy


class TestFdRunnerFlag(unittest.TestCase):
    """Verify the module loaded with FD runner enabled."""

    def test_fd_runner_enabled(self):
        self.assertTrue(_USE_FD_RUNNER)


class TestGetEntropy(unittest.TestCase):
    """Test the core entropy calculation function."""

    def test_uniform_distribution(self):
        # Uniform logits => max entropy = ln(vocab_size)
        logits = paddle.zeros([1, 4], dtype="float32")
        entropy = get_entropy(logits)
        import math

        self.assertAlmostEqual(float(entropy[0]), math.log(4), places=5)

    def test_deterministic_distribution(self):
        # One logit dominates => entropy near 0
        logits = paddle.to_tensor([[100.0, 0.0, 0.0, 0.0]], dtype="float32")
        entropy = get_entropy(logits)
        self.assertAlmostEqual(float(entropy[0]), 0.0, places=5)

    def test_negative_inf_handling(self):
        logits = paddle.to_tensor([[10.0, -float("inf"), -float("inf")]], dtype="float32")
        entropy = get_entropy(logits)
        # After clipping -inf, effectively one dominant logit
        self.assertGreaterEqual(float(entropy[0]), 0.0)


class TestCalculateLogitsEntropyFdRunner(unittest.TestCase):
    """Test calculate_logits_entropy under FD runner (1D shape, direct indexing)."""

    def _make_share_inputs(self, bsz, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, stop_flags, req_ids):
        return {
            "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time, dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder, dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder, dtype="int32"),
            "entropy_list": [[] for _ in range(bsz)],
            "stop_flags": paddle.to_tensor(stop_flags, dtype="bool"),
            "req_ids": req_ids,
        }

    def test_basic_accumulation(self):
        # FD runner: logits shape [max_bsz, vocab], 1D seq_lens
        share_inputs = self._make_share_inputs(
            bsz=3,
            seq_lens_this_time=[1, 1, 0],
            seq_lens_encoder=[0, 0, 0],
            seq_lens_decoder=[10, 10, 0],
            stop_flags=[False, False, False],
            req_ids=["req_a", "req_b", "req_c"],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # slot 0
                [1.0, 1.0, 10.0],  # slot 1
                [5.0, 5.0, 5.0],  # slot 2 (seq_lens_this_time=0, skipped)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 0)
        # Same logits pattern => same entropy
        self.assertAlmostEqual(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][1][0],
            places=6,
        )

    def test_stop_flags_clear(self):
        share_inputs = self._make_share_inputs(
            bsz=2,
            seq_lens_this_time=[1, 1],
            seq_lens_encoder=[0, 0],
            seq_lens_decoder=[10, 10],
            stop_flags=[True, False],
            req_ids=["req_a", "req_b"],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([2], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        # slot 0: stop_flags=True + seq_lens_decoder!=0 => cleared
        self.assertEqual(len(share_inputs["entropy_list"][0]), 0)
        # slot 1: stop_flags=False => kept
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)

    def test_temperature_scaling(self):
        share_inputs = self._make_share_inputs(
            bsz=2,
            seq_lens_this_time=[1, 1],
            seq_lens_encoder=[0, 0],
            seq_lens_decoder=[10, 10],
            stop_flags=[False, False],
            req_ids=["req_a", "req_b"],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [10.0, 1.0, 1.0],
            ],
            dtype="float32",
        )
        # slot 0: temp=1.0 (no scaling), slot 1: temp=0.5 (sharper distribution)
        temperature = paddle.to_tensor([1.0, 0.5], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        # Lower temperature => lower entropy (more peaked)
        self.assertGreater(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][1][0],
        )


class TestSpeculateCalculateLogitsEntropyFdRunner(unittest.TestCase):
    """
    Test speculate_calculate_logits_entropy under FD runner + MTP.

    Key: logits shape is [sum(seq_lens_this_time), vocab], containing all positions
    (accepted + rejected). The function must use accepted_idx to extract correct rows.
    """

    def _make_share_inputs(
        self, bsz, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, stop_flags, req_ids, accept_num
    ):
        return {
            "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time, dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder, dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder, dtype="int32"),
            "entropy_list": [[] for _ in range(bsz)],
            "stop_flags": paddle.to_tensor(stop_flags, dtype="bool"),
            "req_ids": req_ids,
            "accept_num": paddle.to_tensor(accept_num, dtype="int32"),
        }

    def test_accepted_idx_extraction(self):
        """
        Scenario: 3 slots, seq_lens_this_time=[2, 3, 1], accept_num=[1, 2, 1]
        logits shape: [2+3+1=6, vocab]
        - slot 0: positions [0,1], accepted=1 => take row 0
        - slot 1: positions [2,3,4], accepted=2 => take rows 2,3
        - slot 2: positions [5], accepted=1 => take row 5
        """
        share_inputs = self._make_share_inputs(
            bsz=3,
            seq_lens_this_time=[2, 3, 1],
            seq_lens_encoder=[0, 0, 0],
            seq_lens_decoder=[10, 10, 10],
            stop_flags=[False, False, False],
            req_ids=["req_a", "req_b", "req_c"],
            accept_num=[1, 2, 1],
        )
        # 6 rows total, each with distinct logits so we can verify correct extraction
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # row 0: slot 0, position 0 (accepted)
                [1.0, 1.0, 1.0],  # row 1: slot 0, position 1 (rejected)
                [1.0, 10.0, 1.0],  # row 2: slot 1, position 0 (accepted)
                [1.0, 1.0, 10.0],  # row 3: slot 1, position 1 (accepted)
                [5.0, 5.0, 5.0],  # row 4: slot 1, position 2 (rejected)
                [10.0, 10.0, 1.0],  # row 5: slot 2, position 0 (accepted)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        # slot 0: 1 accepted token
        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        # slot 1: 2 accepted tokens
        self.assertEqual(len(share_inputs["entropy_list"][1]), 2)
        # slot 2: 1 accepted token
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        # Verify the extracted logits produce correct entropy values
        # row 0: [10, 1, 1] => same as row 2: [1, 10, 1] (symmetric)
        self.assertAlmostEqual(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][1][0],
            places=6,
        )
        # row 3: [1, 1, 10] => same entropy as [10, 1, 1]
        self.assertAlmostEqual(
            share_inputs["entropy_list"][1][1],
            share_inputs["entropy_list"][0][0],
            places=6,
        )
        # row 5: [10, 10, 1] => different entropy (more uniform among top-2)
        self.assertGreater(
            share_inputs["entropy_list"][2][0],
            share_inputs["entropy_list"][0][0],
        )

    def test_zero_accepted_with_stop_flags(self):
        """
        When total_accepted_num=0 but stop_flags is set, ENTROPY-DONE should trigger
        and clear the entropy_list.
        """
        share_inputs = self._make_share_inputs(
            bsz=2,
            seq_lens_this_time=[1, 1],
            seq_lens_encoder=[0, 0],
            seq_lens_decoder=[10, 10],
            stop_flags=[True, False],
            req_ids=["req_a", "req_b"],
            accept_num=[0, 0],
        )
        # Pre-fill some entropy values to verify clearing
        share_inputs["entropy_list"][0] = [1.0, 2.0, 3.0]
        share_inputs["entropy_list"][1] = [4.0, 5.0]

        logits = paddle.zeros([2, 3], dtype="float32")  # irrelevant, won't be used

        speculate_calculate_logits_entropy(logits, share_inputs, paddle.ones([2], dtype="float32"))

        # slot 0: stop_flags=True => cleared
        self.assertEqual(share_inputs["entropy_list"][0], [])
        # slot 1: stop_flags=False => unchanged
        self.assertEqual(share_inputs["entropy_list"][1], [4.0, 5.0])

    def test_warmup_skip(self):
        """
        Warmup/dummy requests with empty req_id should not accumulate entropy.
        """
        share_inputs = self._make_share_inputs(
            bsz=3,
            seq_lens_this_time=[2, 2, 2],
            seq_lens_encoder=[0, 0, 0],
            seq_lens_decoder=[10, 10, 10],
            stop_flags=[False, False, False],
            req_ids=["", "req_real", "  "],  # slot 0 and 2 are warmup
            accept_num=[1, 1, 1],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # slot 0, pos 0 (accepted) - warmup
                [5.0, 5.0, 5.0],  # slot 0, pos 1 (rejected)
                [1.0, 10.0, 1.0],  # slot 1, pos 0 (accepted) - real
                [5.0, 5.0, 5.0],  # slot 1, pos 1 (rejected)
                [1.0, 1.0, 10.0],  # slot 2, pos 0 (accepted) - warmup (whitespace)
                [5.0, 5.0, 5.0],  # slot 2, pos 1 (rejected)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        # slot 0: warmup, skipped
        self.assertEqual(len(share_inputs["entropy_list"][0]), 0)
        # slot 1: real request, accumulated
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        # slot 2: whitespace req_id, skipped
        self.assertEqual(len(share_inputs["entropy_list"][2]), 0)

    def test_partial_accept(self):
        """
        Mixed accept counts: some slots accept 2, some accept 0.
        Verifies correct indexing when accept_num varies per slot.
        """
        share_inputs = self._make_share_inputs(
            bsz=4,
            seq_lens_this_time=[2, 2, 2, 2],
            seq_lens_encoder=[0, 0, 0, 0],
            seq_lens_decoder=[10, 10, 10, 10],
            stop_flags=[False, False, False, False],
            req_ids=["req_a", "req_b", "req_c", "req_d"],
            accept_num=[2, 0, 1, 2],
        )
        # total rows = sum(seq_lens_this_time) = 8
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # row 0: slot 0, pos 0 (accepted)
                [1.0, 10.0, 1.0],  # row 1: slot 0, pos 1 (accepted)
                [5.0, 5.0, 5.0],  # row 2: slot 1, pos 0 (not accepted)
                [5.0, 5.0, 5.0],  # row 3: slot 1, pos 1 (not accepted)
                [1.0, 1.0, 10.0],  # row 4: slot 2, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # row 5: slot 2, pos 1 (rejected)
                [10.0, 10.0, 1.0],  # row 6: slot 3, pos 0 (accepted)
                [1.0, 10.0, 10.0],  # row 7: slot 3, pos 1 (accepted)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([4], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 2)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][3]), 2)

        # Verify values: row 0 [10,1,1] and row 1 [1,10,1] have same entropy
        self.assertAlmostEqual(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][0][1],
            places=6,
        )
        # row 4 [1,1,10] same entropy as row 0
        self.assertAlmostEqual(
            share_inputs["entropy_list"][2][0],
            share_inputs["entropy_list"][0][0],
            places=6,
        )
        # row 6 [10,10,1] and row 7 [1,10,10] same entropy (symmetric)
        self.assertAlmostEqual(
            share_inputs["entropy_list"][3][0],
            share_inputs["entropy_list"][3][1],
            places=6,
        )

    def test_stop_flags_after_accept(self):
        """
        Slot finishes (stop_flags=True) after accepting tokens in same step.
        Entropy should be accumulated then immediately cleared.
        """
        share_inputs = self._make_share_inputs(
            bsz=2,
            seq_lens_this_time=[2, 2],
            seq_lens_encoder=[0, 0],
            seq_lens_decoder=[10, 10],
            stop_flags=[True, False],
            req_ids=["req_a", "req_b"],
            accept_num=[2, 1],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # slot 0, pos 0 (accepted)
                [1.0, 10.0, 1.0],  # slot 0, pos 1 (accepted)
                [1.0, 1.0, 10.0],  # slot 1, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # slot 1, pos 1 (rejected)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([2], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        # slot 0: accepted 2, then stop_flags=True => cleared
        self.assertEqual(share_inputs["entropy_list"][0], [])
        # slot 1: accepted 1, stop_flags=False => kept
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)

    def test_temperature_scaling_mtp(self):
        """Verify temperature scaling is applied per-slot in MTP path."""
        share_inputs = self._make_share_inputs(
            bsz=2,
            seq_lens_this_time=[2, 2],
            seq_lens_encoder=[0, 0],
            seq_lens_decoder=[10, 10],
            stop_flags=[False, False],
            req_ids=["req_a", "req_b"],
            accept_num=[1, 1],
        )
        # Same logits for both slots
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # slot 0, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # slot 0, pos 1 (rejected)
                [10.0, 1.0, 1.0],  # slot 1, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # slot 1, pos 1 (rejected)
            ],
            dtype="float32",
        )
        # slot 0: temp=1.0, slot 1: temp=0.5
        temperature = paddle.to_tensor([1.0, 0.5], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        # Lower temperature => lower entropy
        self.assertGreater(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][1][0],
        )

    def test_encoder_prefill_slot(self):
        """
        When seq_lens_encoder != 0, real_seq_lens is forced to 1 regardless of
        seq_lens_this_time. Verify accepted_idx handles this correctly.
        """
        share_inputs = self._make_share_inputs(
            bsz=3,
            seq_lens_this_time=[2, 100, 2],  # slot 1 is prefill (large seq_lens_this_time)
            seq_lens_encoder=[0, 100, 0],  # slot 1 is encoder (prefill)
            seq_lens_decoder=[10, 0, 10],
            stop_flags=[False, False, False],
            req_ids=["req_a", "req_b", "req_c"],
            accept_num=[1, 1, 1],
        )
        # real_seq_lens = [2, 1, 2] (slot 1 forced to 1 because encoder != 0)
        # total rows = sum(real_seq_lens) = 2 + 1 + 2 = 5
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # row 0: slot 0, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # row 1: slot 0, pos 1
                [1.0, 10.0, 1.0],  # row 2: slot 1, pos 0 (accepted, prefill)
                [1.0, 1.0, 10.0],  # row 3: slot 2, pos 0 (accepted)
                [5.0, 5.0, 5.0],  # row 4: slot 2, pos 1
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        # All accepted logits are [10,1,1]-symmetric => same entropy
        self.assertAlmostEqual(
            share_inputs["entropy_list"][0][0],
            share_inputs["entropy_list"][1][0],
            places=6,
        )
        self.assertAlmostEqual(
            share_inputs["entropy_list"][1][0],
            share_inputs["entropy_list"][2][0],
            places=6,
        )

    def test_multi_step_accumulation(self):
        """
        Simulate multiple MTP steps and verify entropy accumulates correctly
        across steps until stop_flags triggers ENTROPY-DONE.
        """
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([2, 2], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([0, 0], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([10, 10], dtype="int32"),
            "entropy_list": [[], []],
            "stop_flags": paddle.to_tensor([False, False], dtype="bool"),
            "req_ids": ["req_a", "req_b"],
            "accept_num": paddle.to_tensor([2, 1], dtype="int32"),
        }

        logits_step1 = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
                [5.0, 5.0, 5.0],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([2], dtype="float32")

        # Step 1
        speculate_calculate_logits_entropy(logits_step1, share_inputs, temperature)
        self.assertEqual(len(share_inputs["entropy_list"][0]), 2)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)

        # Step 2: same structure, slot 1 finishes
        share_inputs["accept_num"] = paddle.to_tensor([1, 1], dtype="int32")
        share_inputs["stop_flags"] = paddle.to_tensor([False, True], dtype="bool")
        logits_step2 = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [10.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
            ],
            dtype="float32",
        )

        speculate_calculate_logits_entropy(logits_step2, share_inputs, temperature)

        # slot 0: 2 + 1 = 3 accumulated
        self.assertEqual(len(share_inputs["entropy_list"][0]), 3)
        # slot 1: had 1 from step1, got 1 more, then cleared by stop_flags
        self.assertEqual(share_inputs["entropy_list"][1], [])


class TestCrossRunnerConsistency(unittest.TestCase):
    """
    Verify FD runner and ernie5_runner paths produce the same entropy values
    when given equivalent inputs (accepted-only logits).
    """

    def test_same_entropy_both_paths(self):
        """
        Construct inputs such that the accepted logits are identical for both paths,
        then compare the entropy values.
        """
        # The accepted logits we want both paths to process
        accepted_logits_data = [
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ]

        # --- FD runner path (current module) ---
        # logits = [sum(seq_lens_this_time), vocab] with extra rejected rows
        share_inputs_fd = {
            "seq_lens_this_time": paddle.to_tensor([2, 2, 2], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([0, 0, 0], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([10, 10, 10], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([False, False, False], dtype="bool"),
            "req_ids": ["req_a", "req_b", "req_c"],
            "accept_num": paddle.to_tensor([1, 1, 1], dtype="int32"),
        }
        logits_fd = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],  # slot 0, pos 0 (accepted)
                [99.0, 99.0, 99.0],  # slot 0, pos 1 (rejected - should be ignored)
                [1.0, 10.0, 1.0],  # slot 1, pos 0 (accepted)
                [99.0, 99.0, 99.0],  # slot 1, pos 1 (rejected)
                [1.0, 1.0, 10.0],  # slot 2, pos 0 (accepted)
                [99.0, 99.0, 99.0],  # slot 2, pos 1 (rejected)
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits_fd, share_inputs_fd, temperature)

        # --- Direct entropy calculation on accepted logits ---
        direct_entropy = get_entropy(paddle.to_tensor(accepted_logits_data, dtype="float32")).tolist()

        # Compare
        for i in range(3):
            self.assertAlmostEqual(
                share_inputs_fd["entropy_list"][i][0],
                direct_entropy[i],
                places=6,
                msg=f"Mismatch at slot {i}",
            )


if __name__ == "__main__":
    unittest.main()
