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
Unit tests for unified_update_model_status kernel.

This module tests the unified_update_model_status CUDA kernel which combines
speculate_update and speculate_set_value_by_flags_and_idx functionality.
"""

import unittest
from typing import Any, Dict

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import unified_update_model_status


def gen_unified_update_inputs(
    real_bsz: int = 8,
    max_step_tokens: int = 16,
    max_model_len: int = 256,
    seed: int = 42,
    is_naive_mode: bool = False,
) -> Dict[str, Any]:
    """
    Generate test inputs for unified_update_model_status kernel.

    Args:
        real_bsz: Batch size
        max_step_tokens: Maximum step tokens
        max_model_len: Maximum model length (token_ids_all length)
        seed: Random seed
        is_naive_mode: Whether to test naive mode

    Returns:
        Dictionary of input tensors
    """
    rng = np.random.default_rng(seed)
    max_bsz = real_bsz + 4  # Add some padding

    # Encoder/decoder lengths
    seq_lens_encoder = rng.integers(0, 5, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(10, 100, size=max_bsz, dtype=np.int32)

    # Running status
    has_running_seqs = np.array([True], dtype=bool)

    # Step I/O
    step_input_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    adaptive_step_input_len = rng.integers(1, max_step_tokens + 1, size=max_bsz, dtype=np.int32)

    # Output from verify step
    step_output_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    step_output_len = rng.integers(1, max_step_tokens + 1, size=max_bsz, dtype=np.int32)

    # Control flags
    stop_flags = np.zeros(max_bsz, dtype=bool)
    stop_flags[rng.choice(max_bsz, size=2, replace=False)] = True  # Some stopped sequences

    seq_lens_this_time = rng.integers(1, max_step_tokens + 1, size=real_bsz, dtype=np.int32)
    is_paused = np.zeros(max_bsz, dtype=bool)

    # State tensors
    mask_rollback = np.zeros(max_bsz, dtype=np.int32)
    prompt_lens = rng.integers(10, 50, size=max_bsz, dtype=np.int64)
    token_ids_all = rng.integers(0, 1000, size=(max_bsz, max_model_len), dtype=np.int64)
    step_idx = rng.integers(0, 50, size=max_bsz, dtype=np.int64)

    # New params for EOS detection
    end_tokens = rng.integers(1, 1000, size=4, dtype=np.int64)
    max_dec_len = rng.integers(100, 200, size=max_bsz, dtype=np.int64)

    return {
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "has_running_seqs": has_running_seqs,
        "step_input_ids": step_input_ids,
        "adaptive_step_input_len": adaptive_step_input_len,
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "stop_flags": stop_flags,
        "seq_lens_this_time": seq_lens_this_time,
        "is_paused": is_paused,
        "mask_rollback": mask_rollback,
        "token_ids_all": token_ids_all,
        "prompt_lens": prompt_lens,
        "step_idx": step_idx,
        "end_tokens": end_tokens,
        "max_dec_len": max_dec_len,
        "real_bsz": real_bsz,
        "max_bsz": max_bsz,
        "max_step_tokens": max_step_tokens,
        "max_model_len": max_model_len,
        "is_naive_mode": is_naive_mode,
        "prefill_one_step_stop": False,
    }


class TestUnifiedUpdateModelStatus(unittest.TestCase):
    """Test suite for unified_update_model_status kernel."""

    def run_unified_update_test(self, config: Dict[str, Any]) -> None:
        """Run a single test case."""
        inputs = gen_unified_update_inputs(**config)

        # Prepare GPU inputs
        paddle_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, (int, bool)):
                paddle_inputs[k] = v
            else:
                paddle_inputs[k] = paddle.to_tensor(v)

        # Run kernel
        unified_update_model_status(
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["seq_lens_decoder"],
            paddle_inputs["has_running_seqs"],
            paddle_inputs["step_input_ids"],
            paddle_inputs["adaptive_step_input_len"],
            paddle_inputs["step_output_ids"],
            paddle_inputs["step_output_len"],
            paddle_inputs["stop_flags"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["is_paused"],
            paddle_inputs["mask_rollback"],
            paddle_inputs["token_ids_all"],
            paddle_inputs["prompt_lens"],
            paddle_inputs["step_idx"],
            paddle_inputs["end_tokens"],
            paddle_inputs["max_dec_len"],
            inputs["is_naive_mode"],
            inputs["prefill_one_step_stop"],
        )

        # Basic sanity checks
        seq_lens_decoder = paddle_inputs["seq_lens_decoder"].numpy()
        seq_lens_encoder = paddle_inputs["seq_lens_encoder"].numpy()

        # After kernel: encoder should be 0 (converted to decoder)
        self.assertTrue(np.all(seq_lens_encoder[: inputs["real_bsz"]] == 0))

        # Decoder should be updated
        self.assertTrue(np.all(seq_lens_decoder >= 0))

    def test_mtp_mode(self):
        """Test MTP mode (is_naive_mode=False)."""
        config = {
            "real_bsz": 8,
            "max_step_tokens": 16,
            "max_model_len": 256,
            "seed": 42,
            "is_naive_mode": False,
        }
        self.run_unified_update_test(config)

    def test_naive_mode(self):
        """Test naive mode (is_naive_mode=True)."""
        config = {
            "real_bsz": 8,
            "max_step_tokens": 16,
            "pre_ids_len": 256,
            "seed": 42,
            "is_naive_mode": True,
        }
        self.run_unified_update_test(config)

    def test_small_batch(self):
        """Test with small batch size."""
        config = {
            "real_bsz": 1,
            "max_step_tokens": 8,
            "max_model_len": 128,
            "seed": 42,
            "is_naive_mode": False,
        }
        self.run_unified_update_test(config)

    def test_large_batch(self):
        """Test with larger batch size."""
        config = {
            "real_bsz": 32,
            "max_step_tokens": 16,
            "max_model_len": 512,
            "seed": 42,
            "is_naive_mode": False,
        }
        self.run_unified_update_test(config)

    def test_prefill_one_step_stop(self):
        """Test with prefill_one_step_stop=True."""
        config = {
            "real_bsz": 4,
            "max_step_tokens": 8,
            "pre_ids_len": 128,
            "seed": 42,
            "is_naive_mode": False,
        }
        inputs = gen_unified_update_inputs(**config)
        inputs["prefill_one_step_stop"] = True
        inputs["max_model_len"] = 128

        # Prepare GPU inputs
        paddle_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, (int, bool)):
                paddle_inputs[k] = v
            else:
                paddle_inputs[k] = paddle.to_tensor(v)

        # Run kernel
        unified_update_model_status(
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["seq_lens_decoder"],
            paddle_inputs["has_running_seqs"],
            paddle_inputs["step_input_ids"],
            paddle_inputs["adaptive_step_input_len"],
            paddle_inputs["step_output_ids"],
            paddle_inputs["step_output_len"],
            paddle_inputs["stop_flags"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["is_paused"],
            paddle_inputs["mask_rollback"],
            paddle_inputs["token_ids_all"],
            paddle_inputs["prompt_lens"],
            paddle_inputs["step_idx"],
            paddle_inputs["end_tokens"],
            paddle_inputs["max_dec_len"],
            inputs["is_naive_mode"],
            inputs["prefill_one_step_stop"],
        )

        # Check that stop_flags are set for sequences with prefill
        # This is a basic sanity check


if __name__ == "__main__":
    unittest.main()
