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
Unit tests for naive_update_model_status kernel.

Kernel semantics (from naive_update_model_status.cu):
  - Launched as <<<1, 1024>>>, one thread per real batch slot.
  - Guard: seq_lens_this_time[bid] > 0 (already zeroed for stopped/paused
    slots by pre_process before this kernel runs).
  - Scatters sampled token from packed next_tokens (indexed by cu_seqlens_q_output)
    into accept_tokens[bid, 0].
  - Sets accept_num[bid] = 1 for running slots, 0 otherwise.
  - Sets seq_lens_this_time[bid] = 1 for running, 0 otherwise.

  cu_seqlens_q_output layout:
    next_tokens[cu_seqlens_q_output[i] .. cu_seqlens_q_output[i+1]-1]
    are the output tokens for request i (exactly 1 for decode, 0 for stopped).
"""

import unittest
from typing import Any, Dict

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.ops.gpu import naive_update_model_status

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict → GPU paddle tensors."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any]):
    """Call naive_update_model_status kernel (5 tensor args)."""
    naive_update_model_status(
        paddle_inputs["accept_tokens"],
        paddle_inputs["accept_num"],
        paddle_inputs["seq_lens_this_time"],
        paddle_inputs["next_tokens"],
        paddle_inputs["cu_seqlens_q_output"],
    )


OUTPUT_KEYS = [
    "accept_tokens",
    "accept_num",
    "seq_lens_this_time",
]


def get_outputs(paddle_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: paddle_inputs[k].numpy() for k in OUTPUT_KEYS}


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_inputs(
    real_bsz: int = 8,
    max_step_tokens: int = 4,
    seed: int = 42,
    seq_lens_this_time: np.ndarray = None,
) -> Dict[str, Any]:
    """Generate randomized test inputs.

    seq_lens_this_time: per-slot values pre-set by caller (mirrors what
    pre_process produces). Slots with value > 0 are treated as running;
    slots with value == 0 are stopped/paused. If None, a random mix is
    generated (~75% running).

    cu_seqlens_q_output: cumulative token offsets. Running slots (seq_lens > 0)
    get 1 token; stopped/paused slots get 0 tokens.
    """
    rng = np.random.default_rng(seed)

    if seq_lens_this_time is None:
        # ~75% running: values in [1, 9]; ~25% stopped: value = 0
        seq_lens_this_time = rng.integers(1, 10, size=real_bsz, dtype=np.int32)
        n_stop = max(0, real_bsz // 4)
        if n_stop > 0:
            stop_idxs = rng.choice(real_bsz, size=n_stop, replace=False)
            seq_lens_this_time[stop_idxs] = 0

    is_running = seq_lens_this_time > 0

    # Build cu_seqlens_q_output: running slots contribute 1 token each
    tokens_per_slot = is_running.astype(np.int32)
    cu_seqlens_q_output = np.zeros(real_bsz + 1, dtype=np.int32)
    cu_seqlens_q_output[1:] = np.cumsum(tokens_per_slot)
    total_tokens = int(cu_seqlens_q_output[-1])

    # Sample tokens for each running slot (packed)
    next_tokens = rng.integers(0, 50000, size=max(total_tokens, 1), dtype=np.int64)

    # Pre-allocate accept_tokens/accept_num with noise
    accept_tokens = rng.integers(0, 100, size=(real_bsz, max_step_tokens), dtype=np.int64)
    accept_num = rng.integers(0, 5, size=real_bsz, dtype=np.int32)

    return {
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
        "seq_lens_this_time": seq_lens_this_time.copy(),
        "next_tokens": next_tokens,
        "cu_seqlens_q_output": cu_seqlens_q_output,
        # meta (not passed to kernel)
        "real_bsz": real_bsz,
        "max_step_tokens": max_step_tokens,
        "is_running": is_running,
    }


# ============================================================
# Layer 3: Reference implementation (1:1 with CUDA kernel)
# ============================================================


def reference_impl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference of naive_update_model_status_kernel.

    Guard: seq_lens_this_time[bid] > 0 (matches simplified kernel).
    """
    accept_tokens = inputs["accept_tokens"].copy()
    accept_num = inputs["accept_num"].copy()
    seq_lens_this_time = inputs["seq_lens_this_time"].copy()

    next_tokens = inputs["next_tokens"]
    cu_seqlens_q_output = inputs["cu_seqlens_q_output"]
    real_bsz = inputs["real_bsz"]

    for bid in range(real_bsz):
        if seq_lens_this_time[bid] > 0:
            # Write last (only) token for this slot
            accept_tokens[bid, 0] = next_tokens[cu_seqlens_q_output[bid + 1] - 1]
            accept_num[bid] = 1
            seq_lens_this_time[bid] = 1
        else:
            accept_num[bid] = 0
            seq_lens_this_time[bid] = 0

    return {
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
        "seq_lens_this_time": seq_lens_this_time,
    }


# ============================================================
# Layer 4a: TEST_CONFIGS
# ============================================================

TEST_CONFIGS = [
    {
        "name": "all_running",
        "real_bsz": 8,
        "max_step_tokens": 4,
        "seed": 42,
        "seq_lens_this_time": np.array([3, 5, 1, 7, 2, 4, 6, 8], dtype=np.int32),
    },
    {
        "name": "mixed_stop",
        "real_bsz": 8,
        "max_step_tokens": 4,
        "seed": 100,
        "seq_lens_this_time": np.array([1, 0, 3, 0, 5, 0, 2, 4], dtype=np.int32),
    },
    {
        "name": "all_stopped",
        "real_bsz": 4,
        "max_step_tokens": 4,
        "seed": 42,
        "seq_lens_this_time": np.zeros(4, dtype=np.int32),
    },
    {
        "name": "single_slot",
        "real_bsz": 1,
        "max_step_tokens": 4,
        "seed": 42,
        "seq_lens_this_time": np.array([1], dtype=np.int32),
    },
    {
        "name": "large_batch",
        "real_bsz": 64,
        "max_step_tokens": 8,
        "seed": 200,
        "seq_lens_this_time": None,  # randomly generated
    },
    {
        "name": "bsz_1024",
        "real_bsz": 1024,
        "max_step_tokens": 4,
        "seed": 42,
        "seq_lens_this_time": None,  # randomly generated
    },
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestNaiveUpdateModelStatus(unittest.TestCase):

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA")

    def _run_and_compare(self, inputs: Dict[str, Any]):
        """Run reference + kernel, compare all outputs."""
        ref = reference_impl(inputs)
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs)
        outputs = get_outputs(paddle_inputs)

        for key in OUTPUT_KEYS:
            np.testing.assert_array_equal(
                outputs[key],
                ref[key],
                err_msg=f"{key} mismatch",
            )

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                inputs = gen_inputs(**test_cfg)
                self._run_and_compare(inputs)

    def test_running_slots_get_token(self):
        """Running slots should have accept_tokens[bid, 0] = next_tokens[cu_q[bid+1]-1]."""
        seq_lens = np.array([1, 0, 1, 1], dtype=np.int32)
        inputs = gen_inputs(real_bsz=4, seed=42, seq_lens_this_time=seq_lens)
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        cu_q = inputs["cu_seqlens_q_output"]
        next_tokens = inputs["next_tokens"]
        for bid in range(4):
            if seq_lens[bid] > 0:
                expected_token = next_tokens[cu_q[bid + 1] - 1]
                self.assertEqual(ref["accept_tokens"][bid, 0], expected_token)
                self.assertEqual(ref["accept_num"][bid], 1)
                self.assertEqual(ref["seq_lens_this_time"][bid], 1)

    def test_stopped_slots_cleared(self):
        """Stopped slots (seq_lens_this_time=0): accept_num=0, seq_lens_this_time=0."""
        seq_lens = np.array([0, 1, 0, 1], dtype=np.int32)
        inputs = gen_inputs(real_bsz=4, seed=42, seq_lens_this_time=seq_lens)
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        for bid in [0, 2]:
            self.assertEqual(ref["accept_num"][bid], 0)
            self.assertEqual(ref["seq_lens_this_time"][bid], 0)

    def test_all_stopped(self):
        """All stopped: all accept_num=0, seq_lens_this_time=0."""
        seq_lens = np.zeros(8, dtype=np.int32)
        inputs = gen_inputs(real_bsz=8, seed=42, seq_lens_this_time=seq_lens)
        self._run_and_compare(inputs)
        ref = reference_impl(inputs)
        np.testing.assert_array_equal(ref["accept_num"], 0)
        np.testing.assert_array_equal(ref["seq_lens_this_time"], 0)

    def test_mixed_prefill_decode(self):
        """Mixed prefill+decode: stopped slots (seq_lens=0) get 0 tokens in packed next_tokens."""
        # slots 0,2 are decode (seq_lens > 0); slots 1,3 are stopped (seq_lens = 0)
        seq_lens = np.array([1, 0, 1, 0], dtype=np.int32)
        inputs = gen_inputs(real_bsz=4, seed=77, seq_lens_this_time=seq_lens)
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        for bid in [0, 2]:
            self.assertEqual(ref["accept_num"][bid], 1)
        for bid in [1, 3]:
            self.assertEqual(ref["accept_num"][bid], 0)

    def test_seq_lens_normalized_to_one(self):
        """Running slots with seq_lens_this_time > 1 are normalized to 1 after kernel."""
        # pre_process may set values > 1 for some slots; kernel normalizes to 1
        seq_lens = np.array([3, 7, 0, 5], dtype=np.int32)
        inputs = gen_inputs(real_bsz=4, seed=99, seq_lens_this_time=seq_lens)
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        for bid in [0, 1, 3]:
            self.assertEqual(ref["seq_lens_this_time"][bid], 1)
        self.assertEqual(ref["seq_lens_this_time"][2], 0)

    @pytest.mark.gpu
    def test_bsz_exceeds_block_size(self):
        """real_bsz > 1024 should raise."""
        with self.assertRaises(Exception):
            seq_lens = np.ones(1025, dtype=np.int32)
            inputs = gen_inputs(real_bsz=1025, seed=42, seq_lens_this_time=seq_lens)
            paddle_inputs = to_paddle_inputs(inputs)
            run_kernel(paddle_inputs)


if __name__ == "__main__":
    unittest.main()
