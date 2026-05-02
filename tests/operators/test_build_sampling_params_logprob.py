# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from typing import Any, Dict

import numpy as np
import paddle

# --- Import ops (bypass fastdeploy.__init__) ---
try:
    import os
    import sys

    _fd_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _fd_root not in sys.path:
        sys.path.insert(0, _fd_root)
    from fastdeploy.import_ops import import_custom_ops

    _package = "fastdeploy.model_executor.ops.gpu"
    import_custom_ops(_package, ".fastdeploy_ops", globals())
except ImportError as e:
    print(f"Import error: {e}")
    raise

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers -- tensor creation / kernel invocation / output extraction
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict -> paddle tensors on GPU. Scalar attrs are passed through."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs, inputs):
    """Call build_sampling_params_logprob with paddle tensors + scalar attrs."""
    return build_sampling_params_logprob(  # noqa: F821
        paddle_inputs["input_params"],
        paddle_inputs["token_num_per_batch"],
        inputs["token_num_output_cpu"],
    )


def get_outputs(result) -> Dict[str, np.ndarray]:
    """Extract output tensor to numpy."""
    return {"output_params": result.numpy()}


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_inputs(
    real_bsz=8,
    max_tokens_per_batch=5,
    dtype=np.float32,
    seed=42,
) -> Dict[str, Any]:
    """Generate randomized test inputs.

    Args:
        real_bsz: number of batch items
        max_tokens_per_batch: max token count per batch item
        dtype: numpy dtype for input_params (np.float32, np.int32, np.bool_)
        seed: random seed
    """
    rng = np.random.default_rng(seed)

    # Random token counts per batch, allow zeros (empty slots)
    token_num_per_batch = rng.integers(0, max_tokens_per_batch + 1, size=real_bsz).astype(np.int32)
    token_num_output_cpu = int(token_num_per_batch.sum())

    # Generate per-batch param values
    if dtype == np.float32:
        input_params = rng.uniform(0.0, 1.0, size=real_bsz).astype(np.float32)
    elif dtype == np.int32:
        input_params = rng.integers(0, 100, size=real_bsz).astype(np.int32)
    elif dtype == np.bool_:
        input_params = rng.choice([False, True], size=real_bsz)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return {
        "input_params": input_params,
        "token_num_per_batch": token_num_per_batch,
        "token_num_output_cpu": token_num_output_cpu,
    }


# ============================================================
# Layer 3: Reference implementation (pure Python/NumPy)
# ============================================================


def reference_build_sampling_params_logprob(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference -- must match CUDA kernel logic exactly.

    Kernel logic:
    1. Initialize output with safe defaults (bool->False, int32->1, float32->1.0)
    2. For each batch bi, fill output[start_offset..start_offset+cur_token_num-1]
       with input_params[bi], where start_offset = sum(token_num_per_batch[0..bi-1])
    """
    input_params = inputs["input_params"].copy()
    token_num_per_batch = inputs["token_num_per_batch"].copy()
    token_num_output_cpu = inputs["token_num_output_cpu"]
    real_bsz = len(input_params)
    dtype = input_params.dtype

    # Initialize output with safe defaults (matching kernel behavior)
    if dtype == np.bool_:
        output_params = np.full(token_num_output_cpu, False, dtype=dtype)
    elif dtype == np.int32:
        output_params = np.full(token_num_output_cpu, 1, dtype=dtype)
    elif dtype == np.float32:
        output_params = np.full(token_num_output_cpu, 1.0, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    for bi in range(real_bsz):
        start_offset = int(token_num_per_batch[:bi].sum())
        cur_token_num = int(token_num_per_batch[bi])
        if cur_token_num <= 0:
            continue
        val = input_params[bi]
        for i in range(cur_token_num):
            idx = start_offset + i
            if idx < token_num_output_cpu:
                output_params[idx] = val

    return {"output_params": output_params}


# ============================================================
# Layer 4a: TEST_CONFIGS -- all pure-parameter test scenarios
# ============================================================

TEST_CONFIGS = [
    # --- basic coverage, float32 ---
    {"name": "float32_small_batch", "real_bsz": 2, "max_tokens_per_batch": 3, "dtype": np.float32, "seed": 42},
    {"name": "float32_medium_batch", "real_bsz": 16, "max_tokens_per_batch": 8, "dtype": np.float32, "seed": 42},
    {"name": "float32_large_batch", "real_bsz": 64, "max_tokens_per_batch": 16, "dtype": np.float32, "seed": 42},
    # --- int32 dtype ---
    {"name": "int32_small_batch", "real_bsz": 4, "max_tokens_per_batch": 5, "dtype": np.int32, "seed": 42},
    {"name": "int32_large_batch", "real_bsz": 32, "max_tokens_per_batch": 10, "dtype": np.int32, "seed": 42},
    # --- bool dtype ---
    {"name": "bool_small_batch", "real_bsz": 4, "max_tokens_per_batch": 5, "dtype": np.bool_, "seed": 42},
    {"name": "bool_large_batch", "real_bsz": 32, "max_tokens_per_batch": 10, "dtype": np.bool_, "seed": 42},
    # --- edge cases ---
    {"name": "single_batch_single_token", "real_bsz": 1, "max_tokens_per_batch": 1, "dtype": np.float32, "seed": 42},
    {"name": "single_batch_many_tokens", "real_bsz": 1, "max_tokens_per_batch": 64, "dtype": np.float32, "seed": 42},
    {"name": "many_batch_one_token", "real_bsz": 64, "max_tokens_per_batch": 1, "dtype": np.float32, "seed": 42},
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestBuildSamplingParamLogprob(unittest.TestCase):

    # ------ shared helpers ------

    def _run_and_get(self, inputs):
        paddle_inputs = to_paddle_inputs(inputs)
        result = run_kernel(paddle_inputs, inputs)
        return get_outputs(result)

    def _check_all_outputs(self, inputs, outputs):
        """Compare ALL output tensors against reference."""
        ref = reference_build_sampling_params_logprob(inputs)
        np.testing.assert_array_equal(outputs["output_params"], ref["output_params"], err_msg="output_params mismatch")

    def _run_full_test(self, config):
        inputs = gen_inputs(**config)
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        return outputs

    # ------ test cases ------

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest (one subTest per config)."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                self._run_full_test(test_cfg)

    def test_all_zero_token_counts(self):
        """All batch items have zero tokens -- output should be empty array."""
        inputs = gen_inputs(real_bsz=4, max_tokens_per_batch=1, dtype=np.float32, seed=42)
        # Force all token counts to zero
        inputs["token_num_per_batch"] = np.zeros(4, dtype=np.int32)
        inputs["token_num_output_cpu"] = 0
        outputs = self._run_and_get(inputs)
        self.assertEqual(outputs["output_params"].size, 0)

    def test_exact_golden_float32(self):
        """Exact golden values for float32 -- hand-verified."""
        inputs = {
            "input_params": np.array([0.5, 0.9, 0.1], dtype=np.float32),
            "token_num_per_batch": np.array([2, 3, 1], dtype=np.int32),
            "token_num_output_cpu": 6,
        }
        outputs = self._run_and_get(inputs)
        expected = np.array([0.5, 0.5, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(outputs["output_params"], expected)

    def test_exact_golden_int32(self):
        """Exact golden values for int32 -- hand-verified."""
        inputs = {
            "input_params": np.array([10, 20, 30], dtype=np.int32),
            "token_num_per_batch": np.array([1, 2, 3], dtype=np.int32),
            "token_num_output_cpu": 6,
        }
        outputs = self._run_and_get(inputs)
        expected = np.array([10, 20, 20, 30, 30, 30], dtype=np.int32)
        np.testing.assert_array_equal(outputs["output_params"], expected)

    def test_exact_golden_bool(self):
        """Exact golden values for bool -- hand-verified."""
        inputs = {
            "input_params": np.array([True, False, True], dtype=np.bool_),
            "token_num_per_batch": np.array([3, 2, 1], dtype=np.int32),
            "token_num_output_cpu": 6,
        }
        outputs = self._run_and_get(inputs)
        expected = np.array([True, True, True, False, False, True], dtype=np.bool_)
        np.testing.assert_array_equal(outputs["output_params"], expected)

    def test_mixed_with_empty_slots(self):
        """Some batch items have zero tokens (empty slots)."""
        inputs = {
            "input_params": np.array([0.5, 0.9, 0.1, 0.7], dtype=np.float32),
            "token_num_per_batch": np.array([2, 0, 3, 0], dtype=np.int32),
            "token_num_output_cpu": 5,
        }
        outputs = self._run_and_get(inputs)
        # bi=0: tokens 0,1 -> 0.5; bi=1: empty; bi=2: tokens 2,3,4 -> 0.1; bi=3: empty
        expected = np.array([0.5, 0.5, 0.1, 0.1, 0.1], dtype=np.float32)
        np.testing.assert_array_equal(outputs["output_params"], expected)


if __name__ == "__main__":
    unittest.main()
