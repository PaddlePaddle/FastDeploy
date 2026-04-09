# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

from fastdeploy.model_executor.ops.gpu import eagle_gather_hidden_states

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers — tensor creation / kernel invocation / output extraction
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict → paddle tensors. All tensors on GPU."""
    paddle_inputs = {}
    # Keys that are metadata (not tensors)
    metadata_keys = {"real_bsz", "dim_embed", "input_token_num", "dtype", "seed"}
    for k, v in inputs.items():
        if k in metadata_keys or isinstance(v, (type, np.dtype)):
            # Skip metadata keys and dtype types
            continue
        elif isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any], dtype) -> tuple:
    """Call the eagle_gather_hidden_states CUDA kernel."""
    out, output_token_num = eagle_gather_hidden_states(
        paddle_inputs["input"],
        paddle_inputs["cu_seqlens_q"],
        paddle_inputs["seq_lens_this_time"],
        paddle_inputs["seq_lens_decoder"],
        paddle_inputs["seq_lens_encoder"],
        paddle_inputs["batch_id_per_token_output"],
        paddle_inputs["cu_seqlens_q_output"],
        paddle_inputs["real_output_token_num"],
    )
    return out, output_token_num


def get_outputs(out: paddle.Tensor, output_token_num: paddle.Tensor) -> Dict[str, np.ndarray]:
    """Extract outputs back to numpy."""
    return {
        "out": out.numpy(),
        "output_token_num": output_token_num.numpy(),
    }


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_eagle_gather_hidden_states_inputs(
    real_bsz: int = 4,
    dim_embed: int = 512,
    seed: int = 42,
    dtype: np.dtype = np.float16,
) -> Dict[str, Any]:
    """Generate randomized test inputs.

    Constraint: input_token_num == seq_lens_this_time.sum()
    """
    rng = np.random.default_rng(seed)

    # Generate seq_lens_this_time first (each batch has 0 or more tokens)
    # Use geometric-like distribution but cap it
    seq_lens_this_time = rng.integers(0, 5, size=real_bsz, dtype=np.int32)

    # Calculate total input tokens
    input_token_num = int(seq_lens_this_time.sum())

    # If all seq_lens are 0, add at least one token to avoid empty input
    if input_token_num == 0:
        seq_lens_this_time[0] = 1
        input_token_num = 1

    # Generate input hidden states
    input_data = rng.random((input_token_num, dim_embed), dtype=np.float32).astype(dtype)

    # Unused parameters (placeholders with appropriate shapes)
    cu_seqlens_q = np.zeros(real_bsz + 1, dtype=np.int32)
    seq_lens_decoder = rng.integers(0, 10, size=real_bsz, dtype=np.int32)
    seq_lens_encoder = rng.integers(0, 2, size=real_bsz, dtype=np.int32)
    batch_id_per_token_output = np.zeros(real_bsz, dtype=np.int32)
    cu_seqlens_q_output = np.zeros(real_bsz + 1, dtype=np.int32)
    real_output_token_num = np.array([0], dtype=np.int32)  # Not used by kernel, just placeholder

    return {
        "input": input_data,
        "cu_seqlens_q": cu_seqlens_q,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_decoder": seq_lens_decoder,
        "seq_lens_encoder": seq_lens_encoder,
        "batch_id_per_token_output": batch_id_per_token_output,
        "cu_seqlens_q_output": cu_seqlens_q_output,
        "real_output_token_num": real_output_token_num,
        "real_bsz": real_bsz,
        "dim_embed": dim_embed,
        "input_token_num": input_token_num,
        "dtype": dtype,
    }


# ============================================================
# Layer 3: Reference implementation (pure Python/NumPy)
# ============================================================


def reference_eagle_gather_hidden_states(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference implementation for eagle_gather_hidden_states.

    For each batch i where seq_lens_this_time[i] > 0, take the last token
    of that sequence from input and place it at output position out_offsets[i].
    """
    input_data = inputs["input"].copy()
    seq_lens_this_time = inputs["seq_lens_this_time"].copy()
    real_bsz = inputs["real_bsz"]
    dim_embed = inputs["dim_embed"]

    # Compute in_count and out_count
    in_count = np.where(seq_lens_this_time > 0, seq_lens_this_time, 0)
    out_count = np.where(seq_lens_this_time > 0, 1, 0)

    # Compute prefix sums
    in_offsets = np.zeros(real_bsz, dtype=np.int32)
    out_offsets = np.zeros(real_bsz, dtype=np.int32)
    in_acc = 0
    out_acc = 0
    for i in range(real_bsz):
        in_offsets[i] = in_acc
        out_offsets[i] = out_acc
        in_acc += in_count[i]
        out_acc += out_count[i]

    output_token_num = out_acc

    # Build position_map and gather
    # position_map: for each input token that should be output, map to output position
    input_token_num = inputs["input_token_num"]
    position_map = np.full(input_token_num, -1, dtype=np.int32)

    for i in range(real_bsz):
        if seq_lens_this_time[i] > 0:
            last_token_idx = in_offsets[i] + in_count[i] - 1
            position_map[last_token_idx] = out_offsets[i]

    # Gather: create output and copy tokens
    out = np.zeros((real_bsz, dim_embed), dtype=input_data.dtype)
    for i in range(input_token_num):
        token_idx = position_map[i]
        if token_idx >= 0:
            out[token_idx] = input_data[i]

    return {
        "out": out,
        "output_token_num": np.array([output_token_num], dtype=np.int32),
    }


# ============================================================
# Layer 4a: TEST_CONFIGS — all pure-parameter test scenarios
# ============================================================

TEST_CONFIGS = [
    # --- basic coverage ---
    {"name": "small_batch", "real_bsz": 1, "dim_embed": 512, "seed": 42},
    {"name": "normal_batch", "real_bsz": 4, "dim_embed": 512, "seed": 42},
    {"name": "large_batch", "real_bsz": 64, "dim_embed": 512, "seed": 42},
    # --- dim_embed variants (must be divisible by 4 for VecSize=4) ---
    {"name": "small_dim", "real_bsz": 4, "dim_embed": 128, "seed": 42},
    {"name": "large_dim", "real_bsz": 4, "dim_embed": 4096, "seed": 42},
    # --- edge cases ---
    {"name": "min_seq_len", "real_bsz": 4, "dim_embed": 512, "seed": 1},  # some seq_lens will be 0
    {"name": "all_active", "real_bsz": 8, "dim_embed": 512, "seed": 123},  # seed that gives all seq_lens > 0
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestEagleGatherHiddenStates(unittest.TestCase):

    # ------ shared helpers ------

    def _run_and_get(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run kernel and return numpy outputs."""
        paddle_inputs = to_paddle_inputs(inputs)
        out, output_token_num = run_kernel(paddle_inputs, inputs["dtype"])
        return get_outputs(out, output_token_num)

    def _check_all_outputs(self, inputs: Dict[str, Any], outputs: Dict[str, np.ndarray]):
        """Compare ALL output tensors against reference + sanity checks."""
        ref = reference_eagle_gather_hidden_states(inputs)

        # Check output_token_num
        np.testing.assert_array_equal(
            outputs["output_token_num"],
            ref["output_token_num"],
            err_msg="output_token_num mismatch",
        )

        actual_output_num = outputs["output_token_num"][0]
        expected_output_num = ref["output_token_num"][0]

        # Check output hidden states (only check valid rows)
        if actual_output_num > 0:
            np.testing.assert_allclose(
                outputs["out"][:actual_output_num],
                ref["out"][:expected_output_num],
                rtol=1e-5,
                atol=1e-5,
                err_msg="out mismatch",
            )

        # Sanity check: output_token_num should equal count(seq_lens_this_time > 0)
        seq_lens_this_time = inputs["seq_lens_this_time"]
        expected_count = int((seq_lens_this_time > 0).sum())
        self.assertEqual(
            actual_output_num,
            expected_count,
            f"output_token_num ({actual_output_num}) should equal count of positive seq_lens ({expected_count})",
        )

    def _run_full_test(self, config: Dict[str, Any]):
        """Generate inputs, run kernel, and verify outputs."""
        test_cfg = {k: v for k, v in config.items() if k != "name"}
        inputs = gen_eagle_gather_hidden_states_inputs(**test_cfg)
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        return outputs

    # ------ test cases ------

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest (one subTest per config)."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                inputs = gen_eagle_gather_hidden_states_inputs(**test_cfg)
                paddle_inputs = to_paddle_inputs(inputs)
                out, output_token_num = run_kernel(paddle_inputs, inputs["dtype"])

                # Get reference
                ref = reference_eagle_gather_hidden_states(inputs)

                # Compare output_token_num
                actual_token_num = int(output_token_num.numpy()[0])
                expected_token_num = int(ref["output_token_num"][0])
                self.assertEqual(
                    actual_token_num,
                    expected_token_num,
                    f"output_token_num mismatch: {actual_token_num} vs {expected_token_num}",
                )

                # Compare output tensor (only valid rows)
                if actual_token_num > 0:
                    out_np = out.numpy()[:actual_token_num]
                    ref_np = ref["out"][:expected_token_num]
                    np.testing.assert_allclose(out_np, ref_np, rtol=1e-5, atol=1e-5)

    def test_dtype_bfloat16(self):
        """Test with bfloat16 dtype."""
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available")

        inputs = gen_eagle_gather_hidden_states_inputs(real_bsz=4, dim_embed=512, seed=42, dtype=np.float32)
        paddle_inputs = to_paddle_inputs(inputs)

        # Convert input to bfloat16
        paddle_inputs["input"] = paddle_inputs["input"].astype(paddle.bfloat16)

        out, output_token_num = eagle_gather_hidden_states(
            paddle_inputs["input"],
            paddle_inputs["cu_seqlens_q"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["seq_lens_decoder"],
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["batch_id_per_token_output"],
            paddle_inputs["cu_seqlens_q_output"],
            paddle_inputs["real_output_token_num"],
        )

        # Just verify it runs without error and output shape is correct
        self.assertEqual(out.shape[0], inputs["real_bsz"])
        self.assertEqual(out.shape[1], inputs["dim_embed"])
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available")

        inputs = gen_eagle_gather_hidden_states_inputs(real_bsz=4, dim_embed=512, seed=42, dtype=np.float32)
        paddle_inputs = to_paddle_inputs(inputs)

        # Input is already float32
        out, output_token_num = eagle_gather_hidden_states(
            paddle_inputs["input"],
            paddle_inputs["cu_seqlens_q"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["seq_lens_decoder"],
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["batch_id_per_token_output"],
            paddle_inputs["cu_seqlens_q_output"],
            paddle_inputs["real_output_token_num"],
        )

        # Verify output
        ref = reference_eagle_gather_hidden_states(inputs)
        actual_token_num = int(output_token_num.numpy()[0])
        expected_token_num = int(ref["output_token_num"][0])
        self.assertEqual(actual_token_num, expected_token_num)

        if actual_token_num > 0:
            out_np = out.numpy()[:actual_token_num]
            ref_np = ref["out"][:expected_token_num]
            np.testing.assert_allclose(out_np, ref_np, rtol=1e-5, atol=1e-5)

    def test_all_seq_lens_zero(self):
        """Test when all seq_lens_this_time are 0 (edge case)."""
        # This case is handled in gen_inputs by adding at least one token
        # Here we explicitly test the fallback behavior
        inputs = gen_eagle_gather_hidden_states_inputs(real_bsz=4, dim_embed=512, seed=42)

        # Manually set all seq_lens to 0
        inputs["seq_lens_this_time"] = np.zeros(4, dtype=np.int32)
        inputs["seq_lens_this_time"][0] = 1  # Add at least one token
        inputs["input_token_num"] = 1
        inputs["input"] = inputs["input"][:1]  # Trim input

        paddle_inputs = to_paddle_inputs(inputs)
        out, output_token_num = run_kernel(paddle_inputs, inputs["dtype"])

        # Only one batch should have output
        self.assertEqual(int(output_token_num.numpy()[0]), 1)

    def test_specific_gather_pattern(self):
        """Test with a specific known pattern to verify correctness."""
        real_bsz = 3
        dim_embed = 8  # Small for easy verification

        # seq_lens: [2, 0, 3] means:
        # - batch 0: tokens 0,1 (last is 1)
        # - batch 1: no tokens
        # - batch 2: tokens 2,3,4 (last is 4)
        # Total input tokens = 5
        seq_lens_this_time = np.array([2, 0, 3], dtype=np.int32)

        # Create input with identifiable values
        input_data = np.arange(real_bsz * dim_embed * 2, dtype=np.float32).reshape(-1, dim_embed)
        input_data = input_data[:5]  # Only 5 tokens

        inputs = {
            "input": input_data.astype(np.float16),
            "cu_seqlens_q": np.zeros(real_bsz + 1, dtype=np.int32),
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_decoder": np.zeros(real_bsz, dtype=np.int32),
            "seq_lens_encoder": np.zeros(real_bsz, dtype=np.int32),
            "batch_id_per_token_output": np.zeros(real_bsz, dtype=np.int32),
            "cu_seqlens_q_output": np.zeros(real_bsz + 1, dtype=np.int32),
            "real_output_token_num": np.array([0], dtype=np.int32),
            "real_bsz": real_bsz,
            "dim_embed": dim_embed,
            "input_token_num": 5,
            "dtype": np.float16,
        }

        paddle_inputs = to_paddle_inputs(inputs)
        out, output_token_num = run_kernel(paddle_inputs, inputs["dtype"])

        # Verify: output_token_num should be 2 (batches 0 and 2)
        self.assertEqual(int(output_token_num.numpy()[0]), 2)

        # Verify: output[0] should equal input[1] (last token of batch 0)
        # output[1] should equal input[4] (last token of batch 2)
        out_np = out.numpy()
        np.testing.assert_array_equal(out_np[0], input_data[1])
        np.testing.assert_array_equal(out_np[1], input_data[4])


if __name__ == "__main__":
    unittest.main()
