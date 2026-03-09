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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

paddle.seed(42)
np.random.seed(42)


def _unpack_int32(packed_np, num_bits):
    """Unpack int32 array into individual quantized values.

    Each int32 contains (32 // num_bits) quantized values packed from LSB.
    Returns a 1D sorted array of all quantized values (as uint32).
    """
    mask = np.uint32((1 << num_bits) - 1)
    pack_factor = 32 // num_bits
    flat = packed_np.flatten().astype(np.uint32)
    values = []
    for shift in range(pack_factor):
        values.append((flat >> np.uint32(shift * num_bits)) & mask)
    return np.sort(np.concatenate(values))


def _make_random_packed_weights(size_k, size_n, num_bits):
    """Create random int32-packed quantized weight tensor on GPU."""
    pack_factor = 32 // num_bits
    rows = size_k // pack_factor
    shape = (rows, size_n)
    data = np.random.randint(0, 2**32, size=shape, dtype=np.uint32).view(np.int32)
    return paddle.to_tensor(data, place=paddle.CUDAPlace(0))


def _make_perm(size_k, act_order=False):
    """Create perm tensor on GPU.

    If act_order=True, returns a random permutation of 0..size_k-1.
    If act_order=False, returns an empty int32 tensor of shape [0].
    """
    if act_order:
        perm = np.random.permutation(size_k).astype(np.int32)
    else:
        perm = np.zeros([0], dtype=np.int32)
    return paddle.to_tensor(perm, place=paddle.CUDAPlace(0))


def _expected_output_shape(size_k, size_n, num_bits):
    """Compute expected output shape: [size_k // 16, size_n * 16 / pack_factor]."""
    pack_factor = 32 // num_bits
    return [size_k // 16, size_n * 16 // pack_factor]


class TestGptqMarlinRepack(unittest.TestCase):
    """Unit tests for the gptq_marlin_repack custom operator."""

    def setUp(self):
        paddle.set_device("gpu")

    # ------------------------------------------------------------------
    # Numerical Correctness — Conservation (multiset invariant)
    # ------------------------------------------------------------------
    def test_conservation_4bit_no_perm(self):
        """Unpacked values from input and output must be identical
        multisets (4-bit, no act_order)."""
        size_k, size_n, num_bits = 64, 128, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)

        np.testing.assert_array_equal(
            input_vals,
            output_vals,
            err_msg="4-bit conservation failed: unpacked value multisets differ",
        )

    def test_conservation_8bit_no_perm(self):
        """Unpacked values from input and output must be identical
        multisets (8-bit, no act_order)."""
        size_k, size_n, num_bits = 64, 128, 8
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)

        np.testing.assert_array_equal(
            input_vals,
            output_vals,
            err_msg="8-bit conservation failed: unpacked value multisets differ",
        )

    def test_conservation_4bit_with_perm(self):
        """Conservation holds when act_order permutation is applied (4-bit)."""
        size_k, size_n, num_bits = 64, 128, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=True)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)

        np.testing.assert_array_equal(
            input_vals,
            output_vals,
            err_msg="4-bit with perm conservation failed",
        )

    def test_conservation_8bit_with_perm(self):
        """Conservation holds when act_order permutation is applied (8-bit)."""
        size_k, size_n, num_bits = 64, 128, 8
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=True)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)

        np.testing.assert_array_equal(
            input_vals,
            output_vals,
            err_msg="8-bit with perm conservation failed",
        )

    # ------------------------------------------------------------------
    # Shape Validation
    # ------------------------------------------------------------------
    def test_output_shape_4bit(self):
        """Output shape is [size_k // 16, size_n * 2] for 4-bit."""
        for size_k, size_n in [(16, 64), (64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                b_q_weight = _make_random_packed_weights(size_k, size_n, 4)
                perm = _make_perm(size_k, act_order=False)
                out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, 4)
                expected = _expected_output_shape(size_k, size_n, 4)
                self.assertEqual(list(out.shape), expected)

    def test_output_shape_8bit(self):
        """Output shape is [size_k // 16, size_n * 4] for 8-bit."""
        for size_k, size_n in [(16, 64), (64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                b_q_weight = _make_random_packed_weights(size_k, size_n, 8)
                perm = _make_perm(size_k, act_order=False)
                out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, 8)
                expected = _expected_output_shape(size_k, size_n, 8)
                self.assertEqual(list(out.shape), expected)

    def test_output_dtype_int32(self):
        """Output dtype must be int32."""
        size_k, size_n, num_bits = 16, 64, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)
        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        self.assertEqual(out.dtype, paddle.int32)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------
    def test_determinism_4bit(self):
        """Same inputs produce identical outputs across two runs (4-bit)."""
        size_k, size_n, num_bits = 64, 128, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out1 = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        out2 = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_determinism_8bit_with_perm(self):
        """Same inputs produce identical outputs across two runs
        (8-bit, act_order)."""
        size_k, size_n, num_bits = 64, 128, 8
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=True)

        out1 = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        out2 = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    # ------------------------------------------------------------------
    # Edge Sizes
    # ------------------------------------------------------------------
    def test_minimum_tile_4bit(self):
        """Minimum possible size (one tile: size_k=16, size_n=64) for 4-bit."""
        size_k, size_n, num_bits = 16, 64, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertEqual(list(out.shape), _expected_output_shape(size_k, size_n, num_bits))
        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)
        np.testing.assert_array_equal(input_vals, output_vals)

    def test_minimum_tile_8bit(self):
        """Minimum possible size (one tile: size_k=16, size_n=64) for 8-bit."""
        size_k, size_n, num_bits = 16, 64, 8
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertEqual(list(out.shape), _expected_output_shape(size_k, size_n, num_bits))
        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)
        np.testing.assert_array_equal(input_vals, output_vals)

    def test_larger_size_4bit_with_perm(self):
        """Larger matrix (size_k=128, size_n=256) with act_order (4-bit)."""
        size_k, size_n, num_bits = 128, 256, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=True)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertEqual(list(out.shape), _expected_output_shape(size_k, size_n, num_bits))
        input_vals = _unpack_int32(b_q_weight.numpy(), num_bits)
        output_vals = _unpack_int32(out.numpy(), num_bits)
        np.testing.assert_array_equal(input_vals, output_vals)

    # ------------------------------------------------------------------
    # Zero-input invariant
    # ------------------------------------------------------------------
    def test_zero_input_4bit(self):
        """All-zero packed weights must produce all-zero output (4-bit)."""
        size_k, size_n, num_bits = 32, 64, 4
        pack_factor = 32 // num_bits

        b_q_weight = paddle.zeros([size_k // pack_factor, size_n], dtype="int32").cuda()
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        np.testing.assert_array_equal(out.numpy(), np.zeros(out.shape, dtype=np.int32))

    def test_zero_input_8bit(self):
        """All-zero packed weights must produce all-zero output (8-bit)."""
        size_k, size_n, num_bits = 16, 64, 8
        pack_factor = 32 // num_bits

        b_q_weight = paddle.zeros([size_k // pack_factor, size_n], dtype="int32").cuda()
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)
        np.testing.assert_array_equal(out.numpy(), np.zeros(out.shape, dtype=np.int32))

    # ------------------------------------------------------------------
    # Uniform-value invariant
    # ------------------------------------------------------------------
    def test_uniform_value_4bit(self):
        """All quantized slots set to the same value must remain uniform."""
        size_k, size_n, num_bits = 16, 64, 4
        pack_factor = 32 // num_bits

        # Pack a uniform 4-bit value (0xA = 10) into every slot.
        uniform_val = np.uint32(0xA)
        packed = np.uint32(0)
        for p in range(pack_factor):
            packed |= uniform_val << np.uint32(p * num_bits)

        b_q_np = np.full((size_k // pack_factor, size_n), packed, dtype=np.uint32).view(np.int32)
        b_q_weight = paddle.to_tensor(b_q_np, place=paddle.CUDAPlace(0))
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        vals_out = _unpack_int32(out.numpy(), num_bits)
        # Every extracted value should equal uniform_val.
        self.assertTrue(
            np.all(vals_out == uniform_val),
            "Not all output values equal the uniform input value",
        )

    # ------------------------------------------------------------------
    # Non-trivial output
    # ------------------------------------------------------------------
    def test_output_not_all_zeros(self):
        """Output is not all zeros when input has non-zero values."""
        size_k, size_n, num_bits = 64, 128, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        self.assertFalse(
            np.all(out.numpy() == 0),
            "Output is all zeros — repacking likely failed",
        )

    def test_output_differs_from_input_layout(self):
        """Output raw int32 layout differs from input (repacking is
        non-trivial)."""
        size_k, size_n, num_bits = 64, 128, 4
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=False)

        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        # Flattened raw int32 arrays should differ (different packing layout)
        input_flat = b_q_weight.numpy().flatten()
        output_flat = out.numpy().flatten()
        common_len = min(len(input_flat), len(output_flat))
        self.assertFalse(
            np.array_equal(input_flat[:common_len], output_flat[:common_len]),
            "Output raw layout is identical to input — repacking may be a no-op",
        )


if __name__ == "__main__":
    unittest.main()
