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
    """Unpack int32 array into individual quantized values (sorted)."""
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
    data = np.random.randint(0, 2**32, size=(size_k // pack_factor, size_n), dtype=np.uint32).view(np.int32)
    return paddle.to_tensor(data, place=paddle.CUDAPlace(0))


def _make_perm(size_k, act_order=False):
    """Create perm tensor (random permutation if act_order, else empty)."""
    if act_order:
        return paddle.to_tensor(np.random.permutation(size_k).astype(np.int32), place=paddle.CUDAPlace(0))
    return paddle.to_tensor(np.zeros([0], dtype=np.int32), place=paddle.CUDAPlace(0))


class TestGptqMarlinRepack(unittest.TestCase):
    """Tests for gptq_marlin_repack — value conservation across repacking."""

    def setUp(self):
        paddle.set_device("gpu")

    def _check_conservation(self, size_k, size_n, num_bits, act_order=False):
        """Verify unpacked value multisets are identical before and after repack."""
        b_q_weight = _make_random_packed_weights(size_k, size_n, num_bits)
        perm = _make_perm(size_k, act_order=act_order)
        out = gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        expected_shape = [size_k // 16, size_n * 16 // (32 // num_bits)]
        self.assertEqual(list(out.shape), expected_shape)
        self.assertEqual(out.dtype, paddle.int32)

        np.testing.assert_array_equal(
            _unpack_int32(b_q_weight.numpy(), num_bits),
            _unpack_int32(out.numpy(), num_bits),
        )

    def test_4bit_no_perm(self):
        """4-bit repacking without act_order, multiple sizes."""
        for size_k, size_n in [(16, 64), (64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                self._check_conservation(size_k, size_n, 4, act_order=False)

    def test_8bit_no_perm(self):
        """8-bit repacking without act_order, multiple sizes."""
        for size_k, size_n in [(16, 64), (64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                self._check_conservation(size_k, size_n, 8, act_order=False)

    def test_4bit_with_perm(self):
        """4-bit repacking with act_order permutation."""
        for size_k, size_n in [(64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                self._check_conservation(size_k, size_n, 4, act_order=True)

    def test_8bit_with_perm(self):
        """8-bit repacking with act_order permutation."""
        for size_k, size_n in [(64, 128), (128, 256)]:
            with self.subTest(size_k=size_k, size_n=size_n):
                self._check_conservation(size_k, size_n, 8, act_order=True)


if __name__ == "__main__":
    unittest.main()
