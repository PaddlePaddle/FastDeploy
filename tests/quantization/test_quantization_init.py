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
Tests for quantization module initialization and parse_quant_config.
"""

import unittest

from fastdeploy.model_executor.layers.quantization import (
    _compute_hadamard_block_size,
    get_quantization_config,
)


class TestComputeHadamardBlockSize(unittest.TestCase):
    """Tests for _compute_hadamard_block_size function."""

    def test_basic_case(self):
        """Test basic computation."""
        result = _compute_hadamard_block_size(4096, 2)
        self.assertGreater(result, 0)
        self.assertTrue(result & (result - 1) == 0)  # Power of 2

    def test_not_divisible_raises(self):
        """Test that non-divisible moe_intermediate_size raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _compute_hadamard_block_size(4095, 2)
        self.assertIn("must be divisible", str(ctx.exception))


class TestGetQuantizationConfig(unittest.TestCase):
    """Tests for get_quantization_config function."""

    def test_valid_quantization_method(self):
        """Test getting config for valid quantization method."""
        for method in ["wint4", "wint8", "block_wise_fp8", "w4afp8"]:
            config_cls = get_quantization_config(method)
            self.assertIsNotNone(config_cls)

    def test_invalid_quantization_method_raises(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_quantization_config("invalid_method")
        self.assertIn("Invalid quantization method", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
