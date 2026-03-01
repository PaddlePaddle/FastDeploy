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
from unittest.mock import Mock, patch

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


class TestParseQuantConfigFallback(unittest.TestCase):
    """Tests for parse_quant_config fallback branch (lines 95-96)."""

    def test_parse_quant_config_with_valid_quantization(self):
        """Test parse_quant_config with valid quantization dict."""
        from fastdeploy.model_executor.layers.quantization import parse_quant_config

        # Create mock args
        args = Mock()
        args.quantization = {"quantization": "wint4"}
        args.dynamic_load_weight = False

        # Create mock model_config
        model_config = Mock()
        model_config.model_format = "paddle"
        model_config.quantization_config = None
        model_config.is_quantized = False

        # Call parse_quant_config
        with patch("fastdeploy.model_executor.layers.quantization.get_quantization_config") as mock_get_config:
            mock_config_cls = Mock()
            mock_config_instance = Mock()
            mock_config_cls.from_config.return_value = mock_config_instance
            mock_get_config.return_value = mock_config_cls

            result = parse_quant_config(args, model_config, is_ernie=False, is_v1_loader=False)

            # Verify the function was called correctly
            mock_get_config.assert_called_once()

    def test_parse_quant_config_fallback_on_update_error(self):
        """Test fallback when quantization_config.update raises exception (lines 95-96)."""
        from fastdeploy.model_executor.layers.quantization import parse_quant_config

        # Create mock args with a quantization that will cause {}.update(obj) to fail
        # but obj["quantization"] to succeed. We need an object whose __iter__ raises
        # (so dict.update fails) but __getitem__ works.
        args = Mock()

        class NonIterableMapping:
            """Object that supports [] access but fails when iterated (as dict.update does)."""

            def __getitem__(self, key):
                if key == "quantization":
                    return "wint4"
                raise KeyError(key)

            def keys(self):
                raise TypeError("Simulated iteration failure")

            def __iter__(self):
                raise TypeError("Simulated iteration failure")

        args.quantization = NonIterableMapping()
        args.dynamic_load_weight = False

        # Create mock model_config
        model_config = Mock()
        model_config.model_format = "paddle"
        model_config.quantization_config = None
        model_config.is_quantized = False

        # Call parse_quant_config - should use fallback path
        with patch("fastdeploy.model_executor.layers.quantization.get_quantization_config") as mock_get_config:
            mock_config_cls = Mock()
            mock_config_instance = Mock()
            mock_config_cls.from_config.return_value = mock_config_instance
            mock_get_config.return_value = mock_config_cls

            result = parse_quant_config(args, model_config, is_ernie=False, is_v1_loader=False)

            # Verify fallback was used and quantization was extracted
            mock_get_config.assert_called_once_with("wint4")


if __name__ == "__main__":
    unittest.main()
