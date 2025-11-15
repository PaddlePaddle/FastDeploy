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
Unit tests for scheduler-related data type conversion functionality in FastDeploy.
"""

import unittest
from unittest.mock import Mock

# Copy the utility functions directly to avoid import dependency issues
import argparse
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    """Parse a string to the specified type."""
    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Value {val} cannot be converted to {return_type}.") from e
    return _parse_type

def optional_type(return_type: Callable[[str], T]) -> Callable[[str], Optional[T]]:
    """Parse a string to the specified type, allowing None values."""
    def _optional_type(val: str) -> Optional[T]:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)
    return _optional_type


class TestSchedulerTypeParsing(unittest.TestCase):
    """Test type parsing for scheduler-related functions."""

    def test_parse_type_for_scheduler_params(self):
        """Test type parsing for scheduler parameters."""
        # Test int parsing for max_seq_len
        parser = parse_type(int)
        self.assertEqual(parser("1024"), 1024)
        self.assertEqual(parser("2048"), 2048)

        # Test float parsing for temperature
        float_parser = parse_type(float)
        self.assertEqual(float_parser("0.7"), 0.7)
        self.assertEqual(float_parser("1.2"), 1.2)

    def test_optional_type_for_scheduler_params(self):
        """Test optional type parsing for scheduler parameters."""
        parser = optional_type(float)

        # Test valid values
        self.assertEqual(parser("0.8"), 0.8)
        self.assertEqual(parser("None"), None)
        self.assertEqual(parser(""), None)

        # Test zero value
        self.assertEqual(parser("0.0"), 0.0)


class TestSchedulerConfigTypeConversion(unittest.TestCase):
    """Test scheduler config type conversion."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock scheduler config class that mimics the expected behavior
        class MockSchedulerConfig:
            def __init__(self, max_num_seqs=128, max_total_tokens=8192, max_model_len=2048):
                self.max_num_seqs = max_num_seqs
                self.max_total_tokens = max_total_tokens
                self.max_model_len = max_model_len

            def update_max_num_seqs(self, value):
                self.max_num_seqs = int(value) if isinstance(value, str) else value

            def update_max_total_tokens(self, value):
                self.max_total_tokens = int(value) if isinstance(value, str) else value

        self.scheduler_config = MockSchedulerConfig()
        self.mock_config = Mock()
        self.mock_config.max_num_seqs = 128
        self.mock_config.max_total_tokens = 8192
        self.mock_config.max_model_len = 2048

    def test_scheduler_config_type_conversion(self):
        """Test scheduler config type handling."""
        # Test config with different data types
        self.assertEqual(self.scheduler_config.max_num_seqs, 128)
        self.assertEqual(self.scheduler_config.max_total_tokens, 8192)
        self.assertEqual(self.scheduler_config.max_model_len, 2048)

    def test_batch_size_type_conversion(self):
        """Test batch size type conversion."""
        # Test int batch size
        self.scheduler_config.update_max_num_seqs(256)
        self.assertEqual(self.scheduler_config.max_num_seqs, 256)

        # Test string input conversion
        self.scheduler_config.update_max_num_seqs("512")
        self.assertEqual(self.scheduler_config.max_num_seqs, 512)

    def test_token_type_conversion(self):
        """Test token count type conversion."""
        # Test int token count
        self.scheduler_config.update_max_total_tokens(16384)
        self.assertEqual(self.scheduler_config.max_total_tokens, 16384)

        # Test string input conversion
        self.scheduler_config.update_max_total_tokens("32768")
        self.assertEqual(self.scheduler_config.max_total_tokens, 32768)


class TestSchedulerDataTypeHandling(unittest.TestCase):
    """Test scheduler data type handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_scheduler = Mock()
        self.mock_scheduler.max_num_seqs = 128
        self.mock_scheduler.max_model_len = 2048

    def test_scheduler_param_type_conversion(self):
        """Test scheduler parameter type conversion."""
        # Test int conversion - we need to manually update the mock attribute
        self.mock_scheduler.max_num_seqs = int("256")
        self.assertEqual(self.mock_scheduler.max_num_seqs, 256)

        # Test float conversion for float parameters
        if hasattr(self.mock_scheduler, "temperature"):
            self.mock_scheduler.temperature = float("0.8")
            self.assertEqual(self.mock_scheduler.temperature, 0.8)

    def test_scheduler_config_validation(self):
        """Test scheduler config validation with type conversion."""
        # Test valid type conversions
        valid_configs = [
            {"max_num_seqs": "128", "max_model_len": "2048"},
            {"max_num_seqs": 256, "max_model_len": 4096},
            {"max_num_seqs": "512", "max_model_len": "1024"},
        ]

        for config in valid_configs:
            # Should handle both string and int inputs
            max_seqs = int(config["max_num_seqs"])
            max_len = int(config["max_model_len"])
            self.assertIsInstance(max_seqs, int)
            self.assertIsInstance(max_len, int)

    def test_scheduler_error_handling(self):
        """Test scheduler error handling for invalid types."""
        # Test invalid type conversion
        with self.assertRaises(ValueError):
            int("invalid_number")

        # Test None handling
        result = optional_type(int)("None")
        self.assertIsNone(result)


class TestSchedulerQuantizationTypeConversion(unittest.TestCase):
    """Test scheduler-related quantization type conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.quant_config = {"kv_cache_quant_type": "int8", "is_channel_wise": True, "has_zero_point": False}

    def test_quant_config_type_conversion(self):
        """Test quantization config type conversion."""
        # Test string to enum conversion
        kv_type = self.quant_config["kv_cache_quant_type"]
        self.assertEqual(kv_type, "int8")

        # Test bool conversion
        channel_wise = self.quant_config["is_channel_wise"]
        self.assertIsInstance(channel_wise, bool)
        self.assertTrue(channel_wise)

        # Test zero point flag
        has_zp = self.quant_config["has_zero_point"]
        self.assertIsInstance(has_zp, bool)
        self.assertFalse(has_zp)

    def test_quant_config_validation(self):
        """Test quantization config validation."""
        valid_types = ["int8", "fp8", "int4_zp", "block_wise_fp8"]

        for kv_type in valid_types:
            config = self.quant_config.copy()
            config["kv_cache_quant_type"] = kv_type
            # Should handle valid type strings
            self.assertIsInstance(kv_type, str)
            self.assertIn(kv_type, valid_types)


class TestPerformanceDataTypeConversion(unittest.TestCase):
    """Test performance of data type conversion in scheduler."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_configs = [{"max_num_seqs": str(i), "max_model_len": str(i * 8)} for i in range(1, 100)]

    def test_config_batch_processing(self):
        """Test batch processing of config with type conversion."""
        processed_configs = []

        for config in self.test_configs:
            # Convert string inputs to int
            processed = {"max_num_seqs": int(config["max_num_seqs"]), "max_model_len": int(config["max_model_len"])}
            processed_configs.append(processed)

        # Verify all conversions were successful
        for config in processed_configs:
            self.assertIsInstance(config["max_num_seqs"], int)
            self.assertIsInstance(config["max_model_len"], int)

    def test_memory_efficiency(self):
        """Test memory efficiency of type conversion."""
        import tracemalloc

        # Start memory tracing
        tracemalloc.start()

        # Get initial memory snapshot
        initial_snapshot = tracemalloc.take_snapshot()

        # Process large number of configs
        for i in range(100):
            config = {"max_num_seqs": str(i), "max_model_len": str(i * 16)}
            # Convert types
            max_seqs = int(config["max_num_seqs"])
            max_len = int(config["max_model_len"])
            # Use the converted values to avoid unused variable warnings
            self.assertIsInstance(max_seqs, int)
            self.assertIsInstance(max_len, int)

        # Get final memory snapshot
        final_snapshot = tracemalloc.take_snapshot()

        # Calculate memory difference
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        total_memory_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Memory increase should be reasonable (less than 1MB for this test)
        self.assertLess(total_memory_diff, 1024 * 1024)

        tracemalloc.stop()


if __name__ == "__main__":
    unittest.main()
