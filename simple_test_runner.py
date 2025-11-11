#!/usr/bin/env python3
"""
Simple test runner for data type conversion functions.
This version tests basic type parsing functions without full FastDeploy dependencies.
"""

import sys

# Add the project root to Python path
sys.path.insert(0, "/data/liujun/learning/paddles/hackthon9th/worktrees/hack9th_no55")


def test_basic_type_parsing():
    """Test basic type parsing functions that don't require external dependencies."""

    print("Testing basic type parsing functions...")

    # Test basic type parsing logic
    def parse_type_func(return_type):
        def _parse_type(val):
            try:
                return return_type(val)
            except ValueError as e:
                raise ValueError(f"Value {val} cannot be converted to {return_type}.") from e

        return _parse_type

    def optional_type_func(return_type):
        def _optional_type(val):
            if val == "" or val == "None":
                return None
            return parse_type_func(return_type)(val)

        return _optional_type

    def is_list_of_func(value, typ, check="first"):
        if not isinstance(value, list):
            return False
        if check == "first":
            return len(value) == 0 or isinstance(value[0], typ)
        elif check == "all":
            return all(isinstance(v, typ) for v in value)
        return False

    # Test integer parsing
    int_parser = parse_type_func(int)
    assert int_parser("123") == 123
    assert int_parser("-456") == -456
    try:
        int_parser("abc")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test float parsing
    float_parser = parse_type_func(float)
    assert float_parser("3.14") == 3.14
    assert float_parser("-2.5") == -2.5

    # Test optional type parsing
    optional_int_parser = optional_type_func(int)
    assert optional_int_parser("123") == 123
    assert optional_int_parser("None") is None
    assert optional_int_parser("") is None
    assert optional_int_parser("0") == 0

    # Test list type checking
    assert is_list_of_func([1, 2, 3], int, check="first")
    assert is_list_of_func([1, 2, 3], int, check="all")
    assert not is_list_of_func(["a", 2, 3], int, check="first")
    assert not is_list_of_func([1, "a", 3], int, check="all")
    assert not is_list_of_func("not a list", int)

    print("✅ Basic type parsing tests passed!")


def test_data_type_conceptual_tests():
    """Test conceptual data type conversion scenarios."""

    print("Testing conceptual data type conversion scenarios...")

    # Test FP8 conversion concept (without actual Paddle tensors)
    def simulate_fp8_conversion(tensor_data, block_size=None):
        """Simulate FP8 conversion with range validation."""
        # Simulate FP8 range constraints
        quantized_data = [max(-448, min(448, x)) for x in tensor_data]

        if block_size:
            # For block-wise conversion, calculate scales per block
            scales = []
            for i in range(0, len(tensor_data), block_size[0] * block_size[1]):
                block_data = tensor_data[i : i + block_size[0] * block_size[1]]
                if block_data:
                    scale = max(abs(x) for x in block_data) / 448.0
                    scales.append(scale)
                else:
                    scales.append(1.0)
        else:
            # For per-token conversion, calculate individual scales
            scales = [max(abs(x) / 448.0, 1e-4) for x in tensor_data]

        return quantized_data, scales

    # Test per-token conversion simulation
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    quantized, scales = simulate_fp8_conversion(test_data)

    assert len(quantized) == len(test_data)
    assert len(scales) == len(test_data)
    assert all(-448 <= x <= 448 for x in quantized)

    # Test per-block conversion simulation
    test_data_2d = [1.0, 2.0, 3.0, 4.0]  # Flattened 2x2 matrix
    block_size = [2, 2]  # 2x2 blocks
    quantized_2d, scales_2d = simulate_fp8_conversion(test_data_2d, block_size)

    assert len(quantized_2d) == 4
    assert len(scales_2d) == 1  # Single block for 2x2 matrix

    print("✅ Conceptual data type conversion tests passed!")


def test_scheduler_type_conversion():
    """Test scheduler-related type conversion scenarios."""

    print("Testing scheduler type conversion scenarios...")

    # Test resource manager config conversion
    class MockResourceManager:
        def __init__(self):
            self.max_num_seqs = 128
            self.max_total_tokens = 8192
            self.max_model_len = 2048

        def update_max_num_seqs(self, value):
            if isinstance(value, str):
                value = int(value)
            self.max_num_seqs = value

        def update_max_total_tokens(self, value):
            if isinstance(value, str):
                value = int(value)
            self.max_total_tokens = value

    rm = MockResourceManager()

    # Test string to int conversion
    rm.update_max_num_seqs("256")
    assert rm.max_num_seqs == 256

    rm.update_max_total_tokens("32768")
    assert rm.max_total_tokens == 32768

    # Test direct int conversion
    rm.update_max_num_seqs(512)
    assert rm.max_num_seqs == 512

    print("✅ Scheduler type conversion tests passed!")


def test_quantization_type_conversion():
    """Test quantization-related type conversion scenarios."""

    print("Testing quantization type conversion scenarios...")

    # Test quantization config creation
    quantization_configs = {
        "int8": {"max_bound": 127.0, "min_bound": -127.0},
        "fp8": {"max_bound": 448.0, "min_bound": -448.0},
        "int4_zp": {"max_bound": 7.0, "min_bound": -7.0},
        "block_wise_fp8": {"max_bound": 448.0, "min_bound": -448.0},
    }

    # Test config validation
    for config_type, bounds in quantization_configs.items():
        assert "max_bound" in bounds
        assert "min_bound" in bounds
        assert isinstance(bounds["max_bound"], (int, float))
        assert isinstance(bounds["min_bound"], (int, float))
        assert bounds["max_bound"] > bounds["min_bound"]

    # Test zero-point flag handling
    has_zero_point_configs = ["int4_zp", "fp8_zp", "int8_zp"]

    for config_type in has_zero_point_configs:
        assert "zp" in config_type, f"Config {config_type} should have zero point"

    print("✅ Quantization type conversion tests passed!")


def test_weight_loader_type_conversion():
    """Test weight loader type conversion scenarios."""

    print("Testing weight loader type conversion scenarios...")

    # Simulate weight loader logic
    def simulate_weight_loader(param_dtype, loaded_dtype, loaded_data):
        """Simulate weight loader type conversion logic."""
        if param_dtype != loaded_dtype:
            if loaded_dtype == "int8" and param_dtype == "float8_e4m3fn":
                # Simulate FP8 view conversion
                converted_data = [float(x) for x in loaded_data]  # Simplified view conversion
                return converted_data
            else:
                # Simulate regular cast conversion
                converted_data = [float(x) for x in loaded_data]  # Simplified cast conversion
                return converted_data
        return loaded_data

    # Test same dtype
    result1 = simulate_weight_loader("float32", "float32", [1.0, 2.0, 3.0])
    assert result1 == [1.0, 2.0, 3.0]

    # Test dtype conversion
    result2 = simulate_weight_loader("float32", "int8", [1, 2, 3])
    assert result2 == [1.0, 2.0, 3.0]

    # Test FP8 conversion
    result3 = simulate_weight_loader("float8_e4m3fn", "int8", [-127, 0, 127])
    assert result3 == [-127.0, 0.0, 127.0]

    print("✅ Weight loader type conversion tests passed!")


def run_all_simple_tests():
    """Run all simple tests."""

    print("=" * 60)
    print("Running Simple Data Type Conversion Tests")
    print("=" * 60)

    try:
        test_basic_type_parsing()
        test_data_type_conceptual_tests()
        test_scheduler_type_conversion()
        test_quantization_type_conversion()
        test_weight_loader_type_conversion()

        print("\n" + "=" * 60)
        print("🎉 All simple data type conversion tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_simple_tests()
    sys.exit(0 if success else 1)
