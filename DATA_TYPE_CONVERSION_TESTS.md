# Data Type Conversion Unit Tests for FastDeploy

This directory contains comprehensive unit tests for data type conversion functionality in FastDeploy.

## Overview

The data type conversion tests cover the following key areas:

### 1. Core Type Conversion Utilities (`test_data_type_conversion.py`)
- **Type Parsing**: `parse_type()`, `optional_type()`, `is_list_of()`
- **Tensor Utilities**: `get_tensor()`, `temporary_dtype()`
- **FP8 Conversion**: `per_token_cast_to_fp8()`, `per_block_cast_to_fp8()`
- **Quantization Type Conversion**: Various quantization methods and configs
- **Weight Loader**: `default_weight_loader()` type handling
- **Edge Cases**: Invalid inputs, memory efficiency, accuracy tests

### 2. Scheduler Type Conversion (`test_scheduler_type_conversion.py`)
- **Resource Manager**: Config type handling and validation
- **Scheduler Parameters**: Batch size, token count, model length conversion
- **Quantization Config**: Type validation and conversion
- **Performance Tests**: Memory efficiency and batch processing

## Test Coverage

### Core Data Type Functions

#### Type Parsing and Validation
- ✅ Integer, float, boolean type parsing
- ✅ Optional type handling (None, empty strings)
- ✅ List type checking with "first" and "all" modes
- ✅ Error handling for invalid types

#### Tensor Conversion Functions
- ✅ Paddle tensor to tensor (no conversion)
- ✅ Numpy array to paddle tensor conversion
- ✅ String path to tensor (mocked)
- ✅ Context manager for temporary dtype changes

#### FP8 Quantization Functions
- ✅ Per-token FP8 conversion with range validation
- ✅ Per-block FP8 conversion with block size handling
- ✅ Scale tensor shape validation
- ✅ FP8 range constraint testing (-448 to +448)

#### Quantization Type Conversion
- ✅ KV cache quant config creation and validation
- ✅ Zero-point and scale tensor type casting
- ✅ Weight-only quantization config handling
- ✅ Tensor-wise FP8 method creation

#### Weight Loading Type Conversion
- ✅ Same dtype tensor copying
- ✅ Different dtype casting (int8 to float32)
- ✅ FP8 tensor view conversion
- ✅ Shape mismatch handling with reshape

### Scheduler-Specific Tests

#### Resource Management
- ✅ Config parameter type conversion
- ✅ String to integer conversion for batch sizes
- ✅ Token count type handling
- ✅ Model length parameter validation

#### Performance and Accuracy
- ✅ Batch processing of multiple configs
- ✅ Memory efficiency validation
- ✅ Type conversion consistency
- ✅ Large tensor handling

## Running Tests

### Prerequisites
- Python 3.8+
- PaddlePaddle
- NumPy
- Unittest (built-in)

### Running All Tests
```bash
python run_data_type_conversion_tests.py
```

### Running Specific Test File
```bash
python run_data_type_conversion_tests.py tests/test_data_type_conversion.py
python run_data_type_conversion_tests.py tests/test_scheduler_type_conversion.py
```

### Running Individual Test Classes
```bash
python -m unittest tests.test_data_type_conversion.TestTypeParsing
python -m unittest tests.test_data_type_conversion.TestTensorUtils
python -m unittest tests.test_data_type_conversion.TestFP8Conversion
```

## Test Files Structure

```
tests/
├── test_data_type_conversion.py          # Main data type conversion tests
├── test_scheduler_type_conversion.py      # Scheduler-specific tests
├── run_data_type_conversion_tests.py     # Test runner script
├── utils.py                             # Test utilities (base config)
└── DATA_TYPE_CONVERSION_TESTS.md        # This documentation
```

## Key Test Features

### 1. Comprehensive Coverage
- Type parsing and validation
- Tensor conversion utilities
- FP8 quantization functions
- Weight loading type handling
- Edge cases and error conditions
- Performance and memory tests

### 2. Realistic Test Data
- Uses actual FastDeploy data types and configurations
- Mocks complex components for isolated testing
- Tests both valid and invalid inputs
- Includes large tensors for performance testing

### 3. Error Handling
- Validates type conversion error cases
- Tests graceful handling of invalid inputs
- Verifies error messages and exception types
- Tests boundary conditions and edge cases

### 4. Performance Testing
- Memory efficiency validation
- Large batch processing capabilities
- Performance benchmarking for critical functions
- Stress testing with high-dimensional tensors

## Integration with FastDeploy

These tests are designed to integrate seamlessly with the FastDeploy testing framework:

1. **Import Structure**: Uses proper FastDeploy imports
2. **Mock Objects**: Mocks complex FastDeploy components
3. **Test Data**: Uses realistic FastDeploy configurations
4. **Error Handling**: Tests FastDeploy-specific error conditions
5. **API Compatibility**: Maintains compatibility with FastDeploy APIs

## Adding New Tests

When adding new data type conversion tests:

1. **Identify New Functions**: Find new data type conversion functions
2. **Create Test Classes**: Group related tests in logical classes
3. **Use Mocks**: Mock external dependencies when needed
4. **Test Edge Cases**: Include both valid and invalid inputs
5. **Performance Considerations**: Add performance tests for critical functions
6. **Documentation**: Update this README with new test coverage

## Maintaining Tests

- Regularly update tests when FastDeploy APIs change
- Add new tests for newly implemented type conversion functions
- Remove obsolete tests for deprecated functions
- Ensure all tests pass on CI/CD systems
- Monitor test execution time and performance metrics

