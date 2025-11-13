# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastDeploy is a high-performance inference and deployment toolkit for Large Language Models (LLMs) and Vision Language Models (VLMs) based on PaddlePaddle. It provides production-ready deployment solutions with advanced optimization techniques like load-balanced PD disaggregation, unified KV cache transmission, and comprehensive quantization support.

## Add unit test

- 添加单元测试的时候，参考 tests/multimodal/test_multimodal_utils.py
- 跳过验证 log 功能
- 代码写完以后，需要执行: pre-commit run --files <test file> 进行代码检查，确保代码风格统一。

## Development Commands

### Build and Installation

```bash
# Build FastDeploy wheel (includes custom ops)
./build.sh 1 python3.10

# Build only custom ops without wheel
./build.sh 0 python3.10

# Build with specific CUDA architectures
export FD_BUILDING_ARCS="[80, 90, 100]"
./build.sh 1 python3.10

# Build with CPU BF16 support
FD_CPU_USE_BF16=true ./build.sh 1 python3.10

# Install FastDeploy with specific device support
pip install fastdeploy-gpu          # NVIDIA GPU
pip install fastdeploy-xpu          # Kunlunxin XPU
pip install fastdeploy-npu          # Ascend NPU
pip install fastdeploy-cpu          # CPU only
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/v1/test_schedule_output.py

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/v1/cache_manager/test_prefix_cache.py
```

### Code Quality

```bash
# Format code with Black
black fastdeploy/ tests/

# Sort imports with isort
isort fastdeploy/ tests/

# Lint with ruff
ruff check fastdeploy/ tests/

# Format with ruff
ruff format fastdeploy/ tests/
```

## Architecture Overview

### Core Components

1. **Model Executor** (`fastdeploy/model_executor/`): Core model execution engine with device-specific optimizations

   - `ops/`: Custom operators for GPU/CPU/XPU/NPU devices
   - `models/`: Model implementations and wrappers
   - `layers/`: Neural network layer implementations

2. **Cache Management** (`fastdeploy/cache_manager/`): KV cache management and transfer

   - Prefix caching for context reuse
   - KV cache transmission across devices (NVLink/RDMA)
   - Memory block management

3. **Engine Core** (`fastdeploy/engine/`): Main inference engine

   - Request scheduling and batch management
   - Resource management and allocation
   - Graph optimization and CUDA graph support

4. **Scheduler** (`fastdeploy/scheduler/`): Request scheduling algorithms

   - Load balancing and priority management
   - Batch formation strategies

5. **Input Processing** (`fastdeploy/input/`): Multi-modal input preprocessing

   - ERNIE-4.5-VL processor (`ernie4_5_vl_processor/`)
   - Qwen-VL processor (`qwen_vl_processor/`)
   - PaddleOCR-VL processor (`paddleocr_vl_processor/`)

6. **Device Platforms** (`fastdeploy/platforms/`): Hardware-specific implementations
   - CUDA, ROCm, XPU, NPU, CPU, Intel HPU support
   - Device memory management
   - Hardware-specific optimizations

### Entry Points

1. **LLM Interface** (`fastdeploy/entrypoints/llm.py`): Main Python API for LLM inference
2. **OpenAI API Server** (`fastdeploy/entrypoints/openai/`): OpenAI-compatible API server
3. **CLI Tools** (`fastdeploy/entrypoints/cli/`): Command-line interface tools

### Configuration System

- **Main Config** (`fastdeploy/config.py`): Central configuration with FDConfig class
- **Cache Config**: KV cache and memory management settings
- **Parallel Config**: Distributed execution settings
- **Scheduler Config**: Request scheduling parameters

## Key Features Implementation

### Multi-Hardware Support

The codebase uses a dynamic device detection system in `setup.py`:

- Automatically detects available hardware (CUDA, XPU, NPU, etc.)
- Compiles device-specific custom operations
- Loads appropriate kernels based on runtime detection

### Custom Operations Build System

- **Custom Ops Directory** (`custom_ops/`): Device-specific kernel implementations
- **Build Script** (`custom_ops/setup_ops.py`): Compilation configuration
- **Runtime Loading** (`fastdeploy/import_ops.py`): Dynamic operator loading

### Performance Optimizations

- **CUDA Graphs** (`fastdeploy/graph_optimization/`): Static graph optimization
- **Speculative Decoding** (`fastdeploy/spec_decode/`): Draft model acceleration
- **Chunked Prefill**: Efficient long sequence processing
- **Quantization Support**: Multiple precision formats (W8A16, W4A8, FP8, etc.)

## Development Guidelines

### Code Style

- Line length: 119 characters (configured in pyproject.toml)
- Use Black for code formatting
- Use isort for import sorting
- Use ruff for linting (configured in pyproject.toml)

### Testing Strategy

- Unit tests for core components in `tests/v1/`
- Integration tests for cache management
- Performance benchmarks in `benchmarks/`
- Use pytest framework with standard assertions

### Multi-Device Development

- Always test on target hardware platform
- Use device detection utilities from `fastdeploy/platforms/`
- Consider memory constraints for different device types
- Verify custom ops compilation for target architecture

### Model Integration

- New model implementations go in `fastdeploy/model_executor/models/`
- Follow existing model wrapper patterns
- Implement proper input preprocessing in `fastdeploy/input/`
- Add model-specific configuration options to FDConfig

## Common Development Patterns

### Device Detection

```python
from fastdeploy.platforms import get_device_type
device_type = get_device_type()  # Returns 'gpu', 'xpu', 'npu', etc.
```

### Configuration Usage

```python
from fastdeploy import FDConfig, SamplingParams
config = FDConfig(model="path/to/model")
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
```

### Custom Ops Loading

```python
import fastdeploy.import_ops as import_ops
import_ops.import_custom_ops()  # Loads device-specific operators
```

### Error Handling

- Use proper logging through `fastdeploy.logger`
- Implement graceful degradation for unsupported features
- Provide clear error messages for hardware incompatibilities
