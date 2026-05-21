[简体中文](../zh/usage/code_overview.md)

# FastDeploy Code Structure Overview

This document provides a detailed overview of the FastDeploy codebase structure, helping developers quickly understand each module's functionality for development and feature extension.

---

## Directory Overview

```
FastDeploy/
├── fastdeploy/          # Core code directory
├── custom_ops/          # C++/CUDA custom operators
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── tools/               # Development tools
├── docs/                # Documentation
├── examples/            # Example code
├── benchmarks/          # Performance benchmarks
├── dockerfiles/         # Docker image build files
└── setup.py             # Python package installation script
```

---

## I. Core Code Directory (fastdeploy/)

The main entry file `fastdeploy/__init__.py` exports core classes:

- `LLM` - Main entry class, offline inference interface
- `SamplingParams` - Sampling parameter configuration
- `ModelRegistry` - Model registry
- `version` - Version information

### 1. engine/ - Core Engine Module

**Function**: Manages LLM inference lifecycle and coordinates components.

| File | Function | Development Guide |
|------|----------|-------------------|
| `engine.py` | `LLMEngine` core engine class, manages scheduler, preprocessor, resource manager | Entry point for modifying engine behavior, adding new components |
| `async_llm.py` | Async LLM interface, `AsyncRequestQueue` request queue management | Async inference, streaming output development |
| `request.py` | Core request data structures: `Request`, `RequestOutput`, `RequestStatus` | Adding request fields, modifying request processing logic |
| `sampling_params.py` | `SamplingParams` sampling parameter configuration | Adding new sampling strategy parameters |
| `args_utils.py` | `EngineArgs` engine argument parsing | Adding new engine configuration parameters |
| `resource_manager.py` | GPU/CPU resource management | Resource allocation optimization |

**Subdirectory**:

- `sched/` - Core scheduling implementation, contains `resource_manager_v1.py` (**core scheduling logic**)

---

### 2. model_executor/ - Model Executor

**Function**: Core execution module for model inference, containing model definitions, layers, operators.

#### 2.1 models/ - Model Implementations

| File/Directory | Function | Development Guide |
|----------------|----------|-------------------|
| `model_base.py` | `ModelRegistry` model registration base class | **Must read for adding new models** |
| `deepseek_v3.py` | DeepSeek V3 model | MoE large model reference |
| `ernie4_5_moe.py` | ERNIE 4.5 MoE model | Baidu's flagship model |
| `ernie4_5_mtp.py` | ERNIE 4.5 MTP multi-token prediction | Speculative decoding model |
| `qwen2.py` | Qwen2 model | General model reference |
| `qwen3.py` | Qwen3 model | Latest model reference |
| `ernie4_5_vl/` | ERNIE 4.5 vision-language model | Multimodal model development reference |
| `qwen2_5_vl/` | Qwen2.5 VL multimodal model | VL model reference |
| `paddleocr_vl/` | PaddleOCR VL model | OCR multimodal reference |

#### 2.2 layers/ - Network Layer Implementations

| Subdirectory/File | Function | Development Guide |
|-------------------|----------|-------------------|
| `attention/` | Attention mechanism implementations (flash_attn, append_attn, mla_attn) | **First choice for attention performance optimization** |
| `moe/` | MoE layer implementations (Cutlass, Triton, DeepGEMM backends) | MoE performance optimization |
| `quantization/` | Quantization layers (FP8, W4A8, WINT2, Weight-only) | Quantization scheme development |
| `linear.py` | Linear layer implementation | Matrix multiplication optimization |
| `embeddings.py` | Embedding layer implementation | Word embedding modification |
| `normalization.py` | Normalization layers (RMSNorm, LayerNorm) | Normalization optimization |
| `rotary_embedding.py` | Rotary Position Encoding ROPE | Position encoding modification |
| `sample/` | Sampler implementation | Sampling strategy development |
| `backends/` | Hardware backend implementations (cuda, xpu, dcu, hpu, metax, gcu, npu) | **Entry point for new hardware adaptation** |

#### 2.3 Other Submodules

| Directory | Function | Development Guide |
|-----------|----------|-------------------|
| `model_loader/` | Model weight loader | New model format support |
| `guided_decoding/` | Guided decoding (JSON/regex constrained output) | Structured output development |
| `graph_optimization/` | Graph optimization (CUDA Graph) | Inference performance optimization |
| `logits_processor/` | Logits processor | Output control logic |
| `ops/` | Python-callable operators (organized by hardware platform) | Operator call entry point |

**Key Files**:

- `model_base.py` - Model base class, registry definition
- `pre_and_post_process.py` - Pre/post processing utilities

---

### 3. scheduler/ - Scheduler Module

**Function**: Request scheduling, supporting single-node, distributed, PD disaggregation scenarios.

> **Note**:
> - Core scheduling logic is mainly implemented in `engine/sched/resource_manager_v1.py`
> - Schedulers in this directory are being **gradually deprecated**. For PD disaggregation scheduling, use `router/` or `golang_router/`

| File | Function | Development Guide |
|------|----------|-------------------|
| `global_scheduler.py` | `GlobalScheduler` distributed scheduler (Redis) | (Being deprecated) |
| `local_scheduler.py` | `LocalScheduler` local scheduler | (Being deprecated) |
| `splitwise_scheduler.py` | `SplitwiseScheduler` PD disaggregation scheduling | (Being deprecated, use router) |
| `dp_scheduler.py` | Data parallel scheduler | (Being deprecated) |
| `config.py` | `SchedulerConfig` scheduling configuration | Scheduling parameter adjustment |
| `storage.py` | Storage adapter, wraps Redis connection | Storage layer modification |

**Core Scheduling Implementation** (`engine/sched/`):

| File | Function | Development Guide |
|------|----------|-------------------|
| `resource_manager_v1.py` | Core scheduling logic, contains `ScheduledDecodeTask`, `ScheduledPreemptTask` task classes | **First choice for scheduling strategy modification** |

---

### 4. entrypoints/ - API Entry Points

**Function**: External service interfaces, including offline inference and online API services.

| File | Function | Development Guide |
|------|----------|-------------------|
| `llm.py` | `LLM` main entry class, offline inference interface | **Entry point for using FastDeploy** |
| `engine_client.py` | Engine client | Request forwarding logic modification |

#### 4.1 openai/ - OpenAI Compatible API

| File | Function | Development Guide |
|------|----------|-------------------|
| `api_server.py` | FastAPI server | **Deployment service entry point** |
| `protocol.py` | OpenAI protocol definition | API format modification |
| `serving_chat.py` | Chat Completion API | Chat interface development |
| `serving_completion.py` | Completion API | Completion interface development |
| `serving_embedding.py` | Embedding API | Vectorization interface |
| `tool_parsers/` | Tool call parsers | Function Calling development |

---

### 5. worker/ - Worker Process Module

**Function**: Actual execution process for model inference.

| File | Function | Development Guide |
|------|----------|-------------------|
| `gpu_model_runner.py` | **GPU model runner** (core inference loop) | **First choice for inference flow modification** |
| `gpu_worker.py` | GPU Worker process management | Worker lifecycle management |
| `xpu_model_runner.py` | XPU model runner | Kunlun chip adaptation |
| `hpu_model_runner.py` | HPU model runner | Intel HPU adaptation |
| `worker_process.py` | Worker process base class | Process management logic |

---

### 6. input/ - Input Processing Module

**Function**: Input data preprocessing, including tokenization, multimodal input processing.

| File | Function | Development Guide |
|------|----------|-------------------|
| `text_processor.py` | `BaseDataProcessor` text processor base class | Input processing extension |
| `multimodal_processor.py` | Unified multimodal processor | Multimodal input processing |
| `ernie4_5_tokenizer.py` | ERNIE 4.5 tokenizer | Tokenization logic modification |
| `preprocess.py` | Input preprocessing utilities | Preprocessing flow |

**Multimodal Processing Subdirectories**:

| Directory | Function |
|-----------|----------|
| `encodings/` | Model-specific encoding strategies (Ernie, Qwen, PaddleOCR) |
| `image_processors/` | Image preprocessing (Adaptive, Qwen, Qwen3, PaddleOCR) |
| `multimodal_processor.py` | Unified multimodal processor |

---

### 7. output/ - Output Processing Module

**Function**: Inference result post-processing, streaming output management.

| File | Function | Development Guide |
|------|----------|-------------------|
| `token_processor.py` | `TokenProcessor` token output processing | Streaming output, speculative decoding |
| `pooler.py` | Pooling output processing | Embedding output |
| `stream_transfer_data.py` | Streaming transfer data structure | Data transfer format |

---

### 8. cache_manager/ - Cache Management Module

**Function**: KV Cache management, supporting prefix caching, cross-device transfer.

| File | Function | Development Guide |
|------|----------|-------------------|
| `prefix_cache_manager.py` | `PrefixCacheManager` prefix tree cache | **First choice for KV Cache optimization** |
| `cache_transfer_manager.py` | KV Cache cross-device transfer | PD disaggregation cache transfer |
| `cache_data.py` | `BlockNode`, `CacheStatus` data structures | Cache data definition |
| `multimodal_cache_manager.py` | Multimodal cache management | Multimodal caching |

**Subdirectory**:

- `transfer_factory/` - Cache transfer factory (IPC, RDMA)

---

### 9. platforms/ - Hardware Platform Support

**Function**: Multi-hardware platform adaptation, defining operators and features for each platform.

| File | Function | Development Guide |
|------|----------|-------------------|
| `base.py` | `Platform` base class, `_Backend` enum | **Entry point for new hardware adaptation** |
| `cuda.py` | NVIDIA CUDA platform | GPU optimization |
| `xpu.py` | Baidu Kunlun XPU platform | Kunlun chip adaptation |
| `dcu.py` | AMD DCU (ROCm) platform | AMD GPU adaptation |
| `maca.py` | MetaX GPU (MACA) platform | Biren GPU adaptation |
| `intel_hpu.py` | Intel HPU platform | Intel Gaudi adaptation |
| `iluvatar.py` | Iluvatar GPU platform | Iluvatar adaptation |

---

### 10. metrics/ - Monitoring Metrics Module

**Function**: Prometheus metric collection, performance monitoring.

| File | Function | Development Guide |
|------|----------|-------------------|
| `metrics.py` | Prometheus metric definition | Adding new monitoring metrics |
| `stats.py` | ZMQ metric statistics | Distributed monitoring |
| `trace_util.py` | OpenTelemetry distributed tracing | Link tracing |

---

### 11. Other Important Modules

| Directory | Function | Development Guide |
|-----------|----------|-------------------|
| `inter_communicator/` | Inter-process communication (ZMQ) | Engine-Worker communication modification |
| `spec_decode/` | Speculative decoding (MTP, N-gram) | Speculative decoding strategy development |
| `distributed/` | Distributed communication (AllReduce) | Distributed inference development |
| `multimodal/` | Multimodal data processing | Multimodal feature extension |
| `reasoning/` | Reasoning mode parsing (DeepSeek R1 style) | Chain-of-thought parsing |
| `router/` | Request router, **recommended for PD disaggregation** | **First choice for PD disaggregation deployment** |
| `golang_router/` | Go-implemented router, better PD inter-scheduling performance | **High-performance PD disaggregation scenarios** |
| `eplb/` | Expert Parallel load balancing | MoE load balancing |
| `rl/` | Reinforcement learning Rollout | RLHF scenarios |
| `plugins/` | Plugin system | Custom extensions |
| `logger/` | Logging module | Log format modification |
| `trace/` | Tracing module | Performance analysis |

---

### 12. Configuration Files

| File | Function | Development Guide |
|------|----------|-------------------|
| `config.py` | `FDConfig` main configuration class | **Entry point for configuration parameter modification** |
| `envs.py` | Environment variable configuration | Adding new environment variables |
| `utils.py` | General utility functions | Utility function reuse |

---

## II. Custom Operators Directory (custom_ops/)

**Function**: C++/CUDA high-performance operator implementations, organized by hardware platform.

```
custom_ops/
├── gpu_ops/           # NVIDIA GPU operators (main)
├── cpu_ops/           # CPU operators
├── xpu_ops/           # Baidu Kunlun XPU operators
├── iluvatar_ops/      # Iluvatar GPU operators
├── metax_ops/         # MetaX GPU operators
├── utils/             # Common utilities
└── third_party/       # Third-party libraries (cutlass, DeepGEMM)
```

### gpu_ops/ - GPU Operator Details

| Directory/File | Function | Development Guide |
|----------------|----------|-------------------|
| `append_attn/` | Append Attention implementation | **First choice for attention optimization** |
| `moe/` | MoE operators (fused_moe, expert_dispatch) | MoE performance optimization |
| `flash_mask_attn/` | Flash Mask Attention | Attention mask optimization |
| `mla_attn/` | Multi-Head Latent Attention | MLA model support |
| `machete/` | Machete GEMM | Matrix multiplication optimization |
| `quantization/` | Quantization operators | Quantization performance optimization |
| `sample_kernels/` | Sampling operators | Sampling performance optimization |
| `speculate_decoding/` | Speculative decoding operators | Speculative decoding optimization |
| `cutlass_kernels/` | CUTLASS kernels | High-performance GEMM |
| `cpp_extensions.cc` | C++ extension entry | **Entry point for new operator registration** |
| `append_attention.cu` | Append Attention core | Attention core implementation |

**Key Operator Files**:

- `fused_rotary_position_encoding.cu` - Fused rotary position encoding
- `multi_head_latent_attention.cu` - MLA attention
- `per_token_quant_fp8.cu` - FP8 quantization

---

## III. Test Directory (tests/)

**Function**: Unit tests and end-to-end tests, organized by module.

```
tests/
├── e2e/               # End-to-end service tests
├── operators/         # Operator unit tests
├── model_executor/    # Model executor tests
├── model_loader/      # Model loading tests
├── layers/            # Network layer tests
├── scheduler/         # Scheduler tests
├── cache_manager/     # Cache management tests
├── entrypoints/       # API entry tests
├── input/             # Input processing tests
├── output/            # Output processing tests
├── metrics/           # Metric tests
├── distributed/       # Distributed tests
├── graph_optimization/# Graph optimization tests
├── quantization/      # Quantization tests
├── multimodal/        # Multimodal tests
├── xpu_ci/            # XPU CI tests
├── ci_validation/     # CI validation tests
├── ci_use/            # CI utility tests
└── conftest.py        # pytest configuration
```

### Test Directory Details

| Directory | Content | Development Guide |
|-----------|---------|-------------------|
| `e2e/` | Complete service tests for each model (ERNIE, Qwen, DeepSeek, etc.) | **Service integration testing** |
| `operators/` | Operator unit tests (`test_fused_moe.py`, `test_flash_mask_attn.py`, etc.) | **Required tests for operator development** |
| `layers/` | Network layer tests (attention, moe, quantization) | Network layer testing |
| `model_executor/` | Model execution flow tests | Model execution testing |
| `scheduler/` | Scheduler function tests | Scheduling logic verification |
| `cache_manager/` | Cache management tests | Cache logic verification |

---

## IV. Scripts Directory (scripts/)

**Function**: CI/CD, performance tuning, utility scripts.

| File | Function | Usage Scenario |
|------|----------|----------------|
| `run_unittest.sh` | Unit test runner | Local testing |
| `run_ci_xpu.sh` | XPU CI runner | Kunlun CI |
| `run_ci_hpu.sh` | HPU CI runner | Intel HPU CI |
| `run_ci_dcu.sh` | DCU CI runner | AMD DCU CI |
| `coverage_run.sh` | Code coverage statistics | Code quality |
| `tune_cublaslt_int8_gemm.py` | cuBLASLt INT8 GEMM tuning | Performance tuning |
| `tune_cutlass_fp8_gemm.py` | CUTLASS FP8 GEMM tuning | Performance tuning |
| `offline_w4a8.py` | Offline W4A8 quantization tool | Model quantization |
| `extract_mtp_weight_from_safetensor.py` | MTP weight extraction | Model processing |

---

## V. Other Directories

### docs/ - Documentation

- Usage documentation, API documentation, architecture design documents

### examples/ - Example Code

- Model usage examples, deployment examples

### benchmarks/ - Performance Benchmarks

- Performance test scripts, benchmark data

### tools/ - Development Tools

- `codestyle/` - Code style checking tools
- `dockerfile/` - Docker build tools

### dockerfiles/ - Docker Images

- Dockerfiles for each platform runtime environment

---

## VI. Quick Development Guide

### Adding a New Model

1. Reference `models/model_base.py` to understand model registration mechanism
2. Create new model file under `models/`
3. Add corresponding input processor under `input/`
4. Add tests under `tests/model_executor/`

### Adding a New Operator

1. Implement CUDA operator under `custom_ops/gpu_ops/`
2. Register operator in `cpp_extensions.cc`
3. Add Python wrapper under `model_executor/ops/gpu/`
4. Add tests under `tests/operators/`

### New Hardware Platform Adaptation

1. Reference `platforms/base.py` to create new platform class
2. Create hardware operator directory under `custom_ops/`
3. Create backend implementation under `model_executor/layers/backends/`
4. Create model runner under `worker/`

### Optimizing Inference Performance

1. Attention optimization: `custom_ops/gpu_ops/append_attn/`
2. MoE optimization: `custom_ops/gpu_ops/moe/`
3. Graph optimization: `fastdeploy/model_executor/graph_optimization/`

### PD Disaggregation Deployment

1. Router: `router/router.py` (Python implementation, recommended)
2. High-performance router: `golang_router/` (Go implementation, better PD inter-scheduling performance)
3. Cache transfer: `cache_manager/cache_transfer_manager.py`

---

## VII. Configuration System

```
FDConfig (config.py)
├── ModelConfig      # Model configuration
├── CacheConfig      # Cache configuration
├── ParallelConfig   # Parallel configuration
├── SchedulerConfig  # Scheduler configuration
├── LoRAConfig       # LoRA configuration
└── ...

Environment Variable Configuration (envs.py)
├── FD_* series environment variables
└── Runtime behavior control
```

---

This document covers the main modules and key files of the FastDeploy codebase. It can be used as a code navigation and development reference. For questions, please refer to detailed documentation of each module or source code comments.
