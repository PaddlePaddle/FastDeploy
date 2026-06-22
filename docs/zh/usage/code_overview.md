[English](../../usage/code_overview.md)

# FastDeploy 代码结构详解

本文档详细介绍 FastDeploy 代码库的结构，帮助开发者快速了解各模块功能，便于开发和新功能扩展。

---

## 目录结构总览

```
FastDeploy/
├── fastdeploy/          # 核心代码目录
├── custom_ops/          # C++/CUDA 自定义算子
├── tests/               # 单元测试
├── scripts/             # 工具脚本
├── tools/               # 开发工具
├── docs/                # 文档
├── examples/            # 示例代码
├── benchmarks/          # 性能基准测试
├── dockerfiles/         # Docker 镜像构建文件
└── setup.py             # Python 包安装脚本
```

---

## 一、核心代码目录 (fastdeploy/)

主入口文件 `fastdeploy/__init__.py` 导出核心类：

- `LLM` - 主入口类，离线推理接口
- `SamplingParams` - 采样参数配置
- `ModelRegistry` - 模型注册器
- `version` - 版本信息

### 1. engine/ - 核心引擎模块

**功能**: 管理 LLM 推理的生命周期，协调各组件工作。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `engine.py` | `LLMEngine` 核心引擎类，管理调度器、预处理器、资源管理器 | 修改引擎行为、添加新组件的入口 |
| `async_llm.py` | 异步 LLM 接口，`AsyncRequestQueue` 请求队列管理 | 异步推理、流式输出相关开发 |
| `request.py` | 请求核心数据结构：`Request`, `RequestOutput`, `RequestStatus` | 新增请求字段、修改请求处理逻辑 |
| `sampling_params.py` | `SamplingParams` 采样参数配置 | 添加新采样策略参数 |
| `args_utils.py` | `EngineArgs` 引擎参数解析 | 新增引擎配置参数 |
| `resource_manager.py` | GPU/CPU 资源管理 | 资源分配优化 |

**子目录**:

- `sched/` - 调度核心实现，包含 `resource_manager_v1.py` (**核心调度逻辑所在**)

---

### 2. model_executor/ - 模型执行器

**功能**: 模型推理的核心执行模块，包含模型定义、网络层、算子等。

#### 2.1 models/ - 模型实现

| 文件/目录 | 功能 | 开发指引 |
|-----------|------|----------|
| `model_base.py` | `ModelRegistry` 模型注册基类 | **添加新模型必看** |
| `deepseek_v3.py` | DeepSeek V3 模型 | MoE 大模型参考 |
| `ernie4_5_moe.py` | ERNIE 4.5 MoE 模型 | 百度主力模型 |
| `ernie4_5_mtp.py` | ERNIE 4.5 MTP 多token预测 | 推测解码模型 |
| `qwen2.py` | Qwen2 模型 | 通用模型参考 |
| `qwen3.py` | Qwen3 模型 | 最新模型参考 |
| `ernie4_5_vl/` | ERNIE 4.5 视觉语言模型 | 多模态模型开发参考 |
| `qwen2_5_vl/` | Qwen2.5 VL 多模态模型 | VL 模型参考 |
| `paddleocr_vl/` | PaddleOCR VL 模型 | OCR 多模态参考 |

#### 2.2 layers/ - 网络层实现

| 子目录/文件 | 功能 | 开发指引 |
|-------------|------|----------|
| `attention/` | 注意力机制实现 (flash_attn, append_attn, mla_attn) | **优化注意力性能首选** |
| `moe/` | MoE 层实现 (Cutlass, Triton, DeepGEMM 后端) | MoE 性能优化 |
| `quantization/` | 量化层 (FP8, W4A8, WINT2, Weight-only) | 量化方案开发 |
| `linear.py` | 线性层实现 | 矩阵乘法优化 |
| `embeddings.py` | 嵌入层实现 | 词嵌入修改 |
| `normalization.py` | 归一化层 (RMSNorm, LayerNorm) | 归一化优化 |
| `rotary_embedding.py` | 旋转位置编码 ROPE | 位置编码修改 |
| `sample/` | 采样器实现 | 采样策略开发 |
| `backends/` | 硬件后端实现 (cuda, xpu, dcu, hpu, metax, gcu, npu) | **新硬件适配入口** |

#### 2.3 其他子模块

| 目录 | 功能 | 开发指引 |
|------|------|----------|
| `model_loader/` | 模型权重加载器 | 新模型格式支持 |
| `guided_decoding/` | 引导解码 (JSON/regex 约束输出) | 结构化输出开发 |
| `graph_optimization/` | 图优化 (CUDA Graph) | 推理性能优化 |
| `logits_processor/` | Logits 处理器 | 输出控制逻辑 |
| `ops/` | Python 可调用算子 (按硬件平台组织) | 算子调用入口 |

**关键文件**:

- `model_base.py` - 模型基类、注册器定义
- `pre_and_post_process.py` - 前后处理工具

---

### 3. scheduler/ - 调度器模块

**功能**: 请求调度，支持单机、分布式、PD 分离等场景。

> **注意**:
> - 核心调度逻辑主要在 `engine/sched/resource_manager_v1.py` 中实现
> - 本目录下的 scheduler 正在**逐步废弃**，PD 分离调度请使用 `router/` 或 `golang_router/`

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `global_scheduler.py` | `GlobalScheduler` 分布式调度器 (Redis) | (逐步废弃) |
| `local_scheduler.py` | `LocalScheduler` 本地调度器 | (逐步废弃) |
| `splitwise_scheduler.py` | `SplitwiseScheduler` PD 分离调度 | (逐步废弃，请使用 router) |
| `dp_scheduler.py` | 数据并行调度器 | (逐步废弃) |
| `config.py` | `SchedulerConfig` 调度配置 | 调度参数调整 |
| `storage.py` | 存储适配器，封装 Redis 连接 | 存储层修改 |

**调度核心实现** (`engine/sched/`):

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `resource_manager_v1.py` | 核心调度逻辑，包含 `ScheduledDecodeTask`、`ScheduledPreemptTask` 等任务类 | **调度策略修改首选** |

---

### 4. entrypoints/ - API 入口

**功能**: 对外服务接口，包括离线推理和在线 API 服务。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `llm.py` | `LLM` 主入口类，离线推理接口 | **使用 FastDeploy 入口** |
| `engine_client.py` | 引擎客户端 | 请求转发逻辑修改 |

#### 4.1 openai/ - OpenAI 兼容 API

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `api_server.py` | FastAPI 服务器 | **部署服务入口** |
| `protocol.py` | OpenAI 协议定义 | API 格式修改 |
| `serving_chat.py` | Chat Completion API | 聊天接口开发 |
| `serving_completion.py` | Completion API | 补全接口开发 |
| `serving_embedding.py` | Embedding API | 向量化接口 |
| `tool_parsers/` | 工具调用解析器 | Function Calling 开发 |

---

### 5. worker/ - Worker 进程模块

**功能**: 模型推理的实际执行进程。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `gpu_model_runner.py` | **GPU 模型运行器** (核心推理循环) | **推理流程修改首选** |
| `gpu_worker.py` | GPU Worker 进程管理 | Worker 生命周期管理 |
| `xpu_model_runner.py` | XPU 模型运行器 | 昆仑芯片适配 |
| `hpu_model_runner.py` | HPU 模型运行器 | Intel HPU 适配 |
| `worker_process.py` | Worker 进程基类 | 进程管理逻辑 |

---

### 6. input/ - 输入处理模块

**功能**: 输入数据预处理，包括分词、多模态输入处理。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `text_processor.py` | `BaseDataProcessor` 文本处理器基类 | 输入处理扩展 |
| `multimodal_processor.py` | 统一多模态处理器 | 多模态输入处理 |
| `ernie4_5_tokenizer.py` | ERNIE 4.5 分词器 | 分词逻辑修改 |
| `preprocess.py` | 输入预处理工具 | 预处理流程 |

**多模态处理子目录**:

| 目录 | 功能 |
|------|------|
| `encodings/` | 模型特定编码策略 (Ernie, Qwen, PaddleOCR) |
| `image_processors/` | 图像预处理 (Adaptive, Qwen, Qwen3, PaddleOCR) |
| `multimodal_processor.py` | 统一多模态处理器 |

---

### 7. output/ - 输出处理模块

**功能**: 推理结果后处理，流式输出管理。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `token_processor.py` | `TokenProcessor` Token 输出处理 | 流式输出、推测解码 |
| `pooler.py` | 池化输出处理 | Embedding 输出 |
| `stream_transfer_data.py` | 流式传输数据结构 | 数据传输格式 |

---

### 8. cache_manager/ - 缓存管理模块

**功能**: KV Cache 管理，支持前缀缓存、跨设备传输。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `prefix_cache_manager.py` | `PrefixCacheManager` 前缀树缓存 | **KV Cache 优化首选** |
| `cache_transfer_manager.py` | KV Cache 跨设备传输 | PD 分离缓存传输 |
| `cache_data.py` | `BlockNode`, `CacheStatus` 数据结构 | 缓存数据定义 |
| `multimodal_cache_manager.py` | 多模态缓存管理 | 多模态缓存 |

**子目录**:

- `transfer_factory/` - 缓存传输工厂 (IPC, RDMA)

---

### 9. platforms/ - 硬件平台支持

**功能**: 多硬件平台适配，定义各平台的算子和特性。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `base.py` | `Platform` 基类，`_Backend` 枚举 | **新硬件适配入口** |
| `cuda.py` | NVIDIA CUDA 平台 | GPU 优化 |
| `xpu.py` | 百度昆仑 XPU 平台 | 昆仑芯片适配 |
| `dcu.py` | AMD DCU (ROCm) 平台 | AMD GPU 适配 |
| `maca.py` | MetaX GPU (MACA) 平台 | 壁仞 GPU 适配 |
| `intel_hpu.py` | Intel HPU 平台 | Intel Gaudi 适配 |
| `iluvatar.py` | 天数智芯 GPU 平台 | 天数智芯适配 |

---

### 10. metrics/ - 监控指标模块

**功能**: Prometheus 指标收集，性能监控。

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `metrics.py` | Prometheus 指标定义 | 新增监控指标 |
| `stats.py` | ZMQ 指标统计 | 分布式监控 |
| `trace_util.py` | OpenTelemetry 分布式追踪 | 链路追踪 |

---

### 11. 其他重要模块

| 目录 | 功能 | 开发指引 |
|------|------|----------|
| `inter_communicator/` | 进程间通信 (ZMQ) | Engine-Worker 通信修改 |
| `spec_decode/` | 推测解码 (MTP, N-gram) | 推测解码策略开发 |
| `distributed/` | 分布式通信 (AllReduce) | 分布式推理开发 |
| `multimodal/` | 多模态数据处理 | 多模态功能扩展 |
| `reasoning/` | 推理模式解析 (DeepSeek R1 风格) | 思考链解析 |
| `router/` | 请求路由器，**PD 分离调度推荐使用** | **PD 分离部署首选** |
| `golang_router/` | Go 语言实现的路由器，PD 间调度性能更优 | **高性能 PD 分离场景** |
| `eplb/` | Expert Parallel 负载均衡 | MoE 负载均衡 |
| `rl/` | 强化学习 Rollout | RLHF 场景 |
| `plugins/` | 插件系统 | 自定义扩展 |
| `logger/` | 日志模块 | 日志格式修改 |
| `trace/` | 追踪模块 | 性能分析 |

---

### 12. 配置文件

| 文件 | 功能 | 开发指引 |
|------|------|----------|
| `config.py` | `FDConfig` 总配置类 | **配置参数修改入口** |
| `envs.py` | 环境变量配置 | 新增环境变量 |
| `utils.py` | 通用工具函数 | 工具函数复用 |

---

## 二、自定义算子目录 (custom_ops/)

**功能**: C++/CUDA 高性能算子实现，按硬件平台组织。

```
custom_ops/
├── gpu_ops/           # NVIDIA GPU 算子 (主要)
├── cpu_ops/           # CPU 算子
├── xpu_ops/           # 百度昆仑 XPU 算子
├── iluvatar_ops/      # 天数智芯 GPU 算子
├── metax_ops/         # MetaX GPU 算子
├── utils/             # 公共工具
└── third_party/       # 第三方库 (cutlass, DeepGEMM)
```

### gpu_ops/ - GPU 算子详解

| 目录/文件 | 功能 | 开发指引 |
|-----------|------|----------|
| `append_attn/` | Append Attention 实现 | **注意力优化首选** |
| `moe/` | MoE 算子 (fused_moe, expert_dispatch) | MoE 性能优化 |
| `flash_mask_attn/` | Flash Mask Attention | 注意力掩码优化 |
| `mla_attn/` | Multi-Head Latent Attention | MLA 模型支持 |
| `machete/` | Machete GEMM | 矩阵乘法优化 |
| `quantization/` | 量化算子 | 量化性能优化 |
| `sample_kernels/` | 采样算子 | 采样性能优化 |
| `speculate_decoding/` | 推测解码算子 | 推测解码优化 |
| `cutlass_kernels/` | CUTLASS 内核 | 高性能 GEMM |
| `cpp_extensions.cc` | C++ 扩展入口 | **新增算子注册入口** |
| `append_attention.cu` | Append Attention 核心 | 注意力核心实现 |

**关键算子文件**:

- `fused_rotary_position_encoding.cu` - 融合旋转位置编码
- `multi_head_latent_attention.cu` - MLA 注意力
- `per_token_quant_fp8.cu` - FP8 量化

---

## 三、测试目录 (tests/)

**功能**: 单元测试和端到端测试，按模块组织。

```
tests/
├── e2e/               # 端到端服务测试
├── operators/         # 算子单元测试
├── model_executor/    # 模型执行器测试
├── model_loader/      # 模型加载测试
├── layers/            # 网络层测试
├── scheduler/         # 调度器测试
├── cache_manager/     # 缓存管理测试
├── entrypoints/       # API 入口测试
├── input/             # 输入处理测试
├── output/            # 输出处理测试
├── metrics/           # 指标测试
├── distributed/       # 分布式测试
├── graph_optimization/# 图优化测试
├── quantization/      # 量化测试
├── multimodal/        # 多模态测试
├── xpu_ci/            # XPU CI 测试
├── ci_validation/     # CI 验证测试
├── ci_use/            # CI 工具测试
└── conftest.py        # pytest 配置
```

### 测试目录详解

| 目录 | 内容 | 开发指引 |
|------|------|----------|
| `e2e/` | 各模型服务完整测试 (ERNIE, Qwen, DeepSeek 等) | **服务集成测试** |
| `operators/` | 算子单元测试 (`test_fused_moe.py`, `test_flash_mask_attn.py` 等) | **算子开发必写测试** |
| `layers/` | 网络层测试 (attention, moe, quantization) | 网络层测试 |
| `model_executor/` | 模型执行流程测试 | 模型执行测试 |
| `scheduler/` | 调度器功能测试 | 调度逻辑验证 |
| `cache_manager/` | 缓存管理测试 | 缓存逻辑验证 |

---

## 四、脚本工具目录 (scripts/)

**功能**: CI/CD、性能调优、工具脚本。

| 文件 | 功能 | 使用场景 |
|------|------|----------|
| `run_unittest.sh` | 单元测试运行 | 本地测试 |
| `run_ci_xpu.sh` | XPU CI 运行 | 昆仑 CI |
| `run_ci_hpu.sh` | HPU CI 运行 | Intel HPU CI |
| `run_ci_dcu.sh` | DCU CI 运行 | AMD DCU CI |
| `coverage_run.sh` | 代码覆盖率统计 | 代码质量 |
| `tune_cublaslt_int8_gemm.py` | cuBLASLt INT8 GEMM 调优 | 性能调优 |
| `tune_cutlass_fp8_gemm.py` | CUTLASS FP8 GEMM 调优 | 性能调优 |
| `offline_w4a8.py` | 离线 W4A8 量化工具 | 模型量化 |
| `extract_mtp_weight_from_safetensor.py` | MTP 权重提取 | 模型处理 |

---

## 五、其他目录

### docs/ - 文档

- 使用文档、API 文档、架构设计文档

### examples/ - 示例代码

- 各模型使用示例、部署示例

### benchmarks/ - 性能基准

- 性能测试脚本、基准数据

### tools/ - 开发工具

- `codestyle/` - 代码风格检查工具
- `dockerfile/` - Docker 构建工具

### dockerfiles/ - Docker 镜像

- 各平台运行环境 Dockerfile

---

## 六、开发指引速查

### 添加新模型

1. 参考 `models/model_base.py` 了解模型注册机制
2. 在 `models/` 下创建新模型文件
3. 在 `input/` 下添加对应的输入处理器
4. 在 `tests/model_executor/` 下添加测试

### 添加新算子

1. 在 `custom_ops/gpu_ops/` 下实现 CUDA 算子
2. 在 `cpp_extensions.cc` 中注册算子
3. 在 `model_executor/ops/gpu/` 下添加 Python 封装
4. 在 `tests/operators/` 下添加测试

### 新硬件平台适配

1. 参考 `platforms/base.py` 创建新平台类
2. 在 `custom_ops/` 下创建硬件算子目录
3. 在 `model_executor/layers/backends/` 下创建后端实现
4. 在 `worker/` 下创建模型运行器

### 优化推理性能

1. 注意力优化：`custom_ops/gpu_ops/append_attn/`
2. MoE 优化：`custom_ops/gpu_ops/moe/`
3. 图优化：`fastdeploy/model_executor/graph_optimization/`

### PD 分离部署

1. 路由器：`router/router.py` (Python 实现，推荐)
2. 高性能路由：`golang_router/` (Go 实现，PD 间调度性能更优)
3. 缓存传输：`cache_manager/cache_transfer_manager.py`

---

## 七、配置体系

```
FDConfig (config.py)
├── ModelConfig      # 模型配置
├── CacheConfig      # 缓存配置
├── ParallelConfig   # 并行配置
├── SchedulerConfig  # 调度配置
├── LoRAConfig       # LoRA 配置
└── ...

环境变量配置 (envs.py)
├── FD_* 系列环境变量
└── 运行时行为控制
```

---

本文档涵盖了 FastDeploy 代码库的主要模块和关键文件，可作为代码导航和开发参考使用。如有疑问，请参考各模块的详细文档或源码注释。
