# FastDeploy 确定性支持分析报告

## 一、参考方案：SGLang 确定性推理实现

参考 [SGLang Issue #10278](https://github.com/sgl-project/sglang/issues/10278) 和 [SGLang 确定性推理博客](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)，SGLang 通过以下方式实现确定性推理：

### 1. Attention Backend (已实现)

| 后端 | 状态 | 说明 |
|------|------|------|
| **CUDA Graph** | ✅ | 支持 CUDA Graph 加速确定性推理 (#10645) |
| **Temperature > 0** | ✅ | 支持非零温度的确定性推理 (#10678) |
| **FlashInfer** | ✅ | [PR #1675](https://github.com/flashinfer-ai/flashinfer/pull/1675) (#10645) |
| **Triton** | ✅ | 支持 (#10425, #10694) |
| **FlashAttention-3 (FA3)** | ✅ | 支持 (#10651) |
| **批量不变 Triton Kernels** | ✅ | 已移入 SGLang 仓库 (#10695) |
| **单元测试** | ✅ | 完整测试覆盖 (#11095, #10994, #11368) |

### 2. 通信 (NCCL)

| 功能 | 状态 | 说明 |
|------|------|------|
| **确定性 All-Reduce (NVIDIA)** | ✅ | 支持 Tensor Parallelism (#10930) |
| **确定性 All-Reduce (AMD)** | ✅ | 支持 (#15340) |
| **Blackwell TP4** | ❌ | 已知问题：Blackwell 架构 TP4 下不具确定性 (#11513) |

### 3. Radix Cache 支持

| 功能 | 状态 | 说明 |
|------|------|------|
| **FA3** | ✅ | 由于 FA3 本身是单阶段 prefill，天然支持 |
| **Triton** | ✅ | [PR #11147](https://github.com/sgl-project/sglang/pull/11147) |
| **FlashInfer** | ❌ | 待支持 |
| **Prefill 与 Radix Cache 输出一致性** | ❌ | 确保有/无 Radix Cache 的 prefill 输出相同 |

### 4. 模型支持

| 模型 | 状态 | 说明 |
|------|------|------|
| **Qwen3 Dense** | ✅ | 已支持 |
| **Qwen3 MOE** | ✅ | 已支持 |
| **DeepSeek v3** | ✅ | 确定性支持 (#12095) |
| **Qwen3 Next (Linear Attention)** | ❌ | 待支持 (#12845) |

### 5. 量化支持 (待实现)

| 功能 | 状态 | 说明 |
|------|------|------|
| **Blockwise FP8 Kernel** | ❌ | [PR #11491](https://github.com/sgl-project/sglang/pull/11491) |
| **Per-channel FP8 Gemm** | ❌ | 待实现 |
| **FP8 Fused MoE** | ❌ | 待实现 |
| **nvfp4 Gemm/MoE** | ❌ | 待实现 |

### 6. 并行支持 (待实现)

| 功能 | 状态 | 说明 |
|------|------|------|
| **DP Attention** | ❌ | [PR #11023](https://github.com/sgl-project/sglang/pull/11023) |
| **EP (Expert Parallelism)** | ❌ | 待实现 |

### 7. Speculative Decoding (待实现)

| 功能 | 状态 | 说明 |
|------|------|------|
| **Drafters** | ❌ | [#8391](https://github.com/sgl-project/sglang/issues/8391), [#9877](https://github.com/sgl-project/sglang/issues/9877) |

### 8. 性能优化

| 功能 | 状态 | 说明 |
|------|------|------|
| **加速批量不变 Triton Kernels** | ✅ | (#12142, #12144) |

### 9. 可用性与文档

| 功能 | 状态 | 说明 |
|------|------|------|
| **默认参数设置** | ✅ | 如 Attention Backend 默认值 (#11801) |
| **官方文档** | ✅ | https://docs.sglang.ai/ (#11956) |

### 核心原理

1. **Batch-Invariant Kernels**: 不同批量大小时，reduction splitting 过程会变化，导致浮点运算顺序不同。使用批量不变实现确保相同输入产生相同输出。
2. **Chunked Prefill 对齐**: 确保每个单元能被同一个 attention kernel 完整处理。
3. **确定性采样**: 使用带种子的哈希函数生成 Gumbel 噪声，相同的 `(inputs, seed)` 对总是产生相同的采样结果。
4. **确定性 All-Reduce**: Tensor Parallelism 通信使用确定性 kernel 避免浮点累加顺序差异。

### 相关资源

- [Thinking Machines Lab Blog: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [batch_invariant_ops GitHub](https://github.com/thinking-machines-lab/batch_invariant_ops)

## 二、FastDeploy 需要修改的模块对比

| 模块 | 需要修改的内容 | FastDeploy 现状 | SGLang 参考 |
|------|---------------|----------------|-------------|
| **Batch-Invariant Kernels** | 矩阵乘法、mean、log_softmax 等算子需要批量不变实现 | ✅ **已实现！** 位于 `fastdeploy/model_executor/layers/batch_invariant_ops/batch_invariant_ops.py` | ✅ 已实现并移入主仓库 |
| **Attention Kernels** | 实现固定 split-KV 大小的批量不变注意力 | ✅ **已实现！** `flash_attn_backend.py:249-255` 通过 `envs.FD_DETERMINISTIC_MODE` 控制 `decoder_block_shape_q` | ✅ 支持多个后端 (FlashInfer, Triton, FA3) |
| **Chunked Prefill** | 截断点与 split_kv_size 的整数倍对齐 | ✅ **已实现！** | ✅ 已实现 |
| **Sampler** | 实现带种子的确定性采样（multinomial_with_seed） | ⚠️ 已有 seed 支持，确定性模式下默认 seed=42（`sampling_params.py:271`） | ✅ 支持非零温度 |
| **Attention Backend** | FlashAttention 后端配置 | ✅ **已实现！** `flash_attn_backend.py:249-255` 通过 `envs.FD_DETERMINISTIC_MODE` 覆盖 `decoder_block_shape_q` | ✅ 多后端支持 |
| **CUDA Graph** | 支持 CUDA Graph 加速确定性推理 | ✅ **已支持！** CUDA Graph 功能已实现（`graph_optimization_backend.py`），确定性测试中已启用（`start_fd.sh: use_cudagraph:true`）。待验证：确定性模式下 CUDA Graph replay 结果是否严格一致 | ✅ 已实现 |
| **NCCL All-Reduce** | 确定性 Tensor Parallelism 通信 | ✅ **已实现！** `communication.py:114-166` 通过 `CustomAllreduce` 实现，支持 float32/float16/bfloat16 | ✅ 已实现 (NVIDIA & AMD) |
| **Scheduler 确定性** | 移除随机选择逻辑 | ⚠️ `splitwise_scheduler.py:614-626` 已实现 hash-based 选择；`global_scheduler.py:476,481` 仍使用 `random.sample`/`random.choice`（⚡ 低优：仅影响多机场景） | N/A |
| **Radix Cache** | 确保有/无缓存输出一致 | ❌ 待实现 | ⚠️ 部分支持 (Triton, FA3) |
| **量化支持** | FP8/nvfp4 确定性 kernel | ❌ 待实现 | ❌ 开发中 |

## 三、FastDeploy 关键代码位置

| 功能 | 文件位置 | 说明 |
|------|---------|------|
| 批量不变算子 | `fastdeploy/model_executor/layers/batch_invariant_ops/batch_invariant_ops.py` | Triton 实现的 matmul、log_softmax、mean |
| 批量不变开关 | `batch_invariant_ops.py:572-583` | `set_batch_invariant_mode()` 上下文管理器 |
| attention block size | `batch_invariant_ops.py:589-590` | `get_batch_invariant_attention_block_size()` 返回 `block_m=16, block_n=16` |
| 环境变量定义 | `fastdeploy/envs.py:210-214` | `FD_DETERMINISTIC_MODE`、`FD_DETERMINISTIC_SPLIT_KV_SIZE`、`FD_DETERMINISTIC_LOG_MODE` |
| seed 确定性 | `fastdeploy/engine/sampling_params.py:269-274` | `FD_DETERMINISTIC_MODE` 时 seed 固定为 42，否则随机生成 |
| seed 随位置偏移 | `fastdeploy/model_executor/layers/sample/sampler.py:86-92` | `padding_sampling_params()` 根据解码位置调整 seed：`(seed + offset) % MAX_INFER_SEED` |
| 确定性对齐 | `fastdeploy/engine/sched/resource_manager_v1.py:466-491` | `_get_num_new_tokens()` 中的 3 分支对齐逻辑（final chunk / boundary / defer） |
| FlashAttention 后端 | `fastdeploy/model_executor/layers/attention/flash_attn_backend.py:249-255` | ✅ 已实现：通过 `envs.FD_DETERMINISTIC_MODE` 覆盖 `decoder_block_shape_q` |
| 确定性 All-Reduce | `fastdeploy/distributed/communication.py:114-166` | ✅ 已实现：`FD_DETERMINISTIC_MODE` 时强制 `CustomAllreduce`，支持 float32/float16/bfloat16 |
| CustomAllreduce 实现 | `fastdeploy/distributed/custom_all_reduce.py` | 确定性 All-Reduce kernel 实现 |
| Splitwise 确定性选择 | `fastdeploy/scheduler/splitwise_scheduler.py:606-626` | ✅ 已实现：基于 `hash(str(req.request_id))` 的确定性节点选择 |
| Global Scheduler 随机性 | `fastdeploy/scheduler/global_scheduler.py:476,481` | ⚡ **低优**：`random.sample()` 和 `random.choice()` 仍在使用，仅影响多机场景，单机确定性不受影响 |
| 死代码（需清理） | `fastdeploy/worker/gpu_model_runner.py:1639-1641` | 引用不存在的 `scheduler_config.enable_deterministic_mode`，`getattr` 永远返回 False，实际无效 |

### 批量不变算子详细实现
FastDeploy 已实现以下批量不变算子：

1. **matmul_persistent** - 基于 Triton 的持久化矩阵乘法
2. **log_softmax** - Triton 实现的 log_softmax
3. **mean_dim** - Triton 实现的 mean 归约
4. **addmm_batch_invariant** - 批量不变的 addmm

可以通过以下方式启用：
```python
from fastdeploy.model_executor.layers.batch_invariant_ops import set_batch_invariant_mode

# 启用批量不变模式
with set_batch_invariant_mode(True):
    # 执行推理
    pass
```

## 四、确定性相关测试文件

### 4.1 批量不变算子测试

| 测试文件 | 说明 |
|---------|------|
| `tests/batch_invariant/test_batch_invariance_op_mm.py` | 矩阵乘法批量不变性验证（单行 vs 全批次切片） |
| `tests/batch_invariant/test_batch_invariance_op_mean.py` | Mean 归约批量不变性验证 |
| `tests/batch_invariant/test_batch_invariance_op_logsoftmax.py` | LogSoftmax 批量不变性验证（含极端数值范围 trap tensor） |
| `tests/batch_invariant/test_batch_invariance_op_addmm.py` | AddMM 批量不变性验证（float32、bfloat16） |
| `tests/batch_invariant_ops/test_batch_invariant_ops.py` | 综合单测：模式控制、matmul_persistent（多形状/多 dtype）、log_softmax、mean_dim、addmm、mm、attention block size |

### 4.2 Scheduler 确定性测试

| 测试文件 | 说明 |
|---------|------|
| `tests/scheduler/test_chunked_prefill_determinism.py` | Chunked Prefill 确定性：token 对齐、boundary case、FlashAttention 支持、多模态输入、多请求竞争、边界值 |
| `tests/scheduler/deterministic/test_global_scheduler_determinism.py` | Global Scheduler 确定性：hash-based 选择策略、self-exclusion、环境变量集成（测试已写，生产代码未改） |

### 4.3 分布式确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/distributed/allreduce_deterministic.py` | 多 GPU All-Reduce 确定性（float32/float16/bfloat16，20 轮），验证 custom all-reduce vs NCCL | 2+ GPU |
| `tests/distributed/test_allreduce_deterministic_launch.py` | 上述测试的 pytest 启动器（通过 `paddle.distributed.launch --gpus 0,1`） | 2+ GPU |

### 4.4 Attention 确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/layers/test_flash_attention_versions_determinism.py` | FA2/FA3 确定性（20 个测试：batch invariance、序列长度、GQA、dtype） | GPU |
| `tests/layers/test_paddle_attention_determinism.py` | Paddle SDPA + FastDeploy 集成确定性测试 | GPU |
| `tests/layers/test_paddle_attention_determinism_standalone.py` | 独立 Paddle SDPA 确定性测试（无 FastDeploy 依赖） | GPU |

### 4.5 采样与环境变量测试

| 测试文件 | 说明 |
|---------|------|
| `tests/engine/test_sampling_params_determinism.py` | SamplingParams seed 行为：确定性模式 seed=42、显式 seed、边界值 |
| `tests/envs/test_deterministic_env_vars.py` | 环境变量默认值、组合设置、power-of-two 校验（子进程隔离运行） |

### 4.6 端到端确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/deterministic/test_determinism_offline.py` | 离线推理确定性（13 个场景：same-prompt、batch invariance、不同 batch size、长序列、不同温度等） | GPU + 模型 |
| `tests/deterministic/test_determinism_standalone.py` | 轻量级独立确定性测试（seed、环境变量、token 对齐，无需模型加载） | 无 |
| `tests/ce/deterministic/test_determinism_verification.py` | CE 端到端验证：向运行中的 FastDeploy 服务发送请求，验证 MD5 哈希一致性 | 运行中的服务 |
| `tests/ce/deterministic/start_fd.sh` | 单 GPU 启动脚本（TP=1，Qwen2.5-7B） | 1 GPU |
| `tests/ce/deterministic/start_fd_tp4.sh` | 多 GPU 启动脚本（TP=4，Qwen2.5-7B） | 4 GPU |

### 4.7 其他相关测试（非确定性专项）

| 测试文件 | 说明 |
|---------|------|
| `tests/cache_manager/test_cache_data.py` | 缓存数据结构测试 |
| `tests/cache_manager/test_cache_messager.py` | 缓存消息传递测试 |
| `tests/cache_manager/test_cache_transfer_manager.py` | 缓存传输管理测试 |
| `tests/cache_manager/test_prefix_cache_manager.py` | 前缀缓存管理测试 |
| `tests/cache_manager/test_rdma_transfer.py` | RDMA 传输测试 |
| `tests/engine/test_sampling_params.py` | 采样参数基础测试 |
| `tests/engine/test_sampling_params_extended.py` | 采样参数扩展测试 |
| `tests/layers/test_min_sampling.py` | Min-P 采样测试 |
| `tests/scheduler/test_local_scheduler.py` | 本地调度器测试 |
| `tests/scheduler/test_dp_scheduler.py` | 数据并行调度器测试 |
| `tests/scheduler/test_splitwise_scheduler.py` | 拆分调度器测试（需要 Redis） |

## 五、实现确定性的关键修改点

### 1. 已实现部分 ✅

#### 1.1 批量不变算子
FastDeploy 已经实现了批量不变算子，可以直接使用：

```python
from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
    enable_batch_invariant_mode,
    disable_batch_invariant_mode
)

# 方式1: 使用上下文管理器
with set_batch_invariant_mode(True):
    # 执行推理
    result = model.generate(...)

# 方式2: 全局启用
enable_batch_invariant_mode()
try:
    result = model.generate(...)
finally:
    disable_batch_invariant_mode()

# 检查是否已启用
if is_batch_invariant_mode_enabled():
    print("Batch invariant mode is enabled")
```

#### 1.2 Chunked Prefill 对齐
已在 `resource_manager_v1.py:466-491` 中实现，通过 `envs.FD_DETERMINISTIC_MODE` 控制，包含 3 个分支：

```python
# 实际代码 (resource_manager_v1.py:466-491)
if envs.FD_DETERMINISTIC_MODE:
    split_kv_size = envs.FD_DETERMINISTIC_SPLIT_KV_SIZE
    current_pos = request.num_computed_tokens
    remaining_tokens = request.need_prefill_tokens - current_pos

    # Case 1: Final chunk（剩余 token 不足一个 block）- 无需对齐
    if remaining_tokens < split_kv_size:
        aligned_end = current_pos + remaining_tokens
    else:
        # Case 2: 对齐到 split_kv_size 边界
        next_boundary = ((current_pos + split_kv_size - 1) // split_kv_size) * split_kv_size
        tokens_to_boundary = next_boundary - current_pos

        # Case 3: Budget 不足以到达下一个边界 - 延迟到下一轮
        if token_budget < tokens_to_boundary:
            return 0

        # 对齐到 budget 范围内尽可能多的完整边界
        aligned_end = ((current_pos + token_budget) // split_kv_size) * split_kv_size

    num_new_tokens = aligned_end - current_pos
    num_new_tokens = min(num_new_tokens, token_budget, remaining_tokens)
```

#### 1.3 环境变量支持
所有环境变量定义在 `fastdeploy/envs.py:210-214`，通过 `envs.FD_DETERMINISTIC_MODE` 等方式在代码中访问：

```bash
# 启用确定性模式
export FD_DETERMINISTIC_MODE=1
# Split KV block size（必须为 2 的幂，默认 16）
export FD_DETERMINISTIC_SPLIT_KV_SIZE=16
# 启用确定性日志（打印 MD5 哈希和调试信息，用于验证确定性）
export FD_DETERMINISTIC_LOG_MODE=1
```

#### 1.4 FlashAttention 后端确定性模式
已在 `flash_attn_backend.py:249-255` 中实现：

```python
# 实际代码 (flash_attn_backend.py:249-255)
self.enable_deterministic_mode = envs.FD_DETERMINISTIC_MODE
self.deterministic_split_kv_size = envs.FD_DETERMINISTIC_SPLIT_KV_SIZE

if self.enable_deterministic_mode:
    self.decoder_block_shape_q = self.deterministic_split_kv_size
```

#### 1.5 确定性 All-Reduce (Tensor Parallelism)
已在 `communication.py:114-166` 中实现。当 `FD_DETERMINISTIC_MODE` 开启时，自动使用 `CustomAllreduce` 替代标准 NCCL：

```python
# 实际代码 (communication.py:114-166)
if envs.FD_DETERMINISTIC_MODE:
    # Lazy initialization of custom all-reduce
    if _TP_AR is None:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        if tp_group is not None and tp_group.nranks > 1:
            use_custom_allreduce(tp_group)
    # dtype 检查：仅支持 float32, float16, bfloat16
    if input_.dtype not in SUPPORTED_DTYPES:
        raise AssertionError(...)
    # 16-byte 对齐检查
    if inp_size % 16 != 0:
        raise RuntimeError(...)
```

特性：
- 支持 float32/float16/bfloat16
- 16-byte 对齐检查
- Lazy initialization：首次调用时自动初始化
- 静态模式下 fail-fast（不允许在静态模式使用非确定性 NCCL）

#### 1.6 Splitwise Scheduler 确定性选择
已在 `splitwise_scheduler.py:614-626` 中实现：

```python
# 实际代码 (splitwise_scheduler.py:614-626)
if envs.FD_DETERMINISTIC_MODE:
    # 基于 request_id 的 hash 确定性选择节点
    node_index = hash(str(req.request_id)) % (blur_idx + 1)
    node = nodes[node_index]
else:
    # 保持原有的随机选择行为
    node = random.choice(nodes[:blur_idx + 1])
```

### 2. 需要添加的功能

#### 2.1 CUDA Graph 支持
SGLang 已实现 CUDA Graph 加速确定性推理 (#10645)，FastDeploy 需要添加：
- CUDA Graph 捕获和回放支持
- 确定性 kernel 的 CUDA Graph 兼容性

#### 2.2 死代码清理
`gpu_model_runner.py:1639-1641` 存在一段死代码，引用了 `SchedulerConfig` 中不存在的属性：
```python
# 死代码 (gpu_model_runner.py:1639-1641)
# SchedulerConfig 没有 enable_deterministic_mode 属性，getattr 永远返回 False
if getattr(self.scheduler_config, "enable_deterministic_mode", False):
    decoder_block_shape_q = self.scheduler_config.deterministic_split_kv_size
```
实际起作用的是 `flash_attn_backend.py:249-255` 中通过 `envs.FD_DETERMINISTIC_MODE` 读取的逻辑。
此死代码需要删除或改为使用 `envs.FD_DETERMINISTIC_MODE`。

#### 2.3 Radix Cache 确定性
确保有/无 Radix Cache 的 prefill 输出一致：
```python
# 需要验证
with deterministic_mode():
    output1 = model.prefill(prompt)  # 无缓存
    output2 = model.prefill(prompt)  # 有缓存
    assert output1 == output2  # 应该相等
```

#### 2.4 量化确定性
支持 FP8/nvfp4 的确定性推理：
- Blockwise FP8 Kernel
- Per-channel FP8 Gemm
- FP8 Fused MoE
- nvfp4 Gemm/MoE

#### 2.5 Temperature > 0 支持
确保非零温度下的采样确定性：
```python
def deterministic_sampling(logits, temperature, seed):
    # 使用确定性 Gumbel 噪声生成
    gumbel_noise = -torch.log(-torch.log(hash_from_seed(seed, logits.shape)))
    sampled_tokens = (logits / temperature + gumbel_noise).argmax(dim=-1)
    return sampled_tokens
```

#### 2.6 Global Scheduler 随机性修复（⚡ 低优：仅影响多机场景）
`global_scheduler.py:476,481` 的 `random.sample()` 和 `random.choice()` **仍未修复**，但此问题仅影响多机调度场景，单机确定性不受影响。
测试 `tests/scheduler/deterministic/test_global_scheduler_determinism.py` 已实现 hash-based 选择策略，
但生产代码尚未应用：

```python
# 当前生产代码 (global_scheduler.py:476,481) — 仍为随机
extend_scheduler_names = random.sample(members, k=min(10, len(members)))
lucky = random.choice(extend_scheduler_names)

# 需要改为确定性选择（参考测试中的 hash-based 策略）
if envs.FD_DETERMINISTIC_MODE:
    extend_scheduler_names = members[:min(10, len(members))]
    node_index = hash(self.name + str(self.load_slot_for_getting_request)) % len(extend_scheduler_names)
    lucky = extend_scheduler_names[node_index]
else:
    extend_scheduler_names = random.sample(members, k=min(10, len(members)))
    lucky = random.choice(extend_scheduler_names)
```

注意：`splitwise_scheduler.py:614-626` 已正确实现确定性选择（见 1.6 节）。

### 3. 已有的端到端确定性测试

以下测试已存在于代码库中（详见第四节）：

- `tests/deterministic/test_determinism_offline.py` — 离线推理确定性（13 个场景）
- `tests/deterministic/test_determinism_standalone.py` — 轻量级独立确定性测试
- `tests/ce/deterministic/test_determinism_verification.py` — CE 端到端 MD5 验证
- `tests/distributed/allreduce_deterministic.py` — 多 GPU All-Reduce 确定性

## 六、实施建议

### 阶段一：基础确定性（已完成 ✅）
1. ✅ 添加 `FD_DETERMINISTIC_MODE`、`FD_DETERMINISTIC_SPLIT_KV_SIZE`、`FD_DETERMINISTIC_LOG_MODE` 环境变量
2. ✅ 实现批量不变算子（matmul、mean、log_softmax、addmm）
3. ✅ 实现 Chunked Prefill 对齐逻辑（`resource_manager_v1.py:466-491`）
4. ✅ 添加基础确定性单测

### 阶段二：增强确定性（大部分完成 ✅）
1. ✅ 实现 FlashAttention 后端确定性模式（`flash_attn_backend.py:249-255`）
2. ✅ 实现 Splitwise Scheduler 确定性选择（`splitwise_scheduler.py:614-626`）
3. ⚡ **Global Scheduler 随机选择未修复（低优）**：`global_scheduler.py:476,481` 仍使用 `random.sample`/`random.choice`（测试已写，生产代码未改；仅影响多机场景）
4. ⚠️ Sampler seed 处理：确定性模式下硬编码 seed=42（`sampling_params.py:271`），所有未指定 seed 的请求使用相同种子
5. ✅ 端到端确定性测试套件（`tests/ce/deterministic/`、`tests/deterministic/`）

### 阶段三：并行确定性（大部分完成 ✅）
1. ✅ 实现确定性 All-Reduce kernel（`communication.py:114-166` + `custom_all_reduce.py`）
2. ✅ 支持 float32/float16/bfloat16
3. ✅ 添加多 GPU 确定性测试（`tests/distributed/allreduce_deterministic.py`）
4. ❌ Blackwell 架构支持（SGLang 也存在同类问题 #11513）

### 阶段四：高级特性（待实现 ⏳）
1. ❌ CUDA Graph 支持
2. ❌ Radix Cache 确定性
3. ❌ 量化确定性（FP8, nvfp4）
4. ❌ DP Attention 和 EP 支持

### 阶段五：模型支持（待实现 ⏳）
1. Qwen3 Dense/MOE 验证
2. DeepSeek v3 支持
3. Linear Attention 支持
4. MoE 模型确定性验证

### 阶段六：性能优化（持续 🔄）
1. 加速批量不变 Triton Kernels
2. CUDA Graph 优化
3. 内存优化
4. 量化 kernel 优化

## 七、参考资料

- [SGLang Issue #10278: Support deterministic inference with Batch Invariant Ops](https://github.com/sgl-project/sglang/issues/10278)
- [SGLang 确定性推理博客](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [Thinking Machines Lab Blog: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [batch_invariant_ops GitHub](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [FastDeploy Batch Invariant 代码](fastdeploy/model_executor/layers/batch_invariant_ops/batch_invariant_ops.py)
- [SGLang 官方文档](https://docs.sglang.ai/)

## 八、总结

FastDeploy 已经有了 **batch-invariant 算子** 的基础实现（从 Thinking Machines Lab 移植），并完成了确定性 All-Reduce、FlashAttention 后端确定性模式、Chunked Prefill 对齐等关键功能。确定性相关测试文件共 **19 个**，覆盖算子、调度器、分布式通信、Attention、采样、端到端等多个层面。

### 已完成 ✅
1. 批量不变算子（matmul、mean、log_softmax、addmm）
2. Chunked Prefill 对齐（3 分支逻辑：final chunk / boundary / defer）
3. 环境变量支持（`FD_DETERMINISTIC_MODE`、`FD_DETERMINISTIC_SPLIT_KV_SIZE`、`FD_DETERMINISTIC_LOG_MODE`）
4. FlashAttention 后端确定性模式（`flash_attn_backend.py:249-255`）
5. 确定性 All-Reduce（`communication.py:114-166`，支持 float32/float16/bfloat16）
6. Splitwise Scheduler 确定性选择（`splitwise_scheduler.py:614-626`，hash-based）
7. 端到端确定性测试套件（离线推理、CE 验证、多 GPU 测试）
8. 确定性模式 seed 固定（`sampling_params.py:271`，seed=42）

### 待实现 ⏳
1. CUDA Graph 支持
2. Radix Cache 确定性
3. 量化确定性（FP8, nvfp4）
4. **Global Scheduler 随机性修复**（`global_scheduler.py:476,481`，测试已写但生产代码未改）— ⚡ 低优：当前不影响单机确定性，多机场景再处理
5. DP Attention 和 EP 支持
6. Blackwell 架构 TP 确定性

### 待清理
1. ~~`gpu_model_runner.py:1639-1641` 死代码（引用不存在的 `scheduler_config.enable_deterministic_mode`）~~ ✅ 已修复：改为使用 `envs.FD_DETERMINISTIC_MODE` / `envs.FD_DETERMINISTIC_SPLIT_KV_SIZE`

### 与 SGLang 对比

| 功能 | SGLang | FastDeploy |
|------|--------|-----------|
| 批量不变算子 | ✅ | ✅ |
| Chunked Prefill 对齐 | ✅ | ✅ |
| FlashAttention 确定性 | ✅ | ✅ |
| 确定性 All-Reduce | ✅ | ✅ |
| CUDA Graph | ✅ | ❌ |
| 温度 > 0 | ✅ | ⚠️ |
| Radix Cache | ⚠️ | ❌ |
| FP8 量化 | ❌ | ❌ |
| 单元测试 | ✅ | ✅ |
