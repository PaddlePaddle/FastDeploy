# FastDeploy 确定性推理分析报告

## 一、背景与参考

参考 [SGLang Issue #10278](https://github.com/sgl-project/sglang/issues/10278) 和 [SGLang 确定性推理博客](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)，LLM 推理的不确定性主要来源于以下几方面：

1. **Batch-Invariant Kernels**：不同批量大小时，reduction splitting 过程会变化，导致浮点运算顺序不同。使用批量不变实现确保相同输入产生相同输出。
2. **Chunked Prefill 对齐**：确保每个单元能被同一个 attention kernel 完整处理。
3. **确定性采样**：使用带种子的哈希函数生成 Gumbel 噪声，相同的 `(inputs, seed)` 对总是产生相同的采样结果。
4. **确定性 All-Reduce**：Tensor Parallelism 通信使用确定性 kernel 避免浮点累加顺序差异。

### 参考资料

- [SGLang Issue #10278: Support deterministic inference with Batch Invariant Ops](https://github.com/sgl-project/sglang/issues/10278)
- [SGLang 确定性推理博客](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [Thinking Machines Lab Blog: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [batch_invariant_ops GitHub](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [SGLang 官方文档](https://docs.sglang.ai/)

---

## 二、SGLang 确定性实现现状

### Attention Backend

| 后端 | 状态 | 说明 |
|------|------|------|
| CUDA Graph | ✅ | 支持 CUDA Graph 加速确定性推理 (#10645) |
| Temperature > 0 | ✅ | 支持非零温度的确定性推理 (#10678) |
| FlashInfer | ✅ | [PR #1675](https://github.com/flashinfer-ai/flashinfer/pull/1675) (#10645) |
| Triton | ✅ | #10425, #10694 |
| FlashAttention-3 | ✅ | #10651 |
| 批量不变 Triton Kernels | ✅ | 已移入主仓库 (#10695) |

### 通信 (NCCL)

| 功能 | 状态 | 说明 |
|------|------|------|
| 确定性 All-Reduce (NVIDIA) | ✅ | 支持 Tensor Parallelism (#10930) |
| 确定性 All-Reduce (AMD) | ✅ | #15340 |
| Blackwell TP4 | ❌ | 已知问题 (#11513) |

### Radix Cache

| 功能 | 状态 | 说明 |
|------|------|------|
| FA3 | ✅ | 单阶段 prefill，天然支持 |
| Triton | ✅ | [PR #11147](https://github.com/sgl-project/sglang/pull/11147) |
| FlashInfer | ❌ | 待支持 |
| Prefill 与 Radix Cache 输出一致性 | ❌ | 待验证 |

### 模型支持

| 模型 | 状态 | 说明 |
|------|------|------|
| Qwen3 Dense / MOE | ✅ | 已支持 |
| DeepSeek v3 | ✅ | #12095 |
| Qwen3 Next (Linear Attention) | ❌ | #12845 |

### 量化 / 并行 / 投机解码

| 功能 | 状态 | 说明 |
|------|------|------|
| Blockwise FP8 Kernel | ❌ | [PR #11491](https://github.com/sgl-project/sglang/pull/11491) |
| Per-channel FP8 / FP8 MoE / nvfp4 | ❌ | 待实现 |
| DP Attention | ❌ | [PR #11023](https://github.com/sgl-project/sglang/pull/11023) |
| EP (Expert Parallelism) | ❌ | 待实现 |
| Speculative Decoding | ❌ | [#8391](https://github.com/sgl-project/sglang/issues/8391), [#9877](https://github.com/sgl-project/sglang/issues/9877) |

---

## 三、FastDeploy 与 SGLang 对比

| 功能 | SGLang | FastDeploy | 说明 |
|------|--------|-----------|------|
| 批量不变算子 | ✅ | ✅ | matmul、mean、log_softmax、addmm |
| Chunked Prefill 对齐 | ✅ | ✅ | 3 分支对齐逻辑 |
| FlashAttention 确定性 | ✅ | ✅ | 通过 `FD_DETERMINISTIC_MODE` 控制 |
| 确定性 All-Reduce | ✅ | ✅ | 支持 float32/float16/bfloat16 |
| 确定性采样 (seed) | ✅ | ✅ | 确定性模式下 seed=42 |
| Splitwise Scheduler | N/A | ✅ | hash-based 确定性选择 |
| CUDA Graph | ✅ | ⚠️ | 功能已有，确定性下待验证 |
| 温度 > 0 采样 | ✅ | ⚠️ | 有 seed 支持，待完善 |
| Radix Cache | ⚠️ | ❌ | 待实现 |
| FP8 量化 | ❌ | ❌ | 双方均未完成 |
| DP Attention / EP | ❌ | ❌ | 双方均未完成 |
| Qwen3 Dense | ✅ | ❓ | 待验证 |
| Qwen3 MOE | ✅ | ❓ | 待验证 |
| DeepSeek v3 | ✅ | ❓ | 待验证 |
| Linear Attention (Qwen3 Next) | ❌ | ❌ | 双方均未支持 |
| MoE 模型确定性 | ✅ | ❓ | 待验证（涉及 Expert 路由确定性） |

---

## 四、FastDeploy 已实现功能详解

### 4.1 环境变量

定义在 `fastdeploy/envs.py:210-214`：

```bash
export FD_DETERMINISTIC_MODE=1              # 启用确定性模式
export FD_DETERMINISTIC_SPLIT_KV_SIZE=16    # Split KV block size（必须为 2 的幂，默认 16）
export FD_DETERMINISTIC_LOG_MODE=1          # 启用确定性日志（MD5 哈希 + 调试信息）
```

### 4.2 批量不变算子

**代码位置**：`fastdeploy/model_executor/layers/batch_invariant_ops/batch_invariant_ops.py`

已实现的算子：
- `matmul_persistent` — 基于 Triton 的持久化矩阵乘法
- `log_softmax` — Triton 实现
- `mean_dim` — Triton 实现的 mean 归约
- `addmm_batch_invariant` — 批量不变的 addmm

使用方式：
```python
from fastdeploy.model_executor.layers.batch_invariant_ops import set_batch_invariant_mode

with set_batch_invariant_mode(True):
    result = model.generate(...)
```

Attention block size 由 `get_batch_invariant_attention_block_size()` 返回，默认 `block_m=16, block_n=16`。

### 4.3 Chunked Prefill 对齐

**代码位置**：`fastdeploy/engine/sched/resource_manager_v1.py:466-491`

通过 `FD_DETERMINISTIC_MODE` 控制，包含 3 个分支：

| 分支 | 条件 | 行为 |
|------|------|------|
| Final chunk | 剩余 token < split_kv_size | 无需对齐，直接处理 |
| Boundary 对齐 | budget 足够到达下一边界 | 对齐到 split_kv_size 整数倍 |
| Defer | budget 不足到达下一边界 | 返回 0，延迟到下一轮 |

### 4.4 FlashAttention 后端

**代码位置**：`fastdeploy/model_executor/layers/attention/flash_attn_backend.py:249-255`

确定性模式下自动将 `decoder_block_shape_q` 设置为 `FD_DETERMINISTIC_SPLIT_KV_SIZE`。

### 4.5 确定性 All-Reduce

**代码位置**：`fastdeploy/distributed/communication.py:114-166`，实现在 `fastdeploy/distributed/custom_all_reduce.py`

`FD_DETERMINISTIC_MODE` 开启时，自动使用 `CustomAllreduce` 替代标准 NCCL。

特性：
- 支持 float32/float16/bfloat16
- 16-byte 对齐检查
- Lazy initialization：首次调用时自动初始化
- 静态模式下 fail-fast

### 4.6 Splitwise Scheduler 确定性选择

**代码位置**：`fastdeploy/scheduler/splitwise_scheduler.py:614-626`

确定性模式下基于 `hash(str(req.request_id))` 选择节点，替代 `random.choice()`。

### 4.7 Sampler seed 处理

**代码位置**：
- `fastdeploy/engine/sampling_params.py:269-274` — 确定性模式下 seed 固定为 42
- `fastdeploy/model_executor/layers/sample/sampler.py:86-92` — 根据解码位置偏移 seed：`(seed + offset) % MAX_INFER_SEED`

---

## 五、待实现功能

### 5.1 Global Scheduler 随机性修复（低优）

**代码位置**：`fastdeploy/scheduler/global_scheduler.py:476,481`

`random.sample()` 和 `random.choice()` 仍在使用。仅影响多机调度场景，单机确定性不受影响。测试已在 `tests/scheduler/deterministic/test_global_scheduler_determinism.py` 中实现 hash-based 策略，但生产代码尚未应用。

### 5.2 Radix Cache 确定性

确保有/无 Radix Cache 的 prefill 输出一致。SGLang 在 Triton 和 FA3 后端已部分支持。

### 5.3 量化确定性

支持 FP8/nvfp4 的确定性推理（Blockwise FP8 Kernel、Per-channel FP8 Gemm、FP8 Fused MoE、nvfp4）。SGLang 同样处于开发中。

### 5.4 CUDA Graph 确定性验证

CUDA Graph 功能已实现（`graph_optimization_backend.py`），确定性测试中已启用（`start_fd.sh: use_cudagraph:true`）。待验证确定性模式下 CUDA Graph replay 结果是否严格一致。

### 5.5 Temperature > 0 确定性采样

完善非零温度下的采样确定性，参考 SGLang 的 Gumbel 噪声方案。

### 5.6 DP Attention / EP 支持

SGLang 同样处于开发阶段。

### 5.7 模型确定性验证

SGLang 已验证 Qwen3 Dense/MOE、DeepSeek v3 的确定性推理。FastDeploy 需要逐模型验证：

| 模型 | SGLang | FastDeploy | 说明 |
|------|--------|-----------|------|
| Qwen3 Dense | ✅ | ❓ 待验证 | 基础 Dense 模型 |
| Qwen3 MOE | ✅ | ❓ 待验证 | MoE 路由涉及 Top-K 选择，需验证 Expert 分配确定性 |
| DeepSeek v3 | ✅ | ❓ 待验证 | SGLang #12095 |
| Qwen3 Next (Linear Attention) | ❌ | ❌ | 新架构，双方均未支持 (SGLang #12845) |

验证要点：
- **Dense 模型**：主要验证 batch-invariant 算子 + attention + all-reduce 的端到端确定性
- **MoE 模型**：额外需要验证 Expert 路由（Top-K gating）的确定性，不同 batch 下 Expert 分配是否一致
- **长序列场景**：验证 Chunked Prefill 在不同切分下输出是否一致

---

## 六、关键代码位置速查

| 功能 | 文件位置 | 说明 |
|------|---------|------|
| 环境变量定义 | `fastdeploy/envs.py:210-214` | `FD_DETERMINISTIC_MODE` 等 |
| 批量不变算子 | `fastdeploy/model_executor/layers/batch_invariant_ops/batch_invariant_ops.py` | matmul、log_softmax、mean、addmm |
| 批量不变开关 | 同上 `:572-583` | `set_batch_invariant_mode()` 上下文管理器 |
| Attention block size | 同上 `:589-590` | `get_batch_invariant_attention_block_size()` |
| Chunked Prefill 对齐 | `fastdeploy/engine/sched/resource_manager_v1.py:466-491` | 3 分支对齐逻辑 |
| FlashAttention 后端 | `fastdeploy/model_executor/layers/attention/flash_attn_backend.py:249-255` | 覆盖 `decoder_block_shape_q` |
| 确定性 All-Reduce | `fastdeploy/distributed/communication.py:114-166` | 强制 `CustomAllreduce` |
| CustomAllreduce 实现 | `fastdeploy/distributed/custom_all_reduce.py` | kernel 实现 |
| Sampler seed | `fastdeploy/engine/sampling_params.py:269-274` | seed=42 |
| Seed 位置偏移 | `fastdeploy/model_executor/layers/sample/sampler.py:86-92` | `(seed + offset) % MAX_INFER_SEED` |
| Splitwise 确定性选择 | `fastdeploy/scheduler/splitwise_scheduler.py:606-626` | hash-based |
| Global Scheduler 随机性 | `fastdeploy/scheduler/global_scheduler.py:476,481` | **待修复**（低优） |

---

## 七、测试文件总览

### 批量不变算子测试

| 测试文件 | 说明 |
|---------|------|
| `tests/batch_invariant/test_batch_invariance_op_mm.py` | 矩阵乘法批量不变性验证 |
| `tests/batch_invariant/test_batch_invariance_op_mean.py` | Mean 归约批量不变性验证 |
| `tests/batch_invariant/test_batch_invariance_op_logsoftmax.py` | LogSoftmax 批量不变性验证（含极端数值范围） |
| `tests/batch_invariant/test_batch_invariance_op_addmm.py` | AddMM 批量不变性验证 |
| `tests/batch_invariant_ops/test_batch_invariant_ops.py` | 综合单测：模式控制、多形状/多 dtype |

### Scheduler 确定性测试

| 测试文件 | 说明 |
|---------|------|
| `tests/scheduler/test_chunked_prefill_determinism.py` | Chunked Prefill 确定性：token 对齐、boundary case 等 |
| `tests/scheduler/deterministic/test_global_scheduler_determinism.py` | Global Scheduler 确定性（测试已写，生产代码未改） |

### 分布式确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/distributed/allreduce_deterministic.py` | 多 GPU All-Reduce 确定性（20 轮） | 2+ GPU |
| `tests/distributed/test_allreduce_deterministic_launch.py` | pytest 启动器 | 2+ GPU |

### Attention 确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/layers/test_flash_attention_versions_determinism.py` | FA2/FA3 确定性（20 个测试） | GPU |
| `tests/layers/test_paddle_attention_determinism.py` | Paddle SDPA + FastDeploy 集成测试 | GPU |
| `tests/layers/test_paddle_attention_determinism_standalone.py` | 独立 Paddle SDPA 确定性测试 | GPU |

### 采样与环境变量测试

| 测试文件 | 说明 |
|---------|------|
| `tests/engine/test_sampling_params_determinism.py` | SamplingParams seed 行为 |
| `tests/envs/test_deterministic_env_vars.py` | 环境变量默认值、组合设置、power-of-two 校验 |

### 端到端确定性测试

| 测试文件 | 说明 | 环境要求 |
|---------|------|---------|
| `tests/deterministic/test_determinism_offline.py` | 离线推理确定性（13 个场景） | GPU + 模型 |
| `tests/deterministic/test_determinism_standalone.py` | 轻量级独立确定性测试 | 无 |
| `tests/ce/deterministic/test_determinism_verification.py` | CE 端到端 MD5 验证 | 运行中的服务 |
| `tests/ce/deterministic/start_fd.sh` | 单 GPU 启动脚本（TP=1，Qwen2.5-7B） | 1 GPU |
| `tests/ce/deterministic/start_fd_tp4.sh` | 多 GPU 启动脚本（TP=4，Qwen2.5-7B） | 4 GPU |

---

## 八、实施路线

### 阶段一：基础确定性 ✅ 已完成
- 环境变量（`FD_DETERMINISTIC_MODE`、`FD_DETERMINISTIC_SPLIT_KV_SIZE`、`FD_DETERMINISTIC_LOG_MODE`）
- 批量不变算子（matmul、mean、log_softmax、addmm）
- Chunked Prefill 对齐
- FlashAttention 后端确定性模式
- 基础确定性单测

### 阶段二：并行确定性 ✅ 已完成
- 确定性 All-Reduce kernel（float32/float16/bfloat16）
- Splitwise Scheduler hash-based 选择
- 多 GPU 确定性测试
- 端到端确定性测试套件

### 阶段三：增强与验证 ✅ 已完成
- CUDA Graph 确定性验证
- Global Scheduler 随机性修复（低优，仅影响多机场景）
- Temperature > 0 确定性采样完善

### 阶段四：高级特性 ⏳ 待实现
- Radix Cache 确定性
- 量化确定性（FP8, nvfp4）
- DP Attention / EP 支持
- Blackwell 架构 TP 确定性

### 阶段五：模型验证 ⏳ 待实现
- Qwen3 Dense 确定性验证
- Qwen3 MOE 确定性验证（含 Expert 路由确定性）
- DeepSeek v3 确定性验证
- MoE 模型 Expert 分配确定性验证
- Linear Attention 支持（依赖 SGLang 进展）

### 阶段六：性能优化 ⏳ 待实现
- 加速批量不变 Triton Kernels
- CUDA Graph 性能优化
