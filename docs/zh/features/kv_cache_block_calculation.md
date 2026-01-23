# KV Cache Block数量计算公式

本文档详细说明FastDeploy中如何根据剩余GPU显存计算KV Cache的Block数量。

## 概述

FastDeploy在启动时会进行内存分析（profiling），以确定可以为KV Cache分配多少GPU显存，并据此计算出可用的Block数量。这个过程确保系统不会因内存不足（OOM）而崩溃。

## 计算流程

### 1. 内存分析阶段

内存分析过程在 `worker_process.py` 和 `gpu_worker.py` 中实现，主要步骤如下：

#### 步骤1：记录Profile运行前的内存状态

```python
# 获取当前GPU设备的内存信息
paddle.device.cuda.reset_max_memory_reserved(local_rank)
paddle.device.cuda.reset_max_memory_allocated(local_rank)
paddle_reserved_mem_before_run = paddle.device.cuda.max_memory_reserved(local_rank)
paddle_allocated_mem_before_run = paddle.device.cuda.max_memory_allocated(local_rank)

# 使用NVML获取设备总内存信息
before_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# 包含: total, used, free
```

#### 步骤2：执行Profile运行

```python
self.model_runner.profile_run()
```

这会运行一次模型前向传播，以测量峰值内存使用量。

#### 步骤3：统计内存信息并计算可用KV Cache内存

```python
# Profile后的内存状态
paddle_allocated_mem_after_run = paddle.device.cuda.max_memory_allocated(local_rank)
paddle_peak_increase = paddle_allocated_mem_after_run - paddle_allocated_mem_before_run

# 计算可用于KV Cache的内存
available_kv_cache_memory = (
    after_run_meminfo.total * gpu_memory_utilization
    - after_run_meminfo.used
    - paddle_peak_increase
    + model_block_memory_used * total_block_num
)
```

**公式说明：**
- `after_run_meminfo.total * gpu_memory_utilization`：GPU总内存 × 内存利用率（可通过参数配置）
- `after_run_meminfo.used`：当前已使用的GPU内存
- `paddle_peak_increase`：Profile运行期间的内存峰值增量
- `model_block_memory_used * total_block_num`：Profile时分配的初始Block内存（需要加回来重新计算）

### 2. 单个Block的内存需求计算

单个Block所需的内存由 `cal_theortical_kvcache()` 函数计算（位于各个model_runner中）：

```python
def cal_theortical_kvcache(self):
    # 确定数据类型的字节数
    if cache_quant_dtype is not None:  # int8, int8_zp, fp8, fp8_zp
        byte_of_dtype = 1
    else:  # default(bf16)
        byte_of_dtype = 2
    
    # 计算隐藏维度
    hidden_dim = head_dim * kv_num_heads
    
    # 计算层数（MTP架构可能需要额外的层）
    num_layers = num_hidden_layers + num_gpu_block_expand_ratio  # 如果使用MTP
    
    # 计算单个Block所需内存
    if mla_cache:  # Multi-Head Latent Attention
        required_memory = (
            byte_of_dtype 
            * (kv_lora_rank + qk_rope_head_dim)
            * block_size
            * num_layers
        )
    else:  # 标准Attention
        required_memory = (
            byte_of_dtype 
            * 2  # K和V两个缓存
            * block_size 
            * hidden_dim 
            * num_layers
        )
    
    return required_memory
```

### 3. 计算Block数量

```python
# 计算可以分配的Block数量
num_blocks_local = int(available_kv_cache_memory // model_block_memory_used)

# 安全限制：避免过多Block导致非法内存访问
if num_blocks_local > 40000:
    num_blocks_local = min(40000, num_blocks_local)

# 多卡同步：取所有设备中的最小值
if ranks > 1:
    dist.all_reduce(num_blocks_local, op=dist.ReduceOp.MIN)
```

## 完整计算公式

### 标准Attention架构

```
可用KV Cache内存 = GPU总内存 × gpu_memory_utilization 
                  - 当前已使用内存 
                  - Profile峰值内存增量
                  + 初始Block内存

单Block内存 = byte_of_dtype × 2 × block_size × hidden_dim × num_layers

Block数量 = ⌊可用KV Cache内存 / 单Block内存⌋

其中：
- byte_of_dtype: 数据类型字节数（bf16=2, int8/fp8=1）
- block_size: 每个Block的token数量（配置参数）
- hidden_dim: head_dim × kv_num_heads
- num_layers: 模型层数
- ⌊⌋: 向下取整
```

### Multi-Head Latent Attention (MLA)架构

```
单Block内存 = byte_of_dtype × (kv_lora_rank + qk_rope_head_dim) × block_size × num_layers

其他计算步骤相同
```

## 关键参数

以下参数会影响Block数量的计算：

| 参数 | 说明 | 影响 |
|-----|------|------|
| `gpu_memory_utilization` | GPU内存利用率 | 值越大，可用Block越多 |
| `block_size` | 每个Block的token数 | 值越小，单Block内存越小，可用Block越多 |
| `kv_cache_quant_type` | KV Cache量化类型 | int8/fp8可减少一半内存 |
| `max_num_batched_tokens` | 最大批处理token数 | 影响Profile时的内存峰值 |

## 代码位置

主要实现代码位于：

1. **内存分析和Block计算**：
   - `fastdeploy/worker/worker_process.py`: 第570-625行
   - `fastdeploy/worker/gpu_worker.py`: 第102-176行

2. **单Block内存计算**：
   - `fastdeploy/worker/gpu_model_runner.py`: 第2716-2759行
   - `fastdeploy/worker/xpu_model_runner.py`: 第1731-1768行
   - `fastdeploy/worker/metax_model_runner.py`: 第2596-2633行
   - `fastdeploy/worker/hpu_model_runner.py`: 第1791-1828行
   - `fastdeploy/worker/gcu_model_runner.py`: 第1155-1192行

## 注意事项

1. **安全限制**：系统会将Block数量限制在40000以内，避免潜在的内存访问错误
2. **多卡同步**：在多卡环境下，所有设备会取最小的Block数量以保持一致性
3. **动态调整**：如果计算出的Block数量≤0，系统会抛出错误，提示增加`gpu_memory_utilization`或减少`max_num_batched_tokens`

## 优化建议

如果需要增加可用的Block数量，可以考虑：

1. 增加 `gpu_memory_utilization` 参数（默认0.9）
2. 减小 `block_size`（默认16）
3. 启用KV Cache量化（int8或fp8）
4. 减小 `max_num_batched_tokens` 以降低峰值内存
