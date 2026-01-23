# KV Cache Block Count Calculation Formula

This document explains in detail how FastDeploy calculates the number of KV Cache blocks based on available GPU memory.

## Overview

FastDeploy performs memory profiling during startup to determine how much GPU memory can be allocated for KV Cache and calculates the number of available blocks accordingly. This process ensures the system does not crash due to out-of-memory (OOM) errors.

## Calculation Process

### 1. Memory Profiling Phase

The memory profiling process is implemented in `worker_process.py` and `gpu_worker.py`, with the following main steps:

#### Step 1: Record Memory State Before Profile Run

```python
# Get current GPU device memory information
paddle.device.cuda.reset_max_memory_reserved(local_rank)
paddle.device.cuda.reset_max_memory_allocated(local_rank)
paddle_reserved_mem_before_run = paddle.device.cuda.max_memory_reserved(local_rank)
paddle_allocated_mem_before_run = paddle.device.cuda.max_memory_allocated(local_rank)

# Use NVML to get device total memory information
before_run_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# Contains: total, used, free
```

#### Step 2: Execute Profile Run

```python
self.model_runner.profile_run()
```

This runs one model forward pass to measure peak memory usage.

#### Step 3: Calculate Available KV Cache Memory

```python
# Memory state after profile
paddle_allocated_mem_after_run = paddle.device.cuda.max_memory_allocated(local_rank)
paddle_peak_increase = paddle_allocated_mem_after_run - paddle_allocated_mem_before_run

# Calculate memory available for KV Cache
available_kv_cache_memory = (
    after_run_meminfo.total * gpu_memory_utilization
    - after_run_meminfo.used
    - paddle_peak_increase
    + model_block_memory_used * total_block_num
)
```

**Formula Explanation:**
- `after_run_meminfo.total * gpu_memory_utilization`: Total GPU memory × memory utilization rate (configurable parameter)
- `after_run_meminfo.used`: Currently used GPU memory
- `paddle_peak_increase`: Peak memory increase during profile run
- `model_block_memory_used * total_block_num`: Initial block memory allocated during profiling (needs to be added back for recalculation)

### 2. Single Block Memory Requirement Calculation

The memory required for a single block is calculated by the `cal_theortical_kvcache()` function (located in various model_runners):

```python
def cal_theortical_kvcache(self):
    # Determine data type byte size
    if cache_quant_dtype is not None:  # int8, int8_zp, fp8, fp8_zp
        byte_of_dtype = 1
    else:  # default(bf16)
        byte_of_dtype = 2
    
    # Calculate hidden dimension
    hidden_dim = head_dim * kv_num_heads
    
    # Calculate number of layers (MTP architecture may need extra layers)
    num_layers = num_hidden_layers + num_gpu_block_expand_ratio  # If using MTP
    
    # Calculate memory required per block
    if mla_cache:  # Multi-Head Latent Attention
        required_memory = (
            byte_of_dtype 
            * (kv_lora_rank + qk_rope_head_dim)
            * block_size
            * num_layers
        )
    else:  # Standard Attention
        required_memory = (
            byte_of_dtype 
            * 2  # K and V caches
            * block_size 
            * hidden_dim 
            * num_layers
        )
    
    return required_memory
```

### 3. Calculate Block Count

```python
# Calculate number of blocks that can be allocated
num_blocks_local = int(available_kv_cache_memory // model_block_memory_used)

# Safety limit: Avoid illegal memory access from too many blocks
if num_blocks_local > 40000:
    num_blocks_local = min(40000, num_blocks_local)

# Multi-device synchronization: Take minimum across all devices
if ranks > 1:
    dist.all_reduce(num_blocks_local, op=dist.ReduceOp.MIN)
```

## Complete Calculation Formula

### Standard Attention Architecture

```
Available KV Cache Memory = Total GPU Memory × gpu_memory_utilization 
                           - Currently Used Memory 
                           - Profile Peak Memory Increase
                           + Initial Block Memory

Memory per Block = byte_of_dtype × 2 × block_size × hidden_dim × num_layers

Block Count = ⌊Available KV Cache Memory / Memory per Block⌋

Where:
- byte_of_dtype: Data type byte size (bf16=2, int8/fp8=1)
- block_size: Number of tokens per block (configuration parameter)
- hidden_dim: head_dim × kv_num_heads
- num_layers: Number of model layers
- ⌊⌋: Floor operation
```

### Multi-Head Latent Attention (MLA) Architecture

```
Memory per Block = byte_of_dtype × (kv_lora_rank + qk_rope_head_dim) × block_size × num_layers

Other calculation steps remain the same
```

## Key Parameters

The following parameters affect block count calculation:

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `gpu_memory_utilization` | GPU memory utilization rate | Higher value means more available blocks |
| `block_size` | Number of tokens per block | Smaller value means less memory per block, more available blocks |
| `kv_cache_quant_type` | KV Cache quantization type | int8/fp8 can reduce memory by half |
| `max_num_batched_tokens` | Maximum batched tokens | Affects peak memory during profiling |

## Code Locations

Main implementation code locations:

1. **Memory Profiling and Block Calculation**:
   - `fastdeploy/worker/worker_process.py`: Lines 570-625
   - `fastdeploy/worker/gpu_worker.py`: Lines 102-176

2. **Single Block Memory Calculation**:
   - `fastdeploy/worker/gpu_model_runner.py`: Lines 2716-2759
   - `fastdeploy/worker/xpu_model_runner.py`: Lines 1731-1768
   - `fastdeploy/worker/metax_model_runner.py`: Lines 2596-2633
   - `fastdeploy/worker/hpu_model_runner.py`: Lines 1791-1828
   - `fastdeploy/worker/gcu_model_runner.py`: Lines 1155-1192

## Important Notes

1. **Safety Limit**: The system limits block count to 40000 to avoid potential memory access errors
2. **Multi-Device Sync**: In multi-device environments, all devices use the minimum block count for consistency
3. **Dynamic Adjustment**: If calculated block count ≤ 0, the system throws an error suggesting to increase `gpu_memory_utilization` or decrease `max_num_batched_tokens`

## Optimization Recommendations

To increase available block count, consider:

1. Increase `gpu_memory_utilization` parameter (default 0.9)
2. Decrease `block_size` (default 16)
3. Enable KV Cache quantization (int8 or fp8)
4. Decrease `max_num_batched_tokens` to reduce peak memory
