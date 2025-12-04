[简体中文](../zh/best_practices/FAQ.md)

# FAQ
## 1.CUDA out of memory
1. when starting the service：
- Check the minimum number of deployment GPUs corresponding to the model and quantification method. If it is not met, increase the number of deployment GPUs.
- If CUDAGraph is enabled, try to reserve more GPU memory for CUDAGraph by lowering `gpu_memory_utilization`, or reduce the GPU memory usage of CUDAGraph by reducing `max_num_seqs` and setting `cudagraph_capture_sizes`。

2. during service operation:
- Check whether there is information similar to the following in the log. If so, it is usually caused by insufficient output blocks. You need to reduce `kv-cache-ratio`
```
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 133， encoder block len: 24
recover seq_id: 2， free_list_len: 144， used_list_len: 134
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 144， encoder_block_len: 24
```

It is recommended to enable the service management global block. You need add environment variables before starting the service.
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

## 2.Poor model performance
1. First, check whether the output length meets expectations and whether it is caused by excessive decoding length. If the output is long, please check whether there is similar information as follows in the log. If so, it is usually caused by insufficient output blocks and you need to reduce `kv-cache-ratio`
```
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 133， encoder block len: 24
recover seq_id: 2， free_list_len: 144， used_list_len: 134
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 144， encoder_block_len: 24
```

It is also recommended to enable the service management global block. You need add environment variables before starting the service.
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

2. Check whether the KVCache blocks allocated by the automatic profile are as expected. If the automatic profile is affected by the fluctuation of video memory and may result in less allocation, you can manually set the `num_gpu_blocks_override` parameter to expand the KVCache block.

## 3. How much concurrency can the service support?

1. It is recommended to configure the following environment variable when deploying the service:

   ```
   export ENABLE_V1_KVCACHE_SCHEDULER=1
   ```

2. When starting the service, you need to configure `max-num-seqs`.
   This parameter specifies the maximum batch size during the Decode phase.
   If the concurrency exceeds this value, the extra requests will be queued.
   Under normal circumstances, you can set `max-num-seqs` to **128** to keep it relatively high; the actual concurrency is determined by the load-testing client.

3. `max-num-seqs` represents only the upper limit you configure.
   The **actual** concurrency the service can handle depends on the size of the **KVCache**.
   After the service starts, check `log/worker_process.log` and look for logs similar to:

   ```
   num_blocks_global: 17131
   ```

   This indicates that the current service has **17131 KVCache blocks**.
   With `block_size = 64` (default), the total number of tokens that can be cached is:

   ```
   17131 * 64 = 1,096,384 tokens
   ```

   If the average total number of tokens per request (input + output) is **20K**, then the service can actually support approximately:

   ```
   1,096,384 / 20,000 ≈ 53 concurrent requests
   ```

## 4. Inference Request Stalls After Enabling logprobs

When **logprobs** is enabled, the inference output includes the log-probability of each token, which **significantly increases the size of each message body**. Under default settings, this may exceed the limits of the **System V Message Queue**, causing the inference request to **stall**.

The increase in message size differs between MTP and non-MTP modes. The calculations are shown below.

### Message Size Calculation

1. **Non-MTP + logprobs enabled**
   Size of a single message:

   ```
   ((512 * (20 + 1)) + 2) * 8
   + 512 * (20 + 1) * 4
   + 512 * 8
   = 133136 bytes
   ```

2. **MTP + logprobs enabled**
   Size of a single message:

   ```
   (512 * 6 * (20 + 1) + 512 + 3) * 8
   + 512 * 6 * (20 + 1) * 4
   + 512 * 6 * 8
   = 802840 bytes
   ```

### Root Cause

Running `ipcs -l` typically shows the default System V message queue limits:

```
------ Messages Limits --------
max queues system wide = 32000
max size of message (bytes) = 8192
default max size of queue (bytes) = 16384
```

If a single message **exceeds the `max size of message` limit (usually 8192 bytes)**, inter-process communication becomes blocked, causing the inference task to stall.

### Solution

**Increase the System V message queue size limits.**

Since message sizes can approach 800 KB in MTP mode, it is recommended to increase the **maximum message size to at least 1 MB (1048576 bytes)**.

Use the following commands on Linux:

```
# Increase maximum size of a single message
sysctl -w kernel.msgmax=1048576

# Increase maximum capacity of a message queue
sysctl -w kernel.msgmnb=268435456
```

> **Note:** If running inside a Docker container, privileged mode (`--privileged`) is required, or you must explicitly set these kernel parameters via container startup options.

### Deprecation Notice

This System V message queue–based communication mechanism will be **deprecated in future releases**. Subsequent versions will migrate to a more robust communication method that eliminates the limitations described above.
