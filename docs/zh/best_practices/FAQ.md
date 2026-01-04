[English](../../best_practices/FAQ.md)

# 常见问题FAQ
## 1.显存不足
1. 启动服务时显存不足：
- 核对模型和量化方式对应的部署最小卡数，如果不满足则需要增加部署卡数
- 如果开启了CUDAGraph，尝试通过降低 `gpu_memory_utilization`来为CUDAGraph留存更多的显存，或通过减少 `max_num_seqs`，设置`cudagraph_capture_sizes`来减少CUDAGraph的显存占用。

2. 服务运行期间显存不足：
- 检查log中是否有类似如下信息，如有，通常是输出block不足导致，需要减小`kv-cache-ratio`
```
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 133， encoder block len: 24
recover seq_id: 2， free_list_len: 144， used_list_len: 134
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 144， encoder_block_len: 24
```

建议启用服务管理全局 Block功能，在启动服务前，加入环境变量
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

## 2.模型性能差
1. 首先检查输出长度是否符合预期，是否是解码过长导致。
如果场景输出本身较长，请检查log中是否有类似如下信息，如有，通常是输出block不足导致，需要减小`kv-cache-ratio`
```
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 133， encoder block len: 24
recover seq_id: 2， free_list_len: 144， used_list_len: 134
need_block_len: 1， free_list_len: 0
step max_id: 2， max_num: 144， encoder_block_len: 24
```
同样建议启用服务管理全局 Block功能，在启动服务前，加入环境变量
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

2. 检查自动profile分配的KVCache block是否符合预期，如果自动profile中受到显存波动影响可能导致分配偏少，可以通过手工设置`num_gpu_blocks_override`参数扩大KVCache block。

## 3.服务可以支持多大并发？

1. 服务部署时推荐配置以下环境变量

   ```
   export ENABLE_V1_KVCACHE_SCHEDULER=1
   ```

2. 服务启动时需要配置 `max-num-seqs`
   该参数表示 Decode 阶段的**最大 Batch 数**，当并发超过该值时，多余的请求会进入排队等待处理。
   一般情况下，你可以将 `max-num-seqs` 配置为 **128**，保持在较高范围；实际并发能力由压测客户端决定。

3. `max-num-seqs` 仅表示**配置的上限**，但服务真正能支持的并发量取决于 **KVCache 的总大小**
   服务启动后，在 `log/worker_process.log` 中会看到类似：

   ```
   num_blocks_global: 17131
   ```

   这表示当前服务的 KVCache Block 数量为 **17131**，若 `block_size = 64`（默认），则可缓存 Token 总量为：

   ```
   17131 * 64 = 1,096,384 tokens
   ```

   如果你的请求平均（输入 + 输出）为 **20K tokens**，那么服务实际能支持的并发大约为：

   ```
   1,096,384 / 20,000 ≈ 53
   ```

## 4. 启用 logprobs 后推理请求卡住

启用 **logprobs** 后，推理结果会附带每个 token 的logprobs信息，使**单条消息体显著变大**。在默认配置下，这可能触发 **System V Message Queue** 的消息大小限制，从而导致推理任务token输出**卡住**。

不同模式下（MTP / 非 MTP）logprobs 会导致消息体膨胀的规模不同，具体计算如下。

### 消息体大小计算

1. **非 MTP 模式 + logprobs**
   单条消息体大小：

   ```
   ((512 * (20 + 1)) + 2) * 8
   + 512 * (20 + 1) * 4
   + 512 * 8
   = 133136 bytes
   ```

2. **MTP 模式 + logprobs**
   单条消息体大小：

   ```
   (512 * 6 * (20 + 1) + 512 + 3) * 8
   + 512 * 6 * (20 + 1) * 4
   + 512 * 6 * 8
   = 802840 bytes
   ```

### 问题原因

通过 `ipcs -l` 查看系统默认的 System V 消息队列限制，常见设置如下：

```
------ Messages Limits --------
max queues system wide = 32000
max size of message (bytes) = 8192
default max size of queue (bytes) = 16384
```

当单条消息体大小**超过 max size of message（默认 8192 bytes）** 时，进程间通信会被阻塞，最终表现为推理请求卡住。

### 解决方案

**调大 System V Message Queue 的消息大小限制。**

由于 MTP 下的消息体可接近 800 KB，建议将**单条消息大小限制提升至 1MB（1048576 bytes）**。

Linux 系统可通过以下命令调整：

```
# 提高单条消息的最大允许大小
sysctl -w kernel.msgmax=1048576

# 提高单个消息队列的最大容量
sysctl -w kernel.msgmnb=268435456
```

> **注意**: 若在 Docker 容器中运行，需要启用特权模式（`--privileged`），或在启动参数中显式设置相关内核参数。

### 废弃说明

当前基于 System V Message Queue 的通信机制将在后续版本中被废弃。未来将迁移到更稳定、更高效的通信方式，以彻底解决上述限制问题。
