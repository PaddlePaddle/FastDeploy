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
1. 服务部署时推荐配置环境变量
```
export ENABLE_V1_KVCACHE_SCHEDULER=1
```

2. 服务在启动时需要配置max-num-seqs，此参数用于表示Decode阶段的最大Batch数，如果并发超过此值，则超出的请求会排队等待处理, 常规情况下你可以将max-num-seqs配置为128，保持在较高的范围，实际并发由发压客户端来决定。

3. max-num-seqs仅表示设定的上限，但实际上服务能并发处理的上限取决于KVCache的大小，在启动服务后，查看log/worker_process.log会看到类似num_blocks_global: 17131的日志，这表明当前服务的KVCache Block数量为17131, 17131block_size(默认64）即知道总共可缓存的Token数量，例如此处为1713164=1096384。如果你的请求数据平均输入和输出Token之和为20K，那么服务实际可以处理的并发大概为1096384/20k=53
