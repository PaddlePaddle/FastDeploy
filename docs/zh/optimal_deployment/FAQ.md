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
