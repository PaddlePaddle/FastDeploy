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
