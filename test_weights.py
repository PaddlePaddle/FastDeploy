import numpy as np
import paddle
import torch

# 加载两个框架保存的权重分片
vllm_shard_torch = torch.load("/home/aistudio/work/vllm/vllm_qkv_weight_shard_rank0.pt")
fd_shard_paddle = paddle.load("fd_qkv_weight_shard_rank0.pdparams")

# 将它们都转换为 numpy 数组
vllm_shard_np = vllm_shard_torch.numpy()
fd_shard_np = fd_shard_paddle.numpy()

print(f"vLLM shard shape: {vllm_shard_np.shape}") # 预期 [out, in] e.g., [6144, 6144]
print(f"FD shard shape:   {fd_shard_np.shape}")   # 预期 [in, out] e.g., [6144, 6144]

# 检查 shape 是否匹配 (转置后)
if vllm_shard_np.shape != fd_shard_np.T.shape:
    print("\n[ERROR] Shape mismatch after transpose!")
    print(f"vLLM shape: {vllm_shard_np.shape}, FD transposed shape: {fd_shard_np.T.shape}")
    exit()

# 使用 numpy.allclose 进行逐元素比较
# atol (absolute tolerance) 是关键参数，我们设一个比较严格但合理的阈值
atol = 1e-4 
is_close = np.allclose(vllm_shard_np, fd_shard_np.T, atol=atol)

print(f"\nComparing with absolute tolerance (atol) = {atol}")
if is_close:
    print("\n[SUCCESS] The weight shards are numerically close!")
else:
    print("\n[FAILURE] The weight shards are DIFFERENT!")
    
    # 如果不一致，计算并打印差异的详细信息
    diff = np.abs(vllm_shard_np - fd_shard_np.T)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
    
    print("\n--- Difference Analysis ---")
    print(f"  Maximum absolute difference: {max_diff:.8f}")
    print(f"  Average absolute difference: {avg_diff:.8f}")
    print(f"  Location of max difference: {max_diff_indices}")
    print(f"  Value at vLLM: {vllm_shard_np[max_diff_indices]}")
    print(f"  Value at FD:   {fd_shard_np.T[max_diff_indices]}")
    print("-" * 25)

# 打印统计数据以供参考
print("\n--- vLLM Shard Stats ---")
print(f"  Mean: {np.mean(vllm_shard_np):.8f}, Std: {np.std(vllm_shard_np):.8f}")
print(f"  Max:  {np.max(vllm_shard_np):.8f}, Min: {np.min(vllm_shard_np):.8f}")

print("\n--- FD Shard Stats (Transposed) ---")
print(f"  Mean: {np.mean(fd_shard_np.T):.8f}, Std: {np.std(fd_shard_np.T):.8f}")
print(f"  Max:  {np.max(fd_shard_np.T):.8f}, Min: {np.min(fd_shard_np.T):.8f}")