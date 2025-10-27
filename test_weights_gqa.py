import numpy as np
import paddle
import torch

# --- 配置 ---
# 指定要比较的层号
LAYER_ID = 7 
# -----------------

def print_stats(tensor_np, name):
    """打印 Numpy 数组的统计信息"""
    print(f"\n--- {name} Stats ---")
    if tensor_np is None or tensor_np.size == 0:
        print("  Tensor is None or empty.")
        return
        
    print(f"  Shape: {tensor_np.shape}")
    print(f"  Dtype: {tensor_np.dtype}")
    print(f"  Mean:  {np.mean(tensor_np):.8f}")
    print(f"  Std:   {np.std(tensor_np):.8f}")
    print(f"  Max:   {np.max(tensor_np):.8f}")
    print(f"  Min:   {np.min(tensor_np):.8f}")
    print(f"  First 5 flat values: {tensor_np.flatten()[:5]}")
    print("-" * (len(name) + 12))


try:
    # 1. 加载两个框架保存的权重分片
    print(f"Loading vLLM weight shard for GQA Layer {LAYER_ID}...")
    vllm_shard_torch = torch.load(f"/home/aistudio/work/vllm/vllm_gqa_l7_qkv_weight_shard_rank0.pt")
    
    print(f"Loading FastDeploy weight shard for GQA Layer {LAYER_ID}...")
    fd_shard_paddle = paddle.load(f"fd_gqa_l{LAYER_ID}_qkv_weight_shard_rank0.pdparams")

    # 2. 将它们都转换为 numpy 数组
    vllm_shard_np = vllm_shard_torch.numpy()
    fd_shard_np = fd_shard_paddle.numpy()

    print_stats(vllm_shard_np, "vLLM Shard")
    print_stats(fd_shard_np, "FastDeploy Shard")

    # 3. 检查 shape 是否匹配 (FD的需要转置)
    #    vLLM (Torch) layout: [output_features, input_features]
    #    FD (Paddle) layout:  [input_features, output_features]
    print(f"\nTransposing FastDeploy shard for comparison (from {fd_shard_np.shape} to {fd_shard_np.T.shape})...")
    if vllm_shard_np.shape != fd_shard_np.T.shape:
        print("\n[FATAL ERROR] Shape mismatch after transpose!")
        print(f"vLLM shape: {vllm_shard_np.shape}, FD transposed shape: {fd_shard_np.T.shape}")
        exit()
    else:
        print("Shapes match after transpose. Proceeding to comparison.")

    # 4. 使用 numpy.allclose 进行逐元素比较
    #    atol (absolute tolerance) 是关键参数，我们设一个比较严格但合理的阈值
    atol = 1e-4 
    is_close = np.allclose(vllm_shard_np, fd_shard_np.T, atol=atol)

    print(f"\nComparing with absolute tolerance (atol) = {atol}...")
    if is_close:
        print("\n" + "="*15 + " [SUCCESS] " + "="*15)
        print("The weight shards for GQA Layer are numerically close!")
        print("="*41)
    else:
        print("\n" + "!"*15 + " [FAILURE] " + "!"*15)
        print("The weight shards for GQA Layer are DIFFERENT!")
        print("!"*41)
        
        # 如果不一致，计算并打印差异的详细信息
        diff = np.abs(vllm_shard_np - fd_shard_np.T)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
        
        print("\n--- Difference Analysis ---")
        print(f"  Maximum absolute difference: {max_diff:.8f}")
        print(f"  Average absolute difference: {avg_diff:.8f}")
        print(f"  Location of max difference (row, col): {max_diff_indices}")
        print(f"  Value at this location in vLLM shard: {vllm_shard_np[max_diff_indices]:.8f}")
        print(f"  Value at this location in FD shard (transposed): {fd_shard_np.T[max_diff_indices]:.8f}")
        print("-" * 25)

    # 打印转置后的 FD 统计数据以供直接对比
    print_stats(fd_shard_np.T, "FastDeploy Shard (Transposed)")

except FileNotFoundError as e:
    print(f"\n[ERROR] Could not find a weight file: {e.filename}")
    print("Please make sure you have run both vLLM and FastDeploy with the dump code enabled.")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")