import triton
import triton.language as tl
import paddle
from typing import Optional, Tuple


@triton.jit
def _prefill_qk_kernel_mixed(
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k,
    cu_seqlens_pooled,
    seq_lens_q,
    is_prefill,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_ob, stride_oh, stride_on,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SHARED_TOKENS: tl.constexpr,
    K_SHARED_TOKENS: tl.constexpr,
    NUM_POOLS: tl.constexpr,
    NUM_HEAD_TOKENS: tl.constexpr,
):
    """
    支持 prefill 和 decode 混合的 batch
    """
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    
    # 检查当前 batch 是否是 prefill
    is_prefill_batch = tl.load(is_prefill + off_b)
    
    if is_prefill_batch == 0:
        return
    
    num_tiles_n = tl.cdiv(max_seqlen_k, BLOCK_N)
    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n
    
    # Load batch boundaries
    if off_b == 0:
        start_q = 0
        start_k = 0
        start_pooled = 0
    else:
        start_q = tl.load(cu_seqlens_q + off_b)
        start_k = tl.load(cu_seqlens_k + off_b)
        start_pooled = tl.load(cu_seqlens_pooled + off_b)
    
    end_q = tl.load(cu_seqlens_q + off_b + 1)
    end_k = tl.load(cu_seqlens_k + off_b + 1)
    end_pooled = tl.load(cu_seqlens_pooled + off_b + 1)
    
    seqlen_q = end_q - start_q
    seqlen_k = end_k - start_k
    seqlen_pooled = end_pooled - start_pooled
    
    if seqlen_q <= 0 or seqlen_k <= 0:
        return
    
    m_start = tile_m * BLOCK_M
    n_start = tile_n * BLOCK_N
    
    if m_start >= seqlen_q or n_start >= seqlen_k:
        return
    
    # ==================== Offsets ====================
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # ==================== Load Q ====================
    q_ptrs = (
        Q
        + (start_q + offs_m)[:, None] * stride_qb
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    
    mask_q = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    # ==================== Load K ====================
    k_ptrs = (
        K
        + (start_k + offs_n)[:, None] * stride_kb
        + off_h * stride_kh
        + offs_d[None, :] * stride_kd
    )
    
    mask_k = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=mask_k, other=0.0)
    k = tl.trans(k)
    
    # ==================== Compute QK^T ====================
    qk = tl.dot(q, k, allow_tf32=True)
    
    # ==================== Causal mask（考虑 K 的 pooling）====================
    # Q 的全局位置
    q_pos = start_q + offs_m
    
    # K pooled token 的原始结束位置
    # K pooled token i 代表原始 [i*K_SHARED_TOKENS, (i+1)*K_SHARED_TOKENS)
    k_pool_end_pos = (start_k + offs_n + 1) * K_SHARED_TOKENS - 1
    
    # Causal mask
    causal_mask = q_pos[:, None] >= k_pool_end_pos[None, :]
    
    # Boundary mask
    boundary_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    
    # Final mask
    final_mask = causal_mask & boundary_mask
    
    # ==================== Q-dimension pooling ====================
    if Q_SHARED_TOKENS > 1:
        qk = tl.where(final_mask, qk, float('-1e5'))
        
        # Reshape and pool
        qk_reshaped = tl.reshape(qk, (NUM_POOLS, Q_SHARED_TOKENS, BLOCK_N))
        qk_pooled = tl.max(qk_reshaped, axis=1)  # [NUM_POOLS, BLOCK_N]
        
        # ==================== 应用 trick（向量化版本）====================
        # 计算每个 pooled Q token 的位置信息
        pool_offsets = tl.arange(0, NUM_POOLS) * Q_SHARED_TOKENS
        pool_start_in_batch = m_start + pool_offsets
        
        # 每个 pool 的最后一个 Q token 的全局位置
        pool_end_in_batch = tl.minimum(pool_start_in_batch + Q_SHARED_TOKENS, seqlen_q)
        last_q_pos = start_q + pool_end_in_batch - 1  # [NUM_POOLS]
        
        # 每个 pool 可见的最后一个 K pooled token index
        max_visible_k_pool = (last_q_pos + 1) // K_SHARED_TOKENS - start_k  # [NUM_POOLS]
        
        # K pooled token 的全局索引
        k_pool_indices = n_start + offs_n  # [BLOCK_N]
        
        # Trick 1: 前 NUM_HEAD_TOKENS 个 K positions
        head_mask = k_pool_indices[None, :] < NUM_HEAD_TOKENS  # [1, BLOCK_N]
        
        # Trick 2: 最后一个可见的 K position
        # 对每个 pool，检查哪些 K positions 是最后一个可见的
        last_visible_mask = k_pool_indices[None, :] == max_visible_k_pool[:, None]  # [NUM_POOLS, BLOCK_N]
        
        # 组合两个 mask
        trick_mask = head_mask | last_visible_mask  # [NUM_POOLS, BLOCK_N]
        
        # 应用 trick：将 mask 为 True 的位置设为 1e5
        qk_pooled = tl.where(trick_mask, 1e5, qk_pooled)
        
        # ==================== 计算输出位置 ====================
        pool_idx_in_batch = pool_start_in_batch // Q_SHARED_TOKENS
        global_pool_indices = start_pooled + pool_idx_in_batch
        
        out_mask = (
            (pool_idx_in_batch[:, None] < seqlen_pooled) &
            (offs_n[None, :] < seqlen_k)
        )
        
        out_ptrs = (
            Out
            + global_pool_indices[:, None] * stride_ob
            + off_h * stride_oh
            + offs_n[None, :]
        )
        
        tl.store(out_ptrs, qk_pooled, mask=out_mask)
    else:
        # 不做 pooling
        qk = tl.where(final_mask, qk, 0.0)
        
        out_ptrs = (
            Out
            + (start_pooled + offs_m)[:, None] * stride_ob
            + off_h * stride_oh
            + offs_n[None, :]
        )
        
        tl.store(out_ptrs, qk, mask=final_mask)


def prefill_qk_varlen(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_q: paddle.Tensor,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    block_m: int = 64,
    block_n: int = 64,
    q_shared_tokens: int = 1,
    k_shared_tokens: int = 8,
    num_head_tokens: int = 64,
) -> paddle.Tensor:
    """
    支持 prefill 和 decode 混合的 Q @ K^T 计算
    """
    # ==================== Input validation ====================
    assert q.ndim == 3 and k.ndim == 3
    total_q, num_heads, head_dim = q.shape
    
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    
    batch_size = seq_lens_q.shape[0]
    
    if max_seqlen_q is None:
        max_seqlen_q = int(seq_lens_q.max().item())
    
    if max_seqlen_k is None:
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
        max_seqlen_k = int(seqlens_k.max().item())
    
    assert block_m > 0 and (block_m & (block_m - 1)) == 0
    assert block_n > 0 and (block_n & (block_n - 1)) == 0
    
    if q_shared_tokens > 1:
        assert q_shared_tokens > 0 and (q_shared_tokens & (q_shared_tokens - 1)) == 0
        assert block_m % q_shared_tokens == 0
        num_pools = block_m // q_shared_tokens
    else:
        num_pools = 1
    
    # ==================== 区分 prefill 和 decode ====================
    seq_lens_q_1d = seq_lens_q.reshape([-1])
    is_prefill = (seq_lens_q_1d > 1).cast('int32')
    
    # 计算 pooled Q 的长度
    if q_shared_tokens > 1:
        pooled_seqlens = paddle.where(
            seq_lens_q_1d > 1,
            (seq_lens_q_1d + q_shared_tokens - 1) // q_shared_tokens,
            paddle.ones_like(seq_lens_q_1d)
        ).cast('int32')
    else:
        pooled_seqlens = seq_lens_q_1d.cast('int32')
    
    # 计算累积长度
    cumsum_pooled = paddle.cumsum(pooled_seqlens, axis=0)
    cu_seqlens_pooled = paddle.concat([
        paddle.zeros([1], dtype='int32'),
        cumsum_pooled.cast('int32')
    ])
    
    total_pooled = int(cu_seqlens_pooled[-1].item())
    
    # ==================== Allocate output ====================
    out = paddle.zeros(
        [total_pooled, num_heads, max_seqlen_k],
        dtype=q.dtype
    )
    
    # ==================== Launch kernel ====================
    num_tiles_m = triton.cdiv(max_seqlen_q, block_m)
    num_tiles_n = triton.cdiv(max_seqlen_k, block_n)
    grid = (num_tiles_m * num_tiles_n, num_heads, batch_size)
    num_warps = 4 if block_m <= 64 else 8
    
    _prefill_qk_kernel_mixed[grid](
        Q=q, K=k, Out=out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_pooled=cu_seqlens_pooled,
        seq_lens_q=seq_lens_q_1d,
        is_prefill=is_prefill,
        stride_qb=q.strides[0],
        stride_qh=q.strides[1],
        stride_qd=q.strides[2],
        stride_kb=k.strides[0],
        stride_kh=k.strides[1],
        stride_kd=k.strides[2],
        stride_ob=out.strides[0],
        stride_oh=out.strides[1],
        stride_on=out.strides[2],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        Q_SHARED_TOKENS=q_shared_tokens,
        K_SHARED_TOKENS=k_shared_tokens,
        NUM_POOLS=num_pools,
        NUM_HEAD_TOKENS=num_head_tokens,
        num_warps=num_warps,
        num_stages=2,
    )
    breakpoint()
    return out


# ==================== Reference 实现 ====================
def reference_qk_pooled(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    q_shared_tokens: int = 8,
    k_shared_tokens: int = 8,
    num_head_tokens: int = 64,
) -> paddle.Tensor:
    """Reference 实现"""
    batch_size = seq_lens_q.shape[0]
    num_heads = q.shape[1]
    
    # 计算输出大小
    pooled_seqlens = []
    for i in range(batch_size):
        seqlen = seq_lens_q[i].item()
        if seqlen > 1:
            pooled_len = (seqlen + q_shared_tokens - 1) // q_shared_tokens
        else:
            pooled_len = 1
        pooled_seqlens.append(pooled_len)
    
    total_pooled = sum(pooled_seqlens)
    max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
    
    out_ref = paddle.zeros([total_pooled, num_heads, max_seqlen_k], dtype=q.dtype)
    
    pooled_offset = 0
    
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()
        
        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        
        if seqlen_q <= 1:
            pooled_offset += 1
            continue
        
        q_batch = q[start_q:end_q]
        k_batch = k[start_k:end_k]
        
        # Compute QK^T
        qk = paddle.matmul(
            q_batch.transpose([1, 0, 2]),
            k_batch.transpose([1, 2, 0])
        ).transpose([1, 0, 2])
        
        # Apply causal mask
        for i in range(seqlen_q):
            q_pos = start_q + i
            for j in range(seqlen_k):
                k_end_pos = (start_k + j + 1) * k_shared_tokens - 1
                if q_pos < k_end_pos:
                    qk[i, :, j] = float('-1e5')
        
        # Q dimension max pooling
        num_pools = (seqlen_q + q_shared_tokens - 1) // q_shared_tokens
        qk_pooled = paddle.zeros([num_pools, num_heads, seqlen_k], dtype=q.dtype)
        
        for pool_idx in range(num_pools):
            pool_start = pool_idx * q_shared_tokens
            pool_end = min(pool_start + q_shared_tokens, seqlen_q)
            
            qk_pool = qk[pool_start:pool_end, :, :]
            qk_pooled[pool_idx] = paddle.max(qk_pool, axis=0)
            
            # Apply trick
            last_q_pos = start_q + pool_end - 1
            max_visible_k_pool = (last_q_pos + 1) // k_shared_tokens - start_k
            
            for j in range(seqlen_k):
                if j < num_head_tokens:
                    qk_pooled[pool_idx, :, j] = 1e5
                if j == max_visible_k_pool:
                    qk_pooled[pool_idx, :, j] = 1e5
        
        out_ref[pooled_offset:pooled_offset + num_pools, :, :seqlen_k] = qk_pooled
        pooled_offset += num_pools
    
    return out_ref


# ==================== 测试代码 ====================
def test_with_reference():
    """测试并与 reference 对比"""
    paddle.set_device('gpu:0')
    
    num_heads = 4
    head_dim = 128
    q_shared_tokens = 8
    k_shared_tokens = 8
    num_head_tokens = 64
    
    # Mixed batch
    seqlens_q = [121, 1, 64]
    batch_size = len(seqlens_q)
    
    seqlens_k_original = [240, 160, 128]
    seqlens_k = [(s + k_shared_tokens - 1) // k_shared_tokens for s in seqlens_k_original]
    
    cu_seqlens_q = paddle.to_tensor(
        [0] + [sum(seqlens_q[:i+1]) for i in range(len(seqlens_q))],
        dtype='int32'
    )
    cu_seqlens_k = paddle.to_tensor(
        [0] + [sum(seqlens_k[:i+1]) for i in range(len(seqlens_k))],
        dtype='int32'
    )
    seq_lens_q_tensor = paddle.to_tensor(seqlens_q, dtype='int32')
    
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    
    print("="*60)
    print("Test Configuration:")
    for i in range(batch_size):
        print(f"Batch {i}: Q={seqlens_q[i]:3d} tokens, "
              f"K={seqlens_k[i]:3d} pooled tokens (original: {seqlens_k_original[i]})")
    print(f"Total Q: {total_q}, Total pooled K: {total_k}")
    print("="*60)
    
    # 创建测试数据
    q = paddle.randn([total_q, num_heads, head_dim], dtype='bfloat16')
    k = paddle.randn([total_k, num_heads, head_dim], dtype='bfloat16')
    
    # Triton kernel
    print("\nRunning Triton kernel...")
    out_triton = prefill_qk_varlen(
        q, k,
        cu_seqlens_q,
        seq_lens_q_tensor,
        cu_seqlens_k,
        q_shared_tokens=q_shared_tokens,
        k_shared_tokens=k_shared_tokens,
        num_head_tokens=num_head_tokens,
    )
    
    # Reference
    print("Running reference implementation...")
    out_ref = reference_qk_pooled(
        q, k,
        cu_seqlens_q,
        seq_lens_q_tensor,
        cu_seqlens_k,
        q_shared_tokens=q_shared_tokens,
        k_shared_tokens=k_shared_tokens,
        num_head_tokens=num_head_tokens,
    )
    
    print(f"\nTriton output shape: {out_triton.shape}")
    print(f"Reference output shape: {out_ref.shape}")
    
    # 对比
    print("\n" + "="*60)
    print("Comparing results...")
    
    pooled_offset = 0
    all_passed = True
    
    for b in range(batch_size):
        seqlen_q = seqlens_q[b]
        seqlen_k = seqlens_k[b]
        
        if seqlen_q > 1:
            num_pools = (seqlen_q + q_shared_tokens - 1) // q_shared_tokens
            
            triton_batch = out_triton[pooled_offset:pooled_offset + num_pools, :, :seqlen_k]
            ref_batch = out_ref[pooled_offset:pooled_offset + num_pools, :, :seqlen_k]
            
            diff = paddle.abs(triton_batch - ref_batch).max().item()
            mean_diff = paddle.abs(triton_batch - ref_batch).mean().item()
            
            print(f"Batch {b} (prefill, {num_pools} pooled tokens):")
            print(f"  Max diff: {diff:.6f}, Mean diff: {mean_diff:.6f}")
            
            if diff > 1e-3:
                print(f"  ✗ FAILED!")
                print(f"  Triton sample: {triton_batch[0, 0, :10]}")
                print(f"  Ref sample:    {ref_batch[0, 0, :10]}")
                all_passed = False
            else:
                print(f"  ✓ Passed")
            
            pooled_offset += num_pools
        else:
            print(f"Batch {b} (decode, skipped)")
            pooled_offset += 1
    
    print("="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")


if __name__ == "__main__":
    test_with_reference()