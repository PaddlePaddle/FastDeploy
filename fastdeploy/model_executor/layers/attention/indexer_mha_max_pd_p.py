import triton
import triton.language as tl
import paddle
from typing import Optional, Tuple


@triton.jit
def _prefill_qk_kernel_mixed(
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k,
    cu_seqlens_pooled,
    seq_lens_q,  # ← 添加：每个 batch 的实际 Q 序列长度 [batch_size]
    is_prefill,  # ← 添加：标记每个 batch 是否是 prefill [batch_size]
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_ob, stride_oh, stride_on,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SHARED_TOKENS: tl.constexpr,
    NUM_POOLS: tl.constexpr,
):
    """
    支持 prefill 和 decode 混合的 batch
    - Prefill: 做 max pooling 压缩
    - Decode: 直接跳过（或者保持原样输出）
    """
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    
    # ==================== 检查当前 batch 是否是 prefill ====================
    is_prefill_batch = tl.load(is_prefill + off_b)
    
    # 如果是 decode，直接返回（跳过计算）
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
    
    # ==================== Causal mask ====================
    causal_mask = (offs_m[:, None] + start_q) >= (offs_n[None, :] + start_k)
    boundary_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    final_mask = causal_mask & boundary_mask
    
    # ==================== Prefill: Q-dimension pooling ====================
    if Q_SHARED_TOKENS > 1:
        qk = tl.where(final_mask, qk, float('-1e5'))
        
        # Reshape and pool
        qk_reshaped = tl.reshape(qk, (NUM_POOLS, Q_SHARED_TOKENS, BLOCK_N))
        qk_pooled = tl.max(qk_reshaped, axis=1)
        
        # 使用 batch 内的相对位置
        pool_offsets = tl.arange(0, NUM_POOLS) * Q_SHARED_TOKENS
        pool_start_in_batch = m_start + pool_offsets
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
        qk = tl.where(final_mask, qk, 0.0)
        
        out_ptrs = (
            Out
            + (start_q + offs_m)[:, None] * stride_ob
            + off_h * stride_oh
            + offs_n[None, :]
        )
        
        tl.store(out_ptrs, qk, mask=final_mask)


def prefill_qk_varlen(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_q: paddle.Tensor,  # ← 添加：实际的 Q 序列长度
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    block_m: int = 64,
    block_n: int = 64,
    q_shared_tokens: int = 1,
) -> paddle.Tensor:
    """
    支持 prefill 和 decode 混合的 Q @ K^T 计算
    
    Args:
        q: Query tensor [total_tokens, num_heads, head_dim]
        k: Key tensor [total_tokens, num_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for Q [batch_size + 1]
        seq_lens_q: Actual Q sequence lengths [batch_size]
                   - prefill: seqlen > 1 (会被 pooling)
                   - decode: seqlen = 1 (跳过)
        cu_seqlens_k: Cumulative sequence lengths for K [batch_size + 1]
        max_seqlen_q: Maximum sequence length in Q
        max_seqlen_k: Maximum sequence length in K
        block_m: Tile size for Q dimension
        block_n: Tile size for K dimension
        q_shared_tokens: Number of Q tokens to pool together
    
    Returns:
        Output tensor with pooled prefill tokens
        
    Example:
        >>> # Mixed batch:
        >>> # Batch 0: prefill, 121 tokens -> 16 pooled tokens
        >>> # Batch 1: decode, 1 token -> skipped (0 pooled tokens)
        >>> # Batch 2: prefill, 121 tokens -> 16 pooled tokens
        >>> # Total output: 32 pooled tokens
    """
    # ==================== Input validation ====================
    assert q.ndim == 3 and k.ndim == 3
    total_q, num_heads, head_dim = q.shape
    
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    
    batch_size = cu_seqlens_q.shape[0] - 1
    assert seq_lens_q.shape[0] == batch_size, "seq_lens_q must match batch_size"
    
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
    seq_lens_q_cpu = seq_lens_q.cpu().numpy()
    is_prefill_flags = []
    pooled_seqlens = []
    
    for i in range(batch_size):
        seqlen = seq_lens_q_cpu[i]
        
        if seqlen > 1:  # Prefill
            is_prefill_flags.append(1)
            if q_shared_tokens > 1:
                pooled_len = (seqlen + q_shared_tokens - 1) // q_shared_tokens
            else:
                pooled_len = seqlen
            pooled_seqlens.append(pooled_len)
        else:  # Decode (seqlen = 1)
            is_prefill_flags.append(0)
            pooled_seqlens.append(0)  # Decode 不输出
    
    # 转换为 tensor
    is_prefill = paddle.to_tensor(is_prefill_flags, dtype='int32')
    
    # 计算 pooled 累积长度
    cu_seqlens_pooled = paddle.to_tensor(
        [0] + [sum(pooled_seqlens[:i+1]) for i in range(len(pooled_seqlens))],
        dtype='int32'
    )
    
    total_pooled = cu_seqlens_pooled[-1].item()
    
    print(f"Original seqlens: {seq_lens_q_cpu}")
    print(f"Is prefill: {is_prefill_flags}")
    print(f"Pooled seqlens: {pooled_seqlens}")
    print(f"Total pooled tokens: {total_pooled}")
    print(f"cu_seqlens_pooled: {cu_seqlens_pooled}")
    
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
        seq_lens_q=seq_lens_q,
        is_prefill=is_prefill,  # ← 传入 prefill 标记
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
        NUM_POOLS=num_pools,
        num_warps=num_warps,
        num_stages=2,
    )
    
    return out


# ==================== 测试代码 ====================
def test_mixed_batch():
    """测试混合 prefill 和 decode 的 batch"""
    paddle.set_device('gpu:0')
    
    num_heads = 4
    head_dim = 128
    q_shared_tokens = 8
    
    # Mixed batch:
    # Batch 0: prefill, 121 tokens
    # Batch 1: decode, 1 token
    # Batch 2: prefill, 121 tokens
    # Batch 3: decode, 1 token
    # Batch 4: prefill, 64 tokens
    
    seqlens_q = [121, 1, 121, 1, 64]  # 实际的 Q 序列长度
    batch_size = len(seqlens_q)
    
    # 为了简化，K 的长度假设与历史一致
    seqlens_k = [200, 150, 300, 100, 180]
    
    cu_seqlens_q = paddle.to_tensor(
        [0] + [sum(seqlens_q[:i+1]) for i in range(len(seqlens_q))],
        dtype='int32'
    )
    cu_seqlens_k = paddle.to_tensor(
        [0] + [sum(seqlens_k[:i+1]) for i in range(len(seqlens_k))],
        dtype='int32'
    )
    seq_lens_q_tensor = paddle.to_tensor(seqlens_q, dtype='int32')
    
    total_tokens_q = cu_seqlens_q[-1].item()
    total_tokens_k = cu_seqlens_k[-1].item()
    
    print("="*60)
    print("Mixed Batch Configuration:")
    print(f"Batch 0: prefill, {seqlens_q[0]} Q tokens, {seqlens_k[0]} K tokens")
    print(f"Batch 1: decode,  {seqlens_q[1]} Q tokens, {seqlens_k[1]} K tokens")
    print(f"Batch 2: prefill, {seqlens_q[2]} Q tokens, {seqlens_k[2]} K tokens")
    print(f"Batch 3: decode,  {seqlens_q[3]} Q tokens, {seqlens_k[3]} K tokens")
    print(f"Batch 4: prefill, {seqlens_q[4]} Q tokens, {seqlens_k[4]} K tokens")
    print(f"Total Q tokens: {total_tokens_q}")
    print(f"Total K tokens: {total_tokens_k}")
    print("="*60)
    
    # 创建测试数据
    q = paddle.randn([total_tokens_q, num_heads, head_dim], dtype='float16')
    k = paddle.randn([total_tokens_k, num_heads, head_dim], dtype='float16')
    
    # 运行 kernel
    print("\nRunning mixed batch kernel...")
    out_pooled = prefill_qk_varlen(
        q, k, 
        cu_seqlens_q, 
        seq_lens_q_tensor,  # ← 传入实际序列长度
        cu_seqlens_k,
        q_shared_tokens=q_shared_tokens
    )
    
    print(f"\nOutput shape: {out_pooled.shape}")
    
    # 计算期望的输出大小
    expected_pooled = 0
    for i, seqlen in enumerate(seqlens_q):
        if seqlen > 1:  # Prefill
            pooled = (seqlen + q_shared_tokens - 1) // q_shared_tokens
            expected_pooled += pooled
            print(f"Batch {i}: {seqlen} tokens -> {pooled} pooled tokens")
        else:  # Decode
            print(f"Batch {i}: {seqlen} token  -> 0 pooled tokens (skipped)")
    
    print(f"\nExpected total pooled tokens: {expected_pooled}")
    assert out_pooled.shape[0] == expected_pooled, \
        f"Output size mismatch! Got {out_pooled.shape[0]}, expected {expected_pooled}"
    
    print("\n" + "="*60)
    print("✓ Mixed batch test passed!")


def test_pure_prefill():
    """测试纯 prefill batch（与之前保持兼容）"""
    paddle.set_device('gpu:0')
    
    batch_size = 3
    num_heads = 4
    head_dim = 128
    seqlens = [121, 121, 121]
    q_shared_tokens = 8
    
    cu_seqlens = paddle.to_tensor(
        [0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
        dtype='int32'
    )
    seq_lens_q = paddle.to_tensor(seqlens, dtype='int32')
    
    total_tokens = cu_seqlens[-1].item()
    
    q = paddle.randn([total_tokens, num_heads, head_dim], dtype='float16')
    k = paddle.randn([total_tokens, num_heads, head_dim], dtype='float16')
    
    print("="*60)
    print("Pure Prefill Batch Test")
    print("="*60)
    
    out_pooled = prefill_qk_varlen(
        q, k, cu_seqlens, seq_lens_q, q_shared_tokens=q_shared_tokens
    )
    
    expected_pooled_per_batch = (121 + 8 - 1) // 8  # = 16
    expected_total = expected_pooled_per_batch * batch_size  # = 48
    
    print(f"\nOutput shape: {out_pooled.shape}")
    print(f"Expected: ({expected_total}, {num_heads}, {max(seqlens)})")
    
    assert out_pooled.shape[0] == expected_total
    print("\n✓ Pure prefill test passed!")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Test 1: Mixed Prefill/Decode Batch")
    print("#"*60 + "\n")
    test_mixed_batch()
    
    print("\n\n" + "#"*60)
    print("# Test 2: Pure Prefill Batch (Backward Compatibility)")
    print("#"*60 + "\n")
    test_pure_prefill()