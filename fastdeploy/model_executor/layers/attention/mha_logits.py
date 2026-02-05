import triton
import triton.language as tl
import paddle
from typing import Optional, Tuple


@triton.jit
def _prefill_qk_kernel(
    # Input/Output pointers
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k,
    # Strides for Q [total_tokens, num_heads, head_dim]
    stride_qb, stride_qh, stride_qd,
    # Strides for K [total_tokens, num_heads, head_dim]
    stride_kb, stride_kh, stride_kd,
    # Strides for Out [total_tokens, num_heads, max_seqlen_k]
    stride_ob, stride_oh, stride_on,
    # Dimensions
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute Q @ K^T with causal masking for prefill phase.
    
    Grid: (num_tiles_m * num_tiles_n, num_heads, batch_size)
    
    Args:
        Q: Query tensor [total_tokens, num_heads, head_dim]
        K: Key tensor [total_tokens, num_heads, head_dim]
        Out: Output tensor [total_tokens, num_heads, max_seqlen_k]
        cu_seqlens_q: Cumulative sequence lengths for Q [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for K [batch_size + 1]
        stride_*: Strides for memory access
        max_seqlen_q: Maximum sequence length in Q
        max_seqlen_k: Maximum sequence length in K
        HEAD_DIM: Dimension of each attention head (must be power of 2)
        BLOCK_M: Tile size for Q dimension (must be power of 2)
        BLOCK_N: Tile size for K dimension (must be power of 2)
    """
    # Program IDs
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    
    # Compute tile indices
    num_tiles_n = tl.cdiv(max_seqlen_k, BLOCK_N)
    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n
    
    # Load batch boundaries
    if off_b == 0:
        start_q = 0
        start_k = 0
    else:
        start_q = tl.load(cu_seqlens_q + off_b)
        start_k = tl.load(cu_seqlens_k + off_b)
    
    end_q = tl.load(cu_seqlens_q + off_b + 1)
    end_k = tl.load(cu_seqlens_k + off_b + 1)
    
    seqlen_q = end_q - start_q
    seqlen_k = end_k - start_k
    
    # Early exit for empty batches
    if seqlen_q <= 0 or seqlen_k <= 0:
        return
    
    # Compute tile starting positions (relative to batch)
    m_start = tile_m * BLOCK_M
    n_start = tile_n * BLOCK_N
    
    # Early exit for out-of-bounds tiles
    if m_start >= seqlen_q or n_start >= seqlen_k:
        return
    
    # ==================== Compute offsets ====================
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # ==================== Load Q block ====================
    # q_ptrs = Q + start_q * stride_qb + off_h * stride_qh
    # q_ptrs = q_ptrs + offs_m[:, None] * stride_qb + offs_d[None, :]
    q_ptrs = (
        Q
        + (start_q + offs_m)[:, None] * stride_qb
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )


    
    mask_q = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    # ==================== Load K block ====================
    # k_ptrs = K + start_k * stride_kb + off_h * stride_kh
    # k_ptrs = k_ptrs + offs_n[:, None] * stride_kb + offs_d[None, :]
    k_ptrs = (
        K
        + (start_k + offs_n)[:, None] * stride_kb
        + off_h * stride_kh
        + offs_d[None, :] * stride_kd
    )

    
    mask_k = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=mask_k, other=0.0)
    
    # Transpose K for matrix multiplication
    k = tl.trans(k)
    
    # ==================== Compute QK^T ====================
    # q: [BLOCK_M, HEAD_DIM], k: [HEAD_DIM, BLOCK_N]
    # qk: [BLOCK_M, BLOCK_N]
    qk = tl.dot(q, k, allow_tf32=True)
    
    # Apply scaling factor
    # sm_scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    # qk = qk * sm_scale
    
    # ==================== Apply masks ====================
    # Causal mask: position i can only attend to positions <= i
    # causal_mask = offs_m[:, None] >= offs_n[None, :]
    causal_mask = (offs_m[:, None] + start_q) >= (offs_n[None, :] + start_k)

    
    # Boundary mask: ensure we don't access out-of-bounds positions
    boundary_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    
    # Combine masks
    final_mask = causal_mask & boundary_mask

    # # ==================== Q-dim max pooling ====================
    # # Preconditions:
    # #   Q_SHARED divides BLOCK_M
    # #   Q_SHARED <= BLOCK_M
    # Q_SHARED = tl.constexpr(8)
    # Q_GROUPS = BLOCK_M // Q_SHARED

    # # reshape: [Q_GROUPS, Q_SHARED, BLOCK_N]
    # qk_reshaped = tl.reshape(qk, (Q_GROUPS, Q_SHARED, BLOCK_N))

    # # max over Q_SHARED dimension
    # qk_max = tl.max(qk_reshaped, axis=1)

    # # broadcast back to [Q_GROUPS, Q_SHARED, BLOCK_N]
    # qk_pooled = tl.broadcast_to(
    #     qk_max[:, None, :],
    #     (Q_GROUPS, Q_SHARED, BLOCK_N)
    # )

    # # flatten back to [BLOCK_M, BLOCK_N]
    # qk = tl.reshape(qk_pooled, (BLOCK_M, BLOCK_N))
    # # ==================== Q-dim max pooling ====================


    # # Apply mask (set masked positions to 0)
    # qk = tl.where(final_mask, qk, 0.0)
    
    # ==================== Store output ====================
    out_ptrs = Out + start_q * stride_ob + off_h * stride_oh
    out_ptrs = out_ptrs + offs_m[:, None] * stride_ob + offs_n[None, :]
    
    tl.store(out_ptrs, qk, mask=final_mask)


def prefill_qk_varlen(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    block_m: int = 64,
    block_n: int = 64,
) -> paddle.Tensor:
    """
    Compute Q @ K^T with causal masking for variable-length sequences.
    
    This function is optimized for the prefill phase of transformer attention,
    where each batch may have different sequence lengths.
    
    Args:
        q: Query tensor of shape [total_tokens, num_heads, head_dim]
        k: Key tensor of shape [total_tokens, num_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for Q, shape [batch_size + 1]
                     Example: [0, 128, 640, 1664] for batch_size=3
        cu_seqlens_k: Cumulative sequence lengths for K, shape [batch_size + 1]
                     If None, uses cu_seqlens_q (default for self-attention)
        max_seqlen_q: Maximum sequence length in Q. If None, computed from cu_seqlens_q
        max_seqlen_k: Maximum sequence length in K. If None, computed from cu_seqlens_k
        block_m: Tile size for Q dimension (must be power of 2, default: 64)
        block_n: Tile size for K dimension (must be power of 2, default: 64)
    
    Returns:
        Output tensor of shape [total_tokens, num_heads, max_seqlen_k]
        
        Note: The output is padded to max_seqlen_k. For each batch, only the
        first seqlen_k positions contain valid data.
    
    Raises:
        AssertionError: If input tensors have invalid shapes or strides
        ValueError: If block sizes are not powers of 2
    
    Example:
        >>> batch_size = 2
        >>> seqlens = [128, 512]
        >>> cu_seqlens = paddle.tensor([0, 128, 640], dtype=paddle.int32, device='cuda')
        >>> total_tokens = 640
        >>> q = paddle.randn(total_tokens, 8, 128, dtype=paddle.float16, device='cuda')
        >>> k = paddle.randn(total_tokens, 8, 128, dtype=paddle.float16, device='cuda')
        >>> out = prefill_qk_varlen(q, k, cu_seqlens)
        >>> out.shape
        paddle.Size([640, 8, 512])
    """
    # ==================== Input validation ====================
    assert q.dim() == 3 and k.dim() == 3, \
        f"Q and K must be 3D tensors, got Q.dim()={q.dim()}, K.dim()={k.dim()}"
    
    total_q, num_heads, head_dim = q.shape
    total_k, num_heads_k, head_dim_k = k.shape
    
    assert num_heads == num_heads_k, \
        f"Number of heads must match: Q has {num_heads}, K has {num_heads_k}"
    assert head_dim == head_dim_k, \
        f"Head dimension must match: Q has {head_dim}, K has {head_dim_k}"
    assert head_dim in [64, 128, 256], \
        f"HEAD_DIM must be 64, 128, or 256 (power of 2), got {head_dim}"
    
    # Handle default arguments
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    
    batch_size = cu_seqlens_q.shape[0] - 1
    assert cu_seqlens_k.shape[0] == batch_size + 1, \
        "cu_seqlens_q and cu_seqlens_k must have same batch size"
    
    if max_seqlen_q is None:
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
        max_seqlen_q = int(seqlens_q.max().item())
    
    if max_seqlen_k is None:
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
        max_seqlen_k = int(seqlens_k.max().item())
    
    # Validate block sizes
    assert block_m > 0 and (block_m & (block_m - 1)) == 0, \
        f"block_m must be power of 2, got {block_m}"
    assert block_n > 0 and (block_n & (block_n - 1)) == 0, \
        f"block_n must be power of 2, got {block_n}"
    
    # ==================== Allocate output ====================
    out = paddle.zeros(
        (total_q, num_heads, max_seqlen_k),
        device=q.device,
        dtype=q.dtype
    )
    
    # ==================== Configure kernel ====================
    num_tiles_m = triton.cdiv(max_seqlen_q, block_m)
    num_tiles_n = triton.cdiv(max_seqlen_k, block_n)
    
    grid = (num_tiles_m * num_tiles_n, num_heads, batch_size)
    
    # Choose number of warps based on block size
    num_warps = 4 if block_m <= 64 else 8
    
    # ==================== Launch kernel ====================
    _prefill_qk_kernel[grid](
        Q=q, K=k, Out=out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_on=out.stride(2),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    
    return out