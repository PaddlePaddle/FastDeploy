import triton
import triton.language as tl
import paddle
from typing import Optional, Tuple


@triton.jit
def _prefill_qk_kernel_pooled(
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k,
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
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    num_tiles_n = tl.cdiv(max_seqlen_k, BLOCK_N)
    tile_m = pid // num_tiles_n       # pooled-Q tile
    tile_n = pid % num_tiles_n

    # ---------------- batch seqlens ----------------
    start_q = tl.load(cu_seqlens_q + off_b)
    start_k = tl.load(cu_seqlens_k + off_b)
    end_q   = tl.load(cu_seqlens_q + off_b + 1)
    end_k   = tl.load(cu_seqlens_k + off_b + 1)

    seqlen_q = end_q - start_q
    seqlen_k = end_k - start_k
    if seqlen_q <= 0 or seqlen_k <= 0:
        return

    pooled_seqlen_q = (seqlen_q + Q_SHARED_TOKENS - 1) // Q_SHARED_TOKENS

    # ---------------- pooled Q tile ----------------
    pooled_m_start = tile_m * NUM_POOLS
    if pooled_m_start >= pooled_seqlen_q:
        return

    m_start = pooled_m_start * Q_SHARED_TOKENS
    n_start = tile_n * BLOCK_N
    if n_start >= seqlen_k:
        return

    # ---------------- offsets ----------------
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # ---------------- load Q ----------------
    q_ptrs = (
        Q
        + (start_q + offs_m)[:, None] * stride_qb
        + off_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---------------- load K ----------------
    k_ptrs = (
        K
        + (start_k + offs_n)[:, None] * stride_kb
        + off_h * stride_kh
        + offs_d[None, :] * stride_kd
    )
    k_mask = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    k = tl.trans(k)

    # ---------------- QK ----------------
    qk = tl.dot(q, k, allow_tf32=True)

    # ---------------- causal + boundary mask ----------------
    causal = (offs_m[:, None] + start_q) >= (offs_n[None, :] + start_k)
    valid  = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    final_mask = causal & valid
    qk = tl.where(final_mask, qk, float("-1e5"))

    # ---------------- Q pooling ----------------
    qk = tl.reshape(qk, (NUM_POOLS, Q_SHARED_TOKENS, BLOCK_N))
    qk_pooled = tl.max(qk, axis=1)

    pool_offsets = tl.arange(0, NUM_POOLS)
    global_pool_idx = pooled_m_start + pool_offsets

    out_mask = (
        (global_pool_idx[:, None] < pooled_seqlen_q) &
        (offs_n[None, :] < seqlen_k)
    )

    out_ptrs = (
        Out
        + global_pool_idx[:, None] * stride_ob
        + off_h * stride_oh
        + offs_n[None, :]
    )
    tl.store(out_ptrs, qk_pooled, mask=out_mask)


def prefill_qk_varlen(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    block_m: int = 64,
    block_n: int = 64,
    q_shared_tokens: int = 1,
):
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    total_q, num_heads, head_dim = q.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())

    assert block_m % q_shared_tokens == 0
    num_pools = block_m // q_shared_tokens

    pooled_total_q = (total_q + q_shared_tokens - 1) // q_shared_tokens
    out = paddle.zeros(
        [pooled_total_q, num_heads, max_seqlen_k],
        dtype=q.dtype,
    )

    pooled_max_seqlen_q = (max_seqlen_q + q_shared_tokens - 1) // q_shared_tokens
    num_tiles_m = triton.cdiv(pooled_max_seqlen_q, num_pools)
    num_tiles_n = triton.cdiv(max_seqlen_k, block_n)

    grid = (num_tiles_m * num_tiles_n, num_heads, batch_size)

    num_warps = 4 if block_m <= 64 else 8

    _prefill_qk_kernel_pooled[grid](
        Q=q,
        K=k,
        Out=out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
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






# q = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/unittest_tile_sparse/test_packqk/q")
# k = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/unittest_tile_sparse/test_packqk/k")
# cu_seqlen_q = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/unittest_tile_sparse/test_packqk/cu_seqlens_q")

# # (Pdb) p q.shape
# # paddle.Size([17189, 4, 128])
# # (Pdb) p k.shape
# # paddle.Size([4, 128, 2149])
# # Tensor(shape=[2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
# #        [0    , 17189])

# # for i in range(0):
# score = paddle.matmul(q.transpose([1,0,2]).contiguous(),k)

# # breakpoint()
# out = prefill_qk_varlen(
#     q, 
#     k.transpose([2,0,1]).contiguous(),
#     cu_seqlen_q,
#     (cu_seqlen_q+7)//8,
# ).transpose([1,0,2])

# print(score-out)
# # breakpoint()

# score_qzip = paddle.nn.functional.max_pool1d(
#     score.transpose([0, 2, 1]).contiguous(), 
#     kernel_size=8, 
#     stride=8, 
#     ceil_mode=True
# ).transpose([2, 0, 1]).reshape([-1,score.shape[-1]])

# breakpoint()
# out1 = prefill_qk_varlen(
#     q, 
#     k.transpose([2,0,1]).contiguous(),
#     cu_seqlen_q,
#     (cu_seqlen_q+7)//8,
#     q_shared_tokens=8,
# ).reshape(-1,k.shape[-1])

# print(score_qzip-out1)
# breakpoint()
# # print(score-out)
# # breakpoint()







# # ==================== 测试代码 ====================
# def test_pooling():
#     """测试 Q-dimension max pooling"""
#     paddle.set_device('gpu:0')
    
#     batch_size = 2
#     num_heads = 4
#     head_dim = 64
#     seqlens = [128, 256]
#     q_shared_tokens = 8
    
#     cu_seqlens = paddle.to_tensor(
#         [0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
#         dtype='int32'
#     )
#     total_tokens = cu_seqlens[-1].item()
    
#     print(f"Total tokens: {total_tokens}")
#     print(f"cu_seqlens: {cu_seqlens}")
#     print(f"q_shared_tokens: {q_shared_tokens}")
    
#     # 创建测试数据
#     q = paddle.randn([total_tokens, num_heads, head_dim], dtype='float16')
#     k = paddle.randn([total_tokens, num_heads, head_dim], dtype='float16')
    
#     # 不带 pooling
#     print("\n" + "="*60)
#     print("Testing without pooling...")
#     out_no_pool = prefill_qk_varlen(q, k, cu_seqlens, q_shared_tokens=0)
#     print(f"Output shape (no pooling): {out_no_pool.shape}")
#     expected_shape = (total_tokens, num_heads, max(seqlens))
#     assert tuple(out_no_pool.shape) == expected_shape
#     print("✓ No pooling test passed!")
    
#     # 带 pooling
#     print("\n" + "="*60)
#     print("Testing with pooling...")
#     out_pooled = prefill_qk_varlen(q, k, cu_seqlens, q_shared_tokens=q_shared_tokens)
#     print(f"Output shape (pooled): {out_pooled.shape}")
#     expected_pooled = ((total_tokens + q_shared_tokens - 1) // q_shared_tokens, num_heads, max(seqlens))
#     assert tuple(out_pooled.shape) == expected_pooled
#     print(f"✓ Shape matches expected: {expected_pooled}")
    
#     # 验证 pooling 正确性
#     print("\n" + "="*60)
#     print("Verifying pooling correctness...")
    
#     for b in range(batch_size):
#         start = cu_seqlens[b].item()
#         end = cu_seqlens[b + 1].item()
#         seqlen = seqlens[b]
        
#         num_pools_in_batch = (seqlen + q_shared_tokens - 1) // q_shared_tokens
        
#         print(f"\nBatch {b}: tokens [{start}, {end}), seqlen={seqlen}, pools={num_pools_in_batch}")
        
#         for pool_idx in range(min(3, num_pools_in_batch)):  # 只检查前 3 个 pool
#             # 计算这个 pool 包含的 token 范围
#             pool_start = start + pool_idx * q_shared_tokens
#             pool_end = min(pool_start + q_shared_tokens, end)
            
#             # Reference: 从 no_pool 版本手动计算 max
#             ref_max = paddle.max(
#                 out_no_pool[pool_start:pool_end, :, :seqlen],
#                 axis=0
#             )
            
#             # Triton pooled output
#             pooled_global_idx = pool_start // q_shared_tokens
#             triton_out = out_pooled[pooled_global_idx, :, :seqlen]
            
#             diff = paddle.abs(ref_max - triton_out).max().item()
            
#             print(f"  Pool {pool_idx} (tokens [{pool_start}, {pool_end})): diff={diff:.6f}", end="")
            
#             if diff > 1e-2:
#                 print(" ✗ FAILED")
#                 print(f"    Ref max sample: {ref_max[0, :5]}")
#                 print(f"    Triton sample:  {triton_out[0, :5]}")
#             else:
#                 print(" ✓")
    
#     print("\n" + "="*60)
#     print("✓ All pooling tests passed!")


# if __name__ == "__main__":
#     test_pooling()