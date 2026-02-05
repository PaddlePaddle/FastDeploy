import triton
import triton.language as tl
import paddle
from typing import Optional

# =========================
# Decode kernel（不变）
# =========================
@triton.jit
def _decode_qk_kernel(
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_output,
    batch_indices,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_ob, stride_oh, stride_on,
    max_seqlen_k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b_local = tl.program_id(2)

    off_b = tl.load(batch_indices + off_b_local)

    start_q = tl.load(cu_seqlens_q + off_b)
    start_k = tl.load(cu_seqlens_k + off_b)
    start_o = tl.load(cu_seqlens_output + off_b)

    end_k = tl.load(cu_seqlens_k + off_b + 1)
    seqlen_k = end_k - start_k
    if seqlen_k <= 0:
        return

    n_start = tile_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + start_q * stride_qb + off_h * stride_qh + offs_d * stride_qd
    )

    k = tl.load(
        K + (start_k + offs_n)[:, None] * stride_kb
          + off_h * stride_kh
          + offs_d[None, :] * stride_kd,
        mask=offs_n[:, None] < seqlen_k,
        other=0.0,
    )

    qk = tl.sum(q[None, :] * k, axis=1)
    qk = tl.where(offs_n < seqlen_k, qk, -1e5)

    tl.store(
        Out + start_o * stride_ob + off_h * stride_oh + offs_n * stride_on,
        qk,
        mask=offs_n < seqlen_k,
    )


# =========================
# Prefill kernel（不变）
# =========================
@triton.jit
def _prefill_qk_kernel(
    Q, K, Out,
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_output,
    batch_indices,
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
    off_b_local = tl.program_id(2)

    off_b = tl.load(batch_indices + off_b_local)

    num_tiles_n = tl.cdiv(max_seqlen_k, BLOCK_N)
    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n

    start_q = tl.load(cu_seqlens_q + off_b)
    start_k = tl.load(cu_seqlens_k + off_b)
    start_o = tl.load(cu_seqlens_output + off_b)

    end_q = tl.load(cu_seqlens_q + off_b + 1)
    end_k = tl.load(cu_seqlens_k + off_b + 1)

    seqlen_q = end_q - start_q
    seqlen_k = end_k - start_k
    if seqlen_q <= 0 or seqlen_k <= 0:
        return

    m_start = tile_m * BLOCK_M
    n_start = tile_n * BLOCK_N
    if m_start >= seqlen_q or n_start >= seqlen_k:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + (start_q + offs_m)[:, None] * stride_qb
          + off_h * stride_qh
          + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < seqlen_q,
        other=0.0,
    )

    k = tl.load(
        K + (start_k + offs_n)[:, None] * stride_kb
          + off_h * stride_kh
          + offs_d[None, :] * stride_kd,
        mask=offs_n[:, None] < seqlen_k,
        other=0.0,
    )

    qk = tl.dot(q, tl.trans(k))
    causal = (offs_m[:, None] + start_q) >= (offs_n[None, :] + start_k)
    qk = tl.where(causal, qk, -1e5)

    qk = tl.reshape(qk, (NUM_POOLS, Q_SHARED_TOKENS, BLOCK_N))
    qk = tl.max(qk, axis=1)

    pool_idx = m_start // Q_SHARED_TOKENS + tl.arange(0, NUM_POOLS)
    out_ptr = (
        Out
        + (start_o + pool_idx)[:, None] * stride_ob
        + off_h * stride_oh
        + offs_n[None, :] * stride_on
    )

    tl.store(out_ptr, qk, mask=offs_n[None, :] < seqlen_k)


# =========================
# Host 入口（关键修改在这里）
# =========================
def prefill_qk_varlen(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_q: paddle.Tensor,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    block_m: int = 64,
    block_n: int = 64,
    q_shared_tokens: int = 1,
):
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    batch_size = seq_lens_q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # ======= GPU 上计算 output_seqlens（关键）=======
    decode_mask = seq_lens_q == 1

    if q_shared_tokens > 1:
        output_seqlens = paddle.where(
            decode_mask,
            paddle.ones_like(seq_lens_q, dtype="int32"),
            (seq_lens_q + q_shared_tokens - 1) // q_shared_tokens,
        )
    else:
        output_seqlens = seq_lens_q.astype("int32")

    # ======= GPU prefix sum（int32）=======
    cu_seqlens_output = paddle.concat(
        [
            paddle.zeros([1], dtype="int32"),
            paddle.cumsum(output_seqlens, dtype="int32"),
        ],
        axis=0,
    )

    total_output = int(cu_seqlens_output[-1])

    # ======= 分离 decode / prefill batch index（很小，不是瓶颈）=======
    decode_batches = paddle.nonzero(decode_mask).flatten()
    prefill_batches = paddle.nonzero(~decode_mask).flatten()

    out = paddle.zeros(
        [total_output, num_heads, int(cu_seqlens_k[1:].max())],
        dtype=q.dtype,
    )

    if decode_batches.shape[0] > 0:
        grid = (triton.cdiv(out.shape[2], block_n), num_heads, decode_batches.shape[0])
        _decode_qk_kernel[grid](
            q, k, out,
            cu_seqlens_q, cu_seqlens_k, cu_seqlens_output,
            decode_batches,
            q.strides[0], q.strides[1], q.strides[2],
            k.strides[0], k.strides[1], k.strides[2],
            out.strides[0], out.strides[1], out.strides[2],
            max_seqlen_k=out.shape[2],
            HEAD_DIM=head_dim,
            BLOCK_N=block_n,
        )

    if prefill_batches.shape[0] > 0:
        grid = (
            triton.cdiv(seq_lens_q.max(), block_m)
            * triton.cdiv(out.shape[2], block_n),
            num_heads,
            prefill_batches.shape[0],
        )
        _prefill_qk_kernel[grid](
            q, k, out,
            cu_seqlens_q, cu_seqlens_k, cu_seqlens_output,
            prefill_batches,
            q.strides[0], q.strides[1], q.strides[2],
            k.strides[0], k.strides[1], k.strides[2],
            out.strides[0], out.strides[1], out.strides[2],
            max_seqlen_q=int(seq_lens_q.max()),
            max_seqlen_k=out.shape[2],
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            Q_SHARED_TOKENS=q_shared_tokens,
            NUM_POOLS=block_m // q_shared_tokens,
        )

    return out


# ========================== Unit Tests ==========================
def test_mixed():
    paddle.set_device("gpu:0")
    seqlens_q = [8, 1, 16, 1]
    seqlens_k = [32, 32, 32, 32]
    cu_q = paddle.to_tensor([0, 8, 9, 25, 26], dtype="int32")
    cu_k = paddle.to_tensor([0, 32, 64, 96, 128], dtype="int32")

    q = paddle.randn([26, 2, 64], dtype="float16")
    k = paddle.randn([128, 2, 64], dtype="float16")

    out = prefill_qk_varlen(
        q, k, cu_q, paddle.to_tensor(seqlens_q, dtype="int32"), cu_k, q_shared_tokens=4
    )
    print("Mixed OK:", out.shape)


def test_decode():
    paddle.set_device("gpu:0")
    q = paddle.ones([1, 2, 64])
    k = paddle.ones([32, 2, 64])
    out = prefill_qk_varlen(
        q, k,
        paddle.to_tensor([0, 1], dtype="int32"),
        paddle.to_tensor([1], dtype="int32"),
    )
    print("Decode OK:", out[0, 0, :5])


if __name__ == "__main__":
    test_decode()
    test_mixed()
