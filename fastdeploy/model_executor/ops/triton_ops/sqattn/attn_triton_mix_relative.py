import math

import paddle
import triton
import triton.language as tl


@triton.jit
def attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    kv_len,
    K_ptrs,
    V_ptrs,
    stride_kn,
    stride_vn,
    start_m,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    is_P_fp8_quant: tl.constexpr,
    current_int8_window_ratio,
    current_int4_window_ratio,
    SINK_SIZE: tl.constexpr,
):
    """
    Window Attention 前向计算的内部循环函数，支持 Sink + 分层窗口（相对窗口版本）。
    窗口大小现在是相对于序列长度的比例 (0-1)，但会对齐到128的倍数
    Sink size保持固定大小
    """
    if STAGE == 1:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        lo, hi = 0, tl.minimum(SINK_SIZE, query_start)
    elif STAGE == 2:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        int4_window_size_raw = tl.cast(current_int4_window_ratio * kv_len, tl.int32)
        int8_window_size_raw = tl.cast(current_int8_window_ratio * kv_len, tl.int32)
        int4_window_size = (int4_window_size_raw + 127) // 128 * 128
        int8_window_size = (int8_window_size_raw + 127) // 128 * 128
        int4_window_left = tl.maximum(SINK_SIZE, query_end - int4_window_size)
        int8_window_left = tl.maximum(SINK_SIZE, query_end - int8_window_size)
        condition = int4_window_left < int8_window_left
        lo = tl.where(condition, int4_window_left, 0)
        hi = tl.where(condition, int8_window_left, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 3:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        int8_window_size_raw = tl.cast(current_int8_window_ratio * kv_len, tl.int32)
        int8_window_size = (int8_window_size_raw + 127) // 128 * 128
        int8_window_left = tl.maximum(SINK_SIZE, query_end - int8_window_size)
        int8_window_right = query_start
        condition = int8_window_left < int8_window_right
        lo = tl.where(condition, int8_window_left, 0)
        hi = tl.where(condition, int8_window_right, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 4:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        lo = query_start
        hi = query_end
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < kv_len - start_n
        k = tl.load(K_ptrs, mask=k_mask)
        qk = tl.dot(q, k).to(tl.float32)
        if STAGE == 4:
            causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk + tl.where(causal_mask, 0, -1000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_mask = offs_n[:, None] < kv_len - start_n
        v = tl.load(V_ptrs, mask=v_mask)
        if is_P_fp8_quant:
            p = (p * 448).to(tl.float8e4nv).to(tl.float16) / 448
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def attn_prefill_fwd(
    Q_int8,
    K_int8,
    Q_int4,
    K_int4,
    V,
    Out,
    Lse,
    Int8WindowRatios,
    Int4WindowRatios,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    is_P_fp8_quant: tl.constexpr,
    SINK_SIZE: tl.constexpr,
):
    """
    分层 Window Attention Prefill Kernel，支持 Sink + 多精度窗口（相对窗口版本）
    窗口大小是相对比例，但会对齐到128的倍数；Sink size保持固定
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(2)
    off_h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_int8_ptrs = (
        Q_int8
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    K_int8_ptrs = (
        K_int8
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    Q_int4_ptrs = (
        Q_int4
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    K_int4_ptrs = (
        K_int4
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    V_ptrs = (
        V
        + (off_z * stride_vz + off_h // num_kv_groups * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + (off_z * stride_oz + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q_mask = offs_m[:, None] < qo_len
    q_int8 = tl.load(Q_int8_ptrs, mask=q_mask)
    q_int4 = tl.load(Q_int4_ptrs, mask=q_mask)
    current_int8_window_ratio = tl.load(Int8WindowRatios + off_h)
    current_int4_window_ratio = tl.load(Int4WindowRatios + off_h)
    acc, l_i, m_i = attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        1,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_ratio,
        current_int4_window_ratio,
        SINK_SIZE,
    )
    acc, l_i, m_i = attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int4,
        kv_len,
        K_int4_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        2,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_ratio,
        current_int4_window_ratio,
        SINK_SIZE,
    )
    acc, l_i, m_i = attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        3,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_ratio,
        current_int4_window_ratio,
        SINK_SIZE,
    )
    acc, l_i, m_i = attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_ratio,
        current_int4_window_ratio,
        SINK_SIZE,
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=offs_m < qo_len)


def attn_hierarchical_relative_window(
    q_int8,
    k_int8,
    q_int4,
    k_int4,
    v,
    int8_window_ratios,
    int4_window_ratios,
    sink_size,
    tensor_layout="HND",
    output_dtype=paddle.float16,
    return_lse=False,
    is_P_fp8_quant=False,
):
    """
    分层 Window Attention 接口，支持 Sink + 多精度窗口（相对窗口版本）

    Args:
        q_int8, k_int8: INT8 精度的 Query, Key 张量
        q_int4, k_int4: 4-bit 精度的 Query, Key 张量
        v: 单一精度的 Value 张量
        int8_window_ratios: INT8 窗口大小比例 (0-1 的浮点数)，会对齐到128倍数
        int4_window_ratios: 4-bit 窗口大小比例 (0-1 的浮点数)，会对齐到128倍数
        sink_size: Sink 窗口大小 (固定的整数值)
    """
    seq_dim = 1 if tensor_layout == "NHD" else 2
    qo_len = q_int8.shape[seq_dim]
    kv_len = k_int8.shape[seq_dim]
    if tensor_layout == "HND":
        H = q_int8.shape[1]
    else:
        H = q_int8.shape[2]
    if isinstance(int8_window_ratios, (int, float)):
        int8_ratios_tensor = paddle.full(
            [H], float(int8_window_ratios), dtype=paddle.float32, device=q_int8.place
        )
    elif isinstance(int8_window_ratios, (list, tuple)):
        int8_ratios_tensor = paddle.tensor(
            int8_window_ratios, dtype=paddle.float32, device=q_int8.place
        )
    else:
        int8_ratios_tensor = int8_window_ratios.to(
            device=q_int8.place, dtype=paddle.float32
        )
    if isinstance(int4_window_ratios, (int, float)):
        int4_ratios_tensor = paddle.full(
            [H], float(int4_window_ratios), dtype=paddle.float32, device=q_int8.place
        )
    elif isinstance(int4_window_ratios, (list, tuple)):
        int4_ratios_tensor = paddle.tensor(
            int4_window_ratios, dtype=paddle.float32, device=q_int8.place
        )
    else:
        int4_ratios_tensor = int4_window_ratios.to(
            device=q_int8.place, dtype=paddle.float32
        )
    int8_ratios_tensor = paddle.clamp(int8_ratios_tensor, 0.0, 1.0)
    int4_ratios_tensor = paddle.clamp(int4_ratios_tensor, 0.0, 1.0)
    if not isinstance(sink_size, int) or sink_size < 0:
        raise ValueError(f"sink_size must be a non-negative integer, got {sink_size}")
    head_dim_og = q_int8.shape[-1]
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q_int8, k_int8 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = paddle.compat.pad(v, (0, pad_size))
    elif head_dim_og > 64 and head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q_int8, k_int8 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = paddle.compat.pad(v, (0, pad_size))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    sm_scale = 1.0 / head_dim_og**0.5 * 1.44269504
    q_int8 = q_int8 * sm_scale
    q_int4 = q_int4 * sm_scale
    o = paddle.empty(q_int8.shape, dtype=output_dtype, device=q_int8.place)
    if tensor_layout == "HND":
        b, h_qo, _, head_dim = q_int8.shape
        _, h_kv, _, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            q_int8.strides[0],
            q_int8.strides[1],
            q_int8.strides[2],
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            k_int8.strides[0],
            k_int8.strides[1],
            k_int8.strides[2],
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.strides[0], v.strides[1], v.strides[2]
        stride_bz_o, stride_h_o, stride_seq_o = o.strides[0], o.strides[1], o.strides[2]
    elif tensor_layout == "NHD":
        b, _, h_qo, head_dim = q_int8.shape
        _, _, h_kv, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            q_int8.strides[0],
            q_int8.strides[2],
            q_int8.strides[1],
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            k_int8.strides[0],
            k_int8.strides[2],
            k_int8.strides[1],
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.strides[0], v.strides[2], v.strides[1]
        stride_bz_o, stride_h_o, stride_seq_o = o.strides[0], o.strides[2], o.strides[1]
    num_kv_groups = h_qo // h_kv
    lse = (
        paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=q_int8.place)
        if return_lse
        else paddle.empty([0], dtype=paddle.float32, device="cpu")
    )
    BLOCK_M = 128
    BLOCK_N = 64
    grid = triton.cdiv(qo_len, BLOCK_M), h_qo, b
    attn_prefill_fwd[grid](
        q_int8,
        k_int8,
        q_int4,
        k_int4,
        v,
        o,
        lse,
        int8_ratios_tensor,
        int4_ratios_tensor,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        STAGE=4,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
        is_P_fp8_quant=is_P_fp8_quant,
        SINK_SIZE=sink_size,
    )
    o = o[..., :head_dim_og]
    return (o, lse) if return_lse else o


"""
output = attn_hierarchical_window(q_int8, k_int8, q_int4, k_int4, v,
                                int8_window_ratios=0.25,  # 25%的序列长度，会对齐到128倍数
                                int4_window_ratios=0.5,   # 50%的序列长度，会对齐到128倍数
                                sink_size=64)             # 固定的sink大小
"""
