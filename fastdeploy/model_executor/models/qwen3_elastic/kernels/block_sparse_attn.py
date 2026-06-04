"""Thin paddle wrapper around the Block-Sparse-Attention CUDA custom op.

The CUDA kernels (Block-Sparse-Attention/csrc/*) are compiled into a
**standalone** Paddle extension ``block_sparse_attn_ops`` via
``custom_ops/gpu_ops/block_sparse_attn/setup.py``. They are NOT part of the
main ``fastdeploy_ops`` build because BSA bundles its own (incompatible)
CUTLASS version. After building, the op is exposed as
``block_sparse_attn_ops.block_sparse_attn_fwd``.

Signature mirrors PyTorch ``block_sparse_attn_func`` (forward only):

    block_sparse_attn_fwd(
        q, k, v,                  # [total_T, H, D]
        cu_seqlens_q, cu_seqlens_k,  # int32 [B+1]
        head_mask_type,           # int32 [H]
        streaming_info,           # int32 [2*H] or None
        base_blockmask,           # bool  [B,H,Qb,Kb]
        max_seqlen_q, max_seqlen_k,
        p_dropout, softmax_scale,
        is_causal, exact_streaming, deterministic,
    ) -> attn_out [total_T, H, D]
"""

from __future__ import annotations

import paddle


def _import_bsa():
    """Lazy import so import-time errors don't kill the package when BSA is
    not yet compiled (lets unit tests for router etc. still run)."""
    try:
        # BSA is built as a STANDALONE Paddle extension (`block_sparse_attn_ops`)
        # via custom_ops/gpu_ops/block_sparse_attn/setup.py — it is NOT merged
        # into the main `fastdeploy_ops` build because BSA bundles its own
        # CUTLASS 3.3 which conflicts with FastDeploy's newer CUTLASS.
        from block_sparse_attn_ops import block_sparse_attn_fwd
        return block_sparse_attn_fwd
    except Exception as e:  # pragma: no cover - depends on build
        raise RuntimeError(
            "block_sparse_attn_fwd custom op not available. Build & install it via "
            "`cd FastDeploy/custom_ops/gpu_ops/block_sparse_attn && python setup.py install`."
        ) from e


def _replace_ones_with_count(head_mask_type: paddle.Tensor):
    """Replace each 1 in head_mask_type with its sequential 1-based count.

    The CUDA kernel indexes blockmask via ``(mask_type - 1)`` as the sparse-head
    axis. All sparse heads naively share mask_type=1, so they would all read the
    same blockmask row. This function assigns unique indices 1, 2, 3, ... to
    each sparse head from left to right, mirroring PyTorch
    ``block_sparse_attn_interface.replace_ones_with_count``.

    Returns: (modified_head_mask_type, num_sparse_heads_int)
    """
    ones_mask = (head_mask_type == 1)
    num_sparse = int(ones_mask.sum().item())
    if num_sparse == 0:
        return head_mask_type, 0
    # cumsum gives sequential 1, 2, 3, ... at positions of 1s; 0 elsewhere
    count = paddle.cumsum(ones_mask.astype("int32"), axis=-1).astype("int32") * ones_mask.astype("int32")
    result = paddle.where(ones_mask, count, head_mask_type)
    return result, num_sparse


def _convert_blockmask_row_reverse(blockmask: paddle.Tensor) -> paddle.Tensor:
    """Convert boolean blockmask to sorted-descending K-block indices.

    Input:  [B, H_sparse, Qb, Kb] bool (True = this K-block is attended to)
    Output: [B, H_sparse, Qb, Kb] int32 where each row is a sorted-descending
            list of K-block indices; padding positions contain -1.

    The CUDA binary-search (``fwdBlockmask::max_no_larger``) requires the row to
    be sorted in descending order so that it can binary-search for the largest
    K-block index <= the current causal bound. Padding is -1.

    Mirrors PyTorch ``block_sparse_attn_interface.convert_blockmask_row_reverse``.
    """
    # Cast bool → int32: sort doesn't operate on bool reliably
    bm = blockmask.astype("int32")
    # Argsort ascending along K-block axis: 0s land first, 1s land last
    sorted_idx = paddle.argsort(bm, axis=-1, stable=True, descending=False)
    sorted_vals = paddle.sort(bm, axis=-1, stable=True, descending=False)
    # Positions whose sorted value is 0 are padding → mark as -1
    sorted_idx = paddle.where(
        sorted_vals == 0,
        paddle.full_like(sorted_idx, -1),
        sorted_idx,
    )
    # Flip to descending order: largest valid K-block index first, -1s at end
    return paddle.flip(sorted_idx, axis=[-1]).astype("int32").contiguous()


@paddle.no_grad()
def block_sparse_attn_paddle(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    head_mask_type: paddle.Tensor,
    streaming_info,
    base_blockmask: paddle.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float = 0.0,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_size_left: int = -1,
    window_size_right: int = -1,
    m_block_dim: int = 128,
    n_block_dim: int = 128,
    exact_streaming: bool = False,
    deterministic: bool = True,
    return_softmax: bool = False,
):
    if softmax_scale is None:
        softmax_scale = float(q.shape[-1]) ** -0.5
    fwd = _import_bsa()

    # Give each sparse head a unique 1-based index so the kernel can address
    # each head's own blockmask row via (mask_type - 1).
    # Mirrors PyTorch block_sparse_attn_func::replace_ones_with_count.
    head_mask_type, _ = _replace_ones_with_count(head_mask_type)

    # Convert boolean blockmask to sorted-descending K-block index format
    # expected by the CUDA binary-search iterator.
    # Mirrors PyTorch BlockSparseAttnFunc.forward::convert_blockmask_row_reverse.
    if base_blockmask is not None:
        base_blockmask = _convert_blockmask_row_reverse(base_blockmask)

    if is_causal:
        window_size_right = 0
    out = fwd(
        q.contiguous() if not q.is_contiguous() else q,
        k.contiguous() if not k.is_contiguous() else k,
        v.contiguous() if not v.is_contiguous() else v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(p_dropout),
        float(softmax_scale),
        bool(is_causal),
        int(window_size_left),
        int(window_size_right),
        int(m_block_dim),
        int(n_block_dim),
        bool(exact_streaming),
        bool(return_softmax),
    )
    # Op returns [out, softmax_lse]; xattention only needs out.
    if isinstance(out, (list, tuple)):
        return out[0]
    return out
