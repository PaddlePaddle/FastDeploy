from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import paddle

from fastdeploy.model_executor.layers.attention.cute import (
    flash_attn_varlen_func as _flash_attn_varlen_func,
)


def _maybe_contiguous(x: Optional[paddle.Tensor]) -> Optional[paddle.Tensor]:
    return x.contiguous() if x is not None and x.strides[-1] != 1 else x


def flash_attn_varlen_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (-1, -1),
    learnable_sink: Optional[paddle.Tensor] = None,
    sinks: Optional[paddle.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    **_: object,
):

    q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
    cu_seqlens_q, cu_seqlens_k = [_maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)]
    seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
    page_table = _maybe_contiguous(page_table)

    if learnable_sink is None and sinks is not None:
        learnable_sink = sinks

    if window_size == (-1, -1):
        window_size = (None, None)

    result = _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result


def flash_attn_with_kvcache(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    k: Optional[paddle.Tensor] = None,
    v: Optional[paddle.Tensor] = None,
    qv: Optional[paddle.Tensor] = None,
    rotary_cos: Optional[paddle.Tensor] = None,
    rotary_sin: Optional[paddle.Tensor] = None,
    cache_seqlens: Optional[Union[int, paddle.Tensor]] = None,
    cache_batch_idx: Optional[paddle.Tensor] = None,
    cache_leftpad: Optional[paddle.Tensor] = None,
    page_table: Optional[paddle.Tensor] = None,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k_new: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[paddle.Tensor] = None,
    q_descale: Optional[paddle.Tensor] = None,
    k_descale: Optional[paddle.Tensor] = None,
    v_descale: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: Optional[int] = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    sinks: Optional[paddle.Tensor] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_softmax_lse: bool = False,
    **_: object,
):
    if k is not None or v is not None or qv is not None:
        raise NotImplementedError("FA4 does not support updating KV cache in-place.")
    if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
        raise NotImplementedError("FA4 path does not support rotary embedding.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError("FA4 path does not support non-consecutive batch indices or left padding.")
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError("FA4 path does not support descale.")

    if isinstance(cache_seqlens, int):
        cache_seqlens = paddle.full(
            (k_cache.shape[0],),
            cache_seqlens,
            dtype=paddle.int32,
        )

    result = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap != 0.0 else None,
        window_size=window_size,
        num_splits=num_splits if num_splits != 0 else 1,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_softmax_lse=True,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result
