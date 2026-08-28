"""Misc paddle helpers for Elastic-Attention.

Contains:
- ``AttentionRouter``  : 3-layer MLP head router (per KV-head 0/1)
- ``derive_head_mask_type`` : retrieval/toggle -> {1, 0, -1} per Q-head
- ``ctx_q_pool``       : per-sequence mean-pooling of K (post k_norm, pre RoPE)
"""

from __future__ import annotations

import paddle
from paddle import nn

from fastdeploy.model_executor.utils import set_weight_attrs


class _LinearTransposed(nn.Linear):
    """nn.Linear that flags its weight for HF -> paddle transpose at load time."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias_attr=bias)
        set_weight_attrs(self.weight, {"weight_need_transpose": True})


class AttentionRouter(nn.Layer):
    """Inference-only 3-layer MLP router (matches PawQwen3 ``AttentionRouter``
    with ``use_softmax=True``)."""

    def __init__(self, num_kv_heads: int, d_feature: int = 128):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.d_feature = d_feature
        mid = 4 * d_feature

        self.cls_feat_extractor = nn.Sequential(
            _LinearTransposed(d_feature, mid),
            nn.Silu(),
            _LinearTransposed(mid, d_feature),
        )
        self.cls_router_head_agnostic = nn.Sequential(
            _LinearTransposed(d_feature, mid),
            nn.Silu(),
            _LinearTransposed(mid, d_feature),
            nn.Silu(),
            _LinearTransposed(d_feature, 2),
        )

    @paddle.no_grad()
    def forward(self, k_pooled: paddle.Tensor) -> paddle.Tensor:
        """Args:
            k_pooled: [B, H_kv, D] -- post-k_norm, pre-RoPE, seq-mean.
        Returns:
            z_kv: [B, H_kv] int32 with values in {0,1}.
        """
        h = self.cls_feat_extractor(k_pooled)
        logits = self.cls_router_head_agnostic(h)
        return logits.argmax(axis=-1).astype("int32")


def ctx_q_pool(k_post_norm: paddle.Tensor, cu_seq_lens: paddle.Tensor | None = None) -> paddle.Tensor:
    """Pool K over the sequence axis per request.

    Mirrors the HF reference ``AttentionRouter`` else-branch (eval path with
    ``cu_seq_len is None``) at ``modeling_flash_qwen.py``:

        target = torch.concat([x[:, :100, :], x[:, -100:, :]], dim=1).mean(dim=1)

    i.e. mean over the first 100 + last 100 tokens (with overlap when
    ``T < 200``, matching HF byte-for-byte).

    BS=1 fast path: ``k_post_norm`` is ``[T, H_kv, D]`` -> returns
    ``[1, H_kv, D]``. For general varlen, use ``cu_seq_lens`` ``[B+1]``.
    """
    HEAD = 100
    TAIL = 100

    def _pool_segment(seg: paddle.Tensor) -> paddle.Tensor:
        # seg: [Ti, H, D]
        Ti = seg.shape[0]
        head = seg[: min(HEAD, Ti)]
        tail = seg[-min(TAIL, Ti) :]
        cat = paddle.concat([head, tail], axis=0)  # [head+tail, H, D]
        return cat.astype("float32").mean(axis=0, keepdim=True).astype(seg.dtype)

    if k_post_norm.ndim == 3 and cu_seq_lens is None:
        return _pool_segment(k_post_norm)
    if cu_seq_lens is None:
        raise ValueError("cu_seq_lens required for varlen ctx_q_pool")
    B = int(cu_seq_lens.shape[0]) - 1
    out = []
    for i in range(B):
        s = int(cu_seq_lens[i].item())
        e = int(cu_seq_lens[i + 1].item())
        out.append(_pool_segment(k_post_norm[s:e]))
    return paddle.concat(out, axis=0)


def derive_head_mask_type(
    z_kv: paddle.Tensor,
    retrieval_mode: str,
    toggle_type: str,
    group_size: int = 1,
) -> paddle.Tensor:
    """Return ``head_mask_type`` for BSA: {1=block-sparse, 0=full, -1=streaming}.

    ``z_kv`` is ``[H_kv]`` int. If ``group_size>1`` (GQA), the result is
    ``repeat_interleave``'d to ``[H_kv*group_size]`` to match Q-heads.
    """
    z = z_kv.astype("int32")
    zero = paddle.zeros_like(z)
    one = paddle.ones_like(z)
    neg = -one
    key = (retrieval_mode, toggle_type)
    if key == ("full", "xattn"):
        out = (1 - z).astype("int32")
    elif key == ("full", "streaming"):
        out = paddle.where(z == 1, zero, neg).astype("int32")
    elif key == ("xattn", "streaming"):
        out = paddle.where(z == 1, one, neg).astype("int32")
    elif key == ("xattn", "xattn"):
        out = one
    elif key == ("full", "full"):
        out = zero
    else:
        raise NotImplementedError(f"unsupported (retrieval_mode, toggle_type) = {key}")
    if group_size > 1:
        out = paddle.repeat_interleave(out, group_size, axis=0)
    return out
