# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GDN (Gated Delta Network) Attention Backend for Qwen3.5 linear attention.

Architecture (inspired by SGLang's gdn_backend.py):
  - GDNKernelDispatcher: strategy pattern routing to Triton / Paddle fallback kernels
  - GDNAttentionBackend(AttentionBackend): unified forward entry for model layer

Call chain:
  Model Layer: Qwen3_5GatedDeltaNet.forward()
    |-- projections (qkv, z, b, a)
    |-- forward_meta.gdn_attn_backend.forward(mixed_qkv, a, b, layer, forward_meta)
        |
        GDNAttentionBackend.forward()
          |-- causal_conv1d (decode / prefill / fallback)
          |-- split Q,K,V + fused_gdn_gating
          |-- GVA repeat (if needed)
          |-- kernel_dispatcher.decode/extend/fallback(...)
              |
              GDNKernelDispatcher
                |-- Triton FLA kernels (decode: fused_recurrent, extend: chunk)
                |-- Paddle fallback (paddle_recurrent / paddle_chunk)
    |-- gated RMSNorm + output projection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import paddle

from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

logger = logging.getLogger(__name__)


# ==============================================================================
# Helper functions (extracted from qwen3_5.py)
# ==============================================================================


def l2norm(x: paddle.Tensor, axis: int = -1, eps: float = 1e-6) -> paddle.Tensor:
    """L2 normalization, aligns with FLA library."""
    inv_norm = paddle.rsqrt((x * x).sum(axis=axis, keepdim=True) + eps)
    return x * inv_norm


def fused_gdn_gating(
    A_log: paddle.Tensor,
    a: paddle.Tensor,
    b: paddle.Tensor,
    dt_bias: paddle.Tensor,
) -> tuple:
    """Compute GDN gating values.

    Args:
        A_log: [num_heads] - log of A matrix
        a: [num_tokens, num_heads] - alpha values
        b: [num_tokens, num_heads] - beta values
        dt_bias: [num_heads] - delta-time bias

    Returns:
        g: gating values, same shape as a
        beta: sigmoid(b), same shape as b
    """
    x = a.cast(paddle.float32) + dt_bias.cast(paddle.float32)
    softplus_x = paddle.nn.functional.softplus(x)
    g = -paddle.exp(A_log.cast(paddle.float32)) * softplus_x
    beta = paddle.nn.functional.sigmoid(b.cast(paddle.float32))
    return g, beta


def _causal_conv1d_single_seq(
    x: paddle.Tensor,
    conv_weights: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    activation: str,
    kernel_size: int,
) -> paddle.Tensor:
    """Apply causal conv1d to a single sequence (pure Paddle fallback).

    Args:
        x: [seq_len, channels]
        conv_weights: [channels, kernel_size]
    Returns:
        [seq_len, channels]
    """
    seq_len, channels = x.shape
    x = x.transpose([1, 0]).unsqueeze(0)  # [1, channels, seq_len]
    weight = conv_weights.unsqueeze(1)  # [channels, 1, kernel_size]
    padding = kernel_size - 1
    x = paddle.nn.functional.conv1d(x, weight, bias, padding=padding, groups=channels)
    x = x[:, :, :seq_len]
    x = x.squeeze(0).transpose([1, 0])  # [seq_len, channels]
    if activation == "silu":
        x = paddle.nn.functional.silu(x)
    return x


def _causal_conv1d_fn_fallback(
    x: paddle.Tensor,
    conv_weights: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    activation: str = "silu",
    cu_seqlens: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Causal conv1d for packed sequences (pure Paddle fallback).

    Args:
        x: [num_tokens, channels]
        conv_weights: [channels, kernel_size]
        cu_seqlens: [batch_size + 1] cumulative sequence lengths
    Returns:
        [num_tokens, channels]
    """
    kernel_size = conv_weights.shape[-1]
    if cu_seqlens is None:
        return _causal_conv1d_single_seq(x, conv_weights, bias, activation, kernel_size)

    batch_size = cu_seqlens.shape[0] - 1
    cu_seqlens_np = cu_seqlens.numpy()
    output = paddle.zeros_like(x)

    for i in range(batch_size):
        start = int(cu_seqlens_np[i])
        end = int(cu_seqlens_np[i + 1])
        if end <= start:
            continue
        seq_x = x[start:end, :]
        seq_out = _causal_conv1d_single_seq(seq_x, conv_weights, bias, activation, kernel_size)
        output[start:end, :] = seq_out

    return output


def _paddle_chunk_gated_delta_rule(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[paddle.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple:
    """Chunked Gated Delta Rule (pure Paddle fallback for prefill).

    Args:
        query: [batch, seq_len, num_heads, head_k_dim]
        key: [batch, seq_len, num_heads, head_k_dim]
        value: [batch, seq_len, num_heads, head_v_dim]
        g: [batch, seq_len, num_heads]
        beta: [batch, seq_len, num_heads]
    Returns:
        (output, last_state)
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = l2norm(query.cast(paddle.float32), axis=-1)
        key = l2norm(key.cast(paddle.float32), axis=-1)

    query = query.transpose([0, 2, 1, 3]).cast(paddle.float32)
    key = key.transpose([0, 2, 1, 3]).cast(paddle.float32)
    value = value.transpose([0, 2, 1, 3]).cast(paddle.float32)
    beta = beta.transpose([0, 2, 1]).cast(paddle.float32)
    g = g.transpose([0, 2, 1]).cast(paddle.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = paddle.nn.functional.pad(query, [0, 0, 0, pad_size])
    key = paddle.nn.functional.pad(key, [0, 0, 0, pad_size])
    value = paddle.nn.functional.pad(value, [0, 0, 0, pad_size])
    beta = paddle.nn.functional.pad(beta, [0, pad_size])
    g = paddle.nn.functional.pad(g, [0, pad_size])
    total_sequence_length = sequence_length + pad_size

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query = query.reshape([batch_size, num_heads, -1, chunk_size, k_head_dim])
    key = key.reshape([batch_size, num_heads, -1, chunk_size, k_head_dim])
    value = value.reshape([batch_size, num_heads, -1, chunk_size, v_head_dim])
    k_beta = k_beta.reshape([batch_size, num_heads, -1, chunk_size, k_head_dim])
    v_beta = v_beta.reshape([batch_size, num_heads, -1, chunk_size, v_head_dim])
    g = g.reshape([batch_size, num_heads, -1, chunk_size])

    g = g.cumsum(axis=-1)
    decay_mask = paddle.tril(paddle.exp((g.unsqueeze(-1) - g.unsqueeze(-2)).tril()))

    mask = paddle.triu(paddle.ones([chunk_size, chunk_size], dtype="bool"), diagonal=0)
    attn = -((k_beta @ key.transpose([0, 1, 2, 4, 3])) * decay_mask)
    attn = paddle.where(mask, paddle.zeros_like(attn), attn)

    for i in range(1, chunk_size):
        row = attn[:, :, :, i, :i].clone()
        sub = attn[:, :, :, :i, :i].clone()
        attn[:, :, :, i, :i] = row + (row.unsqueeze(-1) * sub).sum(axis=-2)

    attn = attn + paddle.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        paddle.zeros([batch_size, num_heads, k_head_dim, v_head_dim])
        if initial_state is None
        else initial_state.cast(value.dtype)
    )

    core_attn_out = paddle.zeros_like(value)
    mask = paddle.triu(paddle.ones([chunk_size, chunk_size], dtype="bool"), diagonal=1)

    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]

        attn_i = q_i @ k_i.transpose([0, 1, 3, 2]) * decay_mask[:, :, i]
        attn_i = paddle.where(mask, paddle.zeros_like(attn_i), attn_i)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new

        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose([0, 1, 3, 2]) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape([batch_size, num_heads, -1, v_head_dim])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose([0, 2, 1, 3]).cast(initial_dtype)

    return core_attn_out, last_recurrent_state


def _paddle_recurrent_gated_delta_rule(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    initial_state: Optional[paddle.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple:
    """Recurrent Gated Delta Rule (pure Paddle fallback for decode).

    Args:
        query: [batch, seq_len, num_heads, head_k_dim]
        key: [batch, seq_len, num_heads, head_k_dim]
        value: [batch, seq_len, num_heads, head_v_dim]
        g: [batch, seq_len, num_heads]
        beta: [batch, seq_len, num_heads]
    Returns:
        (output, last_state)
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = l2norm(query.cast(paddle.float32), axis=-1)
        key = l2norm(key.cast(paddle.float32), axis=-1)

    query = query.transpose([0, 2, 1, 3]).cast(paddle.float32)
    key = key.transpose([0, 2, 1, 3]).cast(paddle.float32)
    value = value.transpose([0, 2, 1, 3]).cast(paddle.float32)
    beta = beta.transpose([0, 2, 1]).cast(paddle.float32)
    g = g.transpose([0, 2, 1]).cast(paddle.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = paddle.zeros([batch_size, num_heads, sequence_length, v_head_dim])
    last_recurrent_state = (
        paddle.zeros([batch_size, num_heads, k_head_dim, v_head_dim])
        if initial_state is None
        else initial_state.cast(core_attn_out.dtype)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(axis=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(axis=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose([0, 2, 1, 3]).cast(initial_dtype)
    return core_attn_out, last_recurrent_state


def _fused_recurrent_gated_delta_rule_fallback(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    initial_state: Optional[paddle.Tensor] = None,
) -> tuple:
    """Fallback wrapper: converts 3D [num_tokens, H, D] to 4D and dispatches.

    Returns:
        (output [num_tokens, H, DV], last_state)
    """
    num_tokens = q.shape[0]
    q_4d = q.unsqueeze(0)
    k_4d = k.unsqueeze(0)
    v_4d = v.unsqueeze(0)
    g_4d = g.unsqueeze(0)
    beta_4d = beta.unsqueeze(0)

    if num_tokens > 1:
        out, last_state = _paddle_chunk_gated_delta_rule(
            q_4d,
            k_4d,
            v_4d,
            g_4d,
            beta_4d,
            chunk_size=64,
            initial_state=initial_state,
            output_final_state=False,
        )
    else:
        out, last_state = _paddle_recurrent_gated_delta_rule(
            q_4d,
            k_4d,
            v_4d,
            g_4d,
            beta_4d,
            initial_state=initial_state,
            output_final_state=False,
        )
    return out.squeeze(0), last_state


# ==============================================================================
# Kernel Dispatcher (strategy pattern, inspired by SGLang GDNKernelDispatcher)
# ==============================================================================


class GDNKernelDispatcher:
    """Strategy pattern — routes SSM kernel calls to Triton or Paddle fallback.

    Currently: Triton (FLA) kernels only.
    Future: add FlashInfer/CUDA by selecting different kernels at construction time.
    """

    def decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        g: paddle.Tensor,
        beta: paddle.Tensor,
        *,
        ssm_pool: paddle.Tensor,
        slot_ids: paddle.Tensor,
    ) -> paddle.Tensor:
        """Decode: fused recurrent kernel (pool-indexed, in-place).

        Args:
            q,k: [batch, 1, H, DK]
            v: [batch, 1, H, DV]
            g,beta: [batch, 1, H]
            ssm_pool: [pool_size, H, K, V]
            slot_ids: [batch] int32 (already offset: PAD→0)
        Returns:
            [batch, 1, H, DV]
        """
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            fused_recurrent_gated_delta_rule_update,
        )

        return fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_pool=ssm_pool,
            ssm_indices=slot_ids,
            use_qk_l2norm_in_kernel=True,
        )

    def extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        g: paddle.Tensor,
        beta: paddle.Tensor,
        *,
        ssm_pool: paddle.Tensor,
        slot_ids: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
    ) -> paddle.Tensor:
        """Prefill/extend: chunk kernel + state writeback.

        Args:
            q,k: [1, total_tokens, H, DK]
            v: [1, total_tokens, H, DV]
            g,beta: [1, total_tokens, H]
            ssm_pool: [pool_size, H, K, V]
            slot_ids: [batch] int32 (already offset: PAD→0)
            cu_seqlens: [batch+1] int32
        Returns:
            [1, total_tokens, H, DV]
        """
        from fastdeploy.model_executor.ops.triton_ops.fla import chunk_gated_delta_rule

        # Clone initial state from pool — chunk kernel updates in-place
        initial_state = ssm_pool[slot_ids].clone()  # [batch, H, K, V]

        o, _h = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=None,
            initial_state=initial_state,
            initial_state_indices=paddle.arange(slot_ids.shape[0], dtype=paddle.int32),
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        # Write back final states (updated in-place by the kernel)
        batch_size = slot_ids.shape[0]
        for i in range(batch_size):
            sid = int(slot_ids[i])
            if sid > 0:  # skip padding sentinel (slot 0)
                ssm_pool[sid] = initial_state[i]

        return o

    def decode_fallback(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        g: paddle.Tensor,
        beta: paddle.Tensor,
    ) -> paddle.Tensor:
        """Fallback decode: pure Paddle recurrent (no pool).

        Args: q,k,v,g,beta in 3D [num_tokens, H, D] format
        Returns: [num_tokens, H, DV]
        """
        return _fused_recurrent_gated_delta_rule_fallback(q, k, v, g, beta)[0]

    def extend_fallback(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        g: paddle.Tensor,
        beta: paddle.Tensor,
    ) -> paddle.Tensor:
        """Fallback extend: pure Paddle chunk (no pool).

        Args: q,k,v,g,beta in 3D [num_tokens, H, D] format
        Returns: [num_tokens, H, DV]
        """
        return _fused_recurrent_gated_delta_rule_fallback(q, k, v, g, beta)[0]


# ==============================================================================
# GDN Attention Backend
# ==============================================================================


class GDNAttentionBackend(AttentionBackend):
    """GDN (Gated Delta Network) linear attention backend.

    Inherits AttentionBackend for formal consistency with FD/vLLM/SGLang.
    The model layer calls forward_meta.gdn_attn_backend.forward() directly
    (not through the standard Attention trampoline).

    Internal flow:
      1. Causal Conv1d (decode: triton update / prefill: triton varlen / fallback)
      2. Split Q,K,V + fused_gdn_gating
      3. GVA repeat (if num_v_heads > num_k_heads)
      4. SSM kernel via GDNKernelDispatcher (decode / extend / fallback)
    """

    def __init__(self):
        self.kernel_dispatcher = GDNKernelDispatcher()

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """GDN does not need standard attention metadata."""
        pass

    def forward(
        self,
        mixed_qkv: paddle.Tensor,
        a: paddle.Tensor,
        b: paddle.Tensor,
        layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Unified forward entry — model layer calls this single method.

        Args:
            mixed_qkv: [num_tokens, conv_dim] — projected QKV (after in_proj_qkv)
            a: [num_tokens, num_v_heads] — alpha gating input
            b: [num_tokens, num_v_heads] — beta gating input
            layer: Qwen3_5GatedDeltaNet instance (provides conv_weight, A_log, dt_bias, dims)
            forward_meta: ForwardMeta (contains pool, slot_ids, forward_mode, etc.)

        Returns:
            [num_tokens, num_v_heads_local, head_v_dim]
        """
        from fastdeploy.cache_manager.gdn_state_pool import GDNStatePool

        num_tokens = mixed_qkv.shape[0]
        is_decode = forward_meta.forward_mode.is_decode()

        # Get pool views for this layer
        gdn_pool = forward_meta.gdn_state_pool
        raw_slot_ids = forward_meta.gdn_slot_ids

        conv_pool = gdn_pool.get_layer_conv_pool(layer.gdn_layer_idx) if gdn_pool is not None else None
        ssm_pool = gdn_pool.get_layer_ssm_pool(layer.gdn_layer_idx) if gdn_pool is not None else None

        # Offset slot_ids: PAD_SLOT_ID (-1) → slot 0
        slot_ids = GDNStatePool.offset_slot_ids(raw_slot_ids) if raw_slot_ids is not None else None

        has_pool = conv_pool is not None and slot_ids is not None

        # ============================================================
        # 1. Causal Conv1d
        # ============================================================
        conv_weight_local = layer.conv1d_weight[: layer.conv_dim]

        if is_decode and has_pool:
            from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
                causal_conv1d_update as triton_conv1d_update,
            )

            mixed_qkv = triton_conv1d_update(
                x=mixed_qkv,
                conv_state=conv_pool,
                weight=conv_weight_local,
                bias=None,
                activation="silu",
                conv_state_indices=slot_ids,
            )
        elif not is_decode and has_pool:
            from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
                causal_conv1d_fn as triton_conv1d_fn,
            )

            cu_seqlens = forward_meta.cu_seqlens_q
            mixed_qkv = triton_conv1d_fn(
                x=mixed_qkv.T,  # [dim, total_tokens]
                weight=conv_weight_local,
                bias=None,
                conv_states=conv_pool,
                query_start_loc=cu_seqlens,
                seq_lens_cpu=forward_meta.gdn_seq_lens_cpu,
                cache_indices=slot_ids,
                has_initial_state=forward_meta.gdn_has_initial_state,
                activation="silu",
            ).T  # [total_tokens, dim]
        else:
            cu_seqlens = forward_meta.cu_seqlens_q
            if cu_seqlens is None and forward_meta.seq_lens_this_time is not None:
                seq_lens = forward_meta.seq_lens_this_time.numpy()
                cu_seqlens_list = [0]
                for sl in seq_lens:
                    cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
                cu_seqlens = paddle.to_tensor(cu_seqlens_list, dtype="int32")
            mixed_qkv = _causal_conv1d_fn_fallback(
                mixed_qkv,
                conv_weight_local,
                bias=None,
                activation="silu",
                cu_seqlens=cu_seqlens,
            )

        # ============================================================
        # 2. Split Q, K, V
        # ============================================================
        key_dim_local = layer.num_k_heads_local * layer.head_k_dim
        value_dim_local = layer.num_v_heads_local * layer.head_v_dim

        q, k, v = paddle.split(
            mixed_qkv,
            [
                key_dim_local,
                key_dim_local,
                value_dim_local,
            ],
            axis=-1,
        )

        q = q.reshape([num_tokens, layer.num_k_heads_local, layer.head_k_dim])
        k = k.reshape([num_tokens, layer.num_k_heads_local, layer.head_k_dim])
        v = v.reshape([num_tokens, layer.num_v_heads_local, layer.head_v_dim])

        # ============================================================
        # 3. GDN Gating
        # ============================================================
        A_log_local = layer.A_log[: layer.num_v_heads_local]
        dt_bias_local = layer.dt_bias[: layer.num_v_heads_local]
        a_local = a[:, : layer.num_v_heads_local]
        b_local = b[:, : layer.num_v_heads_local]

        g, beta = fused_gdn_gating(A_log_local, a_local, b_local, dt_bias_local)

        # ============================================================
        # 4. GVA repeat (if num_v_heads > num_k_heads)
        # ============================================================
        if layer.num_v_heads_per_k_head > 1:
            q = (
                q.unsqueeze(2)
                .expand([num_tokens, layer.num_k_heads_local, layer.num_v_heads_per_k_head, layer.head_k_dim])
                .reshape([num_tokens, layer.num_v_heads_local, layer.head_k_dim])
            )
            k = (
                k.unsqueeze(2)
                .expand([num_tokens, layer.num_k_heads_local, layer.num_v_heads_per_k_head, layer.head_k_dim])
                .reshape([num_tokens, layer.num_v_heads_local, layer.head_k_dim])
            )

        # ============================================================
        # 5. Core SSM Attention (via dispatcher)
        # ============================================================
        if is_decode and has_pool:
            # Decode: fused recurrent (Triton)
            q_4d = q.unsqueeze(1)  # [batch, 1, H, K]
            k_4d = k.unsqueeze(1)
            v_4d = v.unsqueeze(1)
            g_4d = g.unsqueeze(1)  # [batch, 1, H]
            beta_4d = beta.unsqueeze(1)

            o = self.kernel_dispatcher.decode(
                q_4d,
                k_4d,
                v_4d,
                g_4d,
                beta_4d,
                ssm_pool=ssm_pool,
                slot_ids=slot_ids,
            )
            core_attn_out = o.squeeze(1)  # [batch, H, V]

        elif not is_decode and has_pool:
            # Prefill/extend: chunk (Triton)
            cu_seqlens = forward_meta.cu_seqlens_q
            q_4d = q.unsqueeze(0)  # [1, total_tokens, H, K]
            k_4d = k.unsqueeze(0)
            v_4d = v.unsqueeze(0)
            g_4d = g.unsqueeze(0)  # [1, total_tokens, H]
            beta_4d = beta.unsqueeze(0)

            o = self.kernel_dispatcher.extend(
                q_4d,
                k_4d,
                v_4d,
                g_4d,
                beta_4d,
                ssm_pool=ssm_pool,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
            )
            core_attn_out = o.squeeze(0)  # [total_tokens, H, V]

        else:
            # Fallback: pure Paddle (no pool)
            core_attn_out = self.kernel_dispatcher.extend_fallback(q, k, v, g, beta)

        return core_attn_out
