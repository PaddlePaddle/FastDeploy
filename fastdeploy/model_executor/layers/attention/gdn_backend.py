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
  - GDNKernelDispatcher: strategy pattern routing to Triton (FLA) kernels
  - GDNAttentionBackend(AttentionBackend): unified forward entry for model layer

Call chain:
  Model Layer: Qwen3_5GatedDeltaNet.forward()
    |-- projections (qkv, z, b, a)
    |-- self.gdn_attn(mixed_qkv, a, b, self, forward_meta)  [GDNAttention trampoline]
        |-- forward_meta.gdn_attn_backend.forward(mixed_qkv, a, b, layer, forward_meta)
            |
            GDNAttentionBackend.forward()
              |-- causal_conv1d (decode: triton update / prefill: triton varlen)
              |-- split Q,K,V + fused_gdn_gating
              |-- GVA repeat (if needed)
              |-- kernel_dispatcher.decode/extend(...)
                  |
                  GDNKernelDispatcher
                    |-- Triton FLA kernels (decode: fused_recurrent, extend: chunk)
    |-- gated RMSNorm + output projection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


# ==============================================================================
# Kernel Dispatcher (strategy pattern, inspired by SGLang GDNKernelDispatcher)
# ==============================================================================


class GDNKernelDispatcher:
    """Strategy pattern — routes SSM kernel calls to Triton (FLA) kernels.

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


# ==============================================================================
# GDN Attention Backend
# ==============================================================================


class GDNAttentionBackend(AttentionBackend):
    """GDN (Gated Delta Network) linear attention backend.

    Inherits AttentionBackend for formal consistency with FD/vLLM/SGLang.
    The model layer calls through GDNAttention trampoline:
      self.gdn_attn(mixed_qkv, a, b, self, forward_meta)
        → forward_meta.gdn_attn_backend.forward(...)

    Internal flow:
      1. Causal Conv1d (decode: triton update / prefill: triton varlen)
      2. Split Q,K,V + fused_gdn_gating
      3. GVA repeat (if num_v_heads > num_k_heads)
      4. SSM kernel via GDNKernelDispatcher (decode / extend)
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

        conv_pool = gdn_pool.get_layer_conv_pool(layer.gdn_layer_idx)
        ssm_pool = gdn_pool.get_layer_ssm_pool(layer.gdn_layer_idx)

        # Offset slot_ids: PAD_SLOT_ID (-1) → slot 0
        slot_ids = GDNStatePool.offset_slot_ids(raw_slot_ids)

        # ============================================================
        # 1. Causal Conv1d
        # ============================================================
        conv_weight_local = layer.conv1d_weight[: layer.conv_dim]

        if is_decode:
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
        else:
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
        if is_decode:
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

        else:
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

        return core_attn_out
