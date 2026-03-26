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
GDN (Gated Delta Network) Triton Kernel 单元测试。

测试覆盖:
  1. TestFusedRecurrentGDN  — fused_recurrent_gated_delta_rule (Decode 路径)
  2. TestChunkGDN           — chunk_gated_delta_rule (Prefill 路径)
  3. TestCausalConv1dUpdate — causal_conv1d_update (Decode conv)
  4. TestCausalConv1dFn     — causal_conv1d_fn (Prefill conv, varlen)

参考基准:
  - GDN: Transformers 动态图中的 torch_recurrent_gated_delta_rule /
         torch_chunk_gated_delta_rule（Pure PyTorch 实现，与 FLA 论文对齐）
         移植为纯 paddle 实现后作为 baseline。
  - Conv1d: torch_causal_conv1d_update（来自 Transformers 的 F.conv1d 参考）
            移植为纯 paddle 实现后作为 baseline。

运行方法:
  cd /root/.../FastDeploy
  python -m pytest tests/model_executor/ops/triton_ops/test_gdn_kernels.py -v
  # 或
  python tests/model_executor/ops/triton_ops/test_gdn_kernels.py
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

# ============================================================
# Pure-Paddle Reference Implementations (ported from Transformers)
# ============================================================


def _l2norm_paddle(x: paddle.Tensor, dim: int = -1, eps: float = 1e-6) -> paddle.Tensor:
    """L2 norm, aligned with FLA's l2norm_fwd."""
    inv_norm = paddle.rsqrt((x * x).sum(axis=dim, keepdim=True) + eps)
    return x * inv_norm


def paddle_causal_conv1d_update_ref(
    hidden_states: paddle.Tensor,
    conv_state: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor = None,
    activation: str = "silu",
) -> paddle.Tensor:
    """
    Pure-Paddle reference for single-token causal conv1d update.

    Args:
        hidden_states: [batch, dim, 1]  (unsqueezed)
        conv_state: [batch, dim, state_len]  (single-sequence, NOT pool)
        weight: [dim, width]
        bias: [dim,] or None
        activation: "silu" or None

    Returns:
        out: [batch, dim]
        (conv_state is updated in-place)
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = paddle.concat([conv_state, hidden_states], axis=-1).cast(weight.dtype)
    # update conv_state in-place (shift left)
    conv_state_new = hidden_states_new[:, :, -state_len:]
    for i in range(conv_state.shape[0]):
        conv_state[i] = conv_state_new[i]
    # grouped conv1d: weight [dim, width] → [dim, 1, width]
    w = weight.unsqueeze(1)  # [dim, 1, width]
    out = F.conv1d(hidden_states_new, w, bias, padding=0, groups=hidden_size)
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    out = out[:, :, -seq_len:]  # keep last seq_len output
    return out.squeeze(-1)  # [batch, dim]


def paddle_recurrent_gated_delta_rule_ref(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple:
    """
    Pure-Paddle reference for fused recurrent GDN (Decode 路径).

    Args:
        query, key: [B, T, H, K]
        value: [B, T, H, V]
        g: [B, T, H]   log decay (negative)
        beta: [B, T, H] write gate
        initial_state: [B, H, K, V] or None
        output_final_state: bool
        use_qk_l2norm_in_kernel: bool

    Returns:
        out: [B, T, H, V]
        last_state: [B, H, K, V] if output_final_state else None
    """
    if use_qk_l2norm_in_kernel:
        query = _l2norm_paddle(query, dim=-1)
        key = _l2norm_paddle(key, dim=-1)

    # Transpose to [B, H, T, D] and cast to float32
    query, key, value, beta, g = [
        x.transpose([0, 2, 1, 3]).cast(paddle.float32) if x.ndim == 4 else x.transpose([0, 2, 1]).cast(paddle.float32)
        for x in (query, key, value, beta, g)
    ]

    B, H, T, K = key.shape
    V = value.shape[-1]
    scale = 1.0 / (K**0.5)
    query = query * scale

    out = paddle.zeros([B, H, T, V], dtype=paddle.float32)
    last_state = (
        paddle.zeros([B, H, K, V], dtype=paddle.float32)
        if initial_state is None
        else initial_state.cast(paddle.float32)
    )

    for i in range(T):
        q_t = query[:, :, i]  # [B, H, K]
        k_t = key[:, :, i]  # [B, H, K]
        v_t = value[:, :, i]  # [B, H, V]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)  # [B, H, 1]

        last_state = last_state * g_t
        kv_mem = (last_state * k_t.unsqueeze(-1)).sum(axis=-2)  # [B, H, V]
        delta = (v_t - kv_mem) * beta_t  # [B, H, V]
        last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (last_state * q_t.unsqueeze(-1)).sum(axis=-2)

    if not output_final_state:
        last_state = None

    out = out.transpose([0, 2, 1, 3])  # [B, T, H, V]
    return out, last_state


def paddle_chunk_gated_delta_rule_ref(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    chunk_size: int = 64,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple:
    """
    Pure-Paddle reference for chunk GDN (Prefill 路径).

    Closely mirrors Transformers' torch_chunk_gated_delta_rule.
    """
    if use_qk_l2norm_in_kernel:
        query = _l2norm_paddle(query, dim=-1)
        key = _l2norm_paddle(key, dim=-1)

    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose([0, 2, 1, 3]).cast(paddle.float32) if x.ndim == 4 else x.transpose([0, 2, 1]).cast(paddle.float32)
        for x in (query, key, value, beta, g)
    ]

    B, H, T, K = key.shape
    V = value.shape[-1]
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    query = F.pad(query, [0, 0, 0, pad_size])
    key = F.pad(key, [0, 0, 0, pad_size])
    value = F.pad(value, [0, 0, 0, pad_size])
    beta = F.pad(beta, [0, pad_size])
    g = F.pad(g, [0, pad_size])
    TT = T + pad_size

    scale = 1.0 / (K**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    NC = TT // chunk_size
    query = query.reshape([B, H, NC, chunk_size, K])
    key = key.reshape([B, H, NC, chunk_size, K])
    value = value.reshape([B, H, NC, chunk_size, V])
    k_beta = k_beta.reshape([B, H, NC, chunk_size, K])
    v_beta = v_beta.reshape([B, H, NC, chunk_size, V])
    g = g.reshape([B, H, NC, chunk_size])

    mask = paddle.triu(paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=0)

    g = g.cumsum(axis=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().cast(paddle.float32)).tril()

    attn = -((k_beta @ key.transpose([0, 1, 2, 4, 3])) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[:, :, :, i, :i].clone()
        sub = attn[:, :, :, :i, :i].clone()
        attn[:, :, :, i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    eye = paddle.eye(chunk_size, dtype=attn.dtype)
    attn = attn + eye

    value_new = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = (
        paddle.zeros([B, H, K, V], dtype=value_new.dtype)
        if initial_state is None
        else initial_state.cast(value_new.dtype)
    )
    core_attn_out = paddle.zeros_like(value_new)
    mask2 = paddle.triu(paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=1)

    for i in range(NC):
        q_i = query[:, :, i]  # [B, H, cs, K]
        k_i = key[:, :, i]
        v_i = value_new[:, :, i]
        attn_i = (q_i @ k_i.transpose([0, 1, 3, 2]) * decay_mask[:, :, i]).masked_fill(mask2, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new_i = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new_i
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp().unsqueeze(-1)).transpose([0, 1, 3, 2]) @ v_new_i
        )

    if not output_final_state:
        last_state = None

    core_attn_out = core_attn_out.reshape([B, H, TT, V])
    core_attn_out = core_attn_out[:, :, :T]
    core_attn_out = core_attn_out.transpose([0, 2, 1, 3]).cast(initial_dtype)
    return core_attn_out, last_state


# ============================================================
# Test Cases
# ============================================================


class TestFusedRecurrentGDN(unittest.TestCase):
    """测试 fused_recurrent_gated_delta_rule (Decode 路径 SSM kernel)."""

    def setUp(self):
        paddle.seed(42)
        self.dtype = paddle.bfloat16
        self.B, self.T = 2, 8
        self.H, self.K, self.V = 4, 64, 64

    def _make_inputs(self, T=None):
        T = T or self.T
        B, H, K, V = self.B, self.H, self.K, self.V
        q = paddle.randn([B, T, H, K], dtype=paddle.float32).cast(self.dtype)
        k = paddle.randn([B, T, H, K], dtype=paddle.float32).cast(self.dtype)
        v = paddle.randn([B, T, H, V], dtype=paddle.float32).cast(self.dtype)
        # g: negative log decay
        g = -F.softplus(paddle.randn([B, T, H], dtype=paddle.float32)).cast(self.dtype)
        beta = paddle.sigmoid(paddle.randn([B, T, H], dtype=paddle.float32)).cast(self.dtype)
        return q, k, v, g, beta

    def test_fused_recurrent_no_state(self):
        """不带初始状态，kernel 输出应与 baseline 一致。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            fused_recurrent_gated_delta_rule,
        )

        q, k, v, g, beta = self._make_inputs()

        ref_out, _ = paddle_recurrent_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

        kernel_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="fused_recurrent_gated_delta_rule (no state) mismatch",
        )

    def test_fused_recurrent_with_l2norm(self):
        """带 L2 norm，kernel 输出应与 baseline 一致。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            fused_recurrent_gated_delta_rule,
        )

        q, k, v, g, beta = self._make_inputs()

        ref_out, _ = paddle_recurrent_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        kernel_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="fused_recurrent_gated_delta_rule (l2norm) mismatch",
        )

    def test_fused_recurrent_output_final_state(self):
        """output_final_state=True 时，验证最终状态形状与数值正确。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            fused_recurrent_gated_delta_rule,
        )

        q, k, v, g, beta = self._make_inputs()

        ref_out, ref_state = paddle_recurrent_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            output_final_state=True,
        )

        kernel_out, kernel_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=True,
        )

        self.assertIsNotNone(kernel_state)
        self.assertEqual(kernel_state.shape, [self.B, self.H, self.K, self.V])

        np.testing.assert_allclose(
            kernel_state.cast(paddle.float32).numpy(),
            ref_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="fused_recurrent final state mismatch",
        )

    def test_fused_recurrent_with_initial_state(self):
        """带初始 SSM 状态，验证状态传播正确。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            fused_recurrent_gated_delta_rule,
        )

        q, k, v, g, beta = self._make_inputs()
        init_state = paddle.randn([self.B, self.H, self.K, self.V], dtype=paddle.float32)

        ref_out, ref_state = paddle_recurrent_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            initial_state=init_state.clone(),
            output_final_state=True,
        )

        kernel_out, kernel_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state.clone(),
            output_final_state=True,
        )
        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="fused_recurrent (with init state) output mismatch",
        )
        np.testing.assert_allclose(
            kernel_state.cast(paddle.float32).numpy(),
            ref_state.numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="fused_recurrent (with init state) final state mismatch",
        )


class TestChunkGDN(unittest.TestCase):
    """测试 chunk_gated_delta_rule (Prefill 路径 SSM kernel)."""

    def setUp(self):
        paddle.seed(42)
        self.dtype = paddle.bfloat16
        self.B = 1
        self.H_k, self.H_v = 4, 4  # num_k_heads, num_v_heads (no GVA for simplicity)
        self.K, self.V = 64, 64
        self.T = 128  # must be multiple of chunk_size=64

    def _make_inputs(self, T=None):
        T = T or self.T
        B, Hk, Hv, K, V = self.B, self.H_k, self.H_v, self.K, self.V
        q = paddle.randn([B, T, Hk, K], dtype=paddle.float32).cast(self.dtype)
        k = paddle.randn([B, T, Hk, K], dtype=paddle.float32).cast(self.dtype)
        v = paddle.randn([B, T, Hv, V], dtype=paddle.float32).cast(self.dtype)
        g = -F.softplus(paddle.randn([B, T, Hk], dtype=paddle.float32)).cast(self.dtype)
        beta = paddle.sigmoid(paddle.randn([B, T, Hk], dtype=paddle.float32)).cast(self.dtype)
        return q, k, v, g, beta

    def test_chunk_gdn_no_state(self):
        """不带初始状态，chunk kernel 输出应与 baseline 一致（使用 l2norm 保证数值稳定）。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import chunk_gated_delta_rule

        q, k, v, g, beta = self._make_inputs()

        ref_out, _ = paddle_chunk_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            chunk_size=64,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,  # l2norm 保证数值不溢出 bf16
        )

        kernel_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=2e-2,
            atol=2e-2,
            err_msg="chunk_gated_delta_rule (no state) mismatch",
        )

    def test_chunk_gdn_with_l2norm(self):
        """带 L2 norm，chunk kernel 输出应与 baseline 一致。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import chunk_gated_delta_rule

        q, k, v, g, beta = self._make_inputs()

        ref_out, _ = paddle_chunk_gated_delta_rule_ref(
            q.cast(paddle.float32),
            k.cast(paddle.float32),
            v.cast(paddle.float32),
            g.cast(paddle.float32),
            beta.cast(paddle.float32),
            chunk_size=64,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        kernel_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=2e-2,
            atol=2e-2,
            err_msg="chunk_gated_delta_rule (l2norm) mismatch",
        )

    def test_chunk_recurrent_consistency(self):
        """chunk 和 recurrent 在相同输入下输出应接近（数值等价性验证）。"""
        from fastdeploy.model_executor.ops.triton_ops.fla import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        # Use short T=64 for recurrent to be affordable
        q, k, v, g, beta = self._make_inputs(T=64)

        chunk_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        np.testing.assert_allclose(
            chunk_out.cast(paddle.float32).numpy(),
            recurrent_out.cast(paddle.float32).numpy(),
            rtol=2e-2,
            atol=2e-2,
            err_msg="chunk vs recurrent output mismatch",
        )


class TestCausalConv1dUpdate(unittest.TestCase):
    """测试 causal_conv1d_update (Decode 单 token conv)."""

    def setUp(self):
        paddle.seed(42)
        self.dtype = paddle.bfloat16
        self.batch = 4
        self.dim = 512  # conv_dim = key_dim * 2 + value_dim
        self.kernel_width = 4  # conv_kernel_size
        self.state_len = self.kernel_width - 1  # = 3

    def _make_inputs(self):
        batch, dim, width = self.batch, self.dim, self.kernel_width
        state_len = self.state_len
        x = paddle.randn([batch, dim], dtype=paddle.float32).cast(self.dtype)
        # conv_state pool: [max_seqs, dim, state_len]
        max_seqs = batch + 2
        conv_pool = paddle.randn([max_seqs, dim, state_len], dtype=paddle.float32).cast(self.dtype)
        weight = paddle.randn([dim, width], dtype=paddle.float32).cast(self.dtype)
        bias = paddle.randn([dim], dtype=paddle.float32).cast(self.dtype)
        # slot ids: each batch item maps to a pool slot
        slot_ids = paddle.arange(batch, dtype=paddle.int32)
        return x, conv_pool, weight, bias, slot_ids

    def _paddle_ref(self, x, conv_state_per_seq, weight, bias, activation):
        """Pure-Paddle reference (per-sequence, no pool)."""
        batch, dim = x.shape
        state_len = conv_state_per_seq.shape[-1]

        outs = []
        for i in range(batch):
            h = conv_state_per_seq[i : i + 1]  # [1, dim, state_len]
            xi = x[i : i + 1].unsqueeze(-1)  # [1, dim, 1]
            # concat and update
            combined = paddle.concat([h, xi], axis=-1)  # [1, dim, state_len+1]
            h_new = combined[:, :, -state_len:]
            conv_state_per_seq[i] = h_new[0]
            # conv1d
            w = weight.unsqueeze(1)  # [dim, 1, width]
            out = F.conv1d(combined, w, bias, padding=0, groups=dim)
            if activation in ["silu", "swish"]:
                out = F.silu(out)
            outs.append(out[:, :, -1])  # last token
        return paddle.concat(outs, axis=0)  # [batch, dim]

    def test_causal_conv1d_update_no_bias(self):
        """无 bias，causal_conv1d_update 与纯 Paddle 基准对齐。"""
        from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
            causal_conv1d_update,
        )

        x, conv_pool, weight, bias, slot_ids = self._make_inputs()

        # Extract per-seq states for reference (using slot_ids)
        ref_conv_state = conv_pool[slot_ids].clone()  # [batch, dim, state_len]

        ref_out = self._paddle_ref(
            x.cast(paddle.float32),
            ref_conv_state.cast(paddle.float32),
            weight.cast(paddle.float32),
            None,
            activation="silu",
        )

        pool_for_kernel = conv_pool.clone()
        kernel_out = causal_conv1d_update(
            x,
            pool_for_kernel,
            weight,
            bias=None,
            activation="silu",
            conv_state_indices=slot_ids,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="causal_conv1d_update (no bias) mismatch",
        )

    def test_causal_conv1d_update_with_bias(self):
        """有 bias，causal_conv1d_update 与纯 Paddle 基准对齐。"""
        from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
            causal_conv1d_update,
        )

        x, conv_pool, weight, bias, slot_ids = self._make_inputs()

        ref_conv_state = conv_pool[slot_ids].clone()
        ref_out = self._paddle_ref(
            x.cast(paddle.float32),
            ref_conv_state.cast(paddle.float32),
            weight.cast(paddle.float32),
            bias.cast(paddle.float32),
            activation="silu",
        )

        pool_for_kernel = conv_pool.clone()
        kernel_out = causal_conv1d_update(
            x,
            pool_for_kernel,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=slot_ids,
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg="causal_conv1d_update (with bias) mismatch",
        )

    def test_causal_conv1d_update_state_inplace(self):
        """验证 conv_state pool 被正确 in-place 更新（滑窗移位）。"""
        from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
            causal_conv1d_update,
        )

        x, conv_pool, weight, bias, slot_ids = self._make_inputs()

        ref_conv_state = conv_pool[slot_ids].clone()

        # Build expected new states via reference
        for i in range(self.batch):
            h = ref_conv_state[i : i + 1]
            xi = x[i : i + 1].cast(paddle.float32).unsqueeze(-1)
            combined = paddle.concat([h.cast(paddle.float32), xi], axis=-1)
            ref_conv_state[i] = combined[:, :, -self.state_len :].cast(self.dtype)[0]

        pool_for_kernel = conv_pool.clone()
        _ = causal_conv1d_update(
            x,
            pool_for_kernel,
            weight,
            activation="silu",
            conv_state_indices=slot_ids,
        )

        # Check pool slots updated correctly
        for i in range(self.batch):
            slot = slot_ids[i].item()
            np.testing.assert_allclose(
                pool_for_kernel[slot].cast(paddle.float32).numpy(),
                ref_conv_state[i].cast(paddle.float32).numpy(),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"conv_state pool slot {slot} not updated correctly",
            )


class TestCausalConv1dFn(unittest.TestCase):
    """测试 causal_conv1d_fn (Prefill varlen conv)."""

    def setUp(self):
        paddle.seed(42)
        self.dtype = paddle.bfloat16
        self.dim = 256
        self.kernel_width = 4
        self.state_len = self.kernel_width - 1

    def _make_varlen_inputs(self, seq_lens):
        """
        构造 varlen 输入。

        Returns:
            x: [dim, total_tokens] (channel-last layout)
            weight: [dim, kernel_width]
            bias: [dim,]
            conv_pool: [max_seqs, dim, state_len]
            slot_ids: [N]
            has_initial_state: [N] bool
            query_start_loc: [N+1]
            seq_lens_cpu: List[int]
        """
        dim, width, state_len = self.dim, self.kernel_width, self.state_len
        N = len(seq_lens)
        total = sum(seq_lens)
        # channel-last: (dim, total_tokens)
        x = paddle.randn([dim, total], dtype=paddle.float32).cast(self.dtype)
        weight = paddle.randn([dim, width], dtype=paddle.float32).cast(self.dtype)
        bias = paddle.randn([dim], dtype=paddle.float32).cast(self.dtype)
        max_seqs = N + 2
        conv_pool = paddle.zeros([max_seqs, dim, state_len], dtype=self.dtype)
        slot_ids = paddle.arange(N, dtype=paddle.int32)
        has_initial_state = paddle.zeros([N], dtype=paddle.bool)
        offsets = [0]
        for l in seq_lens:
            offsets.append(offsets[-1] + l)
        query_start_loc = paddle.to_tensor(offsets, dtype=paddle.int32)
        return x, weight, bias, conv_pool, slot_ids, has_initial_state, query_start_loc, seq_lens

    def _paddle_ref_prefill(self, x, weight, bias, seq_lens, activation):
        """
        Pure-Paddle reference: process each sequence independently.

        x: [dim, total_tokens]  (channel-last)
        Returns: [dim, total_tokens]
        """
        dim, width = weight.shape
        state_len = width - 1
        out_parts = []
        offset = 0
        for seqlen in seq_lens:
            x_seq = x[:, offset : offset + seqlen]  # [dim, seqlen]
            # pad left with zeros (no initial state)
            padded = F.pad(x_seq.unsqueeze(0), [state_len, 0])  # [1, dim, seqlen+state_len]
            w = weight.unsqueeze(1)  # [dim, 1, width]
            out = F.conv1d(
                padded.cast(paddle.float32),
                w.cast(paddle.float32),
                bias.cast(paddle.float32) if bias is not None else None,
                padding=0,
                groups=dim,
            )
            if activation in ["silu", "swish"]:
                out = F.silu(out)
            out_parts.append(out.squeeze(0))  # [dim, seqlen]
            offset += seqlen
        return paddle.concat(out_parts, axis=-1)  # [dim, total_tokens]

    def test_causal_conv1d_fn_no_initial_state(self):
        """无初始状态（全零）的 prefill varlen conv。"""
        from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
            causal_conv1d_fn,
        )

        seq_lens = [16, 32, 8]
        x, weight, bias, conv_pool, slot_ids, has_init, query_start_loc, seq_lens_cpu = self._make_varlen_inputs(
            seq_lens
        )

        ref_out = self._paddle_ref_prefill(
            x.cast(paddle.float32),
            weight.cast(paddle.float32),
            bias.cast(paddle.float32),
            seq_lens,
            activation="silu",
        )

        kernel_out = causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_pool,
            query_start_loc,
            seq_lens_cpu,
            cache_indices=slot_ids,
            has_initial_state=has_init,
            activation="silu",
        )

        np.testing.assert_allclose(
            kernel_out.cast(paddle.float32).numpy(),
            ref_out.numpy(),
            rtol=2e-2,
            atol=5e-2,
            err_msg="causal_conv1d_fn (no initial state) mismatch",
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    unittest.main()
