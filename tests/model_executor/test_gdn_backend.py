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
GDNAttentionBackend integration tests — shape + numerical correctness.

Tests decode / prefill (extend) / mixed mode forward pass through the full
GDN backend pipeline: conv1d → split Q,K,V → gating → SSM kernel.

Numerical correctness is verified by comparing the backend output against a
manually-constructed reference pipeline that calls the same Triton kernels
step-by-step, ensuring the backend's data orchestration (reshape, slice,
unsqueeze, GVA repeat, etc.) is correct.

Run:
  cd FastDeploy
  python -m pytest tests/model_executor/ops/triton_ops/test_gdn_backend.py -v
"""

import unittest
from enum import Enum, auto
from types import SimpleNamespace

import numpy as np
import paddle

from fastdeploy.cache_manager.gdn_state_pool import GDNStatePool
from fastdeploy.model_executor.layers.attention.gdn_backend import GDNAttentionBackend
from fastdeploy.model_executor.ops.triton_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from fastdeploy.model_executor.ops.triton_ops.fla import (
    chunk_gated_delta_rule,
    fused_gdn_gating,
    fused_recurrent_gated_delta_rule_update,
)

# ============================================================
# Mock ForwardMode
# ============================================================


class ForwardMode(Enum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED


# ============================================================
# Test dimensions (small for speed)
# ============================================================

NUM_K_HEADS = 2
NUM_V_HEADS = 4
HEAD_K_DIM = 16
HEAD_V_DIM = 16
CONV_KERNEL_SIZE = 4
NUM_V_HEADS_PER_K_HEAD = NUM_V_HEADS // NUM_K_HEADS

KEY_DIM_LOCAL = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM_LOCAL = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM_LOCAL * 2 + VALUE_DIM_LOCAL


# ============================================================
# Helpers
# ============================================================


def make_pool(max_num_seqs=8, num_layers=1):
    return GDNStatePool(
        max_num_seqs=max_num_seqs,
        num_gdn_layers=num_layers,
        conv_dim=CONV_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        num_v_heads=NUM_V_HEADS,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
    )


def make_layer():
    layer = SimpleNamespace()
    layer.gdn_layer_idx = 0
    layer.num_k_heads_local = NUM_K_HEADS
    layer.num_v_heads_local = NUM_V_HEADS
    layer.head_k_dim = HEAD_K_DIM
    layer.head_v_dim = HEAD_V_DIM
    layer.num_v_heads_per_k_head = NUM_V_HEADS_PER_K_HEAD
    layer.conv_dim = CONV_DIM
    layer.conv1d_weight = paddle.randn([CONV_DIM, CONV_KERNEL_SIZE], dtype="bfloat16")
    layer.A_log = paddle.randn([NUM_V_HEADS], dtype="float32")
    layer.dt_bias = paddle.randn([NUM_V_HEADS], dtype="float32")
    return layer


def make_meta_decode(batch_size, pool):
    raw_slot_ids = paddle.arange(0, batch_size, dtype="int32")
    meta = SimpleNamespace()
    meta.forward_mode = ForwardMode.DECODE
    meta.gdn_state_pool = pool
    meta.gdn_slot_ids = raw_slot_ids
    meta.gdn_has_initial_state = paddle.ones([batch_size], dtype="bool")
    meta.gdn_seq_lens_cpu = [1] * batch_size
    meta.cu_seqlens_q = paddle.arange(0, batch_size + 1, dtype="int32")
    return meta


def make_meta_extend(batch_size, seq_lens, pool):
    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    raw_slot_ids = paddle.arange(0, batch_size, dtype="int32")
    meta = SimpleNamespace()
    meta.forward_mode = ForwardMode.EXTEND
    meta.gdn_state_pool = pool
    meta.gdn_slot_ids = raw_slot_ids
    meta.gdn_has_initial_state = paddle.zeros([batch_size], dtype="bool")
    meta.gdn_seq_lens_cpu = seq_lens
    meta.cu_seqlens_q = paddle.to_tensor(cu, dtype="int32")
    return meta


def make_meta_mixed(num_decode, num_extend, extend_seq_lens, pool):
    batch_size = num_decode + num_extend
    all_seq_lens = [1] * num_decode + extend_seq_lens
    cu = [0]
    for sl in all_seq_lens:
        cu.append(cu[-1] + sl)
    raw_slot_ids = paddle.arange(0, batch_size, dtype="int32")
    has_initial = [True] * num_decode + [False] * num_extend
    meta = SimpleNamespace()
    meta.forward_mode = ForwardMode.MIXED
    meta.gdn_state_pool = pool
    meta.gdn_slot_ids = raw_slot_ids
    meta.gdn_has_initial_state = paddle.to_tensor(has_initial, dtype="bool")
    meta.gdn_seq_lens_cpu = all_seq_lens
    meta.cu_seqlens_q = paddle.to_tensor(cu, dtype="int32")
    return meta


# ============================================================
# Reference pipeline: manually call the same Triton kernels
# ============================================================


def ref_decode_pipeline(mixed_qkv, a, b, layer, conv_pool, ssm_pool, slot_ids):
    """Manual decode pipeline using Triton kernels directly."""
    conv_weight = layer.conv1d_weight[: layer.conv_dim]

    # 1. Conv1d update
    x = causal_conv1d_update(
        x=mixed_qkv,
        conv_state=conv_pool,
        weight=conv_weight,
        bias=None,
        activation="silu",
        conv_state_indices=slot_ids,
    )

    # 2. Split Q, K, V
    num_tokens = x.shape[0]
    q, k, v = paddle.split(
        x,
        [KEY_DIM_LOCAL, KEY_DIM_LOCAL, VALUE_DIM_LOCAL],
        axis=-1,
    )
    q = q.reshape([num_tokens, NUM_K_HEADS, HEAD_K_DIM])
    k = k.reshape([num_tokens, NUM_K_HEADS, HEAD_K_DIM])
    v = v.reshape([num_tokens, NUM_V_HEADS, HEAD_V_DIM])

    # 3. Gating
    g, beta = fused_gdn_gating(
        layer.A_log[:NUM_V_HEADS],
        a[:, :NUM_V_HEADS],
        b[:, :NUM_V_HEADS],
        layer.dt_bias[:NUM_V_HEADS],
    )

    # 4. GVA repeat
    if NUM_V_HEADS_PER_K_HEAD > 1:
        q = (
            q.unsqueeze(2)
            .expand([num_tokens, NUM_K_HEADS, NUM_V_HEADS_PER_K_HEAD, HEAD_K_DIM])
            .reshape([num_tokens, NUM_V_HEADS, HEAD_K_DIM])
        )
        k = (
            k.unsqueeze(2)
            .expand([num_tokens, NUM_K_HEADS, NUM_V_HEADS_PER_K_HEAD, HEAD_K_DIM])
            .reshape([num_tokens, NUM_V_HEADS, HEAD_K_DIM])
        )

    # 5. SSM kernel (decode)
    q_4d = q.unsqueeze(1)
    k_4d = k.unsqueeze(1)
    v_4d = v.unsqueeze(1)
    g_4d = g.unsqueeze(1)
    beta_4d = beta.unsqueeze(1)

    o = fused_recurrent_gated_delta_rule_update(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g_4d,
        beta=beta_4d,
        ssm_pool=ssm_pool,
        ssm_indices=slot_ids,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(1)  # [batch, H, V]


def ref_extend_pipeline(
    mixed_qkv, a, b, layer, conv_pool, ssm_pool, slot_ids, cu_seqlens, seq_lens_cpu, has_initial_state
):
    """Manual extend pipeline using Triton kernels directly."""
    conv_weight = layer.conv1d_weight[: layer.conv_dim]

    # 1. Conv1d fn (varlen)
    x = causal_conv1d_fn(
        x=mixed_qkv.T,
        weight=conv_weight,
        bias=None,
        conv_states=conv_pool,
        query_start_loc=cu_seqlens,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=slot_ids,
        has_initial_state=has_initial_state,
        activation="silu",
    ).T

    # 2. Split Q, K, V
    num_tokens = x.shape[0]
    q, k, v = paddle.split(
        x,
        [KEY_DIM_LOCAL, KEY_DIM_LOCAL, VALUE_DIM_LOCAL],
        axis=-1,
    )
    q = q.reshape([num_tokens, NUM_K_HEADS, HEAD_K_DIM])
    k = k.reshape([num_tokens, NUM_K_HEADS, HEAD_K_DIM])
    v = v.reshape([num_tokens, NUM_V_HEADS, HEAD_V_DIM])

    # 3. Gating
    g, beta = fused_gdn_gating(
        layer.A_log[:NUM_V_HEADS],
        a[:, :NUM_V_HEADS],
        b[:, :NUM_V_HEADS],
        layer.dt_bias[:NUM_V_HEADS],
    )

    # 4. GVA repeat
    if NUM_V_HEADS_PER_K_HEAD > 1:
        q = (
            q.unsqueeze(2)
            .expand([num_tokens, NUM_K_HEADS, NUM_V_HEADS_PER_K_HEAD, HEAD_K_DIM])
            .reshape([num_tokens, NUM_V_HEADS, HEAD_K_DIM])
        )
        k = (
            k.unsqueeze(2)
            .expand([num_tokens, NUM_K_HEADS, NUM_V_HEADS_PER_K_HEAD, HEAD_K_DIM])
            .reshape([num_tokens, NUM_V_HEADS, HEAD_K_DIM])
        )

    # 5. SSM kernel (chunk)
    batch_size = slot_ids.shape[0]
    q_4d = q.unsqueeze(0)
    k_4d = k.unsqueeze(0)
    v_4d = v.unsqueeze(0)
    g_4d = g.unsqueeze(0)
    beta_4d = beta.unsqueeze(0)

    initial_state = ssm_pool[slot_ids].clone()
    o, _h = chunk_gated_delta_rule(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g_4d,
        beta=beta_4d,
        scale=None,
        initial_state=initial_state,
        initial_state_indices=paddle.arange(batch_size, dtype=paddle.int32),
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    # Write back states
    for i in range(batch_size):
        sid = int(slot_ids[i])
        if sid > 0:
            ssm_pool[sid] = initial_state[i]

    return o.squeeze(0)  # [total_tokens, H, V]


# ============================================================
# Test Cases
# ============================================================


class TestGDNBackendDecodeNumerical(unittest.TestCase):
    """Decode: backend vs manual reference — numerical match."""

    def test_decode_numerical(self):
        batch_size = 3
        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()
        meta = make_meta_decode(batch_size, pool)

        mixed_qkv = paddle.randn([batch_size, CONV_DIM], dtype="bfloat16")
        a = paddle.randn([batch_size, NUM_V_HEADS], dtype="bfloat16")
        b = paddle.randn([batch_size, NUM_V_HEADS], dtype="bfloat16")

        # Clone pool + inputs so both paths see the same initial state
        pool_ref = make_pool(max_num_seqs=8)
        slot_ids = GDNStatePool.offset_slot_ids(meta.gdn_slot_ids)

        # Reference
        ref_out = ref_decode_pipeline(
            mixed_qkv.clone(),
            a.clone(),
            b.clone(),
            layer,
            pool_ref.get_layer_conv_pool(0),
            pool_ref.get_layer_ssm_pool(0),
            slot_ids,
        )

        # Backend
        out = backend.forward(mixed_qkv.clone(), a.clone(), b.clone(), layer, meta)

        self.assertEqual(out.shape, [batch_size, NUM_V_HEADS, HEAD_V_DIM])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


class TestGDNBackendExtendNumerical(unittest.TestCase):
    """Extend: backend vs manual reference — numerical match."""

    def test_extend_single_seq(self):
        seq_lens = [10]
        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()
        meta = make_meta_extend(1, seq_lens, pool)

        total = sum(seq_lens)
        mixed_qkv = paddle.randn([total, CONV_DIM], dtype="bfloat16")
        a = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")
        b = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")

        pool_ref = make_pool(max_num_seqs=8)
        slot_ids = GDNStatePool.offset_slot_ids(meta.gdn_slot_ids)

        ref_out = ref_extend_pipeline(
            mixed_qkv.clone(),
            a.clone(),
            b.clone(),
            layer,
            pool_ref.get_layer_conv_pool(0),
            pool_ref.get_layer_ssm_pool(0),
            slot_ids,
            meta.cu_seqlens_q.clone(),
            list(meta.gdn_seq_lens_cpu),
            meta.gdn_has_initial_state.clone(),
        )

        out = backend.forward(mixed_qkv.clone(), a.clone(), b.clone(), layer, meta)

        self.assertEqual(out.shape, [total, NUM_V_HEADS, HEAD_V_DIM])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_extend_multi_seq(self):
        seq_lens = [5, 8, 3]
        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()
        meta = make_meta_extend(3, seq_lens, pool)

        total = sum(seq_lens)
        mixed_qkv = paddle.randn([total, CONV_DIM], dtype="bfloat16")
        a = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")
        b = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")

        pool_ref = make_pool(max_num_seqs=8)
        slot_ids = GDNStatePool.offset_slot_ids(meta.gdn_slot_ids)

        ref_out = ref_extend_pipeline(
            mixed_qkv.clone(),
            a.clone(),
            b.clone(),
            layer,
            pool_ref.get_layer_conv_pool(0),
            pool_ref.get_layer_ssm_pool(0),
            slot_ids,
            meta.cu_seqlens_q.clone(),
            list(meta.gdn_seq_lens_cpu),
            meta.gdn_has_initial_state.clone(),
        )

        out = backend.forward(mixed_qkv.clone(), a.clone(), b.clone(), layer, meta)

        self.assertEqual(out.shape, [total, NUM_V_HEADS, HEAD_V_DIM])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


class TestGDNBackendMixedNumerical(unittest.TestCase):
    """Mixed mode: backend vs manual reference — numerical match."""

    def test_mixed_numerical(self):
        """2 decode (seqlen=1) + 1 extend (seqlen=6), all through extend path."""
        num_decode = 2
        num_extend = 1
        extend_seq_lens = [6]

        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()
        meta = make_meta_mixed(num_decode, num_extend, extend_seq_lens, pool)

        total = num_decode + sum(extend_seq_lens)
        mixed_qkv = paddle.randn([total, CONV_DIM], dtype="bfloat16")
        a = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")
        b = paddle.randn([total, NUM_V_HEADS], dtype="bfloat16")

        pool_ref = make_pool(max_num_seqs=8)
        slot_ids = GDNStatePool.offset_slot_ids(meta.gdn_slot_ids)

        ref_out = ref_extend_pipeline(
            mixed_qkv.clone(),
            a.clone(),
            b.clone(),
            layer,
            pool_ref.get_layer_conv_pool(0),
            pool_ref.get_layer_ssm_pool(0),
            slot_ids,
            meta.cu_seqlens_q.clone(),
            list(meta.gdn_seq_lens_cpu),
            meta.gdn_has_initial_state.clone(),
        )
        out = backend.forward(mixed_qkv.clone(), a.clone(), b.clone(), layer, meta)

        self.assertEqual(out.shape, [total, NUM_V_HEADS, HEAD_V_DIM])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


class TestGDNBackendStateUpdate(unittest.TestCase):
    """Verify SSM/conv states persist across prefill → decode."""

    def test_prefill_then_decode_state_persists(self):
        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()

        # Step 1: Prefill seqlen=5
        meta_p = make_meta_extend(1, [5], pool)
        mixed_qkv = paddle.randn([5, CONV_DIM], dtype="bfloat16")
        a = paddle.randn([5, NUM_V_HEADS], dtype="bfloat16")
        b = paddle.randn([5, NUM_V_HEADS], dtype="bfloat16")
        out1 = backend.forward(mixed_qkv, a, b, layer, meta_p)
        self.assertEqual(out1.shape, [5, NUM_V_HEADS, HEAD_V_DIM])

        # SSM state should be non-zero after prefill (slot 0 → offset 1)
        ssm_state = pool.get_layer_ssm_pool(0)[1].numpy()
        self.assertFalse((ssm_state == 0).all(), "SSM state should be non-zero after prefill")

        # Conv state should be non-zero after prefill
        conv_state = pool.get_layer_conv_pool(0)[1].cast("float32").numpy()
        self.assertFalse((conv_state == 0).all(), "Conv state should be non-zero after prefill")

        # Step 2: Decode 1 token (same slot)
        meta_d = make_meta_decode(1, pool)
        meta_d.gdn_has_initial_state = paddle.to_tensor([True], dtype="bool")
        out2 = backend.forward(
            paddle.randn([1, CONV_DIM], dtype="bfloat16"),
            paddle.randn([1, NUM_V_HEADS], dtype="bfloat16"),
            paddle.randn([1, NUM_V_HEADS], dtype="bfloat16"),
            layer,
            meta_d,
        )
        self.assertEqual(out2.shape, [1, NUM_V_HEADS, HEAD_V_DIM])

        # SSM state should have changed after decode step
        ssm_state_after = pool.get_layer_ssm_pool(0)[1].numpy()
        self.assertFalse(
            np.allclose(ssm_state, ssm_state_after, atol=1e-10), "SSM state should change after decode step"
        )

    def test_output_not_all_zeros(self):
        """Sanity: output should not be trivially zero."""
        pool = make_pool(max_num_seqs=8)
        layer = make_layer()
        backend = GDNAttentionBackend()
        meta = make_meta_extend(1, [8], pool)

        out = backend.forward(
            paddle.randn([8, CONV_DIM], dtype="bfloat16"),
            paddle.randn([8, NUM_V_HEADS], dtype="bfloat16"),
            paddle.randn([8, NUM_V_HEADS], dtype="bfloat16"),
            layer,
            meta,
        )
        self.assertFalse(
            (out.cast("float32").numpy() == 0).all(),
            "Backend output should not be all zeros",
        )


if __name__ == "__main__":
    unittest.main()
