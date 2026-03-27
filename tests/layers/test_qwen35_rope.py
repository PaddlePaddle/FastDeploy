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
Unit tests for Qwen3.5 RoPE support:
  1. QwenRotaryEmbedding — partial_rotary_factor < 1 and mrope_section
  2. gqa_rope_write_cache — neox partial rotary path with head_dim=256
"""

import unittest

import numpy as np
import paddle
from fastdeploy.model_executor.layers.rotary_embedding import QwenRotaryEmbedding

paddle.set_default_dtype("float16")
seed = 42
np.random.seed(seed)
paddle.seed(seed)

def rotate_half(x):
    """Rotates half the hidden dims of the input. Supports both numpy and paddle Tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    if isinstance(x, np.ndarray):
        return np.concatenate([-x2, x1], axis=-1)
    return paddle.concat([-x2, x1], axis=-1)

def apply_neox_partial_rope_ref(q, k, cos, sin):
    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    if isinstance(q_embed, np.ndarray):
        q_embed = np.concatenate([q_embed, q_pass], axis=-1)
        k_embed = np.concatenate([k_embed, k_pass], axis=-1)
    else:
        q_embed = paddle.concat([q_embed, q_pass], axis=-1)
        k_embed = paddle.concat([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Tests for QwenRotaryEmbedding
# ---------------------------------------------------------------------------

class TestQwenRotaryEmbedding(unittest.TestCase):
    """Tests for QwenRotaryEmbedding with partial_rotary_factor and mrope_section."""

    def setUp(self):
        paddle.set_device("gpu")
        self.QwenRotaryEmbedding = QwenRotaryEmbedding

    # ------------------------------------------------------------------
    # partial_rotary_factor < 1 (Qwen3.5 text)
    # ------------------------------------------------------------------

    def test_partial_rotary_dim(self):
        """rotary_dim is correctly scaled by partial_rotary_factor."""
        emb = self.QwenRotaryEmbedding(rotary_dim=256, base=500000.0, partial_rotary_factor=0.25)
        self.assertEqual(emb.rotary_dim, 64)

    def test_full_rotary_dim(self):
        """partial_rotary_factor=1.0 leaves rotary_dim unchanged (Qwen3)."""
        emb = self.QwenRotaryEmbedding(rotary_dim=128, base=10000.0, partial_rotary_factor=1.0)
        self.assertEqual(emb.rotary_dim, 128)

    def test_output_shape_partial(self):
        """Output shape is (2, bsz, seq_len, 1, rotary_dim) for partial rotary."""
        rotary_dim = 256
        partial_rotary_factor = 0.25
        expected_rot_dim = int(rotary_dim * partial_rotary_factor)  # 64
        bsz, seq_len = 2, 16
        position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([bsz, -1])
        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=500000.0,
                                       partial_rotary_factor=partial_rotary_factor)
        rot_emb = emb(position_ids)
        self.assertEqual(list(rot_emb.shape), [2, bsz, seq_len, 1, expected_rot_dim])

    def test_output_shape_full(self):
        """Output shape is (2, bsz, seq_len, 1, rotary_dim) for full rotary."""
        rotary_dim = 128
        bsz, seq_len = 1, 32
        position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([bsz, -1])
        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0, partial_rotary_factor=1.0)
        rot_emb = emb(position_ids)
        self.assertEqual(list(rot_emb.shape), [2, bsz, seq_len, 1, rotary_dim])

    def test_cos_sin_values_partial(self):
        """cos/sin values match numpy reference for partial rotary case."""
        rotary_dim = 256
        partial_rotary_factor = 0.25
        actual_rot_dim = int(rotary_dim * partial_rotary_factor)  # 64
        base = 500000.0
        bsz, seq_len = 1, 8

        position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0)  # [1, S]
        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=base,
                                       partial_rotary_factor=partial_rotary_factor)
        rot_emb = emb(position_ids)  # [2, 1, S, 1, 64]

        # Reference via numpy
        half = actual_rot_dim // 2
        inv_freq = base ** (-np.arange(0, actual_rot_dim, 2, dtype="float32") / actual_rot_dim)
        positions = np.arange(seq_len, dtype="float32")
        freqs = np.outer(positions, inv_freq)                           # [S, half]
        emb_np = np.concatenate([freqs, freqs], axis=-1)               # [S, rot_dim]
        cos_ref = np.cos(emb_np)[np.newaxis, :, np.newaxis, :]         # [1, S, 1, rot_dim]
        sin_ref = np.sin(emb_np)[np.newaxis, :, np.newaxis, :]

        np.testing.assert_allclose(
            rot_emb[0].numpy(), cos_ref, rtol=1e-5, atol=1e-5,
            err_msg="cos values mismatch for partial rotary"
        )
        np.testing.assert_allclose(
            rot_emb[1].numpy(), sin_ref, rtol=1e-5, atol=1e-5,
            err_msg="sin values mismatch for partial rotary"
        )

    # ------------------------------------------------------------------
    # mrope_section (Qwen3.5-VL multi-modal RoPE)
    # ------------------------------------------------------------------

    def test_mrope_section_output_shape(self):
        """mrope_section path outputs (2, bsz, seq_len, 1, actual_rotary_dim)."""
        rotary_dim = 256
        partial_rotary_factor = 0.25
        mrope_section = [10, 11, 11]          # sum == rotary_dim // 2 == 32
        bsz, seq_len = 2, 10
        actual_rotary_dim = int(rotary_dim * partial_rotary_factor)  # 64

        # position_ids shape: (3, bsz, seq_len) — one per modal dimension
        position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.expand([3, bsz, seq_len])

        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0,
                                       partial_rotary_factor=partial_rotary_factor,
                                       mrope_section=mrope_section)
        rot_emb = emb(position_ids)
        self.assertEqual(list(rot_emb.shape), [2, bsz, seq_len, 1, actual_rotary_dim])

    def test_mrope_2d_position_ids_broadcast(self):
        """2D position_ids (bsz, seq_len) are automatically broadcast to (3, bsz, seq_len)."""
        rotary_dim = 256
        partial_rotary_factor = 0.25
        mrope_section = [10, 11, 11]
        bsz, seq_len = 1, 6
        actual_rotary_dim = int(rotary_dim * partial_rotary_factor)  # 64

        position_ids_2d = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([bsz, -1])
        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0,
                                       partial_rotary_factor=partial_rotary_factor,
                                       mrope_section=mrope_section)
        rot_emb = emb(position_ids_2d)
        self.assertEqual(list(rot_emb.shape), [2, bsz, seq_len, 1, actual_rotary_dim])

    def test_apply_interleaved_mrope_sum_check(self):
        """apply_interleaved_mrope raises on mismatched mrope_section sum."""
        rotary_dim = 128
        bad_mrope_section = [10, 10, 10]   # sum=30 != 64
        emb = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0,
                                       partial_rotary_factor=1.0,
                                       mrope_section=bad_mrope_section)
        dummy_freqs = paddle.zeros([3, 1, 4, rotary_dim // 2])
        with self.assertRaises(AssertionError):
            emb.apply_interleaved_mrope(dummy_freqs)

    def test_mrope_no_section_unchanged(self):
        """Without mrope_section, output matches the standard (non-mrope) path."""
        rotary_dim = 128
        bsz, seq_len = 1, 8
        position_ids = paddle.arange(seq_len, dtype="int64").unsqueeze(0).expand([bsz, -1])

        emb_std = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0,
                                           partial_rotary_factor=1.0, mrope_section=None)
        emb_mrope = self.QwenRotaryEmbedding(rotary_dim=rotary_dim, base=10000.0,
                                             partial_rotary_factor=1.0,
                                             mrope_section=[32, 16, 16])
        rot_std = emb_std(position_ids)

        # For mrope: provide identical positions in all 3 dimensions → result equals std
        position_ids_3d = position_ids.unsqueeze(0).expand([3, -1, -1])
        rot_mrope = emb_mrope(position_ids_3d)
        # With identical T/H/W positions, interleaved mrope == standard embedding
        np.testing.assert_allclose(
            rot_std.numpy(), rot_mrope.numpy(), rtol=1e-5, atol=1e-5,
            err_msg="mrope with uniform positions should match standard embedding"
        )


# ---------------------------------------------------------------------------
# Tests for gqa_rope_write_cache — Qwen3.5 partial neox path (head_dim=256)
# ---------------------------------------------------------------------------
def _build_rotary_emb(max_seq_len, head_dim=256, base=100000.0):
    if head_dim==256:
        partial_rotary_factor = 0.25
        mrope_section = [10, 11, 11]
    else:
        partial_rotary_factor = 1.0
        mrope_section = None
    rot_emb = QwenRotaryEmbedding(rotary_dim=head_dim, base=base, partial_rotary_factor=partial_rotary_factor, mrope_section=mrope_section)
    pos_ids = paddle.arange(max_seq_len, dtype="int64").unsqueeze(0)
    return rot_emb(pos_ids)
    
def _build_gqa_rope_write_cache_inputs(
    bsz, q_num_head, kv_num_head, seq_len,
    head_dim, blocksize, max_seq_len, dtype="bfloat16"):
    """Build all tensors needed by gqa_rope_write_cache for a pure-prefill batch."""
    token_num = bsz * seq_len

    # QKV: [token_num, (q + 2*kv) * head_dim]
    qkv_dim = (q_num_head + 2 * kv_num_head) * head_dim
    qkv = paddle.randn([token_num, qkv_dim], dtype=dtype) * 0.02

    # Rotary embedding: shape (2, 1, max_seq_len, 1, rotary_dim)
    # (matches what QwenRotaryEmbedding produces for partial rotary)
    rot_emb = _build_rotary_emb(max_seq_len, head_dim, base=500000.0)
    # gqa_rope_write_cache expects shape (2, 1, max_seq_len, 1, rotary_dim)
    rotary_embs = rot_emb  # already [2, 1, S, 1, rotary_dim]
    rotary_dim = rot_emb.shape[-1]

    # Block tables: each batch gets one block
    num_blocks = bsz
    block_tables = paddle.zeros([bsz, max_seq_len // blocksize + 1], dtype="int32")
    for i in range(bsz):
        block_tables[i, 0] = i

    # KV cache
    key_cache   = paddle.zeros([num_blocks, kv_num_head, blocksize, head_dim], dtype=dtype)
    value_cache = paddle.zeros([num_blocks, kv_num_head, blocksize, head_dim], dtype=dtype)

    # Sequence metadata (pure prefill: encoder only)
    seq_lens_this_time = paddle.to_tensor([seq_len] * bsz, dtype="int32")
    seq_lens_encoder   = paddle.to_tensor([seq_len] * bsz, dtype="int32")
    seq_lens_decoder   = paddle.zeros([bsz], dtype="int32")

    # batch_id_per_token, cu_seqlens_q/k
    batch_id_per_token = paddle.zeros([token_num], dtype="int32")
    for i in range(bsz):
        batch_id_per_token[i * seq_len : (i + 1) * seq_len] = i

    cu_seqlens_q = paddle.zeros([bsz + 1], dtype="int32")
    cu_seqlens_k = paddle.zeros([bsz + 1], dtype="int32")
    for i in range(bsz):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seq_lens_this_time[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seq_lens_this_time[i]

    kv_token_num = int(cu_seqlens_k[-1])

    # Tile/batch ids for KV cache write (minimal: one tile per seq)
    kv_batch_ids        = paddle.zeros([bsz], dtype="int32")
    kv_tile_ids         = paddle.zeros([bsz], dtype="int32")
    kv_num_blocks       = paddle.to_tensor([1] * bsz, dtype="int32")
    cache_batch_ids     = paddle.zeros([bsz], dtype="int32")
    cache_tile_ids      = paddle.zeros([bsz], dtype="int32")
    cache_num_blocks    = paddle.to_tensor([1] * bsz, dtype="int32")

    return dict(
        qkv=qkv, key_cache=key_cache, value_cache=value_cache,
        rotary_embs=rotary_embs,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        seq_lens_this_time=seq_lens_this_time,
        seq_lens_encoder=seq_lens_encoder, seq_lens_decoder=seq_lens_decoder,
        batch_id_per_token=batch_id_per_token,
        block_tables=block_tables,
        kv_batch_ids=kv_batch_ids, kv_tile_ids_per_batch=kv_tile_ids,
        kv_num_blocks=kv_num_blocks,
        cache_batch_ids=cache_batch_ids, cache_tile_ids_per_batch=cache_tile_ids,
        cache_num_blocks=cache_num_blocks,
        kv_token_num=kv_token_num,
        head_dim=head_dim, rotary_dim=rotary_dim,
        q_num_head=q_num_head, kv_num_head=kv_num_head,
        token_num=token_num, seq_len=seq_len,
    )


class TestGqaRopeWriteCacheQwen35(unittest.TestCase):
    """Tests for gqa_rope_write_cache — Qwen3.5 neox partial rotary (head_dim=256)."""

    def setUp(self):
        paddle.set_device("gpu")
        from fastdeploy.model_executor.layers.attention.ops import gqa_rope_write_cache
        self.gqa_rope_write_cache = gqa_rope_write_cache

    def _run(self, inputs, max_seq_len):
        q, k, v, qkv_out = self.gqa_rope_write_cache(
            inputs["qkv"],
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            inputs["rotary_embs"],
            inputs["seq_lens_this_time"],
            inputs["seq_lens_encoder"],
            inputs["seq_lens_decoder"],
            inputs["batch_id_per_token"],
            inputs["block_tables"],
            inputs["kv_batch_ids"],
            inputs["kv_tile_ids_per_batch"],
            inputs["kv_num_blocks"],
            inputs["cache_batch_ids"],
            inputs["cache_tile_ids_per_batch"],
            inputs["cache_num_blocks"],
            kv_token_num=inputs["kv_token_num"],
            max_seq_len=max_seq_len,
            use_neox_rotary_style=True,
            cache_quant_type="none",
        )
        return q, k, v, qkv_out

    # ------------------------------------------------------------------
    # Output shape tests
    # ------------------------------------------------------------------

    def test_output_shapes_qwen35(self):
        """q/k/v/qkv_out have the expected shapes for Qwen3.5 (head_dim=256)."""
        bsz, q_num_head, kv_num_head = 1, 4, 2
        seq_len, head_dim = 8, 256
        blocksize, max_seq_len = 64, 128
        dtype = "float16"

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, dtype
        )
        q, k, v, qkv_out = self._run(inputs, max_seq_len)

        token_num = inputs["token_num"]
        kv_token_num = inputs["kv_token_num"]
        self.assertEqual(list(q.shape),     [token_num, q_num_head, head_dim])
        self.assertEqual(list(k.shape),     [kv_token_num, kv_num_head, head_dim])
        self.assertEqual(list(v.shape),     [kv_token_num, kv_num_head, head_dim])
        self.assertEqual(list(qkv_out.shape), list(inputs["qkv"].shape))

    def test_output_shapes_bfloat16(self):
        """Shapes are correct with bfloat16 dtype."""
        bsz, q_num_head, kv_num_head = 2, 4, 1
        seq_len, head_dim = 16, 256
        blocksize, max_seq_len = 64, 128

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, dtype="bfloat16"
        )
        q, k, v, qkv_out = self._run(inputs, max_seq_len)
        self.assertEqual(q.dtype, paddle.bfloat16)

    # ------------------------------------------------------------------
    # Correctness: RoPE output matches reference
    # ------------------------------------------------------------------

    def test_neox_partial_rope_correctness(self):
        """Rotated Q matches Python reference rotate_half on [0, rotary_dim)."""
        bsz, q_num_head, kv_num_head = 1, 2, 1
        seq_len, head_dim = 4, 256
        blocksize, max_seq_len = 64, 128
        dtype = "float16"

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, dtype
        )

        # Extract original Q before rotation: [token_num, q_num_head, head_dim]
        qkv_np = inputs["qkv"].cast("float32").numpy()
        q_orig = qkv_np.reshape(seq_len, q_num_head + 2 * kv_num_head, head_dim)[:, :q_num_head, :]
        k_orig = qkv_np.reshape(seq_len, q_num_head + 2 * kv_num_head, head_dim)[:, q_num_head:q_num_head + kv_num_head, :]

        q, k, v, qkv_out = self._run(inputs, max_seq_len)
        q_np = q.cast("float32").numpy()  # [token_num, q_num_head, head_dim]
        k_np = k.cast("float32").numpy()  # [token_num, q_num_head, head_dim]

        # Build reference cos/sin for positions 0..seq_len-1
        rot_emb_np = inputs["rotary_embs"].numpy()  # [2, 1, max_seq_len, 1, rotary_dim]
        cos_np = rot_emb_np[0, 0, :seq_len, :, :]  # [seq_len, rotary_dim]
        sin_np = rot_emb_np[1, 0, :seq_len, :, :]

        q_ref = q_orig.copy()
        k_ref = k_orig.copy()
        q_ref, k_ref = apply_neox_partial_rope_ref(q_ref, k_ref, cos_np, sin_np)

        np.testing.assert_allclose(q_np, q_ref, rtol=1e-2, atol=1e-2,
                                   err_msg="Q RoPE output mismatch vs reference")
        np.testing.assert_allclose(k_np, k_ref, rtol=1e-2, atol=1e-2,
                                   err_msg="K RoPE output mismatch vs reference")

    def test_v_passthrough(self):
        """V is not rotated — output V matches raw V from QKV input."""
        bsz, q_num_head, kv_num_head = 1, 2, 1
        seq_len, head_dim = 4, 256
        blocksize, max_seq_len = 64, 128
        dtype = "float16"

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, dtype
        )

        # Extract V directly from packed QKV
        qkv_np = inputs["qkv"].cast("float32").numpy()
        v_orig = qkv_np.reshape(seq_len, q_num_head + 2 * kv_num_head, head_dim)[
            :, q_num_head + kv_num_head :, :
        ]  # [seq_len, kv_num_head, head_dim]

        _, _, v, _ = self._run(inputs, max_seq_len)
        v_np = v.cast("float32").numpy()

        np.testing.assert_allclose(v_np, v_orig, rtol=1e-3, atol=1e-3,
                                   err_msg="V should be unchanged (no rotation)")

    def test_passthrough_region_unchanged(self):
        """head_dim[rotary_dim:] of Q is not modified by partial RoPE."""
        bsz, q_num_head, kv_num_head = 1, 2, 1
        seq_len, head_dim = 4, 256
        blocksize, max_seq_len = 64, 128
        dtype = "float16"

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, dtype
        )
        rotary_dim = inputs["rotary_dim"]
        qkv_np = inputs["qkv"].cast("float32").numpy()
        q_orig_pass = qkv_np.reshape(seq_len, q_num_head + 2 * kv_num_head, head_dim)[
            :, :q_num_head, rotary_dim:
        ]  # [S, H, head_dim - rotary_dim]

        q, _, _, _ = self._run(inputs, max_seq_len)
        q_pass = q.cast("float32").numpy()[:, :, rotary_dim:]

        np.testing.assert_allclose(q_pass, q_orig_pass, rtol=1e-3, atol=1e-3,
                                   err_msg="Pass-through region [rotary_dim:] should be unchanged")

    # ------------------------------------------------------------------
    # Multi-batch test
    # ------------------------------------------------------------------

    def test_multi_batch(self):
        """Kernel produces correct shapes for bsz > 1."""
        bsz, q_num_head, kv_num_head = 3, 4, 2
        seq_len, head_dim = 8, 256
        blocksize, max_seq_len = 64, 128

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, "float16"
        )
        q, k, v, qkv_out = self._run(inputs, max_seq_len)
        token_num = bsz * seq_len
        self.assertEqual(list(q.shape), [token_num, q_num_head, head_dim])
        self.assertEqual(list(k.shape), [token_num, kv_num_head, head_dim])

    # ------------------------------------------------------------------
    # Qwen3 baseline: head_dim=128, full rotary still works
    # ------------------------------------------------------------------

    def test_qwen3_head_dim_128(self):
        """gqa_rope_write_cache still handles Qwen3 head_dim=128 correctly."""
        bsz, q_num_head, kv_num_head = 1, 8, 2
        seq_len, head_dim = 8, 128
        blocksize, max_seq_len = 64, 128

        inputs = _build_gqa_rope_write_cache_inputs(
            bsz, q_num_head, kv_num_head, seq_len,
            head_dim, blocksize, max_seq_len, "float16"
        )
        q, k, v, _ = self._run(inputs, max_seq_len)
        self.assertEqual(list(q.shape), [seq_len, q_num_head, head_dim])
        self.assertEqual(list(k.shape), [seq_len, kv_num_head, head_dim])


if __name__ == "__main__":
    unittest.main()
