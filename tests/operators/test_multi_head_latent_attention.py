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

import math
import unittest

import numpy as np
import paddle

try:
    from fastdeploy.model_executor.ops.gpu import multi_head_latent_attention
except (ImportError, AttributeError):
    multi_head_latent_attention = None


def _get_sm_version():
    """Get CUDA SM version (e.g., 90 for H100, 80 for A100)."""
    try:
        if not paddle.device.is_compiled_with_cuda():
            return 0
        if paddle.device.cuda.device_count() == 0:
            return 0
        prop = paddle.device.cuda.get_device_properties()
        return prop.major * 10 + prop.minor
    except Exception:
        return 0


SM_VERSION = _get_sm_version()
HAS_CUDA = SM_VERSION > 0
OP_AVAILABLE = multi_head_latent_attention is not None


def _reference_mla_decode(
    query_np,
    kv_cache_np,
    block_tables_np,
    seq_len,
    q_num_heads,
    kv_num_heads,
    head_dim_qk,
    head_dim_v,
    block_size,
    softmax_scale,
):
    """
    NumPy reference implementation for MLA decode attention.

    Computes: output = softmax(Q @ K^T / sqrt(d)) @ V
    with paged KV cache and GQA (Grouped Query Attention) support.

    Args:
        query_np: [token_num, q_num_heads * head_dim_qk] float32
        kv_cache_np: [total_blocks, kv_num_heads, block_size, head_dim_qk] float32
        block_tables_np: [batch_size, max_blocks_per_seq] int32
        seq_len: Number of tokens in KV cache
        q_num_heads: Number of query heads
        kv_num_heads: Number of KV heads
        head_dim_qk: Query/Key dimension
        head_dim_v: Value dimension (nope_size)
        block_size: Tokens per cache block
        softmax_scale: Attention scaling factor

    Returns:
        [token_num, q_num_heads * head_dim_v] float32
    """
    token_num = query_np.shape[0]
    q = query_np.reshape(token_num, q_num_heads, head_dim_qk).astype(np.float64)

    # Gather K, V from paged cache
    keys = []
    values = []
    for pos in range(seq_len):
        block_idx = pos // block_size
        offset = pos % block_size
        actual_block = block_tables_np[0, block_idx]
        keys.append(kv_cache_np[actual_block, :, offset, :head_dim_qk])
        values.append(kv_cache_np[actual_block, :, offset, :head_dim_v])

    k = np.stack(keys, axis=0).astype(np.float64)  # [seq_len, kv_num_heads, head_dim_qk]
    v = np.stack(values, axis=0).astype(np.float64)  # [seq_len, kv_num_heads, head_dim_v]

    # GQA: repeat KV heads to match query heads
    repeats = q_num_heads // kv_num_heads
    k = np.repeat(k, repeats, axis=1)  # [seq_len, q_num_heads, head_dim_qk]
    v = np.repeat(v, repeats, axis=1)  # [seq_len, q_num_heads, head_dim_v]

    # Per-head attention
    out = np.zeros((token_num, q_num_heads, head_dim_v), dtype=np.float64)
    for t in range(token_num):
        for h in range(q_num_heads):
            # Attention scores: [1, seq_len]
            scores = q[t : t + 1, h, :] @ k[:, h, :].T * softmax_scale
            # Stable softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            probs = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
            # Weighted sum of values
            out[t, h, :] = probs @ v[:, h, :]

    return out.reshape(token_num, q_num_heads * head_dim_v).astype(np.float32)


@unittest.skipUnless(
    HAS_CUDA and OP_AVAILABLE,
    "Requires CUDA GPU and compiled FastDeploy custom ops.",
)
class TestMultiHeadLatentAttention(unittest.TestCase):
    """
    Unit tests for the multi_head_latent_attention custom operator.

    This op implements Multi-head Latent Attention (MLA) from DeepSeek V2/V3
    with paged KV cache support. The tensorcore kernel path requires SM >= 90
    (NVIDIA H100 / H200).

    Test categories covered:
        A — Numerical correctness (bf16, fp16)
        B — Output shape validation
        D — Edge case (zero-length decode)
        E — Determinism
        F — Error handling (unsupported dtype, SM requirement)
    """

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.sm_version = SM_VERSION

        # Minimal MLA config for unit testing
        self.batch_size = 1
        self.token_num = 1
        self.q_num_heads = 8
        self.kv_num_heads = 1
        self.head_dim_qk = 128
        self.head_dim_v = 128  # nope_size
        self.block_size = 64
        self.max_blocks_per_seq = 2
        self.max_seq_len = self.max_blocks_per_seq * self.block_size
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim_qk)

    def _build_inputs(self, dtype_str="bfloat16", seq_len=5, max_dec_len=1):
        """
        Build all required input tensors for the MLA op.

        Returns:
            (args, query_ref_np, kv_cache_ref_np, block_tables_np)
            where *_ref_np are float32 arrays of the quantized values
            (cast to dtype then back to fp32) for reference computation.
        """
        total_blocks = self.batch_size * self.max_blocks_per_seq
        q_hidden = self.q_num_heads * self.head_dim_qk

        # Generate data in fp32 → cast to target dtype → back to fp32 for reference
        query_fp32 = paddle.to_tensor(np.random.randn(self.token_num, q_hidden).astype(np.float32))
        kv_cache_fp32 = paddle.to_tensor(
            np.random.randn(total_blocks, self.kv_num_heads, self.block_size, self.head_dim_qk).astype(np.float32)
        )

        if dtype_str == "float32":
            query = query_fp32
            kv_cache = kv_cache_fp32
        else:
            query = query_fp32.cast(dtype_str)
            kv_cache = kv_cache_fp32.cast(dtype_str)

        # Quantized reference values (what the kernel actually sees)
        query_ref = query.cast("float32").numpy()
        kv_cache_ref = kv_cache.cast("float32").numpy()

        # Sequence metadata
        seq_lens_decoder = paddle.to_tensor([seq_len], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([self.token_num], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, self.token_num], dtype="int32")
        batch_id_per_token = paddle.zeros([self.token_num], dtype="int32")

        # Block tables: sequential mapping [0, 1, ..., max_blocks-1]
        block_tables = paddle.arange(self.max_blocks_per_seq, dtype="int32").unsqueeze(0)

        # Tile scheduling - minimal for batch_size=1, single-block decode
        kv_batch_ids = paddle.zeros([1], dtype="int32")
        kv_tile_ids_per_batch = paddle.zeros([1], dtype="int32")
        kv_num_blocks = paddle.to_tensor([1], dtype="int32")

        decoder_batch_ids = paddle.zeros([1], dtype="int32")
        decoder_tile_ids_per_batch = paddle.zeros([1], dtype="int32")
        decoder_num_blocks = paddle.to_tensor([1], dtype="int32")
        decoder_chunk_size_device = paddle.to_tensor([self.block_size], dtype="int32")

        max_dec_len_this_time = paddle.to_tensor([max_dec_len], dtype="int32")
        max_len_kv = paddle.to_tensor([seq_len], dtype="int32")

        # compute_dtype must be "bf16" or "fp16" (NOT "bfloat16"/"float16")
        if dtype_str == "bfloat16":
            compute_dtype = "bf16"
        elif dtype_str == "float16":
            compute_dtype = "fp16"
        else:
            compute_dtype = "fp16"  # Fallback; kernel checks query.dtype() first

        args = [
            query,
            kv_cache,
            kv_cache,  # key_cache == value_cache (MLA latent cache)
            seq_lens_decoder,
            seq_lens_this_time,
            cu_seqlens_q,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            decoder_chunk_size_device,
            max_dec_len_this_time,
            max_len_kv,
            # Optional tensors (all None for basic test)
            None,  # attn_mask
            None,  # query_bias
            None,  # query_out_scales
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # out_linear_shifts
            None,  # out_linear_smooths
            # Scalar attributes
            compute_dtype,
            "none",  # cache_quant_type
            self.head_dim_v,  # nope_size
            self.max_seq_len,  # max_input_length
            self.softmax_scale,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            0.0,  # out_linear_in_scale
            0,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        ]

        return args, query_ref, kv_cache_ref, block_tables.numpy()

    # ------------------------------------------------------------------
    # Category F: Error handling
    # ------------------------------------------------------------------

    def test_unsupported_dtype_raises(self):
        """Float32 input must raise RuntimeError (only fp16/bf16 supported)."""
        args, _, _, _ = self._build_inputs(dtype_str="float32", seq_len=1, max_dec_len=0)
        with self.assertRaises(RuntimeError):
            multi_head_latent_attention(*args)

    def test_sm_requirement_on_older_gpu(self):
        """On SM < 90, MLA tensorcore kernel must raise about SM requirement."""
        if self.sm_version >= 90:
            self.skipTest("SM >= 90 detected; SM-requirement error won't trigger.")
        args, _, _, _ = self._build_inputs(dtype_str="bfloat16", seq_len=1, max_dec_len=1)
        with self.assertRaises(RuntimeError):
            multi_head_latent_attention(*args)

    # ------------------------------------------------------------------
    # Category D: Edge case — zero-length decode
    # ------------------------------------------------------------------

    def test_zero_decode_returns_zeros(self):
        """When max_dec_len=0, output should be all zeros with correct shape."""
        if self.sm_version < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        args, _, _, _ = self._build_inputs(dtype_str="bfloat16", seq_len=0, max_dec_len=0)
        out = multi_head_latent_attention(*args)

        expected_shape = [self.token_num, self.q_num_heads * self.head_dim_v]
        self.assertEqual(list(out.shape), expected_shape)
        np.testing.assert_array_equal(
            out.cast("float32").numpy(),
            np.zeros(expected_shape, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Category B: Output shape validation
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output shape must be [token_num, q_num_heads * head_dim_v]."""
        if self.sm_version < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        args, _, _, _ = self._build_inputs(dtype_str="bfloat16", seq_len=5, max_dec_len=5)
        out = multi_head_latent_attention(*args)
        expected_shape = [self.token_num, self.q_num_heads * self.head_dim_v]
        self.assertEqual(list(out.shape), expected_shape)

    # ------------------------------------------------------------------
    # Category A: Numerical correctness
    # ------------------------------------------------------------------

    def test_decode_correctness_bf16(self):
        """Numerical correctness for BF16 single-token decode against reference."""
        if self.sm_version < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        seq_len = 5
        args, query_ref, kv_cache_ref, block_tables_np = self._build_inputs(
            dtype_str="bfloat16", seq_len=seq_len, max_dec_len=seq_len
        )
        out = multi_head_latent_attention(*args)
        out_np = out.cast("float32").numpy()

        ref = _reference_mla_decode(
            query_ref,
            kv_cache_ref,
            block_tables_np,
            seq_len,
            self.q_num_heads,
            self.kv_num_heads,
            self.head_dim_qk,
            self.head_dim_v,
            self.block_size,
            self.softmax_scale,
        )
        np.testing.assert_allclose(out_np, ref, rtol=5e-2, atol=5e-2)

    def test_decode_correctness_fp16(self):
        """Numerical correctness for FP16 single-token decode against reference."""
        if self.sm_version < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        seq_len = 5
        args, query_ref, kv_cache_ref, block_tables_np = self._build_inputs(
            dtype_str="float16", seq_len=seq_len, max_dec_len=seq_len
        )
        out = multi_head_latent_attention(*args)
        out_np = out.cast("float32").numpy()

        ref = _reference_mla_decode(
            query_ref,
            kv_cache_ref,
            block_tables_np,
            seq_len,
            self.q_num_heads,
            self.kv_num_heads,
            self.head_dim_qk,
            self.head_dim_v,
            self.block_size,
            self.softmax_scale,
        )
        np.testing.assert_allclose(out_np, ref, rtol=1e-2, atol=1e-2)

    # ------------------------------------------------------------------
    # Category E: Determinism
    # ------------------------------------------------------------------

    def test_determinism(self):
        """Multiple calls with identical inputs must produce identical outputs."""
        if self.sm_version < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        args, _, _, _ = self._build_inputs(dtype_str="bfloat16", seq_len=5, max_dec_len=5)
        out1 = multi_head_latent_attention(*args)
        out2 = multi_head_latent_attention(*args)
        np.testing.assert_array_equal(
            out1.cast("float32").numpy(),
            out2.cast("float32").numpy(),
        )


if __name__ == "__main__":
    unittest.main()
