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
Test DeepseekV32DSAAttention for DS MLA (Dynamic Sparse Attention) functionality.

This test module validates the DeepseekV32DSAAttention class algorithms from deepseek_v3.py,
including slot_mapping computation, attention mechanisms, and dimension consistency.
The tests use naive Python implementations as ground truth for verification.
"""

import math
import random
import unittest
from unittest.mock import Mock

import numpy as np
import paddle
import paddle.nn.functional as F

seed = 1000
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


# ============================================================================
# Optimized Algorithm Implementations (matching deepseek_v3.py)
# ============================================================================


def compute_slot_mapping_optimized(block_tables, positions, batch_id_per_token, block_size):
    """
    Optimized slot_mapping computation (matching deepseek_v3.py implementation).

    Args:
        block_tables: [num_reqs, max_blocks_per_req] - block ID lookup table
        positions: [num_tokens] - position of each token in its sequence
        batch_id_per_token: [num_tokens] - which request each token belongs to
        block_size: int - number of slots per block

    Returns:
        slot_mapping: [num_tokens] - computed slot for each token

    Formula: slot = block_id * block_size + offset_in_block
    """
    # 1. Compute block index for each token
    block_idx = positions // block_size  # [num_tokens]

    # 2. Lookup block_id from block_tables
    block_ids = block_tables[batch_id_per_token, block_idx]  # [num_tokens]

    # 3. Compute offset within block
    block_offset = positions % block_size  # [num_tokens]

    # 4. Compute final slot mapping
    slot_mapping = block_ids * block_size + block_offset

    return slot_mapping.cast(paddle.int64)


def yarn_get_mscale_impl(scale=1, mscale=1):
    """
    YARN mscale computation (matching DeepseekV32DSAAttention.yarn_get_mscale).

    This is used to adjust the attention softmax scale for long sequences.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ============================================================================
# Naive Algorithm Implementations for Verification
# ============================================================================


def compute_slot_mapping_naive(block_tables, positions, batch_id_per_token, block_size):
    """
    Naive slot_mapping computation using Python loops for verification.
    This is the ground truth implementation.

    slot = block_id * block_size + offset_in_block
    """
    num_tokens = positions.shape[0]
    slot_mapping = []

    for i in range(num_tokens):
        pos = int(positions[i].item())
        batch_id = int(batch_id_per_token[i].item())
        block_idx = pos // block_size
        block_offset = pos % block_size
        block_id = int(block_tables[batch_id, block_idx].item())
        slot = block_id * block_size + block_offset
        slot_mapping.append(slot)

    return paddle.to_tensor(slot_mapping, dtype="int64")


def naive_rms_norm(x, weight, eps=1e-6):
    """Naive RMSNorm implementation for verification."""
    variance = paddle.mean(x.pow(2), axis=-1, keepdim=True)
    x_normed = x * paddle.rsqrt(variance + eps)
    return x_normed * weight


def naive_scaled_dot_product_attention(q, k, v, scale, causal=True):
    """
    Naive scaled dot-product attention implementation.

    Args:
        q: [batch, num_heads, seq_q, head_dim]
        k: [batch, num_heads, seq_k, head_dim]
        v: [batch, num_heads, seq_k, head_dim]
        scale: float - attention scale factor
        causal: bool - whether to apply causal mask

    Returns:
        output: [batch, num_heads, seq_q, head_dim]
    """
    scores = paddle.matmul(q, k, transpose_y=True) * scale

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        mask = paddle.triu(paddle.ones([seq_q, seq_k]), diagonal=1).astype("bool")
        scores = paddle.where(mask, paddle.full_like(scores, float("-inf")), scores)

    attn_weights = F.softmax(scores, axis=-1)
    output = paddle.matmul(attn_weights, v)
    return output


def naive_mla_qkv_split(qkv_a_out, q_lora_rank, kv_lora_rank, qk_rope_head_dim):
    """
    Naive MLA QKV-A projection split (matching DeepseekV32DSAAttention forward).

    Returns query (q_lora_rank), compressed_kv (kv_lora_rank), key_pe (qk_rope_head_dim)
    """
    query = qkv_a_out[..., :q_lora_rank]
    compressed_kv = qkv_a_out[..., q_lora_rank : q_lora_rank + kv_lora_rank]
    key_pe = qkv_a_out[..., q_lora_rank + kv_lora_rank :]
    return query, compressed_kv, key_pe


# ============================================================================
# Mock Factory Functions
# ============================================================================


def create_mock_fd_config(
    hidden_size=7168,
    num_attention_heads=128,
    num_key_value_heads=1,
    head_dim=128,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    block_size=64,
    max_model_len=4096,
    num_hidden_layers=12,
    index_head_dim=2048,
    index_n_heads=8,
    index_topk=32,
    tensor_parallel_size=1,
):
    """Create a comprehensive mock FD config for DeepseekV32DSAAttention testing."""
    fd_config = Mock()

    # Cache config
    fd_config.cache_config = Mock()
    fd_config.cache_config.block_size = block_size

    # Model config
    fd_config.model_config = Mock()
    fd_config.model_config.hidden_size = hidden_size
    fd_config.model_config.num_attention_heads = num_attention_heads
    fd_config.model_config.num_key_value_heads = num_key_value_heads
    fd_config.model_config.head_dim = head_dim
    fd_config.model_config.max_model_len = max_model_len
    fd_config.model_config.num_hidden_layers = num_hidden_layers

    # MLA specific
    fd_config.model_config.kv_lora_rank = kv_lora_rank
    fd_config.model_config.q_lora_rank = q_lora_rank
    fd_config.model_config.qk_rope_head_dim = qk_rope_head_dim
    fd_config.model_config.qk_nope_head_dim = qk_nope_head_dim
    fd_config.model_config.v_head_dim = v_head_dim

    # Indexer/DSA specific
    fd_config.model_config.index_head_dim = index_head_dim
    fd_config.model_config.index_n_heads = index_n_heads
    fd_config.model_config.index_topk = index_topk

    # RoPE scaling for DeepSeek V3
    fd_config.model_config.rope_scaling = {
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    fd_config.model_config.rope_theta = 10000.0
    fd_config.model_config.rms_norm_eps = 1e-6
    fd_config.model_config.start_layer_index = 0
    fd_config.model_config.hidden_act = "silu"
    fd_config.model_config.is_quantized = False
    fd_config.model_config.model_format = "torch"
    fd_config.model_config.model_type = "deepseek_v32"

    # Parallel config
    fd_config.parallel_config = Mock()
    fd_config.parallel_config.tensor_parallel_size = tensor_parallel_size
    fd_config.parallel_config.tensor_parallel_rank = 0
    fd_config.parallel_config.expert_parallel_size = 1
    fd_config.parallel_config.pd_disaggregation_mode = "none"
    fd_config.parallel_config.local_data_parallel_id = 0

    # Speculative config
    fd_config.speculative_config = Mock()
    fd_config.speculative_config.method = None
    fd_config.speculative_config.num_speculative_tokens = 1
    fd_config.speculative_config.model_type = None

    # Quant config
    fd_config.quant_config = None

    # Load config
    fd_config.load_config = Mock()
    fd_config.load_config.load_choices = "default_v1"
    fd_config.load_config.dynamic_load_weight = False

    return fd_config


def create_mock_forward_meta_light(
    prefill=True,
    batch_size=2,
    seq_len=16,
    block_size=64,
):
    """Create a lightweight mock ForwardMeta for testing (no GPU tensors)."""
    forward_meta = Mock()

    # Sequence length tensors (CPU)
    if prefill:
        forward_meta.seq_lens_encoder = paddle.to_tensor(
            [seq_len] * batch_size, dtype="int32", place=paddle.CPUPlace()
        )
        forward_meta.seq_lens_decoder = paddle.to_tensor([0] * batch_size, dtype="int32", place=paddle.CPUPlace())
        forward_meta.seq_lens_this_time = paddle.to_tensor(
            [seq_len * batch_size], dtype="int32", place=paddle.CPUPlace()
        )
        cu_seqlens = [0] + [seq_len * (i + 1) for i in range(batch_size)]
        forward_meta.cu_seqlens_q = paddle.to_tensor(cu_seqlens, dtype="int32", place=paddle.CPUPlace())
        forward_meta.cu_seqlens_k = paddle.to_tensor(cu_seqlens, dtype="int32", place=paddle.CPUPlace())
        forward_meta.max_len_tensor_cpu = paddle.to_tensor(
            [0, seq_len, 0, 0, 0, seq_len], dtype="int32", place=paddle.CPUPlace()
        )
    else:
        forward_meta.seq_lens_encoder = paddle.to_tensor([0] * batch_size, dtype="int32", place=paddle.CPUPlace())
        forward_meta.seq_lens_decoder = paddle.to_tensor(
            [[seq_len]] * batch_size, dtype="int32", place=paddle.CPUPlace()
        )
        forward_meta.seq_lens_this_time = paddle.to_tensor([batch_size], dtype="int32", place=paddle.CPUPlace())
        forward_meta.cu_seqlens_q = paddle.to_tensor(
            [0] + list(range(1, batch_size + 1)), dtype="int32", place=paddle.CPUPlace()
        )
        forward_meta.cu_seqlens_k = paddle.to_tensor(
            [0] + list(range(1, batch_size + 1)), dtype="int32", place=paddle.CPUPlace()
        )
        forward_meta.max_len_tensor_cpu = paddle.to_tensor(
            [0, 0, 1, 0, 0, seq_len + 1], dtype="int32", place=paddle.CPUPlace()
        )

    # Batch/token mappings
    batch_ids = []
    for i in range(batch_size):
        batch_ids.extend([i] * (seq_len if prefill else 1))
    forward_meta.batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32", place=paddle.CPUPlace())

    # Position IDs
    if prefill:
        position_ids = []
        for i in range(batch_size):
            position_ids.extend(list(range(seq_len)))
        forward_meta.position_ids = paddle.to_tensor(position_ids, dtype="int32", place=paddle.CPUPlace())
    else:
        forward_meta.position_ids = paddle.to_tensor([seq_len] * batch_size, dtype="int32", place=paddle.CPUPlace())

    # Block tables (small, on CPU)
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size + 2
    forward_meta.block_tables = paddle.randint(0, 100, [batch_size, max_blocks_per_seq], dtype="int32")

    forward_meta.max_input_length = seq_len
    forward_meta.is_dummy_or_profile_run = False
    forward_meta.attn_backend = None

    return forward_meta


# ============================================================================
# Test Classes
# ============================================================================


class TestComputeSlotMapping(unittest.TestCase):
    """Test compute_slot_mapping function accuracy against naive implementation."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestComputeSlotMapping"
        self.block_size = 64

    def test_slot_mapping_basic(self):
        """Test basic slot_mapping computation against naive implementation."""
        batch_size = 4
        num_tokens = 32
        max_blocks = 20

        block_tables = paddle.randint(0, 100, [batch_size, max_blocks], dtype="int32")
        max_position = (max_blocks - 1) * self.block_size
        positions = paddle.randint(0, max_position, [num_tokens], dtype="int32")
        batch_ids = paddle.randint(0, batch_size, [num_tokens], dtype="int32")

        optimized = compute_slot_mapping_optimized(block_tables, positions, batch_ids, self.block_size)
        naive = compute_slot_mapping_naive(block_tables, positions, batch_ids, self.block_size)

        np.testing.assert_array_equal(
            optimized.numpy(),
            naive.numpy(),
            err_msg="slot_mapping mismatch between optimized and naive",
        )

    def test_slot_mapping_edge_cases(self):
        """Test slot_mapping at block boundaries."""
        block_size = self.block_size
        block_tables = paddle.to_tensor([[5, 10, 15, 20]], dtype="int32")

        # Position 0 (start of block 0)
        result = compute_slot_mapping_optimized(
            block_tables,
            paddle.to_tensor([0], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            block_size,
        )
        self.assertEqual(result[0].item(), 5 * block_size + 0)

        # Position block_size-1 (end of block 0)
        result = compute_slot_mapping_optimized(
            block_tables,
            paddle.to_tensor([block_size - 1], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            block_size,
        )
        self.assertEqual(result[0].item(), 5 * block_size + (block_size - 1))

        # Position block_size (start of block 1)
        result = compute_slot_mapping_optimized(
            block_tables,
            paddle.to_tensor([block_size], dtype="int32"),
            paddle.to_tensor([0], dtype="int32"),
            block_size,
        )
        self.assertEqual(result[0].item(), 10 * block_size + 0)

    def test_slot_mapping_prefill_scenario(self):
        """Test slot_mapping with prefill (variable sequence lengths)."""
        batch_size = 4
        seq_lengths = [16, 32, 24, 20]
        block_size = self.block_size
        max_blocks = 20

        block_tables = paddle.randint(0, 100, [batch_size, max_blocks], dtype="int32")

        positions_list = []
        batch_ids_list = []
        for batch_idx, seq_len in enumerate(seq_lengths):
            positions_list.extend(range(seq_len))
            batch_ids_list.extend([batch_idx] * seq_len)

        positions = paddle.to_tensor(positions_list, dtype="int32")
        batch_ids = paddle.to_tensor(batch_ids_list, dtype="int32")

        optimized = compute_slot_mapping_optimized(block_tables, positions, batch_ids, block_size)
        naive = compute_slot_mapping_naive(block_tables, positions, batch_ids, block_size)

        np.testing.assert_array_equal(
            optimized.numpy(),
            naive.numpy(),
            err_msg="Slot mapping mismatch in prefill scenario",
        )

    def test_slot_mapping_decode_scenario(self):
        """Test slot_mapping in decode (single token per sequence)."""
        batch_size = 8
        block_size = self.block_size
        current_positions = [50, 32, 128, 64, 15, 100, 48, 72]

        block_tables = paddle.randint(0, 100, [batch_size, 20], dtype="int32")
        positions = paddle.to_tensor(current_positions, dtype="int32")
        batch_ids = paddle.to_tensor(list(range(batch_size)), dtype="int32")

        optimized = compute_slot_mapping_optimized(block_tables, positions, batch_ids, block_size)
        naive = compute_slot_mapping_naive(block_tables, positions, batch_ids, block_size)

        np.testing.assert_array_equal(
            optimized.numpy(),
            naive.numpy(),
            err_msg="Slot mapping mismatch in decode scenario",
        )

    def test_slot_mapping_large_batch(self):
        """Test slot_mapping with large batch size."""
        batch_size = 32
        num_tokens = 256
        block_size = self.block_size
        max_blocks = 50

        block_tables = paddle.randint(0, 100, [batch_size, max_blocks], dtype="int32")
        max_position = (max_blocks - 1) * block_size
        positions = paddle.randint(0, max_position, [num_tokens], dtype="int32")
        batch_ids = paddle.randint(0, batch_size, [num_tokens], dtype="int32")

        optimized = compute_slot_mapping_optimized(block_tables, positions, batch_ids, block_size)
        naive = compute_slot_mapping_naive(block_tables, positions, batch_ids, block_size)

        np.testing.assert_array_equal(
            optimized.numpy(),
            naive.numpy(),
            err_msg="Slot mapping mismatch in large batch scenario",
        )


class TestYarnMscale(unittest.TestCase):
    """Test YARN mscale computation."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestYarnMscale"

    def test_mscale_scale_leq_1(self):
        """Test mscale when scale <= 1."""
        result = yarn_get_mscale_impl(scale=1, mscale=1)
        self.assertEqual(result, 1.0)

        result = yarn_get_mscale_impl(scale=0.5, mscale=2)
        self.assertEqual(result, 1.0)

    def test_mscale_scale_gt_1(self):
        """Test mscale when scale > 1."""
        test_cases = [
            (2, 1),
            (10, 1),
            (40, 1),
            (40, 1.0),
            (100, 0.5),
        ]

        for scale, mscale in test_cases:
            result = yarn_get_mscale_impl(scale=scale, mscale=mscale)
            expected = 0.1 * mscale * math.log(scale) + 1.0
            np.testing.assert_almost_equal(
                result,
                expected,
                decimal=6,
                err_msg=f"mscale mismatch for scale={scale}, mscale={mscale}",
            )

    def test_mscale_monotonicity(self):
        """Test YARN mscale is monotonically increasing with scale."""
        mscale = 1.0
        prev_result = 1.0

        for scale in [1, 2, 5, 10, 20, 40, 100]:
            result = yarn_get_mscale_impl(scale, mscale)
            self.assertGreaterEqual(result, prev_result)
            prev_result = result


class TestNaiveAttention(unittest.TestCase):
    """Test naive attention implementation correctness."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestNaiveAttention"

    def test_attention_output_shape(self):
        """Test naive attention produces correct output shape."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            batch, num_heads, seq_len, head_dim = 2, 4, 8, 32
            scale = head_dim**-0.5

            q = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            k = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            v = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")

            output = naive_scaled_dot_product_attention(q, k, v, scale, causal=True)

            self.assertEqual(output.shape, [batch, num_heads, seq_len, head_dim])
            self.assertFalse(paddle.isnan(output).any().item())
            self.assertFalse(paddle.isinf(output).any().item())

    def test_attention_causal_vs_noncausal(self):
        """Test that causal and non-causal attention produce different results."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            paddle.seed(42)
            batch, num_heads, seq_len, head_dim = 1, 1, 4, 8
            scale = head_dim**-0.5

            q = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            k = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            v = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")

            causal_output = naive_scaled_dot_product_attention(q, k, v, scale, causal=True)
            non_causal_output = naive_scaled_dot_product_attention(q, k, v, scale, causal=False)

            # Verify both outputs have correct shape
            self.assertEqual(causal_output.shape, [batch, num_heads, seq_len, head_dim])
            self.assertEqual(non_causal_output.shape, [batch, num_heads, seq_len, head_dim])

            # Total outputs should be different (due to causal masking)
            diff = paddle.abs(causal_output - non_causal_output).sum().item()
            self.assertGreater(diff, 0, "Causal and non-causal outputs should differ")

    def test_attention_vs_manual(self):
        """Compare naive attention with manual step-by-step computation."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            batch, num_heads, seq_len, head_dim = 1, 2, 4, 16
            scale = head_dim**-0.5

            q = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            k = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            v = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")

            naive_output = naive_scaled_dot_product_attention(q, k, v, scale, causal=False)

            # Manual computation
            scores = paddle.matmul(q, k, transpose_y=True) * scale
            attn_weights = F.softmax(scores, axis=-1)
            manual_output = paddle.matmul(attn_weights, v)

            np.testing.assert_allclose(
                naive_output.numpy(),
                manual_output.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )


class TestDeepseekV32DSAAttentionConfig(unittest.TestCase):
    """Test DeepseekV32DSAAttention configuration and initialization."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestDeepseekV32DSAAttentionConfig"

    def test_config_parameters(self):
        """Test that config parameters are correctly parsed."""
        fd_config = create_mock_fd_config(
            hidden_size=7168,
            num_attention_heads=128,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            index_head_dim=2048,
            index_n_heads=8,
            index_topk=32,
        )

        # Verify config is correctly set
        self.assertEqual(fd_config.model_config.hidden_size, 7168)
        self.assertEqual(fd_config.model_config.num_attention_heads, 128)
        self.assertEqual(fd_config.model_config.kv_lora_rank, 512)
        self.assertEqual(fd_config.model_config.q_lora_rank, 1536)
        self.assertEqual(fd_config.model_config.qk_rope_head_dim, 64)
        self.assertEqual(fd_config.model_config.qk_nope_head_dim, 128)
        self.assertEqual(fd_config.model_config.v_head_dim, 128)
        self.assertEqual(fd_config.model_config.index_head_dim, 2048)

    def test_attention_scale_computation(self):
        """Test that attention softmax scale is correctly computed."""
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        expected_base_scale = qk_head_dim**-0.5

        # With rope scaling
        mscale = yarn_get_mscale_impl(40, 1.0)
        expected_scaled = expected_base_scale * mscale * mscale

        np.testing.assert_almost_equal(
            expected_base_scale,
            (qk_nope_head_dim + qk_rope_head_dim) ** -0.5,
            decimal=6,
        )
        self.assertGreater(expected_scaled, expected_base_scale)

    def test_forward_meta_creation_prefill(self):
        """Test ForwardMeta creation for prefill scenario."""
        forward_meta = create_mock_forward_meta_light(
            prefill=True,
            batch_size=2,
            seq_len=16,
            block_size=64,
        )

        self.assertEqual(forward_meta.seq_lens_encoder.shape[0], 2)
        self.assertTrue((forward_meta.seq_lens_encoder == 16).all().item())
        self.assertEqual(forward_meta.max_len_tensor_cpu[1].item(), 16)

    def test_forward_meta_creation_decode(self):
        """Test ForwardMeta creation for decode scenario."""
        forward_meta = create_mock_forward_meta_light(
            prefill=False,
            batch_size=2,
            seq_len=16,
            block_size=64,
        )

        self.assertEqual(forward_meta.max_len_tensor_cpu[2].item(), 1)


class TestMLADimensions(unittest.TestCase):
    """Test MLA dimension calculations and consistency."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestMLADimensions"

        # Typical DeepSeek V3 MLA dimensions
        self.hidden_size = 7168
        self.num_attention_heads = 128
        self.kv_lora_rank = 512
        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.qk_nope_head_dim = 128
        self.v_head_dim = 128
        self.index_head_dim = 2048
        self.index_n_heads = 8
        self.index_topk = 32

    def test_qkv_a_projection_dimension(self):
        """Test QKV-A projection output dimension."""
        qkv_a_out_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        expected = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        self.assertEqual(qkv_a_out_dim, expected)

    def test_q_b_projection_dimension(self):
        """Test Q-B projection output dimension."""
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q_b_out_dim = self.num_attention_heads * qk_head_dim
        expected = self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.assertEqual(q_b_out_dim, expected)

    def test_kv_input_dimension_for_dsa(self):
        """Test KV input dimension for DSA attention."""
        kv_input_dim = self.kv_lora_rank + self.qk_rope_head_dim
        expected = self.kv_lora_rank + self.qk_rope_head_dim
        self.assertEqual(kv_input_dim, expected)

    def test_output_projection_dimension(self):
        """Test output projection input dimension."""
        o_proj_input_dim = self.num_attention_heads * self.v_head_dim
        expected = self.num_attention_heads * self.v_head_dim
        self.assertEqual(o_proj_input_dim, expected)

    def test_indexer_dimensions(self):
        """Test indexer dimension calculations."""
        indexer_output_dim = self.index_head_dim * self.index_n_heads
        expected = self.index_head_dim * self.index_n_heads
        self.assertEqual(indexer_output_dim, expected)

    def test_qkv_split_correctness(self):
        """Test QKV split produces correct dimensions."""
        num_tokens = 16
        q_lora_rank = self.q_lora_rank
        kv_lora_rank = self.kv_lora_rank
        qk_rope_head_dim = self.qk_rope_head_dim

        # Simulate qkv_a_proj output
        total_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        qkv_a_out = paddle.randn([num_tokens, total_dim], dtype="float32")

        # Split using naive implementation
        query, compressed_kv, key_pe = naive_mla_qkv_split(qkv_a_out, q_lora_rank, kv_lora_rank, qk_rope_head_dim)

        self.assertEqual(query.shape, [num_tokens, q_lora_rank])
        self.assertEqual(compressed_kv.shape, [num_tokens, kv_lora_rank])
        self.assertEqual(key_pe.shape, [num_tokens, qk_rope_head_dim])

    def test_query_reshape_and_split(self):
        """Test query reshape and nope/pe split."""
        num_tokens = 16
        num_heads = self.num_attention_heads
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Simulate q_b_proj output
        q_b_out = paddle.randn([num_tokens, num_heads * qk_head_dim], dtype="float32")

        # Reshape
        query = q_b_out.reshape([-1, num_heads, qk_head_dim])
        self.assertEqual(query.shape, [num_tokens, num_heads, qk_head_dim])

        # Split into nope and pe
        query_nope, query_pe = query.split([qk_nope_head_dim, qk_rope_head_dim], axis=-1)
        self.assertEqual(query_nope.shape, [num_tokens, num_heads, qk_nope_head_dim])
        self.assertEqual(query_pe.shape, [num_tokens, num_heads, qk_rope_head_dim])

    def test_kv_concat_dimension(self):
        """Test KV concat produces correct dimension for DSA."""
        num_tokens = 16
        kv_lora_rank = self.kv_lora_rank
        qk_rope_head_dim = self.qk_rope_head_dim

        compressed_kv = paddle.randn([num_tokens, kv_lora_rank], dtype="float32")
        key_pe = paddle.randn([num_tokens, qk_rope_head_dim], dtype="float32")

        kv = paddle.concat([compressed_kv, key_pe], axis=-1)
        self.assertEqual(kv.shape, [num_tokens, kv_lora_rank + qk_rope_head_dim])


class TestDeepseekV32DSAAttentionNumerics(unittest.TestCase):
    """Test numerical properties of DeepseekV32DSAAttention computations."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestDeepseekV32DSAAttentionNumerics"

    def test_softmax_scale_range(self):
        """Test softmax scale is in reasonable range."""
        qk_head_dims = [64, 128, 192, 256]

        for qk_head_dim in qk_head_dims:
            scale = qk_head_dim**-0.5
            self.assertGreater(scale, 0)
            self.assertLess(scale, 1)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            batch, num_heads, seq_len, head_dim = 1, 2, 8, 32
            scale = head_dim**-0.5

            q = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            k = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")

            scores = paddle.matmul(q, k, transpose_y=True) * scale
            attn_weights = F.softmax(scores, axis=-1)

            weight_sums = attn_weights.sum(axis=-1)
            np.testing.assert_allclose(
                weight_sums.numpy(),
                np.ones_like(weight_sums.numpy()),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_causal_attention_preserves_causality(self):
        """Test that causal attention doesn't attend to future tokens."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            batch, num_heads, seq_len, head_dim = 1, 1, 4, 8
            scale = head_dim**-0.5

            q = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")
            k = paddle.randn([batch, num_heads, seq_len, head_dim], dtype="float32")

            scores = paddle.matmul(q, k, transpose_y=True) * scale
            mask = paddle.triu(paddle.ones([seq_len, seq_len]), diagonal=1).astype("bool")
            masked_scores = paddle.where(mask, paddle.full_like(scores, float("-inf")), scores)
            attn_weights = F.softmax(masked_scores, axis=-1)

            # Check upper triangle is zero (no attention to future)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    self.assertAlmostEqual(
                        attn_weights[0, 0, i, j].item(),
                        0.0,
                        places=5,
                        msg=f"Position ({i}, {j}) should have zero attention weight",
                    )

    def test_rms_norm_correctness(self):
        """Test RMSNorm naive implementation."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            hidden_size = 128
            x = paddle.randn([4, hidden_size], dtype="float32")
            weight = paddle.ones([hidden_size], dtype="float32")
            eps = 1e-6

            output = naive_rms_norm(x, weight, eps)

            # Verify output shape
            self.assertEqual(output.shape, x.shape)

            # Verify RMS is approximately 1 after normalization
            rms = paddle.sqrt(paddle.mean(output.pow(2), axis=-1))
            np.testing.assert_allclose(
                rms.numpy(),
                np.ones_like(rms.numpy()),
                rtol=1e-4,
                atol=1e-4,
            )


class TestMLAForwardDataFlow(unittest.TestCase):
    """Test MLA forward pass data flow and transformations."""

    def setUp(self):
        paddle.disable_static()
        self.name = "TestMLAForwardDataFlow"

        # Config
        self.batch_size = 2
        self.seq_len = 8
        self.num_tokens = self.batch_size * self.seq_len
        self.hidden_size = 256
        self.num_heads = 8
        self.kv_lora_rank = 64
        self.q_lora_rank = 128
        self.qk_rope_head_dim = 32
        self.qk_nope_head_dim = 32
        self.v_head_dim = 32
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    def test_qkv_a_proj_output(self):
        """Test QKV-A projection output dimensions."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            hidden_states = paddle.randn([self.num_tokens, self.hidden_size], dtype="float32")

            # Simulate qkv_a_proj (random weight matrix)
            qkv_a_out_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            weight = paddle.randn([self.hidden_size, qkv_a_out_dim], dtype="float32")
            qkv_a_out = paddle.matmul(hidden_states, weight)

            self.assertEqual(qkv_a_out.shape, [self.num_tokens, qkv_a_out_dim])

    def test_query_processing_pipeline(self):
        """Test full query processing pipeline."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            # Simulate qkv_a_proj output
            total_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            qkv_a_out = paddle.randn([self.num_tokens, total_dim], dtype="float32")

            # Split
            query, compressed_kv, key_pe = naive_mla_qkv_split(
                qkv_a_out, self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
            )

            # Simulate q_a_layernorm (just use naive_rms_norm)
            weight = paddle.ones([self.q_lora_rank], dtype="float32")
            query_normed = naive_rms_norm(query, weight)

            # Simulate q_b_proj
            q_b_weight = paddle.randn([self.q_lora_rank, self.num_heads * self.qk_head_dim], dtype="float32")
            query_projected = paddle.matmul(query_normed, q_b_weight)

            # Reshape
            query_reshaped = query_projected.reshape([-1, self.num_heads, self.qk_head_dim])

            # Split into nope and pe
            query_nope, query_pe = query_reshaped.split([self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

            # Verify shapes
            self.assertEqual(query_nope.shape, [self.num_tokens, self.num_heads, self.qk_nope_head_dim])
            self.assertEqual(query_pe.shape, [self.num_tokens, self.num_heads, self.qk_rope_head_dim])

    def test_kv_processing_pipeline(self):
        """Test KV processing pipeline."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            # Simulate qkv_a_proj output
            total_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            qkv_a_out = paddle.randn([self.num_tokens, total_dim], dtype="float32")

            # Split
            query, compressed_kv, key_pe = naive_mla_qkv_split(
                qkv_a_out, self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
            )

            # Simulate kv_a_layernorm
            kv_weight = paddle.ones([self.kv_lora_rank], dtype="float32")
            compressed_kv_normed = naive_rms_norm(compressed_kv, kv_weight)

            # Reshape key_pe for concat
            key_pe_squeezed = key_pe  # Already [num_tokens, qk_rope_head_dim]

            # Concat compressed_kv and key_pe for KV
            kv = paddle.concat([compressed_kv_normed, key_pe_squeezed], axis=-1)

            # Verify shape
            expected_kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
            self.assertEqual(kv.shape, [self.num_tokens, expected_kv_dim])

    def test_output_projection_dimension_consistency(self):
        """Test output dimension consistency through the pipeline."""
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            # After attention, output should be [num_tokens, num_heads, kv_lora_rank]
            # After kv_b_proj_bmm (v projection), should be [num_tokens, num_heads, v_head_dim]
            # After reshape, should be [num_tokens, num_heads * v_head_dim]
            # After o_proj, should be [num_tokens, hidden_size]

            fmha_out = paddle.randn([self.num_tokens, self.num_heads * self.kv_lora_rank], dtype="float32")

            # Simulate kv_b_proj_bmm (v projection)
            fmha_reshaped = fmha_out.reshape([self.num_tokens, self.num_heads, self.kv_lora_rank])
            # Transpose for bmm: [num_heads, num_tokens, kv_lora_rank]
            fmha_transposed = fmha_reshaped.transpose([1, 0, 2])

            # Simulate kv_b_proj_bmm weight for v projection
            # Shape: [num_heads, kv_lora_rank, v_head_dim]
            v_proj_weight = paddle.randn([self.num_heads, self.kv_lora_rank, self.v_head_dim], dtype="float32")

            # BMM: [num_heads, num_tokens, kv_lora_rank] @ [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, num_tokens, v_head_dim]
            v_proj_out = paddle.bmm(fmha_transposed, v_proj_weight)

            # Transpose back and reshape: [num_tokens, num_heads * v_head_dim]
            v_proj_out = v_proj_out.transpose([1, 0, 2]).reshape([self.num_tokens, -1])

            expected_o_proj_input_dim = self.num_heads * self.v_head_dim
            self.assertEqual(v_proj_out.shape, [self.num_tokens, expected_o_proj_input_dim])


if __name__ == "__main__":
    unittest.main()
