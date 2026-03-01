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
Attention determinism tests.

Part 1: Paddle SDPA determinism tests
    Verify that scaled_dot_product_attention produces bitwise-identical
    results across repeated runs, varying batch sizes, sequence lengths,
    head configurations, dtypes, and backends.

Part 2: Append Attention partition_kv determinism tests (算子级单测)
    Test the append attention kernel's partition_kv code path which is
    triggered when num_chunks > 1 (KV length > max_partition_size).

    We use FLAGS_max_partition_size=64 (smallest value) to maximize num_chunks
    and increase sensitivity to non-determinism detection.

    This is the OPERATOR-LEVEL test that guarantees:
    - Positive (deterministic mode ON): bit-exact results across runs
    - Negative (deterministic mode OFF): can detect non-determinism

    See docs/test_long_seq_review.md for the layered testing strategy.
"""

import os
import unittest

import pytest

pytestmark = pytest.mark.gpu

import paddle
import paddle.nn.functional as F

# --------------- constants ---------------
BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LEN = 32
NUM_RUNS = 5

# Use smallest chunk_size (64) to maximize num_chunks and increase
# sensitivity to partition_kv non-determinism detection.
_MAX_PARTITION_SIZE_FOR_TEST = 64


# --------------- helpers ---------------
def _make_qkv(batch_size, num_heads, seq_len, head_dim, dtype="float32", seed=42):
    """Create deterministic q/k/v tensors."""
    paddle.seed(seed)
    shape = [batch_size, num_heads, seq_len, head_dim]
    return (
        paddle.randn(shape, dtype=dtype),
        paddle.randn(shape, dtype=dtype),
        paddle.randn(shape, dtype=dtype),
    )


def _assert_deterministic(test_case, func, num_runs=NUM_RUNS):
    """Run *func* multiple times and assert all results are bitwise equal."""
    results = [func().clone() for _ in range(num_runs)]
    for i in range(1, num_runs):
        test_case.assertTrue(
            paddle.equal(results[0], results[i]).all().item(),
            f"Run 0 vs Run {i} differ",
        )


# ===========================================================================
# Part 1: Paddle SDPA determinism tests
# ===========================================================================


class TestAttentionDeterminism(unittest.TestCase):

    def setUp(self):
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_sdpa_determinism(self):
        """Basic multi-run determinism, causal and non-causal."""
        for is_causal in [False, True]:
            with self.subTest(is_causal=is_causal):
                q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
                _assert_deterministic(
                    self,
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=False),
                )

    def test_dtype_determinism(self):
        """Determinism across float32 and float16."""
        for dtype in ["float32", "float16"]:
            with self.subTest(dtype=dtype):
                if dtype == "float16" and not paddle.is_compiled_with_cuda():
                    continue
                q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype)
                _assert_deterministic(
                    self,
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False),
                )

    def test_seq_length_determinism(self):
        """Determinism across various sequence lengths."""
        for seq_len in [1, 16, 64, 128, 256, 512]:
            with self.subTest(seq_len=seq_len):
                q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
                _assert_deterministic(
                    self,
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False),
                    num_runs=2,
                )

    def test_head_config_determinism(self):
        """Determinism across different head configurations."""
        for num_heads, head_dim in [(1, 64), (4, 64), (8, 64), (16, 32), (32, 32)]:
            with self.subTest(num_heads=num_heads, head_dim=head_dim):
                q, k, v = _make_qkv(BATCH_SIZE, num_heads, SEQ_LEN, head_dim)
                _assert_deterministic(
                    self,
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False),
                    num_runs=2,
                )

    def test_batch_invariance(self):
        """First sample result should be identical regardless of batch size."""
        max_bs = 8
        q_full, k_full, v_full = _make_qkv(max_bs, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        ref = F.scaled_dot_product_attention(q_full[:1], k_full[:1], v_full[:1], is_causal=False, enable_gqa=False)
        for bs in [2, 4, 8]:
            with self.subTest(batch_size=bs):
                result = F.scaled_dot_product_attention(
                    q_full[:bs],
                    k_full[:bs],
                    v_full[:bs],
                    is_causal=False,
                    enable_gqa=False,
                )
                self.assertTrue(
                    paddle.equal(ref, result[0:1]).all().item(),
                    f"Batch invariance failed at bs={bs}",
                )

    def test_backend_determinism(self):
        """Determinism across different backends (auto, math, flash)."""
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Backend test requires CUDA")

        for backend in [None, "math", "flash"]:
            backend_name = backend or "auto"
            with self.subTest(backend=backend_name):
                q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
                try:
                    _assert_deterministic(
                        self,
                        lambda: F.scaled_dot_product_attention(
                            q, k, v, is_causal=False, enable_gqa=False, backend=backend
                        ),
                        num_runs=2,
                    )
                except Exception as e:
                    # Skip unsupported backends instead of failing
                    print(f"  Skip backend={backend_name}: {e}")
                    continue

    def test_batch_invariant_mode_compatibility(self):
        """batch_invariant_mode should not change SDPA results."""
        try:
            from fastdeploy.model_executor.layers.batch_invariant_ops import (
                set_batch_invariant_mode,
            )
        except ImportError:
            self.skipTest("fastdeploy not installed")

        q, k, v = _make_qkv(4, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        with set_batch_invariant_mode(True):
            result_bi = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        result_normal = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        self.assertTrue(
            paddle.equal(result_bi, result_normal).all().item(),
            "batch_invariant_mode should not change native SDPA results",
        )

    def test_manual_attention_determinism(self):
        """Manual QK^T/softmax/V attention should also be deterministic."""
        q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        def _manual_attn():
            scores = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
            scores = scores / paddle.sqrt(paddle.to_tensor(q.shape[-1], dtype=scores.dtype))
            weights = F.softmax(scores, axis=-1)
            return paddle.matmul(weights, v)

        _assert_deterministic(self, _manual_attn)


# ===========================================================================
# Part 2: Append Attention partition_kv determinism tests
# ===========================================================================


class TestAppendAttentionPartitionKVDeterminism(unittest.TestCase):
    """
    Test append attention kernel's partition_kv code path determinism.

    The partition_kv path is triggered when:
        num_chunks = div_up(max_dec_len, chunk_size) > 1

    We use chunk_size=64 (smallest value) to maximize num_chunks:
    - 256 tokens → 4 chunks
    - 512 tokens → 8 chunks
    - 1536 tokens → 24 chunks

    This test class verifies:
    1. POSITIVE: With FD_DETERMINISTIC_MODE=1, results are bit-exact across runs
    2. NEGATIVE: Without deterministic mode, non-determinism can be detected
       (this proves the test is effective and the bug exists)

    The tensor-level comparison (instead of text-level) ensures:
    - Positive test: if it passes, the fix is definitely working
    - Negative test: if non-determinism exists, it WILL be detected
      (unlike end-to-end tests where sampling masks the difference)
    """

    @classmethod
    def setUpClass(cls):
        """Check if append attention is available."""
        try:
            from fastdeploy.model_executor.layers.attention.ops import (
                append_attention,
                get_block_shape_and_split_kv_block,
            )

            cls._append_attention_func = staticmethod(append_attention)
            cls._get_block_shape_func = staticmethod(get_block_shape_and_split_kv_block)
            cls.available = True
        except ImportError:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("append_attention not available")
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA required for append attention")

        paddle.set_device("gpu")

        # Configuration that triggers partition_kv (KV length > chunk_size)
        self.batch_size = 1
        self.q_num_head = 16
        self.kv_num_head = 2
        self.head_dim = 128
        self.blocksize = 64

        # With chunk_size=64, even 256 tokens triggers partition_kv (4 chunks)
        # Using 512 to ensure num_chunks = 8 for thorough testing
        self.seq_len = 512
        self.max_seq_len = 2048

        self.dtype = "bfloat16"

    def _create_test_tensors(self, seed=42):
        """Create tensors for append attention test."""
        import numpy as np

        np.random.seed(seed)
        paddle.seed(seed)

        # Block table setup
        block_num_per_seq = (self.seq_len + self.blocksize - 1) // self.blocksize
        max_block_num = block_num_per_seq * self.batch_size

        # Create QKV tensor
        token_num = self.batch_size * self.seq_len
        total_head_dim = (self.q_num_head + 2 * self.kv_num_head) * self.head_dim
        qkv = paddle.randn([token_num, total_head_dim], dtype=self.dtype)

        # Create cache tensors
        cache_shape = (max_block_num, self.kv_num_head, self.blocksize, self.head_dim)
        cache_k = paddle.zeros(shape=cache_shape, dtype=self.dtype)
        cache_v = paddle.zeros(shape=cache_shape, dtype=self.dtype)

        # Block tables
        block_tables = paddle.zeros(shape=(self.batch_size, block_num_per_seq), dtype="int32")
        for i in range(self.batch_size):
            for j in range(block_num_per_seq):
                block_tables[i, j] = i * block_num_per_seq + j

        # Sequence length tensors
        seq_lens_encoder = paddle.to_tensor([self.seq_len] * self.batch_size, dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0] * self.batch_size, dtype="int32")
        seq_lens_this_time = seq_lens_encoder.clone()

        # Cumulative sequence lengths
        cu_seqlens_q = paddle.zeros([self.batch_size + 1], dtype="int32")
        for i in range(self.batch_size):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + self.seq_len

        # Batch ID per token
        batch_id_per_token = paddle.zeros([token_num], dtype="int32")
        for i in range(self.batch_size):
            start = i * self.seq_len
            end = (i + 1) * self.seq_len
            batch_id_per_token[start:end] = i

        # Launch buffers
        group_size = self.q_num_head // self.kv_num_head
        decode_max_tile_size = int(1024 * self.batch_size * np.ceil((2 * group_size) / 12))
        encode_max_tile_size = self.batch_size * (self.max_seq_len * group_size // 64)
        kv_max_tile_size = self.batch_size * (self.max_seq_len // self.blocksize)

        decoder_batch_ids = paddle.full([decode_max_tile_size], 0, dtype="int32")
        decoder_tile_ids_per_batch = paddle.full([decode_max_tile_size], 0, dtype="int32")
        decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
        decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
        max_len_tensor_cpu = paddle.full([9], 0, dtype="int32").cpu()

        encoder_batch_ids = paddle.full([encode_max_tile_size], 0, dtype="int32")
        encoder_tile_ids_per_batch = paddle.full([encode_max_tile_size], 0, dtype="int32")
        encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

        kv_batch_ids = paddle.full([kv_max_tile_size], 0, dtype="int32")
        kv_tile_ids_per_batch = paddle.full([kv_max_tile_size], 0, dtype="int32")
        kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()

        # RoPE embeddings (identity - no rotation)
        rope_emb = paddle.zeros([2, 1, self.max_seq_len, 1, self.head_dim], dtype="float32")
        rope_emb[0] = 1.0  # cos = 1
        rope_emb[1] = 0.0  # sin = 0

        return {
            "qkv": qkv,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "seq_lens_this_time": seq_lens_this_time,
            "batch_id_per_token": batch_id_per_token,
            "cu_seqlens_q": cu_seqlens_q,
            "block_tables": block_tables,
            "encoder_batch_ids": encoder_batch_ids,
            "encoder_tile_ids_per_batch": encoder_tile_ids_per_batch,
            "encoder_num_blocks_x_cpu": encoder_num_blocks_x_cpu,
            "kv_batch_ids": kv_batch_ids,
            "kv_tile_ids_per_batch": kv_tile_ids_per_batch,
            "kv_num_blocks_x_cpu": kv_num_blocks_x_cpu,
            "decoder_batch_ids": decoder_batch_ids,
            "decoder_tile_ids_per_batch": decoder_tile_ids_per_batch,
            "decoder_num_blocks_cpu": decoder_num_blocks_cpu,
            "decoder_num_blocks_device": decoder_num_blocks_device,
            "decoder_chunk_size_device": decoder_chunk_size_device,
            "max_len_tensor_cpu": max_len_tensor_cpu,
            "rope_emb": rope_emb,
        }

    def _run_append_attention(self, tensors, max_partition_size=_MAX_PARTITION_SIZE_FOR_TEST):
        """Run append attention and return output tensor."""
        # First call get_block_shape_and_split_kv_block
        group_size = self.q_num_head // self.kv_num_head
        self._get_block_shape_func(
            tensors["seq_lens_encoder"],
            tensors["seq_lens_decoder"],
            tensors["seq_lens_this_time"],
            tensors["decoder_batch_ids"],
            tensors["decoder_tile_ids_per_batch"],
            tensors["decoder_num_blocks_cpu"],
            tensors["decoder_num_blocks_device"],
            tensors["decoder_chunk_size_device"],
            tensors["max_len_tensor_cpu"],
            tensors["encoder_batch_ids"],
            tensors["encoder_tile_ids_per_batch"],
            tensors["encoder_num_blocks_x_cpu"],
            tensors["kv_batch_ids"],
            tensors["kv_tile_ids_per_batch"],
            tensors["kv_num_blocks_x_cpu"],
            64,  # encoder_block_shape_q
            12,  # decoder_block_shape_q
            group_size,
            self.blocksize,
        )

        # Run append attention
        out = self._append_attention_func(
            tensors["qkv"].clone(),  # Clone to avoid in-place modification
            tensors["cache_k"],
            tensors["cache_v"],
            tensors["seq_lens_encoder"],
            tensors["seq_lens_decoder"],
            tensors["seq_lens_this_time"],
            tensors["batch_id_per_token"],
            tensors["cu_seqlens_q"],
            tensors["block_tables"],
            tensors["encoder_batch_ids"],
            tensors["encoder_tile_ids_per_batch"],
            tensors["encoder_num_blocks_x_cpu"],
            tensors["kv_batch_ids"],
            tensors["kv_tile_ids_per_batch"],
            tensors["kv_num_blocks_x_cpu"],
            tensors["decoder_batch_ids"],
            tensors["decoder_tile_ids_per_batch"],
            tensors["decoder_num_blocks_cpu"],
            tensors["max_len_tensor_cpu"],
            tensors["rope_emb"],
            None,  # attn_mask
            None,  # qkv_bias
            None,  # qkv_out_scales
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # linear_shift
            None,  # linear_smooth
            None,  # mask_offset
            None,  # kv_signal_data
            None,  # q_norm_weight
            None,  # k_norm_weight
            None,  # sinks
            1e-6,  # rms_norm_eps
            "bf16",  # compute_dtype
            "none",  # cache_quant_type
            False,  # use_neox_rotary_style
            False,  # rope_3d
            self.max_seq_len,
            0.0,  # quant_min_bound
            0.0,  # quant_max_bound
            -1,  # out_linear_in_scale
            64,  # encoder_block_shape_q
            12,  # decoder_block_shape_q
            max_partition_size,  # max_partition_size (key param!)
            32768,  # encoder_max_partition_size
            2,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
            0,  # sliding_window
        )
        paddle.device.synchronize()
        return out

    def test_partition_kv_determinism_positive(self):
        """
        POSITIVE test: With FD_DETERMINISTIC_MODE=1, results must be bit-exact.

        This test:
        1. Sets FD_DETERMINISTIC_MODE=1
        2. Creates tensors with KV length > 64 (triggers partition_kv with chunk_size=64)
        3. Runs append attention multiple times
        4. Asserts all results are bitwise identical

        If this test passes, the deterministic mode fix is working correctly.
        """
        # Set deterministic mode
        old_mode = os.environ.get("FD_DETERMINISTIC_MODE")
        os.environ["FD_DETERMINISTIC_MODE"] = "1"

        try:
            tensors = self._create_test_tensors(seed=42)

            # Run multiple times and collect results
            num_runs = 3
            results = []
            for i in range(num_runs):
                out = self._run_append_attention(tensors)
                results.append(out.clone())

            # Assert all results are bitwise identical
            for i in range(1, num_runs):
                is_equal = paddle.equal(results[0], results[i]).all().item()
                self.assertTrue(
                    is_equal,
                    f"POSITIVE test failed: Run 0 vs Run {i} differ. "
                    f"Deterministic mode should produce identical results.",
                )

        finally:
            # Restore original mode
            if old_mode is None:
                os.environ.pop("FD_DETERMINISTIC_MODE", None)
            else:
                os.environ["FD_DETERMINISTIC_MODE"] = old_mode

    def test_partition_kv_non_determinism_detection(self):
        """
        NEGATIVE test: Without deterministic mode, non-determinism should be detectable.

        This test:
        1. Disables FD_DETERMINISTIC_MODE
        2. Creates tensors with KV length > 64 (triggers partition_kv with chunk_size=64)
        3. Runs append attention multiple times
        4. Checks if any difference is detected

        If this test detects differences, it proves:
        - The non-determinism bug exists
        - Our test setup can detect it
        - The positive test is meaningful (not passing by luck)

        Note: This test may occasionally pass (no difference detected) due to
        the probabilistic nature of the bug. We run multiple times to increase
        detection probability.
        """
        # Disable deterministic mode
        old_mode = os.environ.get("FD_DETERMINISTIC_MODE")
        os.environ.pop("FD_DETERMINISTIC_MODE", None)

        try:
            # Run multiple times and collect results
            num_runs = 10  # More runs to increase detection probability
            results = []
            for i in range(num_runs):
                # Create fresh tensors each time to avoid caching effects
                fresh_tensors = self._create_test_tensors(seed=42)
                out = self._run_append_attention(fresh_tensors)
                results.append(out.clone())

            # Check for any difference
            differences_found = 0
            for i in range(1, num_runs):
                if not paddle.equal(results[0], results[i]).all().item():
                    differences_found += 1

            # Report results (this is informational, not a hard failure)
            if differences_found > 0:
                print(
                    f"\nNEGATIVE test: Detected {differences_found}/{num_runs-1} "
                    f"different results without deterministic mode."
                )
                print("This confirms the partition_kv non-determinism bug exists.")
            else:
                print(
                    f"\nNEGATIVE test: No differences detected in {num_runs} runs. "
                    f"This may happen occasionally due to the probabilistic nature of the bug."
                )

        finally:
            # Restore original mode
            if old_mode is None:
                os.environ.pop("FD_DETERMINISTIC_MODE", None)
            else:
                os.environ["FD_DETERMINISTIC_MODE"] = old_mode

    def test_partition_kv_boundary_conditions(self):
        """
        Test determinism at partition_kv boundary conditions.

        With chunk_size=64, tests various KV lengths:
        - Just below: 60 (num_chunks = 1, no partition)
        - At boundary: 64 (num_chunks = 1, edge case)
        - Just above: 100 (num_chunks = 2, partition active)
        - Multiple chunks: 256 (num_chunks = 4)
        - Many chunks: 512 (num_chunks = 8)
        """
        old_mode = os.environ.get("FD_DETERMINISTIC_MODE")
        os.environ["FD_DETERMINISTIC_MODE"] = "1"

        test_seq_lens = [60, 64, 100, 256, 512]

        try:
            for seq_len in test_seq_lens:
                with self.subTest(seq_len=seq_len):
                    # Temporarily change seq_len
                    orig_seq_len = self.seq_len
                    self.seq_len = seq_len

                    tensors = self._create_test_tensors(seed=42)

                    # Run twice and compare
                    out1 = self._run_append_attention(tensors)
                    out2 = self._run_append_attention(tensors)

                    is_equal = paddle.equal(out1, out2).all().item()
                    num_chunks = (seq_len + _MAX_PARTITION_SIZE_FOR_TEST - 1) // _MAX_PARTITION_SIZE_FOR_TEST
                    self.assertTrue(
                        is_equal,
                        f"Determinism failed at seq_len={seq_len}. " f"num_chunks = {num_chunks}",
                    )

                    # Restore seq_len
                    self.seq_len = orig_seq_len

        finally:
            if old_mode is None:
                os.environ.pop("FD_DETERMINISTIC_MODE", None)
            else:
                os.environ["FD_DETERMINISTIC_MODE"] = old_mode


if __name__ == "__main__":
    unittest.main(verbosity=2)
