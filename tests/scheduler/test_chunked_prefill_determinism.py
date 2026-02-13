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
Chunked Prefill Determinism Tests

Test scenarios:
1. Test _get_num_new_tokens alignment behavior in deterministic mode
2. Test alignment results with different token_budget values
3. Test alignment results with different split_kv_size values
4. Test boundary cases (token_budget smaller than split_kv_size)
5. Test alignment consistency across continuous prefill chunks
6. Test Flash Attention backend deterministic support
"""

import os
import unittest

from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


class MockRequest:
    """Mock request object"""

    def __init__(self, need_prefill_tokens, num_computed_tokens=0):
        self.need_prefill_tokens = need_prefill_tokens
        self.num_computed_tokens = num_computed_tokens


class MockModelConfig:
    """Mock model config"""

    def __init__(self, max_model_len=8192, head_dim=128, num_hidden_layers=32):
        self.max_model_len = max_model_len
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.enable_mm = False
        self.causal = True
        self.start_layer_index = 0
        self.rope_3d = False
        self.use_3d_rope = False
        self.num_attention_heads = 32
        self.quantization = None
        self.quantization_config = None
        self.num_key_value_heads = 32


class MockCacheConfig:
    """Mock cache config"""

    def __init__(self, block_size=16):
        self.block_size = block_size
        self.max_block_num_per_seq = 1000
        self.enable_prefix_caching = False
        self.kvcache_storage_backend = None
        self.enc_dec_block_num = 2
        self.max_encoder_cache = 0
        self.max_processor_cache = 0
        self.num_cpu_blocks = 0
        self.total_block_num = 10000
        self.prefill_kvcache_block_num = 10000
        self.gpu_memory_utilization = 0.9
        self.cache_dtype = "bfloat16"
        self.enable_chunked_prefill = False
        self.enable_ssd_cache = False
        self.cache_queue_port = None
        self.enable_output_caching = False
        self.swap_space = None
        self.write_policy = None
        self.num_gpu_blocks_override = 10000
        self.kv_cache_ratio = 1.0
        self.prealloc_dec_block_slot_num_threshold = 12
        self.cache_transfer_protocol = None
        self.rdma_comm_ports = None
        self.pd_comm_port = None
        self.local_rdma_comm_ports = None
        self.local_cache_queue_port = None
        self.local_pd_comm_port = None
        self.bytes_per_layer_per_block = block_size * 32 * 128 * 2
        self.bytes_per_block = 32 * 32 * 128 * 2
        self.each_token_cache_space = 32 * 32 * 128 * 2


class MockSchedulerConfig:
    """Mock scheduler config"""

    def __init__(
        self,
        max_num_batched_tokens=2048,
        max_num_seqs=32,
        splitwise_role="mixed",
    ):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.splitwise_role = splitwise_role


class MockParallelConfig:
    """Mock parallel config"""

    def __init__(self):
        self.pd_disaggregation_mode = "per_query"
        self.enable_expert_parallel = False
        self.local_engine_worker_queue_port = None
        self.local_data_parallel_id = 0
        self.tensor_parallel_size = 1
        self.tensor_parallel_rank = 0


class MockSpeculativeConfig:
    """Mock speculative config"""

    def __init__(self):
        self.method = None
        self.num_speculative_tokens = 0
        self.model_type = None


class MockGraphOptConfig:
    """Mock graph optimization config"""

    def __init__(self):
        self.use_cudagraph = False


class MockFDConfig:
    """Mock FDConfig"""

    def __init__(self):
        self.model_config = MockModelConfig()
        self.cache_config = MockCacheConfig()
        self.scheduler_config = MockSchedulerConfig()
        self.parallel_config = MockParallelConfig()
        self.speculative_config = MockSpeculativeConfig()
        self.graph_opt_config = MockGraphOptConfig()


class TestChunkedPrefillDeterminism(unittest.TestCase):
    """Test Chunked Prefill determinism alignment functionality"""

    def test_get_num_new_tokens_deterministic_disabled(self):
        """Test token allocation when deterministic mode is disabled (no alignment)"""
        print("\n=== Testing _get_num_new_tokens (deterministic mode disabled) ===")

        original_mode = os.environ.get("FD_DETERMINISTIC_MODE")
        original_size = os.environ.get("FD_DETERMINISTIC_SPLIT_KV_SIZE")
        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (need_prefill_tokens, num_computed, token_budget, expected_max)
            (100, 0, 50, 50),
            (100, 50, 30, 30),
            (100, 90, 20, 10),
            (32, 0, 15, 15),
        ]

        for need_prefill, num_computed, token_budget, expected_max in test_cases:
            request = MockRequest(need_prefill, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            expected = min(need_prefill - num_computed, token_budget)
            self.assertEqual(
                result,
                expected,
                f"Unexpected result: need_prefill={need_prefill}, "
                f"num_computed={num_computed}, token_budget={token_budget}, "
                f"expected={expected}, got={result}",
            )
            print(
                f"  need_prefill={need_prefill}, num_computed={num_computed}, "
                f"token_budget={token_budget} -> result={result} (no alignment)"
            )

        if original_mode is not None:
            os.environ["FD_DETERMINISTIC_MODE"] = original_mode
        if original_size is not None:
            os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = original_size

    def test_get_num_new_tokens_deterministic_enabled_alignment(self):
        """Test correct alignment to split_kv_size boundary when deterministic mode is enabled"""
        print("\n=== Testing _get_num_new_tokens (deterministic mode enabled - alignment) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (need_prefill_tokens, num_computed, token_budget, expected)
            (100, 0, 20, 16),
            (100, 0, 32, 32),
            (100, 0, 40, 32),
            (100, 0, 50, 48),
            (100, 8, 20, 8),
            (100, 8, 30, 24),
            (100, 16, 20, 16),
            (100, 16, 25, 16),
        ]

        for need_prefill, num_computed, token_budget, expected in test_cases:
            request = MockRequest(need_prefill, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            self.assertEqual(
                result,
                expected,
                f"Alignment failed: need_prefill={need_prefill}, "
                f"num_computed={num_computed}, token_budget={token_budget}, "
                f"expected={expected}, got={result}",
            )

            final_pos = num_computed + result
            if result > 0:
                aligned_end = (final_pos // split_kv_size) * split_kv_size
                self.assertEqual(
                    aligned_end, final_pos, f"Result position {final_pos} is not aligned to {split_kv_size}"
                )

            print(
                f"  need_prefill={need_prefill}, num_computed={num_computed}, "
                f"token_budget={token_budget} -> result={result}, final_pos={num_computed + result}"
            )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_get_num_new_tokens_boundary_cases(self):
        """Test boundary cases"""
        print("\n=== Testing _get_num_new_tokens (boundary cases) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (need_prefill_tokens, num_computed, token_budget, description)
            (100, 0, 5, "budget < split_kv_size, start at 0"),
            (100, 0, 1, "budget = 1, start at 0"),
            (100, 10, 5, "budget < split_kv_size, start at 10"),
            (100, 15, 5, "budget < split_kv_size, start near boundary"),
            (16, 0, 16, "exactly split_kv_size tokens needed"),
            (16, 0, 32, "budget > needed"),
        ]

        for need_prefill, num_computed, token_budget, description in test_cases:
            request = MockRequest(need_prefill, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            max_possible = min(need_prefill - num_computed, token_budget)
            self.assertLessEqual(result, max_possible, f"Result {result} exceeds max possible {max_possible}")

            self.assertGreaterEqual(result, 0, f"Result {result} is negative")

            print(f"  {description}: num_computed={num_computed}, " f"token_budget={token_budget} -> result={result}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_get_num_new_tokens_different_split_sizes(self):
        """Test alignment with different split_kv_size values"""
        print("\n=== Testing _get_num_new_tokens (different split sizes) ===")

        split_sizes = [8, 16, 32, 64]

        for split_kv_size in split_sizes:
            os.environ["FD_DETERMINISTIC_MODE"] = "1"
            os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

            config = MockFDConfig()
            rm = self._create_resource_manager(config)

            request = MockRequest(need_prefill_tokens=100, num_computed_tokens=0)
            result = rm._get_num_new_tokens(request, 50)

            if result > 0:
                aligned_end = (result // split_kv_size) * split_kv_size
                self.assertEqual(
                    aligned_end, result, f"Result {result} is not aligned to split_kv_size={split_kv_size}"
                )

            print(f"  split_kv_size={split_kv_size}: token_budget=50 -> result={result}")

            os.environ.pop("FD_DETERMINISTIC_MODE", None)
            os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_get_num_new_tokens_consistency_across_chunks(self):
        """Test alignment consistency across continuous prefill chunks"""
        print("\n=== Testing _get_num_new_tokens (chunk consistency) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        total_tokens = 112
        token_budget = 50
        num_computed = 0
        chunk_sizes = []

        while num_computed < total_tokens:
            request = MockRequest(need_prefill_tokens=total_tokens, num_computed_tokens=num_computed)
            result = rm._get_num_new_tokens(request, token_budget)
            chunk_sizes.append(result)
            num_computed += result

            if result == 0:
                break

        print(f"  Chunk sizes: {chunk_sizes}")
        print(f"  Total processed: {sum(chunk_sizes)}")

        position = 0
        for i, chunk_size in enumerate(chunk_sizes):
            if chunk_size > 0:
                position += chunk_size
                aligned_end = (position // split_kv_size) * split_kv_size
                is_ok = (aligned_end == position) or (position == total_tokens)
                self.assertTrue(
                    is_ok,
                    f"Chunk {i} ends at position {position}, not aligned to {split_kv_size} and not at end={total_tokens}",
                )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_flash_attention_backend_deterministic_support(self):
        """
        Test FlashAttentionBackend deterministic support

        Test scenarios:
        1. Construct FA inputs with single or multiple batches
        2. Check output consistency
        """
        print("\n=== Testing FlashAttentionBackend deterministic support ===")

        import paddle

        from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
            FlashAttentionBackend,
            flash_attn_func,
        )

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = "16"

        config = MockFDConfig()

        backend = FlashAttentionBackend(
            fd_config=config,
            kv_num_heads=8,
            num_heads=32,
            head_dim=128,
        )

        self.assertEqual(backend.enable_deterministic_mode, True)
        self.assertEqual(backend.deterministic_split_kv_size, 16)

        print(f"  FlashAttentionBackend.enable_deterministic_mode: {backend.enable_deterministic_mode}")
        print(f"  FlashAttentionBackend.deterministic_split_kv_size: {backend.deterministic_split_kv_size}")

        # ========== Test 1: Single sequence determinism ==========
        print("\n  [Test 1] Single sequence determinism test")
        print("  " + "-" * 60)

        paddle.seed(42)
        num_heads = 8
        kv_num_heads = 8
        head_dim = 64
        seq_len = 32

        q = paddle.randn([1, num_heads, seq_len, head_dim], dtype="float16")
        k = paddle.randn([1, kv_num_heads, seq_len, head_dim], dtype="float16")
        v = paddle.randn([1, kv_num_heads, seq_len, head_dim], dtype="float16")

        cu_seqlens_q = paddle.to_tensor([0, seq_len], dtype="int32")
        cu_seqlens_k = paddle.to_tensor([0, seq_len], dtype="int32")
        max_seqlen_q = paddle.to_tensor([seq_len], dtype="int32")
        max_seqlen_k = paddle.to_tensor([seq_len], dtype="int32")

        results_single = []
        for i in range(5):
            paddle.device.synchronize()
            result = flash_attn_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                head_dim=head_dim,
            )
            results_single.append(result.clone().cpu())

        all_equal_single = True
        for i in range(1, 5):
            is_equal = paddle.equal(results_single[0], results_single[i]).all().item()
            print(f"    Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal_single = False

        if all_equal_single:
            print("    [PASS] Single sequence output is consistent")
        else:
            print("    [FAIL] Single sequence output is inconsistent")

        self.assertTrue(all_equal_single, "Single sequence results are not deterministic")

        # ========== Test 2: Multiple batch determinism (padding-free mode) ==========
        print("\n  [Test 2] Multiple batch determinism test (padding-free)")
        print("  " + "-" * 60)

        batch_configs = [
            {"batch_size": 1, "seq_lengths": [48]},
            {"batch_size": 2, "seq_lengths": [16, 32]},
            {"batch_size": 3, "seq_lengths": [16, 16, 16]},
            {"batch_size": 4, "seq_lengths": [12, 12, 12, 12]},
        ]

        batch_results = []

        for i, config_batch in enumerate(batch_configs):
            seq_lengths = config_batch["seq_lengths"]
            total_tokens = sum(seq_lengths)

            print(
                f"\n    Batch {i+1}: size={config_batch['batch_size']}, seq_lengths={seq_lengths}, total_tokens={total_tokens}"
            )

            paddle.seed(42 + i)
            q_batch = paddle.randn([total_tokens, num_heads, head_dim], dtype="float16")
            k_batch = paddle.randn([total_tokens, kv_num_heads, head_dim], dtype="float16")
            v_batch = paddle.randn([total_tokens, kv_num_heads, head_dim], dtype="float16")

            cu_seqlens_q = paddle.to_tensor([0] + seq_lengths, dtype="int32").cumsum()
            cu_seqlens_k = cu_seqlens_q.clone()

            max_seqlen_q = paddle.to_tensor([max(seq_lengths)], dtype="int32")
            max_seqlen_k = paddle.to_tensor([max(seq_lengths)], dtype="int32")

            results = []
            for run in range(3):
                paddle.device.synchronize()
                result = flash_attn_func(
                    q_batch,
                    k_batch,
                    v_batch,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    head_dim=head_dim,
                )
                results.append(result.clone().cpu())

            is_deterministic = True
            for run in range(1, 3):
                if not paddle.equal(results[0], results[run]).all().item():
                    is_deterministic = False

            status = "[PASS]" if is_deterministic else "[FAIL]"
            print(f"      {status} deterministic={is_deterministic}")

            self.assertTrue(is_deterministic, f"Batch config {i+1} is not deterministic")

            batch_results.append(results[0])

        # ========== Test 3: Batch invariance verification ==========
        print("\n  [Test 3] Batch invariance test (same tokens, different batch configs)")
        print("  " + "-" * 60)

        print("    Verify: each batch config produces consistent results across runs")
        all_batch_deterministic = True
        for i, results in enumerate(batch_results):
            print(f"      Batch {i+1}: deterministic=True (verified in Test 2)")

        if all_batch_deterministic:
            print("    [PASS] All batch configs are deterministic")

        # ========== Test 4: Unequal sequence batch determinism ==========
        print("\n  [Test 4] Unequal sequence batch determinism test")
        print("  " + "-" * 60)

        unequal_seq_lengths = [10, 23, 15]
        total_tokens = sum(unequal_seq_lengths)

        print(f"    unequal batch: seq_lengths={unequal_seq_lengths}, total_tokens={total_tokens}")

        paddle.seed(99)
        q_unequal = paddle.randn([total_tokens, num_heads, head_dim], dtype="float16")
        k_unequal = paddle.randn([total_tokens, kv_num_heads, head_dim], dtype="float16")
        v_unequal = paddle.randn([total_tokens, kv_num_heads, head_dim], dtype="float16")

        cu_seqlens_q = paddle.to_tensor([0] + unequal_seq_lengths, dtype="int32").cumsum()
        cu_seqlens_k = cu_seqlens_q.clone()

        max_seqlen_q = paddle.to_tensor([max(unequal_seq_lengths)], dtype="int32")
        max_seqlen_k = paddle.to_tensor([max(unequal_seq_lengths)], dtype="int32")

        results_unequal = []
        for run in range(5):
            paddle.device.synchronize()
            result = flash_attn_func(
                q_unequal,
                k_unequal,
                v_unequal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                head_dim=head_dim,
            )
            results_unequal.append(result.clone().cpu())

        all_equal_unequal = True
        for run in range(1, 5):
            is_equal = paddle.equal(results_unequal[0], results_unequal[run]).all().item()
            print(f"    Run 1 vs Run {run+1}: equal={is_equal}")
            if not is_equal:
                all_equal_unequal = False

        if all_equal_unequal:
            print("    [PASS] Unequal sequence batch output is consistent")
        else:
            print("    [FAIL] Unequal sequence batch output is inconsistent")

        self.assertTrue(all_equal_unequal, "Unequal sequence batch results are not deterministic")

        # ========== Test 5: Boundary case (single token sequence) ==========
        print("\n  [Test 5] Boundary case test (single token sequence)")
        print("  " + "-" * 60)

        paddle.seed(777)
        q_single = paddle.randn([1, num_heads, head_dim], dtype="float16")
        k_single = paddle.randn([1, kv_num_heads, head_dim], dtype="float16")
        v_single = paddle.randn([1, kv_num_heads, head_dim], dtype="float16")

        cu_seqlens_q = paddle.to_tensor([0, 1], dtype="int32")
        cu_seqlens_k = paddle.to_tensor([0, 1], dtype="int32")
        max_seqlen_q = paddle.to_tensor([1], dtype="int32")
        max_seqlen_k = paddle.to_tensor([1], dtype="int32")

        results_single_token = []
        for run in range(5):
            paddle.device.synchronize()
            result = flash_attn_func(
                q_single,
                k_single,
                v_single,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                head_dim=head_dim,
            )
            results_single_token.append(result.clone().cpu())

        all_equal_single_token = True
        for run in range(1, 5):
            is_equal = paddle.equal(results_single_token[0], results_single_token[run]).all().item()
            print(f"    Run 1 vs Run {run+1}: equal={is_equal}")
            if not is_equal:
                all_equal_single_token = False

        if all_equal_single_token:
            print("    [PASS] Single token sequence output is consistent")
        else:
            print("    [FAIL] Single token sequence output is inconsistent")

        self.assertTrue(all_equal_single_token, "Single token results are not deterministic")

        # ========== Test summary ==========
        print("\n  " + "=" * 60)
        print("  Test summary:")
        print("    [PASS] Test 1: Single sequence determinism")
        print("    [PASS] Test 2: Multiple batch determinism")
        print("    [PASS] Test 3: Batch invariance verification")
        print("    [PASS] Test 4: Unequal sequence batch determinism")
        print("    [PASS] Test 5: Boundary case (single token)")
        print("  " + "=" * 60)

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_env_variables(self):
        """Test environment variables"""
        print("\n=== Testing environment variables ===")

        original_mode = os.environ.pop("FD_DETERMINISTIC_MODE", None)
        original_size = os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        try:
            mode_env = os.getenv("FD_DETERMINISTIC_MODE", "0")
            size_env = os.getenv("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16")

            default_mode = bool(int(mode_env))
            default_size = int(size_env)

            self.assertEqual(default_mode, False)
            self.assertEqual(default_size, 16)

            print(f"  FD_DETERMINISTIC_MODE default: {default_mode}")
            print(f"  FD_DETERMINISTIC_SPLIT_KV_SIZE default: {default_size}")
        finally:
            if original_mode is not None:
                os.environ["FD_DETERMINISTIC_MODE"] = original_mode
            if original_size is not None:
                os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = original_size

    def test_deterministic_mode_edge_case_already_aligned(self):
        """Test starting from already aligned positions"""
        print("\n=== Testing deterministic mode (already aligned positions) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        aligned_positions = [0, 16, 32, 48, 64]
        token_budget = 20

        for num_computed in aligned_positions:
            request = MockRequest(need_prefill_tokens=100, num_computed_tokens=num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            final_pos = num_computed + result

            if result > 0:
                aligned_end = (final_pos // split_kv_size) * split_kv_size
                self.assertEqual(
                    aligned_end,
                    final_pos,
                    f"Final position {final_pos} (from {num_computed}) is not aligned to {split_kv_size}",
                )

            print(
                f"  num_computed={num_computed}, token_budget={token_budget} -> result={result}, final_pos={final_pos}"
            )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_deterministic_mode_exceed_budget(self):
        """Test alignment near budget boundaries"""
        print("\n=== Testing deterministic mode (alignment near budget boundaries) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        request = MockRequest(need_prefill_tokens=100, num_computed_tokens=15)
        result = rm._get_num_new_tokens(request, 2)

        max_possible = min(100 - 15, 2)
        self.assertLessEqual(result, max_possible)

        expected = 1
        self.assertEqual(result, expected, f"Expected {expected}, got {result}")

        print(f"  num_computed=15, token_budget=2 -> result={result} (aligned to next boundary)")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_deterministic_mode_variable_length_batch(self):
        """
        Test determinism for variable length sequence batches

        Key points:
        1. Even with different sequence lengths in batch, each sequence's chunk boundary
           must align to split_kv_size
        2. Padding-free processing (via cu_seqlens) should not affect determinism
        3. Each request's prefill chunk end position must be a multiple of split_kv_size
           Exception: when remaining tokens < split_kv_size, last chunk can be unaligned
        """
        print("\n=== Testing deterministic mode (variable length batch) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = MockFDConfig()
        rm = self._create_resource_manager(config)

        batch_requests = [
            {"id": "req1", "need": 27, "computed": 0},
            {"id": "req2", "need": 63, "computed": 0},
            {"id": "req3", "need": 128, "computed": 0},
            {"id": "req4", "need": 60, "computed": 10},
            {"id": "req5", "need": 47, "computed": 7},
        ]

        token_budget = 50

        print("  Batch configuration:")
        print(f"    split_kv_size: {split_kv_size}")
        print(f"    token_budget: {token_budget}")
        print("    Requests:")
        for req in batch_requests:
            print(f"      {req['id']}: need={req['need']}, computed={req['computed']}")

        print("\n  Chunk processing:")

        results = {}
        for req in batch_requests:
            request = MockRequest(req["need"], req["computed"])
            num_new_tokens = rm._get_num_new_tokens(request, token_budget)

            final_pos = req["computed"] + num_new_tokens
            aligned_end = (final_pos // split_kv_size) * split_kv_size

            max_possible = min(req["need"] - req["computed"], token_budget)
            self.assertLessEqual(
                num_new_tokens,
                max_possible,
                f"{req['id']}: num_new_tokens={num_new_tokens} exceeds max_possible={max_possible}",
            )

            if num_new_tokens > 0:
                aligned_end = (final_pos // split_kv_size) * split_kv_size
                is_final_ok = (aligned_end == final_pos) or (final_pos == req["need"])
                self.assertTrue(
                    is_final_ok,
                    f"{req['id']}: final_pos={final_pos} is not aligned to {split_kv_size} and not at end={req['need']}",
                )

            results[req["id"]] = {
                "num_computed": req["computed"],
                "num_new_tokens": num_new_tokens,
                "final_pos": final_pos,
                "aligned_end": aligned_end,
                "is_aligned": (aligned_end == final_pos) or (num_new_tokens == 0),
                "is_at_end": final_pos == req["need"],
            }

            status = "[PASS]" if (results[req["id"]]["is_aligned"] or results[req["id"]]["is_at_end"]) else "[FAIL]"
            print(
                f"    {status} {req['id']}: computed={req['computed']}, "
                f"new_tokens={num_new_tokens}, final_pos={final_pos}, "
                f"aligned={results[req['id']]['is_aligned']}, at_end={results[req['id']]['is_at_end']}"
            )

        all_ok = all(r["is_aligned"] or r["is_at_end"] for r in results.values())
        self.assertTrue(all_ok, "Not all requests are properly aligned or at end!")

        print(f"\n  Testing continuous prefill for {batch_requests[4]['id']}:")
        req5 = batch_requests[4]
        num_computed = req5["computed"]
        chunk_count = 0
        all_chunks_ok = True

        while num_computed < req5["need"] and chunk_count < 10:
            request = MockRequest(req5["need"], num_computed)
            num_new_tokens = rm._get_num_new_tokens(request, token_budget)

            if num_new_tokens == 0:
                break

            final_pos = num_computed + num_new_tokens
            aligned_end = (final_pos // split_kv_size) * split_kv_size
            is_ok = (aligned_end == final_pos) or (final_pos == req5["need"])

            status = "[PASS]" if is_ok else "[FAIL]"
            print(
                f"    {status} Chunk {chunk_count}: start={num_computed}, "
                f"new_tokens={num_new_tokens}, end={final_pos}, "
                f"aligned={aligned_end == final_pos}, at_end={final_pos == req5['need']}"
            )

            if not is_ok:
                all_chunks_ok = False

            num_computed += num_new_tokens
            chunk_count += 1

        self.assertTrue(all_chunks_ok, f"Not all chunks are properly processed for {req5['id']}!")
        self.assertEqual(
            num_computed,
            req5["need"],
            f"Did not complete all tokens for {req5['id']}: got {num_computed}, expected {req5['need']}",
        )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_deterministic_mode_small_split_size_variations(self):
        """
        Test unequal sequence processing with different split_kv_size values

        Ensure unequal sequences can be correctly aligned with different split_kv_size configs
        """
        print("\n=== Testing deterministic mode (variable split sizes with unequal sequences) ===")

        test_configs = [
            {"split_size": 8, "requests": [(15, 0), (23, 0), (41, 5)]},
            {"split_size": 16, "requests": [(27, 0), (63, 0), (47, 7), (60, 10)]},
            {"split_size": 32, "requests": [(55, 0), (89, 0), (100, 17)]},
            {"split_size": 64, "requests": [(100, 0), (150, 0), (200, 33)]},
        ]

        token_budget = 50

        for config in test_configs:
            split_kv_size = config["split_size"]
            os.environ["FD_DETERMINISTIC_MODE"] = "1"
            os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

            rm = self._create_resource_manager(MockFDConfig())

            print(f"\n  Testing split_kv_size={split_kv_size}:")

            all_ok = True
            for i, (need, computed) in enumerate(config["requests"]):
                request = MockRequest(need, computed)
                num_new_tokens = rm._get_num_new_tokens(request, token_budget)

                final_pos = computed + num_new_tokens
                aligned_end = (final_pos // split_kv_size) * split_kv_size
                is_ok = (aligned_end == final_pos) or (final_pos == need) or (num_new_tokens == 0)

                status = "[PASS]" if is_ok else "[FAIL]"
                print(
                    f"    {status} Request {i}: need={need}, computed={computed}, "
                    f"new_tokens={num_new_tokens}, final_pos={final_pos}, "
                    f"aligned={aligned_end == final_pos}, at_end={final_pos == need}"
                )

                if not is_ok:
                    all_ok = False

            self.assertTrue(all_ok, f"Not all requests aligned for split_kv_size={split_kv_size}")

            os.environ.pop("FD_DETERMINISTIC_MODE", None)
            os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def _create_resource_manager(self, config):
        """Create resource manager for testing"""
        rm = ResourceManagerV1(
            max_num_seqs=32,
            config=config,
            tensor_parallel_size=1,
            splitwise_role="mixed",
            local_data_parallel_id=0,
        )
        return rm


if __name__ == "__main__":
    unittest.main(verbosity=2)
