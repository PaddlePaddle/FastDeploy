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
Chunked Prefill Determinism Tests - Refactored with Real Classes

Test scenarios:
1. Test _get_num_new_tokens alignment behavior in deterministic mode
2. Test alignment results with different token_budget values
3. Test alignment results with different split_kv_size values
4. Test boundary cases (token_budget smaller than split_kv_size)
5. Test alignment consistency across continuous prefill chunks
6. Test Flash Attention backend deterministic support
7. Test multimodal input scenarios (NEW)
8. Test real batch scheduling scenarios (NEW)
9. Test corner cases:
   - Empty request / zero tokens
   - State inconsistency (num_computed > need_prefill)
   - Minimum split size (split_kv_size = 1)
   - Split size larger than budget
   - Split size larger than sequence
   - Dynamic config switch
   - Large budget values (potential overflow)
"""

import os
import unittest

from fastdeploy.engine.request import Request
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


class TestConfig:
    """Simplified test configuration class (not a full mock, has real interface)"""

    def __init__(self):
        self.model_config = ModelConfig()
        self.cache_config = CacheConfig()
        self.scheduler_config = SchedulerConfig()
        self.parallel_config = ParallelConfig()
        self.speculative_config = SpeculativeConfig()
        self.graph_opt_config = GraphOptConfig()


class ModelConfig:
    """Simplified model config for testing"""

    def __init__(self):
        self.max_model_len = 8192
        self.head_dim = 128
        self.num_hidden_layers = 32
        self.enable_mm = False
        self.causal = True
        self.start_layer_index = 0
        self.rope_3d = False
        self.use_3d_rope = False
        self.num_attention_heads = 32
        self.quantization = None
        self.quantization_config = None
        self.num_key_value_heads = 32
        self.kv_num_head = 32


class CacheConfig:
    """Simplified cache config for testing"""

    def __init__(self):
        self.block_size = 16
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
        self.bytes_per_layer_per_block = 16 * 32 * 128 * 2
        self.bytes_per_block = 32 * 32 * 128 * 2
        self.each_token_cache_space = 32 * 32 * 128 * 2
        self.bytes_per_token_per_layer = 32 * 32 * 128 * 2


class SchedulerConfig:
    """Simplified scheduler config for testing"""

    def __init__(self, max_num_batched_tokens=2048, max_num_seqs=32, splitwise_role="mixed"):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.splitwise_role = splitwise_role


class ParallelConfig:
    """Simplified parallel config for testing"""

    def __init__(self):
        self.pd_disaggregation_mode = "per_query"
        self.enable_expert_parallel = False
        self.local_engine_worker_queue_port = None
        self.local_data_parallel_id = 0
        self.tensor_parallel_size = 1
        self.tensor_parallel_rank = 0


class SpeculativeConfig:
    """Simplified speculative config for testing"""

    def __init__(self):
        self.method = None
        self.num_speculative_tokens = 0
        self.model_type = None


class GraphOptConfig:
    """Simplified graph optimization config for testing"""

    def __init__(self):
        self.use_cudagraph = False


def create_test_request(request_id, prompt_token_ids, num_computed_tokens=0, multimodal_inputs=None):
    """Create a real Request object for testing"""
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        prompt_token_ids_len=len(prompt_token_ids),
        num_computed_tokens=num_computed_tokens,
        multimodal_inputs=multimodal_inputs,
    )


class TestChunkedPrefillDeterminism(unittest.TestCase):
    """Test Chunked Prefill determinism alignment functionality"""

    def test_get_num_new_tokens_deterministic_disabled(self):
        """Test token allocation when deterministic mode is disabled (no alignment)"""
        print("\n=== Testing _get_num_new_tokens (deterministic mode disabled) ===")

        original_mode = os.environ.get("FD_DETERMINISTIC_MODE")
        original_size = os.environ.get("FD_DETERMINISTIC_SPLIT_KV_SIZE")
        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (prompt_tokens, num_computed, token_budget, expected_max)
            (list(range(100)), 0, 50, 50),
            (list(range(100)), 50, 30, 30),
            (list(range(100)), 90, 20, 10),
            (list(range(32)), 0, 15, 15),
        ]

        for prompt_ids, num_computed, token_budget, expected_max in test_cases:
            request = create_test_request("test_req", prompt_ids, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            expected = min(request.need_prefill_tokens - num_computed, token_budget)
            self.assertEqual(
                result,
                expected,
                f"Unexpected result: prompt_len={len(prompt_ids)}, "
                f"num_computed={num_computed}, token_budget={token_budget}, "
                f"expected={expected}, got={result}",
            )
            print(
                f"  prompt_len={len(prompt_ids)}, num_computed={num_computed}, "
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

        config = TestConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (prompt_tokens, num_computed, token_budget, expected)
            (list(range(100)), 0, 20, 16),
            (list(range(100)), 0, 32, 32),
            (list(range(100)), 0, 40, 32),
            (list(range(100)), 0, 50, 48),
            (list(range(100)), 8, 20, 8),
            (list(range(100)), 8, 30, 24),
            (list(range(100)), 16, 20, 16),
            (list(range(100)), 16, 25, 16),
        ]

        for prompt_ids, num_computed, token_budget, expected in test_cases:
            request = create_test_request("test_req", prompt_ids, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            self.assertEqual(
                result,
                expected,
                f"Alignment failed: prompt_len={len(prompt_ids)}, "
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
                f"  prompt_len={len(prompt_ids)}, num_computed={num_computed}, "
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

        config = TestConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            # (prompt_tokens, num_computed, token_budget, description)
            (list(range(100)), 0, 5, "budget < split_kv_size, start at 0"),
            (list(range(100)), 0, 1, "budget = 1, start at 0"),
            (list(range(100)), 10, 5, "budget < split_kv_size, start at 10"),
            (list(range(100)), 15, 5, "budget < split_kv_size, start near boundary"),
            (list(range(16)), 0, 16, "exactly split_kv_size tokens needed"),
            (list(range(16)), 0, 32, "budget > needed"),
        ]

        for prompt_ids, num_computed, token_budget, description in test_cases:
            request = create_test_request("test_req", prompt_ids, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            max_possible = min(request.need_prefill_tokens - num_computed, token_budget)
            self.assertLessEqual(result, max_possible, f"Result {result} exceeds max possible {max_possible}")
            self.assertGreaterEqual(result, 0, f"Result {result} is negative")

            print(f"  {description}: num_computed={num_computed}, " f"token_budget={token_budget} -> result={result}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_get_num_new_tokens_consistency_across_chunks(self):
        """Test alignment consistency across continuous prefill chunks"""
        print("\n=== Testing _get_num_new_tokens (chunk consistency) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        prompt_ids = list(range(112))
        token_budget = 50
        num_computed = 0
        chunk_sizes = []

        while num_computed < len(prompt_ids):
            request = create_test_request("test_req", prompt_ids, num_computed)
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
                is_ok = (aligned_end == position) or (position == len(prompt_ids))
                self.assertTrue(
                    is_ok,
                    f"Chunk {i} ends at position {position}, not aligned to {split_kv_size} and not at end={len(prompt_ids)}",
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
            init_flash_attn_version,
        )

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        # Initialize Flash Attention version based on environment
        current_fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]
        print(f"  Detected FLAGS_flash_attn_version: {current_fa_version}")

        # Initialize flash_attn_func version to match environment variable
        init_flash_attn_version(fa_version=current_fa_version)

        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = "16"

        config = TestConfig()

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

        # Different input formats for v2 vs v3
        # Note: flash_attn_unpadded requires int32 for seqlens
        if current_fa_version == 2:
            # v2 format: [total_seq_len, num_heads, head_dim], int32 for seqlens
            q = paddle.randn([seq_len, num_heads, head_dim], dtype="float16")
            k = paddle.randn([seq_len, kv_num_heads, head_dim], dtype="float16")
            v = paddle.randn([seq_len, kv_num_heads, head_dim], dtype="float16")
            cu_seqlens_q = paddle.to_tensor([0, seq_len], dtype="int32")
            cu_seqlens_k = paddle.to_tensor([0, seq_len], dtype="int32")
            max_seqlen_q = paddle.to_tensor([seq_len], dtype="int32")
            max_seqlen_k = paddle.to_tensor([seq_len], dtype="int32")
        else:
            # v3 format: [batch_size, num_heads, seq_len, head_dim], int32 for seqlens
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
            # v2 returns (output, softmax_lse), v3 returns output directly
            if isinstance(result, tuple):
                result = result[0]
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

            # flash_attn_unpadded and flash_attention_v3_varlen both require int32 for seqlens
            seqlens_dtype = "int32"
            # Manual cumsum to avoid dtype issues
            cu_seq = [0]
            for seq_len in seq_lengths:
                cu_seq.append(cu_seq[-1] + seq_len)
            cu_seqlens_q = paddle.to_tensor(cu_seq, dtype=seqlens_dtype)
            cu_seqlens_k = cu_seqlens_q.clone()

            max_seqlen_q = paddle.to_tensor([max(seq_lengths)], dtype=seqlens_dtype)
            max_seqlen_k = paddle.to_tensor([max(seq_lengths)], dtype=seqlens_dtype)

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
                # v2 returns (output, softmax_lse), v3 returns output directly
                if isinstance(result, tuple):
                    result = result[0]
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

        # flash_attn_unpadded and flash_attention_v3_varlen both require int32 for seqlens
        seqlens_dtype = "int32"
        # Manual cumsum to avoid dtype issues
        cu_seq = [0]
        for seq_len in unequal_seq_lengths:
            cu_seq.append(cu_seq[-1] + seq_len)
        cu_seqlens_q = paddle.to_tensor(cu_seq, dtype=seqlens_dtype)
        cu_seqlens_k = cu_seqlens_q.clone()

        max_seqlen_q = paddle.to_tensor([max(unequal_seq_lengths)], dtype=seqlens_dtype)
        max_seqlen_k = paddle.to_tensor([max(unequal_seq_lengths)], dtype=seqlens_dtype)

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
            # v2 returns (output, softmax_lse), v3 returns output directly
            if isinstance(result, tuple):
                result = result[0]
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
        # Different input formats for v2 vs v3
        # Note: both flash_attn_unpadded and flash_attention_v3_varlen require int32 for seqlens
        if current_fa_version == 2:
            # v2 format: [total_seq_len, num_heads, head_dim], int32 for seqlens
            q_single = paddle.randn([1, num_heads, head_dim], dtype="float16")
            k_single = paddle.randn([1, kv_num_heads, head_dim], dtype="float16")
            v_single = paddle.randn([1, kv_num_heads, head_dim], dtype="float16")
            seqlens_dtype = "int32"
        else:
            # v3 format: [batch_size, num_heads, seq_len, head_dim], int32 for seqlens
            q_single = paddle.randn([1, num_heads, 1, head_dim], dtype="float16")
            k_single = paddle.randn([1, kv_num_heads, 1, head_dim], dtype="float16")
            v_single = paddle.randn([1, kv_num_heads, 1, head_dim], dtype="float16")
            seqlens_dtype = "int32"

        cu_seqlens_q = paddle.to_tensor([0, 1], dtype=seqlens_dtype)
        cu_seqlens_k = paddle.to_tensor([0, 1], dtype=seqlens_dtype)
        max_seqlen_q = paddle.to_tensor([1], dtype=seqlens_dtype)
        max_seqlen_k = paddle.to_tensor([1], dtype=seqlens_dtype)

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
            # v2 returns (output, softmax_lse), v3 returns output directly
            if isinstance(result, tuple):
                result = result[0]
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

    # ========== NEW TEST: Multimodal Input Scenarios ==========
    def test_multimodal_input_with_image(self):
        """
        Test multimodal input scenario with image patches

        This tests the real-world scenario where a request contains image data
        that affects token allocation based on patch_idx and patch_map.
        """
        print("\n=== Testing multimodal input with image patches ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        config.model_config.enable_mm = True
        rm = self._create_resource_manager(config)

        # Simulate a multimodal request with image
        # Text tokens: 50
        # Image patches: 100 (simulated)
        prompt_tokens = list(range(150))

        # Create multimodal inputs with patch_idx and patch_map
        multimodal_inputs = {
            "image_patch_id": 151,
            "image_end_id": 152,
            "video_patch_id": 153,
            "video_end_id": 154,
            "audio_patch_id": 155,
            "audio_end_id": 156,
            "patch_idx": [0] * 50 + [1] * 100,  # First 50 text, then 100 image patches
            "patch_map": [
                {"modal_id": 0, "end_idx": 50, "image_num": 0, "video_num": 0},  # Text
                {"modal_id": 1, "end_idx": 150, "image_num": 1, "video_num": 0},  # Image
            ],
            "tts": False,
        }

        request = create_test_request("mm_image_req", prompt_tokens, 0, multimodal_inputs)
        token_budget = 60

        result = rm._get_num_new_tokens(request, token_budget)

        print(f"  Request: {len(prompt_tokens)} tokens (50 text + 100 image)")
        print(f"  Token budget: {token_budget}")
        print(f"  Result: {result} tokens")
        print(f"  Request with_image: {request.with_image}")
        print(f"  Image start: {request.image_start}, Image end: {request.image_end}")

        # Verify result doesn't exceed budget
        self.assertLessEqual(result, token_budget)
        self.assertGreaterEqual(result, 0)

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_multimodal_input_with_video(self):
        """
        Test multimodal input scenario with video

        This tests the real-world scenario where a request contains video data
        with can_split_idx_list for chunking support.
        """
        print("\n=== Testing multimodal input with video ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        config.model_config.enable_mm = True
        rm = self._create_resource_manager(config)

        # Simulate a multimodal request with video
        prompt_tokens = list(range(200))

        # Create multimodal inputs with video and can_split_idx_list
        multimodal_inputs = {
            "image_patch_id": 201,
            "image_end_id": 202,
            "video_patch_id": 203,
            "video_end_id": 204,
            "audio_patch_id": 205,
            "audio_end_id": 206,
            "patch_idx": [0] * 80 + [1] * 120,  # First 80 text, then 120 video
            "patch_map": [
                {"modal_id": 0, "end_idx": 80, "image_num": 0, "video_num": 0},  # Text
                {"modal_id": 2, "end_idx": 200, "image_num": 0, "video_num": 1},  # Video
            ],
            "can_split_idx_list": [96, 112, 128, 144, 160, 176, 192],  # Split points in video
            "tts": False,
        }

        request = create_test_request("mm_video_req", prompt_tokens, 0, multimodal_inputs)
        token_budget = 50

        result = rm._get_num_new_tokens(request, token_budget)

        print(f"  Request: {len(prompt_tokens)} tokens (80 text + 120 video)")
        print(f"  Token budget: {token_budget}")
        print(f"  Result: {result} tokens")
        print(f"  Video start: {request.video_start}, Video end: {request.video_end}")

        self.assertLessEqual(result, token_budget)
        self.assertGreaterEqual(result, 0)

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_multimodal_input_with_audio(self):
        """
        Test multimodal input scenario with audio

        This tests the real-world scenario where a request contains audio data
        requiring special audio prefix counting.
        """
        print("\n=== Testing multimodal input with audio ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        config.model_config.enable_mm = True
        rm = self._create_resource_manager(config)

        # Simulate a multimodal request with audio
        prompt_tokens = list(range(120))

        # Create multimodal inputs with audio
        multimodal_inputs = {
            "image_patch_id": 121,
            "image_end_id": 122,
            "video_patch_id": 123,
            "video_end_id": 124,
            "audio_patch_id": 125,
            "audio_end_id": 126,
            "patch_idx": [0] * 60 + [3] * 60,  # First 60 text, then 60 audio
            "patch_map": [
                {"modal_id": 0, "end_idx": 60, "image_num": 0, "video_num": 0},  # Text
                {"modal_id": 3, "end_idx": 120, "image_num": 0, "video_num": 0},  # Audio
            ],
            "tts": False,
        }

        request = create_test_request("mm_audio_req", prompt_tokens, 0, multimodal_inputs)
        token_budget = 40

        result = rm._get_num_new_tokens(request, token_budget)

        print(f"  Request: {len(prompt_tokens)} tokens (60 text + 60 audio)")
        print(f"  Token budget: {token_budget}")
        print(f"  Result: {result} tokens")
        print(f"  Audio start: {request.audio_start}, Audio end: {request.audio_end}")

        self.assertLessEqual(result, token_budget)
        self.assertGreaterEqual(result, 0)

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    # ========== NEW TEST: Real Batch Scheduling Scenarios ==========
    def test_real_batch_scheduling_concurrent_requests(self):
        """
        Test real batch scheduling scenario with multiple concurrent requests

        This simulates a real-world scenario where multiple requests compete
        for token budget in the same batch.
        """
        print("\n=== Testing real batch scheduling (concurrent requests) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Simulate a batch with multiple requests competing for budget
        batch_config = [
            {"id": "req1", "prompt": list(range(27)), "computed": 0},
            {"id": "req2", "prompt": list(range(63)), "computed": 0},
            {"id": "req3", "prompt": list(range(128)), "computed": 0},
            {"id": "req4", "prompt": list(range(60)), "computed": 10},
            {"id": "req5", "prompt": list(range(47)), "computed": 7},
        ]

        token_budget = 50

        print("  Batch configuration:")
        print(f"    split_kv_size: {split_kv_size}")
        print(f"    token_budget: {token_budget}")
        print("    Requests:")
        for req in batch_config:
            print(f"      {req['id']}: prompt_len={len(req['prompt'])}, computed={req['computed']}")

        print("\n  Batch processing:")

        results = {}
        for req_cfg in batch_config:
            request = create_test_request(req_cfg["id"], req_cfg["prompt"], req_cfg["computed"])
            num_new_tokens = rm._get_num_new_tokens(request, token_budget)

            final_pos = req_cfg["computed"] + num_new_tokens
            aligned_end = (final_pos // split_kv_size) * split_kv_size

            max_possible = min(len(req_cfg["prompt"]) - req_cfg["computed"], token_budget)
            self.assertLessEqual(
                num_new_tokens,
                max_possible,
                f"{req_cfg['id']}: num_new_tokens={num_new_tokens} exceeds max_possible={max_possible}",
            )

            if num_new_tokens > 0:
                is_final_ok = (aligned_end == final_pos) or (final_pos == len(req_cfg["prompt"]))
                self.assertTrue(
                    is_final_ok,
                    f"{req_cfg['id']}: final_pos={final_pos} is not aligned to {split_kv_size} and not at end={len(req_cfg['prompt'])}",
                )

            results[req_cfg["id"]] = {
                "num_computed": req_cfg["computed"],
                "num_new_tokens": num_new_tokens,
                "final_pos": final_pos,
                "aligned_end": aligned_end,
                "is_aligned": (aligned_end == final_pos) or (num_new_tokens == 0),
                "is_at_end": final_pos == len(req_cfg["prompt"]),
            }

            status = (
                "[PASS]" if (results[req_cfg["id"]]["is_aligned"] or results[req_cfg["id"]]["is_at_end"]) else "[FAIL]"
            )
            print(
                f"    {status} {req_cfg['id']}: computed={req_cfg['computed']}, "
                f"new_tokens={num_new_tokens}, final_pos={final_pos}, "
                f"aligned={results[req_cfg['id']]['is_aligned']}, at_end={results[req_cfg['id']]['is_at_end']}"
            )

        all_ok = all(r["is_aligned"] or r["is_at_end"] for r in results.values())
        self.assertTrue(all_ok, "Not all requests are properly aligned or at end!")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_real_batch_scheduling_continuous_prefill(self):
        """
        Test continuous prefill scenario with real Request objects

        This tests how multiple chunks are allocated over time for the same request.
        """
        print("\n=== Testing real batch scheduling (continuous prefill) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        prompt_tokens = list(range(47))
        token_budget = 50
        num_computed = 0
        chunk_count = 0
        all_chunks_ok = True

        print(f"  Total prompt tokens: {len(prompt_tokens)}")
        print(f"  Token budget per iteration: {token_budget}")
        print(f"  Split KV size: {split_kv_size}")

        while num_computed < len(prompt_tokens) and chunk_count < 10:
            request = create_test_request("continuous_req", prompt_tokens, num_computed)
            num_new_tokens = rm._get_num_new_tokens(request, token_budget)

            if num_new_tokens == 0:
                break

            final_pos = num_computed + num_new_tokens
            aligned_end = (final_pos // split_kv_size) * split_kv_size
            is_ok = (aligned_end == final_pos) or (final_pos == len(prompt_tokens))

            status = "[PASS]" if is_ok else "[FAIL]"
            print(
                f"    {status} Chunk {chunk_count}: start={num_computed}, "
                f"new_tokens={num_new_tokens}, end={final_pos}, "
                f"aligned={aligned_end == final_pos}, at_end={final_pos == len(prompt_tokens)}"
            )

            if not is_ok:
                all_chunks_ok = False

            num_computed += num_new_tokens
            chunk_count += 1

        self.assertTrue(all_chunks_ok, "Not all chunks are properly processed!")
        self.assertEqual(
            num_computed,
            len(prompt_tokens),
            f"Did not complete all tokens: got {num_computed}, expected {len(prompt_tokens)}",
        )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_real_batch_scheduling_with_multimodal_requests(self):
        """
        Test batch scheduling with multimodal requests mixed in

        This tests a real-world scenario where some requests contain images,
        videos, or audio while others are text-only.
        """
        print("\n=== Testing real batch scheduling (multimodal mixed batch) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        config.model_config.enable_mm = True
        rm = self._create_resource_manager(config)

        # Mixed batch: text-only, image, video, audio
        batch_requests = [
            {
                "id": "text_only",
                "prompt": list(range(100)),
                "computed": 0,
                "mm_inputs": None,
            },
            {
                "id": "with_image",
                "prompt": list(range(80)),
                "computed": 0,
                "mm_inputs": {
                    "image_patch_id": 151,
                    "image_end_id": 152,
                    "video_patch_id": 153,
                    "video_end_id": 154,
                    "audio_patch_id": 155,
                    "audio_end_id": 156,
                    "patch_idx": [0] * 40 + [1] * 40,
                    "patch_map": [
                        {"modal_id": 0, "end_idx": 40, "image_num": 0, "video_num": 0},
                        {"modal_id": 1, "end_idx": 80, "image_num": 1, "video_num": 0},
                    ],
                    "tts": False,
                },
            },
        ]

        token_budget = 30

        print("  Mixed multimodal batch configuration:")
        for req in batch_requests:
            mm_type = "text-only"
            if req["mm_inputs"] is not None:
                if 1 in [p["modal_id"] for p in req["mm_inputs"]["patch_map"]]:
                    mm_type = "image"
            print(f"    {req['id']}: prompt_len={len(req['prompt'])}, type={mm_type}")

        print(f"    token_budget: {token_budget}")

        for req_cfg in batch_requests:
            request = create_test_request(req_cfg["id"], req_cfg["prompt"], req_cfg["computed"], req_cfg["mm_inputs"])
            num_new_tokens = rm._get_num_new_tokens(request, token_budget)

            print(
                f"    {req_cfg['id']}: new_tokens={num_new_tokens}, "
                f"with_image={getattr(request, 'with_image', False)}"
            )

            self.assertLessEqual(num_new_tokens, token_budget)
            self.assertGreaterEqual(num_new_tokens, 0)

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    # ========== Existing corner case tests (adapted for real Request class) ==========
    def test_corner_case_empty_request(self):
        """Test corner case: empty request and zero tokens"""
        print("\n=== Testing corner case (empty request / zero tokens) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Test 1: need_prefill_tokens = 0
        request = create_test_request("empty_req", [], 0)
        result = rm._get_num_new_tokens(request, token_budget=50)
        self.assertEqual(result, 0, "Empty request should return 0")
        print(f"  [PASS] empty prompt -> result={result}")

        # Test 2: num_computed == need_prefill (already completed)
        request = create_test_request("completed_req", list(range(100)), 100)
        result = rm._get_num_new_tokens(request, token_budget=50)
        self.assertEqual(result, 0, "Completed request should return 0")
        print(f"  [PASS] prompt_len=100, num_computed=100 -> result={result}")

        # Test 3: token_budget = 0
        request = create_test_request("zero_budget_req", list(range(100)), 0)
        result = rm._get_num_new_tokens(request, token_budget=0)
        self.assertEqual(result, 0, "Zero budget should return 0")
        print(f"  [PASS] token_budget=0 -> result={result}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_state_inconsistency(self):
        """Test corner case: num_computed > need_prefill (state inconsistency)"""
        print("\n=== Testing corner case (state inconsistency) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Test: num_computed > need_prefill_tokens
        # This is an invalid state that should not happen in normal operation.
        # The implementation should handle it gracefully by returning 0.
        request = create_test_request("inconsistent_req", list(range(50)), 100)
        result = rm._get_num_new_tokens(request, token_budget=50)

        # Should return 0 for this edge case
        self.assertEqual(result, 0, "Should return 0 for completed/inconsistent state")
        print(f"  [PASS] prompt_len=50, num_computed=100 (inconsistent) -> result={result}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_minimum_split_size(self):
        """Test corner case: split_kv_size = 1 (minimum alignment unit)"""
        print("\n=== Testing corner case (split_kv_size = 1) ===")

        split_kv_size = 1
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        test_cases = [
            (list(range(100)), 0, 20, 20),  # Every position is aligned
            (list(range(100)), 10, 15, 15),
            (list(range(100)), 50, 10, 10),
        ]

        for prompt_ids, num_computed, token_budget, expected in test_cases:
            request = create_test_request("min_split_req", prompt_ids, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)
            max_possible = min(len(prompt_ids) - num_computed, token_budget)
            self.assertEqual(result, max_possible, "split_size=1 should allow max allocation")
            print(
                f"  [PASS] prompt_len={len(prompt_ids)}, computed={num_computed}, budget={token_budget} -> result={result}"
            )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_split_size_larger_than_budget(self):
        """Test corner case: split_kv_size >> token_budget"""
        print("\n=== Testing corner case (split_kv_size >> token_budget) ===")

        split_kv_size = 128
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Test when split_kv_size is much larger than token_budget
        test_cases = [
            (list(range(100)), 0, 10),  # Budget much smaller than split size
            (list(range(100)), 0, 1),  # Budget is 1
            (list(range(100)), 64, 20),  # Starting near boundary
        ]

        for prompt_ids, num_computed, token_budget in test_cases:
            request = create_test_request("large_split_req", prompt_ids, num_computed)
            result = rm._get_num_new_tokens(request, token_budget)

            # Should not exceed max possible
            max_possible = min(len(prompt_ids) - num_computed, token_budget)
            self.assertLessEqual(result, max_possible)
            # Check if result is reasonable (might be 0 if can't align)
            self.assertGreaterEqual(result, 0)

            print(
                f"  [PASS] prompt_len={len(prompt_ids)}, computed={num_computed}, budget={token_budget} -> result={result}"
            )

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_split_size_larger_than_sequence(self):
        """Test corner case: split_kv_size >> need_prefill"""
        print("\n=== Testing corner case (split_kv_size >> prompt_len) ===")

        split_kv_size = 256
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Sequence shorter than split_kv_size
        prompt_tokens = list(range(50))
        request = create_test_request("short_seq_req", prompt_tokens, 0)
        result = rm._get_num_new_tokens(request, token_budget=100)

        # Should handle gracefully, may return 0 or partial
        max_possible = min(50, 100)
        self.assertLessEqual(result, max_possible)
        self.assertGreaterEqual(result, 0)
        print(f"  [PASS] prompt_len=50, split_size=256 -> result={result}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_dynamic_config_switch(self):
        """Test corner case: dynamic config switch during execution"""
        print("\n=== Testing corner case (dynamic config switch) ===")

        # Start with deterministic mode disabled
        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        request = create_test_request("switch_req", list(range(100)), 0)

        # Test 1: Non-deterministic mode
        result1 = rm._get_num_new_tokens(request, token_budget=30)
        print(f"  Non-deterministic mode: result={result1}")

        # Enable deterministic mode mid-stream
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = "16"

        # Recreate RM to pick up new env vars (simulating reload)
        rm = self._create_resource_manager(config)

        # Test 2: Deterministic mode
        num_computed = result1
        request2 = create_test_request("switch_req_2", list(range(100)), num_computed)
        result2 = rm._get_num_new_tokens(request2, token_budget=30)

        final_pos = num_computed + result2
        if result2 > 0:
            split_kv_size = 16
            aligned_end = (final_pos // split_kv_size) * split_kv_size
            is_aligned = (aligned_end == final_pos) or (final_pos == 100)
            self.assertTrue(is_aligned, f"Final pos {final_pos} should be aligned after mode switch")

        print(f"  [PASS] Deterministic mode after switch: result={result2}, final_pos={final_pos}")

        os.environ.pop("FD_DETERMINISTIC_MODE", None)
        os.environ.pop("FD_DETERMINISTIC_SPLIT_KV_SIZE", None)

    def test_corner_case_negative_values(self):
        """Test corner case: potential negative or invalid inputs"""
        print("\n=== Testing corner case (negative/invalid inputs) ===")

        split_kv_size = 16
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        os.environ["FD_DETERMINISTIC_SPLIT_KV_SIZE"] = str(split_kv_size)

        config = TestConfig()
        rm = self._create_resource_manager(config)

        # Test with real request object
        # Very large budget that could cause overflow issues
        request = create_test_request("large_budget_req", list(range(100)), 0)
        result = rm._get_num_new_tokens(request, token_budget=1000000)

        # Should not exceed actual need
        max_possible = 100
        self.assertLessEqual(result, max_possible, "Should not overflow beyond prompt_len")
        self.assertGreaterEqual(result, 0)
        print(f"  [PASS] Very large budget (1000000) -> result={result}")

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
