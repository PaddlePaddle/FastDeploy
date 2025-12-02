import time
import unittest

import numpy as np
import paddle

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
)
from fastdeploy.worker.input_batch import InputBatch, reorder_split_prefill_and_decode


class SimpleModelConfig:
    """Simplified model configuration for testing"""

    def __init__(self):
        self.max_model_len = 32768
        self.vocab_size = 51200
        self.pad_token_id = -1
        self.eos_tokens_lens = 10
        self.top_p = 0.4
        self.temperature = 1
        self.penalty_score = 1
        self.frequency_score = 1
        self.presence_score = 1
        self.min_length = 100
        self.rope_theta = 1
        self.partial_rotary_factor = 1
        self.head_dim = 64
        self.max_stop_seqs_num = 5
        self.stop_seqs_max_len = 10
        self.enable_mm = False
        self.model_type = "test"
        self.architectures = ["TestModel"]  # Add architectures attribute
        self.rope_scaling = {  # Add rope_scaling attribute
            "original_max_position_embeddings": 2048,
            "factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
        }


def create_test_config(max_num_seqs=64):
    """Create simplified test configuration"""
    graph_opt_config = GraphOptimizationConfig(args={})
    scheduler_config = SchedulerConfig(args={})
    scheduler_config.max_num_seqs = max_num_seqs
    parallel_config = ParallelConfig(args={})
    parallel_config.enable_expert_parallel = False

    # Configure cache and speculative decoding
    cache_config = CacheConfig(args={})
    cache_config.block_size = 128
    cache_config.enc_dec_block_num = 8
    cache_config.total_block_num = 1000
    cache_config.kv_cache_ratio = 0.8

    # Configure SpeculativeConfig and StructuredOutputsConfig
    speculative_config = SpeculativeConfig(args={})
    speculative_config.method = None  # Disable speculative decoding for simplicity

    # Configure StructuredOutputsConfig
    structured_outputs_config_args = {
        "reasoning_parser": None,
        "guided_decoding_backend": "off",
        "disable_any_whitespace": True,
        "logits_processors": None,
    }

    model_config = SimpleModelConfig()
    fd_config = FDConfig(
        graph_opt_config=graph_opt_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        model_config=model_config,
        speculative_config=speculative_config,
        structured_outputs_config=StructuredOutputsConfig(args=structured_outputs_config_args),
        test_mode=True,
    )
    return fd_config


class TestInputBatchSwapPerformance(unittest.TestCase):

    def test_swap_states_correctness(self):
        """Test correctness of swap_states functionality"""
        fd_config = create_test_config(max_num_seqs=2)
        input_batch = InputBatch(fd_config)

        # Modify some fields for verification
        input_batch.input_ids[0] = paddle.full_like(input_batch.input_ids[0], 42)
        input_batch.top_p[0] = paddle.full_like(input_batch.top_p[0], 0.8)
        input_batch.stop_flags[0] = paddle.full_like(input_batch.stop_flags[0], True)

        input_batch.input_ids[1] = paddle.full_like(input_batch.input_ids[1], 123)
        input_batch.top_p[1] = paddle.full_like(input_batch.top_p[1], 0.2)
        input_batch.stop_flags[1] = paddle.full_like(input_batch.stop_flags[1], False)

        # Save original values
        original_values_0 = {
            "input_ids": input_batch.input_ids[0].clone(),
            "top_p": input_batch.top_p[0].clone(),
            "stop_flags": input_batch.stop_flags[0].clone(),
        }

        original_values_1 = {
            "input_ids": input_batch.input_ids[1].clone(),
            "top_p": input_batch.top_p[1].clone(),
            "stop_flags": input_batch.stop_flags[1].clone(),
        }

        # Perform swap
        input_batch.swap_states(0, 1)
        # Verify all attributes have been swapped
        self.assertTrue(paddle.equal_all(input_batch.input_ids[0], original_values_1["input_ids"]))
        self.assertTrue(paddle.equal_all(input_batch.top_p[0], original_values_1["top_p"]))
        self.assertTrue(paddle.equal_all(input_batch.stop_flags[0], original_values_1["stop_flags"]))

        self.assertTrue(paddle.equal_all(input_batch.input_ids[1], original_values_0["input_ids"]))
        self.assertTrue(paddle.equal_all(input_batch.top_p[1], original_values_0["top_p"]))
        self.assertTrue(paddle.equal_all(input_batch.stop_flags[1], original_values_0["stop_flags"]))

    def test_swap_states_performance(self):
        """Test performance of InputBatch.swap_states() method"""
        fd_config = create_test_config(max_num_seqs=64)
        input_batch = InputBatch(fd_config)

        # Warm up
        for _ in range(10):
            input_batch.swap_states(0, 1)

        # Performance test
        test_runs = 1000
        start_time = time.time()

        for i in range(test_runs):
            # Swap different index pairs
            idx1 = i % 64
            idx2 = (i + 1) % 64
            input_batch.swap_states_batch(idx1, idx2)

        elapsed = time.time() - start_time
        avg_time = elapsed / test_runs * 1000  # Convert to milliseconds
        print(f"\nPerformance test results (avg over {test_runs} runs):")
        print(f"- Max sequences: {fd_config.scheduler_config.max_num_seqs}")
        print(f"- Average swap time: {avg_time:.4f} ms")
        print(f"- Total test time: {elapsed:.4f} seconds")

        # Ensure performance is within reasonable range
        self.assertLess(avg_time, 10.0, "Swap operation should be efficient")


class TestReorderSplitPrefillAndDecode(unittest.TestCase):
    """Test cases for reorder_split_prefill_and_decode function"""

    def test_all_decode_requests(self):
        """Test when all requests are decode requests (all seq_lens_encoder = 0)"""
        fd_config = create_test_config(max_num_seqs=8)
        input_batch = InputBatch(fd_config)

        # Set all as decode requests
        paddle.assign(paddle.zeros([8, 1], dtype="int32"), input_batch.seq_lens_encoder)
        paddle.assign(paddle.ones([8, 1], dtype="int32") * 10, input_batch.seq_lens_decoder)

        # Reorder
        reorder_split_prefill_and_decode(input_batch)

        # All seq_lens_encoder should still be 0 (all decode)
        expected_encoder = np.zeros([8, 1], dtype="int32")
        np.testing.assert_array_equal(input_batch.seq_lens_encoder.numpy(), expected_encoder)

    def test_all_prefill_requests(self):
        """Test when all requests are prefill requests (all seq_lens_encoder > 0)"""
        fd_config = create_test_config(max_num_seqs=8)
        input_batch = InputBatch(fd_config)

        # Set all as prefill requests
        paddle.assign(paddle.ones([8, 1], dtype="int32") * 20, input_batch.seq_lens_encoder)
        paddle.assign(paddle.zeros([8, 1], dtype="int32"), input_batch.seq_lens_decoder)

        # Reorder
        reorder_split_prefill_and_decode(input_batch)

        # All seq_lens_decoder should still be 0 (all prefill)
        expected_decoder = np.zeros([8, 1], dtype="int32")
        np.testing.assert_array_equal(input_batch.seq_lens_decoder.numpy(), expected_decoder)

    def test_mixed_requests_reordering(self):
        """Test when there is a mix of decode and prefill requests"""
        fd_config = create_test_config(max_num_seqs=8)
        input_batch = InputBatch(fd_config)

        # Create mixed requests: [decode, prefill, decode, prefill, prefill, decode, decode, prefill]
        decoder_len = [10, 0, 5, 0, 0, 8, 12, 0]  # >0 for decode, 0 for prefill

        paddle.assign(
            paddle.to_tensor([0, 15, 0, 25, 30, 0, 0, 20], dtype="int32").reshape([8, 1]), input_batch.seq_lens_encoder
        )
        paddle.assign(paddle.to_tensor(decoder_len, dtype="int32").reshape([8, 1]), input_batch.seq_lens_decoder)

        # Reorder
        reorder_split_prefill_and_decode(input_batch)

        # After reordering, first part should be decode requests (seq_lens_encoder = 0)
        # Last part should be prefill requests (seq_lens_encoder > 0)
        encoder_values = input_batch.seq_lens_encoder.numpy().flatten()
        decoder_values = input_batch.seq_lens_decoder.numpy().flatten()

        # Count decode requests (where seq_lens_encoder == 0)
        decode_count = np.sum(encoder_values == 0)

        # Verify first decode_count entries are decode requests
        for i in range(decode_count):
            self.assertEqual(encoder_values[i], 0, f"Position {i} should be decode request")
            self.assertGreater(decoder_values[i], 0, f"Position {i} should have decoder length > 0")

        # Verify remaining entries are prefill requests
        for i in range(decode_count, len(encoder_values)):
            self.assertGreater(encoder_values[i], 0, f"Position {i} should be prefill request")
            self.assertEqual(decoder_values[i], 0, f"Position {i} should have decoder length = 0")

    def test_single_request(self):
        """Test with a single request (boundary case)"""
        fd_config = create_test_config(max_num_seqs=1)
        input_batch = InputBatch(fd_config)

        # Set as decode request
        paddle.assign(paddle.zeros([1, 1], dtype="int32"), input_batch.seq_lens_encoder)
        paddle.assign(paddle.ones([1, 1], dtype="int32") * 5, input_batch.seq_lens_decoder)

        # Reorder should not crash
        reorder_split_prefill_and_decode(input_batch)

        self.assertEqual(input_batch.seq_lens_encoder.numpy()[0, 0], 0)
        self.assertEqual(input_batch.seq_lens_decoder.numpy()[0, 0], 5)

    def test_alternating_requests(self):
        """Test with alternating decode and prefill requests"""
        fd_config = create_test_config(max_num_seqs=6)
        input_batch = InputBatch(fd_config)

        # Create alternating pattern: [decode, prefill, decode, prefill, decode, prefill]
        decoder_len = [5, 0, 8, 0, 3, 0]
        paddle.assign(
            paddle.to_tensor([0, 10, 0, 20, 0, 15], dtype="int32").reshape([6, 1]), input_batch.seq_lens_encoder
        )
        paddle.assign(paddle.to_tensor(decoder_len, dtype="int32").reshape([6, 1]), input_batch.seq_lens_decoder)

        # Reorder
        reorder_split_prefill_and_decode(input_batch)

        # Verify ordering: first 3 should be decode, last 3 should be prefill
        encoder_values = input_batch.seq_lens_encoder.numpy().flatten()
        decoder_values = input_batch.seq_lens_decoder.numpy().flatten()

        for i in range(3):
            self.assertEqual(encoder_values[i], 0, f"Position {i} should be decode")
            self.assertGreater(decoder_values[i], 0, f"Position {i} should have decoder length")

        for i in range(3, 6):
            self.assertGreater(encoder_values[i], 0, f"Position {i} should be prefill")
            self.assertEqual(decoder_values[i], 0, f"Position {i} should have no decoder length")

    def test_input_ids_reordering(self):
        """Test that input_ids are correctly reordered with decode/prefill requests"""
        fd_config = create_test_config(max_num_seqs=4)
        input_batch = InputBatch(fd_config)

        # Setup: [prefill, decode, prefill, decode] with unique markers
        paddle.assign(paddle.to_tensor([25, 0, 35, 0], dtype="int32").reshape([4, 1]), input_batch.seq_lens_encoder)
        paddle.assign(paddle.to_tensor([0, 8, 0, 12], dtype="int32").reshape([4, 1]), input_batch.seq_lens_decoder)

        # Set unique markers: 1000s for prefill, 2000s for decode
        for i in range(4):
            if i % 2 == 0:  # prefill positions (0, 2)
                marker = (i + 1) * 1000  # 1000, 3000
                input_batch.input_ids[i, 0] = marker
            else:  # decode positions (1, 3)
                marker = (i + 1) * 1000  # 2000, 4000
                input_batch.input_ids[i, 0] = marker

        # Store markers before reordering
        markers_before = input_batch.input_ids.numpy()[:, 0].copy()

        # Reorder
        reorder_split_prefill_and_decode(input_batch)

        # Verify ordering
        encoder_values = input_batch.seq_lens_encoder.numpy().flatten()
        markers_after = input_batch.input_ids.numpy()[:, 0].copy()

        # First two should be decode (encoder=0) with markers 2000, 4000
        self.assertEqual(encoder_values[0], 0)
        self.assertEqual(encoder_values[1], 0)
        self.assertIn(markers_after[0], [2000, 4000])  # decode marker
        self.assertIn(markers_after[1], [2000, 4000])  # decode marker

        # Last two should be prefill (encoder>0) with markers 1000, 3000
        self.assertGreater(encoder_values[2], 0)
        self.assertGreater(encoder_values[3], 0)
        self.assertIn(markers_after[2], [1000, 3000])  # prefill marker
        self.assertIn(markers_after[3], [1000, 3000])  # prefill marker

        # Verify all markers are present (no data loss)
        np.testing.assert_array_equal(np.sort(markers_before), np.sort(markers_after))


if __name__ == "__main__":
    unittest.main()
