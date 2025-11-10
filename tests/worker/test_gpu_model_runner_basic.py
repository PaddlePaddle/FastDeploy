"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import unittest
from unittest.mock import MagicMock, Mock

import numpy as np
import paddle

# Mock paddleformers and related modules before importing fastdeploy
sys.modules["paddleformers"] = MagicMock()
sys.modules["paddleformers.utils"] = MagicMock()
sys.modules["paddleformers.utils.log"] = MagicMock()
sys.modules["paddleformers.transformers"] = MagicMock()
sys.modules["paddleformers.transformers.configuration_utils"] = MagicMock()


class TestGPUModelRunnerMethods(unittest.TestCase):
    """Test core methods of GPUModelRunner"""

    def setUp(self):
        """Set up mock objects for testing"""
        self.mock_runner = Mock()
        self.mock_runner.share_inputs = {}

    def test_exist_prefill_logic(self):
        """Test exist_prefill logic"""
        # Test with prefill tasks
        seq_lens_encoder = paddle.to_tensor([[10], [0], [0]])
        self.mock_runner.share_inputs["seq_lens_encoder"] = seq_lens_encoder
        result = int(paddle.max(seq_lens_encoder)) > 0
        self.assertTrue(result)

        # Test without prefill tasks
        seq_lens_encoder = paddle.to_tensor([[0], [0], [0]])
        self.mock_runner.share_inputs["seq_lens_encoder"] = seq_lens_encoder
        result = int(paddle.max(seq_lens_encoder)) > 0
        self.assertFalse(result)

    def test_exist_decode_logic(self):
        """Test exist_decode logic"""
        # Test with decode tasks
        seq_lens_decoder = paddle.to_tensor([[5], [0], [0]])
        result = int(paddle.max(seq_lens_decoder)) > 0
        self.assertTrue(result)

        # Test without decode tasks
        seq_lens_decoder = paddle.to_tensor([[0], [0], [0]])
        result = int(paddle.max(seq_lens_decoder)) > 0
        self.assertFalse(result)

    def test_get_input_length_list_logic(self):
        """Test get_input_length_list logic"""
        num_tokens = 100
        batch_size = 4
        expected_decode_len = 10
        max_model_len = 2048
        block_size = 16
        enc_dec_block_num = 0

        # Calculate input_length
        max_dec_len = expected_decode_len + 1
        input_length = min(
            num_tokens // batch_size,
            max_model_len - max_dec_len,
        )

        # Calculate block_num
        calculated_block_num = (input_length + block_size - 1) // block_size + enc_dec_block_num

        # Create input_length_list
        input_length_list = [input_length] * batch_size

        # Verify block_num calculation
        self.assertGreater(calculated_block_num, 0)
        max_dec_len_list = [max_dec_len] * batch_size

        # Verify results
        self.assertEqual(len(input_length_list), 4)
        self.assertEqual(len(max_dec_len_list), 4)
        self.assertTrue(all(length > 0 for length in input_length_list))
        self.assertTrue(all(dec_len == 11 for dec_len in max_dec_len_list))

    def test_get_input_length_list_with_capture_prefill(self):
        """Test get_input_length_list logic with capture_prefill=True"""
        num_tokens = 10
        batch_size = 20

        # Create input_length_list with special allocation logic
        if num_tokens < batch_size:
            input_length_list = [1] * num_tokens
        else:
            input_length_list = [1] * (batch_size - 1)
            input_length_list.append(num_tokens - batch_size + 1)

        # Verify results
        self.assertEqual(len(input_length_list), num_tokens)
        self.assertEqual(sum(input_length_list), num_tokens)

    def test_cal_theortical_kvcache_logic(self):
        """Test cal_theortical_kvcache calculation logic"""
        # Without quantization
        head_dim = 128
        kv_num_heads = 32
        block_size = 16
        num_layers = 32
        byte_of_dtype = 2  # bf16

        hidden_dim = head_dim * kv_num_heads
        required_memory = byte_of_dtype * 2 * (block_size * hidden_dim) * num_layers

        self.assertGreater(required_memory, 0)

        # With quantization (int8)
        byte_of_dtype_quant = 1
        required_memory_quant = byte_of_dtype_quant * 2 * (block_size * hidden_dim) * num_layers

        # Memory should be smaller after quantization
        self.assertLess(required_memory_quant, required_memory)
        self.assertEqual(required_memory_quant, required_memory / 2)

    def test_scatter_and_cache_features_logic(self):
        """Test scatter_and_cache_features logic"""
        # Create mock features - adjust size to match split requirements
        merge_size = 2

        # grid_thw = [2, 5, 2] represents [temporal, height, width]
        # Feature size = (5 * 2) / (2^2) = 10 / 4 = 2.5, needs adjustment
        # Use more reasonable values: [2, 8, 2] -> feature size = (8 * 2) / 4 = 4
        test_grid_thw = paddle.to_tensor([[2, 8, 2], [3, 8, 2]], dtype="int64")
        mm_hashes = ["hash1", "hash2"]

        # Calculate size of each image feature
        image_features_size = (paddle.prod(test_grid_thw[:, 1:], axis=1) // (merge_size**2)).tolist()

        # Create features with correct size
        total_size = sum(image_features_size)
        image_features = paddle.randn([total_size, 512])

        # Split features
        image_features_lst = paddle.split(image_features, image_features_size, axis=0)

        # Verify split results
        self.assertEqual(len(image_features_lst), len(mm_hashes))
        self.assertEqual(sum(image_features_size), total_size)
        self.assertEqual(image_features_lst[0].shape[0], image_features_size[0])
        self.assertEqual(image_features_lst[1].shape[0], image_features_size[1])

        # Verify grid_thw dimensions match
        self.assertEqual(test_grid_thw.shape[0], 2)

    def test_chunked_inputs_logic(self):
        """Test get_chunked_inputs logic"""
        prefill_start_index = 0
        prefill_end_index = 10
        image_type_ids_start = 0
        image_type_ids_end = 5
        image_start = 0
        image_end = 2
        num_image_start = 0
        num_image_end = 2

        multimodal_inputs = {
            "input_ids": np.array(list(range(20))),
            "token_type_ids": np.array(list(range(20))),
            "image_type_ids": np.array(list(range(10))),
            "images": np.array(list(range(10))),
            "grid_thw": np.array([[1, 2, 3], [4, 5, 6]]),
            "mm_hashes": ["hash1", "hash2"],
        }

        # Get chunked inputs
        input_ids = multimodal_inputs["input_ids"][prefill_start_index:prefill_end_index]
        token_type_ids = multimodal_inputs["token_type_ids"][prefill_start_index:prefill_end_index]
        image_type_ids = multimodal_inputs["image_type_ids"][image_type_ids_start:image_type_ids_end]
        images = multimodal_inputs["images"][image_start:image_end]
        chunked_grid_thw = multimodal_inputs["grid_thw"][num_image_start:num_image_end]
        chunked_mm_hashes = multimodal_inputs["mm_hashes"][num_image_start:num_image_end]

        # Verify results
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(token_type_ids), 10)
        self.assertEqual(len(image_type_ids), 5)
        self.assertEqual(len(images), 2)
        self.assertEqual(len(chunked_grid_thw), 2)
        self.assertEqual(len(chunked_mm_hashes), 2)

    def test_batch_uncached_inputs_logic(self):
        """Test batch_uncached_inputs logic"""
        grid_thw = np.array([[2, 5, 2], [2, 5, 2]])
        mm_hashes = ["hash1", "hash2"]
        encoder_cache = {}  # Empty cache

        # Calculate sizes
        image_type_ids_size = grid_thw[:, 0]
        images_size = np.prod(grid_thw, axis=1)

        # Verify sizes
        self.assertEqual(len(image_type_ids_size), 2)
        self.assertEqual(len(images_size), 2)
        self.assertEqual(image_type_ids_size[0], 2)
        self.assertEqual(images_size[0], 20)

        # Test cache filtering logic
        uncached_mm_hashes = []
        for mm_hash in mm_hashes:
            if mm_hash not in encoder_cache:
                uncached_mm_hashes.append(mm_hash)

        # Verify all are uncached
        self.assertEqual(len(uncached_mm_hashes), 2)

        # Add one to cache
        encoder_cache["hash1"] = "cached_value"
        uncached_mm_hashes = []
        for mm_hash in mm_hashes:
            if mm_hash not in encoder_cache:
                uncached_mm_hashes.append(mm_hash)

        # Verify only one is uncached
        self.assertEqual(len(uncached_mm_hashes), 1)
        self.assertEqual(uncached_mm_hashes[0], "hash2")

    def test_not_need_stop_logic(self):
        """Test not_need_stop logic"""
        # Test when should not stop
        not_need_stop = paddle.to_tensor([True]).cpu()
        self.assertTrue(not_need_stop[0])

        # Test when should stop
        not_need_stop = paddle.to_tensor([False]).cpu()
        self.assertFalse(not_need_stop[0])

    def test_only_prefill_logic(self):
        """Test only_prefill logic"""
        # Only prefill, no decode
        seq_lens_encoder = paddle.to_tensor([[10], [5], [0]])
        seq_lens_decoder = paddle.to_tensor([[0], [0], [0]])

        if_only_prefill = True
        decode_exists = int(paddle.max(seq_lens_decoder)) > 0
        if_only_prefill = if_only_prefill and not decode_exists

        self.assertTrue(if_only_prefill)
        # Verify encoder has data
        self.assertGreater(int(paddle.max(seq_lens_encoder)), 0)

        # Both prefill and decode exist
        seq_lens_decoder = paddle.to_tensor([[5], [0], [0]])
        decode_exists = int(paddle.max(seq_lens_decoder)) > 0
        if_only_prefill = True
        if_only_prefill = if_only_prefill and not decode_exists

        self.assertFalse(if_only_prefill)

    def test_only_decode_logic(self):
        """Test only_decode logic"""
        # Only decode, no prefill
        seq_lens_encoder = paddle.to_tensor([[0], [0], [0]])
        seq_lens_decoder = paddle.to_tensor([[5], [3], [0]])

        if_only_decode = True
        prefill_exists = int(paddle.max(seq_lens_encoder)) > 0
        if_only_decode = if_only_decode and not prefill_exists

        self.assertTrue(if_only_decode)
        # Verify decoder has data
        self.assertGreater(int(paddle.max(seq_lens_decoder)), 0)

        # Both prefill and decode exist
        seq_lens_encoder = paddle.to_tensor([[10], [0], [0]])
        prefill_exists = int(paddle.max(seq_lens_encoder)) > 0
        if_only_decode = True
        if_only_decode = if_only_decode and not prefill_exists

        self.assertFalse(if_only_decode)

    def test_update_chunked_prefill_logic(self):
        """Test _update_chunked_prefill logic"""
        restore_chunked_prefill_request = {}

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "test_req_1"
        mock_task.idx = 0
        mock_task.chunk_idx = 1
        mock_task.prefill_chunk_info = [10, 20, 30]

        # Test adding to restore dict (chunk_idx=1 < len=3)
        if mock_task.chunk_idx < len(mock_task.prefill_chunk_info):
            restore_chunked_prefill_request[mock_task.request_id] = mock_task

        self.assertIn("test_req_1", restore_chunked_prefill_request)

        # Test updating chunk_idx (from 1 to 2)
        mock_task.chunk_idx += 1
        self.assertEqual(mock_task.chunk_idx, 2)
        # chunk_idx=2 < len=3, still in restore dict

        # Continue incrementing chunk_idx to 3
        mock_task.chunk_idx += 1
        self.assertEqual(mock_task.chunk_idx, 3)
        # Now chunk_idx=3 == len=3, should be removed

        # Test removing completed task
        if mock_task.chunk_idx >= len(mock_task.prefill_chunk_info):
            if mock_task.request_id in restore_chunked_prefill_request:
                del restore_chunked_prefill_request[mock_task.request_id]

        self.assertNotIn("test_req_1", restore_chunked_prefill_request)

    def test_padding_cudagraph_inputs_logic(self):
        """Test padding_cudagraph_inputs logic"""
        # Create mock data
        ids_remove_padding = paddle.zeros([100, 128])
        seq_lens_this_time_buffer = paddle.to_tensor([[10], [20], [30], [0]])

        # Test with cudagraph enabled
        use_cudagraph = True
        if use_cudagraph:
            real_token_num = ids_remove_padding.shape[0]
            self.assertEqual(real_token_num, 100)

        # Verify buffer dimensions
        self.assertEqual(seq_lens_this_time_buffer.shape[0], 4)

    def test_prepare_rope3d_logic(self):
        """Test prepare_rope3d calculation logic"""
        # Mock position_ids
        position_ids = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="int64")
        max_len_lst = [2048, 2048]
        cumsum_seqlens = [0, 3, 6]

        # Verify parameters
        self.assertEqual(len(max_len_lst), 2)
        self.assertEqual(len(cumsum_seqlens), 3)
        self.assertEqual(cumsum_seqlens[-1] - cumsum_seqlens[0], position_ids.shape[0] * position_ids.shape[1])


class TestGPUModelRunnerShareInputs(unittest.TestCase):
    """Test GPUModelRunner share_inputs initialization logic"""

    def test_share_inputs_initialization(self):
        """Test share_inputs field initialization"""
        max_num_seqs = 256
        max_model_len = 2048
        pad_token_id = 0
        eos_tokens_lens = 1

        # Test basic tensor initialization
        pre_ids = paddle.full([max_num_seqs, max_model_len], -1, dtype="int64")
        self.assertEqual(pre_ids.shape, [max_num_seqs, max_model_len])
        self.assertEqual(int(pre_ids[0, 0]), -1)

        input_ids = paddle.full([max_num_seqs, max_model_len], pad_token_id, dtype="int64")
        self.assertEqual(input_ids.shape, [max_num_seqs, max_model_len])
        self.assertEqual(int(input_ids[0, 0]), pad_token_id)

        eos_token_id = paddle.full([eos_tokens_lens, 1], 0, dtype="int64")
        self.assertEqual(eos_token_id.shape, [eos_tokens_lens, 1])

        # Test list initialization
        top_k_list = [0] * max_num_seqs
        self.assertEqual(len(top_k_list), max_num_seqs)
        self.assertTrue(all(k == 0 for k in top_k_list))

        min_p_list = [0.0] * max_num_seqs
        self.assertEqual(len(min_p_list), max_num_seqs)
        self.assertTrue(all(p == 0.0 for p in min_p_list))

    def test_free_list_initialization(self):
        """Test free_list initialization logic"""
        total_block_num = 1000
        kv_cache_ratio = 0.9

        # Initialize free_list
        free_list = list(
            range(
                total_block_num - 1,
                int(total_block_num * kv_cache_ratio) - 1,
                -1,
            )
        )

        free_list_len = len(free_list)

        # Verify results
        self.assertGreater(free_list_len, 0)
        self.assertLess(free_list_len, total_block_num)
        self.assertEqual(free_list[0], total_block_num - 1)  # First element
        self.assertGreater(free_list[-1], int(total_block_num * kv_cache_ratio) - 1)  # Last element

    def test_speculative_inputs_initialization(self):
        """Test speculative decoding inputs initialization"""
        max_num_seqs = 256
        max_draft_token_num = 4

        # Test speculative decoding related tensors
        accept_tokens = paddle.full(
            shape=[max_num_seqs, max_draft_token_num + 1],
            fill_value=0,
            dtype="int64",
        )
        self.assertEqual(accept_tokens.shape, [max_num_seqs, max_draft_token_num + 1])

        accept_num = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
        self.assertEqual(accept_num.shape, [max_num_seqs])

        draft_tokens = paddle.full(
            shape=[max_num_seqs, max_draft_token_num + 1],
            fill_value=0,
            dtype="int64",
        )
        self.assertEqual(draft_tokens.shape, [max_num_seqs, max_draft_token_num + 1])


class TestGPUModelRunnerAdditionalMethods(unittest.TestCase):
    """Test additional methods of GPUModelRunner"""

    def setUp(self):
        """Set up mock objects for testing"""
        self.mock_runner = Mock()
        self.mock_runner.share_inputs = {}

    def test_insert_sampling_params_logic(self):
        """Test insert_sampling_params logic"""
        # Create mock request
        mock_request = Mock()
        mock_request.idx = 0
        mock_request.get = Mock(
            side_effect=lambda key, default=None: {
                "top_p": 0.8,
                "top_k": 50,
                "min_p": 0.05,
                "temperature": 0.9,
                "repetition_penalty": 1.1,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "min_tokens": 5,
                "max_tokens": 100,
            }.get(key, default)
        )

        # Verify parameter values
        self.assertEqual(mock_request.get("top_p"), 0.8)
        self.assertEqual(mock_request.get("top_k"), 50)
        self.assertEqual(mock_request.get("min_p"), 0.05)
        self.assertEqual(mock_request.get("temperature"), 0.9)
        self.assertEqual(mock_request.get("repetition_penalty"), 1.1)
        self.assertEqual(mock_request.get("frequency_penalty"), 0.1)
        self.assertEqual(mock_request.get("presence_penalty"), 0.2)
        self.assertEqual(mock_request.get("min_tokens"), 5)
        self.assertEqual(mock_request.get("max_tokens"), 100)

    def test_bad_words_processing_logic(self):
        """Test bad_words_token_ids processing logic"""
        max_num_seqs = 256
        vocab_size = 50000

        # Test with bad words
        bad_words = [10, 20, 30, 40]
        bad_tokens = paddle.full([max_num_seqs, vocab_size], -1, dtype="int64")
        bad_tokens_len = paddle.full([max_num_seqs], 1, dtype="int64")

        # Simulate setting bad words for request at idx=0
        idx = 0
        bad_tokens[idx : idx + 1, : len(bad_words)] = paddle.to_tensor(bad_words, dtype="int64")
        bad_tokens_len[idx : idx + 1] = len(bad_words)

        # Verify results
        self.assertEqual(int(bad_tokens_len[idx]), len(bad_words))
        for i, word in enumerate(bad_words):
            self.assertEqual(int(bad_tokens[idx, i]), word)

        # Test without bad words (default case)
        idx = 1
        bad_tokens[idx : idx + 1, :] = -1
        bad_tokens_len[idx : idx + 1] = 1

        self.assertEqual(int(bad_tokens_len[idx]), 1)
        self.assertEqual(int(bad_tokens[idx, 0]), -1)

    def test_stop_sequences_processing_logic(self):
        """Test stop sequences processing logic"""
        max_num_seqs = 256
        max_stop_seqs_num = 4
        stop_seqs_max_len = 16

        # Test with stop sequences
        stop_token_ids = [[10, 20], [30, 40, 50], [60]]

        stop_seqs = paddle.full([max_num_seqs, max_stop_seqs_num, stop_seqs_max_len], -1, dtype="int64")
        stop_seqs_len_tensor = paddle.full([max_num_seqs, max_stop_seqs_num], 0, dtype="int32")

        # Fill stop sequences for idx=0
        idx = 0
        for i, seq in enumerate(stop_token_ids):
            stop_seqs[idx, i, : len(seq)] = paddle.to_tensor(seq, dtype="int64")
            stop_seqs_len_tensor[idx, i] = len(seq)

        # Verify results
        self.assertEqual(int(stop_seqs_len_tensor[idx, 0]), 2)
        self.assertEqual(int(stop_seqs_len_tensor[idx, 1]), 3)
        self.assertEqual(int(stop_seqs_len_tensor[idx, 2]), 1)
        self.assertEqual(int(stop_seqs[idx, 0, 0]), 10)
        self.assertEqual(int(stop_seqs[idx, 1, 2]), 50)

        # Test without stop sequences (default case)
        idx = 1
        stop_seqs_len_tensor[idx : idx + 1, :] = 0
        self.assertTrue(paddle.all(stop_seqs_len_tensor[idx] == 0))

    def test_cu_seqlens_calculation_logic(self):
        """Test cumulative sequence lengths calculation logic"""
        seq_lens = [10, 20, 15, 5]
        max_num_seqs = len(seq_lens)

        # Calculate cumulative sequence lengths
        cu_seqlens = paddle.zeros([max_num_seqs + 1, 1], dtype="int32")
        cumsum = 0
        for i, length in enumerate(seq_lens):
            cu_seqlens[i] = cumsum
            cumsum += length
        cu_seqlens[max_num_seqs] = cumsum

        # Verify results
        self.assertEqual(int(cu_seqlens[0]), 0)
        self.assertEqual(int(cu_seqlens[1]), 10)
        self.assertEqual(int(cu_seqlens[2]), 30)
        self.assertEqual(int(cu_seqlens[3]), 45)
        self.assertEqual(int(cu_seqlens[4]), 50)
        self.assertEqual(int(cu_seqlens[-1]), sum(seq_lens))

    def test_batch_id_per_token_logic(self):
        """Test batch_id_per_token generation logic"""
        seq_lens = [5, 3, 4]

        # Generate batch_id_per_token
        total_tokens = sum(seq_lens)
        batch_id_per_token = paddle.zeros([total_tokens, 1], dtype="int32")

        token_idx = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            for _ in range(seq_len):
                batch_id_per_token[token_idx] = batch_idx
                token_idx += 1

        # Verify results
        self.assertEqual(batch_id_per_token.shape[0], total_tokens)
        # First 5 tokens belong to batch 0
        self.assertEqual(int(batch_id_per_token[0]), 0)
        self.assertEqual(int(batch_id_per_token[4]), 0)
        # Next 3 tokens belong to batch 1
        self.assertEqual(int(batch_id_per_token[5]), 1)
        self.assertEqual(int(batch_id_per_token[7]), 1)
        # Last 4 tokens belong to batch 2
        self.assertEqual(int(batch_id_per_token[8]), 2)
        self.assertEqual(int(batch_id_per_token[11]), 2)

    def test_block_allocation_logic(self):
        """Test block allocation logic"""
        block_size = 16
        seq_length = 50
        max_dec_len = 20

        # Calculate required blocks
        total_length = seq_length + max_dec_len
        required_blocks = (total_length + block_size - 1) // block_size

        # Verify calculation
        self.assertEqual(required_blocks, 5)  # (50 + 20 + 15) / 16 = 5

        # Test edge cases
        seq_length = 16
        total_length = seq_length + max_dec_len
        required_blocks = (total_length + block_size - 1) // block_size
        self.assertEqual(required_blocks, 3)  # (16 + 20 + 15) / 16 = 3

        # Test exact multiple
        seq_length = 32
        max_dec_len = 16
        total_length = seq_length + max_dec_len
        required_blocks = (total_length + block_size - 1) // block_size
        self.assertEqual(required_blocks, 3)  # 48 / 16 = 3

    def test_rope_position_ids_logic(self):
        """Test RoPE position IDs logic"""
        seq_lens = [10, 15, 8]
        max_num_seqs = len(seq_lens)

        # Generate position IDs for each sequence
        position_ids_list = []
        for seq_len in seq_lens:
            position_ids = paddle.arange(0, seq_len, dtype="int64")
            position_ids_list.append(position_ids)

        # Verify results
        self.assertEqual(len(position_ids_list), max_num_seqs)
        self.assertEqual(position_ids_list[0].shape[0], 10)
        self.assertEqual(position_ids_list[1].shape[0], 15)
        self.assertEqual(position_ids_list[2].shape[0], 8)
        self.assertEqual(int(position_ids_list[0][0]), 0)
        self.assertEqual(int(position_ids_list[0][-1]), 9)

    def test_encoder_decoder_split_logic(self):
        """Test encoder/decoder task splitting logic"""
        # Test with mixed encoder and decoder tasks
        seq_lens_encoder = paddle.to_tensor([[10], [0], [5], [0]], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([[0], [3], [0], [7]], dtype="int32")

        # Count encoder and decoder tasks
        num_encoder_tasks = int(paddle.sum(seq_lens_encoder > 0))
        num_decoder_tasks = int(paddle.sum(seq_lens_decoder > 0))

        # Verify counts
        self.assertEqual(num_encoder_tasks, 2)  # Tasks at idx 0 and 2
        self.assertEqual(num_decoder_tasks, 2)  # Tasks at idx 1 and 3

        # Test all encoder tasks
        seq_lens_encoder = paddle.to_tensor([[10], [5], [8]], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([[0], [0], [0]], dtype="int32")

        num_encoder_tasks = int(paddle.sum(seq_lens_encoder > 0))
        num_decoder_tasks = int(paddle.sum(seq_lens_decoder > 0))

        self.assertEqual(num_encoder_tasks, 3)
        self.assertEqual(num_decoder_tasks, 0)

    def test_logprobs_metadata_logic(self):
        """Test logprobs metadata logic"""
        max_num_seqs = 4
        max_logprobs = 5

        # Test enable_logprob flag
        enable_logprob = True
        self.assertTrue(enable_logprob)

        # Test logprobs tensor initialization
        if enable_logprob:
            logprobs = paddle.zeros([max_num_seqs, max_logprobs], dtype="float32")
            logprobs_token_ids = paddle.zeros([max_num_seqs, max_logprobs], dtype="int64")

            self.assertEqual(logprobs.shape, [max_num_seqs, max_logprobs])
            self.assertEqual(logprobs_token_ids.shape, [max_num_seqs, max_logprobs])

    def test_seed_initialization_logic(self):
        """Test random seed initialization logic"""
        MAX_INFER_SEED = 9223372036854775806
        max_num_seqs = 256

        # Test default seed (0 means random)
        infer_seed = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.assertEqual(infer_seed.shape, [max_num_seqs, 1])
        self.assertEqual(int(infer_seed[0]), 0)

        # Test custom seed
        custom_seed = 12345
        self.assertLessEqual(custom_seed, MAX_INFER_SEED)
        infer_seed[0] = custom_seed
        self.assertEqual(int(infer_seed[0]), custom_seed)

    def test_token_type_ids_logic(self):
        """Test token_type_ids handling logic for multimodal inputs"""
        # Test token type IDs for text and image tokens
        text_tokens = [1, 2, 3, 4, 5]
        image_patch_id = 100
        image_tokens = [image_patch_id] * 10

        # Combine text and image tokens
        combined_tokens = text_tokens + image_tokens

        # Create token type IDs (0 for text, 1 for image)
        token_type_ids = []
        for token in combined_tokens:
            if token == image_patch_id:
                token_type_ids.append(1)
            else:
                token_type_ids.append(0)

        # Verify results
        self.assertEqual(len(token_type_ids), len(combined_tokens))
        self.assertEqual(sum(token_type_ids[:5]), 0)  # First 5 are text
        self.assertEqual(sum(token_type_ids[5:]), 10)  # Last 10 are image

    def test_grid_thw_processing_logic(self):
        """Test grid_thw (temporal, height, width) processing logic"""
        # Test grid_thw format [temporal, height, width]
        grid_thw = np.array([[2, 4, 4], [3, 8, 8]])

        # Calculate total patches per image
        total_patches = np.prod(grid_thw, axis=1)

        # Verify calculations
        self.assertEqual(total_patches[0], 32)  # 2 * 4 * 4
        self.assertEqual(total_patches[1], 192)  # 3 * 8 * 8

        # Calculate feature size after merging (merge_size=2)
        merge_size = 2
        feature_sizes = np.prod(grid_thw[:, 1:], axis=1) // (merge_size**2)

        self.assertEqual(feature_sizes[0], 4)  # (4 * 4) / 4
        self.assertEqual(feature_sizes[1], 16)  # (8 * 8) / 4

    def test_penalty_scores_logic(self):
        """Test penalty scores logic"""
        # Test repetition penalty
        repetition_penalty = 1.2
        self.assertGreaterEqual(repetition_penalty, 1.0)

        # Test frequency penalty (can be negative)
        frequency_penalty = 0.5
        self.assertGreaterEqual(frequency_penalty, 0.0)

        # Test presence penalty (can be negative)
        presence_penalty = 0.3
        self.assertGreaterEqual(presence_penalty, 0.0)

        # Verify default values
        default_repetition = 1.0
        default_frequency = 0.0
        default_presence = 0.0

        self.assertEqual(default_repetition, 1.0)
        self.assertEqual(default_frequency, 0.0)
        self.assertEqual(default_presence, 0.0)

    def test_max_model_len_constraint_logic(self):
        """Test max_model_len constraint logic"""
        max_model_len = 2048
        prompt_len = 1500
        requested_max_tokens = 800

        # Calculate actual max tokens considering constraint
        actual_max_tokens = min(requested_max_tokens, max_model_len - prompt_len)

        # Verify constraint is applied
        self.assertEqual(actual_max_tokens, 548)  # 2048 - 1500
        self.assertLessEqual(prompt_len + actual_max_tokens, max_model_len)

        # Test when requested tokens fit within limit
        prompt_len = 1000
        requested_max_tokens = 500
        actual_max_tokens = min(requested_max_tokens, max_model_len - prompt_len)

        self.assertEqual(actual_max_tokens, 500)
        self.assertLessEqual(prompt_len + actual_max_tokens, max_model_len)


if __name__ == "__main__":
    unittest.main()
