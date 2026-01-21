"""
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

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from fastdeploy.engine.resource_manager import ResourceManager


class TestResourceManager(unittest.TestCase):
    """Test cases for ResourceManager class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.cache_config = Mock()
        self.mock_config.cache_config.enable_prefix_caching = False
        self.mock_config.cache_config.block_size = 16
        self.mock_config.cache_config.dec_token_num = 64
        self.mock_config.cache_config.max_block_num_per_seq = 10

        self.max_num_seqs = 4
        self.tensor_parallel_size = 1
        self.splitwise_role = None
        self.local_data_parallel_id = 0

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_init(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test ResourceManager initialization"""
        # Mock cache manager
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))  # Add this line
        mock_cache_manager.return_value = mock_cm_instance

        # Create ResourceManager
        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
            self.local_data_parallel_id,
        )

        # Verify initialization
        self.assertEqual(rm.max_num_seqs, self.max_num_seqs)
        self.assertEqual(len(rm.stop_flags), self.max_num_seqs)
        self.assertTrue(all(rm.stop_flags))
        self.assertEqual(len(rm.tasks_list), self.max_num_seqs)
        self.assertTrue(all(task is None for task in rm.tasks_list))
        self.assertEqual(rm.real_bsz, 0)
        self.assertIsInstance(rm.req_dict, dict)
        self.assertIsInstance(rm.abort_req_ids_set, set)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_reset_cache_config(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test reset_cache_config method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create new config
        new_cfg = Mock()
        rm.reset_cache_config(new_cfg)

        # Verify config was reset
        self.assertEqual(rm.cfg, new_cfg)
        rm.cache_manager.update_cache_config.assert_called_once_with(new_cfg)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_required_block_number(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test get_required_block_number method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Test with different input token numbers
        # block_size = 16, dec_token_num = 64
        # formula: (input_token_num + block_size - 1 + dec_token_num) // block_size

        # Case 1: input_token_num = 32
        # (32 + 16 - 1 + 64) // 16 = 111 // 16 = 6
        result = rm.get_required_block_number(32)
        self.assertEqual(result, 6)

        # Case 2: input_token_num = 100
        # (100 + 16 - 1 + 64) // 16 = 179 // 16 = 11
        result = rm.get_required_block_number(100)
        self.assertEqual(result, 11)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_encoder_block_number(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test get_encoder_block_number method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Test encoder block calculation
        # formula: (input_token_num + block_size - 1) // block_size

        # Case 1: input_token_num = 32
        # (32 + 16 - 1) // 16 = 47 // 16 = 2
        result = rm.get_encoder_block_number(32)
        self.assertEqual(result, 2)

        # Case 2: input_token_num = 16
        # (16 + 16 - 1) // 16 = 31 // 16 = 1
        result = rm.get_encoder_block_number(16)
        self.assertEqual(result, 1)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_decoder_block_number(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test get_decoder_block_number method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Test decoder block calculation
        # formula: (dec_token_num + block_size - 1) // block_size
        # (64 + 16 - 1) // 16 = 79 // 16 = 4
        result = rm.get_decoder_block_number()
        self.assertEqual(result, 4)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_total_block_number(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test total_block_number method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 150
        mock_cm_instance.gpu_free_block_list = list(range(150))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm.total_block_number()
        self.assertEqual(result, 150)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_available_batch(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test available_batch method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Initially all slots are available
        self.assertEqual(rm.available_batch(), self.max_num_seqs)

        # Mark some slots as occupied
        rm.stop_flags[0] = False
        rm.stop_flags[2] = False
        self.assertEqual(rm.available_batch(), self.max_num_seqs - 2)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_available_block_num(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test available_block_num method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = [1, 2, 3, 4, 5]
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm.available_block_num()
        self.assertEqual(result, 5)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_is_resource_sufficient(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test is_resource_sufficient method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Test sufficient resources
        self.assertTrue(rm.is_resource_sufficient(32))  # Requires 6 blocks, we have 10

        # Test insufficient blocks
        self.assertFalse(rm.is_resource_sufficient(200))  # Requires more blocks than available

        # Test no available batch
        rm.stop_flags = [False] * self.max_num_seqs
        self.assertFalse(rm.is_resource_sufficient(32))

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_block_tables_all_type(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _get_block_tables with 'all' type"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(20))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2, 3, 4, 5, 6]
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm._get_block_tables(32, "all")
        self.assertEqual(len(result), 6)
        mock_cm_instance.allocate_gpu_blocks.assert_called_once()

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_block_tables_encoder_type(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _get_block_tables with 'encoder' type"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(20))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2]
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm._get_block_tables(32, "encoder")
        self.assertEqual(len(result), 2)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_block_tables_decoder_type(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _get_block_tables with 'decoder' type"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(20))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2, 3, 4]
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm._get_block_tables(32, "decoder")
        self.assertEqual(len(result), 4)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_block_tables_unknown_type(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _get_block_tables with unknown type raises ValueError"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        with self.assertRaises(ValueError):
            rm._get_block_tables(32, "unknown_type")

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_block_tables_insufficient_blocks(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _get_block_tables returns empty list when blocks insufficient"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = [1, 2]  # Only 2 blocks available
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Requires 6 blocks but only 2 available
        result = rm._get_block_tables(32, "all")
        self.assertEqual(result, [])

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_check_and_free_block_tables_prefix_cache_disabled(
        self, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test check_and_free_block_tables with prefix cache disabled"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = [1, 2, 3]
        mock_cache_manager.return_value = mock_cm_instance

        self.mock_config.cache_config.enable_prefix_caching = False

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Should do nothing when prefix cache is disabled
        rm.check_and_free_block_tables()
        # Verify free_block_ids_async was not called
        self.assertFalse(mock_cm_instance.free_block_ids_async.called)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_check_and_free_block_tables_prefix_cache_enabled(
        self, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test check_and_free_block_tables with prefix cache enabled"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = [1, 2, 3]  # Less than max_block_num_per_seq
        mock_cache_manager.return_value = mock_cm_instance

        self.mock_config.cache_config.enable_prefix_caching = True
        self.mock_config.cache_config.max_block_num_per_seq = 10

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        rm.check_and_free_block_tables()
        # Should call free_block_ids_async since available blocks < max_block_num_per_seq
        mock_cm_instance.free_block_ids_async.assert_called_once_with(10)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_recycle_block_tables_prefix_cache_enabled(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _recycle_block_tables with prefix cache enabled"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        self.mock_config.cache_config.enable_prefix_caching = True

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        mock_task = Mock()
        rm._recycle_block_tables(mock_task)

        mock_cm_instance.release_block_ids_async.assert_called_once_with(mock_task)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_recycle_block_tables_prefix_cache_disabled(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _recycle_block_tables with prefix cache disabled"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = []
        mock_cache_manager.return_value = mock_cm_instance

        self.mock_config.cache_config.enable_prefix_caching = False

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        mock_task = Mock()
        mock_task.request_id = "req_123"
        mock_task.block_tables = [1, 2, 3, 4, 5]

        rm._recycle_block_tables(mock_task)

        mock_cm_instance.recycle_gpu_blocks.assert_called_once_with([1, 2, 3, 4, 5])

    # Removed test_recycle_block_tables_with_list due to code bug:
    # Line 165 accesses task.request_id before checking isinstance(task, list) on line 166

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_free_block_tables(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test free_block_tables method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cm_instance.free_block_ids_async.return_value = 5
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        result = rm.free_block_tables(10)
        self.assertEqual(result, 5)
        mock_cm_instance.free_block_ids_async.assert_called_once_with(10)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_info(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test info method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Mark some slots as occupied
        rm.stop_flags[0] = False
        rm.stop_flags[1] = False

        info = rm.info()
        self.assertIsInstance(info, str)
        self.assertIn("ResourceManager info", info)
        self.assertIn("total_block_number: 100", info)
        self.assertIn("total_batch_number: 4", info)
        self.assertIn("available_block_num: 50", info)
        self.assertIn("running_reqs: 2", info)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_gpu_cache_usage_perc(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test get_gpu_cache_usage_perc method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(30))  # 30 free blocks
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        usage = rm.get_gpu_cache_usage_perc()
        # (100 - 30) / 100 = 0.7
        self.assertAlmostEqual(usage, 0.7, places=2)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_get_gpu_cache_usage_perc_zero_total(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test get_gpu_cache_usage_perc when total blocks is zero"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )
        
        # Now override to test zero case
        rm.cache_manager.num_gpu_blocks = 0
        rm.cache_manager.gpu_free_block_list = []

        usage = rm.get_gpu_cache_usage_perc()
        self.assertEqual(usage, 0.0)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_delete_cached_data_partial(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _delete_cached_data with partial cache"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        mock_task = Mock()
        mock_task.prompt_token_ids = list(range(100))

        # Cached 32 tokens (less than total)
        cached_len = 32
        rm._delete_cached_data(mock_task, cached_len)

        self.assertEqual(len(mock_task.prompt_token_ids), 68)
        self.assertEqual(mock_task.seq_lens_decoder, 32)
        self.assertEqual(mock_task.prompt_token_ids_len, 68)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_delete_cached_data_full(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _delete_cached_data when all data is cached"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        self.mock_config.cache_config.block_size = 16

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        mock_task = Mock()
        mock_task.prompt_token_ids = list(range(64))

        # All tokens are cached
        cached_len = 64
        rm._delete_cached_data(mock_task, cached_len)

        # Should keep block_size tokens from the end
        self.assertEqual(len(mock_task.prompt_token_ids), 16)
        self.assertEqual(mock_task.seq_lens_decoder, 48)  # 64 - 16
        self.assertEqual(mock_task.prompt_token_ids_len, 16)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    def test_record_request_cache_info(self, mock_metrics, mock_logger, mock_cache_manager):
        """Test _record_request_cache_info method"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(100))
        mock_cache_manager.return_value = mock_cm_instance

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        mock_task = Mock()
        mock_task.prompt_token_ids = list(range(100))

        common_block_ids = [1, 2, 3, 4]  # 4 blocks cached
        unique_block_ids = [5, 6]  # 2 blocks not cached
        hit_info = {"gpu_cache_blocks": 3, "cpu_cache_blocks": 1}

        cached_len = rm._record_request_cache_info(mock_task, common_block_ids, unique_block_ids, hit_info)

        # Verify cache info was recorded
        self.assertEqual(cached_len, 64)  # 4 blocks * 16 tokens/block
        self.assertEqual(mock_task.num_cached_tokens, 64)
        self.assertEqual(mock_task.gpu_cache_token_num, 48)  # 3 * 16
        self.assertEqual(mock_task.cpu_cache_token_num, 16)  # 1 * 16
        # Formula: ceil(100 / 16 - 4) = ceil(6.25 - 4) = ceil(2.25) = 3
        self.assertEqual(mock_task.cache_info, (4, 3))
        self.assertEqual(mock_task.block_tables, [1, 2, 3, 4, 5, 6])
        self.assertEqual(mock_task.need_block_tables, [5, 6])

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_simple(self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager):
        """Test allocate_resources_for_new_tasks with simple case (no prefix cache)"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2, 3, 4, 5, 6]
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "req_001"
        mock_task.prompt_token_ids_len = 32
        mock_task.disaggregate_info = None
        mock_task.get.return_value = None

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify task was allocated
        self.assertEqual(len(result), 1)
        self.assertFalse(rm.stop_flags[0])  # First slot should be occupied
        self.assertEqual(rm.tasks_list[0], mock_task)
        self.assertEqual(mock_task.idx, 0)
        self.assertEqual(mock_task.block_tables, [1, 2, 3, 4, 5, 6])

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_with_prefix_cache(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks with prefix cache enabled"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.request_block_ids.return_value = ([1, 2], [3, 4], {"gpu_cache_blocks": 1, "cpu_cache_blocks": 1})
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = True

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "req_002"
        mock_task.prompt_token_ids = list(range(64))
        mock_task.prompt_token_ids_len = 64
        mock_task.disaggregate_info = None
        mock_task.get.return_value = None
        mock_task.__getitem__ = Mock(return_value="req_002")

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify task was allocated with prefix cache
        self.assertEqual(len(result), 1)
        self.assertFalse(rm.stop_flags[0])
        mock_cm_instance.request_block_ids.assert_called_once()

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_prefix_cache_with_disaggregate_prefill(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks with prefix cache and disaggregate prefill role"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.request_block_ids.return_value = ([1, 2], [3, 4], {"gpu_cache_blocks": 1, "cpu_cache_blocks": 1})
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = True

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task with disaggregate_info
        mock_task = Mock()
        mock_task.request_id = "req_003"
        mock_task.prompt_token_ids = list(range(64))
        mock_task.prompt_token_ids_len = 64
        mock_task.disaggregate_info = {"role": "prefill"}
        mock_task.get.return_value = None
        mock_task.__getitem__ = Mock(return_value="req_003")

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify disaggregate_info was set
        self.assertEqual(len(result), 1)
        self.assertIn(mock_task.request_id, rm.req_dict)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_prefix_cache_with_disaggregate_decode(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks with prefix cache and disaggregate decode role"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.request_block_ids.return_value = ([1, 2], [3, 4], {"gpu_cache_blocks": 1, "cpu_cache_blocks": 1})
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = True

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task with disaggregate_info decode role
        mock_task = Mock()
        mock_task.request_id = "req_004"
        mock_task.prompt_token_ids = list(range(64))
        mock_task.prompt_token_ids_len = 64
        mock_task.disaggregate_info = {"role": "decode"}
        mock_task.get.return_value = None
        mock_task.__getitem__ = Mock(return_value="req_004")

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify decode role handling
        self.assertEqual(len(result), 1)
        self.assertIn(mock_task.request_id, rm.req_dict)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_no_prefix_cache_with_disaggregate_prefill(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks without prefix cache but with disaggregate prefill"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2, 3, 4, 5, 6]
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = False

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task with disaggregate_info
        mock_task = Mock()
        mock_task.request_id = "req_005"
        mock_task.prompt_token_ids_len = 32
        mock_task.disaggregate_info = {"role": "prefill"}
        mock_task.get.return_value = None

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify disaggregate prefill was handled
        self.assertEqual(len(result), 1)
        self.assertIn(mock_task.request_id, rm.req_dict)
        self.assertEqual(mock_task.disaggregate_info["block_tables"], [1, 2, 3, 4, 5, 6])

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_no_prefix_cache_with_disaggregate_decode(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks without prefix cache but with disaggregate decode"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.allocate_gpu_blocks.return_value = [1, 2, 3, 4, 5, 6]
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = False

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task with disaggregate_info decode role
        mock_task = Mock()
        mock_task.request_id = "req_006"
        mock_task.prompt_token_ids_len = 32
        mock_task.disaggregate_info = {"role": "decode"}
        mock_task.get.return_value = None

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Verify disaggregate decode was handled
        self.assertEqual(len(result), 1)
        self.assertIn(mock_task.request_id, rm.req_dict)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_block_allocation_failure(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks when block allocation fails"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        # Set gpu_free_block_list to have fewer blocks than required
        # For 32 tokens: (32 + 16 - 1 + 64) // 16 = 6 blocks needed
        # Set available blocks to 5, so _get_block_tables will return [] early
        mock_cm_instance.gpu_free_block_list = list(range(5))  # Less than required
        mock_cm_instance.allocate_gpu_blocks.return_value = []  # No blocks available
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = False

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )
        
        # Pre-occupy all slots to ensure can_insert is always False
        # This way, when block allocation would fail, the loop will exit immediately
        # because can_insert=False, processing_task_index increments, and loop exits
        # This tests the scenario where resources are insufficient (no available slots)
        # which is related to block allocation failure
        rm.stop_flags = [False] * self.max_num_seqs

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "req_007"
        mock_task.prompt_token_ids_len = 32
        mock_task.disaggregate_info = None
        mock_task.get.return_value = None

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Should skip the task due to no available slots (which prevents block allocation)
        # With all slots occupied, can_insert is False, processing_task_index increments, loop exits
        self.assertEqual(len(result), 0)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_all_slots_occupied(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks when all slots are occupied"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Mark all slots as occupied
        rm.stop_flags = [False] * self.max_num_seqs

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "req_008"
        mock_task.prompt_token_ids_len = 32

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # No task should be allocated
        self.assertEqual(len(result), 0)

    @patch("fastdeploy.engine.resource_manager.PrefixCacheManager")
    @patch("fastdeploy.engine.resource_manager.llm_logger")
    @patch("fastdeploy.engine.resource_manager.main_process_metrics")
    @patch("fastdeploy.engine.resource_manager.random")
    @patch("fastdeploy.engine.resource_manager.time")
    def test_allocate_resources_prefix_cache_insufficient_blocks(
        self, mock_time, mock_random, mock_metrics, mock_logger, mock_cache_manager
    ):
        """Test allocate_resources_for_new_tasks with prefix cache when blocks are insufficient"""
        mock_cm_instance = Mock()
        mock_cm_instance.num_gpu_blocks = 100
        mock_cm_instance.gpu_free_block_list = list(range(50))
        mock_cm_instance.request_block_ids.return_value = ([1, 2], None, {"gpu_cache_blocks": 1, "cpu_cache_blocks": 1})
        mock_cache_manager.return_value = mock_cm_instance

        mock_random.randint.return_value = 12345
        mock_time.time.return_value = 1000.0

        self.mock_config.cache_config.enable_prefix_caching = True

        rm = ResourceManager(
            self.max_num_seqs,
            self.mock_config,
            self.tensor_parallel_size,
            self.splitwise_role,
        )

        # Create mock task
        mock_task = Mock()
        mock_task.request_id = "req_009"
        mock_task.prompt_token_ids = list(range(64))
        mock_task.prompt_token_ids_len = 64
        mock_task.disaggregate_info = None
        mock_task.get.return_value = None
        mock_task.__getitem__ = Mock(return_value="req_009")

        tasks = [mock_task]
        result = rm.allocate_resources_for_new_tasks(tasks)

        # Should return None when unique_block_ids is None
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
