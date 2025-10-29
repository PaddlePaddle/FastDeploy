"""
Tests for SplitWiseSchedulerConfig class
"""
import unittest
import uuid
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import SplitWiseSchedulerConfig


class TestSplitWiseSchedulerConfig(unittest.TestCase):
    """Test cases for SplitWiseSchedulerConfig"""

    def test_init_with_default_values(self):
        """Test initialization with default values"""
        config = SplitWiseSchedulerConfig()
        
        # Check that nodeid is generated
        self.assertIsNotNone(config.nodeid)
        self.assertIsInstance(config.nodeid, str)
        
        # Check default values
        self.assertEqual(config.redis_host, "127.0.0.1")
        self.assertEqual(config.redis_port, 6379)
        self.assertIsNone(config.redis_password)
        self.assertEqual(config.redis_topic, "fd")
        self.assertEqual(config.ttl, 900)
        self.assertEqual(config.release_load_expire_period, 600)
        self.assertEqual(config.sync_period, 5)
        self.assertEqual(config.expire_period, 3.0)  # 3000ms / 1000.0
        self.assertEqual(config.clear_expired_nodes_period, 60)
        self.assertEqual(config.reader_parallel, 4)
        self.assertEqual(config.reader_batch_size, 200)
        self.assertEqual(config.writer_parallel, 4)
        self.assertEqual(config.writer_batch_size, 200)

    def test_init_with_custom_values(self):
        """Test initialization with custom values"""
        custom_nodeid = "test-node-123"
        config = SplitWiseSchedulerConfig(
            nodeid=custom_nodeid,
            host="192.168.1.100",
            port=6380,
            password="testpass",
            topic="test-topic",
            ttl=1800,
            release_load_expire_period=1200,
            sync_period=10,
            expire_period=6000,
            clear_expired_nodes_period=120,
            reader_parallel=8,
            reader_batch_size=400,
            writer_parallel=8,
            writer_batch_size=400
        )
        
        self.assertEqual(config.nodeid, custom_nodeid)
        self.assertEqual(config.redis_host, "192.168.1.100")
        self.assertEqual(config.redis_port, 6380)
        self.assertEqual(config.redis_password, "testpass")
        self.assertEqual(config.redis_topic, "test-topic")
        self.assertEqual(config.ttl, 1800)
        self.assertEqual(config.release_load_expire_period, 1200)
        self.assertEqual(config.sync_period, 10)
        self.assertEqual(config.expire_period, 6.0)  # 6000ms / 1000.0
        self.assertEqual(config.clear_expired_nodes_period, 120)
        self.assertEqual(config.reader_parallel, 8)
        self.assertEqual(config.reader_batch_size, 400)
        self.assertEqual(config.writer_parallel, 8)
        self.assertEqual(config.writer_batch_size, 400)

    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs"""
        config = SplitWiseSchedulerConfig(
            max_model_len=8192,
            enable_chunked_prefill=True,
            max_num_partial_prefills=8,
            max_long_partial_prefills=4,
            long_prefill_token_threshold=328
        )
        
        self.assertEqual(config.max_model_len, 8192)
        self.assertTrue(config.enable_chunked_prefill)
        self.assertEqual(config.max_num_partial_prefills, 8)
        self.assertEqual(config.max_long_partial_prefills, 4)
        self.assertEqual(config.long_prefill_token_threshold, 328)

    def test_long_prefill_token_threshold_calculation(self):
        """Test automatic calculation of long_prefill_token_threshold"""
        config = SplitWiseSchedulerConfig(
            max_model_len=4096,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2
        )
        
        expected_threshold = int(4096 * 0.04)
        self.assertEqual(config.long_prefill_token_threshold, expected_threshold)

    def test_long_prefill_token_threshold_zero_uses_calculation(self):
        """Test that zero long_prefill_token_threshold triggers calculation"""
        config = SplitWiseSchedulerConfig(
            max_model_len=2048,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=0
        )
        
        expected_threshold = int(2048 * 0.04)
        self.assertEqual(config.long_prefill_token_threshold, expected_threshold)

    def test_long_prefill_token_threshold_none_uses_calculation(self):
        """Test that None long_prefill_token_threshold triggers calculation"""
        config = SplitWiseSchedulerConfig(
            max_model_len=1024,
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=None
        )
        
        expected_threshold = int(1024 * 0.04)
        self.assertEqual(config.long_prefill_token_threshold, expected_threshold)

    def test_assertions_for_required_parameters(self):
        """Test that assertions are raised for missing required parameters"""
        # Test missing enable_chunked_prefill
        with self.assertRaises(AssertionError) as context:
            SplitWiseSchedulerConfig(max_model_len=4096)
        self.assertIn("enable_chunked_prefill must be set", str(context.exception))

        # Test missing max_num_partial_prefills
        with self.assertRaises(AssertionError) as context:
            SplitWiseSchedulerConfig(
                enable_chunked_prefill=True,
                max_model_len=4096
            )
        self.assertIn("max_num_partial_prefills must be set", str(context.exception))

        # Test missing max_long_partial_prefills
        with self.assertRaises(AssertionError) as context:
            SplitWiseSchedulerConfig(
                enable_chunked_prefill=True,
                max_num_partial_prefills=4,
                max_model_len=4096
            )
        self.assertIn("max_long_partial_prefills must be set", str(context.exception))

        # Test missing max_model_len when long_prefill_token_threshold is None
        with self.assertRaises(AssertionError) as context:
            SplitWiseSchedulerConfig(
                enable_chunked_prefill=True,
                max_num_partial_prefills=4,
                max_long_partial_prefills=2,
                long_prefill_token_threshold=None
            )
        self.assertIn("max_model_len must be set", str(context.exception))

    def test_check_method(self):
        """Test the check method"""
        config = SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            max_model_len=4096
        )
        
        # Should not raise any exception
        config.check()

    @patch('fastdeploy.scheduler.splitwise_scheduler.logger')
    def test_print_method(self, mock_logger):
        """Test the print method"""
        config = SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            max_model_len=4096
        )
        
        config.print()
        
        # Verify that logger.info was called
        self.assertTrue(mock_logger.info.called)
        
        # Check that configuration information is logged
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertIn("LocalScheduler Configuration Information :", info_calls)
        self.assertIn("=============================================================", info_calls)

    def test_uuid_generation_when_nodeid_is_none(self):
        """Test that UUID is generated when nodeid is None"""
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="test-uuid-123")
            
            config = SplitWiseSchedulerConfig(nodeid=None)
            self.assertEqual(config.nodeid, "test-uuid-123")

    def test_expire_period_conversion(self):
        """Test that expire_period is correctly converted from ms to seconds"""
        config = SplitWiseSchedulerConfig(expire_period=5000)
        self.assertEqual(config.expire_period, 5.0)

    def test_all_attributes_are_set(self):
        """Test that all expected attributes are set after initialization"""
        config = SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
            max_long_partial_prefills=2,
            max_model_len=4096
        )
        
        expected_attrs = [
            'nodeid', 'redis_host', 'redis_port', 'redis_password', 'redis_topic',
            'ttl', 'release_load_expire_period', 'sync_period', 'expire_period',
            'clear_expired_nodes_period', 'reader_parallel', 'reader_batch_size',
            'writer_parallel', 'writer_batch_size', 'max_model_len',
            'enable_chunked_prefill', 'max_num_partial_prefills',
            'max_long_partial_prefills', 'long_prefill_token_threshold'
        ]
        
        for attr in expected_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")


if __name__ == '__main__':
    unittest.main()
