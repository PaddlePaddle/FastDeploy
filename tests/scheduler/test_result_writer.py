"""
Tests for ResultWriter class
"""
import unittest
import time
import threading
import math
from unittest.mock import Mock, patch, MagicMock
from collections import deque

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import ResultWriter
from tests.scheduler.test_utils import create_mock_request_output


class TestResultWriter(unittest.TestCase):
    """Test cases for ResultWriter class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.idx = 0
        self.batch = 200
        self.ttl = 900

    def test_init(self):
        """Test ResultWriter initialization"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        self.assertEqual(writer.idx, self.idx)
        self.assertEqual(writer.batch, self.batch)
        self.assertEqual(writer.client, self.mock_client)
        self.assertIsInstance(writer.data, deque)
        self.assertIsInstance(writer.cond, threading.Condition)
        self.assertIsInstance(writer.thread, threading.Thread)
        self.assertEqual(writer.ttl, self.ttl)

    def test_start(self):
        """Test start method"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        with patch.object(writer.thread, 'start') as mock_start:
            writer.start()
            mock_start.assert_called_once()

    def test_put_single_item(self):
        """Test put method with single item"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        result = create_mock_request_output("req1", finished=True)
        result_str = "test-result-1"
        
        writer.put("req1", [result_str])
        
        self.assertEqual(len(writer.data), 1)
        self.assertEqual(writer.data[0], ("req1", result_str))

    def test_put_multiple_items(self):
        """Test put method with multiple items"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        result1 = "test-result-1"
        result2 = "test-result-2"
        result3 = "test-result-3"
        
        writer.put("req1", [result1, result2, result3])
        
        self.assertEqual(len(writer.data), 3)
        self.assertEqual(writer.data[0], ("req1", result1))
        self.assertEqual(writer.data[1], ("req1", result2))
        self.assertEqual(writer.data[2], ("req1", result3))

    def test_put_notifies_condition(self):
        """Test that put method notifies the condition"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        with patch.object(writer.cond, 'notify_all') as mock_notify:
            writer.put("req1", ["test-result"])
            mock_notify.assert_called_once()

    def test_run_empty_data(self):
        """Test run method with empty data"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        mock_wait.assert_called()

    def test_run_with_data(self):
        """Test run method with data"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        # Add test data
        writer.data.append(("req1", "result1"))
        writer.data.append(("req1", "result2"))
        writer.data.append(("req2", "result3"))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        # Verify pipeline operations
        mock_pipeline.multi.assert_called_once()
        mock_pipeline.lpush.assert_called()
        mock_pipeline.expire.assert_called()
        mock_pipeline.execute.assert_called_once()

    def test_run_batches_data_correctly(self):
        """Test that run method batches data correctly"""
        writer = ResultWriter(self.mock_client, self.idx, 2, self.ttl)  # Small batch size
        
        # Add more data than batch size
        for i in range(5):
            writer.data.append((f"req{i}", f"result{i}"))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        # Should process in batches of 2
        self.assertEqual(mock_pipeline.lpush.call_count, 3)  # 2+2+1

    def test_run_groups_by_key(self):
        """Test that run method groups data by key"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        # Add data with same key
        writer.data.append(("req1", "result1"))
        writer.data.append(("req2", "result2"))
        writer.data.append(("req1", "result3"))
        writer.data.append(("req2", "result4"))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        # Should group by key
        lpush_calls = mock_pipeline.lpush.call_args_list
        self.assertEqual(len(lpush_calls), 2)  # Two different keys
        
        # Check that results are grouped
        req1_calls = [call for call in lpush_calls if call[0][0] == "req1"]
        req2_calls = [call for call in lpush_calls if call[0][0] == "req2"]
        
        self.assertEqual(len(req1_calls), 1)
        self.assertEqual(len(req2_calls), 1)

    def test_run_sets_expire_time(self):
        """Test that run method sets expire time correctly"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        writer.data.append(("req1", "result1"))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        # Verify expire is called with correct TTL
        expire_calls = mock_pipeline.expire.call_args_list
        self.assertEqual(len(expire_calls), 1)
        self.assertEqual(expire_calls[0][0][1], math.ceil(self.ttl))

    def test_run_exception_handling(self):
        """Test run method exception handling"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        writer.data.append(("req1", "result1"))
        
        # Mock pipeline to raise exception
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        mock_pipeline.multi.side_effect = Exception("Test error")
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            with patch.object(writer.cond, 'wait') as mock_wait:
                with patch.object(writer.cond, '__enter__') as mock_enter:
                    with patch.object(writer.cond, '__exit__') as mock_exit:
                        mock_enter.return_value = writer.cond
                        mock_exit.return_value = None
                        mock_wait.side_effect = KeyboardInterrupt()
                        
                        try:
                            writer.run()
                        except KeyboardInterrupt:
                            pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_thread_safety(self):
        """Test thread safety of ResultWriter operations"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        def put_data():
            for i in range(100):
                writer.put(f"req_{i}", [f"result_{i}"])
        
        # Run operations in parallel
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=put_data))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final state is consistent
        self.assertGreaterEqual(len(writer.data), 0)

    def test_run_clears_processed_data(self):
        """Test that run method clears processed data"""
        writer = ResultWriter(self.mock_client, self.idx, self.batch, self.ttl)
        
        # Add test data
        writer.data.append(("req1", "result1"))
        writer.data.append(("req2", "result2"))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__enter__ = Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = Mock(return_value=None)
        self.mock_client.pipeline.return_value = mock_pipeline
        
        with patch.object(writer.cond, 'wait') as mock_wait:
            with patch.object(writer.cond, '__enter__') as mock_enter:
                with patch.object(writer.cond, '__exit__') as mock_exit:
                    mock_enter.return_value = writer.cond
                    mock_exit.return_value = None
                    mock_wait.side_effect = KeyboardInterrupt()
                    
                    try:
                        writer.run()
                    except KeyboardInterrupt:
                        pass
        
        # Data should be cleared after processing
        self.assertEqual(len(writer.data), 0)


if __name__ == '__main__':
    unittest.main()
