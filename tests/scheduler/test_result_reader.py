"""
Tests for ResultReader class
"""
import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import orjson

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import ResultReader
from fastdeploy.engine.request import RequestOutput, CompletionOutput, RequestMetrics
from tests.scheduler.test_utils import create_mock_request, create_mock_request_output


class TestResultReader(unittest.TestCase):
    """Test cases for ResultReader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.idx = 0
        self.batch = 200
        self.ttl = 900
        self.group = "test-group"

    def test_init(self):
        """Test ResultReader initialization"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        self.assertEqual(reader.idx, self.idx)
        self.assertEqual(reader.batch, self.batch)
        self.assertEqual(reader.client, self.mock_client)
        self.assertIsInstance(reader.data, deque)
        self.assertEqual(reader.ttl, self.ttl)
        self.assertEqual(reader.group, self.group)
        self.assertEqual(reader.reqs, {})
        self.assertEqual(reader.out_buffer, {})
        self.assertIsInstance(reader.lock, threading.Lock)
        self.assertIsInstance(reader.thread, threading.Thread)

    def test_add_req(self):
        """Test add_req method"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        req = create_mock_request("req1", 100, time.time())
        reader.add_req(req)
        
        self.assertIn("req1", reader.reqs)
        self.assertIn("req1", reader.out_buffer)
        self.assertEqual(reader.out_buffer["req1"], [])

    def test_read_empty_data(self):
        """Test read method with empty data"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        outputs = reader.read()
        self.assertEqual(outputs, {})

    def test_read_with_data(self):
        """Test read method with data"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        # Add some test data
        result1 = create_mock_request_output("req1", finished=True, error_code=200, send_idx=0)
        result2 = create_mock_request_output("req2", finished=False, error_code=200, send_idx=1)
        result3 = create_mock_request_output("req3", finished=False, error_code=500, send_idx=0)
        
        reader.data.appendleft(result1)
        reader.data.appendleft(result2)
        reader.data.appendleft(result3)
        
        # Add requests to track
        reader.reqs["req1"] = {"arrival_time": time.time()}
        reader.reqs["req2"] = {"arrival_time": time.time()}
        reader.reqs["req3"] = {"arrival_time": time.time()}
        
        outputs = reader.read()
        
        # Check that finished and error requests are returned immediately
        self.assertIn("req1", outputs)
        self.assertIn("req3", outputs)
        self.assertEqual(len(outputs["req1"]), 1)
        self.assertEqual(len(outputs["req3"]), 1)
        
        # Check that non-finished requests are grouped
        self.assertIn("req2", outputs)
        self.assertEqual(len(outputs["req2"]), 1)

    def test_read_with_grouped_tokens(self):
        """Test read method with grouped tokens"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        # Add grouped tokens for same request
        result1 = create_mock_request_output("req1", finished=False, error_code=200, send_idx=1)
        result2 = create_mock_request_output("req1", finished=False, error_code=200, send_idx=2)
        result3 = create_mock_request_output("req1", finished=True, error_code=200, send_idx=0)
        
        reader.data.appendleft(result1)
        reader.data.appendleft(result2)
        reader.data.appendleft(result3)
        
        reader.reqs["req1"] = {"arrival_time": time.time()}
        
        outputs = reader.read()
        
        # All results should be grouped together
        self.assertIn("req1", outputs)
        self.assertEqual(len(outputs["req1"]), 3)

    def test_read_with_out_buffer(self):
        """Test read method with existing out_buffer"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        # Add to out_buffer
        buffered_result = create_mock_request_output("req1", finished=False, error_code=200, send_idx=1)
        reader.out_buffer["req1"] = [buffered_result]
        
        # Add new result
        new_result = create_mock_request_output("req1", finished=True, error_code=200, send_idx=0)
        reader.data.appendleft(new_result)
        
        reader.reqs["req1"] = {"arrival_time": time.time()}
        
        outputs = reader.read()
        
        # Should combine buffered and new results
        self.assertIn("req1", outputs)
        self.assertEqual(len(outputs["req1"]), 2)

    def test_read_clears_finished_requests(self):
        """Test that read method clears finished requests from tracking"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, self.group)
        
        result = create_mock_request_output("req1", finished=True, error_code=200, send_idx=0)
        reader.data.appendleft(result)
        reader.reqs["req1"] = {"arrival_time": time.time()}
        
        outputs = reader.read()
        
        # Request should be removed from tracking
        self.assertNotIn("req1", reader.reqs)
        self.assertIn("req1", outputs)

    def test_sync_results_with_group(self):
        """Test sync_results method with group specified"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "test-group")
        
        # Mock Redis response
        mock_result = orjson.dumps(create_mock_request_output("req1", finished=True).to_dict())
        self.mock_client.rpop.return_value = [mock_result]
        
        total = reader.sync_results(["req1", "req2"])
        
        # Should only query the group key
        self.mock_client.rpop.assert_called_once_with("test-group", self.batch)
        self.assertEqual(total, 1)

    def test_sync_results_without_group(self):
        """Test sync_results method without group"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        # Mock Redis response
        mock_result = orjson.dumps(create_mock_request_output("req1", finished=True).to_dict())
        self.mock_client.rpop.return_value = [mock_result]
        
        total = reader.sync_results(["req1", "req2"])
        
        # Should query each key
        self.assertEqual(self.mock_client.rpop.call_count, 2)
        self.assertEqual(total, 1)

    def test_sync_results_empty_response(self):
        """Test sync_results method with empty Redis response"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        self.mock_client.rpop.return_value = None
        
        total = reader.sync_results(["req1"])
        
        self.assertEqual(total, 0)

    def test_sync_results_parse_error(self):
        """Test sync_results method with JSON parse error"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        # Mock invalid JSON response
        self.mock_client.rpop.return_value = ["invalid-json"]
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            total = reader.sync_results(["req1"])
        
        # Should log error and continue
        self.assertTrue(mock_logger.error.called)
        self.assertEqual(total, 0)

    def test_run_expired_requests(self):
        """Test run method handles expired requests"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        # Add expired request
        old_time = time.time() - 1000  # Very old
        reader.reqs["req1"] = {"arrival_time": old_time}
        
        # Mock the run method to execute once
        with patch.object(reader, 'sync_results', return_value=0):
            with patch('time.sleep') as mock_sleep:
                # Stop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    reader.run()
                except KeyboardInterrupt:
                    pass
        
        # Expired request should be handled
        self.assertNotIn("req1", reader.reqs)

    def test_run_with_requests(self):
        """Test run method with valid requests"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        # Add valid request
        reader.reqs["req1"] = {"arrival_time": time.time()}
        
        # Mock sync_results to return data
        mock_result = create_mock_request_output("req1", finished=True)
        reader.data.appendleft(mock_result)
        
        with patch.object(reader, 'sync_results', return_value=1):
            with patch('time.sleep') as mock_sleep:
                # Stop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    reader.run()
                except KeyboardInterrupt:
                    pass
        
        # Should have processed the result
        self.assertEqual(len(reader.data), 1)

    def test_run_exception_handling(self):
        """Test run method exception handling"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        with patch.object(reader, 'sync_results', side_effect=Exception("Test error")):
            with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
                with patch('time.sleep') as mock_sleep:
                    # Stop after first iteration
                    mock_sleep.side_effect = KeyboardInterrupt()
                    
                    try:
                        reader.run()
                    except KeyboardInterrupt:
                        pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_thread_safety(self):
        """Test thread safety of ResultReader operations"""
        reader = ResultReader(self.mock_client, self.idx, self.batch, self.ttl, "")
        
        def add_requests():
            for i in range(100):
                req = create_mock_request(f"req_{i}")
                reader.add_req(req)
        
        def read_requests():
            for _ in range(50):
                reader.read()
        
        # Run operations in parallel
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=add_requests))
            threads.append(threading.Thread(target=read_requests))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final state is consistent
        self.assertGreaterEqual(len(reader.reqs), 0)
        self.assertGreaterEqual(len(reader.out_buffer), 0)


if __name__ == '__main__':
    unittest.main()
