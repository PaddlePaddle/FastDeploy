"""
Tests for InferScheduler class
"""
import unittest
import time
import threading
import hashlib
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import orjson

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import InferScheduler, NodeInfo
from tests.scheduler.test_utils import create_mock_config, create_mock_request, create_mock_request_output


class TestInferScheduler(unittest.TestCase):
    """Test cases for InferScheduler class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.scheduler = InferScheduler(self.config)

    def test_init(self):
        """Test InferScheduler initialization"""
        self.assertEqual(self.scheduler.config, self.config)
        self.assertEqual(self.scheduler.nodeid, self.config.nodeid)
        self.assertEqual(self.scheduler.writer_parallel, self.config.writer_parallel)
        self.assertEqual(self.scheduler.writer_batch_size, self.config.writer_batch_size)
        self.assertEqual(self.scheduler.sync_period, self.config.sync_period)
        self.assertEqual(self.scheduler.topic, self.config.redis_topic)
        self.assertEqual(self.scheduler.cluster_key, f"{self.config.redis_topic}.cluster")
        self.assertEqual(self.scheduler.ttl, self.config.ttl)
        self.assertEqual(self.scheduler.release_load_expire_period, self.config.release_load_expire_period)
        self.assertIsInstance(self.scheduler.reqs_queue, deque)
        self.assertEqual(self.scheduler.writers, [])

    def test_start(self):
        """Test start method creates writers and threads"""
        role = "prefill"
        host = "192.168.1.100"
        disaggregated = {"transfer_protocol": ["ipc"]}
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.ResultWriter') as mock_writer_class:
            with patch('fastdeploy.scheduler.splitwise_scheduler.threading.Thread') as mock_thread_class:
                with patch('fastdeploy.scheduler.splitwise_scheduler.NodeInfo') as mock_node_class:
                    mock_writer = Mock()
                    mock_writer_class.return_value = mock_writer
                    mock_thread = Mock()
                    mock_thread_class.return_value = mock_thread
                    mock_node = Mock()
                    mock_node_class.return_value = mock_node
                    
                    self.scheduler.start(role, host, disaggregated)
                    
                    # Should create writers
                    self.assertEqual(mock_writer_class.call_count, self.config.writer_parallel)
                    self.assertEqual(len(self.scheduler.writers), self.config.writer_parallel)
                    
                    # Should create threads
                    self.assertEqual(mock_thread_class.call_count, 3)  # getreq_thread, report_thread, expire_reqs_thread
                    
                    # Should set role, host, and node
                    self.assertEqual(self.scheduler.role, role)
                    self.assertEqual(self.scheduler.host, host)

    def test_routine_report(self):
        """Test routine_report method"""
        # Mock node
        mock_node = Mock()
        mock_node.serialize.return_value = '{"test": "data"}'
        self.scheduler.node = mock_node
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.routine_report()
            except KeyboardInterrupt:
                pass
        
        # Should serialize and report node info
        mock_node.serialize.assert_called()
        self.scheduler.client.hset.assert_called_with(
            self.scheduler.cluster_key,
            self.scheduler.nodeid,
            '{"test": "data"}'
        )

    def test_routine_report_exception_handling(self):
        """Test routine_report exception handling"""
        self.scheduler.client.hset.side_effect = Exception("Redis error")
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            with patch('time.sleep') as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    self.scheduler.routine_report()
                except KeyboardInterrupt:
                    pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_loop_expire_reqs(self):
        """Test loop_expire_reqs method"""
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.loop_expire_reqs()
            except KeyboardInterrupt:
                pass
        
        # Should call expire_reqs on node
        mock_node.expire_reqs.assert_called_with(self.scheduler.release_load_expire_period)

    def test_loop_expire_reqs_exception_handling(self):
        """Test loop_expire_reqs exception handling"""
        mock_node = Mock()
        mock_node.expire_reqs.side_effect = Exception("Expire error")
        self.scheduler.node = mock_node
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            with patch('time.sleep') as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    self.scheduler.loop_expire_reqs()
                except KeyboardInterrupt:
                    pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_loop_get_reqs_with_data(self):
        """Test loop_get_reqs method with data"""
        # Mock Redis response
        req_data = {
            "request_id": "req1",
            "prompt_token_ids_len": 100,
            "arrival_time": time.time(),
            "group": "test-group"
        }
        req_str = orjson.dumps(req_data)
        self.scheduler.client.rpop.return_value = [req_str]
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        self.scheduler.role = "prefill"
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.loop_get_reqs()
            except KeyboardInterrupt:
                pass
        
        # Should add request to queue
        self.assertEqual(len(self.scheduler.reqs_queue), 1)
        req = self.scheduler.reqs_queue[0]
        self.assertEqual(req.request_id, "req1#0#test-group")  # writer_idx=0, group=test-group

    def test_loop_get_reqs_brpop_fallback(self):
        """Test loop_get_reqs method with brpop fallback"""
        # Mock Redis response - rpop returns None, brpop returns data
        req_data = {
            "request_id": "req1",
            "prompt_token_ids_len": 100,
            "arrival_time": time.time(),
            "group": "test-group"
        }
        req_str = orjson.dumps(req_data)
        
        self.scheduler.client.rpop.return_value = None
        self.scheduler.client.brpop.return_value = (b"ReqQ_test", req_str)
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        self.scheduler.role = "prefill"
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.loop_get_reqs()
            except KeyboardInterrupt:
                pass
        
        # Should add request to queue
        self.assertEqual(len(self.scheduler.reqs_queue), 1)

    def test_loop_get_reqs_exception_handling(self):
        """Test loop_get_reqs exception handling"""
        self.scheduler.client.rpop.side_effect = Exception("Redis error")
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            with patch('time.sleep') as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    self.scheduler.loop_get_reqs()
                except KeyboardInterrupt:
                    pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_get_requests_empty_queue(self):
        """Test get_requests method with empty queue"""
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        self.assertEqual(result, [])

    def test_get_requests_insufficient_resources(self):
        """Test get_requests method with insufficient resources"""
        # Add request to queue
        req = create_mock_request("req1", 100)
        self.scheduler.reqs_queue.append(req)
        
        result = self.scheduler.get_requests(
            available_blocks=5,  # Very low
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        # Should return empty due to insufficient blocks
        self.assertEqual(result, [])

    def test_get_requests_batch_processing(self):
        """Test get_requests method with batch processing"""
        # Add multiple requests to queue
        req1 = create_mock_request("req1", 100)
        req2 = create_mock_request("req2", 200)
        req3 = create_mock_request("req3", 300)
        
        self.scheduler.reqs_queue.extend([req1, req2, req3])
        
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=2
        )
        
        # Should return up to batch size
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].request_id, "req1")
        self.assertEqual(result[1].request_id, "req2")

    def test_get_requests_chunked_prefill_long_requests(self):
        """Test get_requests method with chunked prefill and long requests"""
        # Add long request
        long_req = create_mock_request("long_req", 2000)  # Above threshold
        self.scheduler.reqs_queue.append(long_req)
        
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        # Should handle long requests
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].request_id, "long_req")

    def test_get_requests_chunked_prefill_short_requests(self):
        """Test get_requests method with chunked prefill and short requests"""
        # Add short request
        short_req = create_mock_request("short_req", 100)  # Below threshold
        self.scheduler.reqs_queue.append(short_req)
        
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        # Should handle short requests
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].request_id, "short_req")

    def test_get_requests_expired_request(self):
        """Test get_requests method with expired request"""
        # Add expired request
        old_time = time.time() - 1000
        expired_req = create_mock_request("expired_req", 100, old_time)
        self.scheduler.reqs_queue.append(expired_req)
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        # Should skip expired request
        self.assertEqual(result, [])
        mock_node.finish_req.assert_called_with("expired_req")

    def test_get_requests_exception_handling(self):
        """Test get_requests method exception handling"""
        # Add request that will cause exception
        req = create_mock_request("req1", 100)
        req.prompt_token_ids_len = None  # This will cause exception
        self.scheduler.reqs_queue.append(req)
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            result = self.scheduler.get_requests(
                available_blocks=100,
                block_size=16,
                reserved_output_blocks=10,
                max_num_batched_tokens=2048,
                batch=5
            )
        
        # Should return empty list and log error
        self.assertEqual(result, [])
        self.assertTrue(mock_logger.error.called)

    def test_put_results(self):
        """Test put_results method"""
        # Mock writers
        mock_writer1 = Mock()
        mock_writer2 = Mock()
        self.scheduler.writers = [mock_writer1, mock_writer2]
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        self.scheduler.role = "prefill"
        
        # Create test results
        result1 = create_mock_request_output("req1#0#group1", finished=True)
        result2 = create_mock_request_output("req2#1#group2", finished=False)
        results = [result1, result2]
        
        with patch.object(result1, 'to_dict') as mock_dict1:
            with patch.object(result2, 'to_dict') as mock_dict2:
                mock_dict1.return_value = {"request_id": "req1", "finished": True}
                mock_dict2.return_value = {"request_id": "req2", "finished": False}
                
                self.scheduler.put_results(results)
        
        # Should call put on appropriate writers
        mock_writer1.put.assert_called_once()
        mock_writer2.put.assert_called_once()
        
        # Should update request timestamps
        mock_node.update_req_timestamp.assert_called_once()

    def test_put_results_finished_request(self):
        """Test put_results method with finished request"""
        # Mock writers
        mock_writer = Mock()
        self.scheduler.writers = [mock_writer]
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        
        # Create finished result
        result = create_mock_request_output("req1#0#group1", finished=True)
        results = [result]
        
        with patch.object(result, 'to_dict') as mock_dict:
            mock_dict.return_value = {"request_id": "req1", "finished": True}
            
            self.scheduler.put_results(results)
        
        # Should finish request on node
        mock_node.finish_req.assert_called_with("req1#0#group1")

    def test_put_results_error_request(self):
        """Test put_results method with error request"""
        # Mock writers
        mock_writer = Mock()
        self.scheduler.writers = [mock_writer]
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        
        # Create error result
        result = create_mock_request_output("req1#0#group1", finished=False, error_code=500)
        results = [result]
        
        with patch.object(result, 'to_dict') as mock_dict:
            mock_dict.return_value = {"request_id": "req1", "error_code": 500}
            
            self.scheduler.put_results(results)
        
        # Should finish request on node due to error
        mock_node.finish_req.assert_called_with("req1#0#group1")

    def test_put_results_prefill_role_send_idx_zero(self):
        """Test put_results method with prefill role and send_idx=0"""
        # Mock writers
        mock_writer = Mock()
        self.scheduler.writers = [mock_writer]
        
        # Mock node
        mock_node = Mock()
        self.scheduler.node = mock_node
        self.scheduler.role = "prefill"
        
        # Create result with send_idx=0
        result = create_mock_request_output("req1#0#group1", finished=False, send_idx=0)
        results = [result]
        
        with patch.object(result, 'to_dict') as mock_dict:
            mock_dict.return_value = {"request_id": "req1", "outputs": {"send_idx": 0}}
            
            self.scheduler.put_results(results)
        
        # Should set finished=False for prefill with send_idx=0
        self.assertFalse(result.finished)

    def test_select_writer(self):
        """Test select_writer function"""
        # Mock writers
        self.scheduler.writers = [Mock(), Mock(), Mock()]
        
        req = create_mock_request("req1")
        
        # Test that same request always gets same writer
        writer_idx1 = self.scheduler._select_writer(req)
        writer_idx2 = self.scheduler._select_writer(req)
        
        self.assertEqual(writer_idx1, writer_idx2)
        self.assertIn(writer_idx1, range(len(self.scheduler.writers)))

    def test_select_writer_different_requests(self):
        """Test select_writer function with different requests"""
        # Mock writers
        self.scheduler.writers = [Mock(), Mock(), Mock()]
        
        req1 = create_mock_request("req1")
        req2 = create_mock_request("req2")
        
        # Different requests might get different writers
        writer_idx1 = self.scheduler._select_writer(req1)
        writer_idx2 = self.scheduler._select_writer(req2)
        
        # Both should be valid indices
        self.assertIn(writer_idx1, range(len(self.scheduler.writers)))
        self.assertIn(writer_idx2, range(len(self.scheduler.writers)))


if __name__ == '__main__':
    unittest.main()
