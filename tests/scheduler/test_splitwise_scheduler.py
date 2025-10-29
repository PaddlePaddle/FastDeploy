"""
Tests for SplitWiseScheduler main class
"""
import unittest
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import SplitWiseScheduler
from tests.scheduler.test_utils import create_mock_config, create_mock_request, create_mock_request_output


class TestSplitWiseScheduler(unittest.TestCase):
    """Test cases for SplitWiseScheduler main class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.scheduler = SplitWiseScheduler(self.config)

    def test_init(self):
        """Test SplitWiseScheduler initialization"""
        self.assertIsNotNone(self.scheduler.scheduler)
        self.assertIsNotNone(self.scheduler.infer)
        
        # Check that scheduler and infer are properly initialized
        self.assertEqual(self.scheduler.scheduler.nodeid, self.config.nodeid)
        self.assertEqual(self.scheduler.infer.nodeid, self.config.nodeid)

    def test_start(self):
        """Test start method"""
        role = "prefill"
        host = "192.168.1.100"
        disaggregated = {"transfer_protocol": ["ipc"]}
        
        with patch.object(self.scheduler.infer, 'start') as mock_infer_start:
            with patch.object(self.scheduler.scheduler, 'start') as mock_scheduler_start:
                with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
                    self.scheduler.start(role, host, disaggregated)
                    
                    # Should start both infer and scheduler
                    mock_infer_start.assert_called_once_with(role, host, disaggregated)
                    mock_scheduler_start.assert_called_once()
                    
                    # Should log start message
                    mock_logger.info.assert_called_once()

    def test_reset_nodeid(self):
        """Test reset_nodeid method"""
        new_nodeid = "new-node-123"
        
        self.scheduler.reset_nodeid(new_nodeid)
        
        # Should update both scheduler and infer nodeids
        self.assertEqual(self.scheduler.scheduler.nodeid, new_nodeid)
        self.assertEqual(self.scheduler.infer.nodeid, new_nodeid)

    def test_put_requests(self):
        """Test put_requests method"""
        req1 = create_mock_request("req1")
        req2 = create_mock_request("req2")
        reqs = [req1, req2]
        
        # Mock scheduler put_requests
        expected_result = [("req1", None), ("req2", None)]
        self.scheduler.scheduler.put_requests.return_value = expected_result
        
        result = self.scheduler.put_requests(reqs)
        
        # Should delegate to scheduler
        self.scheduler.scheduler.put_requests.assert_called_once_with(reqs)
        self.assertEqual(result, expected_result)

    def test_get_results(self):
        """Test get_results method"""
        # Mock scheduler get_results
        expected_result = {"req1": ["result1"], "req2": ["result2"]}
        self.scheduler.scheduler.get_results.return_value = expected_result
        
        result = self.scheduler.get_results()
        
        # Should delegate to scheduler
        self.scheduler.scheduler.get_results.assert_called_once()
        self.assertEqual(result, expected_result)

    def test_get_requests_insufficient_resources(self):
        """Test get_requests method with insufficient resources"""
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            result = self.scheduler.get_requests(
                available_blocks=5,  # Very low
                block_size=16,
                reserved_output_blocks=10,
                max_num_batched_tokens=2048,
                batch=1
            )
        
        # Should return empty list due to insufficient resources
        self.assertEqual(result, [])
        
        # Should log insufficient resources message
        mock_logger.info.assert_called_once()

    def test_get_requests_insufficient_batch(self):
        """Test get_requests method with insufficient batch size"""
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            result = self.scheduler.get_requests(
                available_blocks=100,
                block_size=16,
                reserved_output_blocks=10,
                max_num_batched_tokens=2048,
                batch=0  # Invalid batch size
            )
        
        # Should return empty list due to invalid batch
        self.assertEqual(result, [])
        
        # Should log insufficient resources message
        mock_logger.info.assert_called_once()

    def test_get_requests_sufficient_resources(self):
        """Test get_requests method with sufficient resources"""
        # Mock infer get_requests
        expected_result = [create_mock_request("req1"), create_mock_request("req2")]
        self.scheduler.infer.get_requests.return_value = expected_result
        
        result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        
        # Should delegate to infer
        self.scheduler.infer.get_requests.assert_called_once_with(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        self.assertEqual(result, expected_result)

    def test_put_results(self):
        """Test put_results method"""
        result1 = create_mock_request_output("req1", finished=True)
        result2 = create_mock_request_output("req2", finished=False)
        results = [result1, result2]
        
        # Mock infer put_results
        expected_result = "success"
        self.scheduler.infer.put_results.return_value = expected_result
        
        result = self.scheduler.put_results(results)
        
        # Should delegate to infer
        self.scheduler.infer.put_results.assert_called_once_with(results)
        self.assertEqual(result, expected_result)

    def test_get_requests_edge_case_available_blocks_equals_reserved(self):
        """Test get_requests method when available_blocks equals reserved_output_blocks"""
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            result = self.scheduler.get_requests(
                available_blocks=10,
                block_size=16,
                reserved_output_blocks=10,  # Equal to available_blocks
                max_num_batched_tokens=2048,
                batch=1
            )
        
        # Should return empty list
        self.assertEqual(result, [])
        
        # Should log insufficient resources message
        mock_logger.info.assert_called_once()

    def test_get_requests_with_different_batch_sizes(self):
        """Test get_requests method with different batch sizes"""
        # Mock infer get_requests
        expected_result = [create_mock_request("req1")]
        self.scheduler.infer.get_requests.return_value = expected_result
        
        # Test with batch size 1
        result1 = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=1
        )
        
        # Test with batch size 10
        result2 = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=10
        )
        
        # Both should succeed
        self.assertEqual(result1, expected_result)
        self.assertEqual(result2, expected_result)
        
        # Should call infer with correct batch sizes
        self.assertEqual(self.scheduler.infer.get_requests.call_count, 2)
        self.scheduler.infer.get_requests.assert_any_call(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=1
        )
        self.scheduler.infer.get_requests.assert_any_call(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=10
        )

    def test_put_requests_with_empty_list(self):
        """Test put_requests method with empty list"""
        result = self.scheduler.put_requests([])
        
        # Should handle empty list gracefully
        self.scheduler.scheduler.put_requests.assert_called_once_with([])
        self.assertEqual(result, [])

    def test_get_results_with_empty_result(self):
        """Test get_results method with empty result"""
        self.scheduler.scheduler.get_results.return_value = {}
        
        result = self.scheduler.get_results()
        
        # Should return empty dict
        self.assertEqual(result, {})

    def test_put_results_with_empty_list(self):
        """Test put_results method with empty list"""
        self.scheduler.infer.put_results.return_value = "success"
        
        result = self.scheduler.put_results([])
        
        # Should handle empty list gracefully
        self.scheduler.infer.put_results.assert_called_once_with([])
        self.assertEqual(result, "success")

    def test_integration_workflow(self):
        """Test complete workflow integration"""
        # Start scheduler
        role = "prefill"
        host = "192.168.1.100"
        disaggregated = {"transfer_protocol": ["ipc"]}
        
        with patch.object(self.scheduler.infer, 'start'):
            with patch.object(self.scheduler.scheduler, 'start'):
                self.scheduler.start(role, host, disaggregated)
        
        # Put requests
        req1 = create_mock_request("req1")
        req2 = create_mock_request("req2")
        reqs = [req1, req2]
        
        self.scheduler.scheduler.put_requests.return_value = [("req1", None), ("req2", None)]
        put_result = self.scheduler.put_requests(reqs)
        self.assertEqual(len(put_result), 2)
        
        # Get requests
        self.scheduler.infer.get_requests.return_value = [req1, req2]
        get_result = self.scheduler.get_requests(
            available_blocks=100,
            block_size=16,
            reserved_output_blocks=10,
            max_num_batched_tokens=2048,
            batch=5
        )
        self.assertEqual(len(get_result), 2)
        
        # Put results
        result1 = create_mock_request_output("req1", finished=True)
        result2 = create_mock_request_output("req2", finished=False)
        results = [result1, result2]
        
        self.scheduler.infer.put_results.return_value = "success"
        put_result = self.scheduler.put_results(results)
        self.assertEqual(put_result, "success")
        
        # Get results
        self.scheduler.scheduler.get_results.return_value = {"req1": [result1], "req2": [result2]}
        get_result = self.scheduler.get_results()
        self.assertEqual(len(get_result), 2)

    def test_reset_nodeid_preserves_other_attributes(self):
        """Test that reset_nodeid preserves other attributes"""
        # Set some initial state
        self.scheduler.scheduler.some_attr = "value1"
        self.scheduler.infer.some_attr = "value2"
        
        new_nodeid = "new-node-456"
        self.scheduler.reset_nodeid(new_nodeid)
        
        # Should update nodeids
        self.assertEqual(self.scheduler.scheduler.nodeid, new_nodeid)
        self.assertEqual(self.scheduler.infer.nodeid, new_nodeid)
        
        # Should preserve other attributes
        self.assertEqual(self.scheduler.scheduler.some_attr, "value1")
        self.assertEqual(self.scheduler.infer.some_attr, "value2")


if __name__ == '__main__':
    unittest.main()
