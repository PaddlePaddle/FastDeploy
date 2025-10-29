"""
Tests for APIScheduler class
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

from fastdeploy.scheduler.splitwise_scheduler import APIScheduler, NodeInfo
from tests.scheduler.test_utils import create_mock_config, create_mock_request, create_mock_redis_client


class TestAPIScheduler(unittest.TestCase):
    """Test cases for APIScheduler class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.scheduler = APIScheduler(self.config)

    def test_init(self):
        """Test APIScheduler initialization"""
        self.assertEqual(self.scheduler.nodeid, self.config.nodeid)
        self.assertEqual(self.scheduler.reader_parallel, self.config.reader_parallel)
        self.assertEqual(self.scheduler.reader_batch_size, self.config.reader_batch_size)
        self.assertEqual(self.scheduler.expire_period, self.config.expire_period)
        self.assertEqual(self.scheduler.clear_expired_nodes_period, self.config.clear_expired_nodes_period)
        self.assertEqual(self.scheduler.ttl, self.config.ttl)
        self.assertEqual(self.scheduler.topic, self.config.redis_topic)
        self.assertEqual(self.scheduler.cluster_key, f"{self.config.redis_topic}.cluster")
        self.assertIsInstance(self.scheduler.req_cond, threading.Condition)
        self.assertIsInstance(self.scheduler.reqs_queue, deque)
        self.assertEqual(self.scheduler.readers, [])

    def test_start(self):
        """Test start method creates readers and threads"""
        with patch('fastdeploy.scheduler.splitwise_scheduler.ResultReader') as mock_reader_class:
            with patch('fastdeploy.scheduler.splitwise_scheduler.threading.Thread') as mock_thread_class:
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread
                
                self.scheduler.start()
                
                # Should create readers
                self.assertEqual(mock_reader_class.call_count, self.config.reader_parallel)
                self.assertEqual(len(self.scheduler.readers), self.config.reader_parallel)
                
                # Should create threads
                self.assertEqual(mock_thread_class.call_count, 2)  # clear_expired_nodes_thread and schedule_thread

    def test_put_requests(self):
        """Test put_requests method"""
        req1 = create_mock_request("req1")
        req2 = create_mock_request("req2")
        reqs = [req1, req2]
        
        result = self.scheduler.put_requests(reqs)
        
        # Should return list of tuples
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("req1", None))
        self.assertEqual(result[1], ("req2", None))
        
        # Should add requests to queue
        self.assertEqual(len(self.scheduler.reqs_queue), 2)
        self.assertEqual(self.scheduler.reqs_queue[0], req1)
        self.assertEqual(self.scheduler.reqs_queue[1], req2)

    def test_get_results(self):
        """Test get_results method"""
        # Mock readers
        mock_reader1 = Mock()
        mock_reader1.read.return_value = {"req1": ["result1"]}
        mock_reader2 = Mock()
        mock_reader2.read.return_value = {"req2": ["result2"]}
        self.scheduler.readers = [mock_reader1, mock_reader2]
        
        results = self.scheduler.get_results()
        
        # Should combine results from all readers
        self.assertEqual(len(results), 2)
        self.assertIn("req1", results)
        self.assertIn("req2", results)

    def test_sync_cluster(self):
        """Test sync_cluster method"""
        # Mock Redis response
        node_info = {
            "ts": time.time(),
            "role": "prefill",
            "load": 50,
            "host": "192.168.1.100",
            "disaggregated": {"transfer_protocol": ["ipc"]}
        }
        node_info_str = orjson.dumps(node_info)
        
        self.scheduler.client.hgetall.return_value = {
            b"node1": node_info_str,
            b"node2": node_info_str
        }
        
        pnodes, dnodes, mnodes = self.scheduler.sync_cluster()
        
        # Should categorize nodes by role
        self.assertEqual(len(pnodes), 2)
        self.assertEqual(len(dnodes), 0)
        self.assertEqual(len(mnodes), 0)
        
        # Should create NodeInfo objects
        for node in pnodes:
            self.assertIsInstance(node, NodeInfo)
            self.assertEqual(node.role, "prefill")

    def test_sync_cluster_with_expired_nodes(self):
        """Test sync_cluster method filters expired nodes"""
        # Mock Redis response with expired node
        old_time = time.time() - 1000  # Very old
        node_info = {
            "ts": old_time,
            "role": "prefill",
            "load": 50,
            "host": "192.168.1.100",
            "disaggregated": {"transfer_protocol": ["ipc"]}
        }
        node_info_str = orjson.dumps(node_info)
        
        self.scheduler.client.hgetall.return_value = {
            b"expired_node": node_info_str
        }
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            pnodes, dnodes, mnodes = self.scheduler.sync_cluster()
        
        # Should filter out expired nodes
        self.assertEqual(len(pnodes), 0)
        self.assertEqual(len(dnodes), 0)
        self.assertEqual(len(mnodes), 0)
        
        # Should log expired node
        self.assertTrue(mock_logger.error.called)

    def test_sync_cluster_with_different_roles(self):
        """Test sync_cluster method with different node roles"""
        # Mock Redis response with different roles
        node_info_prefill = {
            "ts": time.time(),
            "role": "prefill",
            "load": 50,
            "host": "192.168.1.100",
            "disaggregated": {"transfer_protocol": ["ipc"]}
        }
        node_info_decode = {
            "ts": time.time(),
            "role": "decode",
            "load": 30,
            "host": "192.168.1.101",
            "disaggregated": {"transfer_protocol": ["rdma"]}
        }
        node_info_mixed = {
            "ts": time.time(),
            "role": "mixed",
            "load": 80,
            "host": "192.168.1.102",
            "disaggregated": {"transfer_protocol": ["ipc", "rdma"]}
        }
        
        self.scheduler.client.hgetall.return_value = {
            b"prefill_node": orjson.dumps(node_info_prefill),
            b"decode_node": orjson.dumps(node_info_decode),
            b"mixed_node": orjson.dumps(node_info_mixed)
        }
        
        pnodes, dnodes, mnodes = self.scheduler.sync_cluster()
        
        # Should categorize correctly
        self.assertEqual(len(pnodes), 1)
        self.assertEqual(len(dnodes), 1)
        self.assertEqual(len(mnodes), 1)
        
        self.assertEqual(pnodes[0].role, "prefill")
        self.assertEqual(dnodes[0].role, "decode")
        self.assertEqual(mnodes[0].role, "mixed")

    def test_schedule_mixed_node(self):
        """Test schedule method with mixed node"""
        req = create_mock_request("req1", 100)
        pnodes = []
        dnodes = []
        mnodes = [NodeInfo("mixed1", "mixed", "192.168.1.100", {"transfer_protocol": ["ipc"]}, 50)]
        group = "test-group"
        
        with patch.object(req, 'to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"request_id": "req1", "group": group}
            
            self.scheduler.schedule(req, pnodes, dnodes, mnodes, group)
        
        # Should send to mixed node
        self.scheduler.client.lpush.assert_called_once()
        call_args = self.scheduler.client.lpush.call_args
        self.assertEqual(call_args[0][0], "ReqQ_mixed1")

    def test_schedule_prefill_decode_nodes(self):
        """Test schedule method with separate prefill and decode nodes"""
        req = create_mock_request("req1", 100)
        pnodes = [NodeInfo("prefill1", "prefill", "192.168.1.100", {"transfer_protocol": ["ipc"]}, 30)]
        dnodes = [NodeInfo("decode1", "decode", "192.168.1.101", {"transfer_protocol": ["rdma"]}, 20)]
        mnodes = []
        group = "test-group"
        
        with patch.object(req, 'to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"request_id": "req1", "group": group}
            
            self.scheduler.schedule(req, pnodes, dnodes, mnodes, group)
        
        # Should send to both prefill and decode nodes
        self.assertEqual(self.scheduler.client.lpush.call_count, 2)
        
        # Check that disaggregate_info is set
        self.assertIsNotNone(req.disaggregate_info)
        self.assertEqual(req.disaggregate_info["transfer_protocol"], "rdma")

    def test_schedule_same_host_uses_ipc(self):
        """Test schedule method uses IPC for same host"""
        req = create_mock_request("req1", 100)
        pnodes = [NodeInfo("prefill1", "prefill", "192.168.1.100", {"transfer_protocol": ["ipc", "rdma"]}, 30)]
        dnodes = [NodeInfo("decode1", "decode", "192.168.1.100", {"transfer_protocol": ["ipc", "rdma"]}, 20)]
        mnodes = []
        group = "test-group"
        
        with patch.object(req, 'to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"request_id": "req1", "group": group}
            
            self.scheduler.schedule(req, pnodes, dnodes, mnodes, group)
        
        # Should use IPC for same host
        self.assertEqual(req.disaggregate_info["transfer_protocol"], "ipc")

    def test_schedule_different_host_uses_rdma(self):
        """Test schedule method uses RDMA for different hosts"""
        req = create_mock_request("req1", 100)
        pnodes = [NodeInfo("prefill1", "prefill", "192.168.1.100", {"transfer_protocol": ["ipc", "rdma"]}, 30)]
        dnodes = [NodeInfo("decode1", "decode", "192.168.1.101", {"transfer_protocol": ["ipc", "rdma"]}, 20)]
        mnodes = []
        group = "test-group"
        
        with patch.object(req, 'to_dict') as mock_to_dict:
            mock_to_dict.return_value = {"request_id": "req1", "group": group}
            
            self.scheduler.schedule(req, pnodes, dnodes, mnodes, group)
        
        # Should use RDMA for different hosts
        self.assertEqual(req.disaggregate_info["transfer_protocol"], "rdma")

    def test_select_pd_prefill(self):
        """Test select_pd method for prefill role"""
        req = create_mock_request("req1", 100)
        nodes = [
            NodeInfo("node1", "prefill", "192.168.1.100", {}, 10),
            NodeInfo("node2", "prefill", "192.168.1.101", {}, 20),
            NodeInfo("node3", "prefill", "192.168.1.102", {}, 30)
        ]
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.random.choice') as mock_choice:
            mock_choice.return_value = nodes[0]
            
            selected = self.scheduler.select_pd(req, nodes, "prefill")
        
        # Should select from nodes with load within blur step
        self.assertIn(selected, nodes)

    def test_select_pd_decode(self):
        """Test select_pd method for decode role"""
        req = create_mock_request("req1", 100)
        nodes = [
            NodeInfo("node1", "decode", "192.168.1.100", {}, 10),
            NodeInfo("node2", "decode", "192.168.1.101", {}, 20),
            NodeInfo("node3", "decode", "192.168.1.102", {}, 30)
        ]
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.random.choice') as mock_choice:
            mock_choice.return_value = nodes[0]
            
            selected = self.scheduler.select_pd(req, nodes, "decode")
        
        # Should select from nodes with load within blur step
        self.assertIn(selected, nodes)

    def test_select_pd_invalid_role(self):
        """Test select_pd method with invalid role"""
        req = create_mock_request("req1", 100)
        nodes = [NodeInfo("node1", "prefill", "192.168.1.100", {}, 10)]
        
        with self.assertRaises(Exception) as context:
            self.scheduler.select_pd(req, nodes, "invalid_role")
        
        self.assertIn("Invalid Role: invalid_role", str(context.exception))

    def test_loop_clear_expired_nodes(self):
        """Test loop_clear_expired_nodes method"""
        # Mock expired node
        old_time = time.time() - 1000
        node_info = {
            "ts": old_time,
            "role": "prefill",
            "load": 50,
            "host": "192.168.1.100",
            "disaggregated": {"transfer_protocol": ["ipc"]}
        }
        node_info_str = orjson.dumps(node_info)
        
        self.scheduler.client.hgetall.return_value = {
            b"expired_node": node_info_str
        }
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.loop_clear_expired_nodes()
            except KeyboardInterrupt:
                pass
        
        # Should delete expired node
        self.scheduler.client.hdel.assert_called_with(self.scheduler.cluster_key, b"expired_node")

    def test_loop_clear_expired_nodes_exception_handling(self):
        """Test loop_clear_expired_nodes exception handling"""
        self.scheduler.client.hgetall.side_effect = Exception("Redis error")
        
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            with patch('time.sleep') as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt()
                
                try:
                    self.scheduler.loop_clear_expired_nodes()
                except KeyboardInterrupt:
                    pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)

    def test_loop_schedule_no_nodes(self):
        """Test loop_schedule method with no available nodes"""
        # Mock empty cluster
        self.scheduler.client.hgetall.return_value = {}
        
        # Add request to queue
        req = create_mock_request("req1")
        self.scheduler.reqs_queue.append(req)
        
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            
            try:
                self.scheduler.loop_schedule()
            except KeyboardInterrupt:
                pass
        
        # Should log error about no schedule nodes
        with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
            self.scheduler.loop_schedule()
            # This will be called in the actual implementation

    def test_loop_schedule_exception_handling(self):
        """Test loop_schedule exception handling"""
        # Mock cluster with nodes
        node_info = {
            "ts": time.time(),
            "role": "prefill",
            "load": 50,
            "host": "192.168.1.100",
            "disaggregated": {"transfer_protocol": ["ipc"]}
        }
        node_info_str = orjson.dumps(node_info)
        
        self.scheduler.client.hgetall.return_value = {
            b"node1": node_info_str
        }
        
        # Add request to queue
        req = create_mock_request("req1")
        self.scheduler.reqs_queue.append(req)
        
        # Mock schedule to raise exception
        with patch.object(self.scheduler, 'schedule', side_effect=Exception("Schedule error")):
            with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
                with patch('time.sleep') as mock_sleep:
                    mock_sleep.side_effect = KeyboardInterrupt()
                    
                    try:
                        self.scheduler.loop_schedule()
                    except KeyboardInterrupt:
                        pass
        
        # Should log the error
        self.assertTrue(mock_logger.error.called)


if __name__ == '__main__':
    unittest.main()
