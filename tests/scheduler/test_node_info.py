"""
Tests for NodeInfo class
"""
import unittest
import time
import threading
from unittest.mock import patch, Mock
import orjson

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastdeploy.scheduler.splitwise_scheduler import NodeInfo


class TestNodeInfo(unittest.TestCase):
    """Test cases for NodeInfo class"""

    def setUp(self):
        """Set up test fixtures"""
        self.nodeid = "test-node-123"
        self.role = "prefill"
        self.host = "192.168.1.100"
        self.disaggregated = {"transfer_protocol": ["ipc", "rdma"]}
        self.load = 50
        self.ts = time.time()

    def test_init(self):
        """Test NodeInfo initialization"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load, self.ts)
        
        self.assertEqual(node.nodeid, self.nodeid)
        self.assertEqual(node.role, self.role)
        self.assertEqual(node.host, self.host)
        self.assertEqual(node.disaggregated, self.disaggregated)
        self.assertEqual(node.load, self.load)
        self.assertEqual(node.ts, self.ts)
        self.assertIsInstance(node.lock, threading.Lock)
        self.assertEqual(node.reqs, {})

    def test_init_with_default_timestamp(self):
        """Test NodeInfo initialization with default timestamp"""
        with patch('time.time', return_value=1234567890.0):
            node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load)
            self.assertEqual(node.ts, 1234567890.0)

    def test_repr(self):
        """Test NodeInfo string representation"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load, self.ts)
        expected_repr = f"{self.nodeid}({self.load})"
        self.assertEqual(repr(node), expected_repr)

    def test_expired_true(self):
        """Test expired method returns True when node is expired"""
        old_ts = time.time() - 10  # 10 seconds ago
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load, old_ts)
        expire_period = 5  # 5 seconds
        
        self.assertTrue(node.expired(expire_period))

    def test_expired_false(self):
        """Test expired method returns False when node is not expired"""
        recent_ts = time.time() - 1  # 1 second ago
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load, recent_ts)
        expire_period = 5  # 5 seconds
        
        self.assertFalse(node.expired(expire_period))

    def test_serialize(self):
        """Test serialize method"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, self.load, self.ts)
        
        with patch('time.time', return_value=1234567890.0):
            serialized = node.serialize()
            
        # Verify timestamp is updated
        self.assertEqual(node.ts, 1234567890.0)
        
        # Verify serialized data
        data = orjson.loads(serialized)
        expected_data = {
            "ts": 1234567890.0,
            "role": self.role,
            "load": self.load,
            "host": self.host,
            "disaggregated": self.disaggregated
        }
        self.assertEqual(data, expected_data)

    def test_load_from_classmethod(self):
        """Test load_from class method"""
        health_data = {
            "ts": 1234567890.0,
            "role": "decode",
            "load": 75,
            "host": "192.168.1.200",
            "disaggregated": {"transfer_protocol": ["rdma"]}
        }
        health_str = orjson.dumps(health_data)
        
        node = NodeInfo.load_from("test-node-456", health_str)
        
        self.assertEqual(node.nodeid, "test-node-456")
        self.assertEqual(node.ts, 1234567890.0)
        self.assertEqual(node.role, "decode")
        self.assertEqual(node.load, 75)
        self.assertEqual(node.host, "192.168.1.200")
        self.assertEqual(node.disaggregated, {"transfer_protocol": ["rdma"]})

    def test_lt_comparison(self):
        """Test less than comparison"""
        node1 = NodeInfo("node1", self.role, self.host, self.disaggregated, 10, self.ts)
        node2 = NodeInfo("node2", self.role, self.host, self.disaggregated, 20, self.ts)
        
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)

    def test_expire_reqs(self):
        """Test expire_reqs method"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        # Add some requests
        node.add_req("req1", 10)
        node.add_req("req2", 20)
        node.add_req("req3", 30)
        
        self.assertEqual(node.load, 60)
        self.assertEqual(len(node.reqs), 3)
        
        # Mock time to make req2 and req3 expired
        with patch('time.time', return_value=time.time() + 1000):
            with patch('fastdeploy.scheduler.splitwise_scheduler.logger') as mock_logger:
                node.expire_reqs(500)  # 500 seconds TTL
        
        # Only req1 should remain (recently added)
        self.assertEqual(node.load, 10)
        self.assertEqual(len(node.reqs), 1)
        self.assertIn("req1", node.reqs)
        self.assertNotIn("req2", node.reqs)
        self.assertNotIn("req3", node.reqs)

    def test_add_req(self):
        """Test add_req method"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        with patch('time.time', return_value=1234567890.0):
            node.add_req("req1", 25)
        
        self.assertEqual(node.load, 25)
        self.assertEqual(len(node.reqs), 1)
        self.assertIn("req1", node.reqs)
        self.assertEqual(node.reqs["req1"], [25, 1234567890.0])

    def test_add_req_duplicate(self):
        """Test add_req method with duplicate request ID"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        node.add_req("req1", 25)
        node.add_req("req1", 30)  # Duplicate ID
        
        # Should not add duplicate
        self.assertEqual(node.load, 25)
        self.assertEqual(len(node.reqs), 1)

    def test_update_req_timestamp(self):
        """Test update_req_timestamp method"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        node.add_req("req1", 25)
        node.add_req("req2", 30)
        
        original_ts1 = node.reqs["req1"][1]
        original_ts2 = node.reqs["req2"][1]
        
        with patch('time.time', return_value=1234567890.0):
            node.update_req_timestamp(["req1", "req2"])
        
        # Timestamps should be updated
        self.assertEqual(node.reqs["req1"][1], 1234567890.0)
        self.assertEqual(node.reqs["req2"][1], 1234567890.0)
        self.assertNotEqual(node.reqs["req1"][1], original_ts1)
        self.assertNotEqual(node.reqs["req2"][1], original_ts2)

    def test_update_req_timestamp_nonexistent(self):
        """Test update_req_timestamp with non-existent request IDs"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        node.add_req("req1", 25)
        
        # Update non-existent request - should not raise error
        node.update_req_timestamp(["req2", "req3"])
        
        # req1 should remain unchanged
        self.assertEqual(len(node.reqs), 1)
        self.assertIn("req1", node.reqs)

    def test_finish_req(self):
        """Test finish_req method"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        node.add_req("req1", 25)
        node.add_req("req2", 30)
        
        self.assertEqual(node.load, 55)
        self.assertEqual(len(node.reqs), 2)
        
        node.finish_req("req1")
        
        self.assertEqual(node.load, 30)
        self.assertEqual(len(node.reqs), 1)
        self.assertNotIn("req1", node.reqs)
        self.assertIn("req2", node.reqs)

    def test_finish_req_nonexistent(self):
        """Test finish_req with non-existent request ID"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        node.add_req("req1", 25)
        original_load = node.load
        
        # Finish non-existent request - should not raise error
        node.finish_req("req2")
        
        # Load should remain unchanged
        self.assertEqual(node.load, original_load)
        self.assertEqual(len(node.reqs), 1)

    def test_thread_safety(self):
        """Test thread safety of NodeInfo operations"""
        node = NodeInfo(self.nodeid, self.role, self.host, self.disaggregated, 0, self.ts)
        
        def add_requests():
            for i in range(100):
                node.add_req(f"req_{i}", 1)
        
        def finish_requests():
            for i in range(50):
                node.finish_req(f"req_{i}")
        
        # Run operations in parallel
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=add_requests))
            threads.append(threading.Thread(target=finish_requests))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final state is consistent
        self.assertGreaterEqual(node.load, 0)
        self.assertGreaterEqual(len(node.reqs), 0)


if __name__ == '__main__':
    unittest.main()
