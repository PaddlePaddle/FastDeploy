"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from fastdeploy.cache_manager.v1.transfer.rdma.connector import RDMAConnector


class TestRDMAConnectorInit(unittest.TestCase):
    """Test RDMAConnector.__init__."""

    def test_default_config(self):
        """Init with no config uses empty dict."""
        conn = RDMAConnector()
        self.assertEqual(conn.config, {})
        self.assertIsNone(conn._pd)
        self.assertIsNone(conn._cq)
        self.assertIsNone(conn._qp)
        self.assertIsNone(conn._mr)
        self.assertEqual(conn._buffers, {})
        self.assertFalse(conn.is_connected())

    def test_custom_config(self):
        """Init with custom config stores it."""
        cfg = {"device": "mlx5_0", "port": 1, "max_wr": 128, "buffer_size": 4096}
        conn = RDMAConnector(config=cfg)
        self.assertEqual(conn.config, cfg)
        self.assertEqual(conn.config["device"], "mlx5_0")


class TestRDMAConnectorConnect(unittest.TestCase):
    """Test RDMAConnector.connect."""

    def test_connect_returns_true(self):
        """connect() sets _connected=True and returns True."""
        conn = RDMAConnector()
        result = conn.connect()
        self.assertTrue(result)
        self.assertTrue(conn.is_connected())


class TestRDMAConnectorDisconnect(unittest.TestCase):
    """Test RDMAConnector.disconnect."""

    def test_disconnect_clears_all_state(self):
        """disconnect() resets all RDMA resources and disconnects."""
        conn = RDMAConnector()
        conn.connect()
        conn._buffers = {"addr1": b"data"}
        conn._mr = "mock_mr"
        conn._qp = "mock_qp"
        conn._cq = "mock_cq"
        conn._pd = "mock_pd"

        conn.disconnect()

        self.assertEqual(conn._buffers, {})
        self.assertIsNone(conn._mr)
        self.assertIsNone(conn._qp)
        self.assertIsNone(conn._cq)
        self.assertIsNone(conn._pd)
        self.assertFalse(conn.is_connected())

    def test_disconnect_when_already_disconnected(self):
        """disconnect() is safe when already disconnected."""
        conn = RDMAConnector()
        conn.disconnect()
        self.assertFalse(conn.is_connected())


class TestRDMAConnectorSend(unittest.TestCase):
    """Test RDMAConnector.send."""

    def test_send_not_connected_returns_false(self):
        """send() returns False when not connected."""
        conn = RDMAConnector()
        result = conn.send("addr", b"data", 4)
        self.assertFalse(result)

    def test_send_connected_returns_false_placeholder(self):
        """send() returns False (placeholder implementation)."""
        conn = RDMAConnector()
        conn.connect()
        result = conn.send("addr", b"data", 4, dst_offset=0)
        self.assertFalse(result)


class TestRDMAConnectorRecv(unittest.TestCase):
    """Test RDMAConnector.recv."""

    def test_recv_not_connected_returns_false(self):
        """recv() returns False when not connected."""
        conn = RDMAConnector()
        result = conn.recv("addr", bytearray(10), 10)
        self.assertFalse(result)

    def test_recv_connected_returns_false_placeholder(self):
        """recv() returns False (placeholder implementation)."""
        conn = RDMAConnector()
        conn.connect()
        result = conn.recv("addr", bytearray(10), 10, src_offset=0)
        self.assertFalse(result)


class TestRDMAConnectorSendAsync(unittest.TestCase):
    """Test RDMAConnector.send_async."""

    def test_send_async_not_connected_returns_none(self):
        """send_async() returns None when not connected."""
        conn = RDMAConnector()
        result = conn.send_async("addr", b"data", 4)
        self.assertIsNone(result)

    def test_send_async_connected_returns_none_placeholder(self):
        """send_async() returns None (placeholder implementation)."""
        conn = RDMAConnector()
        conn.connect()
        result = conn.send_async("addr", b"data", 4, dst_offset=0)
        self.assertIsNone(result)


class TestRDMAConnectorRecvAsync(unittest.TestCase):
    """Test RDMAConnector.recv_async."""

    def test_recv_async_not_connected_returns_none(self):
        """recv_async() returns None when not connected."""
        conn = RDMAConnector()
        result = conn.recv_async("addr", bytearray(10), 10)
        self.assertIsNone(result)

    def test_recv_async_connected_returns_none_placeholder(self):
        """recv_async() returns None (placeholder implementation)."""
        conn = RDMAConnector()
        conn.connect()
        result = conn.recv_async("addr", bytearray(10), 10, src_offset=0)
        self.assertIsNone(result)


class TestRDMAConnectorWait(unittest.TestCase):
    """Test RDMAConnector.wait."""

    def test_wait_not_connected_returns_false(self):
        """wait() returns False when not connected."""
        conn = RDMAConnector()
        result = conn.wait("some_handle")
        self.assertFalse(result)

    def test_wait_connected_returns_false_placeholder(self):
        """wait() returns False (placeholder implementation)."""
        conn = RDMAConnector()
        conn.connect()
        result = conn.wait("some_handle", timeout=5.0)
        self.assertFalse(result)


class TestRDMAConnectorRegisterBuffer(unittest.TestCase):
    """Test RDMAConnector.register_buffer."""

    def test_register_not_connected_returns_false(self):
        """register_buffer() returns False when not connected."""
        conn = RDMAConnector()
        result = conn.register_buffer(b"data", "addr1")
        self.assertFalse(result)

    def test_register_success(self):
        """register_buffer() stores buffer and returns True."""
        conn = RDMAConnector()
        conn.connect()

        buf = b"x" * 1024
        result = conn.register_buffer(buf, "addr1")

        self.assertTrue(result)
        self.assertIn("addr1", conn._buffers)
        self.assertEqual(conn._buffers["addr1"], buf)

    def test_register_multiple_buffers(self):
        """register_buffer() can register multiple buffers."""
        conn = RDMAConnector()
        conn.connect()

        conn.register_buffer(b"aaa", "buf_a")
        conn.register_buffer(b"bbb", "buf_b")

        self.assertEqual(len(conn._buffers), 2)
        self.assertIn("buf_a", conn._buffers)
        self.assertIn("buf_b", conn._buffers)


class TestRDMAConnectorUnregisterBuffer(unittest.TestCase):
    """Test RDMAConnector.unregister_buffer."""

    def test_unregister_unknown_addr_returns_false(self):
        """unregister_buffer() returns False for unknown addr."""
        conn = RDMAConnector()
        result = conn.unregister_buffer("nonexistent")
        self.assertFalse(result)

    def test_unregister_success(self):
        """unregister_buffer() removes buffer and returns True."""
        conn = RDMAConnector()
        conn.connect()
        conn.register_buffer(b"data", "addr1")

        result = conn.unregister_buffer("addr1")

        self.assertTrue(result)
        self.assertNotIn("addr1", conn._buffers)

    def test_unregister_only_removes_specified(self):
        """unregister_buffer() only removes the specified buffer."""
        conn = RDMAConnector()
        conn.connect()
        conn.register_buffer(b"aaa", "buf_a")
        conn.register_buffer(b"bbb", "buf_b")

        conn.unregister_buffer("buf_a")

        self.assertNotIn("buf_a", conn._buffers)
        self.assertIn("buf_b", conn._buffers)


class TestRDMAConnectorGetStats(unittest.TestCase):
    """Test RDMAConnector.get_stats."""

    def test_stats_empty(self):
        """get_stats() returns base stats + buffer count when empty."""
        conn = RDMAConnector(config={"device": "mlx5_0"})
        stats = conn.get_stats()

        self.assertFalse(stats["connected"])
        self.assertEqual(stats["config"], {"device": "mlx5_0"})
        self.assertEqual(stats["registered_buffers"], 0)

    def test_stats_with_buffers(self):
        """get_stats() reflects registered buffer count."""
        conn = RDMAConnector()
        conn.connect()
        conn.register_buffer(b"a", "addr1")
        conn.register_buffer(b"b", "addr2")

        stats = conn.get_stats()

        self.assertTrue(stats["connected"])
        self.assertEqual(stats["registered_buffers"], 2)


if __name__ == "__main__":
    unittest.main()
