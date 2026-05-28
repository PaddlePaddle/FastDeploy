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
from unittest.mock import MagicMock, patch

from fastdeploy.cache_manager.v1.transfer.ipc.connector import IPCConnector


class TestIPCConnectorInit(unittest.TestCase):
    """Test IPCConnector.__init__."""

    def test_default_config(self):
        """Init with no config sets empty dict and empty buffers."""
        conn = IPCConnector()
        self.assertEqual(conn.config, {})
        self.assertEqual(conn._shm_buffers, {})
        self.assertEqual(conn._shm_paths, {})
        self.assertFalse(conn.is_connected())

    def test_custom_config(self):
        """Init with custom config stores it."""
        cfg = {"shm_path": "/dev/shm/test", "buffer_size": 4096, "max_buffers": 10}
        conn = IPCConnector(config=cfg)
        self.assertEqual(conn.config, cfg)
        self.assertEqual(conn.config["buffer_size"], 4096)


class TestIPCConnectorConnect(unittest.TestCase):
    """Test IPCConnector.connect and disconnect."""

    def test_connect_returns_true(self):
        """connect() sets _connected=True and returns True."""
        conn = IPCConnector()
        result = conn.connect()
        self.assertTrue(result)
        self.assertTrue(conn.is_connected())

    def test_disconnect_clears_state(self):
        """disconnect() closes shm, removes files, clears state."""
        conn = IPCConnector()
        conn.connect()

        # Set up mock shm buffers
        mock_shm = MagicMock()
        conn._shm_buffers = {"addr1": mock_shm}
        conn._shm_paths = {"addr1": "/dev/shm/kv_cache_addr1"}

        with patch("os.unlink") as mock_unlink:
            conn.disconnect()

        mock_shm.close.assert_called_once()
        mock_unlink.assert_called_once_with("/dev/shm/kv_cache_addr1")
        self.assertEqual(conn._shm_buffers, {})
        self.assertEqual(conn._shm_paths, {})
        self.assertFalse(conn.is_connected())

    def test_disconnect_handles_close_exception(self):
        """disconnect() swallows exceptions from shm.close()."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.close.side_effect = OSError("close failed")
        conn._shm_buffers = {"addr1": mock_shm}
        conn._shm_paths = {"addr1": "/dev/shm/kv_cache_addr1"}

        with patch("os.unlink"):
            conn.disconnect()

        # Should not raise, state is cleaned
        self.assertFalse(conn.is_connected())
        self.assertEqual(conn._shm_buffers, {})

    def test_disconnect_handles_unlink_exception(self):
        """disconnect() swallows exceptions from os.unlink()."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        conn._shm_buffers = {"addr1": mock_shm}
        conn._shm_paths = {"addr1": "/dev/shm/kv_cache_addr1"}

        with patch("os.unlink", side_effect=OSError("file not found")):
            conn.disconnect()

        self.assertFalse(conn.is_connected())


class TestIPCConnectorSend(unittest.TestCase):
    """Test IPCConnector.send."""

    def test_send_not_connected_returns_false(self):
        """send() returns False when not connected."""
        conn = IPCConnector()
        result = conn.send("addr", b"data", 4)
        self.assertFalse(result)

    def test_send_unknown_addr_returns_false(self):
        """send() returns False when dst_addr is not registered."""
        conn = IPCConnector()
        conn.connect()
        result = conn.send("unknown_addr", b"data", 4)
        self.assertFalse(result)

    def test_send_success(self):
        """send() writes data to shm buffer at offset."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        conn._shm_buffers["addr1"] = mock_shm

        data = b"hello world"
        result = conn.send("addr1", data, 5, dst_offset=10)

        self.assertTrue(result)
        mock_shm.seek.assert_called_once_with(10)
        mock_shm.write.assert_called_once_with(b"hello")

    def test_send_exception_returns_false(self):
        """send() returns False on exception."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.seek.side_effect = OSError("seek failed")
        conn._shm_buffers["addr1"] = mock_shm

        result = conn.send("addr1", b"data", 4)
        self.assertFalse(result)


class TestIPCConnectorRecv(unittest.TestCase):
    """Test IPCConnector.recv."""

    def test_recv_not_connected_returns_false(self):
        """recv() returns False when not connected."""
        conn = IPCConnector()
        result = conn.recv("addr", bytearray(10), 10)
        self.assertFalse(result)

    def test_recv_unknown_addr_returns_false(self):
        """recv() returns False when src_addr is not registered."""
        conn = IPCConnector()
        conn.connect()
        result = conn.recv("unknown", bytearray(10), 10)
        self.assertFalse(result)

    def test_recv_success(self):
        """recv() reads data from shm buffer into dst_buffer."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.read.return_value = b"hello"
        conn._shm_buffers["addr1"] = mock_shm

        dst_buffer = bytearray(10)
        result = conn.recv("addr1", dst_buffer, 5, src_offset=20)

        self.assertTrue(result)
        mock_shm.seek.assert_called_once_with(20)
        mock_shm.read.assert_called_once_with(5)
        self.assertEqual(dst_buffer[:5], b"hello")

    def test_recv_exception_returns_false(self):
        """recv() returns False on exception."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.read.side_effect = OSError("read failed")
        conn._shm_buffers["addr1"] = mock_shm

        dst_buffer = bytearray(10)
        result = conn.recv("addr1", dst_buffer, 5)
        self.assertFalse(result)


class TestIPCConnectorSendAsync(unittest.TestCase):
    """Test IPCConnector.send_async."""

    def test_send_async_success(self):
        """send_async() delegates to send() and returns handle dict."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        conn._shm_buffers["addr1"] = mock_shm

        handle = conn.send_async("addr1", b"data", 4, dst_offset=0)

        self.assertEqual(handle, {"success": True, "addr": "addr1"})

    def test_send_async_failure(self):
        """send_async() returns failure handle when send fails."""
        conn = IPCConnector()
        conn.connect()
        # No registered buffer for "missing"
        handle = conn.send_async("missing", b"data", 4)
        self.assertEqual(handle, {"success": False, "addr": "missing"})


class TestIPCConnectorRecvAsync(unittest.TestCase):
    """Test IPCConnector.recv_async."""

    def test_recv_async_success(self):
        """recv_async() delegates to recv() and returns handle dict."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.read.return_value = b"test"
        conn._shm_buffers["addr1"] = mock_shm

        dst = bytearray(10)
        handle = conn.recv_async("addr1", dst, 4, src_offset=0)

        self.assertEqual(handle, {"success": True, "addr": "addr1"})

    def test_recv_async_failure(self):
        """recv_async() returns failure handle when recv fails."""
        conn = IPCConnector()
        # Not connected
        dst = bytearray(10)
        handle = conn.recv_async("addr1", dst, 4)
        self.assertEqual(handle, {"success": False, "addr": "addr1"})


class TestIPCConnectorWait(unittest.TestCase):
    """Test IPCConnector.wait."""

    def test_wait_none_handle_returns_false(self):
        """wait() returns False for None handle."""
        conn = IPCConnector()
        self.assertFalse(conn.wait(None))

    def test_wait_success_handle(self):
        """wait() returns True for success handle."""
        conn = IPCConnector()
        self.assertTrue(conn.wait({"success": True, "addr": "x"}))

    def test_wait_failure_handle(self):
        """wait() returns False for failure handle."""
        conn = IPCConnector()
        self.assertFalse(conn.wait({"success": False, "addr": "x"}))

    def test_wait_missing_key_returns_false(self):
        """wait() returns False when 'success' key is missing."""
        conn = IPCConnector()
        self.assertFalse(conn.wait({"addr": "x"}))


class TestIPCConnectorRegisterBuffer(unittest.TestCase):
    """Test IPCConnector.register_buffer."""

    def test_register_not_connected_returns_false(self):
        """register_buffer() returns False when not connected."""
        conn = IPCConnector()
        result = conn.register_buffer(b"x" * 1024, "addr1")
        self.assertFalse(result)

    @patch("mmap.mmap")
    @patch("os.close")
    @patch("os.ftruncate")
    @patch("os.open", return_value=5)
    def test_register_buffer_with_len(self, mock_open, mock_ftruncate, mock_close, mock_mmap):
        """register_buffer() uses len(buffer) for size when available."""
        conn = IPCConnector()
        conn.connect()

        mock_mmap_instance = MagicMock()
        mock_mmap.return_value = mock_mmap_instance

        buffer = b"x" * 2048
        result = conn.register_buffer(buffer, "addr1")

        self.assertTrue(result)
        mock_open.assert_called_once_with("/dev/shm/kv_cache_addr1", 66, 0o666)
        mock_ftruncate.assert_called_once_with(5, 2048)
        mock_mmap.assert_called_once_with(5, 2048)
        mock_close.assert_called_once_with(5)
        self.assertEqual(conn._shm_buffers["addr1"], mock_mmap_instance)
        self.assertEqual(conn._shm_paths["addr1"], "/dev/shm/kv_cache_addr1")

    @patch("mmap.mmap")
    @patch("os.close")
    @patch("os.ftruncate")
    @patch("os.open", return_value=7)
    def test_register_buffer_without_len_uses_config(self, mock_open, mock_ftruncate, mock_close, mock_mmap):
        """register_buffer() falls back to config buffer_size."""
        conn = IPCConnector(config={"buffer_size": 8192})
        conn.connect()

        mock_mmap.return_value = MagicMock()

        # Object without __len__
        buffer = MagicMock(spec=[])
        result = conn.register_buffer(buffer, "addr2")

        self.assertTrue(result)
        mock_ftruncate.assert_called_once_with(7, 8192)

    @patch("os.open", side_effect=OSError("permission denied"))
    def test_register_buffer_exception_returns_false(self, mock_open):
        """register_buffer() returns False on exception."""
        conn = IPCConnector()
        conn.connect()

        result = conn.register_buffer(b"data", "addr1")
        self.assertFalse(result)


class TestIPCConnectorUnregisterBuffer(unittest.TestCase):
    """Test IPCConnector.unregister_buffer."""

    def test_unregister_unknown_addr_returns_false(self):
        """unregister_buffer() returns False for unknown addr."""
        conn = IPCConnector()
        result = conn.unregister_buffer("nonexistent")
        self.assertFalse(result)

    def test_unregister_success(self):
        """unregister_buffer() closes shm, unlinks file, removes entries."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        conn._shm_buffers["addr1"] = mock_shm
        conn._shm_paths["addr1"] = "/dev/shm/kv_cache_addr1"

        with patch("os.unlink") as mock_unlink:
            result = conn.unregister_buffer("addr1")

        self.assertTrue(result)
        mock_shm.close.assert_called_once()
        mock_unlink.assert_called_once_with("/dev/shm/kv_cache_addr1")
        self.assertNotIn("addr1", conn._shm_buffers)
        self.assertNotIn("addr1", conn._shm_paths)

    def test_unregister_without_shm_path(self):
        """unregister_buffer() works even if addr not in _shm_paths."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        conn._shm_buffers["addr1"] = mock_shm
        # No entry in _shm_paths

        result = conn.unregister_buffer("addr1")

        self.assertTrue(result)
        mock_shm.close.assert_called_once()
        self.assertNotIn("addr1", conn._shm_buffers)

    def test_unregister_exception_returns_false(self):
        """unregister_buffer() returns False on exception."""
        conn = IPCConnector()
        conn.connect()

        mock_shm = MagicMock()
        mock_shm.close.side_effect = OSError("close failed")
        conn._shm_buffers["addr1"] = mock_shm

        result = conn.unregister_buffer("addr1")
        self.assertFalse(result)


class TestIPCConnectorGetStats(unittest.TestCase):
    """Test IPCConnector.get_stats."""

    def test_stats_disconnected(self):
        """get_stats() returns base stats + buffer info when disconnected."""
        conn = IPCConnector(config={"key": "val"})
        stats = conn.get_stats()

        self.assertFalse(stats["connected"])
        self.assertEqual(stats["config"], {"key": "val"})
        self.assertEqual(stats["registered_buffers"], 0)
        self.assertEqual(stats["buffer_addresses"], [])

    def test_stats_with_buffers(self):
        """get_stats() includes registered buffer addresses."""
        conn = IPCConnector()
        conn.connect()
        conn._shm_buffers["buf_a"] = MagicMock()
        conn._shm_buffers["buf_b"] = MagicMock()

        stats = conn.get_stats()

        self.assertTrue(stats["connected"])
        self.assertEqual(stats["registered_buffers"], 2)
        self.assertIn("buf_a", stats["buffer_addresses"])
        self.assertIn("buf_b", stats["buffer_addresses"])


if __name__ == "__main__":
    unittest.main()
