"""
Simple tests for ZmqServerBase.recv_result_handle to cover the startup log line.
"""

import unittest

from fastdeploy.inter_communicator.zmq_server import ZmqServerBase


class _DummyServer(ZmqServerBase):
    """Minimal concrete subclass to satisfy abstract methods.

    We do not create any real ZMQ sockets; we only need to call
    recv_result_handle with running=False so the loop is skipped.
    """

    def __init__(self):
        super().__init__()
        self.socket = None
        self.running = False  # skip loop to just hit the startup log

    def _create_socket(self):  # pragma: no cover - not needed in this test
        return None

    def close(self):  # pragma: no cover - not needed in this test
        pass


class TestZmqServerRecvResultHandle(unittest.TestCase):
    def test_recv_result_handle_startup_log(self):
        """Just invoke recv_result_handle to execute the first log line (L123)."""
        srv = _DummyServer()
        # Should not raise; returns None after logging start/finish and skipping loop
        self.assertIsNone(srv.recv_result_handle())


if __name__ == "__main__":
    unittest.main()
