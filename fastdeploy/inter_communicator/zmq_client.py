"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod

import zmq

from fastdeploy.utils import llm_logger


class ZmqClientBase(ABC):
    """
    ZmqClientBase is a base class that provides a client-side interface for sending and receiving messages using ZeroMQ.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _create_socket(self):
        """Abstract method to create and return a ZeroMQ socket."""
        pass

    def _ensure_socket(self):
        """Ensure the socket is created before use."""
        if self.socket is None:
            self.socket = self._create_socket()

    @abstractmethod
    def connect(self):
        """
        Connect to the server using the file name specified in the constructor.
        """
        pass

    def send_json(self, data):
        """
        Send a JSON-serializable object over the socket.
        """
        self._ensure_socket()
        self.socket.send_json(data)

    def recv_json(self):
        """
        Receive a JSON-serializable object from the socket.
        """
        self._ensure_socket()
        return self.socket.recv_json()

    def send_pyobj(self, data):
        """
        Send a Pickle-serializable object over the socket.
        """
        self._ensure_socket()
        self.socket.send_pyobj(data)

    def recv_pyobj(self):
        """
        Receive a Pickle-serializable object from the socket.
        """
        self._ensure_socket()
        return self.socket.recv_pyobj()

    @abstractmethod
    def close(self):
        pass


class ZmqIpcClient(ZmqClientBase):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        self.file_name = f"/dev/shm/{name}.socket"
        self.context = zmq.Context()
        self.socket = self.context.socket(self.mode)

    def _create_socket(self):
        """create and return a ZeroMQ socket."""
        self.context = zmq.Context()
        return self.context.socket(self.mode)

    def connect(self):
        self._ensure_socket()
        self.socket.connect(f"ipc://{self.file_name}")

    def close(self):
        """
        Close the socket and context.
        """
        llm_logger.info("ZMQ client is closing connection...")
        try:
            if self.socket is not None and not self.socket.closed:
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
            if self.context is not None:
                self.context.term()

        except Exception as e:
            llm_logger.warning(f"ZMQ client failed to close connection - {e}")
            return
