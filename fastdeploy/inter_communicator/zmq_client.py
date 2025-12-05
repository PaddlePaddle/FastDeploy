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

import time
from abc import ABC, abstractmethod
from multiprocessing.reduction import ForkingPickler

import zmq
from zmq.utils import jsonapi

from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.metrics.stats import ZMQMetricsStats
from fastdeploy.utils import llm_logger


class ZmqClientBase(ABC):
    """
    ZmqClientBase is a base class that provides a client-side interface for sending and receiving messages using ZeroMQ.
    """

    def __init__(self):
        self.address = None
        pass

    @abstractmethod
    def _create_socket(self):
        """Abstract method to create and return a ZeroMQ socket."""
        pass

    def _ensure_socket(self):
        """Ensure the socket is created before use."""
        if self.socket is None:
            self.socket: zmq.Socket = self._create_socket()

    @abstractmethod
    def connect(self):
        """
        Connect to the server using the file name specified in the constructor.
        """
        pass

    def send_json(self, data, flags: int = 0):
        """Send a Python object as a message using json to serialize.

        Keyword arguments are passed on to json.dumps

        Parameters
        ----------
        obj : Python object
            The Python object to send
        flags : int
            Any valid flags for :func:`Socket.send`
        """
        _zmq_metrics_stats = ZMQMetricsStats()
        try:
            # package data with meta information
            envelope = {"__meta": {"send_ts": time.perf_counter()}, "data": data}
            msg = jsonapi.dumps(envelope)

            # collect zmq send metrics
            _zmq_metrics_stats.msg_bytes_send_total += len(msg)

            return self.socket.send(msg, flags=flags)
        except Exception as e:
            _zmq_metrics_stats.msg_send_failed_total += 1
            raise e
        finally:
            # collect zmq send metrics
            _zmq_metrics_stats.msg_send_total += 1
            main_process_metrics.record_zmq_stats(_zmq_metrics_stats, self.address)

    def recv_json(self, flags: int = 0):
        """
        Receive a JSON-serializable object from the socket.
        """
        self._ensure_socket()
        _zmq_metrics_stats = ZMQMetricsStats()
        try:
            # receive from socket
            msg = self.socket.recv(flags=flags)
            data_dict = self.socket._deserialize(msg, lambda buf: jsonapi.loads(buf))

            # collect zmq recv metrics
            _zmq_metrics_stats.msg_bytes_recv_total += len(msg)
            _zmq_metrics_stats.msg_recv_total += 1

            # first check if the received msg is a dict
            if isinstance(data_dict, dict):
                # then check if the dict has "__meta" key
                if "__meta" in data_dict and "send_ts" in data_dict["__meta"]:
                    # if so, calculate the delay
                    _zmq_metrics_stats.zmq_latency = time.perf_counter() - data_dict["__meta"]["send_ts"]
                    return data_dict["data"]
            return data_dict
        finally:
            main_process_metrics.record_zmq_stats(_zmq_metrics_stats, self.address)

    def send_pyobj(self, data, flags: int = 0):
        """
        Send a Pickle-serializable object over the socket.
        """
        self._ensure_socket()
        _zmq_metrics_stats = ZMQMetricsStats()
        try:
            envelope = {"__meta": {"send_ts": time.perf_counter()}, "data": data}
            data_bytes = ForkingPickler.dumps(envelope)
            _zmq_metrics_stats.msg_bytes_send_total += len(data_bytes)
            self.socket.send(data_bytes, copy=False, flags=flags)
        except Exception as e:
            _zmq_metrics_stats.msg_send_failed_total += 1
            raise e
        finally:
            _zmq_metrics_stats.msg_send_total += 1
            main_process_metrics.record_zmq_stats(_zmq_metrics_stats, self.address)

    def recv_pyobj(self, flags: int = 0):
        """
        Receive a Pickle-serializable object from the socket.
        """
        _zmq_metrics_stats = ZMQMetricsStats()
        self._ensure_socket()
        data_bytes = self.socket.recv(flags=flags)
        envelope = ForkingPickler.loads(data_bytes)
        if isinstance(envelope, dict):
            if "__meta" in envelope and "send_ts" in envelope["__meta"]:
                _zmq_metrics_stats.msg_recv_total += 1
                _zmq_metrics_stats.msg_bytes_recv_total += len(data_bytes)
                _zmq_metrics_stats.zmq_latency = time.perf_counter() - envelope["__meta"]["send_ts"]
                main_process_metrics.record_zmq_stats(_zmq_metrics_stats, self.address)
                return envelope["data"]
        return envelope

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
        address = f"ipc://{self.file_name}"
        self.address = address
        self.socket.connect(address)

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
