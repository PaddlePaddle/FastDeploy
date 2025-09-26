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

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import msgpack
import zmq

from fastdeploy import envs
from fastdeploy.utils import llm_logger


class ZmqServerBase(ABC):
    """
    ZmqServerBase
    """

    def __init__(self):
        self.cached_results = defaultdict(list)
        self.response_token_lock = threading.Lock()

    @abstractmethod
    def _create_socket(self):
        """Abstract method to create and return a ZeroMQ socket."""
        pass

    def _ensure_socket(self):
        """Ensure the socket is created before use."""
        if self.socket is None:
            self.socket = self._create_socket()

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

    def pack_aggregated_data(self, data):
        """
        Aggregate multiple responses into one and send them to the client.
        """
        result = data[0]
        if len(data) > 1:
            for response in data[1:]:
                result.add(response)
        result = msgpack.packb([result.to_dict()])
        return result

    def receive_json_once(self, block=False):
        """
        Receive a single message from the socket.
        """
        self._ensure_socket()
        if self.socket is None or self.socket.closed:
            return "zmp socket has closed", None
        try:
            flags = zmq.NOBLOCK if not block else 0
            return None, self.socket.recv_json(flags=flags)
        except zmq.Again:
            return None, None
        except Exception as e:
            self.close()
            llm_logger.warning(f"{e}")
            return str(e), None

    def receive_pyobj_once(self, block=False):
        """
        Receive a single message from the socket.
        """
        self._ensure_socket()
        if self.socket is None or self.socket.closed:
            return "zmp socket has closed", None
        try:
            flags = zmq.NOBLOCK if not block else 0
            return None, self.socket.recv_pyobj(flags=flags)
        except zmq.Again:
            return None, None
        except Exception as e:
            self.close()
            llm_logger.warning(f"{e}")
            return str(e), None

    def recv_result_handle(self):
        while True:
            try:
                with self.response_token_lock:
                    client, _, request_id = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                req_id_str = request_id.decode("utf-8")
                need_send_after_finished_inference = False
                with self.mutex:
                    self.req_dict[req_id_str] = client
                    if req_id_str in self.cached_results:
                        if self.cached_results[req_id_str][-1][-1].finished:
                            need_send_after_finished_inference = True
                if need_send_after_finished_inference:
                    self.send_response(req_id_str, [])
                    llm_logger.info(f"send_multipart finished, req_id: {req_id_str}")
                    self.req_dict.pop(req_id_str, None)

            except zmq.Again:
                time.sleep(0.001)
                continue
            except Exception as e:
                llm_logger.error(f"recv_result_handle get unknown exception: {e}")
                continue

    def send_response(self, req_id, data):
        """
        Send generated token result to client.
        """
        self._ensure_socket()
        if self.socket is None:
            raise RuntimeError("Router socket not created. Call create_router() first.")
        new_data = []
        has_result_handle = False
        with self.mutex:
            if req_id not in self.req_dict:
                self.cached_results[req_id].append(data)
            else:
                has_result_handle = True
                if req_id in self.cached_results:
                    for history_data in self.cached_results[req_id]:
                        new_data.extend(history_data)
                    llm_logger.info(
                        f"get request {req_id} result handle after cached result, total cached length {len(self.cached_results[req_id])}"
                    )
                    del self.cached_results[req_id]
        if has_result_handle:
            try:
                new_data.extend(data)
                start_send = time.time()
                if self.aggregate_send:
                    result = self.pack_aggregated_data(new_data)
                else:
                    result = msgpack.packb([response.to_dict() for response in new_data])
                with self.response_token_lock:
                    self.socket.send_multipart([self.req_dict[req_id], b"", result])
                llm_logger.debug(
                    f"send_multipart result: {req_id} len {len(new_data)} elapse: {time.time()-start_send}"
                )

            except Exception as e:
                llm_logger.error(f"Send result to zmq client failed: {e}")

        if data and data[-1].finished:
            with self.mutex:
                if req_id in self.req_dict:
                    llm_logger.info(f"send_multipart finished, req_id: {req_id}")
                    self.req_dict.pop(req_id, None)

    @abstractmethod
    def close(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ZmqIpcServer(ZmqServerBase):
    """
    ZmqIpcServer, used when FD_ENABLE_INTERNAL_ADAPTER=0
    """

    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        self.cached_results = defaultdict(list)
        if mode == zmq.PULL:
            self.file_name = f"/dev/shm/{name}.socket"
        elif mode == zmq.ROUTER:
            self.file_name = f"/dev/shm/router_{name}.ipc"
        self.ZMQ_SNDHWM = int(envs.FD_ZMQ_SNDHWM)
        self.aggregate_send = envs.FD_USE_AGGREGATE_SEND
        self.mutex = threading.Lock()
        self.response_token_lock = threading.Lock()
        self.req_dict = dict()
        self.running = True
        self.context = zmq.Context()
        self._create_socket()

    def _create_socket(self):
        """create and return a ZeroMQ socket."""
        self.socket = self.context.socket(self.mode)
        self.socket.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.bind(f"ipc://{self.file_name}")
        return self.socket

    def _clear_ipc(self, name):
        """
        Remove the IPC file with the given name.
        """
        if os.path.exists(name):
            try:
                os.remove(name)
            except OSError as e:
                llm_logger.warning(f"Failed to remove IPC file {name} - {e}")

    def close(self):
        """
        Close the socket and context, and remove the IPC files.
        """
        if not self.running:
            return

        self.running = False
        llm_logger.info("ZMQ server is closing connection...")
        try:
            if self.socket is not None and not self.socket.closed:
                self.socket.close()
            if not self.context.closed:
                self.context.term()
            self._clear_ipc(self.file_name)
        except Exception as e:
            llm_logger.warning(f"ZMQ server failed to close connection - {e}")
            return


class ZmqTcpServer(ZmqServerBase):
    """
    ZmqTcpServer, used when FD_ENABLE_INTERNAL_ADAPTER=1
    """

    def __init__(self, port, mode):
        self.mode = mode
        self.port = port
        self.cached_results = defaultdict(list)
        self.ZMQ_SNDHWM = int(envs.FD_ZMQ_SNDHWM)
        self.aggregate_send = envs.FD_USE_AGGREGATE_SEND

        self.mutex = threading.Lock()
        self.req_dict = dict()
        self.running = True
        self.context = zmq.Context()
        self._create_socket()
        self.response_token_lock = threading.Lock()

    def _create_socket(self):
        """create and return a ZeroMQ socket."""
        self.socket = self.context.socket(self.mode)
        self.socket.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.bind(f"tcp://*:{self.port}")
        return self.socket

    def recv_control_cmd(self):
        """
        Recieve control command from client
        """
        self._ensure_socket()
        try:
            client, _, task_data = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            task = msgpack.unpackb(task_data)
            task_id_str = task["task_id"]
        except zmq.Again:
            return None
        with self.mutex:
            self.req_dict[task_id_str] = client
        return task

    def response_for_control_cmd(self, task_id, result):
        """
        Send command result back to client.
        """
        self._ensure_socket()
        if self.socket is None:
            raise RuntimeError("Router socket not created.")
        try:
            result = msgpack.packb(result)
            self.socket.send_multipart([self.req_dict[task_id], b"", result])

        except Exception as e:
            llm_logger.error(f"Send result to zmq client failed: {e}")

        with self.mutex:
            self.req_dict.pop(task_id, None)
        llm_logger.debug(f"response control cmd finished, task_id: {task_id}")

    def close(self):
        """
        Close the socket and context.
        """
        if not self.running:
            return

        self.running = False
        llm_logger.info("ZMQ server is closing connection...")
        try:
            if self.socket is not None and not self.socket.closed:
                self.socket.close()
            if not self.context.closed:
                self.context.term()

        except Exception as e:
            llm_logger.warning(f"ZMQ server failed to close connection - {e}")
            return
