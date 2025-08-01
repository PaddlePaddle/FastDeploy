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

import threading
import time

import msgpack
import zmq

from fastdeploy import envs
from fastdeploy.utils import llm_logger


class ZmqTcpServer:
    """
    ZmqTcpServer, used when ENABLE_EXTERNAL_MODULE_ACCESS=1
    """

    def __init__(self, port, mode):
        self.context = zmq.Context()
        self.socket = self.context.socket(mode)
        self.mode = mode
        self.port = port
        self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.bind(f"tcp://*:{self.port}")
        self.ZMQ_SNDHWM = int(envs.FD_ZMQ_SNDHWM)
        self.socket.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.aggregate_send = envs.FD_USE_AGGREGATE_SEND

        self.mutex = threading.Lock()
        self.req_dict = dict()
        self.poller = None
        self.running = True
        if self.mode == zmq.PULL:
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)

    def send_json(self, data):
        """
        Send a JSON-serializable object over the socket.
        """
        self.socket.send_json(data)

    def recv_json(self):
        """
        Receive a JSON-serializable object from the socket.
        """
        return self.socket.recv_json()

    def send_pyobj(self, data):
        """
        Send a Pickle-serializable object over the socket.
        """
        self.socket.send_pyobj(data)

    def recv_pyobj(self):
        """
        Receive a Pickle-serializable object from the socket.
        """
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

    def recv_control_cmd(self):
        while self.running:
            try:
                client, _, task_data = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                task = msgpack.unpackb(task_data)
                task_id_str = task["task_id"]
            except zmq.Again:
                time.sleep(0.001)
                continue
            with self.mutex:
                self.req_dict[task_id_str] = client
            return task

    def response_for_control_cmd(self, task_id, result):
        """
        Send a multipart message to the control cmd socket.
        """
        if self.socket is None:
            raise RuntimeError("Router socket not created.")
        try:
            result = msgpack.packb(result)
            self.socket.send_multipart([self.req_dict[task_id], b"", result])

        except Exception as e:
            llm_logger.error(f"Send result to zmq client failed: {e}")

        with self.mutex:
            self.req_dict.pop(task_id, None)
        llm_logger.info(f"response contrl cmd finished, task_id: {task_id}")

    def send_multipart(self, req_id, data):
        """
        Send a multipart message to the router socket.
        """
        if self.socket is None:
            raise RuntimeError("Router socket not created. Call create_router() first.")

        while self.running:
            with self.mutex:
                if req_id not in self.req_dict:
                    try:
                        client, _, request_id = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                        req_id_str = request_id.decode("utf-8")
                        self.req_dict[req_id_str] = client
                    except zmq.Again:
                        time.sleep(0.001)
                        continue
                else:
                    break

        try:
            start_send = time.time()
            if self.aggregate_send:
                result = self.pack_aggregated_data(data)
            else:
                result = msgpack.packb([response.to_dict() for response in data])
            self.socket.send_multipart([self.req_dict[req_id], b"", result])
            llm_logger.debug(f"send_multipart result: {req_id} len {len(data)} elapse: {time.time()-start_send}")

        except Exception as e:
            llm_logger.error(f"Send result to zmq client failed: {e}")

        if data[-1].finished:
            with self.mutex:
                self.req_dict.pop(req_id, None)
            llm_logger.info(f"send_multipart finished, req_id: {req_id}")

    def receive_json_once(self, block=False):
        """
        Receive a single message from the socket.
        """
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

    def close(self):
        """
        Close the socket and context, and remove the IPC files.
        """
        if not self.running:
            return

        self.running = False
        llm_logger.info("Closing ZMQ connection...")
        try:
            if hasattr(self, "socket") and not self.socket.closed:
                self.socket.close()

            if self.socket is not None and not self.socket.closed:
                self.socket.close()

            if not self.context.closed:
                self.context.term()

        except Exception as e:
            llm_logger.warning(f"Failed to close ZMQ connection - {e}")
            return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()