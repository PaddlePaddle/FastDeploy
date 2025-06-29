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

import json
import os
import threading
import time

import zmq

from fastdeploy import envs
from fastdeploy.utils import llm_logger


class ZmqClient:
    """
    ZmqClient is a class that provides a client-side interface for sending and receiving messages using ZeroMQ.
    """

    def __init__(self, name, mode):
        self.context = zmq.Context()
        self.socket = self.context.socket(mode)
        self.file_name = f"/dev/shm/{name}.socket"
        self.router_path = f"/dev/shm/router_{name}.ipc"

        self.ZMQ_SNDHWM = int(envs.FD_ZMQ_SNDHWM)

        self.mutex = threading.Lock()
        self.req_dict = dict()
        self.router = None
        self.poller = None
        self.running = True

    def connect(self):
        """
        Connect to the server using the file name specified in the constructor.
        """
        self.socket.connect(f"ipc://{self.file_name}")

    def start_server(self):
        """
        Start the server using the file name specified in the constructor.
        """
        self.socket.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.bind(f"ipc://{self.file_name}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def create_router(self):
        """
        Create a ROUTER socket and bind it to the specified router path.
        """
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.SNDHWM, self.ZMQ_SNDHWM)
        self.router.setsockopt(zmq.SNDTIMEO, -1)
        self.router.bind(f"ipc://{self.router_path}")

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

    def send_multipart(self, req_id, data):
        """
        Send a multipart message to the router socket.
        """
        if self.router is None:
            raise RuntimeError(
                "Router socket not created. Call create_router() first.")

        while self.running:
            with self.mutex:
                if req_id not in self.req_dict:
                    try:
                        client, _, request_id = self.router.recv_multipart(
                            flags=zmq.NOBLOCK)
                        req_id_str = request_id.decode('utf-8')
                        self.req_dict[req_id_str] = client
                    except zmq.Again:
                        time.sleep(0.001)
                        continue
                else:
                    break

        try:
            result = json.dumps(data.to_dict()).encode('utf-8')
            self.router.send_multipart([self.req_dict[req_id], b'', result])
        except Exception as e:
            llm_logger.error(f"Send result to zmq client failed: {e}")

        if data.finished:
            with self.mutex:
                self.req_dict.pop(data.request_id, None)

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
        llm_logger.info("Closing ZMQ connection...")
        try:
            if hasattr(self, 'socket') and not self.socket.closed:
                self.socket.close()

            if self.router is not None and not self.router.closed:
                self.router.close()

            if not self.context.closed:
                self.context.term()

            self._clear_ipc(self.file_name)
            self._clear_ipc(self.router_path)
        except Exception as e:
            llm_logger.warning(f"Failed to close ZMQ connection - {e}")
            return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
