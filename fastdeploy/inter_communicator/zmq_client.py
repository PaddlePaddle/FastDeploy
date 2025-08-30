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
import traceback

import msgpack
import zmq

from fastdeploy import envs
from fastdeploy.utils import zmq_client_logger


class ZmqClient:
    """
    ZmqClient is a class that provides a client-side interface for sending and receiving messages using ZeroMQ.
    """

    def __init__(self, name, mode):
        self.context = zmq.Context(4)
        self.socket = self.context.socket(mode)
        self.file_name = f"/dev/shm/{name}.socket"
        self.router_path = f"/dev/shm/router_{name}.ipc"

        self.ZMQ_SNDHWM = int(envs.FD_ZMQ_SNDHWM)
        self.aggregate_send = envs.FD_USE_AGGREGATE_SEND

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
        self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.router.setsockopt(zmq.SNDTIMEO, -1)
        self.router.bind(f"ipc://{self.router_path}")
        zmq_client_logger.info(f"router path: {self.router_path}")

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

    def send_multipart(self, req_id, data):
        """
        Send a multipart message to the router socket.
        """
        if self.router is None:
            raise RuntimeError("Router socket not created. Call create_router() first.")

        while self.running:
            with self.mutex:
                if req_id not in self.req_dict:
                    try:
                        client, _, request_id = self.router.recv_multipart(flags=zmq.NOBLOCK)
                        req_id_str = request_id.decode("utf-8")
                        self.req_dict[req_id_str] = client
                    except zmq.Again:
                        time.sleep(0.001)
                        continue
                else:
                    break
        if self.req_dict[req_id] == -1:
            if data[-1].finished:
                with self.mutex:
                    self.req_dict.pop(req_id, None)
            return
        try:
            start_send = time.time()
            if self.aggregate_send:
                result = self.pack_aggregated_data(data)
            else:
                result = msgpack.packb([response.to_dict() for response in data])
            self.router.send_multipart([self.req_dict[req_id], b"", result])
            zmq_client_logger.info(f"send_multipart result: {req_id} len {len(data)} elapse: {time.time()-start_send}")
        except zmq.ZMQError as e:
            zmq_client_logger.error(f"[{req_id}] zmq error: {e}")
            self.req_dict[req_id] = -1
        except Exception as e:
            zmq_client_logger.error(f"Send result to zmq client failed: {e}, {str(traceback.format_exc())}")

        if data[-1].finished:
            with self.mutex:
                self.req_dict.pop(req_id, None)
            zmq_client_logger.info(f"send_multipart finished, req_id: {req_id}")

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
            zmq_client_logger.warning(f"{e}, {str(traceback.format_exc())}")
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
            zmq_client_logger.warning(f"{e}, {str(traceback.format_exc())}")
            return str(e), None

    def _clear_ipc(self, name):
        """
        Remove the IPC file with the given name.
        """
        if os.path.exists(name):
            try:
                os.remove(name)
            except OSError as e:
                zmq_client_logger.warning(f"Failed to remove IPC file {name} - {e}")

    def close(self):
        """
        Close the socket and context, and remove the IPC files.
        """
        if not self.running:
            return

        self.running = False
        zmq_client_logger.info("Closing ZMQ connection...")
        try:
            if hasattr(self, "socket") and not self.socket.closed:
                self.socket.close()

            if self.router is not None and not self.router.closed:
                self.router.close()

            if not self.context.closed:
                self.context.term()

            self._clear_ipc(self.file_name)
            self._clear_ipc(self.router_path)
        except Exception as e:
            zmq_client_logger.warning(f"Failed to close ZMQ connection - {e}, {str(traceback.format_exc())}")
            return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
