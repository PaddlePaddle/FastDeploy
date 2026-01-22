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

"""
Refactored FMQ implementation removing zmq.asyncio.
Uses synchronous ZeroMQ sockets + background threads.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import shared_memory
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable, Dict, Optional

import zmq

from fastdeploy.utils import fmq_logger

# ==========================
# Config & Enum Definitions
# ==========================


class EndpointType(Enum):
    QUEUE = "queue"
    TOPIC = "topic"


class Role(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SocketOptions:
    sndhwm: int = 0
    rcvhwm: int = 0
    linger: int = -1
    sndbuf: int = 32 * 1024 * 1024
    rcvbuf: int = 32 * 1024 * 1024
    immediate: int = 0

    def apply(self, socket: zmq.Socket, is_producer: bool):
        socket.setsockopt(zmq.LINGER, self.linger)
        socket.setsockopt(zmq.IMMEDIATE, self.immediate)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        if is_producer:
            socket.setsockopt(zmq.SNDHWM, self.sndhwm)
            socket.setsockopt(zmq.SNDBUF, self.sndbuf)
        else:
            socket.setsockopt(zmq.RCVHWM, self.rcvhwm)
            socket.setsockopt(zmq.RCVBUF, self.rcvbuf)


@dataclass
class Endpoint:
    protocol: str
    address: str
    io_threads: int = 1
    copy: bool = False


@dataclass
class Config:
    ipc_root: str = "/dev/shm"
    io_threads: int = 1
    copy: bool = False
    endpoints: Dict[str, Endpoint] = field(default_factory=dict)
    socket_config: SocketOptions = field(default_factory=SocketOptions)


# ==========================
# Endpoint Manager
# ==========================


class EndpointManager:
    config: Config = Config()

    @classmethod
    def load_config(cls, _ignored_file_path: str = None):
        # For demonstration only; env reference removed
        cfg_str = None
        if cfg_str:
            try:
                custom_cfg = json.loads(cfg_str)
                for key, value in vars(custom_cfg).items():
                    if value is not None:
                        setattr(cls.config, key, value)
            except Exception:
                pass

    @classmethod
    def get_endpoint(cls, name: str) -> Endpoint:
        if name in cls.config.endpoints:
            return cls.config.endpoints[name]
        address = f"{cls.config.ipc_root}/fmq_{name}.ipc"
        return Endpoint(protocol="ipc", address=address)


# ==========================
# Shared Memory Descriptor
# ==========================


@dataclass
class Descriptor:
    shm_name: str
    size: int

    @staticmethod
    def create(data_bytes: bytes) -> "Descriptor":
        name = f"fmq_shm_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(create=True, size=len(data_bytes), name=name)
        shm.buf[: len(data_bytes)] = data_bytes
        shm.close()
        return Descriptor(shm_name=name, size=len(data_bytes))

    def read_and_unlink(self) -> bytes:
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            data = bytes(shm.buf[: self.size])
            shm.close()
            shm.unlink()
            return data
        except FileNotFoundError:
            return b""


# ==========================
# Message Wrapper
# ==========================


@dataclass
class Message:
    payload: Any
    msg_id: int = None
    timestamp: float = field(default_factory=time.time)
    descriptor: Optional[Descriptor] = None

    def serialize(self) -> bytes:
        return ForkingPickler.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "Message":
        return ForkingPickler.loads(data)


# ==========================
# Base Component
# ==========================


class BaseComponent:
    def __init__(self, context: zmq.Context, endpoint: Endpoint):
        self.context = context
        self.endpoint = endpoint
        self.socket = None
        self.debug = True
        self._lock = threading.Lock()

    def close(self):
        if self.socket:
            self.socket.close()


# ==========================
# FIFO Queue
# ==========================


class Queue(BaseComponent):
    """
    Queue is NOT thread-safe.

    A Queue instance (and its underlying ZMQ socket)
    MUST be used only in the thread where it was created.

    This class performs defensive runtime checks to detect
    cross-thread misuse early.
    """

    def __init__(self, context, name: str, role: str = "producer"):
        self._owner_thread_id = threading.get_ident()

        endpoint = EndpointManager.get_endpoint(name)
        super().__init__(context, endpoint)

        self.name = name
        self.role = Role(role)
        self.copy = endpoint.copy
        self.socket_conf = EndpointManager.config.socket_config

        self.full_ep = f"{endpoint.protocol}://{endpoint.address}"

        self.socket = self.context.socket(zmq.PUSH if self.role == Role.PRODUCER else zmq.PULL)
        self.socket_conf.apply(self.socket, self.role == Role.PRODUCER)

        if self.role == Role.PRODUCER:
            self.socket.connect(self.full_ep)
        else:
            self.socket.bind(self.full_ep)

        last_endpoint = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        fmq_logger.info(
            f"init Queue endpoint:{last_endpoint} role={self.role.name} " f"thread={self._owner_thread_id}"
        )

    # ---------- defensive check ----------

    def _check_thread(self):
        """
        Check if the current thread is the owner of the queue.
        ØMQ has both thread safe socket type and not thread safe socket types.
        Applications MUST NOT use a not thread safe socket from multiple threads except
        after migrating a socket from one thread to another with a "full fence" memory barrier.
        """
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise RuntimeError(
                f"Queue socket used from multiple threads. " f"owner={self._owner_thread_id}, current={current}"
            )

    # ---------- public API ----------

    def put(self, data: Any, shm_threshold: int = 1024 * 1024):
        self._check_thread()

        if self.role != Role.PRODUCER:
            raise PermissionError("Only producers can send messages.")

        if self.socket is None:
            fmq_logger.warning(f"Re-create broken socket:{self.full_ep}")
            self.socket = self.context.socket(zmq.PUSH)
            self.socket_conf.apply(self.socket, True)
            self.socket.connect(self.full_ep)

        desc = None
        payload = data
        if isinstance(data, bytes) and len(data) >= shm_threshold:
            desc = Descriptor.create(data)
            payload = None

        msg = Message(payload=payload, descriptor=desc)

        try:
            raw = msg.serialize()
        except Exception as e:
            fmq_logger.error(f"Failed to serialize message: {e}")
            raise

        while True:
            try:
                self.socket.send(raw, copy=self.copy, flags=zmq.NOBLOCK)
                break
            except zmq.Again:
                fmq_logger.warning("Queue is full, waiting and retrying...")
                time.sleep(0.01)
                continue
            except zmq.ZMQError as e:
                fmq_logger.error(f"Failed to send message: {e}")
                raise
            except Exception as e:
                fmq_logger.error(f"Unknown error occurred: {e}")
                raise

    def get(self) -> Optional[Any]:
        self._check_thread()

        if self.role != Role.CONSUMER:
            raise PermissionError("Only consumers can get messages.")

        try:
            raw = self.socket.recv(copy=self.copy)
            msg = Message.deserialize(raw)
            if msg.descriptor:
                msg.payload = msg.descriptor.read_and_unlink()
            return msg.payload
        except zmq.ZMQError as e:
            fmq_logger.error(f"ZMQ error occurred: {str(e)}")
            return None
        finally:
            if self.debug:
                fmq_logger.info("get message")


# ==========================
# Pub/Sub Topic
# ==========================


class Topic(BaseComponent):
    """
    Topic using XPUB / XSUB to avoid slow-joiner problem.

    PUB socket:
        - XPUB
        - created and used in the caller thread of pub()

    SUB socket:
        - XSUB
        - created and used in an internal dedicated thread
    """

    def __init__(self, context, name: str):
        endpoint = EndpointManager.get_endpoint(name)
        super().__init__(context, endpoint)

        self.name = name

        # PUB side
        self._pub_socket = None
        self._pub_owner_thread = None
        self._has_subscriber = False

        # SUB side
        self._sub_thread = None
        self._sub_running = threading.Event()

    # ---------- PUB ----------

    def _init_pub(self):
        ep = f"{self.endpoint.protocol}://{self.endpoint.address}"

        self._pub_socket = self.context.socket(zmq.XPUB)
        self._pub_socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        self._pub_socket.bind(ep)

        fmq_logger.info(f"Topic XPUB initialized endpoint={ep} " f"thread={self._pub_owner_thread}")

        # 等待至少一个订阅者
        while True:
            evt = self._pub_socket.recv()
            if evt and evt[0] == 1:  # subscribe
                self._has_subscriber = True
                fmq_logger.info("XPUB got subscriber")
                break

    def pub(self, data: Any):
        current_thread = threading.get_ident()

        if self._pub_socket is None:
            self._pub_owner_thread = current_thread
            self._init_pub()
        elif current_thread != self._pub_owner_thread:
            raise RuntimeError(
                "PUB socket used from multiple threads " f"(owner={self._pub_owner_thread}, current={current_thread})"
            )

        if not self._has_subscriber:
            return

        msg = Message(payload=data)

        while True:
            try:
                self._pub_socket.send(msg.serialize(), zmq.NOBLOCK)
                break
            except zmq.Again:
                fmq_logger.warning("XPUB send queue full, waiting...")
                time.sleep(0.01)
            except Exception:
                raise

    # ---------- SUB ----------

    def sub(self, callback: Callable[[Message], Any]):
        if self._sub_thread is not None:
            raise RuntimeError("SUB already started")

        ep = f"{self.endpoint.protocol}://{self.endpoint.address}"
        self._sub_running.set()

        def loop():
            sub_socket = self.context.socket(zmq.XSUB)
            sub_socket.connect(ep)

            sub_socket.send(b"\x01")  # subscribe all

            poller = zmq.Poller()
            poller.register(sub_socket, zmq.POLLIN)

            fmq_logger.info(f"Topic XSUB started endpoint={ep} " f"thread={threading.get_ident()}")

            try:
                while self._sub_running.is_set():
                    events = dict(poller.poll(timeout=500))
                    if sub_socket in events:
                        raw = sub_socket.recv()
                        msg = Message.deserialize(raw)
                        callback(msg)
            finally:
                poller.unregister(sub_socket)
                sub_socket.close()
                fmq_logger.info("Topic XSUB socket closed")

        self._sub_thread = threading.Thread(
            target=loop,
            name=f"TopicSub-{self.name}",
            daemon=True,
        )
        self._sub_thread.start()

    # ---------- lifecycle ----------

    def stop_sub(self):
        if self._sub_thread:
            self._sub_running.clear()
            self._sub_thread.join(timeout=1)
            self._sub_thread = None


# ==========================
# FMQ Main Interface
# ==========================


class FMQ:
    def __init__(self, config_path: str = "fmq_config.json"):
        EndpointManager.load_config(config_path)

        config = EndpointManager.config
        io_threads = config.io_threads or 1

        self._context = zmq.Context(io_threads=io_threads)

        fmq_logger.info(f"FMQ initialized. " f"context={id(self._context)} io_threads={io_threads}")

    def queue(self, name: str, role: str = "producer") -> Queue:
        return Queue(self._context, name, role)

    def topic(self, name: str) -> Topic:
        return Topic(self._context, name)

    def destroy(self):
        if self._context is not None:
            fmq_logger.info(f"FMQ destroying. context={id(self._context)}")
            self._context.term()
            self._context = None
