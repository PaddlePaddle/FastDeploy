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

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import shared_memory
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable, Dict, Optional

import zmq
import zmq.asyncio

from fastdeploy import envs
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
    immediate: int = 1

    def apply(self, socket: zmq.Socket, is_producer: bool):
        # Apply socket-level configurations
        socket.setsockopt(zmq.LINGER, self.linger)
        socket.setsockopt(zmq.IMMEDIATE, self.immediate)

        if is_producer:
            socket.setsockopt(zmq.SNDHWM, self.sndhwm)
            socket.setsockopt(zmq.SNDBUF, self.sndbuf)
        else:
            socket.setsockopt(zmq.RCVHWM, self.rcvhwm)
            socket.setsockopt(zmq.RCVBUF, self.rcvbuf)


@dataclass
class Endpoint:
    # Represents a single endpoint with protocol, address, io_threads, and copy behavior
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
        cfg_str = envs.FMQ_CONFIG_JSON
        if cfg_str:
            try:
                custom_cfg = json.loads(cfg_str)
                for key, value in vars(custom_cfg).items():
                    if value is not None:
                        setattr(cls.config, key, value)
            except Exception as e:
                fmq_logger.error(f"Failed to load FMQ config: {e}")
        fmq_logger.info(f"Loaded FMQ config: {cls.config}")

    @classmethod
    def get_endpoint(cls, name: str) -> Endpoint:
        # Retrieve endpoint object
        if name in cls.config.endpoints:
            return cls.config.endpoints[name]

        # Fallback: auto-generate endpoint
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
        # Create shared memory buffer and store payload
        name = f"fmq_shm_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(create=True, size=len(data_bytes), name=name)
        shm.buf[: len(data_bytes)] = data_bytes
        shm.close()
        return Descriptor(shm_name=name, size=len(data_bytes))

    def read_and_unlink(self) -> bytes:
        # Read and cleanup shared memory
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
        # Serialize message
        return ForkingPickler.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "Message":
        # Deserialize message
        return ForkingPickler.loads(data)


# ==========================
# Base Component
# ==========================


class BaseComponent:
    def __init__(self, context: zmq.asyncio.Context, endpoint: Endpoint):
        self.context = context
        self.endpoint = endpoint
        self.socket = None
        self.lock = asyncio.Lock()

    async def close(self):
        # Close socket
        if self.socket:
            self.socket.close()


# ==========================
# FIFO Queue
# ==========================


class Queue(BaseComponent):
    def __init__(self, context, name: str, role: str = "producer"):
        endpoint = EndpointManager.get_endpoint(name)
        super().__init__(context, endpoint)

        self.name = name
        self.role = Role(role)
        self.copy = endpoint.copy
        self.socket_conf = EndpointManager.config.socket_config
        self._msg_id = 0

        full_ep = f"{endpoint.protocol}://{endpoint.address}"

        self.socket = self.context.socket(zmq.PUSH if self.role == Role.PRODUCER else zmq.PULL)
        self.socket_conf.apply(self.socket, self.role == Role.PRODUCER)

        if self.role == Role.PRODUCER:
            self.socket.connect(full_ep)
        else:
            self.socket.bind(full_ep)

        fmq_logger.info(f"Queue {name}({role}) initialized on {full_ep}")

    async def put(self, data: Any, shm_threshold: int = 1024 * 1024):
        """
        Send data to the queue.

        Args:
            data: The data to send. Can be any serializable object or bytes.
            shm_threshold: Size threshold in bytes. If the data is of type bytes and its size is
                greater than or equal to this threshold, shared memory will be used to send the message.
                Default is 1MB (1024 * 1024 bytes).

        Raises:
            PermissionError: If called by a non-producer role.
        """
        if self.role != Role.PRODUCER:
            raise PermissionError("Only producers can send messages.")

        desc = None
        payload = data

        if isinstance(data, bytes) and len(data) >= shm_threshold:
            desc = Descriptor.create(data)
            payload = None

        msg = Message(msg_id=self._msg_id, payload=payload, descriptor=desc)
        raw = msg.serialize()

        async with self.lock:
            await self.socket.send(raw, copy=self.copy)
            self._msg_id += 1

    async def get(self, timeout: int = None) -> Optional[Message]:
        # Receive data from queue
        if self.role != Role.CONSUMER:
            raise PermissionError("Only consumers can get messages.")

        try:
            if timeout:
                raw = await asyncio.wait_for(self.socket.recv(), timeout / 1000)
            else:
                raw = await self.socket.recv(copy=self.copy)
        except asyncio.TimeoutError:
            fmq_logger.error(f"Timeout receiving message on {self.name}")
            return None

        msg = Message.deserialize(raw)
        if msg.descriptor:
            msg.payload = msg.descriptor.read_and_unlink()

        self._msg_id += 1
        return msg


# ==========================
# Pub/Sub Topic
# ==========================


class Topic(BaseComponent):
    def __init__(self, context, name: str):
        endpoint = EndpointManager.get_endpoint(name)
        super().__init__(context, endpoint)
        self.name = name
        self._pub_socket = None
        self._sub_socket = None
        self._task = None

    async def pub(self, data: Any):
        # Publish a message
        if not self._pub_socket:
            ep = f"{self.endpoint.protocol}://{self.endpoint.address}"
            self._pub_socket = self.context.socket(zmq.PUB)
            self._pub_socket.bind(ep)
            await asyncio.sleep(0.05)

        msg = Message(payload=data)
        async with self.lock:
            await self._pub_socket.send(msg.serialize())

    async def sub(self, callback: Callable[[Message], Any]):
        # Subscribe and handle messages
        if not self._sub_socket:
            ep = f"{self.endpoint.protocol}://{self.endpoint.address}"
            self._sub_socket = self.context.socket(zmq.SUB)
            self._sub_socket.connect(ep)
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        async def loop():
            while True:
                raw = await self._sub_socket.recv()
                msg = Message.deserialize(raw)
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    await result

        self._task = asyncio.create_task(loop())


# ==========================
# FMQ Main Interface
# ==========================


class FMQ:
    _instance = None
    _context = None

    def __new__(cls, config_path="fmq_config.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            EndpointManager.load_config()

            # Determine IO threads based on global defaults
            io_threads = 1
            if EndpointManager.config.endpoints:
                # Use max io_threads among all endpoints
                io_threads = max(ep.io_threads for ep in EndpointManager.config.endpoints.values())

            cls._context = zmq.asyncio.Context(io_threads=io_threads)
        return cls._instance

    def queue(self, name: str, role="producer") -> Queue:
        return Queue(self._context, name, role)

    def topic(self, name: str) -> Topic:
        return Topic(self._context, name)

    async def destroy(self):
        # Destroy ZeroMQ context
        self._context.term()
