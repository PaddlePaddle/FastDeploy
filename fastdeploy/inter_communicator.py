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

Inter-process communication utilities for FastDeploy server.
This module provides:
- ZeroMQ-based client/server communication
- Shared memory utilities for numpy arrays
- Process-safe task queues
"""


import os
import threading
import socket
import json
import time
import numpy as np
from multiprocessing.managers import (AcquirerProxy, BaseManager, ListProxy,
                                      Value, ValueProxy)
from queue import Queue
import zmq
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict, Tuple, List, Any

from fastdeploy.utils import llm_logger


def shared_memory_exists(name: str) -> bool:
    """Check if a shared memory block with the given name exists.

    Args:
        name (str): The unique identifier of the shared memory block.

    Returns:
        bool: True if the shared memory exists, False otherwise.
    """
    try:
        shm = SharedMemory(name=name, create=False)
        shm.close()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False



class ZmqClient:
    """ZeroMQ client for inter-process communication.
    
    Provides both client and server capabilities using ZeroMQ sockets.
    Supports JSON and Python object serialization.
    
    Attributes:
        context (zmq.Context): ZeroMQ context
        socket (zmq.Socket): Primary communication socket
        file_name (str): IPC socket file path
        router_path (str): Router IPC file path
        mutex (threading.Lock): Thread synchronization lock
        req_dict (dict): Request tracking dictionary
        router (zmq.Socket): Router socket for multi-client communication
        poller (zmq.Poller): Socket poller for event monitoring
    """
    def __init__(self, name, mode):
        self.context = zmq.Context()
        self.socket = self.context.socket(mode)
        self.file_name = f"/dev/shm/{name}.socket"
        self.router_path = f"/dev/shm/router_{name}.ipc"

        self.mutex = threading.Lock()
        self.req_dict = dict()
        self.router = None
        self.poller = None

    def connect(self):
        """Connect to the ZeroMQ server.
        
        Uses the IPC file path specified during initialization.
        """
        self.socket.connect(f"ipc://{self.file_name}")

    def start_server(self):
        """Start a ZeroMQ server.
        
        Binds to the IPC file path specified during initialization.
        Also initializes a poller for the socket.
        """
        self.socket.bind(f"ipc://{self.file_name}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def create_router(self):
        """Create and bind a ROUTER socket.
        
        The router socket enables handling multiple client connections.
        Uses the router path specified during initialization.
        """
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(f"ipc://{self.router_path}")

    def send_json(self, data):
        """Send JSON data through the socket.
        
        Args:
            data: JSON-serializable object to send
        """
        self.socket.send_json(data)

    def recv_json(self):
        """Receive JSON data from the socket.
        
        Returns:
            object: Deserialized JSON data
        """
        return self.socket.recv_json()

    def send_pyobj(self, data):
        """Send a Python object through the socket.
        
        Args:
            data: Pickle-serializable Python object
        """
        self.socket.send_pyobj(data)

    def recv_pyobj(self):
        """Receive a Python object from the socket.
        
        Returns:
            object: Deserialized Python object
        """
        return self.socket.recv_pyobj()

    def send_multipart(self, req_id, data):
        """Send a multipart message through the router socket.
        
        Args:
            req_id (str): Request identifier
            data: Data to send (will be JSON-serialized)
            
        Raises:
            RuntimeError: If router socket is not initialized
        """
        if self.router is None:
            raise RuntimeError("Router socket not created. Call create_router() first.")

        while True:
            with self.mutex:
                if req_id not in self.req_dict:
                    try:
                        client, _, request_id = self.router.recv_multipart(flags=zmq.NOBLOCK)
                        req_id_str = request_id.decode('utf-8')
                        self.req_dict[req_id_str] = client
                    except zmq.Again:
                        continue
                else:
                    break
        
        try:
            result = json.dumps(data.to_dict()).encode('utf-8')
            self.router.send_multipart([self.req_dict[req_id], b'', result], zmq.DONTWAIT)
        except Exception as e:
            llm_logger.error(f"Send result to zmq client failed: {e}")
        
        if data["finished"]:
            with self.mutex:
                self.req_dict.pop(data["request_id"], None)
    
    def send_multipart2(self, get_results_handler):
        """Batch send multipart messages through the router socket.
        
        Args:
            get_results_handler (callable): Function that takes request IDs and
                returns a dict mapping request IDs to response data
                
        Raises:
            RuntimeError: If router socket is not initialized
        """
        if self.router is None:
            raise RuntimeError("Router socket not created. Call create_router() first.")
        
        while True:
            with self.mutex:
                try:
                    flags = 0 if len(self.req_dict) == 0 else zmq.NOBLOCK
                    client, _, request_id = self.router.recv_multipart(flags=flags)
                    req_id_str = request_id.decode('utf-8')
                    self.req_dict[req_id_str] = client
                except zmq.Again:
                    time.sleep(0.01)
                    break
        
        req_dict_copy = dict()
        with self.mutex:
            req_dict_copy = self.req_dict.copy()

        finished_req = []
        req_ids = list(req_dict_copy.keys())
        results = get_results_handler(req_ids)

        for req_id, contents in results.items():
            client = req_dict_copy[req_id]
            for data in contents:
                if data["finished"]:
                    finished_req.append(data["request_id"])

                result = json.dumps(data).encode('utf-8')
                try:
                    self.router.send_multipart([client, b'', result], zmq.DONTWAIT)
                except Exception as e:
                    llm_logger.error(f"Send result to zmq client2 failed: {e}")
        
        if len(finished_req) > 0:
            with self.mutex:
                for req_id in finished_req:
                    self.req_dict.pop(req_id, None)

    def receive_json_once(self, block=False):
        """Receive a single JSON message from the socket.
        
        Args:
            block (bool): Whether to block waiting for message
            
        Returns:
            tuple: (error_message, data) where data is None if no message received
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
        """Receive a single Python object from the socket.
        
        Args:
            block (bool): Whether to block waiting for message
            
        Returns:
            tuple: (error_message, data) where data is None if no message received
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
        """Clean up IPC file.
        
        Args:
            name (str): Path to IPC file to remove
        """
        if os.path.exists(name):
            try:
                os.remove(name)
            except OSError as e:
                llm_logger.warning(f"Failed to remove IPC file {name} - {e}")

    def close(self):
        """Clean up all resources.
        
        Closes sockets, terminates context, and removes IPC files.
        Safe to call multiple times.
        """
        if hasattr(self, 'socket') and not self.socket.closed:
            self.socket.close()

        if self.router is not None and not self.router.closed:
            self.router.close()

        if not self.context.closed:
            self.context.term()

        self._clear_ipc(self.file_name)
        self._clear_ipc(self.router_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class IPCSignal:
    """Shared memory wrapper for numpy array IPC.
    
    Provides process-safe shared memory access to numpy arrays.
    
    Attributes:
        shm (SharedMemory): Underlying shared memory block
        value (np.ndarray): Numpy array view of shared memory
    """

    def __init__(self,
                 name: str,
                 array: np.ndarray,
                 dtype: np.dtype,
                 suffix: int = None,
                 create: bool = True) -> None:
        """Initialize or connect to a shared memory block.

        Args:
            name: Unique identifier for the shared memory block.
            array: Numpy array template defining shape and data type.
            dtype: Data type of the array (must match array.dtype).
            suffix: Suffix number that will be appended to the name.
            create: If True, creates new memory block; otherwise connects to existing.

        Raises:
            AssertionError: If create=True but memory already exists, or dtype mismatch.
        """
        assert isinstance(array, np.ndarray), "Input must be a numpy array"
        assert dtype == array.dtype, "Specified dtype must match array dtype"

        # Set a suffix for name to avoid name conflict while there are multiple engine launched
        if suffix is not None:
            name = name + f".{suffix}"

        if create:
            assert not shared_memory_exists(
                name), f"ShareMemory: {name} already exists"
            self.shm = SharedMemory(create=True, size=array.nbytes, name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)
            self.value[:] = array  # Initialize with input array data
        else:
            self.shm = SharedMemory(name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)

    def clear(self) -> None:
        """Release system resources and unlink the shared memory block."""
        self.shm.close()
        self.shm.unlink()


class EngineWorkerQueue:
    """Process-safe task queue for engine-worker communication.
    
    Implements a multi-producer, multi-consumer queue using:
    - Multiprocessing managers for shared state
    - Thread locks for synchronization
    - Network sockets for cross-machine operation
    
    Attributes:
        address (tuple): (host, port) for network binding
        authkey (bytes): Authentication key
        num_client (int): Expected number of clients
        client_id (int): Unique client identifier (-1 for server)
        manager (BaseManager): Proxy manager instance
        tasks (ListProxy): Shared task list
        client_read_flag (ListProxy): Client read status flags  
        lock (AcquirerProxy): Synchronization lock
        read_finish_flag (ValueProxy): Completion flag
        connected_client_counter (ValueProxy): Active client count
    """

    def __init__(self,
                 address: Tuple[str, int] = ('0.0.0.0', 5000),
                 authkey: bytes = b'secret_key',
                 is_server: bool = False,
                 num_client: int = 1,
                 client_id: int = -1) -> None:
        """
        Initialize the communication queue.

        Args:
            address: Network address (IP, port) for the queue server
            authkey: Authentication key for secure connection
            is_server: Whether this instance acts as a server
            num_client: Total number of expected clients
            client_id: Unique identifier for client instances
        """
        self.address: Tuple[str, int] = address
        self.authkey: bytes = authkey
        self.num_client: int = num_client
        self.client_id: int = client_id

        # Custom QueueManager for proxy object registration
        class QueueManager(BaseManager):
            pass

        if is_server:
            # Server-side initialization for shared resources
            self.tasks_init: List[Any] = list()
            self.client_read_flag_init: List[int] = [1] * self.num_client
            self.lock_init: threading.Lock = threading.Lock()
            self.read_finish_flag_init: Value = Value("i", 0)
            self.connected_client_counter_init: Value = Value("i", 0)

            # Register shared objects with proxy types
            QueueManager.register("get_tasks",
                                  callable=lambda: self.tasks_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_client_read_flag",
                                  callable=lambda: self.client_read_flag_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_lock",
                                  callable=lambda: self.lock_init,
                                  proxytype=AcquirerProxy)
            QueueManager.register("get_read_finish_flag",
                                  callable=lambda: self.read_finish_flag_init,
                                  proxytype=ValueProxy)
            QueueManager.register(
                "get_connected_client_counter",
                callable=lambda: self.connected_client_counter_init,
                proxytype=ValueProxy)

            self.manager: BaseManager = QueueManager(address=self.address,
                                                     authkey=self.authkey)
            self.manager.start()
        else:
            # Client-side connection setup
            assert self.client_id >= 0 and self.client_id < self.num_client, (
                f"self.client_id={self.client_id}, self.num_client={self.num_client}"
            )
            QueueManager.register("get_tasks")
            QueueManager.register("get_client_read_flag")
            QueueManager.register("get_lock")
            QueueManager.register("get_read_finish_flag")
            QueueManager.register("get_connected_client_counter")
            self.manager = QueueManager(address=self.address,
                                        authkey=self.authkey)
            self._connect_with_retry()

        # Get proxy objects for shared resources
        self.tasks: ListProxy = self.manager.get_tasks()
        self.client_read_flag: ListProxy = self.manager.get_client_read_flag()
        self.lock: AcquirerProxy = self.manager.get_lock()
        self.read_finish_flag: ValueProxy = self.manager.get_read_finish_flag()
        self.connected_client_counter: ValueProxy = self.manager.get_connected_client_counter(
        )
        assert self.num_client == len(self.client_read_flag)

        if is_server:
            llm_logger.info(f"EngineWorkerQueue server started.")
        else:
            # Update client connection counter
            self.lock.acquire()
            self.connected_client_counter.set(
                self.connected_client_counter.get() + 1)
            self.lock.release()
            llm_logger.info((
                f"Connected EngineWorkerQueue client_id: {self.client_id}, number "
                f"of connected clients: {self.connected_client_counter.get()}"
            ))

    def _connect_with_retry(self,
                            max_retries: int = 5,
                            interval: int = 3) -> None:
        """Connect to server with retry logic.
        
        Args:
            max_retries (int): Maximum connection attempts. Default: 5
            interval (int): Seconds between retries. Default: 3
            
        Raises:
            ConnectionError: If all retries fail
        """
        for _ in range(max_retries):
            try:
                self.manager.connect()
                return
            except ConnectionRefusedError:
                time.sleep(interval)
        raise ConnectionError(f"TaskQueue cannot connect {self.address}")

    def put_tasks(self, tasks: List[Any]) -> None:
        """Add tasks to the shared queue.
        
        Waits until all clients have read previous tasks before adding new ones.
        Thread-safe operation.
        
        Args:
            tasks (list): Tasks to add to queue
        """
        self.lock.acquire()
        while sum(self.client_read_flag) < self.num_client:
            self.lock.release()
            time.sleep(0.001)
            self.lock.acquire()

        self.tasks[:] = list()
        self.client_read_flag[:] = [0] * self.num_client
        self.tasks.append(tasks)
        self.lock.release()

    def get_tasks(self) -> Tuple[List[Any], bool]:
        """Retrieve tasks from shared queue.
        
        Updates read status for this client.
        Thread-safe operation.
        
        Returns:
            tuple: (tasks, all_read) where:
                tasks (list): Retrieved tasks
                all_read (bool): True if all clients have read
        """
        tasks: List[Any] = list()
        self.lock.acquire()
        tasks.extend(self.tasks)
        self.client_read_flag[self.client_id] = 1
        all_client_read: bool = np.sum(
            self.client_read_flag) == self.num_client
        if all_client_read:
            self.tasks[:] = list()
        self.lock.release()
        return tasks, all_client_read

    def num_tasks(self) -> int:
        """Get current task count in queue.
        
        Thread-safe operation.
        
        Returns:
            int: Number of tasks currently in queue
        """
        self.lock.acquire()
        total_num: int = len(self.tasks)
        self.lock.release()
        return total_num
