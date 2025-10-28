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
import traceback
from multiprocessing.managers import (
    AcquirerProxy,
    BaseManager,
    ListProxy,
    Value,
    ValueProxy,
)
from typing import Any, List, Tuple

from fastdeploy.utils import get_logger

logger = get_logger("cache_queue_manager", "cache_queue_manager.log")


class EngineCacheQueue:
    """
    Multiprocessing manager for cache queue between Engine and Worker.
    Manages shared resources using multiprocessing managers for inter-process communication.
    """

    def __init__(
        self,
        address: Tuple[str, int] = ("127.0.0.1", 56666),
        authkey: bytes = b"cache_queue_service",
        is_server: bool = False,
        num_client: int = 1,  # tensor parallel size
        client_id: int = -1,  # tensor parallel id
        local_data_parallel_size: int = 1,  # data parallel size
        local_data_parallel_id: int = 0,  # local data parallel id
    ) -> None:
        """
        Initialize the cache communication queue.

        Args:
            address: Network address (IP, port) for the queue server
            authkey: Authentication key for secure connection
            is_server: Whether this instance acts as a server
            num_client: Total number of expected clients
            client_id: Unique identifier for client instances
            local_data_parallel_size: data parallel size
            local_data_parallel_id: local data parallel id
        """
        self.address: Tuple[str, int] = address
        self.authkey: bytes = authkey
        self.is_server: bool = is_server
        self.num_client: int = num_client
        self.client_id: int = client_id
        self.local_data_parallel_size = local_data_parallel_size
        self.local_data_parallel_id = local_data_parallel_id

        class QueueManager(BaseManager):
            """
            Custom QueueManager for proxy object registration
            """

            pass

        if is_server:
            # Server-side initialization for shared resources
            self.transfer_task_queue_init: List[List[Any]] = [list() for _ in range(self.local_data_parallel_size)]
            self.tansfer_done_queue_init: List[List[Any]] = [list() for _ in range(self.local_data_parallel_size)]
            self.cache_sync_value_init: List[Value] = [Value("i", 0) for _ in range(self.local_data_parallel_size)]
            self.transfer_task_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]
            self.transfer_task_done_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]

            # Initialize barriers
            self.barrier1_init = [threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)]
            self.barrier2_init = [threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)]
            self.barrier3_init = [threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)]
            self.swap_to_cpu_barrier1_init = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.swap_to_cpu_barrier2_init = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.swap_to_gpu_barrier1_init = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.swap_to_gpu_barrier2_init = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]

            # Register shared objects with proxy types
            QueueManager.register(
                "get_transfer_task_queue",
                callable=lambda idx: self.transfer_task_queue_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_tansfer_done_queue",
                callable=lambda idx: self.tansfer_done_queue_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_cache_sync_value",
                callable=lambda idx: self.cache_sync_value_init[idx],
                proxytype=ValueProxy,
            )
            QueueManager.register(
                "get_transfer_task_lock",
                callable=lambda idx: self.transfer_task_lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register(
                "get_transfer_task_done_lock",
                callable=lambda idx: self.transfer_task_done_lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register("get_barrier1", callable=lambda idx: self.barrier1_init[idx])
            QueueManager.register("get_barrier2", callable=lambda idx: self.barrier2_init[idx])
            QueueManager.register("get_barrier3", callable=lambda idx: self.barrier3_init[idx])
            QueueManager.register(
                "get_swap_to_cpu_barrier1",
                callable=lambda idx: self.swap_to_cpu_barrier1_init[idx],
            )
            QueueManager.register(
                "get_swap_to_cpu_barrier2",
                callable=lambda idx: self.swap_to_cpu_barrier2_init[idx],
            )
            QueueManager.register(
                "get_swap_to_gpu_barrier1",
                callable=lambda idx: self.swap_to_gpu_barrier1_init[idx],
            )
            QueueManager.register(
                "get_swap_to_gpu_barrier2",
                callable=lambda idx: self.swap_to_gpu_barrier2_init[idx],
            )

            self.manager: BaseManager = QueueManager(address=self.address, authkey=self.authkey)
            self.manager.start()

            # If the port is 0, an anonymous port will be automatically assigned. The port range can be queried from system configuration,
            # e.g., by running 'cat /proc/sys/net/ipv4/ip_local_port_range'; typically in the range of 10000-60999.
            # After manager.start(), its address attribute will be updated to the actual listening address.
            # We update self.address here so that the real address can be queried later.
            self.address = self.manager.address
            logger.info(f"EngineCacheQueue server started at {self.address}")
        else:
            # Client-side connection setup
            assert (
                0 <= self.client_id < self.num_client
            ), f"client_id must be between 0 and {self.num_client-1}, got {self.client_id}"
            QueueManager.register("get_transfer_task_queue")
            QueueManager.register("get_tansfer_done_queue")
            QueueManager.register("get_cache_sync_value")
            QueueManager.register("get_transfer_task_lock")
            QueueManager.register("get_transfer_task_done_lock")
            QueueManager.register("get_barrier1")
            QueueManager.register("get_barrier2")
            QueueManager.register("get_barrier3")
            QueueManager.register("get_swap_to_cpu_barrier1")
            QueueManager.register("get_swap_to_cpu_barrier2")
            QueueManager.register("get_swap_to_gpu_barrier1")
            QueueManager.register("get_swap_to_gpu_barrier2")

            self.manager = QueueManager(address=self.address, authkey=self.authkey)
            self._connect_with_retry()

        # Get proxy objects for shared resources
        self.transfer_task_queue = self.manager.get_transfer_task_queue(self.local_data_parallel_id)
        self.tansfer_done_queue = self.manager.get_tansfer_done_queue(self.local_data_parallel_id)
        self.task_sync_value = self.manager.get_cache_sync_value(self.local_data_parallel_id)
        self.task_lock = self.manager.get_transfer_task_lock(self.local_data_parallel_id)
        self.task_done_lock = self.manager.get_transfer_task_done_lock(self.local_data_parallel_id)

        # Get barrier proxies
        self.barrier1 = self.manager.get_barrier1(self.local_data_parallel_id)
        self.barrier2 = self.manager.get_barrier2(self.local_data_parallel_id)
        self.barrier3 = self.manager.get_barrier3(self.local_data_parallel_id)
        self.swap_to_cpu_barrier1 = self.manager.get_swap_to_cpu_barrier1(self.local_data_parallel_id)
        self.swap_to_cpu_barrier2 = self.manager.get_swap_to_cpu_barrier2(self.local_data_parallel_id)
        self.swap_to_gpu_barrier1 = self.manager.get_swap_to_gpu_barrier1(self.local_data_parallel_id)
        self.swap_to_gpu_barrier2 = self.manager.get_swap_to_gpu_barrier2(self.local_data_parallel_id)
        self.total_num: int = (1 << self.num_client) - 1

        if not is_server:
            # Setup position and total_num for sync operations
            self.position: int = 1 << self.client_id
            logger.info(f"Connected EngineCacheQueue client_id: {self.client_id}")

    def get_server_port(self) -> int:
        """
        Returns the actual port that the server instance is listening on.
        Calling this method only makes sense on instances where is_server=True.
        """
        if not self.is_server:
            raise RuntimeError("Only the server instance can provide the port.")
        return self.address[1]

    def _connect_with_retry(self, max_retries: int = 5, interval: int = 3) -> None:
        """
        Connect to the server with retry mechanism.

        Args:
            max_retries: Maximum connection attempts
            interval: Retry interval in seconds

        Raises:
            ConnectionError: If all connection attempts fail
        """
        for _ in range(max_retries):
            try:
                self.manager.connect()
                return
            except ConnectionRefusedError:
                time.sleep(interval)
        raise ConnectionError(f"EngineCacheQueue cannot connect to {self.address}")

    def put_transfer_task(self, item):
        """
        put swap task
        """
        self.task_lock.acquire()
        if 0 < self.task_sync_value.get() < self.total_num:
            self.task_lock.release()
            while 0 < self.task_sync_value.get() < self.total_num:
                time.sleep(0.001)
            self.task_lock.acquire()
        self.task_sync_value.set(0)
        self.transfer_task_queue.append(item)
        logger.info(f"put_transfer_task: put swap task {item[-1]} to queue successful")
        self.task_lock.release()

    def get_transfer_task(self):
        """
        get swap task
        """
        data = None
        read_finish = False
        self.task_lock.acquire()
        if self.task_sync_value.get() & self.position == 0 and len(self.transfer_task_queue) > 0:
            data = self.transfer_task_queue[0]
            logger.debug(f"get_transfer_task: Get {data} by {self.client_id} from queue successful")
            set_value = self.task_sync_value.get() | self.position
            logger.info(f"get_transfer_task: rank: {self.client_id} set_value: {set_value}")
            if set_value >= self.total_num:
                self.transfer_task_queue.pop(0)
                set_value = 0
                read_finish = True
            self.task_sync_value.set(set_value)
        self.task_lock.release()
        return data, read_finish

    def put_transfer_done_signal(self, item):
        """
        put swap result
        """
        self.task_done_lock.acquire()
        self.tansfer_done_queue.append(item)
        self.task_done_lock.release()
        logger.info(f"put_transfer_done_signal: put swap task {item[-1]} finished signal to queue successful")

    def get_transfer_done_signal(self):
        """
        get swap result
        """
        data = None
        self.task_done_lock.acquire()
        if len(self.tansfer_done_queue) > 0:
            data = self.tansfer_done_queue.pop(0)
            logger.info(f"get_transfer_done_signal: Get swap task {data[-1]} finished signal from queue successful")
        self.task_done_lock.release()
        return data

    def empty(self):
        """
        check if queue is empty
        """
        try:
            return len(self.transfer_task_queue) == 0
        except Exception as e:
            logger.error(f"empty function meets error: {e}, {str(traceback.format_exc())}")
            raise e
