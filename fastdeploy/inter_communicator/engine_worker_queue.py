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
from multiprocessing.managers import (
    AcquirerProxy,
    BaseManager,
    ListProxy,
    Value,
    ValueProxy,
)
from queue import Queue
from typing import Any, List, Tuple

import numpy as np

from fastdeploy import envs
from fastdeploy.utils import llm_logger, to_tensor


class EngineWorkerQueue:
    """
    Cross-machine and cross-process communication queue between Engine and Worker.
    Manages shared resources using multiprocessing managers for inter-process communication.
    """

    def __init__(
        self,
        address: Tuple[str, int] = ("0.0.0.0", 5000),
        authkey: bytes = b"secret_key",
        is_server: bool = False,
        num_client: int = 1,  # tensor parallel size
        client_id: int = -1,  # tensor parallel id
        local_data_parallel_size: int = 1,  # data parallel size
        local_data_parallel_id: int = 0,  # local data parallel id
    ) -> None:
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
        self.is_server: bool = is_server
        self.num_client: int = num_client
        self.client_id: int = client_id
        self.local_data_parallel_size = local_data_parallel_size
        self.local_data_parallel_id = local_data_parallel_id

        class QueueManager(BaseManager):
            """
            Custom QueueManager for proxy object registration.
            """

            pass

        if is_server:
            # Server-side initialization for shared resources
            self.tasks_init: List[List[Any]] = [list() for _ in range(self.local_data_parallel_size)]
            self.client_read_flag_init: List[List[int]] = [
                [1] * self.num_client for _ in range(self.local_data_parallel_size)
            ]

            self.lock_init: List[threading.Lock] = [threading.Lock() for _ in range(self.local_data_parallel_size)]
            self.read_finish_flag_init: List[Value] = [Value("i", 0) for _ in range(self.local_data_parallel_size)]
            self.connected_client_counter_init: List[Value] = [
                Value("i", 0) for _ in range(self.local_data_parallel_size)
            ]
            self.finished_req_list = [list() for _ in range(self.local_data_parallel_size)]
            self.finished_add_cache_task_list = [list() for _ in range(self.local_data_parallel_size)]
            self.cache_infos_init: List[List[Any]] = [list() for _ in range(self.local_data_parallel_size)]
            self.connect_rdma_tasks_list = [list() for _ in range(self.local_data_parallel_size)]
            self.connect_rdma_tasks_response_list = [list() for _ in range(self.local_data_parallel_size)]
            self.client_read_info_flag_init: List[List[int]] = [
                [0] * self.num_client for _ in range(self.local_data_parallel_size)
            ]
            self.lock_info_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]
            # PD disaggregation
            # Locks
            self.connect_task_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]  # connect rdma task
            self.connect_task_response_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]  # connect rdma task response
            self.finish_add_cache_task_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]  # finish add cache task
            self.finish_send_cache_lock_init: List[threading.Lock] = [
                threading.Lock() for _ in range(self.local_data_parallel_size)
            ]  # finish send cache

            # sync read status for TPs
            self.client_get_connect_task_flag_init: List[List[int]] = [
                [0] * self.num_client for _ in range(self.local_data_parallel_size)
            ]
            self.client_get_connect_task_response_flag_init: List[List[int]] = [
                [0] * self.num_client for _ in range(self.local_data_parallel_size)
            ]
            self.client_get_finished_add_cache_task_flag_init: List[List[int]] = [
                [0] * self.num_client for _ in range(self.local_data_parallel_size)
            ]
            self.client_get_finish_send_cache_flag_init: List[List[int]] = [
                [0] * self.num_client for _ in range(self.local_data_parallel_size)
            ]
            self.can_put_next_connect_task_response_flag_init: List[Value] = [
                Value("i", 1) for _ in range(self.local_data_parallel_size)
            ]
            self.can_put_next_add_task_finished_flag_init: List[Value] = [
                Value("i", 1) for _ in range(self.local_data_parallel_size)
            ]
            self.can_put_next_send_cache_finished_flag_init: List[Value] = [
                Value("i", 1) for _ in range(self.local_data_parallel_size)
            ]

            # barrier
            self.get_connect_task_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.get_connect_task_response_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.finish_add_cache_task_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.begin_send_cache_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.finish_send_cache_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.get_cache_info_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]

            self.finish_request_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]
            self.worker_process_tp_barrier = [
                threading.Barrier(self.num_client) for _ in range(self.local_data_parallel_size)
            ]

            # Register shared objects with proxy types
            QueueManager.register(
                "get_tasks",
                callable=lambda idx: self.tasks_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_client_read_flag",
                callable=lambda idx: self.client_read_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_client_get_connect_task_flag",
                callable=lambda idx: self.client_get_connect_task_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_client_get_connect_task_response_flag",
                callable=lambda idx: self.client_get_connect_task_response_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_client_get_finished_add_cache_task_flag_init",
                callable=lambda idx: self.client_get_finished_add_cache_task_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_client_get_finish_send_cache_flag_init",
                callable=lambda idx: self.client_get_finish_send_cache_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_lock",
                callable=lambda idx: self.lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register(
                "get_read_finish_flag",
                callable=lambda idx: self.read_finish_flag_init[idx],
                proxytype=ValueProxy,
            )
            QueueManager.register(
                "get_can_put_next_connect_task_response_flag",
                callable=lambda idx: self.can_put_next_connect_task_response_flag_init[idx],
                proxytype=ValueProxy,
            )
            QueueManager.register(
                "get_can_put_next_add_task_finished_flag",
                callable=lambda idx: self.can_put_next_add_task_finished_flag_init[idx],
                proxytype=ValueProxy,
            )
            QueueManager.register(
                "get_can_put_next_send_cache_finished_flag",
                callable=lambda idx: self.can_put_next_send_cache_finished_flag_init[idx],
                proxytype=ValueProxy,
            )
            # PD disaggregation
            QueueManager.register(
                "get_connect_task_lock",
                callable=lambda idx: self.connect_task_lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register(
                "get_connect_task_response_lock",
                callable=lambda idx: self.connect_task_response_lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register(
                "get_finish_add_cache_task_lock",
                callable=lambda idx: self.finish_add_cache_task_lock_init[idx],
                proxytype=AcquirerProxy,
            )
            QueueManager.register(
                "get_finish_send_cache_lock",
                callable=lambda idx: self.finish_send_cache_lock_init[idx],
                proxytype=AcquirerProxy,
            )

            QueueManager.register(
                "get_connect_rdma_tasks", callable=lambda idx: self.connect_rdma_tasks_list[idx], proxytype=ListProxy
            )
            QueueManager.register(
                "get_connect_rdma_tasks_responses",
                callable=lambda idx: self.connect_rdma_tasks_response_list[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_connected_client_counter",
                callable=lambda idx: self.connected_client_counter_init[idx],
                proxytype=ValueProxy,
            )

            QueueManager.register(
                "get_finish_request_queue", callable=lambda idx: self.finished_req_list[idx], proxytype=ListProxy
            )

            QueueManager.register(
                "get_finish_add_cache_task_queue",
                callable=lambda idx: self.finished_add_cache_task_list[idx],
                proxytype=ListProxy,
            )

            QueueManager.register(
                "get_cache_infos",
                callable=lambda idx: self.cache_infos_init[idx],
                proxytype=ListProxy,
            )

            QueueManager.register(
                "get_client_read_info_flag",
                callable=lambda idx: self.client_read_info_flag_init[idx],
                proxytype=ListProxy,
            )
            QueueManager.register(
                "get_lock_info",
                callable=lambda idx: self.lock_info_init[idx],
                proxytype=AcquirerProxy,
            )

            self.disaggregate_requests = [Queue() for _ in range(self.local_data_parallel_size)]
            QueueManager.register(
                "get_disaggregate_requests",
                callable=lambda idx: self.disaggregate_requests[idx],
            )

            QueueManager.register(
                "get_finish_request_barrier",
                callable=lambda idx: self.finish_request_barrier[idx],
            )
            QueueManager.register(
                "get_connect_task_barrier",
                callable=lambda idx: self.get_connect_task_barrier[idx],
            )
            QueueManager.register(
                "get_connect_task_response_barrier",
                callable=lambda idx: self.get_connect_task_response_barrier[idx],
            )
            QueueManager.register(
                "get_begin_send_cache_barrier",
                callable=lambda idx: self.begin_send_cache_barrier[idx],
            )
            QueueManager.register(
                "get_finish_send_cache_barrier",
                callable=lambda idx: self.finish_send_cache_barrier[idx],
            )
            QueueManager.register(
                "get_cache_info_barrier",
                callable=lambda idx: self.get_cache_info_barrier[idx],
            )

            QueueManager.register(
                "get_finish_add_cache_task_barrier",
                callable=lambda idx: self.finish_add_cache_task_barrier[idx],
            )

            QueueManager.register(
                "get_worker_process_tp_barrier",
                callable=lambda idx: self.worker_process_tp_barrier[idx],
            )
            self.manager: BaseManager = QueueManager(address=self.address, authkey=self.authkey)
            self.manager.start()

            # If the port is 0, an anonymous port will be automatically assigned. The port range can be queried from system configuration,
            # e.g., by running 'cat /proc/sys/net/ipv4/ip_local_port_range'; typically in the range of 10000-60999.
            # After manager.start(), its address attribute will be updated to the actual listening address.
            # We update self.address here so that the real address can be queried later.
            self.address = self.manager.address
        else:
            # Client-side connection setup
            assert (
                self.client_id >= 0 and self.client_id < self.num_client
            ), f"self.client_id={self.client_id}, self.num_client={self.num_client}"
            QueueManager.register("get_tasks")
            QueueManager.register("get_client_read_flag")
            QueueManager.register("get_lock")
            QueueManager.register("get_read_finish_flag")
            QueueManager.register("get_connected_client_counter")
            QueueManager.register("get_finish_request_queue")
            QueueManager.register("get_finish_add_cache_task_queue")
            QueueManager.register("get_cache_infos")
            QueueManager.register("get_client_read_info_flag")
            QueueManager.register("get_lock_info")
            QueueManager.register("get_disaggregate_requests")
            QueueManager.register("get_finish_request_barrier")
            QueueManager.register("get_finish_add_cache_task_barrier")
            QueueManager.register("get_connect_task_barrier")
            QueueManager.register("get_connect_task_response_barrier")
            QueueManager.register("get_finish_send_cache_barrier")
            QueueManager.register("get_begin_send_cache_barrier")
            QueueManager.register("get_cache_info_barrier")
            QueueManager.register("get_connect_rdma_tasks")
            QueueManager.register("get_client_get_connect_task_flag")
            QueueManager.register("get_client_get_connect_task_response_flag")
            QueueManager.register("get_client_get_finished_add_cache_task_flag_init")
            QueueManager.register("get_client_get_finish_send_cache_flag_init")
            QueueManager.register("get_connect_rdma_tasks_responses")
            QueueManager.register("get_connect_task_lock")
            QueueManager.register("get_connect_task_response_lock")
            QueueManager.register("get_finish_add_cache_task_lock")
            QueueManager.register("get_finish_send_cache_lock")
            QueueManager.register("get_worker_process_tp_barrier")
            QueueManager.register("get_can_put_next_connect_task_response_flag")
            QueueManager.register("get_can_put_next_add_task_finished_flag")
            QueueManager.register("get_can_put_next_send_cache_finished_flag")
            self.manager = QueueManager(address=self.address, authkey=self.authkey)
            self._connect_with_retry()

            # Get proxy objects for shared resources
            self.tasks: ListProxy = self.manager.get_tasks(self.local_data_parallel_id)
            self.client_read_flag: ListProxy = self.manager.get_client_read_flag(self.local_data_parallel_id)
            self.lock: AcquirerProxy = self.manager.get_lock(self.local_data_parallel_id)
            self.read_finish_flag: ValueProxy = self.manager.get_read_finish_flag(self.local_data_parallel_id)
            self.connected_client_counter: ValueProxy = self.manager.get_connected_client_counter(
                self.local_data_parallel_id
            )
            self.cache_infos: ListProxy = self.manager.get_cache_infos(self.local_data_parallel_id)
            self.client_read_info_flag: ListProxy = self.manager.get_client_read_info_flag(self.local_data_parallel_id)
            self.lock_info: AcquirerProxy = self.manager.get_lock_info(self.local_data_parallel_id)

            # p/d 分离获取
            self.disaggregate_requests = self.manager.get_disaggregate_requests(self.local_data_parallel_id)
            self.finish_request_barrier = self.manager.get_finish_request_barrier(self.local_data_parallel_id)
            self.finish_add_cache_task_barrier = self.manager.get_finish_add_cache_task_barrier(
                self.local_data_parallel_id
            )
            self.connect_task_barrier = self.manager.get_connect_task_barrier(self.local_data_parallel_id)
            self.connect_task_response_barrier = self.manager.get_connect_task_response_barrier(
                self.local_data_parallel_id
            )
            self.finish_send_cache_barrier = self.manager.get_finish_send_cache_barrier(self.local_data_parallel_id)
            self.cache_info_barrier = self.manager.get_cache_info_barrier(self.local_data_parallel_id)
            self.begin_send_cache_barrier = self.manager.get_begin_send_cache_barrier(self.local_data_parallel_id)
            self.worker_process_tp_barrier = self.manager.get_worker_process_tp_barrier(self.local_data_parallel_id)
            self.finished_send_cache_list = self.manager.get_finish_request_queue(self.local_data_parallel_id)
            self.finished_add_cache_task_list = self.manager.get_finish_add_cache_task_queue(
                self.local_data_parallel_id
            )
            # p/d互联
            self.connect_rdma_tasks = self.manager.get_connect_rdma_tasks(self.local_data_parallel_id)
            self.client_get_connect_task_flag = self.manager.get_client_get_connect_task_flag(
                self.local_data_parallel_id
            )
            self.client_get_connect_task_response_flag = self.manager.get_client_get_connect_task_response_flag(
                self.local_data_parallel_id
            )
            self.client_get_finished_add_cache_task_flag = (
                self.manager.get_client_get_finished_add_cache_task_flag_init(self.local_data_parallel_id)
            )
            self.client_get_finish_send_cache_flag = self.manager.get_client_get_finish_send_cache_flag_init(
                self.local_data_parallel_id
            )

            self.connect_rdma_task_responses = self.manager.get_connect_rdma_tasks_responses(
                self.local_data_parallel_id
            )
            self.connect_task_lock = self.manager.get_connect_task_lock(self.local_data_parallel_id)
            self.connect_task_response_lock = self.manager.get_connect_task_response_lock(self.local_data_parallel_id)
            self.finish_add_cache_task_lock = self.manager.get_finish_add_cache_task_lock(self.local_data_parallel_id)
            self.finish_send_cache_lock = self.manager.get_finish_send_cache_lock(self.local_data_parallel_id)

            self.can_put_next_add_task_finished_flag = self.manager.get_can_put_next_add_task_finished_flag(
                self.local_data_parallel_id
            )
            self.can_put_next_connect_task_response_flag = self.manager.get_can_put_next_connect_task_response_flag(
                self.local_data_parallel_id
            )
            self.can_put_next_send_cache_finished_flag = self.manager.get_can_put_next_send_cache_finished_flag(
                self.local_data_parallel_id
            )

            assert self.num_client == len(self.client_read_flag)

        if is_server:
            llm_logger.info("EngineWorkerQueue server started.")
        else:
            # Update client connection counter
            self.lock.acquire()
            self.connected_client_counter.set(self.connected_client_counter.get() + 1)
            self.lock.release()
            llm_logger.info(
                f"Connected EngineWorkerQueue client_id: {self.client_id}, number "
                f"of connected clients: {self.connected_client_counter.get()}"
            )

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
        raise ConnectionError(f"TaskQueue cannot connect {self.address}")

    def put_tasks(self, tasks: List[Any]) -> None:
        """
        Add tasks to the shared queue in a thread-safe manner.
        Waits until all clients have read previous tasks before adding new ones.

        Args:
            tasks: Tasks to be added to the queue
        """
        self.lock.acquire()
        while sum(self.client_read_flag) < self.num_client:
            self.lock.release()
            time.sleep(0.001)
            self.lock.acquire()

        if envs.FD_ENABLE_MAX_PREFILL or envs.FD_ENABLE_E2W_TENSOR_CONVERT:
            # multimodal input numpy -> tensor
            to_tensor(tasks[0])

        self.tasks[:] = list()
        self.client_read_flag[:] = [0] * self.num_client
        self.tasks.append(tasks)
        self.lock.release()

    def get_tasks(self) -> Tuple[List[Any], bool]:
        """
        Retrieve tasks from the shared queue and update read status.

        Returns:
            tuple: (list of tasks, bool indicating if all clients have read)
        """
        tasks: List[Any] = list()
        self.lock.acquire()

        tasks.extend(self.tasks)
        self.client_read_flag[self.client_id] = 1
        all_client_read: bool = np.sum(self.client_read_flag) == self.num_client
        if all_client_read:
            self.tasks[:] = list()
        self.lock.release()
        return tasks, all_client_read

    def num_tasks(self) -> int:
        """
        Get current number of tasks in the queue.

        Returns:
            int: Total number of tasks
        """
        self.lock.acquire()
        total_num: int = len(self.tasks)
        self.lock.release()
        return total_num

    def put_connect_rdma_task(self, connect_rdma_task):
        self.connect_task_lock.acquire()
        while sum(self.client_get_connect_task_flag) < self.num_client:
            self.connect_task_lock.release()
            time.sleep(0.001)
            self.connect_task_lock.acquire()

        self.connect_rdma_tasks[:] = list()
        self.client_get_connect_task_flag[:] = [0] * self.num_client
        self.connect_rdma_tasks.append(connect_rdma_task)
        self.connect_task_lock.release()

    def get_connect_rdma_task(self):
        connect_rdma_task = None
        self.connect_task_lock.acquire()
        if len(self.connect_rdma_tasks) > 0:
            connect_rdma_task = self.connect_rdma_tasks[0]
        self.client_get_connect_task_flag[self.client_id] = 1
        all_client_read: bool = np.sum(self.client_get_connect_task_flag) == self.num_client
        if all_client_read:
            self.connect_rdma_tasks[:] = list()
        self.connect_task_lock.release()
        return connect_rdma_task, all_client_read

    def put_connect_rdma_task_response(self, connect_rdma_task_response):
        self.connect_task_response_lock.acquire()
        while not self.can_put_next_connect_task_response_flag.get():
            self.connect_task_response_lock.release()
            time.sleep(0.001)
            self.connect_task_response_lock.acquire()
        self.connect_rdma_task_responses.append(connect_rdma_task_response)
        self.client_get_connect_task_response_flag[self.client_id] = 1
        all_client_put: bool = np.sum(self.client_get_connect_task_response_flag) == self.num_client
        if all_client_put:
            self.can_put_next_connect_task_response_flag.set(0)
        self.connect_task_response_lock.release()
        return all_client_put

    def get_connect_rdma_task_response(self):
        task_response = None
        self.connect_task_response_lock.acquire()
        if len(self.connect_rdma_task_responses) == 0:
            self.connect_task_response_lock.release()
            return task_response
        while sum(self.client_get_connect_task_response_flag) < self.num_client:
            self.connect_task_response_lock.release()
            time.sleep(0.001)
            self.connect_task_response_lock.acquire()
        if len(self.connect_rdma_task_responses) > 0:
            task_response = self.connect_rdma_task_responses[0]
        for tmp_task_response in self.connect_rdma_task_responses:
            task_response["success"] = task_response["success"] and tmp_task_response["success"]
        self.connect_rdma_task_responses[:] = list()
        self.client_get_connect_task_response_flag[:] = [0] * self.num_client
        self.can_put_next_connect_task_response_flag.set(1)
        self.connect_task_response_lock.release()
        return task_response

    def put_cache_info(self, cache_info) -> None:
        """
        Args:
            tasks: Tasks to be added to the queue
        """
        self.lock_info.acquire()
        while sum(self.client_read_info_flag) < self.num_client:
            self.lock_info.release()
            time.sleep(0.001)
            self.lock_info.acquire()

        self.cache_infos[:] = list()
        self.client_read_info_flag[:] = [0] * self.num_client

        self.cache_infos.extend(cache_info)
        llm_logger.debug(
            f"put cache_infos to engine worker queue: {self.cache_infos}, "
            f"local_data_parallel_id:{self.local_data_parallel_id}"
        )
        self.lock_info.release()

    def get_cache_info(self) -> List[Any]:
        """
        Retrieve tasks from the shared queue and update read status.

        Returns:
            tuple: (list of tasks, bool indicating if all clients have read)
        """
        cache_infos: List[Any] = list()
        self.lock_info.acquire()
        if self.client_read_info_flag[self.client_id] == 1:
            self.lock_info.release()
            return cache_infos
        cache_infos.extend(self.cache_infos)
        self.client_read_info_flag[self.client_id] = 1
        all_client_read: bool = np.sum(self.client_read_info_flag) == self.num_client
        if all_client_read:
            self.cache_infos[:] = list()
        self.lock_info.release()
        if len(cache_infos) != 0:
            llm_logger.debug(
                f"get cache infos from engine worker queue: {cache_infos}, "
                f"local_data_parallel_id:{self.local_data_parallel_id}"
            )
        return cache_infos

    def num_cache_infos(self) -> int:
        """
        Get current number of tasks in the queue.

        Returns:
            int: Total number of tasks
        """
        self.lock_info.acquire()
        total_num: int = len(self.cache_infos)
        self.lock_info.release()
        return total_num

    def put_finished_req(self, send_cache_result) -> None:
        """
        Put finished request ID into the queue.

        Args:
            req_ids: Request ID to be added to the queue
        """
        self.finish_send_cache_lock.acquire()
        while not self.can_put_next_send_cache_finished_flag.get():
            self.finish_send_cache_lock.release()
            time.sleep(0.001)
            self.finish_send_cache_lock.acquire()
        self.finished_send_cache_list.append(send_cache_result[0])
        self.client_get_finish_send_cache_flag[self.client_id] = 1
        all_client_put: bool = np.sum(self.client_get_finish_send_cache_flag) == self.num_client
        if all_client_put:
            self.can_put_next_send_cache_finished_flag.set(0)
        self.finish_send_cache_lock.release()
        return all_client_put

    def get_finished_req(self) -> str:
        """
        Get finished request ID from the queue.

        Returns:
            str: Finished request ID
        """
        response = []
        self.finish_send_cache_lock.acquire()
        if len(self.finished_send_cache_list) == 0:
            self.finish_send_cache_lock.release()
            return response
        while sum(self.client_get_finish_send_cache_flag) < self.num_client:
            self.finish_send_cache_lock.release()
            time.sleep(0.001)
            self.finish_send_cache_lock.acquire()
        if len(self.finished_send_cache_list) > 0:
            response = self.finished_send_cache_list[0]
        for tmp_response in self.finished_send_cache_list:
            if "error" in tmp_response[1]:
                response[1] = tmp_response[1]
        if response:
            response = [response]
        self.finished_send_cache_list[:] = list()
        self.client_get_finish_send_cache_flag[:] = [0] * self.num_client
        self.can_put_next_send_cache_finished_flag.set(1)
        self.finish_send_cache_lock.release()
        return response

    def put_finished_add_cache_task_req(self, req_ids) -> None:
        """
        Put finished request ID into the queue.

        Args:
            req_ids: Request ID to be added to the queue
        """
        self.finish_add_cache_task_lock.acquire()
        while not self.can_put_next_add_task_finished_flag.get():
            self.finish_add_cache_task_lock.release()
            time.sleep(0.001)
            self.finish_add_cache_task_lock.acquire()
        self.finished_add_cache_task_list.append(req_ids)
        self.client_get_finished_add_cache_task_flag[self.client_id] = 1
        all_client_put: bool = np.sum(self.client_get_finished_add_cache_task_flag) == self.num_client
        if all_client_put:
            self.can_put_next_add_task_finished_flag.set(0)
        self.finish_add_cache_task_lock.release()
        return all_client_put

    def get_finished_add_cache_task_req(self) -> str:
        """
        Get finished request ID from the queue.

        Returns:
            str: Finished request ID
        """
        response = []
        self.finish_add_cache_task_lock.acquire()
        if len(self.finished_add_cache_task_list) == 0:
            self.finish_add_cache_task_lock.release()
            return response
        while sum(self.client_get_finished_add_cache_task_flag) < self.num_client:
            self.finish_add_cache_task_lock.release()
            time.sleep(0.001)
            self.finish_add_cache_task_lock.acquire()
        if len(self.finished_add_cache_task_list) > 0:
            response = self.finished_add_cache_task_list[0]
        for tmp_response in self.finished_add_cache_task_list:
            assert tmp_response == response
        self.finished_add_cache_task_list[:] = list()
        self.client_get_finished_add_cache_task_flag[:] = [0] * self.num_client
        self.can_put_next_add_task_finished_flag.set(1)
        self.finish_add_cache_task_lock.release()
        return response

    def disaggregate_queue_empty(self):
        """
        Check if the disaggregated task queue is empty.
        """
        return self.disaggregate_requests.qsize() == 0

    def put_disaggregated_tasks(self, item):
        """
        put disaggregated tasks to the queue
        """
        llm_logger.debug("put item to queue")
        self.disaggregate_requests.put(item)
        llm_logger.debug("put item to queue success")

    def get_disaggregated_tasks(self):
        """
        get disaggregated tasks from the queue
        """
        llm_logger.debug("get tasks from queue")
        if self.disaggregate_requests.qsize() == 0:
            return None
        item = []
        while not self.disaggregate_requests.empty():
            item.append(self.disaggregate_requests.get())
        llm_logger.debug("get tasks from queue success")
        return item

    def clear_data(self):
        self.lock.acquire()
        self.tasks[:] = list()
        self.client_read_flag[:] = [1] * self.num_client
        self.lock.release()
        llm_logger.info("clear data for engine worker queue")

    def cleanup(self):
        """
        Exit the worker queue gracefully.
        """
        if self.manager is not None and self.is_server:
            self.manager.shutdown()
