"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import random
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

import crcmod
from redis import ConnectionPool

from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler import utils
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.scheduler.storage import AdaptedRedis
from fastdeploy.scheduler.workers import Task, Workers
from fastdeploy.utils import envs, scheduler_logger


class GlobalScheduler:
    """
    A distributed task scheduler that manages request/response queues using Redis.

    This class provides functionality for:
    - Enqueuing and dequeuing requests
    - Load balancing across multiple scheduler instances
    - Handling request/response lifecycle
    - Maintaining worker health checks
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        topic: str,
        ttl: int,
        min_load_score: float,
        load_shards_num: int,
        enable_chunked_prefill: bool,
        max_num_partial_prefills: int,
        max_long_partial_prefills: int,
        long_prefill_token_threshold: int,
    ):
        """
        Initialize the GlobalScheduler with Redis connection and scheduling parameters.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Optional password for Redis authentication
            topic: Base topic name for queue namespacing
            ttl: Time-to-live in seconds for Redis keys
            min_load_score: Minimum load score for task assignment
            load_shards_num: Number of shards for load balancing table
            enable_chunked_prefill: Whether to enable chunked prefill processing
            max_num_partial_prefills: Maximum number of partial prefills allowed
            max_long_partial_prefills: Maximum number of long partial prefills allowed
            long_prefill_token_threshold: Token count threshold for long prefills

        Initializes:
            - Redis connection pool and client
            - Worker threads for request/response handling
            - Load balancing and request stealing mechanisms
            - Response tracking structures
        """

        self.topic = topic
        self.ttl = ttl
        self.min_load_score = min_load_score
        self.load_shards_num = load_shards_num

        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold

        self.blpop_request_timeout = 2
        self.blpop_response_timeout = 10

        self.crc16_mutex = threading.Lock()
        self.crc16 = crcmod.predefined.Crc("ccitt-false")
        self.load_slot_for_getting_request = 0
        self.load_offset = 0  # const
        self.load_count = 50  # const
        self.load_lookup_num = 5  # const
        self.keep_alive_duration = 30  # const

        connection_pool = ConnectionPool(host=host, port=port, db=db, password=password, max_connections=10)
        self.client = AdaptedRedis(connection_pool=connection_pool)

        self.name, self.shard = self._generate_scheduler_name_and_shard()

        self.keep_alive_workers = threading.Thread(target=self._keep_alive, daemon=True)
        self.keep_alive_workers.start()

        self.put_requests_workers = Workers("put_requests_workers", self._put_requests_worker, 20)
        self.put_requests_workers.start(1)

        self.put_results_workers = Workers("put_results_workers", self._put_results_worker, 300)
        self.put_results_workers.start(1)

        self.mutex = threading.Lock()
        self.local_response_not_empty = threading.Condition(self.mutex)
        self.local_responses: Dict[str, List[ScheduledResponse]] = dict()
        self.stolen_requests: Dict[str, ScheduledRequest] = dict()

        self.get_response_workers = threading.Thread(target=self._get_results_worker, daemon=True)
        self.get_response_workers.start()

        scheduler_logger.info(f"Scheduler: name={self.name} redis_version={self.client.version}")

    def _get_hash_slot(self, data: str) -> int:
        """
        Calculate the hash slot for a given string using CRC16 algorithm.

        This method is thread-safe and used for consistent hashing in distributed scheduling.
        It implements the same CRC16 algorithm (CCITT-FALSE variant) used by Redis Cluster.

        Args:
            data: Input string to be hashed (typically a scheduler or request identifier)

        Returns:
            int: A 16-bit hash value (0-65535) representing the calculated slot

        Implementation Details:
        1. Encodes input string as UTF-8 bytes
        2. Uses thread-safe CRC16 calculation with mutex protection
        3. Resets CRC state after each calculation
        4. Returns raw CRC value without modulo operation

        Note:
        - The result is typically used with modulo operation for sharding (e.g. % num_shards)
        - Matches Redis Cluster's slot distribution algorithm for compatibility
        """
        data = data.encode("utf-8")
        with self.crc16_mutex:
            self.crc16.update(data)
            crc_value = self.crc16.crcValue
            self.crc16.crcValue = self.crc16.initCrc
        return crc_value

    def _instance_name(self, scheduler_name: str) -> str:
        """
        Generate the Redis key name for a scheduler instance.

        Args:
            scheduler_name: Name of the scheduler instance

        Returns:
            Formatted Redis key name
        """
        return f"{self.topic}.ins.{scheduler_name}"

    def _generate_scheduler_name_and_shard(self) -> Tuple[str, int]:
        """
        Generate a unique scheduler name and calculate its shard assignment.

        This method:
        1. Creates a unique identifier using hostname/IP and timestamp
        2. Registers the name in Redis with TTL
        3. Calculates the shard assignment using consistent hashing
        4. Handles naming conflicts by appending incrementing suffixes

        Returns:
            Tuple[str, int]:
                - str: Unique scheduler name
                - int: Assigned shard number (0 to load_shards_num-1)

        Implementation Details:
        - Uses hostname/IP as base identifier, falls back to UUID if unavailable
        - Implements conflict resolution with incrementing suffixes
        - Registers name in Redis with keep-alive duration
        - Calculates shard using CRC16 hash of the name

        Error Handling:
        - Logs IP resolution failures
        - Handles Redis registration conflicts gracefully
        - Ensures unique name generation even in edge cases
        """
        try:
            _, name = utils.get_hostname_ip()
        except Exception as e:
            scheduler_logger.warning(f"Scheduler encountered an error while resolving the IP address. {e}")
            name = str(uuid.uuid4())

        size = len(name)
        count = 1
        while True:
            if self.client.set(
                self._instance_name(name),
                "",
                ex=self.keep_alive_duration,
                nx=True,
            ):
                break
            name = f"{name[:size]}:{count}"
            count += 1

        shard = self._get_hash_slot(name) % self.load_shards_num
        self.client.set(
            self._instance_name(name),
            self._load_table_name(shard=shard),
            ex=self.keep_alive_duration,
        )
        return name, shard

    def _keep_alive(self):
        """
        Background thread that periodically updates the scheduler's TTL in Redis.

        Runs in a loop with interval of keep_alive_duration/2 to maintain instance registration.
        """
        while True:
            try:
                self.client.set(
                    self._instance_name(self.name),
                    self._load_table_name(),
                    ex=self.keep_alive_duration,
                )
                time.sleep(self.keep_alive_duration / 2)
            except Exception as e:
                scheduler_logger.error(f"Scheduler keep alive failed: {e}, {str(traceback.format_exc())}")
                time.sleep(min(3, self.keep_alive_duration / 4))

    def _scheduler_name_from_request_queue(self, request_queue: str) -> str:
        """
        Extract scheduler name from a request queue name.

        Args:
            request_queue: Full request queue name

        Returns:
            The scheduler name portion of the queue name
        """
        prefix_len = len(f"{self.topic}.req.")
        return request_queue[prefix_len:]

    def _request_queue_name(self, scheduler_name: Optional[str] = None) -> str:
        """
        Generate the Redis request queue name for a scheduler.

        Args:
            scheduler_name: Optional specific scheduler name, defaults to current instance

        Returns:
            Formatted request queue name
        """
        if scheduler_name is None:
            return f"{self.topic}.req.{self.name}"
        return f"{self.topic}.req.{scheduler_name}"

    def _response_queue_name(self, scheduler_name: Optional[str] = None) -> str:
        """
        Generate the Redis response queue name for a scheduler.

        Args:
            scheduler_name: Optional specific scheduler name, defaults to current instance

        Returns:
            Formatted response queue name
        """
        if scheduler_name is None:
            return f"{self.topic}.resp.{self.name}"
        return f"{self.topic}.resp.{scheduler_name}"

    def _load_table_name(self, shard: Optional[int] = None, slot: Optional[int] = None) -> str:
        """
        Get the Redis sorted set name used for load balancing.

        Returns:
            The load score key name
        """
        if shard is None and slot is not None:
            shard = slot % self.load_shards_num
        if shard is None:
            shard = self.shard
        return f"{self.topic}.load.{shard}"

    @staticmethod
    def calc_required_blocks(token_num, block_size):
        """
        Calculate the number of blocks needed for a given number of tokens.

        Args:
            token_num: Number of tokens
            block_size: Size of each block

        Returns:
            Number of blocks required (rounded up)
        """
        return (token_num + block_size - 1) // block_size

    @staticmethod
    def _mark_request(request: ScheduledRequest):
        """
        Mark a stolen request with the original queue name.

        Args:
            request: The request to mark
        """
        request.request_id = f"mark<{request.request_queue_name}>{request.request_id}"

    @staticmethod
    def _unmark_response(response: ScheduledResponse, request_queue_name: str):
        """
        Remove marking from a response that came from a stolen request.

        Args:
            response: The response to unmark
            request_queue_name: Original request queue name
        """
        mark = f"mark<{request_queue_name}>"
        if not response.request_id.startswith(mark):
            return
        response.request_id = response.request_id[len(mark) :]

    def _put_requests_worker(self, tasks: List[Task]) -> List[Task]:
        """
        Worker method that adds requests to the shared Redis cache.

        Args:
            tasks: List of tasks containing requests to enqueue

        Returns:
            List of processed tasks (some may be marked as duplicates)
        """
        duplicate = False
        requests: List[ScheduledRequest] = []
        with self.mutex:
            for task in tasks:
                request = ScheduledRequest(
                    task.raw,
                    self._request_queue_name(),
                    self._response_queue_name(),
                )
                task.raw = None

                if request.request_id in self.local_responses:
                    task.reason = "duplicate request_id"
                    duplicate = True
                    continue
                requests.append(request)
                self.local_responses[request.request_id] = []

        if len(requests) > 0:
            serialized_requests = [request.serialize() for request in requests]
            self.client.rpush(self._request_queue_name(), *serialized_requests, ttl=self.ttl)
            self.client.zincrby(
                self._load_table_name(),
                len(serialized_requests),
                self.name,
                rem_amount=0,
                ttl=self.ttl,
            )
            scheduler_logger.info(f"Scheduler has enqueued some requests: {requests}")

        if duplicate:
            scheduler_logger.warning(
                "Scheduler has received some duplicated requests: "
                f"{[task for task in tasks if task.reason is not None]}"
            )
        return tasks

    def put_requests(self, requests: List[Request]) -> List[Tuple[str, Optional[str]]]:
        """
        Public method to add new requests to the scheduler.

        Args:
            requests: List of Request objects to schedule

        Returns:
            List of tuples containing (request_id, error_reason) for each request
        """
        tasks: List[Task] = []
        for request in requests:
            task = Task(request.request_id, request)
            tasks.append(task)

        self.put_requests_workers.add_tasks(tasks)
        results = self.put_requests_workers.get_results(10, 0.001)
        return [(result.id, result.reason) for result in results]

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch=1,
    ) -> List[Request]:
        """
        Get requests from the shared cache based on available resources.

        Args:
            available_blocks: Number of available processing blocks
            block_size: Size of each processing block
            reserved_output_blocks: Blocks reserved for output
            max_num_batched_tokens: Maximum tokens that can be batched
            batch: Preferred batch size

        Returns:
            List of Request objects ready for processing
        """

        if available_blocks <= reserved_output_blocks or batch < 1:
            scheduler_logger.debug(
                f"Scheduler's resource are insufficient: available_blocks={available_blocks} "
                f"reserved_output_blocks={reserved_output_blocks} batch={batch} "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )
            return []

        mini_batch = (batch + 1) // 2
        batches = []
        for _ in range(2):
            if batch >= mini_batch:
                batches.append(mini_batch)
                batch -= mini_batch
                continue

            if batch > 0:
                batches.append(batch)
                batch = 0

        local_request_queue_name = self._request_queue_name()
        serialized_requests: List[Tuple[str, bytes]] = []
        for bs in batches:
            elements = self.client.lpop(local_request_queue_name, bs, ttl=self.ttl)
            if elements is None:
                break
            self.client.zincrby(
                self._load_table_name(),
                -len(elements),
                self.name,
                rem_amount=0,
                ttl=self.ttl,
            )
            serialized_requests += [(local_request_queue_name, element) for element in elements]

        extend_scheduler_names = []
        extend_scheduler_load_table_name = ""
        if len(serialized_requests) == 0 and len(batches) > 0:
            for _ in range(min(self.load_lookup_num, self.load_shards_num)):
                extend_scheduler_load_table_name = self._load_table_name(slot=self.load_slot_for_getting_request)
                serialized_members = self.client.zrangebyscore(
                    extend_scheduler_load_table_name,
                    self.min_load_score,
                    float("+inf"),
                    start=self.load_offset,
                    num=self.load_count,
                )
                self.load_slot_for_getting_request += 1
                if len(serialized_members) > 0:
                    break

            members = [member.decode("utf-8") for member in serialized_members]
            if len(members) > 0:
                extend_scheduler_names = random.sample(members, k=min(10, len(members)))
                extend_scheduler_names = [name for name in extend_scheduler_names if name != self.name]

        # find lucky one
        if len(extend_scheduler_names) > 0:
            lucky = random.choice(extend_scheduler_names)
            lucky_request_queue_name = self._request_queue_name(lucky)

            elements = self.client.lpop(lucky_request_queue_name, batches[0])
            if elements is not None and len(elements) > 0:
                self.client.zincrby(
                    extend_scheduler_load_table_name,
                    -len(elements),
                    lucky,
                    rem_amount=0,
                    ttl=self.ttl,
                )
                serialized_requests += [(lucky_request_queue_name, element) for element in elements]
                scheduler_logger.info(
                    f"Scheduler {self.name} has stolen some requests from another lucky one. "
                    f"(name={lucky} num={len(serialized_requests)})"
                )
            else:
                exist_num = self.client.exists(self._instance_name(lucky))
                if exist_num == 0:
                    if self.client.zrem(extend_scheduler_load_table_name, lucky):
                        scheduler_logger.info(f"Scheduler {lucky} has been removed")

        # blocked read
        if len(serialized_requests) == 0:
            request_queue_names = [local_request_queue_name]
            request_queue_names += [self._request_queue_name(name) for name in extend_scheduler_names]

            element = self.client.blpop(request_queue_names, self.blpop_request_timeout)
            if element is None:
                return []
            request_queue_name = element[0].decode("utf-8")
            scheduler_name = self._scheduler_name_from_request_queue(request_queue_name)
            load_table_name = (
                extend_scheduler_load_table_name if scheduler_name != self.name else self._load_table_name()
            )
            self.client.zincrby(load_table_name, -1, scheduler_name, rem_amount=0, ttl=self.ttl)
            serialized_requests.append((request_queue_name, element[1]))
            if scheduler_name != self.name:
                scheduler_logger.info(
                    f"Scheduler {self.name} has stolen a request from another scheduler. (name={scheduler_name})"
                )

        long_partial_requests = 0
        short_partial_requests = 0
        required_total_blocks = 0
        current_prefill_tokens = 0
        remaining_request: List[Tuple[str, bytes]] = []
        scheduled_requests: List[ScheduledRequest] = []
        for request_queue_name, serialized_request in serialized_requests:
            if len(remaining_request) > 0:
                remaining_request.append((request_queue_name, serialized_request))
                continue

            request: ScheduledRequest = ScheduledRequest.unserialize(serialized_request)
            required_input_blocks = self.calc_required_blocks(request.prompt_tokens_ids_len, block_size)

            current_prefill_tokens += request.prompt_tokens_ids_len
            required_total_blocks += required_input_blocks + reserved_output_blocks

            if required_total_blocks > available_blocks:
                remaining_request.append((request_queue_name, serialized_request))
                continue

            if not envs.FD_ENABLE_MAX_PREFILL:
                if self.enable_chunked_prefill:
                    if request.prompt_tokens_ids_len > self.long_prefill_token_threshold:
                        long_partial_requests += 1
                        if long_partial_requests > self.max_long_partial_prefills:
                            remaining_request.append((request_queue_name, serialized_request))
                            continue
                    else:
                        short_partial_requests += 1

                    if short_partial_requests + long_partial_requests > self.max_num_partial_prefills:
                        remaining_request.append((request_queue_name, serialized_request))
                        continue
                else:
                    if current_prefill_tokens > max_num_batched_tokens:
                        remaining_request.append((request_queue_name, serialized_request))
                        continue

            scheduled_requests.append(request)

        if len(scheduled_requests) > 0:
            with self.mutex:
                for request in scheduled_requests:
                    if request.request_queue_name == local_request_queue_name:
                        continue

                    # self._mark_request(request)
                    if request.request_id not in self.stolen_requests:
                        self.stolen_requests[request.request_id] = request
                        continue

                    scheduler_logger.error(f"Scheduler has received a duplicate request from others: {request}")

        requests: List[Request] = [request.raw for request in scheduled_requests]
        if len(remaining_request) > 0:
            group: Dict[str, List] = dict()
            for request_queue_name, serialized_request in remaining_request:
                if request_queue_name not in group:
                    group[request_queue_name] = []
                group[request_queue_name].append(serialized_request)

            for request_queue_name, serialized_requests in group.items():
                self.client.lpush(request_queue_name, *serialized_requests)
                scheduler_name = self._scheduler_name_from_request_queue(request_queue_name)
                load_table_name = (
                    extend_scheduler_load_table_name if scheduler_name != self.name else self._load_table_name()
                )
                self.client.zincrby(
                    load_table_name,
                    len(serialized_requests),
                    scheduler_name,
                    ttl=self.ttl,
                )

            scheduler_logger.info(f"Scheduler has put remaining request into the queue: {len(remaining_request)}")
            if len(requests) == 0:
                scheduler_logger.debug(
                    f"Scheduler has put all just-pulled request into the queue: {len(remaining_request)}"
                )

        if len(requests) > 0:
            scheduler_logger.info(f"Scheduler has pulled some request: {[request.request_id for request in requests]}")
        return requests

    def _put_results_worker(self, tasks: List[Task]):
        """
        Worker method that adds task results to the appropriate queues.

        Args:
            tasks: List of completed tasks with results
        """
        # count = 0  # for test

        with self.mutex:
            local_request_ids = set(self.local_responses.keys())

            stolen_request_id_request_queue = dict()
            stolen_request_id_response_queue = dict()
            for request_id, request in self.stolen_requests.items():
                stolen_request_id_request_queue[request_id] = request.request_queue_name
                stolen_request_id_response_queue[request_id] = request.response_queue_name

        finished_request_ids: List[str] = list()
        local_responses: Dict[str, List[ScheduledResponse]] = dict()
        stolen_responses: Dict[str, List[bytes]] = dict()

        for task in tasks:
            response = ScheduledResponse(task.raw)
            if response.finished:
                finished_request_ids.append(response.request_id)

            if response.request_id in local_request_ids:
                if response.request_id not in local_responses:
                    local_responses[response.request_id] = []
                local_responses[response.request_id].append(response)
                continue

            if response.request_id in stolen_request_id_request_queue:
                response_queue_name = stolen_request_id_response_queue[response.request_id]
                # request_queue_name = stolen_request_id_request_queue[response.request_id]
                # self._unmark_response(response, request_queue_name)

                if response_queue_name not in stolen_responses:
                    stolen_responses[response_queue_name] = []
                stolen_responses[response_queue_name].append(response.serialize())
                continue

            scheduler_logger.error(f"Scheduler has received a non-existent response from engine: {[response]}")

        with self.mutex:
            for request_id, responses in local_responses.items():
                self.local_responses[request_id] += responses
                # count += len(responses)  # for test

            for request_id in finished_request_ids:
                if request_id in self.stolen_requests:
                    del self.stolen_requests[request_id]

            if len(local_responses) > 0:
                self.local_response_not_empty.notify_all()

        if len(finished_request_ids) > 0:
            scheduler_logger.info(f"Scheduler has received some finished responses: {finished_request_ids}")

        for response_queue_name, responses in stolen_responses.items():
            self.client.rpush(response_queue_name, *responses, ttl=self.ttl)
            # count += len(responses)  # for test
        # return [Task("", count)]  # for test

    def put_results(self, results: List[RequestOutput]):
        """
        Public method to add processing results back to the scheduler.

        Args:
            results: List of RequestOutput objects to return
        """
        tasks: List[Task] = [Task(result.request_id, result) for result in results]
        self.put_results_workers.add_tasks(tasks)

        # ---- for test ----
        # task_results = self.put_results_workers.get_results(10, 0.001)
        # amount = 0
        # for task_result in task_results:
        #     amount += task_result.raw
        # return amount
        # ---- for test ----

    def _get_results_worker(self):
        """
        Background worker that continuously fetches results from Redis.

        Handles both bulk and blocking operations for efficiency.
        Runs in an infinite loop until scheduler shutdown.
        """
        while True:
            try:
                serialized_responses = self.client.lpop(self._response_queue_name(), 300, ttl=self.ttl)

                if serialized_responses is None or len(serialized_responses) == 0:
                    element = self.client.blpop(
                        [self._response_queue_name()],
                        self.blpop_response_timeout,
                    )
                    if element is None or len(element) == 0:
                        continue
                    serialized_responses = [element[1]]

                responses: Dict[str, List[ScheduledResponse]] = dict()
                for serialized_response in serialized_responses:
                    response = ScheduledResponse.unserialize(serialized_response)
                    if response.request_id not in responses:
                        responses[response.request_id] = []
                    responses[response.request_id].append(response)

                with self.mutex:
                    for request_id, contents in responses.items():
                        if request_id not in self.local_responses:
                            scheduler_logger.error(
                                "Scheduler has received some non-existent response from the queue. "
                                f"response:{contents} queue:{self._response_queue_name()}"
                            )
                            continue
                        self.local_responses[request_id] += contents
                    self.local_response_not_empty.notify_all()
            except Exception as e:
                scheduler_logger.error(
                    f"Scheduler get_results_worker exception: {e} " f"traceback: {traceback.format_exc()}"
                )

    def get_results(self) -> Dict[str, List[RequestOutput]]:
        """
        Retrieve all available results from the distributed scheduler.

        This method:
        - Waits for new responses using a condition variable (timeout=0.001s)
        - Returns all currently available responses
        - Automatically removes completed requests from local tracking
        - Logs finished requests

        Behavior Details:
        1. For first call with less than 64 pending responses, returns empty dict
        2. Subsequent calls return all available responses
        3. Uses thread-safe operations with condition variables
        4. Automatically cleans up completed request tracking

        Returns:
            Dict[str, List[RequestOutput]]:
                A dictionary where:
                - Key is the request ID
                - Value is a list of RequestOutput objects for that request
                Completed requests are automatically removed from tracking

        Note:
            - Thread-safe operation using condition variables
            - Short timeout avoids blocking while maintaining responsiveness
            - First call may return empty to batch small responses
            - Automatically logs finished requests via scheduler_logger
        """
        first = True

        def _get_results() -> Dict[str, List[ScheduledResponse]]:
            nonlocal first
            responses: Dict[str, List[ScheduledResponse]] = dict()

            count = 0
            for _, contents in self.local_responses.items():
                count += len(contents)

            if first and count < 64:
                first = False
                return responses

            request_ids = list(self.local_responses.keys())
            for request_id in request_ids:
                responses[request_id] = self.local_responses[request_id]
                self.local_responses[request_id] = []
            return responses

        with self.local_response_not_empty:
            responses: Dict[str, List[ScheduledResponse]] = self.local_response_not_empty.wait_for(_get_results, 0.001)

            results: Dict[str, List[RequestOutput]] = dict()
            for request_id, resps in responses.items():
                finished = False
                results[request_id] = []
                for resp in resps:
                    results[request_id].append(resp.raw)
                    finished |= resp.finished

                if finished:
                    del self.local_responses[request_id]
                    scheduler_logger.info(f"Scheduler has pulled a finished response: {[request_id]}")
            return results

    def reset(self):
        """
        Reset the scheduler to its initial state by:
        1. Clearing all Redis queues associated with this scheduler instance
        2. Removing this instance from the load balancing table
        3. Clearing in-memory tracking of responses and stolen requests

        This method is thread-safe and should be called when:
        - The scheduler needs to be cleanly restarted
        - Recovering from critical errors
        - Preparing for graceful shutdown

        Effects:
        - Deletes the request and response queues in Redis
        - Removes this scheduler's entry from the load balancing sorted set
        - Clears the local_responses dictionary tracking pending responses
        - Clears the stolen_requests dictionary tracking requests taken from other schedulers

        Note:
        - Uses the scheduler's mutex to ensure thread safety
        - Does not affect other scheduler instances in the cluster
        - After reset, the scheduler will need to be reinitialized to be usable again
        """
        with self.mutex:
            self.client.delete(self._request_queue_name(), self._response_queue_name())
            self.client.zrem(self._load_table_name(), self.name)
            self.local_responses = dict()
            self.stolen_requests = dict()
        scheduler_logger.info("Scheduler has been reset")

    def update_config(self, load_shards_num: Optional[int], reallocate: Optional[bool]):
        """
        Update the scheduler's configuration parameters dynamically.

        This method allows runtime modification of:
        - Total number of load balancing shards
        - Current instance's shard assignment

        Args:
            load_shards_num: New total number of load balancing shards (must be > 0)
            reallocate: If True, recalculates this instance's shard assignment

        Effects:
        - Updates internal load balancing configuration
        - Optionally reallocates this instance to a new shard
        - Logs configuration changes for audit purposes

        Note:
        - Changes take effect immediately for new operations
        - Existing in-progress operations continue with old configuration
        - Reallocation may affect request distribution pattern
        """
        with self.mutex:
            old_load_shards_num = self.load_shards_num
            old_shard = self.shard

            if load_shards_num:
                self.load_shards_num = load_shards_num

            if reallocate:
                self.shard = self._get_hash_slot(self.name) % self.load_shards_num

        scheduler_logger.info(
            "Scheduler has reload config, "
            f"load_shards_num({old_load_shards_num} => {self.load_shards_num}) "
            f"shard({old_shard} => {self.shard})"
        )
