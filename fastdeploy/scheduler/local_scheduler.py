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

import threading
import time
from typing import Dict, List, Optional, Tuple

from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.utils import envs, scheduler_logger


class LocalScheduler:
    """
    A local in-memory task scheduler for request/response management.

    This class provides functionality for:
    - Enqueuing and dequeuing requests
    - Managing request lifecycle with TTL
    - Handling request/response flow
    - Thread-safe operations with condition variables
    """

    def __init__(
        self,
        max_size: int,
        ttl: int,
        enable_chunked_prefill: bool,
        max_num_partial_prefills: int,
        max_long_partial_prefills: int,
        long_prefill_token_threshold: int,
    ):
        """
        Initializes a local in-memory scheduler for managing inference requests.

        Args:
            max_size: Maximum number of concurrent requests the scheduler can handle (0 for unlimited)
            ttl: Time-to-live in seconds for requests before automatic timeout
            enable_chunked_prefill: Whether to enable chunked prefill processing
            max_num_partial_prefills: Maximum number of partial prefill operations allowed
            max_long_partial_prefills: Maximum number of long-running partial prefill operations
            long_prefill_token_threshold: Token count threshold to classify as long prefill

        Initializes:
            - Thread synchronization primitives (mutex, condition variables)
            - Request and response tracking structures
            - Chunked prefill configuration parameters
            - Request queue management system

        Note:
            - Uses thread-safe operations for concurrent access
            - Automatically recycles expired requests based on TTL
            - Supports both batched and individual request processing
        """
        self.max_size = max_size
        self.ttl = ttl
        self.mutex = threading.Lock()

        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold

        self.ids_read_cursor = 0
        self.ids: List[str] = list()

        self.requests: Dict[str, ScheduledRequest] = dict()
        self.responses: Dict[str, List[ScheduledResponse]] = dict()
        self.batch_responses_per_step: List[List[ScheduledResponse]] = list()

        self.wait_request_timeout = 10
        self.wait_response_timeout = 0.001

        self.requests_not_empty = threading.Condition(self.mutex)
        self.responses_not_empty = threading.Condition(self.mutex)

    def reset(self):
        """
        Reset the local scheduler to its initial empty state by:
        1. Resetting the request ID tracking cursor to 0
        2. Clearing all stored request IDs
        3. Clearing all pending requests
        4. Clearing all cached responses

        This method is thread-safe and should be called when:
        - The scheduler needs to be cleanly restarted
        - Recovering from critical errors
        - Preparing for graceful shutdown

        Effects:
        - Resets the ids_read_cursor to 0 (request processing position)
        - Clears the ids list tracking all request IDs
        - Clears the requests dictionary tracking pending requests
        - Clears the responses dictionary tracking received responses

        Note:
        - Uses the scheduler's mutex to ensure thread safety
        - Does not affect the scheduler's configuration parameters (max_size, ttl, etc.)
        - After reset, the scheduler will be empty but still operational
        """
        with self.mutex:
            self.ids_read_cursor = 0
            self.ids = list()
            self.requests = dict()
            self.responses = dict()
        scheduler_logger.info("Scheduler has been reset")

    def _recycle(self, request_id: Optional[str] = None):
        """
        Clean up expired or completed requests to free memory.

        Args:
            request_id: Optional specific request ID to remove.
                       If None, removes all expired requests.
        """
        if request_id is not None:
            self.requests.pop(request_id, None)
            self.responses.pop(request_id, None)
            self.ids.pop(self.ids.index(request_id))
            self.ids_read_cursor -= 1
            return

        if self.max_size <= 0:
            return

        if len(self.requests) <= self.max_size:
            return

        now = time.time()
        expired_ids = []
        for request_id in self.ids:
            request = self.requests[request_id]
            if now - request.schedule_time < self.ttl:
                break
            expired_ids.append(request.request_id)

        for i, expired_id in enumerate(expired_ids):
            self.requests.pop(expired_id, None)
            self.responses.pop(expired_id, None)
            self.ids.pop(i)

        if len(expired_ids) > 0:
            if len(expired_ids) - 1 >= self.ids_read_cursor:
                self.ids_read_cursor = 0
            else:
                self.ids_read_cursor -= len(expired_ids)

    def put_requests(self, requests: List[Request]) -> List[Tuple[str, Optional[str]]]:
        """
        Add new requests to the scheduler queue.

        Args:
            requests: List of Request objects to enqueue

        Returns:
            List of tuples containing (request_id, error_message) for each request.
            error_message is None for successful enqueues.
        """
        with self.mutex:
            self._recycle()
            if self.max_size > 0 and len(self.requests) + len(requests) > self.max_size:
                msg = f"Exceeding the max length of the local scheduler (max_size={self.max_size})"
                return [(request.request_id, msg) for request in requests]

            valid_ids = []
            duplicated_ids = []
            for request in requests:
                if request.request_id in self.requests:
                    duplicated_ids.append(request.request_id)
                else:
                    scheduled_request = ScheduledRequest(request)
                    self.requests[scheduled_request.request_id] = scheduled_request
                    valid_ids.append(scheduled_request.request_id)

            self.ids += valid_ids
            self.requests_not_empty.notify_all()
        scheduler_logger.info(f"Scheduler has enqueued some requests: {valid_ids}")

        if len(duplicated_ids) > 0:
            scheduler_logger.warning(f"Scheduler has received some duplicated requests: {duplicated_ids}")

        results = [(request_id, None) for request_id in valid_ids]
        results += [(request_id, "duplicated request_id") for request_id in duplicated_ids]
        return results

    def has_request(self, request_id: str) -> bool:
        """
        Check if there are any pending requests in the scheduler.

        Args:
            request_id: Optional specific request ID to check.
                        If None, checks whether there are any pending requests.

        Returns:
            True if there are pending requests, False otherwise.
        """
        with self.mutex:
            return request_id in self.requests

    def calc_required_blocks(self, token_num, block_size):
        """
        Calculate the number of blocks needed for a given number of tokens.

        Args:
            token_num: Number of tokens
            block_size: Size of each block

        Returns:
            Number of blocks required (rounded up)
        """
        return (token_num + block_size - 1) // block_size

    def get_unhandled_request_num(self):
        return len(self.ids) - self.ids_read_cursor

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch=1,
    ) -> List[Request]:
        """
        Retrieve requests from the scheduler based on available resources.

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

        with self.requests_not_empty:
            batch_ids = self.requests_not_empty.wait_for(
                lambda: self.ids[self.ids_read_cursor : self.ids_read_cursor + batch],
                self.wait_request_timeout,
            )

            requests: List[Request] = []
            required_total_blocks = 0
            current_prefill_tokens = 0
            long_partial_requests, short_partial_requests = 0, 0
            for request_id in batch_ids:
                request = self.requests[request_id]
                required_input_blocks = self.calc_required_blocks(request.prompt_tokens_ids_len, block_size)
                current_prefill_tokens += request.prompt_tokens_ids_len
                required_total_blocks += required_input_blocks + reserved_output_blocks
                if required_total_blocks > available_blocks:
                    break

                if not envs.FD_ENABLE_MAX_PREFILL:
                    if self.enable_chunked_prefill:
                        if request.prompt_tokens_ids_len > self.long_prefill_token_threshold:
                            # 长请求
                            long_partial_requests += 1
                            if long_partial_requests > self.max_long_partial_prefills:
                                break
                        else:
                            short_partial_requests += 1

                        if short_partial_requests + long_partial_requests > self.max_num_partial_prefills:
                            break
                    else:
                        if current_prefill_tokens > max_num_batched_tokens and len(requests) > 0:
                            break
                requests.append(request.raw)

            self.ids_read_cursor += len(requests)

        if len(batch_ids) > 0 and len(requests) == 0:
            scheduler_logger.debug(f"Scheduler has put all just-pulled request into the queue: {len(batch_ids)}")

        if len(requests) > 0:
            scheduler_logger.info(f"Scheduler has pulled some request: {[request.request_id for request in requests]}")

        return requests

    def put_results(self, results: List[RequestOutput]):
        """
        Add processing results back to the scheduler.

        Args:
            results: List of RequestOutput objects containing results
        """
        scheduler_logger.debug(f"put results: {results}")
        responses: List[ScheduledResponse] = [ScheduledResponse(result) for result in results]

        finished_responses = [response.request_id for response in responses if response.finished]
        if len(finished_responses) > 0:
            scheduler_logger.info(f"Scheduler has received some finished responses: {finished_responses}")

        with self.mutex:
            self.batch_responses_per_step.append([response.raw for response in responses])
            for response in responses:
                if response.request_id not in self.requests:
                    scheduler_logger.warning(f"Scheduler has received a expired response: {[response.request_id]}")
                    continue

                if response.request_id not in self.responses:
                    self.responses[response.request_id] = [response]
                    continue
                scheduler_logger.debug(f"append response {response.raw}")
                self.responses[response.request_id].append(response)
            self.responses_not_empty.notify_all()

    def get_results(self) -> Dict[str, List[RequestOutput]]:
        """
        Retrieve all available results from the scheduler and clean up completed requests.

        This method:
        - Waits for new responses using a condition variable
        - Returns all currently available responses
        - Automatically removes completed requests from the scheduler
        - Logs finished requests

        Returns:
            Dict[str, List[RequestOutput]]:
                A dictionary where:
                - Key is the request ID
                - Value is a list of RequestOutput objects for that request
                Completed requests are automatically removed from the scheduler

        Note:
            - Thread-safe operation using condition variables
            - Has a short timeout (0.001s) to avoid blocking
            - Automatically recycles completed requests to free memory
            - Logs finished requests via scheduler_logger
        """

        def _get_results():
            responses = self.responses
            batch_responses_per_step = self.batch_responses_per_step
            self.responses = dict()
            self.batch_responses_per_step = list()
            if not responses:
                return None  # No response yet
            return responses, batch_responses_per_step

        with self.responses_not_empty:
            wait_response_result = self.responses_not_empty.wait_for(_get_results, self.wait_response_timeout)
            if wait_response_result is not None:
                responses, batch_responses_per_step = wait_response_result
            else:
                responses, batch_responses_per_step = dict(), list()
            results = dict()
            for request_id, resps in responses.items():
                finished = False
                results[request_id] = []
                for resp in resps:
                    results[request_id].append(resp.raw)
                    finished |= resp.finished

                if finished:
                    self._recycle(request_id)
                    scheduler_logger.info(f"Scheduler has pulled a finished response: {[request_id]}")

            if results:
                scheduler_logger.debug(f"get responses, {results}")

            if envs.FD_ENABLE_INTERNAL_ADAPTER:
                return batch_responses_per_step
            else:
                return results
