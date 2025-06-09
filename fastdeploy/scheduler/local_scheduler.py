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


from typing import Dict, List, Optional, Tuple
import threading
import time

from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import llm_logger
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse


class LocalScheduler(object):
    """
    LocalScheduler Class
    """

    def __init__(self,
                 max_size: int,
                 ttl: int,
                 wait_response_timeout: float):
        self.max_size = max_size
        self.ttl = ttl
        self.mutex = threading.Lock()
        self.ids_read_cursor = 0
        self.ids: List[str] = list()

        self.requests: Dict[str, ScheduledRequest] = dict()
        self.responses: Dict[str, List[ScheduledResponse]] = dict()

        self.wait_request_timeout = 10
        self.wait_response_timeout = wait_response_timeout

        self.requests_not_empty = threading.Condition(self.mutex)
        self.responses_not_empty = threading.Condition(self.mutex)

    def _recycle(self, request_id: Optional[str] = None):
        """
            recycle memory
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
            if (now - request.scheduled_time < self.ttl):
                break
            expired_ids.append(request.id)

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
        """  submit requests to scheduler
             Args:
                 requests: List[Request]
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
                    self.requests[scheduled_request.id] = scheduled_request
                    valid_ids.append(scheduled_request.id)

            self.ids += valid_ids
            self.requests_not_empty.notify_all()

        llm_logger.info(
                f"Scheduler has put some requests: {valid_ids}")
        main_process_metrics.num_requests_waiting.inc(len(valid_ids))

        if len(duplicated_ids) > 0:
            llm_logger.warning(
                    f"Scheduler has received some duplicated requests: {duplicated_ids}")

        results = [(request_id, None) for request_id in valid_ids]
        results += [(request_id, "duplicated request_id")
                        for request_id in duplicated_ids]
        return results

    def calc_required_blocks(self, token_num, block_size):
        """calculate required blocks for given token number"""
        return (token_num + block_size - 1) // block_size

    def get_requests(self, available_blocks, block_size,
                     reserved_output_blocks, max_num_batched_tokens, batch=1) -> List[Request]:
        """get requests from local cache
            Args:
                available_blocks: int
                block_size: int
                reserved_output_blocks: int
                max_num_batched_tokens: int
                batch: int
        """
        if available_blocks <= reserved_output_blocks or batch < 1:
            llm_logger.debug(
                f"Scheduler's resource are insufficient: available_blocks={available_blocks} "
                f"reserved_output_blocks={reserved_output_blocks} batch={batch} "
                f"max_num_batched_tokens={max_num_batched_tokens}")
            return []

        with self.requests_not_empty:
            batch_ids = self.requests_not_empty.wait_for(
                lambda: self.ids[self.ids_read_cursor:
                                 self.ids_read_cursor + batch], self.wait_request_timeout)

            required_total_blocks = 0
            current_prefill_tokens = 0
            requests: List[Request] = []
            for request_id in batch_ids:
                request = self.requests[request_id]
                required_input_blocks = self.calc_required_blocks(
                    request.size, block_size)
                current_prefill_tokens += request.size
                required_total_blocks += required_input_blocks + reserved_output_blocks
                if required_total_blocks > available_blocks or current_prefill_tokens > max_num_batched_tokens:
                    break
                requests.append(request.raw)
            self.ids_read_cursor += len(requests)

        if len(requests) > 0:
            llm_logger.info(
                    f"Scheduler has pulled some request: {[request.request_id for request in requests]}")
        main_process_metrics.num_requests_waiting.dec(len(requests))
        main_process_metrics.num_requests_running.inc(len(requests))
        return requests

    def put_results(self, results: List[RequestOutput]):
        """put results into local cache"""
        responses: List[ScheduledResponse] = [
            ScheduledResponse(result) for result in results]

        finished_responses = [
            response.id for response in responses if response.finished]
        if len(finished_responses) > 0:
            llm_logger.info(
                f"Scheduler has received a finished response: {finished_responses}")

        with self.mutex:
            for response in responses:
                if response.id not in self.requests:
                    llm_logger.warning(
                        f"Scheduler has received a expired response: {[response.id]}")
                    continue

                if response.id not in self.responses:
                    self.responses[response.id] = [response]
                    continue
                self.responses[response.id].append(response)
            self.responses_not_empty.notify_all()

    def get_results(self, request_ids: List[str]) -> Dict[str, List[RequestOutput]]:
        """get results from local cache"""
        def _get_results():
            responses = dict()
            for request_id in request_ids:
                if request_id not in responses:
                    responses[request_id] = []
                responses[request_id] += self.responses.pop(request_id, [])
            return responses

        with self.responses_not_empty:
            responses = self.responses_not_empty.wait_for(
                _get_results, self.wait_response_timeout)

            results = dict()
            for request_id, resps in responses.items():
                finished = False
                results[request_id] = []
                for resp in resps:
                    results[request_id].append(resp.raw)
                    finished |= resp.finished

                if finished:
                    self._recycle(request_id)
                    llm_logger.info(
                        f"Scheduler has pulled a finished response: {[request_id]}")
            return results
