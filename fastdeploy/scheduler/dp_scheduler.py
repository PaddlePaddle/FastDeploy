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
from multiprocessing import Queue
from typing import Dict, List
from typing import Dict, List, Optional, Tuple

from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledResponse
from fastdeploy.scheduler.local_scheduler import LocalScheduler
from fastdeploy.utils import scheduler_logger


class DPLocalScheduler(LocalScheduler):
    def __init__(
        self,
        max_size: int,
        ttl: int,
        enable_chunked_prefill: bool,
        max_num_partial_prefills: int,
        max_long_partial_prefills: int,
        long_prefill_token_threshold: int,
        splitwise_role: str = 'prefill'
    ):
        super().__init__(
            max_size,
            ttl,
            enable_chunked_prefill,
            max_num_partial_prefills,
            max_long_partial_prefills,
            long_prefill_token_threshold,
        )
        self.splitwise_role = splitwise_role

    def put_results(self, results: List[RequestOutput]):
        """
        Add processing results back to the scheduler.
        Args:
            results: List of RequestOutput objects containing results
        """
        responses: List[ScheduledResponse] = [ScheduledResponse(result) for result in results]

        finished_responses = [response.request_id for response in responses if response.finished]
        if len(finished_responses) > 0:
            scheduler_logger.info(f"Scheduler has received some finished responses: {finished_responses}")

        with self.mutex:
            for response in responses:
                if response.request_id not in self.responses:
                    self.responses[response.request_id] = [response]
                    continue
                self.responses[response.request_id].append(response)
            self.responses_not_empty.notify_all()

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
            if self.splitwise_role == 'decode':
                return
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




class DPScheduler:
    def __init__(
        self,
        max_size: int,
        ttl: int,
        enable_chunked_prefill: bool,
        max_num_partial_prefills: int,
        max_long_partial_prefills: int,
        long_prefill_token_threshold: int,
        splitwise_role: str = 'prefill'
    ):
        self._scheduler = DPLocalScheduler(
            max_size,
            ttl,
            enable_chunked_prefill,
            max_num_partial_prefills,
            max_long_partial_prefills,
            long_prefill_token_threshold,
            splitwise_role
        )

    def start(self, dp_rank: int, request_queues: List[Queue], result_queue: Queue):
        self.dp_rank = dp_rank
        self.request_queues = request_queues
        self.result_queue = result_queue
        threading.Thread(target=self._put_requests_to_local).start()
        threading.Thread(target=self._get_response_from_local).start()

    def put_requests(self, requests: List[Dict]):
        results = []
        for request in requests:
            self.request_queues[request.dp_rank].put(request)
            results.append((request.request_id, None))
        return results

    def _put_requests_to_local(self):
        while True:
            request = self.request_queues[self.dp_rank].get()
            self._scheduler.put_requests([request])

    def _get_response_from_local(self):
        while True:
            results = self._scheduler.get_results()
            if len(results) == 0:
                continue
            self.result_queue.put(results)

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch=1,
    ) -> List[Request]:
        return self._scheduler.get_requests(
            available_blocks, block_size, reserved_output_blocks, max_num_batched_tokens, batch
        )

    def put_results(self, results: List[RequestOutput]):
        self._scheduler.put_results(results)

    def get_results(self) -> Dict[str, List[RequestOutput]]:
        return self.result_queue.get()