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


from typing import List, Optional, Dict, Tuple
import time
from redis import ConnectionPool
from fastdeploy.scheduler.storage import AdaptedRedis
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.scheduler.workers import Workers
from fastdeploy.utils import llm_logger


class GlobalScheduler(object):
    """
    GlobalScheduler class
    """

    def __init__(self,
                 host: str,
                 port: int,
                 db: int,
                 password: Optional[str],
                 topic: str,
                 ttl: int,
                 remote_write_time: int,
                 wait_response_timeout: float
                 ):

        self.topic = topic
        self.ttl = ttl
        self.remote_write_time = remote_write_time
        self.wait_response_timeout = 1.0 if wait_response_timeout < 1.0 else wait_response_timeout
        self.wait_request_timeout = 10

        connection_pool = ConnectionPool(
            host=host, port=port, db=db, password=password, max_connections=10)
        self.client = AdaptedRedis(connection_pool=connection_pool)

        self.put_request_workers = Workers(
            "put_request_worker", self._put_requests_worker, max_batch_size=5)
        self.put_request_workers.start(size=1)

        self.put_response_workers = Workers(
            "put_response_worker", self._put_results_worker, max_batch_size=50)
        self.put_response_workers.start(size=1)

        self.get_response_workers = Workers(
            "get_response_worker", self._get_results_worker, max_batch_size=1)
        self.get_response_workers.start(size=5)
        self.response_max_batch = 50

        llm_logger.info(f"Scheduler: redis version is {self.client.version}")

    def _request_queue_name(self):
        return f"{self.topic}.request"

    def _response_queue_name(self, id: str):
        return f"{self.topic}.response.{id}"

    def _unique_key_name(self, id: str):
        return f"{self.topic}.unique.{id}"

    @staticmethod
    def calc_required_blocks(token_num, block_size):
        """calculate required blocks for given token number"""
        return (token_num + block_size - 1) // block_size

    def _put_requests_worker(self, tasks: List[Tuple[str, Request]]) -> List[Tuple[str, Optional[str]]]:
        """
            add requests to shared cache
        """
        requests: List[ScheduledRequest] = [
            ScheduledRequest(request) for _, request in tasks]

        # check the uniqueness of the request_id
        valid_requests: List[ScheduledRequest] = list()
        duplicated_ids: List[str] = list()
        for request in requests:
            unique_key = self._unique_key_name(request.id)
            if self.client.set(unique_key, "", ex=self.ttl, nx=True):
                valid_requests.append(request)
            else:
                duplicated_ids.append(request.id)

        # add to request queue
        serialized_requests = [request.serialize()
                               for request in valid_requests]
        self.client.rpush(self._request_queue_name(), *serialized_requests)
        llm_logger.info(
            f"Scheduler has put some requests: {[request.id for request in valid_requests]}")
        main_process_metrics.num_requests_waiting.inc(len(valid_requests))

        if len(duplicated_ids) > 0:
            llm_logger.warning(
                f"Scheduler has received some duplicated requests: {duplicated_ids}")

        results = [(request.id, None) for request in valid_requests]
        results += [(request_id, "duplicated request_id")
                    for request_id in duplicated_ids]
        return results

    def put_requests(self, requests: List[Request]) -> List[Tuple[str, Optional[str]]]:
        """
            add requests to scheduler
        """
        tasks: List[Tuple[str, Request]] = [
            (request.request_id, request) for request in requests]
        self.put_request_workers.put_tasks(tasks)
        return self.put_request_workers.get_results(10, 0.005)

    def get_requests(self, available_blocks, block_size, reserved_output_blocks,
                     max_num_batched_tokens, batch=1) -> List[Request]:
        """
            get requests blocked from shared cache
        """

        if available_blocks <= reserved_output_blocks or batch < 1:
            llm_logger.debug(
                f"Scheduler's resource are insufficient: available_blocks={available_blocks} "
                f"reserved_output_blocks={reserved_output_blocks} batch={batch} "
                f"max_num_batched_tokens={max_num_batched_tokens}")
            return []

        batches = []
        piece = (batch + 1) // 2
        while batch > 0:
            batch -= piece
            if batch >= 0:
                batches.append(piece)
            else:
                batches.append(piece + batch)

        serialized_requests = []
        for bs in batches:
            bs_data = self.client.lpop(self._request_queue_name(), bs)
            if bs_data is None:
                break
            serialized_requests += bs_data

        if len(serialized_requests) == 0:
            blocked_data = self.client.blpop(
                self._request_queue_name(), self.wait_request_timeout)
            if blocked_data is None:
                return []
            serialized_requests = blocked_data[1:]

        required_total_blocks = 0
        current_prefill_tokens = 0
        remaining_request = []
        requests: List[Request] = []
        for serialized_request in serialized_requests:
            if len(remaining_request) > 0:
                remaining_request.append(serialized_request)
                continue

            request: ScheduledRequest = ScheduledRequest.unserialize(
                serialized_request)
            if (time.time() - request.scheduled_time) > self.ttl:
                llm_logger.info(
                    f"Request has expired when getting a request from the scheduler: {[request.id]}")
                continue

            required_input_blocks = self.calc_required_blocks(
                request.size, block_size)
            current_prefill_tokens += request.size
            required_total_blocks += required_input_blocks + reserved_output_blocks
            if required_total_blocks > available_blocks or current_prefill_tokens > max_num_batched_tokens:
                remaining_request.append(serialized_request)
                continue
            requests.append(request.raw)

        if len(remaining_request) > 0:
            self.client.lpush(self._request_queue_name(), *remaining_request)

        if len(requests) > 0:
            llm_logger.info(
                f"Scheduler has pulled some request: {[request.request_id for request in requests]}")
        main_process_metrics.num_requests_running.inc(len(requests))
        main_process_metrics.num_requests_waiting.dec(len(requests))
        return requests

    def _put_results_worker(self, tasks: List[Tuple[str, RequestOutput]]):
        """
            add tasks to shared cache
        """
        responses: List[ScheduledResponse] = [
            ScheduledResponse(result) for _, result in tasks]
        sorted_responses = sorted(
            responses, key=lambda response: f"{response.id}.{response.index}")

        finished_responses = [
            response.id for response in responses if response.finished]
        if len(finished_responses) > 0:
            llm_logger.info(
                f"Scheduler has received a finished response: {finished_responses}")

        group = dict()
        for response in sorted_responses:
            serialized_response = response.serialize()
            if response.id not in group:
                group[response.id] = [serialized_response]
                continue
            group[response.id].append(serialized_response)

        for response_id, responses in group.items():
            ttl = self.client.ttl(self._unique_key_name(
                response_id)) - self.remote_write_time
            if ttl <= 0:
                llm_logger.warning(
                    f"Scheduler has received a expired response: {[response.id]}")
                continue

            with self.client.pipeline() as pipe:
                pipe.multi()
                pipe.rpush(self._response_queue_name(response_id), *responses)
                pipe.expire(self._response_queue_name(response_id), ttl)
                pipe.execute()

    def put_results(self, results: List[RequestOutput]):
        """
            add results to shared cache
        """
        tasks: List[Tuple[str, RequestOutput]] = [
            (result.request_id, result) for result in results]
        self.put_response_workers.put_tasks(tasks)

    def _get_results_worker(self, tasks: List[Tuple[str, str]]) -> List[Tuple[str, List[ScheduledResponse]]]:
        """
            get results blocked from shared cache
        """
        if len(tasks) != 1:
            raise ValueError(
                f"Tasks size of _get_results_worker must be 1. ({len(tasks)})")

        task_id, request_id = tasks[0]
        key = self._response_queue_name(request_id)
        size = self.client.llen(key)
        size = min(size, self.response_max_batch)

        serialized_responses = None
        if size > 0:
            serialized_responses = self.client.lpop(key, size)

        if serialized_responses is None or len(serialized_responses) == 0:
            blocked_data = self.client.blpop(key, self.wait_response_timeout)
            if blocked_data is None:
                return []
            serialized_responses = blocked_data[1:]

        output = [(task_id, [])]
        for serialized_response in serialized_responses:
            response = ScheduledResponse.unserialize(serialized_response)
            output[0][1].append(response)
        return output

    def get_results(self, request_ids: List[str]) -> Dict[str, RequestOutput]:
        """
            get results blocked from scheduler.
        """
        tasks = [(request_id, request_id) for request_id in request_ids]
        self.get_response_workers.put_tasks(tasks, deduplication=True)
        batch_responses: List[Tuple[str, List[ScheduledResponse]]] = self.get_response_workers.get_results(
            10, self.wait_response_timeout)

        results = dict()
        for _, responses in batch_responses:
            for response in responses:
                if response.id not in results:
                    results[response.id] = []
                results[response.id].append(response)
                if response.finished:
                    llm_logger.info(
                        f"Scheduler has pulled a finished response: {[response.id]}")

        request_ids = list(results.keys())
        for request_id in request_ids:
            results[request_id] = sorted(
                results[request_id], key=lambda response: f"{response.id}.{response.index}")
            results[request_id] = [
                result.raw for result in results[request_id]]
        return results
