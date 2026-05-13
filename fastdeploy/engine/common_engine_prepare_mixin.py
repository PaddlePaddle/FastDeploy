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

from __future__ import annotations

import threading
import time
import traceback

import fastdeploy.metrics.trace as tracing
from fastdeploy.engine.request import RequestOutput
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
from fastdeploy.utils import envs


class EngineServicePrepareMixin:
    def _fetch_request_mixed(self) -> bool:
        """Fetch and prepare requests for a mixed instance. Returns True if tasks were fetched."""
        # FIXME: to validate if it's necessary for avoiding error when enable mtp
        if len(self.resource_manager.waiting) > 0:
            return False

        num_prefill_batch = min(
            int(self.resource_manager.available_batch()),
            self.cfg.max_prefill_batch,
        )
        max_num_batched_tokens = self.cfg.model_config.max_model_len
        available_blocks = self.cfg.cache_config.max_block_num_per_seq

        tasks = self.scheduler.get_requests(
            available_blocks=available_blocks,
            block_size=self.cfg.cache_config.block_size,
            reserved_output_blocks=0,
            max_num_batched_tokens=max_num_batched_tokens,
            batch=num_prefill_batch,
        )
        if not tasks:
            return False

        for task in tasks:
            task.metrics.engine_get_req_time = time.time()
            trace_print(LoggingEventName.REQUEST_QUEUE_END, task.request_id, getattr(task, "user", ""))

        self.llm_logger.debug(
            f"Engine has fetched tasks from {self.scheduler.__class__.__name__}: {[task.request_id for task in tasks]}"
        )

        for task in tasks:
            task.metrics.add_req_to_resource_manager_time = time.time()
            trace_print(LoggingEventName.RESOURCE_ALLOCATE_START, task.request_id, getattr(task, "user", ""))
            self.resource_manager.add_request(task)

        return True

    def _fetch_request_decode(self) -> bool:
        """Consume scheduler queue for decode instance to prevent memory accumulation.
        Returns True if tasks were consumed."""
        num_prefill_batch = min(
            int(self.resource_manager.available_batch()),
            self.cfg.max_prefill_batch,
        )
        max_num_batched_tokens = self.cfg.scheduler_config.max_num_batched_tokens
        available_blocks = self.cfg.cache_config.max_block_num_per_seq

        tasks = self.scheduler.get_requests(
            available_blocks=available_blocks,
            block_size=self.cfg.cache_config.block_size,
            reserved_output_blocks=0,
            max_num_batched_tokens=max_num_batched_tokens,
            batch=num_prefill_batch,
        )
        # Tasks are intentionally discarded - decode receives requests via _decode_process_splitwise_requests
        return len(tasks) > 0

    def _fetch_request_prefill(self) -> bool:
        """Fetch and prepare requests for a prefill instance. Returns True if tasks were fetched."""
        num_prefill_batch = min(
            int(self.resource_manager.available_batch()),
            self.cfg.max_prefill_batch,
        )
        max_num_batched_tokens = self.cfg.scheduler_config.max_num_batched_tokens
        available_blocks = self.cfg.cache_config.max_block_num_per_seq

        tasks = self.scheduler.get_requests(
            available_blocks=available_blocks,
            block_size=self.cfg.cache_config.block_size,
            reserved_output_blocks=0,
            max_num_batched_tokens=max_num_batched_tokens,
            batch=num_prefill_batch,
        )
        if not tasks:
            return False

        for task in tasks:
            task.metrics.engine_get_req_time = time.time()
            trace_print(LoggingEventName.REQUEST_QUEUE_END, task.request_id, getattr(task, "user", ""))

        self.llm_logger.debug(
            f"Engine has fetched tasks from {self.scheduler.__class__.__name__}: {[task.request_id for task in tasks]}"
        )

        # Start async preprocess for all tasks in this batch
        for task in tasks:
            self.resource_manager.apply_async_preprocess(task)

        # P-side resource preallocation + D-side coordination
        failed_tasks = []
        if envs.PREFILL_CONTINUOUS_REQUEST_DECODE_RESOURCES:
            for task in tasks:
                # assure can allocate block ids in P
                while not self.resource_manager.preallocate_resource_in_p(task):
                    time.sleep(0.005)
                self.llm_logger.debug(
                    f"P has allocated resources and then ask D resource for request: {task.request_id}"
                )
                trace_print(LoggingEventName.ASK_DECODE_RESOURCE_START, task.request_id, getattr(task, "user", ""))
                task.metrics.ask_decode_resource_start_time = time.time()
                while True:
                    self.split_connector.send_splitwise_tasks([task], task.idx)
                    status, msg = self.split_connector.check_decode_allocated(task)
                    if status:
                        task.metrics.ask_decode_resource_finish_time = time.time()
                        trace_print(
                            LoggingEventName.ASK_DECODE_RESOURCE_END,
                            task.request_id,
                            getattr(task, "user", ""),
                        )
                        break
                    else:
                        self.llm_logger.warning(
                            f"D failed to allocate resource for request {task.request_id}, try again."
                        )
                        time.sleep(0.05)

                self.llm_logger.debug(f"D has allocated resource for request: {task.request_id}")
        else:
            for task in tasks:
                # assure can allocate block ids in P
                while not self.resource_manager.preallocate_resource_in_p(task):
                    time.sleep(0.005)

                self.llm_logger.debug(
                    f"P has allocated resources and then ask D resource for req_id: {task.request_id}"
                )
                trace_print(LoggingEventName.ASK_DECODE_RESOURCE_START, task.request_id, getattr(task, "user", ""))
                task.metrics.ask_decode_resource_start_time = time.time()
                self.split_connector.send_splitwise_tasks([task], task.idx)

            for task in tasks:
                # assure fetch block ids from D
                status, msg = self.split_connector.check_decode_allocated(task)
                task.metrics.ask_decode_resource_finish_time = time.time()
                trace_print(LoggingEventName.ASK_DECODE_RESOURCE_END, task.request_id, getattr(task, "user", ""))
                if not status:
                    error_msg = (
                        f"PD Error: prefill failed to apply for resource from decode, "
                        f"req: {task.request_id}, msg:{msg}."
                    )
                    self.llm_logger.error(error_msg)
                    self.scheduler.put_results(
                        [
                            RequestOutput(
                                request_id=task.request_id,
                                finished=True,
                                error_code=500,
                                error_msg=error_msg,
                            )
                        ]
                    )
                    main_process_metrics.reschedule_req_num.inc()
                    failed_tasks.append(task)

        for tmp_task in failed_tasks:
            tasks.remove(tmp_task)
            self.resource_manager.pre_recycle_resource(tmp_task.request_id)

        # Check and wait async preprocess
        if tasks:
            need_check_req_ids = [task.request_id for task in tasks]
            failed_tasks = []

            while need_check_req_ids:
                still_in_progress = False
                for task in tasks:
                    if task.request_id not in need_check_req_ids:
                        continue

                    result = self.resource_manager.waiting_async_process(task)
                    if result is False:  # async preprocess success
                        need_check_req_ids.remove(task.request_id)
                    elif result is True:
                        still_in_progress = True
                    elif result is None:  # async preprocess failed
                        failed_tasks.append(task)
                        need_check_req_ids.remove(task.request_id)
                        self.scheduler.put_results(
                            [
                                RequestOutput(
                                    request_id=task.request_id,
                                    finished=True,
                                    error_code=task.error_code,
                                    error_msg=task.error_message,
                                )
                            ]
                        )

                if still_in_progress:
                    time.sleep(0.005)

            for tmp_task in failed_tasks:
                tasks.remove(tmp_task)
                self.resource_manager.pre_recycle_resource(tmp_task.request_id)

        # Send cache info to messager
        if tasks:
            self.split_connector.send_cache_info_to_messager(tasks, 0)

        # Fetch requests and add them to the scheduling queue
        if tasks:
            for task in tasks:
                task.metrics.add_req_to_resource_manager_time = time.time()
                trace_print(LoggingEventName.RESOURCE_ALLOCATE_START, task.request_id, getattr(task, "user", ""))
            self.resource_manager.add_request_in_p(tasks)
            self.llm_logger.info(f"P add requests into running queue: {[task.request_id for task in tasks]}")

        return True

    def _fetch_loop(self, fetch_fn, thread_idx: int):
        """Fetch loop run by each worker thread."""
        tracing.trace_set_thread_info(f"Prepare Request for Scheduling - thread {thread_idx}")
        while self.running:
            try:
                with self._pause_cond:
                    self._pause_cond.wait_for(lambda: not self.is_paused)
                fetch_fn()
                time.sleep(0.002)
            except Exception as e:
                self.llm_logger.error(f"fetching request error in worker-{thread_idx}: {e} {traceback.format_exc()}")
                time.sleep(0.002)

    def _prepare_request_v1(self):
        """Prepare request and send to the queue for scheduling"""
        tracing.trace_set_thread_info("Prepare Request for Scheduling")
        role = self.cfg.scheduler_config.splitwise_role
        num_workers = envs.FD_PREFILL_PREPARE_REQ_THREAD_NUM if role == "prefill" else 1
        self.llm_logger.info(f"prepare request for scheduling, role: {role}, num_workers: {num_workers}")

        fetch_fn = {
            "mixed": self._fetch_request_mixed,
            "prefill": self._fetch_request_prefill,
            "decode": self._fetch_request_decode,
        }[role]

        self._fetch_threads = []
        for i in range(num_workers):
            t = threading.Thread(
                target=self._fetch_loop,
                args=(fetch_fn, i),
                daemon=True,
                name=f"fetch-{i}",
            )
            t.start()
            self._fetch_threads.append(t)

        # Keep this thread alive for graceful shutdown
        while self.running:
            time.sleep(1.0)
