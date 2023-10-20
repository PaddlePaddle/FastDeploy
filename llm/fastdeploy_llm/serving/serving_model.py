# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import queue, threading, time, copy
from fastdeploy_llm.task import TaskStatus, BatchTask, Task
from fastdeploy_llm.model import Model
from fastdeploy_llm.utils.logging_util import logger
from fastdeploy_llm.config import Config


class ServingModel:
    def __init__(self, config):
        self.config = config

        logger.info("=============== Debug Information ===============")
        for k, v in self.config.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=================================================\n")

        self.model = Model(config)

        self.requests_queue = queue.Queue(maxsize=self.config.max_queue_num)
        self.stop_threshold = self.config.stop_threshold
        self.disable_dynamic_batching = self.config.disable_dynamic_batching
        self.runner_thread = None
        self.stop_queue = False
        self.stop_runner = False

        self.max_need_num = min(self.config.max_batch_size,
                                int(self.config.stop_threshold * 1.5))

    def add_request(self, task):
        logger.debug("Receive task = {}".format(task))
        assert not self.stop_queue, "The serving model is stopped, cannot accept new requests now."
        assert task.text.strip() != "", "The request's text cannot be empty."
        try:
            if not hasattr(task, "token_ids"):
                task.token_ids, task.position_ids = self.model.data_processor.encode(
                    task.text, padding=True)
            else:
                task.position_ids = None
            if self.config.is_ptuning:
                assert len(
                    task.token_ids
                ) + self.config.max_prefix_len <= self.config.max_seq_len, "The request's token number({}) + max_prefix_len({}) = {} is exceeding the setting max_seq_len({}).".format(
                    len(task.token_ids), self.config.max_prefix_len,
                    len(task.token_ids) + self.config.max_prefix_len,
                    self.config.max_seq_len)
            else:
                assert len(
                    task.token_ids
                ) <= self.config.max_seq_len, "The request's token number({}) is exceed the setting max_seq_len({}).".format(
                    len(task.token_ids), self.config.max_seq_len)
            self.requests_queue.put(task, timeout=0.5)
            logger.debug("Task with task_id={} is put into requests queue.".
                         format(task.task_id))
        except Exception as e:
            raise Exception(
                "There's error while inserting request, error={}.".format(e))

    def runner(self):
        batch_tasks = BatchTask(self.config.max_batch_size)
        while not self.stop_runner:
            remaining_slots_num = batch_tasks.remaining_slots_size()

            if self.disable_dynamic_batching:
                get_tasks_num = remaining_slots_num
            else:
                get_tasks_num = min(remaining_slots_num, self.max_need_num)
            get_tasks = list()
            for i in range(get_tasks_num):
                try:
                    task = self.requests_queue.get(timeout=0.1)
                    get_tasks.append(task)
                except Exception as e:
                    break
            if len(get_tasks) == 0 and batch_tasks.unfinished_size() == 0:
                time.sleep(0.1)
                continue

            sender_size = 0
            if self.model.stream_sender is not None:
                sender_size = len(self.model.stream_sender)
            logger.info(
                "Get data from queue, num = {} batch_task_size = {}, reponse_handler_size = {}.".
                format(len(get_tasks), batch_tasks.size(), sender_size))
            get_tasks_num = len(get_tasks)
            batch_tasks.update(get_tasks)

            if batch_tasks.size() == 0:
                logger.info(
                    "There's no task need to process now, wait for new request..."
                )
                continue

            stop_nums = batch_tasks.size() - batch_tasks.unfinished_size()
            if self.config.disable_dynamic_batching:
                stop_nums = batch_tasks.size()
            elif batch_tasks.size(
            ) < self.config.stop_threshold * 2 or self.config.stop_threshold == 0:
                stop_nums = (stop_nums + batch_tasks.size() + 1) // 2
            else:
                stop_nums = min(stop_nums + self.config.stop_threshold,
                                batch_tasks.size())
            finished_num = batch_tasks.size() - batch_tasks.unfinished_size()
            if stop_nums < finished_num:
                stop_nums = finished_num + 1
            logger.info(
                "Insert {} new request, current prediction with batch = {} unfinished_task = {} stop_nums = {}.".
                format(get_tasks_num,
                       batch_tasks.size(),
                       batch_tasks.unfinished_size(), stop_nums))
            logger.debug(
                "BatchTasks with task_id={} is send to engine to predict.".
                format([t.task_id for t in batch_tasks.tasks]))
            self.model.predict(batch_tasks, stop_nums)
            logger.debug("The last batch tasks' status = {}.".format(
                [t.status for t in batch_tasks.tasks]))
            logger.info("Waiting for new requests...")

    def start(self):
        self.runner_thread = threading.Thread(target=self.runner)
        self.runner_thread.start()

    def force_stop(self):
        remaining_requests = self.requests_queue.qsize()
        if remaining_requests > 0:
            logger.warning(
                "There's almost {} requests are not processed, will ignore them.".
                format(remaining_requests))
        if self.model._is_engine_busy():
            logger.warning(
                "The inference engine is still processing requests, will stop them, this may cause some requests lost."
            )
        self.stop_queue = True
        self.stop_runner = True
        self.model.kill_engine()

    def stop(self):
        logger.info("Stop receiving new request now.")
        self.stop_queue = True
        while self.requests_queue.qsize() > 0:
            time.sleep(2)
        logger.info("There's no request in queue now.")
        time.sleep(2)
        logger.info("Wait until engine is finished...")
        while self.model._is_engine_busy():
            time.sleep(2)
        logger.info("Engine is finished now, will stop now.")
        self.stop_runner = True
        self.model.kill_engine()
        logger.info(
            "Engine is killed, will terminate the current process in 10 seconds..."
        )
        time.sleep(10)
        logger.info("ByeBye.")
