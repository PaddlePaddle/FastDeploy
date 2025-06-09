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

from typing import Callable, List, Tuple, Any, Dict, Optional
import functools
import threading
from fastdeploy.utils import llm_logger


class Workers:
    """
        Workers class
    """

    def __init__(self,
                 name: str,
                 work: Callable[[List[Tuple[str, Any]]], Optional[List[Tuple[str, Any]]]],
                 max_batch_size: int = 1):
        self.name = name
        self.work = work
        self.max_batch_size = max_batch_size

        self.mutex = threading.Lock()
        self.pool = []

        self.tasks_not_empty = threading.Condition(self.mutex)
        self.results_not_empty = threading.Condition(self.mutex)

        self.tasks: List[Tuple[str, Any]] = []
        self.results: List[Tuple[str, Any]] = []
        self.running_tasks: Dict[int, List[Tuple[str, Any]]] = dict()

        self.not_stop = threading.Condition(self.mutex)
        self.stop = False
        self.stopped = 0

    def _stop(self, func: Callable):
        """
            a stop decorator
        """
        @functools.wraps(func)
        def wrapper():
            if self.stop:
                return True
            return func()
        return wrapper

    def _worker(self, number: int):
        """
            worker thread
        """
        with self.mutex:
            self.running_tasks[number] = []

        @self._stop
        def _get_tasks():
            self.running_tasks[number] = []
            batch = min((len(self.tasks) + len(self.pool) - 1) //
                        len(self.pool), self.max_batch_size)
            tasks = self.tasks[:batch]
            del self.tasks[:batch]
            self.running_tasks[number] = tasks
            return tasks

        while True:
            with self.tasks_not_empty:
                tasks = self.tasks_not_empty.wait_for(_get_tasks)
                if self.stop:
                    self.stopped += 1
                    if self.stopped == len(self.pool):
                        self.not_stop.notify_all()
                    return

            results = []
            try:
                results = self.work(tasks)
            except Exception as e:
                llm_logger.info(f"Worker {self.name} execute error: {e}")

            if results is not None and len(results) > 0:
                with self.mutex:
                    self.results += results
                    self.results_not_empty.notify_all()

    def start(self, size: int):
        """
            start thread pood
        """
        with self.mutex:
            remain = size - len(self.pool)
            if remain <= 0:
                return

            for i in range(remain):
                t = threading.Thread(target=self._worker, args=(i,))
                t.daemon = True
                t.start()
                self.pool.append(t)

    def terminate(self):
        """
            terminame thread pool
        """
        with self.mutex:
            self.stop = True
            self.tasks_not_empty.notify_all()
            self.results_not_empty.notify_all()

            self.not_stop.wait_for(lambda: self.stopped == len(self.pool))

            self.pool = []
            self.tasks = []
            self.results = []
            self.running_tasks = dict()
            self.stop = False
            self.stopped = 0

    def get_results(self, max_size: int, timeout: float) -> List[Tuple[str, Any]]:
        """
            get results from thread pool.
        """
        @self._stop
        def _get_results():
            results = self.results[:max_size]
            del self.results[:max_size]
            return results

        with self.results_not_empty:
            results = self.results_not_empty.wait_for(_get_results, timeout)
            if self.stop:
                return []
            return results

    def put_tasks(self, tasks: List[Tuple[str, Any]], deduplication: bool = False):
        """
            put tasks into thread pool.
        """
        if len(tasks) == 0:
            return

        with self.mutex:
            if not deduplication:
                self.tasks += tasks
            else:
                task_set = set([t[0] for t in self.tasks])
                for _, running in self.running_tasks.items():
                    task_set.update([t[0] for t in running])

                for task in tasks:
                    if task[0] in task_set:
                        continue
                    self.tasks.append(task)
            self.tasks_not_empty.notify_all()
