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

import functools
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

from fastdeploy.utils import scheduler_logger


class Task:
    """
    A container class representing a unit of work to be processed.

    Attributes:
        id: Unique identifier for the task
        raw: The actual task payload/data
        reason: Optional reason/status message for the task
    """

    def __init__(self, task_id: str, task: Any, reason: Optional[str] = None):
        """
        Initialize a Task instance.

        Args:
            task_id: Unique identifier for the task
            task: The actual task payload/data
            reason: Optional reason/status message
        """

        self.id = task_id
        self.raw = task
        self.reason = reason

    def __repr__(self) -> str:
        return f"task_id:{self.id} reason:{self.reason}"


class Workers:
    """
    A thread pool implementation for parallel task processing.

    Features:
    - Configurable number of worker threads
    - Task batching support
    - Custom task filtering
    - Thread-safe task queue
    - Graceful shutdown
    """

    def __init__(
        self,
        name: str,
        work: Callable[[List[Task]], Optional[List[Task]]],
        max_task_batch_size: int = 1,
        task_filters: Optional[List[Callable[[Task], bool]]] = None,
    ):
        """
        Initialize a Workers thread pool.

        Args:
            name: Identifier for the worker pool
            work: The worker function that processes tasks
            max_task_batch_size: Maximum tasks processed per batch
            task_filters: Optional list of filter functions for task assignment
        """

        self.name: str = name
        self.work: Callable[[List[Task]], Optional[List[Task]]] = work
        self.max_task_batch_size: int = max_task_batch_size
        self.task_filters: List[Callable[[Task], bool]] = task_filters

        self.mutex = threading.Lock()
        self.pool = []

        self.tasks_not_empty = threading.Condition(self.mutex)
        self.results_not_empty = threading.Condition(self.mutex)

        self.tasks: List[Task] = []
        self.results: List[Task] = []
        self.running_tasks: Dict[int, List[Task]] = dict()

        self.not_stop = threading.Condition(self.mutex)
        self.stop = False
        self.stopped_count = 0

    def _get_tasks(self, worker_index: int, filter: Optional[Callable[[Task], bool]] = None):
        """
        Retrieve tasks from the queue for a worker thread.

        Args:
            worker_index: Index of the worker thread
            filter: Optional filter function for task selection

        Returns:
            List of tasks assigned to the worker
        """
        if self.stop:
            return True

        if filter is None:
            tasks = self.tasks[: self.max_task_batch_size]
            del self.tasks[: self.max_task_batch_size]
            self.running_tasks[worker_index] = tasks
            return tasks

        tasks = []
        for i, task in enumerate(self.tasks):
            if not filter(task):
                continue
            tasks.append((i, task))
            if len(tasks) >= self.max_task_batch_size:
                break

        for i, _ in reversed(tasks):
            del self.tasks[i]
        tasks = [task for _, task in tasks]
        self.running_tasks[worker_index] = tasks
        return tasks

    def _worker(self, worker_index: int):
        """
        Worker thread main loop.

        Args:
            worker_index: Index of the worker thread
        """
        with self.mutex:
            self.running_tasks[worker_index] = []

        task_filter = None
        task_filer_size = 0 if self.task_filters is None else len(self.task_filters)
        if task_filer_size > 0:
            task_filter = self.task_filters[worker_index % task_filer_size]

        while True:
            with self.tasks_not_empty:
                tasks = self.tasks_not_empty.wait_for(functools.partial(self._get_tasks, worker_index, task_filter))

                if self.stop:
                    self.stopped_count += 1
                    if self.stopped_count == len(self.pool):
                        self.not_stop.notify_all()
                    return

            results = []
            try:
                results = self.work(tasks)
            except Exception as e:
                scheduler_logger.error(f"Worker {self.name} execute error: {e}, traceback: {traceback.format_exc()}")
                continue

            if results is not None and len(results) > 0:
                with self.mutex:
                    self.results += results
                    self.results_not_empty.notify_all()

    def start(self, workers: int):
        """
        Start the worker threads.

        Args:
            workers: Number of worker threads to start
        """
        with self.mutex:
            remain = workers - len(self.pool)
            if remain <= 0:
                return

            for _ in range(remain):
                index = len(self.pool)
                t = threading.Thread(target=self._worker, args=(index,), daemon=True)
                t.start()
                self.pool.append(t)

    def terminate(self):
        """
        Gracefully shutdown all worker threads.

        Waits for all threads to complete current tasks before stopping.
        """
        with self.mutex:
            self.stop = True
            self.tasks_not_empty.notify_all()
            self.results_not_empty.notify_all()

            self.not_stop.wait_for(lambda: self.stopped_count == len(self.pool))

            self.pool = []
            self.tasks = []
            self.results = []
            self.running_tasks = dict()
            self.stop = False
            self.stopped_count = 0

    def get_results(self, max_size: int, timeout: float) -> List[Task]:
        """
        Retrieve processed task results.

        Args:
            max_size: Maximum number of results to retrieve
            timeout: Maximum wait time in seconds

        Returns:
            List of completed tasks/results
        """

        def _get_results():
            if self.stop:
                return True
            results = self.results[:max_size]
            del self.results[:max_size]
            return results

        with self.results_not_empty:
            results = self.results_not_empty.wait_for(_get_results, timeout)
            if self.stop:
                return []
            return results

    def add_tasks(self, tasks: List[Task], unique: bool = False):
        """
        Add new tasks to the worker pool.

        Args:
            tasks: List of tasks to add
            unique: If True, only adds tasks with unique IDs
        """
        if len(tasks) == 0:
            return

        with self.mutex:
            if not unique:
                self.tasks += tasks
            else:
                task_set = set([t.id for t in self.tasks])
                for _, running in self.running_tasks.items():
                    task_set.update([t.id for t in running])

                for task in tasks:
                    if task.id in task_set:
                        continue
                    self.tasks.append(task)
            self.tasks_not_empty.notify_all()
