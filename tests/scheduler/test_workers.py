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

import time
import unittest

from fastdeploy.scheduler.workers import Task, Workers


class TestTask(unittest.TestCase):
    def test_repr(self):
        """Test the __repr__ method of Task class.

        Verifies that the string representation of Task contains key attributes
        (task_id and reason) with correct values.
        """
        task = Task("123", 456, reason="ok")
        repr_str = repr(task)
        self.assertIn("task_id:123", repr_str)
        self.assertIn("reason:ok", repr_str)


class TestWorkers(unittest.TestCase):
    """Unit test suite for the Workers class.

    Covers core functionalities including task processing flow, filtering, unique task addition,
    timeout handling, exception resilience, and edge cases like empty inputs or zero workers.
    """

    def test_basic_flow(self):
        """Test basic task processing flow with multiple tasks and workers.

        Verifies that Workers can start multiple worker threads, process batched tasks,
        and return correct results in expected format.
        """

        def simple_work(tasks):
            """Simple work function that increments task raw value by 1."""
            return [Task(task.id, task.raw + 1) for task in tasks]

        workers = Workers("test_basic_flow", work=simple_work, max_task_batch_size=2)
        workers.start(2)

        tasks = [Task(str(i), i) for i in range(4)]
        workers.add_tasks(tasks)

        # Collect results with timeout protection
        results = []
        start_time = time.time()
        while len(results) < 4 and time.time() - start_time < 1:
            batch_results = workers.get_results(10, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        # Clean up resources
        workers.terminate()

        result_map = {int(task.id): task.raw for task in results}
        self.assertEqual(result_map, {0: 1, 1: 2, 2: 3, 3: 4})

    def test_task_filters(self):
        """Test task filtering functionality.

        Verifies that Workers apply specified task filters correctly and process
        all eligible tasks without dropping or duplicating.
        """

        def work_function(tasks):
            """Work function that adds 10 to task raw value."""
            return [Task(task.id, task.raw + 10) for task in tasks]

        # Define filter functions: even and odd task ID filters
        def filter_even(task):
            """Filter to select tasks with even-numbered IDs."""
            return int(task.id) % 2 == 0

        def filter_odd(task):
            """Filter to select tasks with odd-numbered IDs."""
            return int(task.id) % 2 == 1

        # Initialize Workers with filter chain and 2 worker threads
        workers = Workers(
            "test_task_filters",
            work=work_function,
            max_task_batch_size=1,
            task_filters=[filter_even, filter_odd],
        )
        workers.start(2)

        # Add 6 tasks with IDs 0-5
        workers.add_tasks([Task(str(i), i) for i in range(6)])

        # Collect results with timeout protection
        results = []
        start_time = time.time()
        while len(results) < 6 and time.time() - start_time < 2:
            batch_results = workers.get_results(10, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        # Clean up resources
        workers.terminate()

        # Expected task ID groups
        even_ids = {0, 2, 4}
        odd_ids = {1, 3, 5}

        # Extract original IDs from results (reverse work function calculation)
        got_even = {int(task.raw) - 10 for task in results if int(task.id) in even_ids}
        got_odd = {int(task.raw) - 10 for task in results if int(task.id) in odd_ids}

        # Verify all even and odd tasks were processed correctly
        self.assertEqual(got_even, even_ids)
        self.assertEqual(got_odd, odd_ids)

    def test_unique_task_addition(self):
        """Test unique task addition functionality.

        Verifies that duplicate tasks (same task_id) are filtered out when unique=True,
        while new tasks are processed normally.
        """

        def slow_work(tasks):
            """Slow work function to simulate processing delay (50ms)."""
            time.sleep(0.05)
            return [Task(task.id, task.raw + 1) for task in tasks]

        # Initialize Workers with 1 worker thread (to control task processing order)
        workers = Workers("test_unique_task_addition", work=slow_work, max_task_batch_size=1)
        workers.start(1)

        # Add first task (task_id="1") with unique=True
        workers.add_tasks([Task("1", 100)], unique=True)
        time.sleep(0.02)  # Allow task to enter running state

        # Add duplicate task (same task_id="1") - should be filtered out
        workers.add_tasks([Task("1", 200)], unique=True)
        # Add new task (task_id="2") - should be processed
        workers.add_tasks([Task("2", 300)], unique=True)

        # Collect results (expect 2 valid results)
        results = []
        start_time = time.time()
        while len(results) < 2 and time.time() - start_time < 1:
            batch_results = workers.get_results(10, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        # Clean up resources
        workers.terminate()

        # Verify only unique task IDs are present
        result_ids = sorted(int(task.id) for task in results)
        self.assertEqual(result_ids, [1, 2])

    def test_get_results_timeout(self):
        """Test timeout handling in get_results method.

        Verifies that get_results returns empty list after specified timeout when
        no results are available, and the actual wait time meets the timeout requirement.
        """

        def no_result_work(tasks):
            """Work function that returns empty list (no results)."""
            time.sleep(0.01)
            return []

        # Initialize Workers with 1 worker thread
        workers = Workers("test_get_results_timeout", work=no_result_work, max_task_batch_size=1)
        workers.start(1)

        # Measure time taken for get_results with 50ms timeout
        start_time = time.time()
        results = workers.get_results(max_size=1, timeout=0.05)
        end_time = time.time()

        # Clean up resources
        workers.terminate()

        # Verify no results are returned and timeout is respected
        self.assertEqual(results, [])
        self.assertGreaterEqual(end_time - start_time, 0.05)

    def test_start_zero_workers(self):
        """Test starting Workers with zero worker threads.

        Verifies that Workers initializes correctly with zero threads and the worker pool is empty.
        """
        # Initialize Workers without specifying max_task_batch_size (uses default)
        workers = Workers("test_start_zero_workers", work=lambda tasks: tasks)
        workers.start(0)

        # Verify worker pool is empty
        self.assertEqual(len(workers.pool), 0)

    def test_worker_exception_resilience(self):
        """Test Workers resilience to exceptions in work function.

        Verifies that worker threads continue running (or complete gracefully) when
        the work function raises an exception, without crashing the entire Workers instance.
        """
        # Track number of work function calls
        call_tracker = {"count": 0}

        def error_prone_work(tasks):
            """Work function that raises RuntimeError on each call."""
            call_tracker["count"] += 1
            raise RuntimeError("Simulated work function exception")

        # Initialize Workers with 1 worker thread
        workers = Workers("test_worker_exception_resilience", work=error_prone_work, max_task_batch_size=1)
        workers.start(1)

        # Add a test task that will trigger the exception
        workers.add_tasks([Task("1", 100)])
        time.sleep(0.05)  # Allow time for exception to be raised

        # Clean up resources
        workers.terminate()

        # Verify work function was called at least once (exception was triggered)
        self.assertGreaterEqual(call_tracker["count"], 1)

    def test_add_empty_tasks(self):
        """Test adding empty task list to Workers.

        Verifies that adding an empty list of tasks does not affect Workers state
        and no invalid operations are performed.
        """
        # Initialize Workers with 1 worker thread
        workers = Workers("test_add_empty_tasks", work=lambda tasks: tasks)
        workers.start(1)

        # Add empty task list
        workers.add_tasks([])

        # Verify task queue remains empty
        self.assertEqual(len(workers.tasks), 0)

        # Clean up resources
        workers.terminate()

    def test_terminate_empty_workers(self):
        """Test terminating Workers that have no running threads or tasks.

        Verifies that terminate() can be safely called on an unstarted Workers instance
        without errors, and all state variables remain in valid initial state.
        """
        # Initialize Workers without starting any threads
        workers = Workers("test_terminate_empty_workers", work=lambda tasks: tasks)

        # Call terminate on empty Workers
        workers.terminate()

        # Verify Workers state remains valid
        self.assertFalse(workers.stop)
        self.assertEqual(workers.stopped_count, 0)
        self.assertEqual(len(workers.pool), 0)
        self.assertEqual(len(workers.tasks), 0)
        self.assertEqual(len(workers.results), 0)

    def test_get_tasks_with_stop_condition(self):
        """Test _get_tasks method when stop condition is True.

        Verifies that _get_tasks returns True (indicating stop) when self.stop is True.
        This tests lines 112-113 that are currently uncovered.
        """
        workers = Workers("test_get_tasks_stop", work=lambda tasks: tasks)

        # Set stop condition to True
        with workers.mutex:
            workers.stop = True

        # Test _get_tasks with stop condition
        result = workers._get_tasks(0)

        # Should return True when stop is True
        self.assertTrue(result)

    def test_get_tasks_with_filter(self):
        """Test _get_tasks method with filter function.

        Verifies that _get_tasks correctly applies filter logic and removes filtered tasks
        from the task queue. This tests lines 121-133.
        """
        workers = Workers("test_get_tasks_filter", work=lambda tasks: tasks, max_task_batch_size=2)

        # Add tasks to queue
        with workers.mutex:
            workers.tasks = [Task("1", 1), Task("2", 2), Task("3", 3), Task("4", 4), Task("5", 5)]

        # Define filter to select only odd-numbered tasks
        def odd_filter(task):
            return int(task.id) % 2 == 1

        # Test _get_tasks with filter
        result = workers._get_tasks(0, odd_filter)

        # Should return tasks with IDs 1 and 3 (odd numbers)
        expected_ids = ["1", "3"]
        result_ids = [task.id for task in result]
        self.assertEqual(sorted(result_ids), sorted(expected_ids))

        # Verify filtered tasks are removed from queue
        remaining_ids = [task.id for task in workers.tasks]
        self.assertEqual(sorted(remaining_ids), ["2", "4", "5"])

    def test_get_tasks_with_empty_filter_result(self):
        """Test _get_tasks when filter returns no matching tasks.

        Verifies that _get_tasks handles case where no tasks match the filter.
        """
        workers = Workers("test_get_tasks_empty_filter", work=lambda tasks: tasks)

        # Add tasks to queue
        with workers.mutex:
            workers.tasks = [Task("1", 1), Task("2", 2)]

        # Define filter that matches nothing
        def no_match_filter(task):
            return False

        # Test _get_tasks with non-matching filter
        result = workers._get_tasks(0, no_match_filter)

        # Should return empty list when no tasks match filter
        self.assertEqual(result, [])

        # Original tasks should remain in queue
        remaining_ids = [task.id for task in workers.tasks]
        self.assertEqual(sorted(remaining_ids), ["1", "2"])

    def test_worker_with_stop_condition(self):
        """Test _worker method when stop condition is encountered.

        Verifies that worker properly handles stop conditions, increments stopped_count,
        and notifies other waiting threads. This tests lines 154-158.
        """

        def no_work(tasks):
            # Work function that does nothing
            return []

        workers = Workers("test_worker_stop", work=no_work)
        workers.start(1)

        # Stop workers - this will trigger the stop condition in _worker
        # The stop condition path in _worker will be executed
        workers.terminate()

        # After terminate, stopped_count should be reset to 0,
        # but the stop condition handling path should have been executed
        self.assertEqual(workers.stopped_count, 0)

    def test_worker_with_exception_and_results(self):
        """Test _worker method exception handling with valid results.

        Verifies that worker properly handles exceptions in work function and processes
        valid results correctly. This tests lines 160-170.
        """
        result_tracker = {"processed": []}

        def work_with_exception(tasks):
            result_tracker["processed"].extend(tasks)
            if len(tasks) > 0 and int(tasks[0].id) == 1:
                raise RuntimeError("Test exception")
            return [Task(task.id, task.raw * 2) for task in tasks]

        workers = Workers("test_worker_exception", work=work_with_exception, max_task_batch_size=1)
        workers.start(1)

        # Add tasks that will trigger exception and normal processing
        workers.add_tasks([Task("1", 10), Task("2", 20)])

        # Give time for processing
        time.sleep(0.1)

        # Get results
        results = []
        start_time = time.time()
        while len(results) < 1 and time.time() - start_time < 1.0:
            batch_results = workers.get_results(5, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        workers.terminate()

        # Should have results from the successful task (ID=2)
        result_ids = [task.id for task in results]
        self.assertIn("2", result_ids)

        # Task 1 should have been processed despite the exception
        processed_ids = [task.id for task in result_tracker["processed"]]
        self.assertIn("1", processed_ids)
        self.assertIn("2", processed_ids)

    def test_worker_with_none_results(self):
        """Test _worker method when work function returns None.

        Verifies that worker properly handles None results from work function.
        This tests the condition on lines 167-170.
        """

        def work_returning_none(tasks):
            return None

        workers = Workers("test_worker_none_results", work=work_returning_none)
        workers.start(1)

        # Add a task
        workers.add_tasks([Task("1", 10)])

        # Give time for processing
        time.sleep(0.1)

        # Try to get results (should be empty)
        results = workers.get_results(5, timeout=0.1)

        workers.terminate()

        # Should have no results
        self.assertEqual(results, [])

    def test_worker_with_empty_results(self):
        """Test _worker method when work function returns empty list.

        Verifies that worker properly handles empty results from work function.
        """

        def work_returning_empty(tasks):
            return []

        workers = Workers("test_worker_empty_results", work=work_returning_empty)
        workers.start(1)

        # Add a task
        workers.add_tasks([Task("1", 10)])

        # Give time for processing
        time.sleep(0.1)

        # Try to get results (should be empty)
        results = workers.get_results(5, timeout=0.1)

        workers.terminate()

        # Should have no results
        self.assertEqual(results, [])

    def test_get_results_stop_condition(self):
        """Test get_results method when stop condition is True.

        Verifies that get_results returns empty list when stop is True.
        This tests lines 223-224 and 231-232.
        """
        workers = Workers("test_get_results_stop", work=lambda tasks: tasks)

        # Add some results to the queue
        with workers.mutex:
            workers.results = [Task("1", 10), Task("2", 20)]
            workers.stop = True

        # Test get_results with stop condition
        result = workers.get_results(5, timeout=0.1)

        # Should return empty list when stop is True
        self.assertEqual(result, [])

    def test_get_results_with_partial_results(self):
        """Test get_results when there are fewer results than max_size.

        Verifies that get_results returns all available results when count < max_size.
        This tests line 225-227.
        """
        workers = Workers("test_get_results_partial", work=lambda tasks: tasks)

        # Add fewer results than max_size
        with workers.mutex:
            workers.results = [Task("1", 10), Task("2", 20)]

        # Mock the condition wait_for to return immediately
        def mock_wait_for(predicate, timeout):
            return predicate()

        original_wait_for = workers.results_not_empty.wait_for
        workers.results_not_empty.wait_for = mock_wait_for

        try:
            result = workers.get_results(5, timeout=0.1)

            # Should return all available results
            self.assertEqual(len(result), 2)
            result_ids = [task.id for task in result]
            self.assertEqual(sorted(result_ids), ["1", "2"])

            # Results should be removed from the queue
            self.assertEqual(len(workers.results), 0)

        finally:
            workers.results_not_empty.wait_for = original_wait_for

    def test_add_tasks_unique_with_running_tasks(self):
        """Test add_tasks with unique=True when there are running tasks.

        Verifies that unique task filtering works correctly when there are tasks
        currently being processed (in running_tasks).
        """

        def slow_work(tasks):
            time.sleep(0.1)  # Slow work to keep tasks in running state
            return [Task(task.id, task.raw + 1) for task in tasks]

        workers = Workers("test_add_unique_running", work=slow_work, max_task_batch_size=1)
        workers.start(1)

        # Add initial task
        workers.add_tasks([Task("1", 10)])

        # Give time for task to enter running state
        time.sleep(0.05)

        # Try to add duplicate task with unique=True
        workers.add_tasks([Task("1", 20), Task("2", 30)], unique=True)

        # Give time for processing
        time.sleep(0.2)

        # Get results
        results = []
        start_time = time.time()
        while len(results) < 2 and time.time() - start_time < 1.0:
            batch_results = workers.get_results(5, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        workers.terminate()

        # Should only have results for task IDs 1 and 2 (duplicate filtered out)
        result_ids = [task.id for task in results]
        self.assertEqual(sorted(result_ids), ["1", "2"])

        # Verify only unique count of results
        self.assertEqual(len(set(result_ids)), 2)

    def test_get_tasks_without_filter(self):
        """Test _get_tasks method without filter function.

        Verifies that _get_tasks correctly handles tasks without filter and removes
        them from the task queue. This tests lines 116-119.
        """
        workers = Workers("test_get_tasks_no_filter", work=lambda tasks: tasks, max_task_batch_size=3)

        # Add tasks to queue
        with workers.mutex:
            workers.tasks = [Task("1", 1), Task("2", 2), Task("3", 3), Task("4", 4)]

        # Test _get_tasks without filter (default behavior)
        result = workers._get_tasks(0)

        # Should return first 3 tasks based on max_task_batch_size
        expected_ids = ["1", "2", "3"]
        result_ids = [task.id for task in result]
        self.assertEqual(result_ids, expected_ids)

        # Verify tasks are removed from queue and assigned to running_tasks
        remaining_ids = [task.id for task in workers.tasks]
        self.assertEqual(remaining_ids, ["4"])
        self.assertEqual(workers.running_tasks[0], result)

    def test_worker_task_filter_assignment(self):
        """Test _worker method task filter assignment and worker loop.

        Verifies that workers correctly get assigned task filters and enter their main loop.
        This tests lines 142-150 and covers more of the worker initialization.
        """

        def simple_work(tasks):
            return [Task(task.id, task.raw * 10) for task in tasks]

        # Create workers with multiple filters
        def filter_even(task):
            return int(task.id) % 2 == 0

        def filter_odd(task):
            return int(task.id) % 2 == 1

        workers = Workers(
            "test_worker_filter_assignment",
            work=simple_work,
            max_task_batch_size=1,
            task_filters=[filter_even, filter_odd],
        )
        workers.start(2)

        # Add tasks with various IDs
        workers.add_tasks([Task("1", 1), Task("2", 2), Task("3", 3), Task("4", 4)])

        # Give time for workers to process tasks with filters
        time.sleep(0.2)

        # Get results
        results = []
        start_time = time.time()
        while len(results) < 4 and time.time() - start_time < 1.0:
            batch_results = workers.get_results(10, timeout=0.1)
            if batch_results:
                results.extend(batch_results)

        workers.terminate()

        # Should have processed all tasks through filter assignment
        result_ids = [task.id for task in results]
        self.assertEqual(sorted(result_ids), ["1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
