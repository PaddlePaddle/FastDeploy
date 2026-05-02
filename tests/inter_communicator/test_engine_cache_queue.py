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

import unittest
from unittest.mock import MagicMock

from fastdeploy.inter_communicator.engine_cache_queue import EngineCacheQueue


class TestEngineCacheQueueLogger(unittest.TestCase):
    """Simple tests to cover logger.debug lines in EngineCacheQueue"""

    def _create_mock_queue(self):
        """Create a mock EngineCacheQueue instance without starting server"""
        queue = object.__new__(EngineCacheQueue)
        queue.task_lock = MagicMock()
        queue.task_done_lock = MagicMock()
        queue.task_sync_value = MagicMock()
        queue.task_sync_value.get.return_value = 0
        queue.transfer_task_queue = []
        queue.tansfer_done_queue = []
        queue.total_num = 1
        queue.position = 1
        queue.client_id = 0
        return queue

    def test_put_transfer_task_logs_debug(self):
        """Cover line 289: logger.debug in put_transfer_task"""
        queue = self._create_mock_queue()
        queue.put_transfer_task("test_item")
        self.assertIn("test_item", queue.transfer_task_queue)

    def test_get_transfer_task_logs_debug(self):
        """Cover lines 305, 307: logger.debug in get_transfer_task"""
        queue = self._create_mock_queue()
        queue.transfer_task_queue.append("task1")
        data, read_finish = queue.get_transfer_task()
        self.assertEqual(data, "task1")
        self.assertTrue(read_finish)

    def test_clear_transfer_task_logs_debug(self):
        """Cover line 326: logger.debug in clear_transfer_task"""
        queue = self._create_mock_queue()
        queue.transfer_task_queue.append("task1")
        queue.clear_transfer_task()
        self.assertEqual(len(queue.transfer_task_queue), 0)

    def test_put_transfer_done_signal_logs_debug(self):
        """Cover line 336: logger.debug in put_transfer_done_signal"""
        queue = self._create_mock_queue()
        queue.put_transfer_done_signal(("data", "task_id"))
        self.assertEqual(queue.tansfer_done_queue[0], ("data", "task_id"))

    def test_get_transfer_done_signal_logs_debug(self):
        """Cover line 346: logger.debug in get_transfer_done_signal"""
        queue = self._create_mock_queue()
        queue.tansfer_done_queue.append(("data", "task_id"))
        data = queue.get_transfer_done_signal()
        self.assertEqual(data, ("data", "task_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
