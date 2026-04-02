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
from unittest.mock import Mock

from fastdeploy.config import FDConfig
from fastdeploy.worker.gpu_worker import GpuWorker


class TestGpuWorkerSleepWakeup(unittest.TestCase):
    """Test cases for GpuWorker sleep and wakeup methods - Coverage for lines 201, 205"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_fd_config = Mock(spec=FDConfig)
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1

    def test_sleep_delegates_to_model_runner(self):
        """Test sleep method delegates to model_runner (line 201)"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call sleep
        worker.sleep(tags="weight")

        # Verify model_runner.sleep was called
        worker.model_runner.sleep.assert_called_once_with(tags="weight")

    def test_sleep_with_multiple_tags(self):
        """Test sleep with multiple tags"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call sleep with multiple tags
        worker.sleep(tags="weight,kv_cache")

        # Verify model_runner.sleep was called with correct tags
        worker.model_runner.sleep.assert_called_once_with(tags="weight,kv_cache")

    def test_sleep_with_kwargs(self):
        """Test sleep passes kwargs to model_runner"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call sleep with kwargs
        worker.sleep(tags="weight", force=True, timeout=100)

        # Verify model_runner.sleep was called with kwargs
        worker.model_runner.sleep.assert_called_once_with(tags="weight", force=True, timeout=100)

    def test_wakeup_delegates_to_model_runner(self):
        """Test wakeup method delegates to model_runner (line 205)"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call wakeup
        worker.wakeup(tags="weight")

        # Verify model_runner.wakeup was called
        worker.model_runner.wakeup.assert_called_once_with(tags="weight")

    def test_wakeup_with_multiple_tags(self):
        """Test wakeup with multiple tags"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call wakeup with multiple tags
        worker.wakeup(tags="weight,kv_cache")

        # Verify model_runner.wakeup was called with correct tags
        worker.model_runner.wakeup.assert_called_once_with(tags="weight,kv_cache")

    def test_wakeup_with_kwargs(self):
        """Test wakeup passes kwargs to model_runner"""
        worker = GpuWorker.__new__(GpuWorker)
        worker.model_runner = Mock()

        # Call wakeup with kwargs
        worker.wakeup(tags="kv_cache", async_load=True)

        # Verify model_runner.wakeup was called with kwargs
        worker.model_runner.wakeup.assert_called_once_with(tags="kv_cache", async_load=True)


if __name__ == "__main__":
    unittest.main()
