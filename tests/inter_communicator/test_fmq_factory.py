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

import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.inter_communicator.fmq_factory import FMQFactory


class TestFMQFactory(unittest.TestCase):

    def setUp(self):
        # ✅ patch 被测模块中的 FMQ
        patcher = patch("fastdeploy.inter_communicator.fmq_factory.FMQ")
        self.addCleanup(patcher.stop)
        self.mock_fmq_cls = patcher.start()

        # 每次 FMQ() 返回的实例
        self.mock_fmq_instance = MagicMock()
        self.mock_fmq_cls.return_value = self.mock_fmq_instance

    # ------------------------------
    # API -> Engine
    # ------------------------------
    def test_q_a2e_producer(self):
        FMQFactory.q_a2e_producer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_a2e", role="producer")

    def test_q_a2e_consumer(self):
        FMQFactory.q_a2e_consumer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_a2e", role="consumer")

    # ------------------------------
    # Engine -> Worker
    # ------------------------------
    def test_q_e2w_producer(self):
        FMQFactory.q_e2w_producer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_e2w", role="producer")

    def test_q_e2w_consumer(self):
        FMQFactory.q_e2w_consumer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_e2w", role="consumer")

    # ------------------------------
    # Worker -> Engine
    # ------------------------------
    def test_q_w2e_producer_with_name(self):
        FMQFactory.q_w2e_producer("worker1")

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_w2e_worker1", role="producer")

    def test_q_w2e_consumer_with_name(self):
        FMQFactory.q_w2e_consumer("worker2")

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_w2e_worker2", role="consumer")

    # ------------------------------
    # Engine -> API
    # ------------------------------
    def test_q_e2a_producer(self):
        FMQFactory.q_e2a_producer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_e2a", role="producer")

    def test_q_e2a_consumer(self):
        FMQFactory.q_e2a_consumer()

        self.mock_fmq_cls.assert_called_once()
        self.mock_fmq_instance.queue.assert_called_once_with("q_e2a", role="consumer")


if __name__ == "__main__":
    unittest.main()
