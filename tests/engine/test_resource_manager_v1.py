# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from unittest.mock import Mock

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request, RequestStatus
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestResourceManagerV1(unittest.TestCase):
    """Test cases for ResourceManagerV1."""

    def setUp(self):
        """Set up test fixtures."""
        engine_args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=8192,
            tensor_parallel_size=1,
            engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")),
            cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")),
        )
        # Create and start the engine service
        mock_config = engine_args.create_engine_config()

        self.manager = ResourceManagerV1(
            max_num_seqs=4,
            config=mock_config,
            tensor_parallel_size=1,
            splitwise_role="mixed",
            local_data_parallel_id=0,
        )

        # Mock cache manager
        self.manager.cache_manager = Mock()
        self.manager.cache_manager.free_blocks = Mock()

    def tearDown(self) -> None:
        self.manager.need_block_num_signal.clear()

    def test_preempted_all_with_no_running_requests(self):
        """Test preempted_all with no running requests."""
        self.assertEqual(len(self.manager.running), 0)
        preempted_reqs = self.manager.preempted_all()
        self.assertEqual(len(preempted_reqs), 0)

    def test_preempted_all_with_normal_requests(self):
        """Test preempted_all with normal running requests."""
        # Add mock running requests
        req1 = Mock(spec=Request)
        req1.request_id = "req1"
        req1.use_extend_tables = False
        req1.status = RequestStatus.RUNNING
        req1.block_tables = [1, 2, 3]
        req1.num_cached_blocks = 0
        req1.idx = 0

        req2 = Mock(spec=Request)
        req2.request_id = "req2"
        req2.use_extend_tables = False
        req2.status = RequestStatus.RUNNING
        req2.block_tables = [4, 5]
        req2.num_cached_blocks = 0
        req2.idx = 1

        self.manager.running = [req1, req2]

        preempted_reqs = self.manager.preempted_all()

        # Verify
        self.assertEqual(len(preempted_reqs), 2)
        self.assertEqual(preempted_reqs[0].request_id, "req2")
        self.assertEqual(preempted_reqs[1].request_id, "req1")

        # Verify request status changed
        self.assertEqual(req1.status, RequestStatus.PREEMPTED)
        self.assertEqual(req2.status, RequestStatus.PREEMPTED)

        # Verify added to to_be_rescheduled_request_id_set
        self.assertIn("req1", self.manager.to_be_rescheduled_request_id_set)
        self.assertIn("req2", self.manager.to_be_rescheduled_request_id_set)

        self.assertEqual(len(self.manager.running), 0)
        self.assertEqual(len(self.manager.waiting), 0)
        self.assertEqual(len(self.manager.to_be_rescheduled_request_id_set), 2)


if __name__ == "__main__":
    unittest.main()
