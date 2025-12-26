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
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.distributed import communication


class TestCommunicationBasic(unittest.TestCase):
    def setUp(self):
        communication._TP_AR = None

    def test_capture_custom_allreduce_no_tp_ar(self):
        with communication.capture_custom_allreduce():
            pass

    def test_capture_custom_allreduce_with_tp_ar(self):
        mock_tp_ar = MagicMock()
        mock_context = MagicMock()
        mock_tp_ar.capture.return_value = mock_context
        communication._TP_AR = mock_tp_ar
        with communication.capture_custom_allreduce():
            pass
        mock_tp_ar.capture.assert_called_once()

    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    @patch("fastdeploy.distributed.custom_all_reduce.CustomAllreduce")
    def test_use_custom_allreduce(self, mock_custom_ar, mock_get_hcg):
        mock_hcg = MagicMock()
        mock_get_hcg.return_value = mock_hcg

        # fake group with required attributes used by CustomAllreduce
        fake_group = MagicMock()
        fake_group.rank = 0
        fake_group.world_size = 2
        mock_hcg.get_model_parallel_group.return_value = fake_group

        communication.use_custom_allreduce()

        self.assertIsNotNone(communication._TP_AR)
        mock_custom_ar.assert_called_once_with(fake_group, 8192 * 1024)

    def test_custom_ar_clear_ipc_handles(self):
        mock_tp_ar = MagicMock()
        communication._TP_AR = mock_tp_ar
        communication.custom_ar_clear_ipc_handles()
        mock_tp_ar.clear_ipc_handles.assert_called_once()

    @patch("fastdeploy.distributed.communication.dist.all_reduce")
    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_tensor_model_parallel_all_reduce(self, mock_get_hcg, mock_all_reduce):
        # ensure group exists
        mock_hcg = MagicMock()
        mock_get_hcg.return_value = mock_hcg
        fake_group = MagicMock()
        fake_group.world_size = 2
        mock_hcg.get_model_parallel_group.return_value = fake_group

        # make all_reduce callable
        def fake_all_reduce(x, group=None):
            return x

        mock_all_reduce.side_effect = fake_all_reduce

        x = paddle.to_tensor([1.0])
        # call should not raise, ensure all_reduce was invoked
        _ = communication.tensor_model_parallel_all_reduce(x)
        mock_all_reduce.assert_called()

    @patch("fastdeploy.distributed.communication.stream.all_reduce")
    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_tensor_model_parallel_all_reduce_custom(self, mock_get_hcg, mock_stream_ar):
        # ensure group exists
        mock_hcg = MagicMock()
        mock_get_hcg.return_value = mock_hcg
        fake_group = MagicMock()
        fake_group.world_size = 2
        mock_hcg.get_model_parallel_group.return_value = fake_group

        # stream.all_reduce may not return value in source; ensure callable
        def fake_stream_all_reduce(x, **kwargs):
            return None

        mock_stream_ar.side_effect = fake_stream_all_reduce

        x = paddle.to_tensor([2.0])
        # the function does not return input_ in source; ensure call succeeds and stream.all_reduce used
        _ = communication.tensor_model_parallel_all_reduce_custom(x)
        mock_stream_ar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
