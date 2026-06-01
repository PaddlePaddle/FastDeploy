"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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


class TestIPCConnectorInit(unittest.TestCase):
    """Tests for IPCConnector.__init__."""

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.get_data_ptr_ipc")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_init_basic_dtype(self, mock_logger, mock_get_data_ptr_ipc, mock_paddle):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCConnector,
        )

        mock_get_data_ptr_ipc.return_value = 12345
        mock_paddle.ones.return_value = MagicMock()
        mock_stream = MagicMock()
        mock_paddle.device.Stream.return_value = mock_stream

        connector = IPCConnector(rank_id_=0, remote_gpu_id_=1, layer_num=3, local_gpu_id_=0, cache_dtype="bfloat16")

        self.assertEqual(connector.rank_id, 0)
        self.assertEqual(connector.remote_gpu_id, 1)
        self.assertEqual(connector.local_gpu_id, 0)
        self.assertEqual(connector.cache_dtype, "bfloat16")
        self.assertEqual(len(connector.remote_key_tensor_ptr_list), 3)
        self.assertEqual(len(connector.remote_value_tensor_ptr_list), 3)
        self.assertEqual(len(connector.remote_key_scale_tensor_ptr_list), 0)
        self.assertEqual(len(connector.remote_value_scale_tensor_ptr_list), 0)
        self.assertEqual(connector.write_stream, mock_stream)
        mock_paddle.device.Stream.assert_called_once_with("gpu:0")

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.get_data_ptr_ipc")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_init_block_wise_fp8_dtype(self, mock_logger, mock_get_data_ptr_ipc, mock_paddle):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCConnector,
        )

        mock_get_data_ptr_ipc.return_value = 99999
        mock_paddle.ones.return_value = MagicMock()
        mock_paddle.device.Stream.return_value = MagicMock()

        connector = IPCConnector(
            rank_id_=2, remote_gpu_id_=3, layer_num=2, local_gpu_id_=1, cache_dtype="block_wise_fp8"
        )

        self.assertEqual(connector.cache_dtype, "block_wise_fp8")
        self.assertEqual(len(connector.remote_key_tensor_ptr_list), 2)
        self.assertEqual(len(connector.remote_value_tensor_ptr_list), 2)
        self.assertEqual(len(connector.remote_key_scale_tensor_ptr_list), 2)
        self.assertEqual(len(connector.remote_value_scale_tensor_ptr_list), 2)
        # 2 layers * 4 calls (key, value, key_scale, val_scale) = 8
        self.assertEqual(mock_get_data_ptr_ipc.call_count, 8)

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.get_data_ptr_ipc")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_init_zero_layers(self, mock_logger, mock_get_data_ptr_ipc, mock_paddle):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCConnector,
        )

        mock_paddle.ones.return_value = MagicMock()
        mock_paddle.device.Stream.return_value = MagicMock()

        connector = IPCConnector(rank_id_=0, remote_gpu_id_=0, layer_num=0, local_gpu_id_=0, cache_dtype="bfloat16")

        self.assertEqual(len(connector.remote_key_tensor_ptr_list), 0)
        self.assertEqual(len(connector.remote_value_tensor_ptr_list), 0)
        mock_get_data_ptr_ipc.assert_not_called()


class TestIPCCommManagerInit(unittest.TestCase):
    """Tests for IPCCommManager.__init__."""

    def test_init_stores_attributes(self):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCCommManager,
        )

        key_tensors = [MagicMock(), MagicMock()]
        value_tensors = [MagicMock(), MagicMock()]
        key_scales = [MagicMock(), MagicMock()]
        value_scales = [MagicMock(), MagicMock()]

        manager = IPCCommManager(
            rank_id_=1,
            gpu_idx_=2,
            local_key_cache_tensor_list=key_tensors,
            local_value_cache_tensor_list=value_tensors,
            local_key_cache_scale_list=key_scales,
            local_value_cache_scale_list=value_scales,
            cache_dtype="bfloat16",
        )

        self.assertEqual(manager.rank_id, 1)
        self.assertEqual(manager.gpu_idx, 2)
        self.assertEqual(manager.cache_dtype, "bfloat16")
        self.assertEqual(manager.local_key_cache_tensor_list, key_tensors)
        self.assertEqual(manager.local_value_cache_tensor_list, value_tensors)
        self.assertEqual(manager.layer_num, 2)
        self.assertEqual(manager.local_key_cache_scale_list, key_scales)
        self.assertEqual(manager.local_value_cache_scale_list, value_scales)
        self.assertEqual(manager.comm_map, {})


class TestIPCCommManagerConnect(unittest.TestCase):
    """Tests for IPCCommManager.connect and is_connected."""

    def _make_manager(self):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCCommManager,
        )

        return IPCCommManager(
            rank_id_=0,
            gpu_idx_=0,
            local_key_cache_tensor_list=[MagicMock()],
            local_value_cache_tensor_list=[MagicMock()],
            local_key_cache_scale_list=[],
            local_value_cache_scale_list=[],
            cache_dtype="bfloat16",
        )

    def test_is_connected_false_initially(self):
        manager = self._make_manager()
        self.assertFalse(manager.is_connected(0))
        self.assertFalse(manager.is_connected(1))

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.IPCConnector")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_connect_creates_connector(self, mock_logger, mock_connector_cls):
        manager = self._make_manager()
        mock_connector_cls.return_value = MagicMock()

        result = manager.connect(remote_gpu_id_=1)

        self.assertTrue(result)
        self.assertTrue(manager.is_connected(1))
        mock_connector_cls.assert_called_once_with(0, 1, 1, 0, "bfloat16")

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.IPCConnector")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_connect_already_connected_returns_true(self, mock_logger, mock_connector_cls):
        manager = self._make_manager()
        mock_connector_cls.return_value = MagicMock()

        manager.connect(remote_gpu_id_=2)
        result = manager.connect(remote_gpu_id_=2)

        self.assertTrue(result)
        # Only one IPCConnector should be created
        mock_connector_cls.assert_called_once()

    def test_is_connected_true_after_manual_insert(self):
        manager = self._make_manager()
        manager.comm_map[5] = MagicMock()
        self.assertTrue(manager.is_connected(5))


class TestIPCCommManagerWriteCache(unittest.TestCase):
    """Tests for IPCCommManager.write_cache."""

    def _make_manager(self, cache_dtype="bfloat16"):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCCommManager,
        )

        key_tensors = [MagicMock(), MagicMock()]
        value_tensors = [MagicMock(), MagicMock()]
        key_scales = [MagicMock(), MagicMock()]
        value_scales = [MagicMock(), MagicMock()]

        return IPCCommManager(
            rank_id_=0,
            gpu_idx_=0,
            local_key_cache_tensor_list=key_tensors,
            local_value_cache_tensor_list=value_tensors,
            local_key_cache_scale_list=key_scales,
            local_value_cache_scale_list=value_scales,
            cache_dtype=cache_dtype,
        )

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.ipc_sent_key_value_cache_by_remote_ptr")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_write_cache_basic(self, mock_logger, mock_paddle, mock_ipc_send):
        manager = self._make_manager(cache_dtype="bfloat16")

        # Pre-insert a mock connector
        mock_comm = MagicMock()
        mock_comm.remote_gpu_id = 1
        mock_comm.remote_key_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.remote_value_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.write_stream = MagicMock()
        mock_comm.write_stream.stream_base.cuda_stream = 42
        manager.comm_map[1] = mock_comm

        # Mock stream_guard as context manager
        mock_paddle.device.stream_guard.return_value.__enter__ = MagicMock()
        mock_paddle.device.stream_guard.return_value.__exit__ = MagicMock(return_value=False)

        result = manager.write_cache(
            ip="192.168.1.1",
            remote_gpu_id=1,
            local_block_ids=[0, 1, 2],
            remote_block_ids=[3, 4, 5],
            layer_idx=0,
        )

        self.assertEqual(result, 0)
        mock_ipc_send.assert_called_once()
        call_args = mock_ipc_send.call_args
        self.assertEqual(call_args[0][6], 3)  # block_num
        self.assertEqual(call_args[0][7], 0)  # gpu_idx
        self.assertEqual(call_args[0][8], 1)  # remote_gpu_id

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.ipc_sent_key_value_cache_by_remote_ptr")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_write_cache_fp8_sends_scales(self, mock_logger, mock_paddle, mock_ipc_send):
        manager = self._make_manager(cache_dtype="block_wise_fp8")

        mock_comm = MagicMock()
        mock_comm.remote_gpu_id = 2
        mock_comm.remote_key_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.remote_value_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.remote_key_scale_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.remote_value_scale_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.write_stream = MagicMock()
        mock_comm.write_stream.stream_base.cuda_stream = 99
        manager.comm_map[2] = mock_comm

        mock_paddle.device.stream_guard.return_value.__enter__ = MagicMock()
        mock_paddle.device.stream_guard.return_value.__exit__ = MagicMock(return_value=False)

        result = manager.write_cache(
            ip="10.0.0.1",
            remote_gpu_id=2,
            local_block_ids=[0],
            remote_block_ids=[1],
            layer_idx=1,
        )

        self.assertEqual(result, 0)
        # Called twice: once for cache, once for scales
        self.assertEqual(mock_ipc_send.call_count, 2)
        # Second call should have is_scale=True
        second_call_args = mock_ipc_send.call_args_list[1]
        self.assertTrue(second_call_args[0][10])  # is_scale=True

    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.IPCConnector")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.ipc_sent_key_value_cache_by_remote_ptr")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_write_cache_auto_connects(self, mock_logger, mock_paddle, mock_ipc_send, mock_connector_cls):
        manager = self._make_manager(cache_dtype="bfloat16")

        mock_comm = MagicMock()
        mock_comm.remote_gpu_id = 3
        mock_comm.remote_key_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.remote_value_tensor_ptr_list = [MagicMock(), MagicMock()]
        mock_comm.write_stream = MagicMock()
        mock_comm.write_stream.stream_base.cuda_stream = 7
        mock_connector_cls.return_value = mock_comm

        mock_paddle.device.stream_guard.return_value.__enter__ = MagicMock()
        mock_paddle.device.stream_guard.return_value.__exit__ = MagicMock(return_value=False)

        # Not connected yet — should auto-connect
        self.assertFalse(manager.is_connected(3))

        result = manager.write_cache(
            ip="10.0.0.1",
            remote_gpu_id=3,
            local_block_ids=[0, 1],
            remote_block_ids=[2, 3],
            layer_idx=0,
        )

        self.assertEqual(result, 0)
        mock_connector_cls.assert_called_once()
        self.assertTrue(manager.is_connected(3))


class TestIPCCommManagerWriteBlockBySync(unittest.TestCase):
    """Tests for IPCCommManager.write_block_by_sync."""

    @patch(
        "fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.ipc_sent_key_value_cache_by_remote_ptr_block_sync"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.paddle")
    @patch("fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer.logger")
    def test_write_block_by_sync(self, mock_logger, mock_paddle, mock_block_sync):
        from fastdeploy.cache_manager.transfer_factory.ipc_cache_transfer import (
            IPCCommManager,
        )

        key_tensors = [MagicMock()]
        value_tensors = [MagicMock()]

        manager = IPCCommManager(
            rank_id_=0,
            gpu_idx_=1,
            local_key_cache_tensor_list=key_tensors,
            local_value_cache_tensor_list=value_tensors,
            local_key_cache_scale_list=[],
            local_value_cache_scale_list=[],
            cache_dtype="bfloat16",
        )

        mock_comm = MagicMock()
        mock_comm.write_stream.stream_base.cuda_stream = 55
        manager.comm_map[2] = mock_comm

        manager.write_block_by_sync(remote_gpu_id=2)

        mock_paddle.set_device.assert_called_once_with("gpu:1")
        mock_block_sync.assert_called_once_with(
            key_tensors[0],
            value_tensors[0],
            55,
        )


if __name__ == "__main__":
    unittest.main()
