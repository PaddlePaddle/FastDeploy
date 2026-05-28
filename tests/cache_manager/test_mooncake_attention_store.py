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

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the attentionstore_sdk before importing the module under test
mock_common_pb2 = MagicMock()
mock_common_pb2.MEDIA_HBM = 1

mock_sdk_module = MagicMock()
mock_tokens_cls = MagicMock()
mock_attention_store_sdk_cls = MagicMock()
mock_attention_store_sdk_error = type("AttentionStoreSDKError", (Exception,), {})
mock_attention_type = MagicMock()
mock_attention_type.MHA = "MHA"

sys.modules["attentionstore_sdk"] = MagicMock()
sys.modules["attentionstore_sdk.api"] = MagicMock()
sys.modules["attentionstore_sdk.api.common"] = MagicMock()
sys.modules["attentionstore_sdk.api.common.common_pb2"] = mock_common_pb2
sys.modules["attentionstore_sdk.sdk"] = MagicMock(
    AttentionStoreSDK=mock_attention_store_sdk_cls,
    Tokens=mock_tokens_cls,
)
sys.modules["attentionstore_sdk.utils"] = MagicMock()
sys.modules["attentionstore_sdk.utils.err"] = MagicMock(
    AttentionStoreSDKError=mock_attention_store_sdk_error,
)
sys.modules["attentionstore_sdk.client"] = MagicMock()
sys.modules["attentionstore_sdk.client.client"] = MagicMock(
    AttentionType=mock_attention_type,
)


class TestAttentionStoreConfig(unittest.TestCase):
    """Tests for AttentionStoreConfig dataclass."""

    def test_default_values(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStoreConfig,
        )

        config = AttentionStoreConfig()
        self.assertEqual(config.namespace, "default_ns")
        self.assertEqual(config.pod_name, "default_pod")
        self.assertEqual(config.model_version, "v0")
        self.assertEqual(config.shard_id, 0)
        self.assertEqual(config.shard_num, 1)
        self.assertEqual(config.layer_num, 1)
        self.assertEqual(config.block_token_size, 64)
        self.assertEqual(config.bytes_per_shard_layer_per_block, 1024)
        self.assertEqual(config.device_id, 0)
        self.assertEqual(config.dp_id, 0)
        self.assertEqual(config.splitwise_role, "mixed")

    def test_custom_values(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStoreConfig,
        )

        config = AttentionStoreConfig(
            namespace="ns1",
            pod_name="pod1",
            model_version="v2",
            shard_id=1,
            shard_num=4,
            layer_num=32,
            block_token_size=128,
            bytes_per_shard_layer_per_block=2048,
            device_id=3,
            dp_id=2,
            splitwise_role="decode",
        )
        self.assertEqual(config.namespace, "ns1")
        self.assertEqual(config.pod_name, "pod1")
        self.assertEqual(config.layer_num, 32)
        self.assertEqual(config.splitwise_role, "decode")


class TestAttentionStoreInit(unittest.TestCase):
    """Tests for AttentionStore.__init__."""

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict(
        "os.environ",
        {"AS_NAMESPACE": "test_ns", "AS_POD_NAME": "test_pod", "AS_MODEL_VERSION": "v3", "ENABLE_EP_DP_IN_FD": "1"},
    )
    def test_init_cuda_platform(self, mock_logger, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=2, block_token_size=64, device_id=0, dp_id=1, splitwise_role="prefill")

        self.assertEqual(store.config.namespace, "test_ns")
        self.assertEqual(store.config.pod_name, "test_pod_prefill_1")
        self.assertEqual(store.config.model_version, "v3")
        self.assertEqual(store.sdk, mock_sdk_instance)
        mock_attention_store_sdk_cls.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "0"})
    def test_init_non_cuda_platform(self, mock_logger, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = False
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=4, block_token_size=32, dp_id=2)

        # When ENABLE_EP_DP_IN_FD=0, pod_name should not be modified
        self.assertEqual(store.config.pod_name, "default_pod")
        self.assertEqual(store.sdk, mock_sdk_instance)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store._ATTENTIONSTORE_AVAILABLE", False)
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    def test_init_sdk_not_available_raises(self, mock_logger):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        with self.assertRaises(ImportError) as ctx:
            AttentionStore()
        self.assertIn("attentionstore_sdk", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_init_sdk_raises_propagates(self, mock_logger, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_attention_store_sdk_cls.side_effect = RuntimeError("connection refused")

        with self.assertRaises(RuntimeError):
            AttentionStore(layer_num=1)

        mock_attention_store_sdk_cls.side_effect = None


class TestAttentionStoreWaitForSdkReady(unittest.TestCase):
    """Tests for AttentionStore.wait_for_sdk_ready."""

    def _make_store(self, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=1, block_token_size=64)

        store.sdk = mock_sdk_instance
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_wait_ready_immediate_success(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.match.return_value = 0

        # Should not raise
        store.wait_for_sdk_ready(timeout=10, delta_t=1)
        store.sdk.match.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.time.sleep")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_wait_ready_retries_on_cuda_not_ready(self, mock_logger, mock_platform, mock_sleep):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()

        # Fail twice with "cuda memory not ready", then succeed
        store.sdk.match.side_effect = [
            mock_attention_store_sdk_error("cuda memory not ready"),
            mock_attention_store_sdk_error("cuda memory not ready"),
            0,
        ]

        store.wait_for_sdk_ready(timeout=30, delta_t=5)
        self.assertEqual(store.sdk.match.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.time.sleep")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_wait_ready_timeout(self, mock_logger, mock_platform, mock_sleep):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()

        # Always fail with "cuda memory not ready"
        store.sdk.match.side_effect = mock_attention_store_sdk_error("cuda memory not ready")

        with self.assertRaises(TimeoutError) as ctx:
            store.wait_for_sdk_ready(timeout=10, delta_t=5)
        self.assertIn("timed out", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_wait_ready_unexpected_error_raises(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()

        store.sdk.match.side_effect = mock_attention_store_sdk_error("some other error")

        with self.assertRaises(RuntimeError) as ctx:
            store.wait_for_sdk_ready(timeout=30, delta_t=5)
        self.assertIn("Unexpected exception", str(ctx.exception))


class TestAttentionStoreReadWrite(unittest.TestCase):
    """Tests for AttentionStore.read and .write methods."""

    def _make_store(self, mock_platform, is_cuda=True):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = is_cuda
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=2, block_token_size=64)

        store.sdk = mock_sdk_instance
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_read_cuda(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=True)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.read.return_value = 3

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=100)), MagicMock(data_ptr=MagicMock(return_value=200))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=300)), MagicMock(data_ptr=MagicMock(return_value=400))]

        result = store.read(
            task_id="task1",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[1, 2, 3],
            gpu_block_ids=[0, 1],
            start_read_block_idx=0,
            timeout=10.0,
        )

        self.assertEqual(result, 3)
        store.sdk.read.assert_called_once()
        call_kwargs = store.sdk.read.call_args
        # On CUDA, remote_addrs=None is passed
        self.assertIn("remote_addrs", call_kwargs.kwargs or {}) or self.assertIsNone(
            call_kwargs[1].get("remote_addrs")
        )

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_read_non_cuda(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=False)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.read.return_value = 2

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=100))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=200))]

        result = store.read(
            task_id="task2",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[4, 5],
            gpu_block_ids=[2],
            start_read_block_idx=1,
        )

        self.assertEqual(result, 2)
        store.sdk.read.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_read_sdk_error_returns_zero(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=True)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.read.side_effect = mock_attention_store_sdk_error("read failed")

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=100))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=200))]

        result = store.read(
            task_id="task_err",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[1],
            gpu_block_ids=[0],
            start_read_block_idx=0,
        )

        self.assertEqual(result, 0)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_write_cuda(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=True)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.write.return_value = 5

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=10))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=20))]

        result = store.write(
            task_id="w_task1",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[10, 11, 12],
            gpu_block_ids=[0, 1, 2],
            start_write_block_idx=0,
            timeout=5.0,
        )

        self.assertEqual(result, 5)
        store.sdk.write.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_write_non_cuda(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=False)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.write.return_value = 4

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=10))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=20))]

        result = store.write(
            task_id="w_task2",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[7, 8],
            gpu_block_ids=[3],
            start_write_block_idx=1,
        )

        self.assertEqual(result, 4)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_write_sdk_error_returns_zero(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform, is_cuda=True)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.write.side_effect = mock_attention_store_sdk_error("write failed")

        key_cache = [MagicMock(data_ptr=MagicMock(return_value=10))]
        val_cache = [MagicMock(data_ptr=MagicMock(return_value=20))]

        result = store.write(
            task_id="w_err",
            key_cache=key_cache,
            val_cache=val_cache,
            token_ids=[1],
            gpu_block_ids=[0],
            start_write_block_idx=0,
        )

        self.assertEqual(result, 0)


class TestAttentionStoreQuery(unittest.TestCase):
    """Tests for AttentionStore.query."""

    def _make_store(self, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=1, block_token_size=64)

        store.sdk = mock_sdk_instance
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_query_success(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.match.return_value = 7

        result = store.query(task_id="q1", token_ids=[1, 2, 3], start_match_block_idx=0, timeout=5.0)

        self.assertEqual(result, 7)
        store.sdk.match.assert_called_once()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_query_sdk_error_returns_zero(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.match.side_effect = mock_attention_store_sdk_error("match error")

        result = store.query(task_id="q_err", token_ids=[1], start_match_block_idx=0)

        self.assertEqual(result, 0)


class TestAttentionStoreFlushTokenIndex(unittest.TestCase):
    """Tests for AttentionStore.flush_token_index."""

    def _make_store(self, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_sdk_instance = MagicMock()
        mock_attention_store_sdk_cls.return_value = mock_sdk_instance

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=2, block_token_size=64)

        store.sdk = mock_sdk_instance
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_flush_reside_in_gpu_true(self, mock_logger, mock_platform):
        import fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store as mod

        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()

        store.flush_token_index(task_id="f1", token_ids=[1, 2, 3], start_block_idx=0, reside_in_gpu=True)

        store.sdk.flush_token_index.assert_called_once()
        call_args = store.sdk.flush_token_index.call_args[0]
        # reside_in_gpu=True: (layers, tokens, start_idx, None, MEDIA_HBM)
        self.assertEqual(call_args[0], [0, 1])
        self.assertIsNone(call_args[3])
        self.assertEqual(call_args[4], mod.common_pb2.MEDIA_HBM)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_flush_reside_in_gpu_false(self, mock_logger, mock_platform):
        import fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store as mod

        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()

        store.flush_token_index(task_id="f2", token_ids=[4, 5], start_block_idx=1, reside_in_gpu=False)

        store.sdk.flush_token_index.assert_called_once()
        call_args = store.sdk.flush_token_index.call_args[0]
        # reside_in_gpu=False: (layers, tokens, start_idx, MEDIA_HBM, None)
        self.assertEqual(call_args[3], mod.common_pb2.MEDIA_HBM)
        self.assertIsNone(call_args[4])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_flush_sdk_error_handled(self, mock_logger, mock_platform):
        store = self._make_store(mock_platform)
        mock_tokens_cls.return_value = MagicMock()
        store.sdk.flush_token_index.side_effect = mock_attention_store_sdk_error("flush error")

        # Should not raise
        store.flush_token_index(task_id="f_err", token_ids=[1], start_block_idx=0, reside_in_gpu=True)


class TestAttentionStoreUnsupportedMethods(unittest.TestCase):
    """Tests for methods that raise NotImplementedError."""

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.current_platform")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store.logger")
    @patch.dict("os.environ", {"ENABLE_EP_DP_IN_FD": "1"})
    def test_unsupported_methods(self, mock_logger, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.attention_store import (
            AttentionStore,
        )

        mock_platform.is_cuda.return_value = True
        mock_attention_store_sdk_cls.return_value = MagicMock()

        with patch.object(AttentionStore, "wait_for_sdk_ready"):
            store = AttentionStore(layer_num=1, block_token_size=64)

        with self.assertRaises(NotImplementedError):
            store.get()
        with self.assertRaises(NotImplementedError):
            store.batch_get()
        with self.assertRaises(NotImplementedError):
            store.set()
        with self.assertRaises(NotImplementedError):
            store.batch_set()
        with self.assertRaises(NotImplementedError):
            store.exists(["key1"])
        with self.assertRaises(NotImplementedError):
            store.clear()
        with self.assertRaises(NotImplementedError):
            store.register_buffer(0, 0)


if __name__ == "__main__":
    unittest.main()
