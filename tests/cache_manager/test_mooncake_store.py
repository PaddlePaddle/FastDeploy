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

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Mock mooncake.store before importing the module under test
mock_mooncake_store_module = MagicMock()
sys.modules["mooncake"] = MagicMock()
sys.modules["mooncake.store"] = mock_mooncake_store_module


class TestByteToGb(unittest.TestCase):
    """Tests for byte_to_gb helper."""

    def test_byte_to_gb(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            byte_to_gb,
        )

        self.assertEqual(byte_to_gb(1024 * 1024 * 1024), 1.0)
        self.assertEqual(byte_to_gb(0), 0.0)
        self.assertAlmostEqual(byte_to_gb(512 * 1024 * 1024), 0.5)


class TestMooncakeStoreConfigCreate(unittest.TestCase):
    """Tests for MooncakeStoreConfig.create()."""

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.current_platform")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_rdma_nics", return_value="mlx5_0"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict(
        "os.environ",
        {
            "MOONCAKE_METADATA_SERVER": "meta:2379",
            "MOONCAKE_MASTER_SERVER_ADDR": "master:8080",
        },
        clear=False,
    )
    def test_create_from_env_vars(self, mock_logger, mock_rdma_nics, mock_host_ip, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        mock_platform.is_cuda.return_value = True
        # Remove MOONCAKE_CONFIG_PATH if set
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)

        config = MooncakeStoreConfig.create()

        self.assertEqual(config.local_hostname, "10.0.0.1")
        self.assertEqual(config.metadata_server, "meta:2379")
        self.assertEqual(config.master_server_addr, "master:8080")
        self.assertEqual(config.protocol, "rdma")
        # rdma_devices empty -> auto-detect
        self.assertEqual(config.rdma_devices, "mlx5_0")

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.current_platform")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.2"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict(
        "os.environ",
        {
            "MOONCAKE_LOCAL_HOSTNAME": "custom_host",
            "MOONCAKE_METADATA_SERVER": "meta:2379",
            "MOONCAKE_MASTER_SERVER_ADDR": "master:8080",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "2048",
            "MOONCAKE_LOCAL_BUFFER_SIZE": "512",
            "MOONCAKE_PROTOCOL": "tcp",
            "MOONCAKE_RDMA_DEVICES": "mlx5_1,mlx5_2",
        },
        clear=False,
    )
    def test_create_from_env_vars_custom(self, mock_logger, mock_host_ip, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        mock_platform.is_cuda.return_value = True
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)

        config = MooncakeStoreConfig.create()

        self.assertEqual(config.local_hostname, "custom_host")
        self.assertEqual(config.global_segment_size, 2048)
        self.assertEqual(config.local_buffer_size, 512)
        self.assertEqual(config.protocol, "tcp")
        self.assertEqual(config.rdma_devices, "mlx5_1,mlx5_2")

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.current_platform")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.3"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_create_from_json_file(self, mock_logger, mock_host_ip, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        mock_platform.is_cuda.return_value = False

        config_data = {
            "local_hostname": "json_host",
            "metadata_server": "json_meta:2379",
            "master_server_addr": "json_master:8080",
            "global_segment_size": 4096,
            "local_buffer_size": 256,
            "protocol": "tcp",
            "rdma_devices": "mlx5_bond",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch.dict("os.environ", {"MOONCAKE_CONFIG_PATH": config_path}):
                config = MooncakeStoreConfig.create()

            self.assertEqual(config.local_hostname, "json_host")
            self.assertEqual(config.metadata_server, "json_meta:2379")
            self.assertEqual(config.master_server_addr, "json_master:8080")
            self.assertEqual(config.global_segment_size, 4096)
            self.assertEqual(config.local_buffer_size, 256)
            self.assertEqual(config.protocol, "tcp")
            self.assertEqual(config.rdma_devices, "mlx5_bond")
        finally:
            os.unlink(config_path)

    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MOONCAKE_CONFIG_PATH": "/nonexistent/path.json"})
    def test_create_file_not_found(self, mock_logger, mock_host_ip):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        with self.assertRaises(FileNotFoundError):
            MooncakeStoreConfig.create()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.current_platform")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MOONCAKE_METADATA_SERVER": "meta:2379"}, clear=False)
    def test_create_missing_master_server_raises(self, mock_logger, mock_host_ip, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        mock_platform.is_cuda.return_value = False
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
        os.environ.pop("MOONCAKE_MASTER_SERVER_ADDR", None)

        with self.assertRaises(ValueError) as ctx:
            MooncakeStoreConfig.create()
        self.assertIn("must be provided", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.current_platform")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict(
        "os.environ",
        {
            "MOONCAKE_LOCAL_HOSTNAME": "localhost",
            "MOONCAKE_METADATA_SERVER": "meta:2379",
            "MOONCAKE_MASTER_SERVER_ADDR": "master:8080",
            "MOONCAKE_RDMA_DEVICES": "mlx5_0",
        },
        clear=False,
    )
    def test_create_localhost_raises(self, mock_logger, mock_host_ip, mock_platform):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        mock_platform.is_cuda.return_value = False
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)

        with self.assertRaises(ValueError) as ctx:
            MooncakeStoreConfig.create()
        self.assertIn("localhost", str(ctx.exception))

    def test_select_rdma_device(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig(
            local_hostname="host",
            metadata_server="meta",
            global_segment_size=1024,
            local_buffer_size=256,
            protocol="rdma",
            rdma_devices="mlx5_0,mlx5_1,mlx5_2",
            master_server_addr="master",
        )

        config.select_rdma_device(tp_rank=0)
        self.assertEqual(config.rdma_devices, "mlx5_0")

        # Reset
        config.rdma_devices = "mlx5_0,mlx5_1,mlx5_2"
        config.select_rdma_device(tp_rank=1)
        self.assertEqual(config.rdma_devices, "mlx5_1")

        config.rdma_devices = "mlx5_0,mlx5_1,mlx5_2"
        config.select_rdma_device(tp_rank=4)  # 4 % 3 = 1
        self.assertEqual(config.rdma_devices, "mlx5_1")


class TestMooncakeStoreInit(unittest.TestCase):
    """Tests for MooncakeStore.__init__."""

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": "0"}, clear=False)
    def test_init_default_mr_size(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            DEFAULT_MC_MAX_MR_SIZE,
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with patch.object(MooncakeStore, "warmup"):
            store = MooncakeStore(tp_rank=None)

        self.assertEqual(store.mc_max_mr_size, DEFAULT_MC_MAX_MR_SIZE)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(512 * 1024 * 1024)}, clear=False)
    def test_init_mr_size_below_min(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MIN_MC_MAX_MR_SIZE,
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with patch.object(MooncakeStore, "warmup"):
            store = MooncakeStore()

        self.assertEqual(store.mc_max_mr_size, MIN_MC_MAX_MR_SIZE)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(8 * 1024 * 1024 * 1024)}, clear=False)
    def test_init_mr_size_above_max(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MAX_MC_MAX_MR_SIZE,
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with patch.object(MooncakeStore, "warmup"):
            store = MooncakeStore()

        self.assertEqual(store.mc_max_mr_size, MAX_MC_MAX_MR_SIZE)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(2 * 1024 * 1024 * 1024)}, clear=False)
    def test_init_mr_size_within_range(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with patch.object(MooncakeStore, "warmup"):
            store = MooncakeStore()

        self.assertEqual(store.mc_max_mr_size, 2 * 1024 * 1024 * 1024)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}, clear=False)
    def test_init_with_tp_rank_selects_rdma(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with patch.object(MooncakeStore, "warmup"):
            MooncakeStore(tp_rank=2)

        mock_config.select_rdma_device.assert_called_once_with(2)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}, clear=False)
    def test_init_setup_failure_raises(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=-1))

        with self.assertRaises(RuntimeError) as ctx:
            MooncakeStore()
        self.assertIn("failed to setup mooncake store", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig")
    @patch(
        "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip", return_value="10.0.0.1"
    )
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    @patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}, clear=False)
    def test_init_local_buffer_exceeds_mr_raises(self, mock_logger, mock_host_ip, mock_config_cls):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 8 * 1024 * 1024 * 1024  # larger than max_mr
        mock_config_cls.create.return_value = mock_config
        mock_mooncake_store_module.MooncakeDistributedStore.return_value = MagicMock(setup=MagicMock(return_value=0))

        with self.assertRaises(ValueError) as ctx:
            MooncakeStore()
        self.assertIn("local_buffer_size", str(ctx.exception))


class TestMooncakeStoreWarmup(unittest.TestCase):
    """Tests for MooncakeStore.warmup."""

    def _make_store(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024

        with (
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig"
            ) as mock_cfg_cls,
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip",
                return_value="10.0.0.1",
            ),
            patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger"),
            patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}),
            patch.object(MooncakeStore, "warmup"),
        ):
            mock_cfg_cls.create.return_value = mock_config
            mock_distributed_store = MagicMock(setup=MagicMock(return_value=0))
            mock_mooncake_store_module.MooncakeDistributedStore.return_value = mock_distributed_store
            store = MooncakeStore()
            store.store = mock_distributed_store
        return store

    def test_warmup_success(self):
        store = self._make_store()
        store.store.put.return_value = 0
        store.store.is_exist.return_value = 1
        store.store.get.return_value = bytes(1024)
        store.store.remove.return_value = 0

        # Should not raise
        store.warmup()
        store.store.put.assert_called_once()
        store.store.is_exist.assert_called_once()
        store.store.get.assert_called_once()
        store.store.remove.assert_called_once()

    def test_warmup_put_failure(self):
        store = self._make_store()
        store.store.put.return_value = -1

        with self.assertRaises(AssertionError):
            store.warmup()

    def test_warmup_exist_failure(self):
        store = self._make_store()
        store.store.put.return_value = 0
        store.store.is_exist.return_value = 0

        with self.assertRaises(AssertionError):
            store.warmup()


class TestMooncakeStoreRegisterBuffer(unittest.TestCase):
    """Tests for MooncakeStore.register_buffer."""

    def _make_store(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024

        with (
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig"
            ) as mock_cfg_cls,
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip",
                return_value="10.0.0.1",
            ),
            patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger"),
            patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}),
            patch.object(MooncakeStore, "warmup"),
        ):
            mock_cfg_cls.create.return_value = mock_config
            mock_distributed_store = MagicMock(setup=MagicMock(return_value=0))
            mock_mooncake_store_module.MooncakeDistributedStore.return_value = mock_distributed_store
            store = MooncakeStore()
            store.store = mock_distributed_store
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_register_small_buffer(self, mock_logger):
        store = self._make_store()
        store.store.register_buffer.return_value = 0

        store.register_buffer(buffer_ptr=0x1000, buffer_size=1024)
        store.store.register_buffer.assert_called_once_with(0x1000, 1024)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_register_large_buffer_splits_into_chunks(self, mock_logger):
        store = self._make_store()
        store.mc_max_mr_size = 1024  # small max for testing
        store.store.register_buffer.return_value = 0

        store.register_buffer(buffer_ptr=0x1000, buffer_size=2500)

        # 2500 / 1024 = 3 chunks (1024, 1024, 452)
        self.assertEqual(store.store.register_buffer.call_count, 3)
        calls = store.store.register_buffer.call_args_list
        self.assertEqual(calls[0][0], (0x1000, 1024))
        self.assertEqual(calls[1][0], (0x1000 + 1024, 1024))
        self.assertEqual(calls[2][0], (0x1000 + 2048, 452))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_register_buffer_type_error(self, mock_logger):
        store = self._make_store()
        store.store.register_buffer.side_effect = TypeError("invalid ptr")

        with self.assertRaises(TypeError) as ctx:
            store.register_buffer(buffer_ptr=0x1000, buffer_size=1024)
        self.assertIn("Mooncake Store Register Buffer Error", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_register_buffer_assertion_error(self, mock_logger):
        store = self._make_store()
        store.store.register_buffer.return_value = -1

        with self.assertRaises(AssertionError):
            store.register_buffer(buffer_ptr=0x1000, buffer_size=1024)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_register_large_buffer_chunk_type_error(self, mock_logger):
        store = self._make_store()
        store.mc_max_mr_size = 1024
        store.store.register_buffer.side_effect = TypeError("chunk error")

        with self.assertRaises(TypeError) as ctx:
            store.register_buffer(buffer_ptr=0x1000, buffer_size=2048)
        self.assertIn("Mooncake Store Register Buffer Error", str(ctx.exception))


class TestMooncakeStoreBatchSetGet(unittest.TestCase):
    """Tests for MooncakeStore.batch_set and batch_get."""

    def _make_store(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024

        with (
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig"
            ) as mock_cfg_cls,
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip",
                return_value="10.0.0.1",
            ),
            patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger"),
            patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}),
            patch.object(MooncakeStore, "warmup"),
        ):
            mock_cfg_cls.create.return_value = mock_config
            mock_distributed_store = MagicMock(setup=MagicMock(return_value=0))
            mock_mooncake_store_module.MooncakeDistributedStore.return_value = mock_distributed_store
            store = MooncakeStore()
            store.store = mock_distributed_store
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_set_length_mismatch(self, mock_logger):
        store = self._make_store()

        with self.assertRaises(ValueError) as ctx:
            store.batch_set(keys=["k1", "k2"], target_locations=[1], target_sizes=[10, 20])
        self.assertIn("must match", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_set_empty_keys(self, mock_logger):
        store = self._make_store()

        with self.assertRaises(ValueError) as ctx:
            store.batch_set(keys=[], target_locations=[], target_sizes=[])
        self.assertIn("greater than zero", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_set_success(self, mock_logger):
        store = self._make_store()
        store.store.batch_put_from.return_value = [0, 0]

        result = store.batch_set(keys=["k1", "k2"], target_locations=[100, 200], target_sizes=[10, 20])
        self.assertEqual(result, [0, 0])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_get_length_mismatch(self, mock_logger):
        store = self._make_store()

        with self.assertRaises(ValueError) as ctx:
            store.batch_get(keys=["k1"], target_locations=[1, 2], target_sizes=[10])
        self.assertIn("must match", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_get_empty_keys(self, mock_logger):
        store = self._make_store()

        with self.assertRaises(ValueError) as ctx:
            store.batch_get(keys=[], target_locations=[], target_sizes=[])
        self.assertIn("greater than zero", str(ctx.exception))

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_batch_get_success(self, mock_logger):
        store = self._make_store()
        store.store.batch_get_into.return_value = [10, 20]

        result = store.batch_get(keys=["k1", "k2"], target_locations=[100, 200], target_sizes=[10, 20])
        self.assertEqual(result, [10, 20])


class TestMooncakeStoreExistsQueryDeleteClear(unittest.TestCase):
    """Tests for exists, query, delete, close, clear."""

    def _make_store(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024

        with (
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig"
            ) as mock_cfg_cls,
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip",
                return_value="10.0.0.1",
            ),
            patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger"),
            patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}),
            patch.object(MooncakeStore, "warmup"),
        ):
            mock_cfg_cls.create.return_value = mock_config
            mock_distributed_store = MagicMock(setup=MagicMock(return_value=0))
            mock_mooncake_store_module.MooncakeDistributedStore.return_value = mock_distributed_store
            store = MooncakeStore()
            store.store = mock_distributed_store
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_exists(self, mock_logger):
        store = self._make_store()
        store.store.batch_is_exist.return_value = [True, False, True]

        result = store.exists(["k1", "k2", "k3"])
        self.assertEqual(result, {"k1": True, "k2": False, "k3": True})

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_query_no_scale(self, mock_logger):
        store = self._make_store()
        # All exist
        store.store.batch_is_exist.return_value = [True, True, True, True]

        result = store.query(k_keys=["k1", "k2"], v_keys=["v1", "v2"])
        self.assertEqual(result, 2)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_query_no_scale_partial_match(self, mock_logger):
        store = self._make_store()
        # k1, k2, v1, v2 — v2 not found
        store.store.batch_is_exist.return_value = [True, True, True, False]

        result = store.query(k_keys=["k1", "k2"], v_keys=["v1", "v2"])
        # Only first pair fully matches, second breaks
        self.assertEqual(result, 1)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_query_with_scale(self, mock_logger):
        store = self._make_store()
        # k1, v1, ks1, vs1 — all exist
        store.store.batch_is_exist.return_value = [True, True, True, True]

        result = store.query(
            k_keys=["k1"],
            v_keys=["v1"],
            k_scale_keys=["ks1"],
            v_scale_keys=["vs1"],
        )
        self.assertEqual(result, 1)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_query_with_scale_missing(self, mock_logger):
        store = self._make_store()
        # k1, v1, ks1, vs1 — vs1 not found
        store.store.batch_is_exist.return_value = [True, True, True, False]

        result = store.query(
            k_keys=["k1"],
            v_keys=["v1"],
            k_scale_keys=["ks1"],
            v_scale_keys=["vs1"],
        )
        self.assertEqual(result, 0)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.time.sleep")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_delete_success_first_try(self, mock_logger, mock_sleep):
        store = self._make_store()
        store.store.remove.return_value = 0

        result = store.delete("key1", timeout=3)
        self.assertTrue(result)
        mock_sleep.assert_not_called()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.time.sleep")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_delete_retries_then_succeeds(self, mock_logger, mock_sleep):
        store = self._make_store()
        store.store.remove.side_effect = [-1, -1, 0]

        result = store.delete("key2", timeout=5)
        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.time.sleep")
    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_delete_timeout(self, mock_logger, mock_sleep):
        store = self._make_store()
        store.store.remove.return_value = -1

        result = store.delete("key3", timeout=2)
        self.assertFalse(result)

    def test_close(self):
        store = self._make_store()
        # close is a no-op, should not raise
        store.close()

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_clear(self, mock_logger):
        store = self._make_store()
        store.store.remove_all.return_value = 5

        result = store.clear()
        self.assertTrue(result)
        store.store.remove_all.assert_called_once()


class TestMooncakeStorePutGetBatchImpl(unittest.TestCase):
    """Tests for _put_batch_zero_copy_impl and _get_batch_zero_copy_impl."""

    def _make_store(self):
        from fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        mock_config = MagicMock()
        mock_config.local_buffer_size = 1024

        with (
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.MooncakeStoreConfig"
            ) as mock_cfg_cls,
            patch(
                "fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.get_host_ip",
                return_value="10.0.0.1",
            ),
            patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger"),
            patch.dict("os.environ", {"MC_MAX_MR_SIZE": str(4 * 1024 * 1024 * 1024)}),
            patch.object(MooncakeStore, "warmup"),
        ):
            mock_cfg_cls.create.return_value = mock_config
            mock_distributed_store = MagicMock(setup=MagicMock(return_value=0))
            mock_mooncake_store_module.MooncakeDistributedStore.return_value = mock_distributed_store
            store = MooncakeStore()
            store.store = mock_distributed_store
        return store

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_put_batch_all_success(self, mock_logger):
        store = self._make_store()
        store.store.batch_put_from.return_value = [0, 0, 0]

        result = store._put_batch_zero_copy_impl(["k1", "k2", "k3"], [100, 200, 300], [10, 20, 30])
        self.assertEqual(result, [0, 0, 0])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_put_batch_partial_failure(self, mock_logger):
        store = self._make_store()
        store.store.batch_put_from.return_value = [0, -1, 0]

        result = store._put_batch_zero_copy_impl(["k1", "k2", "k3"], [100, 200, 300], [10, 20, 30])
        self.assertEqual(result, [0, -1, 0])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_put_batch_exception(self, mock_logger):
        store = self._make_store()
        store.store.batch_put_from.side_effect = RuntimeError("network error")

        with self.assertRaises(RuntimeError):
            store._put_batch_zero_copy_impl(["k1"], [100], [10])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_get_batch_all_success(self, mock_logger):
        store = self._make_store()
        store.store.batch_get_into.return_value = [10, 20]

        result = store._get_batch_zero_copy_impl(["k1", "k2"], [100, 200], [10, 20])
        self.assertEqual(result, [10, 20])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_get_batch_partial_failure(self, mock_logger):
        store = self._make_store()
        store.store.batch_get_into.return_value = [10, -1]

        result = store._get_batch_zero_copy_impl(["k1", "k2"], [100, 200], [10, 20])
        self.assertEqual(result, [10, -1])

    @patch("fastdeploy.cache_manager.transfer_factory.mooncake_store.mooncake_store.logger")
    def test_get_batch_exception(self, mock_logger):
        store = self._make_store()
        store.store.batch_get_into.side_effect = RuntimeError("read error")

        with self.assertRaises(RuntimeError):
            store._get_batch_zero_copy_impl(["k1"], [100], [10])


if __name__ == "__main__":
    unittest.main()
