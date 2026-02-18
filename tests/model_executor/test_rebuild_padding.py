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
import sys
import types

# We need to mock modules BEFORE importing fastdeploy.model_executor.pre_and_post_process
# to control the platform detection and import behavior.

class TestRebuildPadding(unittest.TestCase):
    def setUp(self):
        # Create a mock for fastdeploy.platforms.current_platform
        self.mock_platform = MagicMock()

        # Setup basic module structure mocks
        self.mock_paddle = MagicMock()
        sys.modules["paddle"] = self.mock_paddle

        self.mock_numpy = MagicMock()
        sys.modules["numpy"] = self.mock_numpy

        # Mock fastdeploy and its submodules
        self.mock_fastdeploy = types.ModuleType("fastdeploy")
        self.mock_fastdeploy.platforms = types.ModuleType("fastdeploy.platforms")
        self.mock_fastdeploy.platforms.current_platform = self.mock_platform
        self.mock_fastdeploy.envs = types.ModuleType("fastdeploy.envs")
        self.mock_fastdeploy.envs.FD_DISABLED_RECOVER = "0"
        self.mock_fastdeploy.config = types.ModuleType("fastdeploy.config")
        self.mock_fastdeploy.config.SpeculativeConfig = MagicMock()

        self.mock_fastdeploy.worker = types.ModuleType("fastdeploy.worker")
        self.mock_fastdeploy.worker.input_batch = types.ModuleType("fastdeploy.worker.input_batch")
        self.mock_fastdeploy.worker.input_batch.InputBatch = MagicMock()

        self.mock_fastdeploy.model_executor = types.ModuleType("fastdeploy.model_executor")
        self.mock_fastdeploy.model_executor.entropy_utils = types.ModuleType("fastdeploy.model_executor.entropy_utils")
        self.mock_fastdeploy.model_executor.layers = types.ModuleType("fastdeploy.model_executor.layers")
        self.mock_fastdeploy.model_executor.layers.sample = types.ModuleType("fastdeploy.model_executor.layers.sample")
        self.mock_fastdeploy.model_executor.layers.sample.meta_data = types.ModuleType("fastdeploy.model_executor.layers.sample.meta_data")
        self.mock_fastdeploy.output = types.ModuleType("fastdeploy.output")
        self.mock_fastdeploy.output.pooler = types.ModuleType("fastdeploy.output.pooler")
        self.mock_fastdeploy.output.stream_transfer_data = types.ModuleType("fastdeploy.output.stream_transfer_data")
        self.mock_fastdeploy.worker.output = types.ModuleType("fastdeploy.worker.output")

        # Ops mocks
        self.mock_gpu_ops = types.ModuleType("fastdeploy.model_executor.ops.gpu")
        self.mock_cpu_ops = types.ModuleType("fastdeploy.model_executor.ops.cpu")
        self.mock_iluvatar_ops = types.ModuleType("fastdeploy.model_executor.ops.iluvatar")
        self.mock_gcu_ops = types.ModuleType("fastdeploy.model_executor.ops.gcu")

        sys.modules["fastdeploy"] = self.mock_fastdeploy
        sys.modules["fastdeploy.platforms"] = self.mock_fastdeploy.platforms
        sys.modules["fastdeploy.envs"] = self.mock_fastdeploy.envs
        sys.modules["fastdeploy.config"] = self.mock_fastdeploy.config
        sys.modules["fastdeploy.worker"] = self.mock_fastdeploy.worker
        sys.modules["fastdeploy.worker.input_batch"] = self.mock_fastdeploy.worker.input_batch
        sys.modules["fastdeploy.model_executor"] = self.mock_fastdeploy.model_executor
        sys.modules["fastdeploy.model_executor.ops"] = types.ModuleType("fastdeploy.model_executor.ops")
        sys.modules["fastdeploy.model_executor.ops.gpu"] = self.mock_gpu_ops
        sys.modules["fastdeploy.model_executor.ops.cpu"] = self.mock_cpu_ops
        sys.modules["fastdeploy.model_executor.ops.iluvatar"] = self.mock_iluvatar_ops
        sys.modules["fastdeploy.model_executor.ops.gcu"] = self.mock_gcu_ops

    def tearDown(self):
        # Clean up modules
        if "fastdeploy.model_executor.pre_and_post_process" in sys.modules:
            del sys.modules["fastdeploy.model_executor.pre_and_post_process"]

    def _load_module(self):
        import fastdeploy.model_executor.pre_and_post_process as pnp
        # Reset the global variable to None to force re-initialization
        pnp._rebuild_padding_impl = None
        return pnp

    def test_rebuild_padding_cuda(self):
        self.mock_platform.is_cuda.return_value = True
        self.mock_platform.is_maca.return_value = False
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_dcu.return_value = False
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_cpu.return_value = False

        mock_impl = MagicMock()
        self.mock_gpu_ops.rebuild_padding = mock_impl

        pnp = self._load_module()

        # Test call
        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        self.assertEqual(len(mock_impl.call_args[0]), 9)

    def test_rebuild_padding_cpu(self):
        self.mock_platform.is_cuda.return_value = False
        self.mock_platform.is_maca.return_value = False
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_dcu.return_value = False
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_cpu.return_value = True

        mock_impl = MagicMock()
        self.mock_cpu_ops.rebuild_padding_cpu = mock_impl

        pnp = self._load_module()

        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        # Wrapper should call with 6 args
        self.assertEqual(len(mock_impl.call_args[0]), 6)

    def test_rebuild_padding_dcu(self):
        self.mock_platform.is_cuda.return_value = False
        self.mock_platform.is_maca.return_value = False
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_dcu.return_value = True
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_cpu.return_value = False

        mock_impl = MagicMock()
        self.mock_gpu_ops.rebuild_padding = mock_impl

        pnp = self._load_module()

        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        self.assertEqual(len(mock_impl.call_args[0]), 6)

    def test_rebuild_padding_gcu(self):
        self.mock_platform.is_cuda.return_value = False
        self.mock_platform.is_maca.return_value = False
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_dcu.return_value = False
        self.mock_platform.is_gcu.return_value = True
        self.mock_platform.is_cpu.return_value = False

        mock_impl = MagicMock()
        self.mock_gcu_ops.rebuild_padding = mock_impl

        pnp = self._load_module()

        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        self.assertEqual(len(mock_impl.call_args[0]), 6)

    def test_rebuild_padding_iluvatar(self):
        self.mock_platform.is_cuda.return_value = False
        self.mock_platform.is_maca.return_value = False
        self.mock_platform.is_iluvatar.return_value = True
        self.mock_platform.is_dcu.return_value = False
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_cpu.return_value = False

        mock_impl = MagicMock()
        self.mock_iluvatar_ops.rebuild_padding = mock_impl

        pnp = self._load_module()

        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        self.assertEqual(len(mock_impl.call_args[0]), 9)

    def test_rebuild_padding_maca(self):
        self.mock_platform.is_cuda.return_value = False
        self.mock_platform.is_maca.return_value = True
        self.mock_platform.is_iluvatar.return_value = False
        self.mock_platform.is_dcu.return_value = False
        self.mock_platform.is_gcu.return_value = False
        self.mock_platform.is_cpu.return_value = False

        mock_impl = MagicMock()
        self.mock_gpu_ops.rebuild_padding = mock_impl

        pnp = self._load_module()

        args = [MagicMock() for _ in range(9)]
        pnp.rebuild_padding(*args)

        mock_impl.assert_called_once()
        self.assertEqual(len(mock_impl.call_args[0]), 9)

if __name__ == "__main__":
    unittest.main()
