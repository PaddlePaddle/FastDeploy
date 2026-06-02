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

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from fastdeploy.rl.dynamic_weight_manager import DynamicWeightManager


class TestGetGpuId(unittest.TestCase):
    """Test DynamicWeightManager._get_gpu_id."""

    def _make_manager(self):
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        return mgr

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4", "FLAGS_selected_gpus": "1"})
    def test_returns_correct_gpu_id(self):
        """_get_gpu_id returns the GPU at FLAGS_selected_gpus index in CUDA_VISIBLE_DEVICES."""
        mgr = self._make_manager()
        self.assertEqual(mgr._get_gpu_id(), 3)

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "FLAGS_selected_gpus": "0"})
    def test_returns_first_gpu(self):
        """_get_gpu_id returns first GPU when FLAGS_selected_gpus=0."""
        mgr = self._make_manager()
        self.assertEqual(mgr._get_gpu_id(), 0)

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_when_env_not_set(self):
        """_get_gpu_id returns 0 when env vars not set."""
        mgr = self._make_manager()
        # Defaults: CUDA_VISIBLE_DEVICES="0", FLAGS_selected_gpus="0"
        self.assertEqual(mgr._get_gpu_id(), 0)


class TestValidateParameterMatch(unittest.TestCase):
    """Test DynamicWeightManager._validate_parameter_match."""

    def _make_manager(self):
        return DynamicWeightManager.__new__(DynamicWeightManager)

    def test_valid_match(self):
        """_validate_parameter_match passes with matching shape and dtype."""
        mgr = self._make_manager()
        src = MagicMock(dtype="float32", shape=[10, 20])
        dst = MagicMock(dtype="float32", shape=[10, 20])
        # Should not raise
        mgr._validate_parameter_match("param_name", src, dst)

    def test_dtype_mismatch_raises(self):
        """_validate_parameter_match raises TypeError on dtype mismatch."""
        mgr = self._make_manager()
        src = MagicMock(dtype="float32", shape=[10, 20])
        dst = MagicMock(dtype="float16", shape=[10, 20])
        with self.assertRaises(TypeError) as ctx:
            mgr._validate_parameter_match("weight", src, dst)
        self.assertIn("Type mismatch", str(ctx.exception))
        self.assertIn("weight", str(ctx.exception))

    def test_shape_mismatch_raises(self):
        """_validate_parameter_match raises ValueError on shape mismatch."""
        mgr = self._make_manager()
        src = MagicMock(dtype="float32", shape=[10, 20])
        dst = MagicMock(dtype="float32", shape=[10, 30])
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_parameter_match("bias", src, dst)
        self.assertIn("Shape mismatch", str(ctx.exception))
        self.assertIn("bias", str(ctx.exception))


class TestUpdateModelFromState(unittest.TestCase):
    """Test DynamicWeightManager._update_model_from_state."""

    def _make_manager(self):
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.state_dict = {}
        return mgr

    def test_empty_state_dict_raises(self):
        """_update_model_from_state raises ValueError on empty dict."""
        mgr = self._make_manager()
        with self.assertRaises(ValueError) as ctx:
            mgr._update_model_from_state({}, "test")
        self.assertIn("No parameter found", str(ctx.exception))

    @patch("paddle.no_grad")
    def test_unmatched_param_skipped(self, mock_no_grad):
        """_update_model_from_state skips params not in self.state_dict."""
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()

        mgr = self._make_manager()
        mgr.state_dict = {"existing_param": MagicMock()}

        new_param = MagicMock()
        new_param.stride.return_value = [1]
        new_param.dtype = "float32"
        new_param.shape = [10]

        # "unknown_param" not in state_dict, should be skipped
        mgr._update_model_from_state({"unknown_param": new_param}, "raw")

    @patch("paddle.no_grad")
    def test_matching_stride_shares_buffer(self, mock_no_grad):
        """_update_model_from_state calls _share_buffer_to when strides match."""
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()

        mgr = self._make_manager()
        target_param = MagicMock()
        target_param.stride.return_value = [20, 1]
        target_param.dtype = "float32"
        target_param.shape = [10, 20]

        new_param = MagicMock()
        new_param.stride.return_value = [20, 1]
        new_param.dtype = "float32"
        new_param.shape = [10, 20]

        mgr.state_dict = {"layer.weight": target_param}
        mgr._validate_parameter_match = MagicMock()

        mgr._update_model_from_state({"layer.weight": new_param}, "snapshot")

        new_param._share_buffer_to.assert_called_once_with(target_param)


class TestVerifyParameters(unittest.TestCase):
    """Test DynamicWeightManager._verify_parameters."""

    def _make_manager(self):
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.state_dict = {}
        return mgr

    def test_update_all_initialized_passes(self):
        """_verify_parameters passes when all params are initialized after update."""
        mgr = self._make_manager()
        param = MagicMock()
        param._is_initialized.return_value = True
        mgr.state_dict = {"param1": param, "param2": param}

        # Should not raise
        mgr._verify_parameters("update")

    def test_update_not_initialized_raises(self):
        """_verify_parameters raises RuntimeError when param not initialized after update."""
        mgr = self._make_manager()
        param = MagicMock()
        param._is_initialized.return_value = False
        mgr.state_dict = {"param1": param}

        with self.assertRaises(RuntimeError) as ctx:
            mgr._verify_parameters("update")
        self.assertIn("verification failed", str(ctx.exception))

    def test_clearance_not_initialized_passes(self):
        """_verify_parameters passes when params are NOT initialized after clearance."""
        mgr = self._make_manager()
        param = MagicMock()
        param._is_initialized.return_value = False
        mgr.state_dict = {"param1": param}

        # Should not raise
        mgr._verify_parameters("clearance")

    def test_clearance_still_initialized_raises(self):
        """_verify_parameters raises RuntimeError when param still initialized after clearance."""
        mgr = self._make_manager()
        param = MagicMock()
        param._is_initialized.return_value = True
        mgr.state_dict = {"param1": param}

        with self.assertRaises(RuntimeError) as ctx:
            mgr._verify_parameters("clearance")
        self.assertIn("verification failed", str(ctx.exception))


class TestConvertIpcMetaToTensor(unittest.TestCase):
    """Test DynamicWeightManager._convert_ipc_meta_to_tensor."""

    @patch("paddle.to_tensor")
    @patch("paddle.base.core.LoDTensor._new_shared_cuda")
    @patch.dict(os.environ, {"FLAGS_selected_gpus": "2"})
    def test_converts_meta(self, mock_new_shared, mock_to_tensor):
        """_convert_ipc_meta_to_tensor converts IPC metadata correctly."""
        mock_new_shared.return_value = "raw_tensor"
        mock_to_tensor.return_value = "paddle_tensor"

        # meta format: [str_buffer, ...other_fields..., gpu_id_placeholder]
        # meta[0] gets encoded, meta[6] gets replaced with FLAGS_selected_gpus
        meta = ["buffer_data", 1, 2, 3, 4, 5, 99, 7]
        ipc_meta = {"param_name": meta}

        result = DynamicWeightManager._convert_ipc_meta_to_tensor(ipc_meta)

        self.assertEqual(result, {"param_name": "paddle_tensor"})
        # meta[0] should be encoded to latin-1
        self.assertEqual(meta[0], b"buffer_data")
        # meta[6] should be FLAGS_selected_gpus value
        self.assertEqual(meta[6], 2)
        mock_new_shared.assert_called_once_with(tuple(meta))
        mock_to_tensor.assert_called_once_with("raw_tensor")


class TestFinalizeUpdate(unittest.TestCase):
    """Test DynamicWeightManager.finalize_update."""

    @patch("paddle.distributed.barrier")
    def test_finalize_first_load(self, mock_barrier):
        """finalize_update on first_load does not call _update_shared_status."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True
        mgr.state_dict = {}
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.tensor_parallel_size = 1
        mgr.parallel_config.enable_expert_parallel = False

        mgr._verify_parameters = MagicMock()
        mgr._update_shared_status = MagicMock()

        mgr.finalize_update()

        mgr._verify_parameters.assert_called_once_with("update")
        mgr._update_shared_status.assert_not_called()
        self.assertFalse(mgr.first_load)

    @patch("paddle.distributed.barrier")
    def test_finalize_not_first_load(self, mock_barrier):
        """finalize_update when not first_load calls _update_shared_status."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.state_dict = {}
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.tensor_parallel_size = 1
        mgr.parallel_config.enable_expert_parallel = False

        mgr._verify_parameters = MagicMock()
        mgr._update_shared_status = MagicMock()

        mgr.finalize_update(pid=5)

        mgr._update_shared_status.assert_called_once()

    @patch("paddle.distributed.barrier")
    def test_finalize_with_tp_and_ep(self, mock_barrier):
        """finalize_update calls barrier for both tp and ep groups."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True
        mgr.state_dict = {}
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.tensor_parallel_size = 4
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tp_group = "tp_group"
        mgr.parallel_config.ep_group = "ep_group"

        mgr._verify_parameters = MagicMock()
        mgr._update_shared_status = MagicMock()

        mgr.finalize_update()

        calls = mock_barrier.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][0], "tp_group")
        self.assertEqual(calls[1][0][0], "ep_group")


class TestUpdateSharedStatus(unittest.TestCase):
    """Test DynamicWeightManager._update_shared_status."""

    @patch("fastdeploy.rl.dynamic_weight_manager.SharedMemory")
    def test_updates_status_on_rank_0(self, mock_shm_cls):
        """_update_shared_status writes status when rank == 0."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.rank = 0

        mock_shm = MagicMock()
        mock_shm.buf = bytearray(4)
        mock_shm_cls.return_value = mock_shm

        mgr._update_shared_status(pid=1, status=42)

        mock_shm_cls.assert_called_once_with(create=False, size=4, name="model_weights_status.1")

    @patch("fastdeploy.rl.dynamic_weight_manager.SharedMemory")
    def test_no_write_on_non_zero_rank(self, mock_shm_cls):
        """_update_shared_status does not write when rank != 0."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.rank = 1

        mock_shm = MagicMock()
        mock_shm.buf = bytearray(4)
        mock_shm_cls.return_value = mock_shm

        mgr._update_shared_status(pid=0, status=99)

        # SharedMemory is still opened, but the value should not be set at rank 1
        mock_shm_cls.assert_called_once()


class TestReadModelVersionFromFile(unittest.TestCase):
    """Test DynamicWeightManager.read_model_version_from_file."""

    def _make_manager(self, model_dir):
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = model_dir
        return mgr

    def test_reads_step_from_yaml(self):
        """read_model_version_from_file reads step field from version.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = os.path.join(tmpdir, "version.yaml")
            with open(version_file, "w") as f:
                yaml.dump({"step": 12345}, f)

            mgr = self._make_manager(tmpdir)
            result = mgr.read_model_version_from_file()
            self.assertEqual(result, "12345")

    def test_missing_file_returns_none(self):
        """read_model_version_from_file returns None if file not found."""
        mgr = self._make_manager("/nonexistent/path")
        result = mgr.read_model_version_from_file()
        self.assertIsNone(result)

    def test_missing_step_field_returns_none(self):
        """read_model_version_from_file returns None if step field missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = os.path.join(tmpdir, "version.yaml")
            with open(version_file, "w") as f:
                yaml.dump({"epoch": 5}, f)

            mgr = self._make_manager(tmpdir)
            result = mgr.read_model_version_from_file()
            self.assertIsNone(result)

    def test_non_dict_yaml_returns_none(self):
        """read_model_version_from_file returns None if yaml isn't a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = os.path.join(tmpdir, "version.yaml")
            with open(version_file, "w") as f:
                f.write("- item1\n- item2\n")

            mgr = self._make_manager(tmpdir)
            result = mgr.read_model_version_from_file()
            self.assertIsNone(result)

    def test_invalid_yaml_returns_none(self):
        """read_model_version_from_file returns None for invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            version_file = os.path.join(tmpdir, "version.yaml")
            with open(version_file, "w") as f:
                f.write("{invalid: yaml: content: [}")

            mgr = self._make_manager(tmpdir)
            result = mgr.read_model_version_from_file()
            self.assertIsNone(result)


class TestUpdateParameters(unittest.TestCase):
    """Test DynamicWeightManager.update_parameters."""

    @patch("paddle.device.cuda.empty_cache")
    def test_first_load_ipc_strategy(self, mock_empty_cache):
        """update_parameters calls _update_ipc on first load with ipc strategy."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "ipc"

        mgr._update_ipc = MagicMock()

        mgr.update_parameters()

        mock_empty_cache.assert_called_once()
        mgr._update_ipc.assert_called_once()

    @patch("paddle.device.cuda.empty_cache")
    def test_first_load_ipc_snapshot_strategy(self, mock_empty_cache):
        """update_parameters calls _update_ipc_snapshot with ipc_snapshot strategy."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "ipc_snapshot"

        mgr._update_ipc_snapshot = MagicMock()

        mgr.update_parameters()

        mgr._update_ipc_snapshot.assert_called_once()

    @patch("paddle.device.cuda.empty_cache")
    def test_unsupported_strategy_raises(self, mock_empty_cache):
        """update_parameters raises ValueError for unsupported strategy."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "unknown"

        with self.assertRaises(ValueError) as ctx:
            mgr.update_parameters()
        self.assertIn("Unsupported strategy", str(ctx.exception))

    @patch("paddle.distributed.restart_process_group")
    @patch("paddle.device.cuda.empty_cache")
    def test_not_first_load_with_restart(self, mock_empty_cache, mock_restart):
        """update_parameters restarts process groups when not first_load and restart requested."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.parallel_config.tp_group = "tp_group"
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "ipc"

        mgr._update_ipc = MagicMock()

        mgr.update_parameters(restart_process_group=True)

        # restart_process_group called for default and tp_group
        self.assertEqual(mock_restart.call_count, 2)


class TestRestartCommunicationGroup(unittest.TestCase):
    """Test DynamicWeightManager.restart_communication_group."""

    @patch("paddle.distributed.restart_process_group")
    def test_first_load_does_nothing(self, mock_restart):
        """restart_communication_group does nothing on first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True

        mgr.restart_communication_group()

        mock_restart.assert_not_called()

    @patch("paddle.distributed.restart_process_group")
    def test_not_first_load_restarts(self, mock_restart):
        """restart_communication_group restarts groups when not first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tp_group = "tp"
        mgr.parallel_config.ep_group = "ep"

        mgr.restart_communication_group()

        # default + tp + ep = 3 calls
        self.assertEqual(mock_restart.call_count, 3)


class TestReloadModelWeights(unittest.TestCase):
    """Test DynamicWeightManager.reload_model_weights."""

    def test_first_load_does_nothing(self):
        """reload_model_weights does nothing on first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True

        # Should not raise
        mgr.reload_model_weights()

    def test_not_first_load_calls_handler(self):
        """reload_model_weights calls appropriate handler when not first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "ipc"
        mgr._update_ipc = MagicMock()

        mgr.reload_model_weights()

        mgr._update_ipc.assert_called_once()

    def test_not_first_load_unsupported_raises(self):
        """reload_model_weights raises ValueError for unsupported strategy."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "bad_strategy"

        with self.assertRaises(ValueError) as ctx:
            mgr.reload_model_weights()
        self.assertIn("Unsupported strategy", str(ctx.exception))


class TestClearDeepepBuffer(unittest.TestCase):
    """Test DynamicWeightManager.clear_deepep_buffer."""

    @patch("fastdeploy.model_executor.layers.moe.ep.DeepEPBufferManager")
    def test_clear_deepep_buffer(self, mock_buffer_mgr):
        """clear_deepep_buffer calls DeepEPBufferManager.clear_buffer."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.clear_deepep_buffer()
        mock_buffer_mgr.clear_buffer.assert_called_once()


class TestClearModelWeight(unittest.TestCase):
    """Test DynamicWeightManager.clear_model_weight."""

    def test_clears_all_params(self):
        """clear_model_weight calls _clear_data on all model params."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)

        param1 = MagicMock()
        param2 = MagicMock()
        model = MagicMock()
        model.state_dict.return_value = {"p1": param1, "p2": param2}
        mgr.model_list = [model]

        mgr.clear_model_weight()

        param1._clear_data.assert_called_once()
        param2._clear_data.assert_called_once()


class TestClearCommunicationGroup(unittest.TestCase):
    """Test DynamicWeightManager.clear_communication_group."""

    @patch("paddle.distributed.shutdown_process_group")
    @patch("paddle.distributed.barrier")
    def test_clears_ep_and_tp(self, mock_barrier, mock_shutdown):
        """clear_communication_group shuts down ep and tp groups."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tensor_parallel_size = 4
        mgr.parallel_config.ep_group = "ep_group"
        mgr.parallel_config.tp_group = "tp_group"

        mgr.clear_communication_group()

        self.assertEqual(mock_barrier.call_count, 2)
        self.assertEqual(mock_shutdown.call_count, 2)

    @patch("paddle.distributed.shutdown_process_group")
    @patch("paddle.distributed.barrier")
    def test_no_ep_no_tp(self, mock_barrier, mock_shutdown):
        """clear_communication_group does nothing with ep=False and tp=1."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.parallel_config.tensor_parallel_size = 1

        mgr.clear_communication_group()

        mock_barrier.assert_not_called()
        mock_shutdown.assert_not_called()


class TestCheckModelWeightsStatus(unittest.TestCase):
    """Test DynamicWeightManager.check_model_weights_status."""

    def test_normal_status_returns_immediately(self):
        """check_model_weights_status returns immediately when status is NORMAL."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        model_weights_status = MagicMock()
        model_weights_status.value = [ModelWeightsStatus.NORMAL]
        kv_cache_status = MagicMock()
        model_runner = MagicMock()

        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=True
        )

        model_runner.clear_requests.assert_not_called()
        model_runner.update_parameters.assert_not_called()

    def test_cleared_non_block_returns(self):
        """check_model_weights_status returns on CLEARED when block=False."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        model_weights_status = MagicMock()
        model_weights_status.value = [ModelWeightsStatus.CLEARED]
        kv_cache_status = None
        model_runner = MagicMock()

        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=False
        )

        model_runner.update_parameters.assert_not_called()

    @patch("time.sleep")
    def test_updating_then_normal(self, mock_sleep):
        """check_model_weights_status handles UPDATING -> NORMAL transition."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        # Line 523 logs value[0] first (access 0), then while loop.
        # Access pattern:
        #   logger.info: value[0] (access 0)
        #   outer while: value[0] != NORMAL (access 1) -> True
        #   outer while: block or value[0] != CLEARED (access 2) -> True (block=True)
        #   if value[0] == UPDATING (access 3) -> True
        #   clear_requests, update_parameters
        #   inner while: value[0] != NORMAL (access 4) -> NORMAL -> False (exit)
        #   outer while: value[0] != NORMAL (access 5) -> NORMAL -> False (exit)
        status_sequence = [
            ModelWeightsStatus.UPDATING,  # access 0: logger.info
            ModelWeightsStatus.UPDATING,  # access 1: outer while != NORMAL
            ModelWeightsStatus.UPDATING,  # access 2: block=True so short-circuits (not evaluated)
            ModelWeightsStatus.UPDATING,  # access 3: if == UPDATING -> True
            ModelWeightsStatus.NORMAL,  # access 4: inner while != NORMAL -> False (exit)
            ModelWeightsStatus.NORMAL,  # access 5: outer while != NORMAL -> False (exit)
        ]
        call_count = [0]

        class FakeValue:
            def __getitem__(self, idx):
                val = status_sequence[min(call_count[0], len(status_sequence) - 1)]
                call_count[0] += 1
                return val

            def __setitem__(self, idx, val):
                pass

        model_weights_status = MagicMock()
        model_weights_status.value = FakeValue()
        kv_cache_status = MagicMock()
        kv_cache_status.value = [0]
        model_runner = MagicMock()

        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=True
        )

        model_runner.clear_requests.assert_called()
        model_runner.update_parameters.assert_called_with(0)


class TestUpdateIpc(unittest.TestCase):
    """Test DynamicWeightManager._update_ipc."""

    @patch("paddle.load")
    def test_update_ipc(self, mock_paddle_load):
        """_update_ipc loads ipc meta and updates model state."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.ipc_path = "/shared_ipc_meta/ipc_metas_0"
        mgr._convert_ipc_meta_to_tensor = MagicMock(return_value={"p": "tensor"})
        mgr._update_model_from_state = MagicMock()

        mock_paddle_load.return_value = {"meta": "data"}

        mgr._update_ipc()

        mock_paddle_load.assert_called_once_with("/shared_ipc_meta/ipc_metas_0")
        mgr._convert_ipc_meta_to_tensor.assert_called_once_with({"meta": "data"})
        mgr._update_model_from_state.assert_called_once_with({"p": "tensor"}, "raw")


class TestRecreateDeepepBuffer(unittest.TestCase):
    """Test DynamicWeightManager.recreate_deepep_buffer."""

    @patch("paddle.distributed.barrier")
    @patch("fastdeploy.model_executor.layers.moe.ep.DeepEPBufferManager")
    def test_not_first_load_recreates(self, mock_buffer_mgr, mock_barrier):
        """recreate_deepep_buffer recreates buffer when not first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.ep_group = "ep_group"

        mgr.recreate_deepep_buffer()

        mock_buffer_mgr.recreate_buffer.assert_called_once()
        mock_barrier.assert_called_once_with("ep_group")

    def test_first_load_does_nothing(self):
        """recreate_deepep_buffer does nothing on first_load."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = True

        # Should not raise
        mgr.recreate_deepep_buffer()


class TestCaptureModelState(unittest.TestCase):
    """Test DynamicWeightManager._capture_model_state."""

    @patch("paddle.no_grad")
    def test_captures_params(self, mock_no_grad):
        """_capture_model_state stores all model params in state_dict."""
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()

        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.state_dict = {}

        param1 = MagicMock(shape=[10], dtype="float32", place="gpu:0")
        param2 = MagicMock(shape=[20, 30], dtype="float16", place="gpu:0")
        model = MagicMock()
        model.state_dict.return_value = {"layer.weight": param1, "layer.bias": param2}
        mgr.model_list = [model]

        mgr._capture_model_state()

        self.assertIn("layer.weight", mgr.state_dict)
        self.assertIn("layer.bias", mgr.state_dict)
        self.assertEqual(mgr.state_dict["layer.weight"], param1)
        self.assertEqual(mgr.state_dict["layer.bias"], param2)


class TestLogMemory(unittest.TestCase):
    """Test DynamicWeightManager._log_memory."""

    @patch("paddle.device.cuda.memory_reserved", return_value=2 * 1024**3)
    @patch("paddle.device.cuda.memory_allocated", return_value=1 * 1024**3)
    @patch("paddle.device.cuda.max_memory_reserved", return_value=4 * 1024**3)
    @patch("paddle.device.cuda.max_memory_allocated", return_value=3 * 1024**3)
    def test_log_memory(self, mock_max_alloc, mock_max_res, mock_alloc, mock_res):
        """_log_memory logs GPU memory usage without error."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        # Should not raise
        mgr._log_memory("test_context")


class TestUpdateIpcSnapshot(unittest.TestCase):
    """Test DynamicWeightManager._update_ipc_snapshot."""

    @patch("paddle.load")
    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_priority2_single_file(self, mock_rank, mock_glob, mock_exists, mock_load):
        """_update_ipc_snapshot loads single full pdparams file (priority 2)."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        # No part files
        mock_glob.return_value = []
        # Single file exists
        mock_exists.side_effect = lambda p: p == "/model/model_state.tp0.0.pdparams"
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with("/model/model_state.tp0.0.pdparams", safetensors=True)
        mgr._update_model_from_state.assert_called_once_with({"param": "tensor"}, "snapshot")

    @patch("paddle.load")
    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_priority3_legacy_format(self, mock_rank, mock_glob, mock_exists, mock_load):
        """_update_ipc_snapshot loads legacy format (priority 3)."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        mock_glob.return_value = []
        # Single file does NOT exist, but legacy does
        mock_exists.side_effect = lambda p: p == "/model/model_state.tp00.pdparams"
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with("/model/model_state.tp00.pdparams", safetensors=True)
        mgr._update_model_from_state.assert_called_once_with({"param": "tensor"}, "snapshot")

    @patch("os.path.exists", return_value=False)
    @patch("glob.glob", return_value=[])
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_priority4_fallback_not_found_raises(self, mock_rank, mock_glob, mock_exists):
        """_update_ipc_snapshot raises FileNotFoundError when all priorities fail."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0

        with self.assertRaises(FileNotFoundError) as ctx:
            mgr._update_ipc_snapshot()
        self.assertIn("No snapshot found", str(ctx.exception))

    @patch("gc.collect")
    @patch("paddle.load")
    @patch("glob.glob")
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_priority1_part_files(self, mock_rank, mock_glob, mock_load, mock_gc):
        """_update_ipc_snapshot loads chunked part files (priority 1)."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        # Return part files
        mock_glob.return_value = [
            "/model/model_state.tp0.0.part1.pdparams",
            "/model/model_state.tp0.0.part0.pdparams",
        ]
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        # Should load 2 parts (sorted by index: part0 first, part1 second)
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(mgr._update_model_from_state.call_count, 2)

    @patch("fastdeploy.rl.dynamic_weight_manager.logger")
    @patch("paddle.load")
    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_invalid_part_files_logged_and_skipped(self, mock_rank, mock_glob, mock_exists, mock_load, mock_logger):
        """_update_ipc_snapshot skips invalid part files and falls through."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        # Part file with non-numeric part index (regex doesn't match digits)
        mock_glob.return_value = ["/model/model_state.tp0.0.partabc.pdparams"]
        # Falls through to priority 2
        mock_exists.side_effect = lambda p: p == "/model/model_state.tp0.0.pdparams"
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        # Should fall through to priority 2 since part file name is invalid
        mock_load.assert_called_once_with("/model/model_state.tp0.0.pdparams", safetensors=True)
        # Warning should have been logged for invalid part files
        mock_logger.warning.assert_called_once()


class TestClearParameters(unittest.TestCase):
    """Test DynamicWeightManager.clear_parameters."""

    @patch("paddle.distributed.shutdown_process_group")
    @patch("paddle.distributed.barrier")
    @patch("paddle.device.cuda.empty_cache")
    @patch("fastdeploy.model_executor.layers.moe.ep.DeepEPBufferManager")
    def test_clear_with_ep(self, mock_buffer_mgr, mock_empty_cache, mock_barrier, mock_shutdown):
        """clear_parameters clears EP buffer and model weights."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tensor_parallel_size = 1
        mgr.parallel_config.ep_group = "ep_group"

        param = MagicMock()
        param._is_initialized.return_value = False
        model = MagicMock()
        model.state_dict.return_value = {"p": param}
        mgr.model_list = [model]
        mgr.state_dict = {"p": param}
        mgr._update_shared_status = MagicMock()

        mgr.clear_parameters(pid=0, shutdown_process_group=False)

        mock_buffer_mgr.clear_buffer.assert_called_once()
        param._clear_data.assert_called_once()
        mgr._update_shared_status.assert_called_once()

    @patch("paddle.distributed.shutdown_process_group")
    @patch("paddle.distributed.barrier")
    @patch("paddle.device.cuda.empty_cache")
    def test_clear_with_tp_shutdown(self, mock_empty_cache, mock_barrier, mock_shutdown):
        """clear_parameters shuts down tp group when shutdown_process_group=True."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = False
        mgr.parallel_config.tensor_parallel_size = 4
        mgr.parallel_config.tp_group = "tp_group"

        param = MagicMock()
        param._is_initialized.return_value = False
        model = MagicMock()
        model.state_dict.return_value = {"p": param}
        mgr.model_list = [model]
        mgr.state_dict = {"p": param}
        mgr._update_shared_status = MagicMock()

        with patch("paddle.distributed.collective._get_group_map_by_name", return_value={}):
            mgr.clear_parameters(pid=0, shutdown_process_group=True)

        # barrier for tp, then shutdown for tp, then global shutdown
        mock_barrier.assert_called()
        self.assertTrue(mock_shutdown.call_count >= 2)


class TestUpdateModelFromStateStrideMismatch(unittest.TestCase):
    """Test _update_model_from_state stride mismatch branch."""

    @patch("paddle.no_grad")
    @patch("paddle.empty")
    def test_stride_mismatch_uninitialized(self, mock_empty, mock_no_grad):
        """_update_model_from_state handles stride mismatch with uninitialized param."""
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()
        mock_empty.return_value = "empty_tensor"

        mgr = DynamicWeightManager.__new__(DynamicWeightManager)

        target_param = MagicMock()
        target_param.stride.return_value = [20, 1]
        target_param.dtype = "float32"
        target_param.shape = [10, 20]
        target_param._is_initialized.return_value = False

        new_param = MagicMock()
        new_param.stride.return_value = [1, 10]  # Different stride
        new_param.dtype = "float32"
        new_param.shape = [10, 20]

        mgr.state_dict = {"layer.weight": target_param}
        mgr._validate_parameter_match = MagicMock()

        mgr._update_model_from_state({"layer.weight": new_param}, "snapshot")

        # Should call paddle.empty and assign via [...]
        mock_empty.assert_called_once()
        # target_param[...] should be assigned twice (once with empty, once with new_param)
        self.assertEqual(target_param.__setitem__.call_count, 2)


class TestCheckModelWeightsStatusClearing(unittest.TestCase):
    """Test check_model_weights_status CLEARING branch."""

    @patch("time.sleep")
    def test_clearing_then_cleared(self, mock_sleep):
        """check_model_weights_status handles CLEARING -> CLEARED transition."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        # Line 523 logs value[0] first, then the while loop checks it.
        # Access pattern:
        #   logger.info: value[0] (access 0)
        #   outer while: value[0] != NORMAL (access 1)
        #   outer while: block or value[0] != CLEARED (access 2) -> False or True -> True
        #   if value[0] == UPDATING (access 3) -> False
        #   elif value[0] == CLEARING (access 4) -> True
        #   kv_cache_status write (no read from model_weights_status)
        #   clear_requests, clear_parameters
        #   inner while: value[0] != CLEARED (access 5) -> False (exit)
        #   outer while: value[0] != NORMAL (access 6) -> CLEARED != NORMAL -> True
        #   outer while: block or value[0] != CLEARED (access 7) -> False or False -> False (exit)
        status_sequence = [
            ModelWeightsStatus.CLEARING,  # access 0: logger.info
            ModelWeightsStatus.CLEARING,  # access 1: outer while != NORMAL
            ModelWeightsStatus.CLEARING,  # access 2: block or != CLEARED
            ModelWeightsStatus.CLEARING,  # access 3: if == UPDATING -> False
            ModelWeightsStatus.CLEARING,  # access 4: elif == CLEARING -> True
            ModelWeightsStatus.CLEARED,  # access 5: inner while != CLEARED -> False (exit)
            ModelWeightsStatus.CLEARED,  # access 6: outer while != NORMAL -> True
            ModelWeightsStatus.CLEARED,  # access 7: block or != CLEARED -> False (exit)
        ]
        call_count = [0]

        class FakeValue:
            def __getitem__(self, idx):
                val = status_sequence[min(call_count[0], len(status_sequence) - 1)]
                call_count[0] += 1
                return val

            def __setitem__(self, idx, val):
                pass

        model_weights_status = MagicMock()
        model_weights_status.value = FakeValue()
        kv_cache_status = MagicMock()
        kv_cache_status.value = [0]
        model_runner = MagicMock()

        # block=False so it exits on CLEARED
        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=False
        )

        model_runner.clear_requests.assert_called()
        model_runner.clear_parameters.assert_called_with(0)


class TestCheckModelWeightsStatusElseBranch(unittest.TestCase):
    """Test check_model_weights_status else branch (unknown status -> sleep)."""

    @patch("time.sleep")
    def test_unknown_status_sleeps(self, mock_sleep):
        """check_model_weights_status sleeps on unknown status then exits on NORMAL."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        # Access pattern with logger.info access at start:
        #   logger.info: value[0] (access 0)
        #   outer while: value[0] != NORMAL (access 1) -> True
        #   outer while: block or value[0] != CLEARED (access 2) -> True (block=True)
        #   if value[0] == UPDATING (access 3) -> False
        #   elif value[0] == CLEARING (access 4) -> False
        #   else -> sleep(0.01)
        #   outer while: value[0] != NORMAL (access 5) -> False (exit)
        UNKNOWN_STATUS = 99
        status_sequence = [
            UNKNOWN_STATUS,  # access 0: logger.info
            UNKNOWN_STATUS,  # access 1: outer while != NORMAL
            UNKNOWN_STATUS,  # access 2: block=True short-circuits
            UNKNOWN_STATUS,  # access 3: if == UPDATING -> False
            UNKNOWN_STATUS,  # access 4: elif == CLEARING -> False -> else sleep
            ModelWeightsStatus.NORMAL,  # access 5: outer while == NORMAL -> exit
        ]
        call_count = [0]

        class FakeValue:
            def __getitem__(self, idx):
                val = status_sequence[min(call_count[0], len(status_sequence) - 1)]
                call_count[0] += 1
                return val

            def __setitem__(self, idx, val):
                pass

        model_weights_status = MagicMock()
        model_weights_status.value = FakeValue()
        kv_cache_status = None
        model_runner = MagicMock()

        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=True
        )

        mock_sleep.assert_called_with(0.01)


class TestUpdateParametersWithEP(unittest.TestCase):
    """Test update_parameters with expert parallel enabled."""

    @patch("paddle.distributed.barrier")
    @patch("fastdeploy.model_executor.layers.moe.ep.DeepEPBufferManager")
    @patch("paddle.distributed.restart_process_group")
    @patch("paddle.device.cuda.empty_cache")
    def test_not_first_load_with_ep(self, mock_empty_cache, mock_restart, mock_buffer_mgr, mock_barrier):
        """update_parameters recreates EP buffer when not first_load and EP enabled."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.first_load = False
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tp_group = "tp_group"
        mgr.parallel_config.ep_group = "ep_group"
        mgr.load_config = MagicMock()
        mgr.load_config.load_strategy = "ipc"

        mgr._update_ipc = MagicMock()

        mgr.update_parameters(restart_process_group=True)

        mock_buffer_mgr.recreate_buffer.assert_called_once()
        mock_barrier.assert_called_once_with("ep_group")
        # restart for default, tp, and ep
        self.assertEqual(mock_restart.call_count, 3)


class TestUpdateIpcSnapshotFallback(unittest.TestCase):
    """Test _update_ipc_snapshot fallback to /shared_ipc_meta/ path."""

    @patch("fastdeploy.rl.dynamic_weight_manager.logger")
    @patch("paddle.load")
    @patch("os.path.exists")
    @patch("glob.glob", return_value=[])
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_priority4_fallback_exists(self, mock_rank, mock_glob, mock_exists, mock_load, mock_logger):
        """_update_ipc_snapshot loads from /shared_ipc_meta/ as fallback."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        def exists_side_effect(p):
            if p == "/shared_ipc_meta/model_state.tp0.0.pdparams":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with("/shared_ipc_meta/model_state.tp0.0.pdparams")
        mgr._update_model_from_state.assert_called_once_with({"param": "tensor"}, "snapshot")


class TestUpdateIpcSnapshotInvalidPartIndex(unittest.TestCase):
    """Test _update_ipc_snapshot with part file that has parse error in index."""

    @patch("fastdeploy.rl.dynamic_weight_manager.logger")
    @patch("paddle.load")
    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_part_with_no_regex_match(self, mock_rank, mock_glob, mock_exists, mock_load, mock_logger):
        """_update_ipc_snapshot handles part file where regex doesn't match."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.fd_config = MagicMock()
        mgr.fd_config.model_config.model = "/model"
        mgr.meta_src_id = 0
        mgr._update_model_from_state = MagicMock()

        # File that matches glob but not regex (no .partN.)
        mock_glob.return_value = ["/model/model_state.tp0.0.nopart.pdparams"]
        mock_exists.side_effect = lambda p: p == "/model/model_state.tp0.0.pdparams"
        mock_load.return_value = {"param": "tensor"}

        mgr._update_ipc_snapshot()

        # Falls through to priority 2 (no valid parts, no invalid logged since regex didn't match at all)
        mock_load.assert_called_once_with("/model/model_state.tp0.0.pdparams", safetensors=True)


class TestClearParametersFullShutdown(unittest.TestCase):
    """Test clear_parameters with full shutdown and EP+TP."""

    @patch("paddle.distributed.collective._get_group_map_by_name")
    @patch("paddle.distributed.shutdown_process_group")
    @patch("paddle.distributed.barrier")
    @patch("paddle.device.cuda.empty_cache")
    @patch("fastdeploy.model_executor.layers.moe.ep.DeepEPBufferManager")
    def test_full_shutdown_with_ep_and_tp(
        self, mock_buffer_mgr, mock_empty_cache, mock_barrier, mock_shutdown, mock_get_map
    ):
        """clear_parameters handles full shutdown with both EP and TP."""
        mgr = DynamicWeightManager.__new__(DynamicWeightManager)
        mgr.parallel_config = MagicMock()
        mgr.parallel_config.enable_expert_parallel = True
        mgr.parallel_config.tensor_parallel_size = 4
        mgr.parallel_config.ep_group = "ep_group"
        mgr.parallel_config.tp_group = "tp_group"

        param = MagicMock()
        param._is_initialized.return_value = False
        model = MagicMock()
        model.state_dict.return_value = {"p": param}
        mgr.model_list = [model]
        mgr.state_dict = {"p": param}
        mgr._update_shared_status = MagicMock()

        # Mock the process group map for Gloo cleanup
        mock_pg = MagicMock()
        mock_pg.process_group = MagicMock(spec=[])  # No shutdown attr
        mock_get_map.return_value = {"gloo_group": mock_pg}

        mgr.clear_parameters(pid=0, shutdown_process_group=True)

        mock_buffer_mgr.clear_buffer.assert_called_once()
        param._clear_data.assert_called_once()
        # Multiple barriers and shutdowns
        self.assertTrue(mock_barrier.call_count >= 2)
        self.assertTrue(mock_shutdown.call_count >= 2)


class TestCheckModelWeightsStatusClearingWithSleep(unittest.TestCase):
    """Test check_model_weights_status CLEARING branch inner while sleep."""

    @patch("time.sleep")
    def test_clearing_inner_while_sleeps(self, mock_sleep):
        """check_model_weights_status sleeps in inner while waiting for CLEARED."""
        from fastdeploy.inter_communicator import ModelWeightsStatus

        # Access pattern: logger.info (0), outer while cond x2 (1,2), if (3), elif (4) -> CLEARING
        # Then inner while: first iteration NOT CLEARED (5) -> sleep, second iteration CLEARED (6)
        # Then outer while: (7) != NORMAL -> True, (8) block=False or != CLEARED -> False -> exit
        status_sequence = [
            ModelWeightsStatus.CLEARING,  # access 0: logger.info
            ModelWeightsStatus.CLEARING,  # access 1: outer while != NORMAL
            ModelWeightsStatus.CLEARING,  # access 2: block or != CLEARED
            ModelWeightsStatus.CLEARING,  # access 3: if == UPDATING -> False
            ModelWeightsStatus.CLEARING,  # access 4: elif == CLEARING -> True
            ModelWeightsStatus.CLEARING,  # access 5: inner while != CLEARED -> True -> sleep
            ModelWeightsStatus.CLEARED,  # access 6: inner while != CLEARED -> False (exit)
            ModelWeightsStatus.CLEARED,  # access 7: outer while != NORMAL -> True
            ModelWeightsStatus.CLEARED,  # access 8: block=False or != CLEARED -> False (exit)
        ]
        call_count = [0]

        class FakeValue:
            def __getitem__(self, idx):
                val = status_sequence[min(call_count[0], len(status_sequence) - 1)]
                call_count[0] += 1
                return val

            def __setitem__(self, idx, val):
                pass

        model_weights_status = MagicMock()
        model_weights_status.value = FakeValue()
        kv_cache_status = MagicMock()
        kv_cache_status.value = [0]
        model_runner = MagicMock()

        DynamicWeightManager.check_model_weights_status(
            model_weights_status, kv_cache_status, model_runner, pid=0, block=False
        )

        # sleep should be called at least once (the inner while sleep at line 543)
        mock_sleep.assert_called_with(0.01)
        model_runner.clear_parameters.assert_called_with(0)


class TestInit(unittest.TestCase):
    """Test DynamicWeightManager.__init__."""

    @patch.object(DynamicWeightManager, "finalize_update")
    @patch.object(DynamicWeightManager, "update_parameters")
    @patch.object(DynamicWeightManager, "_capture_model_state")
    @patch.object(DynamicWeightManager, "_get_gpu_id", return_value=0)
    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_ipc_strategy(self, mock_world_size, mock_gpu_id, mock_capture, mock_update, mock_finalize):
        """__init__ with ipc strategy calls update_parameters."""
        fd_config = MagicMock()
        fd_config.load_config.load_strategy = "ipc"
        fd_config.parallel_config.tensor_parallel_rank = 0

        model = MagicMock()
        mgr = DynamicWeightManager(fd_config, model, local_rank=0)

        self.assertEqual(mgr.local_rank, 0)
        self.assertEqual(mgr.rank, 0)
        self.assertEqual(mgr.nranks, 1)
        self.assertTrue(mgr.first_load)  # finalize_update is mocked, won't set False
        self.assertEqual(mgr.model_list, [model])
        self.assertIsNone(mgr.rdma_handle)
        mock_capture.assert_called_once()
        mock_update.assert_called_once()
        mock_finalize.assert_called_once()

    @patch.object(DynamicWeightManager, "finalize_update")
    @patch.object(DynamicWeightManager, "update_weights_by_rdma")
    @patch.object(DynamicWeightManager, "_capture_model_state")
    @patch.object(DynamicWeightManager, "_get_gpu_id", return_value=2)
    @patch("paddle.distributed.get_world_size", return_value=4)
    def test_init_rsync_strategy(self, mock_world_size, mock_gpu_id, mock_capture, mock_rdma, mock_finalize):
        """__init__ with rsync strategy calls update_weights_by_rdma."""
        fd_config = MagicMock()
        fd_config.load_config.load_strategy = "rsync"
        fd_config.parallel_config.tensor_parallel_rank = 1

        models = [MagicMock(), MagicMock()]
        mgr = DynamicWeightManager(fd_config, models, local_rank=1)

        self.assertEqual(mgr.local_rank, 1)
        self.assertEqual(mgr.nranks, 4)
        self.assertEqual(mgr.meta_src_id, 2)
        self.assertEqual(mgr.model_list, models)
        mock_rdma.assert_called_once()
        mock_finalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
