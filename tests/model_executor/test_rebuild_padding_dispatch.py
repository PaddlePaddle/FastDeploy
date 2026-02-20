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
import os
import importlib.util

# Mock essential dependencies BEFORE importing the module under test
mock_paddle = MagicMock()
sys.modules["paddle"] = mock_paddle

mock_numpy = MagicMock()
sys.modules["numpy"] = mock_numpy

# Mock fastdeploy top-level package
fd = types.ModuleType("fastdeploy")
fd.__path__ = [] # Ensure it is treated as a package
sys.modules["fastdeploy"] = fd
fd.envs = MagicMock()
fd.envs.FD_DISABLED_RECOVER = "0"
sys.modules["fastdeploy.envs"] = fd.envs

# Mock fastdeploy.config
fd_config = types.ModuleType("fastdeploy.config")
sys.modules["fastdeploy.config"] = fd_config
fd_config.SpeculativeConfig = MagicMock()

# Mock fastdeploy.platforms
fd_platforms = types.ModuleType("fastdeploy.platforms")
sys.modules["fastdeploy.platforms"] = fd_platforms
current_platform = MagicMock()
fd_platforms.current_platform = current_platform

# Mock fastdeploy.worker
fd_worker = types.ModuleType("fastdeploy.worker")
sys.modules["fastdeploy.worker"] = fd_worker

# Mock fastdeploy.worker.input_batch
fd_worker_input_batch = types.ModuleType("fastdeploy.worker.input_batch")
sys.modules["fastdeploy.worker.input_batch"] = fd_worker_input_batch
fd_worker_input_batch.InputBatch = MagicMock()
fd_worker_input_batch.recover_batch_index_for_output = MagicMock()
fd_worker_input_batch.recover_batch_index_for_sampler_output = MagicMock()

# Mock fastdeploy.worker.output
fd_worker_output = types.ModuleType("fastdeploy.worker.output")
sys.modules["fastdeploy.worker.output"] = fd_worker_output
fd_worker_output.LogprobsTensors = MagicMock()
fd_worker_output.ModelOutputData = MagicMock()
fd_worker_output.SamplerOutput = MagicMock()

# Mock fastdeploy.model_executor
fd_me = types.ModuleType("fastdeploy.model_executor")
fd_me.__path__ = []
sys.modules["fastdeploy.model_executor"] = fd_me

# Mock fastdeploy.model_executor.entropy_utils
fd_me_entropy = types.ModuleType("fastdeploy.model_executor.entropy_utils")
sys.modules["fastdeploy.model_executor.entropy_utils"] = fd_me_entropy
fd_me_entropy.calculate_logits_entropy = MagicMock()
fd_me_entropy.speculate_calculate_logits_entropy = MagicMock()

# Mock fastdeploy.model_executor.layers
fd_me_layers = types.ModuleType("fastdeploy.model_executor.layers")
sys.modules["fastdeploy.model_executor.layers"] = fd_me_layers

# Mock fastdeploy.model_executor.layers.sample
fd_me_layers_sample = types.ModuleType("fastdeploy.model_executor.layers.sample")
sys.modules["fastdeploy.model_executor.layers.sample"] = fd_me_layers_sample

# Mock fastdeploy.model_executor.layers.sample.meta_data
fd_me_layers_sample_meta = types.ModuleType("fastdeploy.model_executor.layers.sample.meta_data")
sys.modules["fastdeploy.model_executor.layers.sample.meta_data"] = fd_me_layers_sample_meta
fd_me_layers_sample_meta.SamplingMetadata = MagicMock()

# Mock fastdeploy.output
fd_output = types.ModuleType("fastdeploy.output")
sys.modules["fastdeploy.output"] = fd_output

# Mock fastdeploy.output.pooler
fd_output_pooler = types.ModuleType("fastdeploy.output.pooler")
sys.modules["fastdeploy.output.pooler"] = fd_output_pooler
fd_output_pooler.PoolerOutput = MagicMock()
fd_output_pooler.PoolingSequenceGroupOutput = MagicMock()

# Mock fastdeploy.output.stream_transfer_data
fd_output_stream = types.ModuleType("fastdeploy.output.stream_transfer_data")
sys.modules["fastdeploy.output.stream_transfer_data"] = fd_output_stream
fd_output_stream.DecoderState = MagicMock()
fd_output_stream.StreamTransferData = MagicMock()

# Mock fastdeploy.model_executor.ops
fd_me_ops = types.ModuleType("fastdeploy.model_executor.ops")
sys.modules["fastdeploy.model_executor.ops"] = fd_me_ops

# Mock ops modules
mock_ops_gpu = types.ModuleType("fastdeploy.model_executor.ops.gpu")
mock_ops_gpu.rebuild_padding = MagicMock(return_value="gpu_result")
# Mock other functions imported from ops.gpu
for attr in [
    "get_padding_offset", "save_output", "set_stop_value_multi_ends", "step_paddle", "update_inputs",
    "save_output_topk", "speculate_get_seq_lens_output", "speculate_limit_thinking_content_length_v1",
    "speculate_limit_thinking_content_length_v2", "speculate_save_output", "speculate_save_output_topk",
    "speculate_set_stop_value_multi_seqs", "speculate_set_value_by_flags_and_idx", "speculate_step_paddle",
    "speculate_step_reschedule", "speculate_step_system_cache", "speculate_update", "step_reschedule",
    "step_system_cache", "update_inputs_v1", "limit_thinking_content_length_v1", "limit_thinking_content_length_v2"
]:
    setattr(mock_ops_gpu, attr, MagicMock())
sys.modules["fastdeploy.model_executor.ops.gpu"] = mock_ops_gpu

mock_ops_cpu = types.ModuleType("fastdeploy.model_executor.ops.cpu")
mock_ops_cpu.rebuild_padding_cpu = MagicMock(return_value="cpu_result")
sys.modules["fastdeploy.model_executor.ops.cpu"] = mock_ops_cpu

mock_ops_iluvatar = types.ModuleType("fastdeploy.model_executor.ops.iluvatar")
mock_ops_iluvatar.rebuild_padding = MagicMock(return_value="iluvatar_result")
for attr in [
    "get_padding_offset", "limit_thinking_content_length_v1", "limit_thinking_content_length_v2",
    "save_output", "set_stop_value_multi_ends", "step_paddle", "update_inputs", "update_inputs_v1"
]:
    setattr(mock_ops_iluvatar, attr, MagicMock())
sys.modules["fastdeploy.model_executor.ops.iluvatar"] = mock_ops_iluvatar

mock_ops_gcu = types.ModuleType("fastdeploy.model_executor.ops.gcu")
mock_ops_gcu.rebuild_padding = MagicMock(return_value="gcu_result")
for attr in [
    "get_padding_offset", "save_output", "set_stop_value_multi_ends", "update_inputs"
]:
    setattr(mock_ops_gcu, attr, MagicMock())
sys.modules["fastdeploy.model_executor.ops.gcu"] = mock_ops_gcu

# Load the module manually because we are mocking the package structure
file_path = "fastdeploy/model_executor/pre_and_post_process.py"
module_name = "fastdeploy.model_executor.pre_and_post_process"

spec = importlib.util.spec_from_file_location(module_name, file_path)
ppp = importlib.util.module_from_spec(spec)
sys.modules[module_name] = ppp
spec.loader.exec_module(ppp)

class TestRebuildPaddingDispatch(unittest.TestCase):
    def setUp(self):
        # Reset the cached implementation before each test
        ppp._rebuild_padding_impl = None

    def test_cuda_dispatch(self):
        # Must patch ppp.current_platform because it was imported into ppp namespace
        with patch.object(ppp.current_platform, "is_cuda", return_value=True), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "gpu_result")
            self.assertIsNotNone(ppp._rebuild_padding_impl)
            pid, impl = ppp._rebuild_padding_impl
            self.assertEqual(pid, os.getpid())
            self.assertEqual(impl, mock_ops_gpu.rebuild_padding)

    def test_dcu_dispatch(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=True), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "gpu_result")
            pid, impl = ppp._rebuild_padding_impl
            self.assertEqual(pid, os.getpid())
            self.assertTrue(callable(impl))
            self.assertNotEqual(impl, mock_ops_gpu.rebuild_padding)

    def test_iluvatar_dispatch(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=True), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "iluvatar_result")

    def test_gcu_dispatch(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=True), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "gcu_result")

    def test_cpu_dispatch(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False), \
             patch.object(ppp.current_platform, "is_cpu", return_value=True):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "cpu_result")

    def test_maca_dispatch(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=True), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False), \
             patch.object(ppp.current_platform, "is_cpu", return_value=False):
            res = ppp.rebuild_padding(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
            self.assertEqual(res, "gpu_result")

    def test_pid_change(self):
        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False), \
             patch.object(ppp.current_platform, "is_cpu", return_value=True):
            ppp.rebuild_padding(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
            self.assertEqual(ppp._rebuild_padding_impl[0], os.getpid())

        ppp._rebuild_padding_impl = (os.getpid() + 1, ppp._rebuild_padding_impl[1])

        with patch.object(ppp.current_platform, "is_cuda", return_value=False), \
             patch.object(ppp.current_platform, "is_iluvatar", return_value=False), \
             patch.object(ppp.current_platform, "is_gcu", return_value=False), \
             patch.object(ppp.current_platform, "is_dcu", return_value=False), \
             patch.object(ppp.current_platform, "is_maca", return_value=False), \
             patch.object(ppp.current_platform, "is_intel_hpu", return_value=False), \
             patch.object(ppp.current_platform, "is_cpu", return_value=True):
            ppp.rebuild_padding(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
            self.assertEqual(ppp._rebuild_padding_impl[0], os.getpid())

if __name__ == "__main__":
    unittest.main()
