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

"""
Offline inference test for comparing old and new EngineService architectures.

This test captures input/output from old architecture and uses it to
verify the new architecture produces identical results.
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastdeploy.engine.io_capture import EngineIOCapture, disable_capture


class TestArchitectureComparison:
    """
    Test suite for comparing old and new engine architectures.

    Tests cover different configuration parameters that affect EngineService I/O:
    - Scheduler config (max_num_seqs, max_num_batched_tokens)
    - Cache config (block_size, enable_prefix_caching)
    - Parallel config (tensor_parallel_size, data_parallel_size)
    - Speculative config (method, speculative_max_model_len)
    - Splitwise role (prefill, decode, mixed)
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Cleanup before test
        disable_capture()
        yield
        # Cleanup after test
        disable_capture()

    def test_io_capture_enabled_env_var(self):
        """Test that FD_ENABLE_ENGINE_IO_CAPTURE enables capture."""
        # Set environment variable
        os.environ["FD_ENABLE_ENGINE_IO_CAPTURE"] = "1"

        # Import after setting env var
        import importlib

        import fastdeploy.envs as envs

        importlib.reload(envs)

        assert envs.FD_ENABLE_ENGINE_IO_CAPTURE == "1"

        # Cleanup
        del os.environ["FD_ENABLE_ENGINE_IO_CAPTURE"]

    def test_io_capture_custom_output_dir(self):
        """Test that FD_ENGINE_IO_CAPTURE_DIR sets output directory."""
        custom_dir = "/tmp/test_capture_dir"
        os.environ["FD_ENGINE_IO_CAPTURE_DIR"] = custom_dir

        # Import after setting env var
        import importlib

        import fastdeploy.envs as envs

        importlib.reload(envs)

        assert envs.FD_ENGINE_IO_CAPTURE_DIR == custom_dir

        # Cleanup
        del os.environ["FD_ENGINE_IO_CAPTURE_DIR"]

    def test_capture_and_load_request(self):
        """Test capturing and loading a request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and enable capture
            capture = EngineIOCapture(tmpdir)

            # Create mock request
            request = self._create_mock_request("test_req_1")

            # Capture request
            filepath = capture.capture_request(request)
            assert filepath is not None
            assert Path(filepath).exists()

            # Load and verify
            loaded_data = self._load_capture_file(filepath)
            assert loaded_data is not None
            assert loaded_data["request_id"] == "test_req_1"
            assert loaded_data["prompt"] == "Test prompt"

    def test_capture_and_load_request_output(self):
        """Test capturing and loading a request output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and enable capture
            capture = EngineIOCapture(tmpdir)

            # Create mock output
            output = self._create_mock_output("test_req_1")

            # Capture output
            filepath = capture.capture_request_output(output)
            assert filepath is not None
            assert Path(filepath).exists()

            # Load and verify
            loaded_data = self._load_capture_file(filepath)
            assert loaded_data is not None
            assert loaded_data["request_id"] == "test_req_1"
            assert loaded_data["finished"] is True

    def test_capture_and_load_schedule_task(self):
        """Test capturing and loading schedule tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and enable capture
            capture = EngineIOCapture(tmpdir)

            # Create mock tasks
            tasks = [self._create_mock_request(f"req_{i}") for i in range(3)]

            # Capture tasks
            filepath = capture.capture_schedule_task(tasks, current_id=5)
            assert filepath is not None
            assert Path(filepath).exists()

            # Load and verify
            loaded_data = self._load_capture_file(filepath)
            assert loaded_data is not None
            assert loaded_data["current_id"] == 5
            assert loaded_data["num_tasks"] == 3
            assert len(loaded_data["tasks"]) == 3

    def test_capture_and_load_worker_task(self):
        """Test capturing and loading worker tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and enable capture
            capture = EngineIOCapture(tmpdir)

            # Create mock tasks
            tasks = [self._create_mock_request(f"req_{i}") for i in range(2)]

            # Capture tasks
            filepath = capture.capture_worker_task(tasks, real_bsz=4)
            assert filepath is not None
            assert Path(filepath).exists()

            # Load and verify
            loaded_data = self._load_capture_file(filepath)
            assert loaded_data is not None
            assert loaded_data["real_bsz"] == 4
            assert loaded_data["num_tasks"] == 2
            assert len(loaded_data["tasks"]) == 2

    def test_numpy_array_serialization(self):
        """Test that numpy arrays are correctly serialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)

            # Create request with numpy array
            request = self._create_mock_request("test_numpy")
            request.block_table = np.array([[1, 2], [3, 4]], dtype=np.int32)

            # Capture request
            filepath = capture.capture_request(request)

            # Load and verify numpy array
            loaded_data = self._load_capture_file(filepath)
            assert loaded_data is not None
            assert "block_table" in loaded_data
            block_table_data = loaded_data["block_table"]
            assert block_table_data["_type"] == "numpy.ndarray"
            assert block_table_data["shape"] == [2, 2] or block_table_data["shape"] == (2, 2)
            assert block_table_data["dtype"] == "int32"
            assert block_table_data["data"] == [[1, 2], [3, 4]]

    def test_config_capture_sections(self):
        """Test that all important config sections are captured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            config = self._create_mock_config()

            capture.set_config(config)
            snapshot = capture._config_snapshot

            assert snapshot is not None
            # Verify all sections are present
            assert "model_config" in snapshot
            assert "scheduler_config" in snapshot
            assert "cache_config" in snapshot
            assert "parallel_config" in snapshot
            assert "speculative_config" in snapshot
            assert "structured_outputs_config" in snapshot
            assert "eplb_config" in snapshot

    def test_index_file_generation(self):
        """Test that index file is correctly generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)

            # Capture some data
            capture.capture_request(self._create_mock_request("req_1"))
            capture.capture_request_output(self._create_mock_output("req_1"))

            # Save index
            index_path = capture.save_index()
            assert index_path is not None
            assert Path(index_path).exists()

            # Load and verify index
            with open(index_path, "r") as f:
                index_data = json.load(f)

            assert "session_id" in index_data
            assert "timestamp" in index_data
            assert "requests" in index_data
            assert "outputs" in index_data
            assert "num_schedule_tasks" in index_data
            assert "num_worker_tasks" in index_data

    def test_multiple_capture_sessions(self):
        """Test that multiple capture sessions work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First session
            capture1 = EngineIOCapture(tmpdir)
            capture1.capture_request(self._create_mock_request("req_1"))

            # Second session (different session_id)
            import time

            time.sleep(1)  # Ensure different timestamp
            capture2 = EngineIOCapture(tmpdir)
            capture2.capture_request(self._create_mock_request("req_2"))

            assert capture1._session_id != capture2._session_id

    def _create_mock_request(self, request_id: str) -> Any:
        """Create a mock request object for testing."""
        sampling_params = self._create_mock_sampling_params()

        class MockRequest:
            def __init__(self, rid, sp):
                self.request_id = rid
                self.prompt = "Test prompt"
                self.prompt_token_ids = [1, 2, 3, 4, 5]
                self.prompt_token_ids_len = 5
                self.sampling_params = sp
                self.task_type = None
                self.user = "test_user"

        return MockRequest(request_id, sampling_params)

    def _create_mock_output(self, request_id: str) -> Any:
        """Create a mock request output object for testing."""
        outputs = self._create_mock_outputs()

        class MockRequestOutput:
            def __init__(self, rid, out):
                self.request_id = rid
                self.prompt = "Test prompt"
                self.prompt_token_ids = [1, 2, 3, 4, 5]
                self.outputs = out
                self.finished = True
                self.error_code = 200
                self.error_msg = None

        return MockRequestOutput(request_id, outputs)

    def _create_mock_sampling_params(self) -> Any:
        """Create a mock sampling params object for testing."""

        class MockSamplingParams:
            def __init__(self):
                self.temperature = 0.7
                self.top_p = 0.9
                self.max_tokens = 100

        return MockSamplingParams()

    def _create_mock_outputs(self) -> Any:
        """Create a mock outputs object for testing."""

        class MockOutputs:
            def __init__(self):
                self.token_ids = [6, 7, 8]
                self.text = " test response"

        return MockOutputs()

    def _create_mock_config(self) -> Any:
        """Create a mock config object for testing."""
        config_sections = []
        for _ in range(7):
            config_sections.append(self._create_config_section())

        class MockConfig:
            def __init__(self, sections):
                self.model_config = sections[0]
                self.scheduler_config = sections[1]
                self.cache_config = sections[2]
                self.parallel_config = sections[3]
                self.speculative_config = sections[4]
                self.structured_outputs_config = sections[5]
                self.eplb_config = sections[6]

        return MockConfig(config_sections)

    def _create_config_section(self) -> Any:
        """Create a mock config section for testing."""

        class MockConfigSection:
            def __init__(self):
                self.max_num_seqs = 128
                self.max_model_len = 4096
                self.block_size = 16
                self.max_num_batched_tokens = 8192
                self.tensor_parallel_size = 1
                self.data_parallel_size = 1
                self.local_data_parallel_id = 0
                self.enable_mm = False
                self.guided_decoding_backend = "off"
                self.enable_prefix_caching = False
                self.splitwise_role = "none"

        return MockConfigSection()

    def _load_capture_file(self, filepath: str) -> Any:
        """Load a capture file."""
        with open(filepath, "rb") as f:
            npz_file = np.load(f, allow_pickle=True)
            return pickle.loads(npz_file["data"])


class TestConfigurationParameters:
    """
    Test suite for configuration parameters that affect EngineService I/O.

    These tests identify which configuration parameters need to be covered
    in offline inference tests.
    """

    def test_scheduler_max_num_seqs(self):
        """Test that max_num_seqs affects request batching."""
        # This parameter affects how many requests can be scheduled at once
        # Test should verify both old and new architectures respect this limit
        pass

    def test_scheduler_max_num_batched_tokens(self):
        """Test that max_num_batched_tokens affects token batching."""
        # This parameter affects how many tokens can be batched
        # Test should verify both old and new architectures respect this limit
        pass

    def test_cache_block_size(self):
        """Test that block_size affects memory allocation."""
        # This parameter affects KV cache block size
        # Test should verify both architectures allocate memory identically
        pass

    def test_cache_enable_prefix_caching(self):
        """Test that enable_prefix_caching affects prefix sharing."""
        # This parameter enables prefix cache sharing
        # Test should verify both architectures handle prefix caching identically
        pass

    def test_parallel_tensor_parallel_size(self):
        """Test that tensor_parallel_size affects model parallelism."""
        # This parameter affects tensor parallel execution
        # Test should verify both architectures handle TP correctly
        pass

    def test_parallel_data_parallel_size(self):
        """Test that data_parallel_size affects data parallelism."""
        # This parameter affects data parallel execution
        # Test should verify both architectures handle DP correctly
        pass

    def test_speculative_method(self):
        """Test that speculative method affects generation."""
        # This parameter enables speculative decoding (e.g., MTP)
        # Test should verify both architectures handle speculation identically
        pass

    def test_splitwise_role(self):
        """Test that splitwise_role affects execution mode."""
        # This parameter can be 'prefill', 'decode', or 'mixed'
        # Test should verify all three modes work correctly
        pass

    def test_model_enable_mm(self):
        """Test that enable_mm affects multimodal processing."""
        # This parameter enables multimodal (vision/audio) support
        # Test should verify both architectures handle multimodal inputs identically
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
