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
Unit test for Engine I/O capture module.

Tests the io_capture module functionality for capturing EngineService
input/output data.
"""

import tempfile
from pathlib import Path

import pytest

from fastdeploy.engine.io_capture import (
    EngineIOCapture,
    disable_capture,
    enable_capture,
    get_global_capture,
    is_capture_enabled,
)


class MockRequest:
    """Mock Request object for testing."""

    def __init__(self, request_id="test_req_1"):
        self.request_id = request_id
        self.prompt = "Hello, world!"
        self.prompt_token_ids = [1, 2, 3, 4, 5]
        self.prompt_token_ids_len = 5
        self.sampling_params = MockSamplingParams()
        self.task_type = None
        self.user = "test_user"


class MockSamplingParams:
    """Mock SamplingParams for testing."""

    def __init__(self):
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 100


class MockRequestOutput:
    """Mock RequestOutput object for testing."""

    def __init__(self, request_id="test_req_1"):
        self.request_id = request_id
        self.prompt = "Hello, world!"
        self.prompt_token_ids = [1, 2, 3, 4, 5]
        self.outputs = MockOutputs()
        self.finished = True
        self.error_code = 200
        self.error_msg = None


class MockOutputs:
    """Mock Outputs for testing."""

    def __init__(self):
        self.token_ids = [6, 7, 8]
        self.text = " test response"


class TestEngineIOCapture:
    """Test suite for EngineIOCapture."""

    def test_capture_initialization(self):
        """Test capture initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            assert capture.output_dir == Path(tmpdir)
            assert capture.is_enabled() is True

    def test_capture_disable_enable(self):
        """Test enable/disable functionality."""
        capture = EngineIOCapture()
        capture.disable()
        assert capture.is_enabled() is False
        capture.enable()
        assert capture.is_enabled() is True

    def test_capture_request(self):
        """Test capturing a request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            request = MockRequest()

            filepath = capture.capture_request(request)

            assert filepath is not None
            assert Path(filepath).exists()
            assert request.request_id in capture._captured_requests

    def test_capture_request_output(self):
        """Test capturing a request output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            output = MockRequestOutput()

            filepath = capture.capture_request_output(output)

            assert filepath is not None
            assert Path(filepath).exists()
            assert output.request_id in capture._captured_outputs

    def test_capture_schedule_task(self):
        """Test capturing schedule tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            tasks = [MockRequest(f"req_{i}") for i in range(3)]

            filepath = capture.capture_schedule_task(tasks, current_id=1)

            assert filepath is not None
            assert Path(filepath).exists()
            assert len(capture._captured_tasks) == 1
            assert capture._captured_tasks[0]["num_tasks"] == 3
            assert capture._captured_tasks[0]["current_id"] == 1

    def test_capture_worker_task(self):
        """Test capturing worker tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            tasks = [MockRequest(f"req_{i}") for i in range(2)]

            filepath = capture.capture_worker_task(tasks, real_bsz=4)

            assert filepath is not None
            assert Path(filepath).exists()
            assert len(capture._captured_worker_tasks) == 1
            assert capture._captured_worker_tasks[0]["num_tasks"] == 2
            assert capture._captured_worker_tasks[0]["real_bsz"] == 4

    def test_capture_when_disabled(self):
        """Test that nothing is captured when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            capture.disable()
            request = MockRequest()

            filepath = capture.capture_request(request)

            assert filepath is None
            assert len(capture._captured_requests) == 0

    def test_save_config_snapshot(self):
        """Test saving config snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            mock_config = MockConfig()
            capture.set_config(mock_config)

            filepath = capture.save_config_snapshot()

            assert filepath is not None
            assert Path(filepath).exists()
            assert capture._config_snapshot is not None

    def test_clear(self):
        """Test clearing captured data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = EngineIOCapture(tmpdir)
            request = MockRequest()
            capture.capture_request(request)

            assert len(capture._captured_requests) == 1

            capture.clear()

            assert len(capture._captured_requests) == 0
            assert len(capture._captured_outputs) == 0
            assert len(capture._captured_tasks) == 0

    def test_global_capture_functions(self):
        """Test global capture functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test enable
            enable_capture(tmpdir)
            assert is_capture_enabled() is True

            # Test get_global_capture
            capture = get_global_capture()
            assert capture is not None
            assert capture.output_dir == Path(tmpdir)

            # Test disable
            disable_capture()
            assert is_capture_enabled() is False


class MockConfig:
    """Mock configuration object."""

    def __init__(self):
        # Create mock config sections
        self.model_config = MockConfigSection()
        self.scheduler_config = MockConfigSection()
        self.cache_config = MockConfigSection()
        self.parallel_config = MockConfigSection()
        self.speculative_config = MockConfigSection()
        self.structured_outputs_config = MockConfigSection()
        self.eplb_config = MockConfigSection()


class MockConfigSection:
    """Mock config section."""

    def __init__(self):
        self.max_num_seqs = 128
        self.max_model_len = 4096
        self.block_size = 16
        self.tensor_parallel_size = 1
        self.data_parallel_size = 1
        self.local_data_parallel_id = 0
        self.enable_mm = False
        self.guided_decoding_backend = "off"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
