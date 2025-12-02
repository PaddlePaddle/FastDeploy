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

"""Pytest configuration and fixtures for XPU CI tests."""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Base path to model files",
    )
    parser.addoption(
        "--xpu-id",
        action="store",
        type=int,
        default=0,
        help="XPU device ID (0 or 1)",
    )


@pytest.fixture(scope="session")
def model_path(request) -> str:
    """Get model path from command line or environment variable.

    Args:
        request: pytest request fixture.

    Returns:
        Path to model files.
    """
    path = request.config.getoption("--model-path")
    if path is None:
        path = os.getenv("MODEL_PATH")
    if path is None:
        pytest.fail("Model path not specified. Use --model-path or set MODEL_PATH environment variable.")
    return path


@pytest.fixture(scope="session")
def xpu_id(request) -> int:
    """Get XPU device ID from command line or environment variable.

    Args:
        request: pytest request fixture.

    Returns:
        XPU device ID (0 or 1).
    """
    xpu_id = request.config.getoption("--xpu-id")
    if xpu_id is None:
        xpu_id = int(os.getenv("XPU_ID", "0"))
    return xpu_id


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "v1_mode: marks tests as V1 mode tests",
    )
    config.addinivalue_line(
        "markers",
        "w4a8: marks tests as W4A8 quantization tests",
    )
    config.addinivalue_line(
        "markers",
        "vl_model: marks tests as VL model tests",
    )
    config.addinivalue_line(
        "markers",
        "expert_parallel: marks tests as Expert Parallel tests",
    )
    config.addinivalue_line(
        "markers",
        "ep4tp4: marks tests as EP4TP4 tests",
    )
    config.addinivalue_line(
        "markers",
        "ep4tp1: marks tests as EP4TP1 tests",
    )
    config.addinivalue_line(
        "markers",
        "all2all: marks tests as all2all tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    # Add timeout marker to all tests if not already present
    for item in items:
        if not any(mark.name == "timeout" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(1800))  # 30 minutes default timeout
