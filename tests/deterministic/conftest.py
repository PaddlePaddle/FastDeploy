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

"""Shared fixtures and helpers for deterministic tests."""

import os
from contextlib import contextmanager

import pytest

DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen2-7B-Instruct"


@contextmanager
def env_override(mapping):
    """Temporarily set env vars, restoring original values on exit."""
    old = {k: os.environ.get(k) for k in mapping}
    os.environ.update(mapping)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def model_path():
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    return os.path.join(model_dir, MODEL_NAME)


@pytest.fixture(autouse=True)
def _reset_deterministic_mode():
    """Ensure every test starts with deterministic mode ON."""
    os.environ["FD_DETERMINISTIC_MODE"] = "1"
    yield
    os.environ["FD_DETERMINISTIC_MODE"] = "1"
