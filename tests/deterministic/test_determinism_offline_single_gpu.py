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
Single-GPU determinism offline inference tests for coverage.

Simplified from tests/e2e/4cards_cases/test_determinism_offline.py
for single-GPU coverage testing.

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_determinism_offline_single_gpu.py -v
"""

import os
from contextlib import contextmanager

import pytest

pytestmark = pytest.mark.gpu

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars before importing fastdeploy (must happen first)."""
    with env_override(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "FD_DETERMINISTIC_MODE": "1",
            "FD_CUSTOM_AR_MAX_SIZE_MB": "64",
        }
    ):
        # Lazy import: env vars must be set before importing fastdeploy
        global LLM, SamplingParams  # noqa: PLW0603
        from fastdeploy import LLM, SamplingParams

        yield


@pytest.fixture(scope="module")
def llm(model_path, _module_env):
    return LLM(
        model=model_path,
        tensor_parallel_size=1,  # Single GPU
        max_model_len=4096,
        enable_prefix_caching=False,
        graph_optimization_config={"use_cudagraph": False},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(llm, prompt, sp):
    """Generate once, return (text, token_ids)."""
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, list(out.outputs.token_ids)


def _assert_deterministic(llm, prompt, sp, runs=2):
    """Run *runs* times and assert all outputs are identical."""
    results = [_generate_text(llm, prompt, sp) for _ in range(runs)]
    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]
    assert all(t == texts[0] for t in texts), "Text outputs differ across runs"
    assert all(t == token_ids[0] for t in token_ids), "Token IDs differ across runs"
    return texts[0], token_ids[0]


# ===================== Core determinism tests =====================


def test_deterministic_same_prompt(llm):
    """Same prompt + same seed produces identical output across 3 runs."""
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=30, seed=123)
    _assert_deterministic(llm, "What is AI?", sp, runs=3)


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
