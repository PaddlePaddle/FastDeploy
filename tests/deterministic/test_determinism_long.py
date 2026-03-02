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
Determinism tests for long sequences and long prompts.

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_determinism_long.py -v
"""

import os

import pytest

pytestmark = pytest.mark.gpu

DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen2-7B-Instruct"

_ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_ENV_FD_DETERMINISTIC_MODE = "FD_DETERMINISTIC_MODE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars before importing fastdeploy (must happen first)."""
    old_cuda = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES)
    old_det = os.environ.get(_ENV_FD_DETERMINISTIC_MODE)

    os.environ[_ENV_CUDA_VISIBLE_DEVICES] = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES, "0")
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"

    global LLM, SamplingParams  # noqa: PLW0603
    from fastdeploy import LLM, SamplingParams

    yield

    if old_cuda is None:
        os.environ.pop(_ENV_CUDA_VISIBLE_DEVICES, None)
    else:
        os.environ[_ENV_CUDA_VISIBLE_DEVICES] = old_cuda
    if old_det is None:
        os.environ.pop(_ENV_FD_DETERMINISTIC_MODE, None)
    else:
        os.environ[_ENV_FD_DETERMINISTIC_MODE] = old_det


@pytest.fixture(autouse=True)
def _reset_deterministic_mode():
    """Ensure every test starts with deterministic mode ON."""
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"
    yield
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"


@pytest.fixture(scope="module")
def model_path():
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    return os.path.join(model_dir, MODEL_NAME)


@pytest.fixture(scope="module")
def llm(model_path, _module_env):
    return LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(llm, prompt, sp):
    """Generate once, return (text, token_ids)."""
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, out.outputs.token_ids


def _assert_deterministic(llm, prompt, sp, runs=2):
    """Run *runs* times and assert all outputs are identical."""
    results = [_generate_text(llm, prompt, sp) for _ in range(runs)]
    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]
    assert all(t == texts[0] for t in texts), "Text outputs differ across runs"
    assert all(t == token_ids[0] for t in token_ids), "Token IDs differ across runs"
    return texts[0], token_ids[0]


# ===================== Long sequence tests =====================


@pytest.mark.parametrize(
    "temp,seed",
    [
        (0.0, 100),
        (0.7, 170),
        (1.0, 200),
    ],
)
def test_deterministic_long_sequence(llm, temp, seed):
    """Long generation (512+ tokens) stays deterministic at various temperatures."""
    prompt = "Please describe the history of AI in detail, including major milestones and key technical breakthroughs."
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, seed=seed)

    text, token_ids = _assert_deterministic(llm, prompt, sp)
    assert len(token_ids) >= 100, f"Expected >= 100 tokens, got {len(token_ids)}"


def test_deterministic_long_prompt(llm):
    """Long input prompt (prefill-heavy) stays deterministic."""
    base = "This is a description about natural language processing. "
    long_prompt = (base * 50) + "Please summarize the above."
    sp = SamplingParams(temperature=0.5, max_tokens=100, seed=2024)

    _assert_deterministic(llm, long_prompt, sp)


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
