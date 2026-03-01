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
Long sequence determinism tests (端到端兜底回归).

This test ensures that the deterministic mode works correctly for long sequences
that trigger the partition_kv code path (num_chunks > 1 when KV length > 1024).

Design rationale (分层测试策略):
- This is an END-TO-END test that only runs POSITIVE (deterministic mode ON)
- Negative testing (detecting non-determinism) is done at the OPERATOR level
  in tests/layers/test_attention_determinism.py
- See docs/test_long_seq_review.md for details

Key requirements:
1. Total KV length (prompt_tokens + max_tokens) must exceed 1024 to trigger partition_kv
2. Recommended: KV length >= 2048 to ensure num_chunks >= 2

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/deterministic/test_determinism_long_sequence.py -v
"""

import gc
import itertools
import os

import pytest

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen2-7B-Instruct"

_ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_ENV_FD_DETERMINISTIC_MODE = "FD_DETERMINISTIC_MODE"
_ENV_FD_CUSTOM_AR_MAX_SIZE_MB = "FD_CUSTOM_AR_MAX_SIZE_MB"
_ENV_FLAGS_MAX_PARTITION_SIZE = "FLAGS_max_partition_size"

# Use smallest chunk_size (64) to maximize num_chunks and increase
# sensitivity to partition_kv non-determinism. With chunk_size=64:
# - 1200 tokens → 19 chunks (vs 2 chunks with default 1024)
# - More chunks = more merge operations = easier to detect non-determinism
_CHUNK_SIZE_FOR_TEST = "64"

# Long prompt to ensure KV length > 1024 (triggers partition_kv path)
# This sentence is ~20 tokens, repeated 40 times = ~800 tokens
_BASE_SENTENCE = (
    "Artificial intelligence has transformed various industries including healthcare, "
    "finance, transportation, and education through machine learning algorithms. "
)
_LONG_PROMPT = _BASE_SENTENCE * 40 + (
    "Based on the above context about AI, please provide a detailed analysis of "
    "the future trends and potential challenges in AI development."
)

# With ~800 token prompt + 1280 max_tokens, total KV length ~2080 > 1024*2
# This ensures num_chunks >= 2, triggering the partition_kv code path
_MAX_TOKENS_LONG = 1280

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars BEFORE importing fastdeploy (must happen first)."""
    old_cuda = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES)
    old_det = os.environ.get(_ENV_FD_DETERMINISTIC_MODE)
    old_ar = os.environ.get(_ENV_FD_CUSTOM_AR_MAX_SIZE_MB)
    old_partition_size = os.environ.get(_ENV_FLAGS_MAX_PARTITION_SIZE)

    os.environ[_ENV_CUDA_VISIBLE_DEVICES] = os.environ.get(_ENV_CUDA_VISIBLE_DEVICES, "0,1,2,3")
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"
    os.environ[_ENV_FD_CUSTOM_AR_MAX_SIZE_MB] = os.environ.get(_ENV_FD_CUSTOM_AR_MAX_SIZE_MB, "57")
    os.environ[_ENV_FLAGS_MAX_PARTITION_SIZE] = _CHUNK_SIZE_FOR_TEST

    global LLM, SamplingParams  # noqa: PLW0603
    from fastdeploy import LLM, SamplingParams

    yield

    # Restore original environment
    if old_cuda is None:
        os.environ.pop(_ENV_CUDA_VISIBLE_DEVICES, None)
    else:
        os.environ[_ENV_CUDA_VISIBLE_DEVICES] = old_cuda
    if old_det is None:
        os.environ.pop(_ENV_FD_DETERMINISTIC_MODE, None)
    else:
        os.environ[_ENV_FD_DETERMINISTIC_MODE] = old_det
    if old_ar is None:
        os.environ.pop(_ENV_FD_CUSTOM_AR_MAX_SIZE_MB, None)
    else:
        os.environ[_ENV_FD_CUSTOM_AR_MAX_SIZE_MB] = old_ar
    if old_partition_size is None:
        os.environ.pop(_ENV_FLAGS_MAX_PARTITION_SIZE, None)
    else:
        os.environ[_ENV_FLAGS_MAX_PARTITION_SIZE] = old_partition_size


@pytest.fixture(autouse=True)
def _reset_deterministic_mode():
    """Ensure every test starts with deterministic mode ON."""
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"
    yield
    os.environ[_ENV_FD_DETERMINISTIC_MODE] = "1"


@pytest.fixture(scope="module")
def model_path():
    """Get model path from environment or use default."""
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    return os.path.join(model_dir, MODEL_NAME)


@pytest.fixture(scope="module")
def llm(model_path, _module_env):
    """Create LLM instance, shared across all tests in this module."""
    instance = LLM(
        model=model_path,
        tensor_parallel_size=int(os.getenv("TP_SIZE", "4")),
        max_model_len=8192,
        enable_prefix_caching=False,  # Disabled for determinism testing
    )
    yield instance
    # Cleanup
    del instance
    gc.collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(llm, prompt, sp):
    """Generate once, return (text, token_ids)."""
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, list(out.outputs.token_ids)


def _assert_deterministic(llm, prompt, sp, runs=3):
    """Run *runs* times and assert all outputs are identical (text AND token_ids)."""
    results = [_generate_text(llm, prompt, sp) for _ in range(runs)]
    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]

    # Check text equality with detailed diff on failure
    if not all(t == texts[0] for t in texts):
        _report_text_diff(texts)
        pytest.fail("Text outputs differ across runs")

    # Check token_ids equality
    assert all(t == token_ids[0] for t in token_ids), "Token IDs differ across runs"

    return texts[0], token_ids[0]


def _report_text_diff(texts):
    """Report detailed diff when texts differ."""
    for i, text in enumerate(texts[1:], start=1):
        if text != texts[0]:
            # Check length difference first
            if len(text) != len(texts[0]):
                print(f"Run {i}: length differs (baseline={len(texts[0])}, got={len(text)})")

            # Find first difference using zip_longest
            for j, (c1, c2) in enumerate(itertools.zip_longest(texts[0], text, fillvalue="")):
                if c1 != c2:
                    print(f"Run {i}: first diff at pos {j}")
                    print(f"  Baseline: {repr(texts[0][max(0, j-10):j+20])}")
                    print(f"  Run {i}:   {repr(text[max(0, j-10):j+20])}")
                    break


# ---------------------------------------------------------------------------
# Tests (端到端正向测试 - 只验证确定性模式有效)
# ---------------------------------------------------------------------------


def test_long_sequence_determinism_basic(llm):
    """
    Basic long sequence test: KV length > 2048 to trigger partition_kv.

    This is the core test that verifies the deterministic mode fix works
    for long sequences that would normally trigger num_chunks > 1.
    """
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=_MAX_TOKENS_LONG, seed=170)
    _, token_ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=3)

    # Verify we actually generated enough tokens to trigger partition_kv
    assert len(token_ids) >= 500, f"Expected >= 500 tokens, got {len(token_ids)}"


@pytest.mark.parametrize(
    "temp,seed",
    [
        (0.0, 100),  # greedy
        (0.3, 130),  # low temp
        (0.5, 150),  # medium temp
        (0.7, 170),  # default temp
        (1.0, 200),  # high temp
    ],
)
def test_long_sequence_temperature_sweep(llm, temp, seed):
    """Long sequence determinism across various temperatures."""
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=_MAX_TOKENS_LONG, seed=seed)
    _, token_ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=2)

    # Should generate substantial output
    assert len(token_ids) >= 100, f"Expected >= 100 tokens, got {len(token_ids)}"


def test_long_sequence_multiple_lengths(llm):
    """
    Test determinism across sequence lengths that cross the chunk boundary.

    With FLAGS_max_partition_size=64 (chunk_size=64), we test:
    - ~800 tokens: 13 chunks
    - ~1200 tokens: 19 chunks
    - ~2000 tokens: 32 chunks
    - ~3000 tokens: 47 chunks
    """
    test_configs = [
        {"max_tokens": 400, "min_expected": 200, "desc": "~1200 total (~19 chunks)"},
        {"max_tokens": 1280, "min_expected": 500, "desc": "~2000 total (~32 chunks)"},
        {"max_tokens": 2200, "min_expected": 1000, "desc": "~3000 total (~47 chunks)"},
    ]

    for config in test_configs:
        sp = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=config["max_tokens"],
            seed=42,
        )
        _, token_ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=2)
        assert (
            len(token_ids) >= config["min_expected"]
        ), f"{config['desc']}: expected >= {config['min_expected']} tokens, got {len(token_ids)}"


def test_long_sequence_batch_invariance(llm):
    """
    Long sequence output should be identical regardless of batch position.

    This tests that the partition_kv fix maintains batch invariance.
    """
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=_MAX_TOKENS_LONG, seed=170)

    # Single request baseline
    baseline_text, baseline_ids = _generate_text(llm, _LONG_PROMPT, sp)

    # Batch with different positions
    filler = "What is machine learning?"
    batch_configs = [
        [_LONG_PROMPT, filler],
        [filler, _LONG_PROMPT],
        [filler, _LONG_PROMPT, filler],
    ]

    for i, batch in enumerate(batch_configs):
        outputs = llm.generate(batch, sp)
        idx = batch.index(_LONG_PROMPT)
        result_text = outputs[idx].outputs.text
        result_ids = list(outputs[idx].outputs.token_ids)

        assert result_text == baseline_text, f"Batch config {i} (pos {idx}): text differs"
        assert result_ids == baseline_ids, f"Batch config {i} (pos {idx}): token_ids differ"


def test_long_prompt_prefill_heavy(llm):
    """
    Prefill-heavy test: long input prompt with short output.

    This tests determinism when the workload is dominated by prefill (attention
    over long input) rather than decode (autoregressive generation).

    With FLAGS_max_partition_size=64, a ~400 token prompt triggers ~6 chunks
    during prefill, exercising the partition_kv merge path.
    """
    base_sentence = "This is a description about natural language processing. "
    long_prompt = (base_sentence * 50) + "Please summarize the above in one sentence."

    sp = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=100, seed=2024)
    _assert_deterministic(llm, long_prompt, sp, runs=3)


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
