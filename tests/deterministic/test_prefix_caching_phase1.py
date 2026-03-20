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
Phase 1: Prefix caching determinism test.
Single GPU, no cudagraph, no allreduce, max_tokens=128.

Validates the unified Triton extend attention kernel in real inference
with prefix caching enabled. Two identical prompts are sent so the second
run hits the prefix cache, exercising the Triton dispatch path.

Usage:
    CUDA_VISIBLE_DEVICES=0 FD_DETERMINISTIC_MODE=1 USE_CUDAGRAPH=0 \
    python -m pytest tests/deterministic/test_prefix_caching_phase1.py -v -s
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

_LONG_PROMPT = (
    "Artificial intelligence has transformed various industries including healthcare, "
    "finance, transportation, and education through machine learning algorithms. "
) * 40 + (
    "Based on the above context about AI, please provide a detailed analysis of "
    "the future trends and potential challenges in AI development."
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars BEFORE importing fastdeploy."""
    saved = {}
    env_overrides = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "FD_DETERMINISTIC_MODE": "1",
        "FLAGS_max_partition_size": "64",
    }
    for k, v in env_overrides.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v

    global LLM, SamplingParams
    from fastdeploy import LLM, SamplingParams

    yield

    for k, old in saved.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


@pytest.fixture(scope="module")
def llm(_module_env):
    model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
    model_path = os.path.join(model_dir, MODEL_NAME)
    instance = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=4096,
        enable_prefix_caching=os.getenv("TEST_PREFIX_CACHE", "1") == "1",
        graph_optimization_config={"use_cudagraph": False},
    )
    # Warm-up: trigger Triton JIT compilation so first real test is deterministic
    _warmup_sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
    instance.generate(["warmup"], _warmup_sp)
    yield instance
    del instance
    gc.collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate(llm, prompt, sp):
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, list(out.outputs.token_ids)


def _report_diff(token_ids_list, sp):
    print("\n" + "=" * 70)
    print("[DIAG] Token-level determinism diagnosis")
    print(f"[DIAG] SamplingParams: temp={sp.temperature}, seed={sp.seed}")
    print("=" * 70)
    baseline = token_ids_list[0]
    for i, tids in enumerate(token_ids_list):
        print(f"[DIAG] Run {i}: {len(tids)} tokens, first 10: {tids[:10]}")
    for i, tids in enumerate(token_ids_list[1:], start=1):
        if tids == baseline:
            print(f"[DIAG] Run {i}: IDENTICAL")
            continue
        for j, (a, b) in enumerate(itertools.zip_longest(baseline, tids)):
            if a != b:
                print(f"[DIAG] Run {i}: FIRST DIFF at position {j} ({a} vs {b})")
                s = max(0, j - 3)
                e = min(len(baseline), j + 4)
                print(f"[DIAG]   baseline[{s}:{e}] = {baseline[s:e]}")
                print(f"[DIAG]   run_{i}[{s}:{e}]   = {tids[s:min(len(tids), e)]}")
                break
    print("=" * 70 + "\n")


def _assert_deterministic(llm, prompt, sp, runs=3):
    """Assert determinism across ALL runs, including cache miss vs cache hit."""
    results = [_generate(llm, prompt, sp) for _ in range(runs)]
    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]

    _report_diff(token_ids, sp)

    # ALL runs must be identical (including cache miss vs cache hit)
    if not all(t == token_ids[0] for t in token_ids):
        pytest.fail(
            f"Token IDs differ across runs. "
            f"Run 0 (miss) == Run 1 (hit): {token_ids[0] == token_ids[1]}, "
            f"Run 1 == Run 2: {token_ids[1] == token_ids[2]}"
        )

    return texts[0], token_ids[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prefix_caching_determinism_greedy(llm):
    """Greedy decoding (temp=0) with prefix caching should be deterministic."""
    sp = SamplingParams(temperature=0.0, max_tokens=10, seed=100)
    _, ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=3)
    assert len(ids) >= 5, f"Expected >= 5 tokens, got {len(ids)}"


def test_prefix_caching_determinism_sampling(llm):
    """Sampling (temp=0.7) with prefix caching should be deterministic with fixed seed."""
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=10, seed=170)
    _, ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=3)
    assert len(ids) >= 5, f"Expected >= 5 tokens, got {len(ids)}"


def test_prefix_caching_cache_hit_vs_miss(llm):
    """
    Core test: same prompt sent 3 times.
    Run 0 is cache miss, Run 1 & 2 are cache hits.
    Cache-hit runs must produce identical output.
    (Cache miss vs hit may differ due to GEMM bf16 rounding with different M.)
    """
    # Use a unique prompt to avoid cache from previous tests
    unique_prompt = (
        "The development of quantum computing represents a paradigm shift "
        "in computational capability. " * 30 + "Summarize the key quantum computing breakthroughs."
    )
    sp = SamplingParams(temperature=0.0, max_tokens=10, seed=42)

    # Run 1: cache miss (cold start)
    text1, ids1 = _generate(llm, unique_prompt, sp)
    # Run 2: cache hit (prefix cached)
    text2, ids2 = _generate(llm, unique_prompt, sp)
    # Run 3: cache hit again
    text3, ids3 = _generate(llm, unique_prompt, sp)

    # ALL runs must be identical
    _report_diff([ids1, ids2, ids3], sp)
    if ids1 != ids2:
        pytest.fail("Cache miss (Run 0) differs from cache hit (Run 1)")
    if ids2 != ids3:
        pytest.fail("Cache hit runs (Run 1 vs Run 2) differ")


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
