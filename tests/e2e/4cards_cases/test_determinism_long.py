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
Long sequence determinism tests.

This test ensures that the deterministic mode works correctly for long sequences
that trigger the partition_kv code path (num_chunks > 1 when KV length > 1024).

Key requirements:
1. Total KV length (prompt_tokens + max_tokens) must exceed 1024 to trigger partition_kv
2. Recommended: KV length >= 2048 to ensure num_chunks >= 2

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/e2e/4cards_cases/test_determinism_long.py -v
"""

import gc
import itertools
import os
from contextlib import contextmanager

import pytest

try:
    import paddle.device.cuda as _paddle_cuda
except Exception:
    _paddle_cuda = None

try:
    from fastdeploy.logger.deterministic_logger import (
        _read_logits_md5_file,
        _reset_logits_md5_file,
    )
except Exception:
    _read_logits_md5_file = None
    _reset_logits_md5_file = None

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


def _is_high_performance_gpu():
    """
    Check if current GPU has performance >= H800.

    Uses compute capability as proxy for performance.
    H800 has compute capability 9.0, so GPUs with 9.0 or higher are considered high performance.
    """
    if _paddle_cuda is None:
        return False
    try:
        props = _paddle_cuda.get_device_properties(0)

        # Compute capability comparison
        # H800: 9.0, H100: 9.0, H200: 9.0+, B100/B200: 10.0
        # Consider GPUs with compute capability >= 9.0 as high performance
        min_cc = 9.0
        current_cc = props.major * 1.0 + props.minor * 0.1

        return current_cc >= min_cc
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use smallest chunk_size (64) to maximize num_chunks and increase
# sensitivity to partition_kv non-determinism. With chunk_size=64:
# - 1200 tokens -> 19 chunks (vs 2 chunks with default 1024)
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

# With ~800 token prompt + 512 max_tokens, total KV length ~1312 > 1024
# This ensures num_chunks >= 2, triggering the partition_kv code path
_MAX_TOKENS_LONG = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _module_env():
    """Set env vars BEFORE importing fastdeploy (must happen first)."""
    with env_override(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
            "FD_DETERMINISTIC_MODE": "1",
            "FD_CUSTOM_AR_MAX_SIZE_MB": os.environ.get("FD_CUSTOM_AR_MAX_SIZE_MB", "57"),
            "FLAGS_max_partition_size": _CHUNK_SIZE_FOR_TEST,
        }
    ):
        # Lazy import: env vars must be set before importing fastdeploy
        global LLM, SamplingParams  # noqa: PLW0603
        from fastdeploy import LLM, SamplingParams

        yield


@pytest.fixture(scope="module")
def llm(model_path, _module_env):
    instance = LLM(
        model=model_path,
        tensor_parallel_size=int(os.getenv("TP_SIZE", "4")),
        max_model_len=8192,
        enable_prefix_caching=False,
        graph_optimization_config={"use_cudagraph": os.getenv("USE_CUDAGRAPH", "0") == "1"},
    )
    yield instance
    del instance
    gc.collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_text(llm, prompt, sp):
    """Generate once, return (text, token_ids)."""
    out = llm.generate([prompt], sp)[0]
    return out.outputs.text, list(out.outputs.token_ids)


def _collect_logits_hashes():
    """Read and clear the per-step logits MD5 hashes written by the worker process."""
    if _read_logits_md5_file is None:
        return []
    try:
        return _read_logits_md5_file()
    except Exception:
        return []


def _reset_logits_hashes():
    """Reset the logits MD5 hash file before a new generate run."""
    if _reset_logits_md5_file is None:
        return
    try:
        _reset_logits_md5_file()
    except Exception:
        pass


def _report_logits_diff(hashes_list):
    """Compare logits hashes between runs and report first divergence."""
    if len(hashes_list) < 2 or not hashes_list[0]:
        print("[DIAG-LOGITS] No logits hashes collected (FD_DETERMINISTIC_LOG_MODE=1 ?)")
        return
    baseline = hashes_list[0]
    for run_idx, hashes in enumerate(hashes_list[1:], start=1):
        min_len = min(len(baseline), len(hashes))
        for step in range(min_len):
            if baseline[step]["logits_md5"] != hashes[step]["logits_md5"]:
                print(f"[DIAG-LOGITS] Run {run_idx}: LOGITS FIRST DIFFER at step {step}")
                print(
                    f"[DIAG-LOGITS]   baseline logits_md5={baseline[step]['logits_md5']}, "
                    f"probs_md5={baseline[step]['probs_md5']}"
                )
                print(
                    f"[DIAG-LOGITS]   run_{run_idx} logits_md5={hashes[step]['logits_md5']}, "
                    f"probs_md5={hashes[step]['probs_md5']}"
                )
                print("[DIAG-LOGITS]   -> Non-determinism is in MODEL COMPUTATION (not sampling)")
                return
        if len(baseline) != len(hashes):
            print(
                f"[DIAG-LOGITS] Run {run_idx}: All logits identical "
                f"but length differs ({len(baseline)} vs {len(hashes)})"
            )
            return
        for step in range(min_len):
            if baseline[step]["probs_md5"] != hashes[step]["probs_md5"]:
                print(f"[DIAG-LOGITS] Run {run_idx}: logits identical but PROBS DIFFER at step {step}")
                print("[DIAG-LOGITS]   -> Non-determinism is in SOFTMAX/PENALTY (not model)")
                return
        print(f"[DIAG-LOGITS] Run {run_idx}: ALL logits AND probs IDENTICAL across {min_len} steps")
        print("[DIAG-LOGITS]   -> Non-determinism is in SAMPLING OPERATOR")


def _report_token_diff(token_ids_list, sp=None):
    """Report detailed token-level diff to diagnose determinism issues."""
    print("\n" + "=" * 70)
    print("[DIAG] Token-level determinism diagnosis")
    print("=" * 70)
    if sp is not None:
        print(f"[DIAG] SamplingParams: temperature={sp.temperature}, seed={sp.seed}, top_p={sp.top_p}")
    for i, tids in enumerate(token_ids_list):
        print(f"[DIAG] Run {i}: {len(tids)} tokens, first 10: {tids[:10]}")

    baseline = token_ids_list[0]
    for i, tids in enumerate(token_ids_list[1:], start=1):
        if tids == baseline:
            print(f"[DIAG] Run {i}: IDENTICAL to baseline")
            continue
        min_len = min(len(baseline), len(tids))
        for j in range(min_len):
            if baseline[j] != tids[j]:
                print(f"[DIAG] Run {i}: FIRST DIVERGENCE at token position {j}")
                print(f"[DIAG]   baseline[{j}] = {baseline[j]}")
                print(f"[DIAG]   run_{i}[{j}]  = {tids[j]}")
                start = max(0, j - 3)
                end = min(min_len, j + 4)
                print(f"[DIAG]   baseline[{start}:{end}] = {baseline[start:end]}")
                print(f"[DIAG]   run_{i}[{start}:{end}]  = {tids[start:end]}")
                total_diff = sum(1 for a, b in zip(baseline[:min_len], tids[:min_len]) if a != b)
                print(f"[DIAG]   Total differing tokens (in shared range): {total_diff}/{min_len}")
                break
        if len(baseline) != len(tids):
            print(f"[DIAG]   Length differs: baseline={len(baseline)}, run_{i}={len(tids)}")
    print("=" * 70 + "\n")


def _report_text_diff(texts):
    """Report detailed diff when texts differ."""
    for i, text in enumerate(texts[1:], start=1):
        if text != texts[0]:
            if len(text) != len(texts[0]):
                print(f"Run {i}: length differs (baseline={len(texts[0])}, got={len(text)})")
            for j, (c1, c2) in enumerate(itertools.zip_longest(texts[0], text, fillvalue="")):
                if c1 != c2:
                    print(f"Run {i}: first diff at pos {j}")
                    print(f"  Baseline: {repr(texts[0][max(0, j-10):j+20])}")
                    print(f"  Run {i}:   {repr(text[max(0, j-10):j+20])}")
                    break


def _assert_deterministic(llm, prompt, sp, runs=2):
    """Run *runs* times and assert all outputs are identical (text AND token_ids)."""
    all_hashes = []
    results = []
    for _ in range(runs):
        _reset_logits_hashes()  # truncate file before each run
        results.append(_generate_text(llm, prompt, sp))
        all_hashes.append(_collect_logits_hashes())

    texts = [r[0] for r in results]
    token_ids = [r[1] for r in results]

    if not all(t == token_ids[0] for t in token_ids):
        _report_token_diff(token_ids, sp)
        _report_logits_diff(all_hashes)
        pytest.fail("Token IDs differ across runs")

    if not all(t == texts[0] for t in texts):
        _report_text_diff(texts)
        pytest.fail("Text outputs differ across runs")

    return texts[0], token_ids[0]


# ===================== Long sequence tests =====================


@pytest.mark.parametrize(
    "temp,seed",
    [
        (0.0, 100),
        (1.0, 200),
    ],
)
def test_deterministic_long_sequence(llm, temp, seed):
    """Long generation (512+ tokens) stays deterministic at various temperatures."""
    prompt = "Please describe the history of AI in detail, including major milestones and key technical breakthroughs."
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=384, seed=seed)

    text, token_ids = _assert_deterministic(llm, prompt, sp)
    assert len(token_ids) >= 100, f"Expected >= 100 tokens, got {len(token_ids)}"


def test_deterministic_long_prompt(llm):
    """Long input prompt (prefill-heavy) stays deterministic."""
    base = "This is a description about natural language processing. "
    long_prompt = (base * 50) + "Please summarize the above."
    sp = SamplingParams(temperature=0.5, max_tokens=100, seed=2024)

    _assert_deterministic(llm, long_prompt, sp)


# ===================== Partition-kv aware tests =====================


def test_long_sequence_determinism_basic(llm):
    """
    Basic long sequence test: KV length > 2048 to trigger partition_kv.

    This is the core test that verifies the deterministic mode fix works
    for long sequences that would normally trigger num_chunks > 1.
    """
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, seed=170)
    _, token_ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=5)

    assert len(token_ids) >= 200, f"Expected >= 200 tokens, got {len(token_ids)}"


@pytest.mark.skipif(
    not _is_high_performance_gpu(),
    reason="Test only runs on GPUs with performance >= H800 (compute capability >= 9.0)",
)
@pytest.mark.parametrize(
    "max_tokens,min_expected,desc",
    [
        (400, 100, "~1200 total (~19 chunks)"),
        (1280, 200, "~2000 total (~32 chunks)"),
        (2200, 300, "~3000 total (~47 chunks)"),
    ],
    ids=["19_chunks", "32_chunks", "47_chunks"],
)
def test_long_sequence_multiple_lengths(llm, max_tokens, min_expected, desc):
    """
    Test determinism across sequence lengths that cross the chunk boundary.

    With FLAGS_max_partition_size=64 (chunk_size=64), we test various chunk counts.

    Note: min_expected is set conservatively because the model may stop early
    due to EOS. The key test is determinism, not exact token count.
    """
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=42,
    )
    _, token_ids = _assert_deterministic(llm, _LONG_PROMPT, sp, runs=5)
    assert len(token_ids) >= min_expected, f"{desc}: expected >= {min_expected} tokens, got {len(token_ids)}"


def test_long_sequence_batch_invariance(llm):
    """
    Long sequence output should be identical regardless of batch position.

    This tests that the partition_kv fix maintains batch invariance.
    """
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=_MAX_TOKENS_LONG, seed=170)

    baseline_text, baseline_ids = _generate_text(llm, _LONG_PROMPT, sp)

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


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
