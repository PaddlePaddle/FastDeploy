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
Unit test: isolate sampling determinism from model computation.

This test fixes the logits (model output) and runs only the sampling
pipeline multiple times.  If the results differ, the bug is in sampling;
if they are always identical, the non-determinism comes from model
computation (logits differ between runs).

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_sampling_determinism.py -v -s
"""

import paddle
import paddle.nn.functional as F
import pytest

pytestmark = pytest.mark.gpu

VOCAB_SIZE = 151936  # Qwen2 vocab size
BATCH_SIZE = 1


def _make_logits(seed: int = 42):
    """Create reproducible random logits that look like real model output."""
    paddle.seed(seed)
    # Simulate logits with realistic distribution (not uniform)
    logits = paddle.randn([BATCH_SIZE, VOCAB_SIZE], dtype="float32")
    # Make it slightly peaked (a few tokens have higher logits)
    logits[0, 100] += 5.0
    logits[0, 200] += 4.5
    logits[0, 300] += 4.0
    return logits


def _sample_with_top_p(logits, top_p_val, seed_val):
    """Run the same sampling pipeline as sampler.forward_cuda (non-greedy path)."""
    probs = F.softmax(logits, axis=-1)
    top_p = paddle.to_tensor([top_p_val], dtype="float32")
    topp_seed = paddle.to_tensor([[seed_val]], dtype="int64")
    _, ids = paddle.tensor.top_p_sampling(probs, top_p, topp_seed=topp_seed, seed=-1, mode="truncated")
    return ids.item()


# ---- Test 1: basic repeated sampling on identical logits ----


def test_sampling_determinism_basic():
    """Same logits + same seed -> must produce same token every time."""
    logits = _make_logits(seed=42)
    results = [_sample_with_top_p(logits, top_p_val=0.95, seed_val=200) for _ in range(20)]
    assert len(set(results)) == 1, f"Sampling non-deterministic! Got {len(set(results))} distinct values: {results}"


# ---- Test 2: simulate multi-step decode (seed increments like real runner) ----


def test_sampling_determinism_multistep():
    """Simulate 100 decode steps with seed incrementing by 4 each step."""
    logits = _make_logits(seed=42)

    def run_steps():
        tokens = []
        for step in range(100):
            seed_val = 200 + step * 4  # real runner increments seed by 4
            tok = _sample_with_top_p(logits, top_p_val=0.95, seed_val=seed_val)
            tokens.append(tok)
        return tokens

    run1 = run_steps()
    run2 = run_steps()
    assert run1 == run2, _diff_msg(run1, run2)


# ---- Test 3: interleave GPU work between sampling calls ----


def test_sampling_determinism_with_gpu_noise():
    """
    Insert GPU matmul work between sampling calls to check if
    GPU state residuals affect sampling determinism.
    """
    logits = _make_logits(seed=42)

    def run_steps_with_noise():
        tokens = []
        for step in range(50):
            # Simulate GPU model forward between steps
            _ = paddle.matmul(paddle.randn([256, 256]), paddle.randn([256, 256]))
            seed_val = 200 + step * 4
            tok = _sample_with_top_p(logits, top_p_val=0.95, seed_val=seed_val)
            tokens.append(tok)
        return tokens

    run1 = run_steps_with_noise()
    run2 = run_steps_with_noise()
    assert run1 == run2, _diff_msg(run1, run2)


# ---- Test 4: flat distribution (temp=1.0 scenario, hardest case) ----


def test_sampling_determinism_flat_distribution():
    """
    Flat probability distribution (simulating temp=1.0 with no dominant token).
    This is the hardest case for determinism.
    """
    paddle.seed(99)
    # Logits close to zero -> softmax gives nearly uniform distribution
    logits = paddle.randn([BATCH_SIZE, VOCAB_SIZE], dtype="float32") * 0.1

    results_per_seed = {}
    for seed_val in [100, 200, 300, 400, 500]:
        results = [_sample_with_top_p(logits, top_p_val=0.95, seed_val=seed_val) for _ in range(10)]
        results_per_seed[seed_val] = results
        assert len(set(results)) == 1, (
            f"seed={seed_val}: sampling non-deterministic on flat dist! "
            f"Got {len(set(results))} distinct values: {results}"
        )


# ---- Test 5: different top_p values ----


@pytest.mark.parametrize("top_p_val", [0.5, 0.8, 0.95, 1.0])
def test_sampling_determinism_various_top_p(top_p_val):
    """Determinism across different top_p values."""
    logits = _make_logits(seed=42)
    results = [_sample_with_top_p(logits, top_p_val=top_p_val, seed_val=200) for _ in range(10)]
    assert len(set(results)) == 1, (
        f"top_p={top_p_val}: non-deterministic! " f"Got {len(set(results))} distinct values: {results}"
    )


# ---- Helpers ----


def _diff_msg(run1, run2):
    for i, (a, b) in enumerate(zip(run1, run2)):
        if a != b:
            return f"First diff at step {i}: run1={a}, run2={b}. Total diffs: {sum(1 for x, y in zip(run1, run2) if x != y)}/{len(run1)}"
    return "Lengths differ"


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
