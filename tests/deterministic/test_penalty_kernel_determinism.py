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
Unit tests for the `update_repeat_times` kernel in token_penalty_multi_scores.cu.

Tests verify both **correctness** and **determinism** of the penalty kernel,
specifically targeting the race condition that was fixed by splitting into
two passes with a __syncthreads() barrier.

Race condition background:
    The `update_repeat_times` kernel processes token_ids_all to build a
    repeat_times array with these semantics:
        0   -> token absent from both prompt and generated tokens
        -1  -> token only in the prompt
        >=1 -> count of occurrences in generated tokens

    The original (buggy) code ran both passes in a single loop without
    synchronization. When a token appeared in BOTH the prompt and generated
    portions, three atomic operations could interleave across warps:
        Pass 1: atomicCAS(&slot, 0, -1)   -- mark as prompt-only
        Pass 2: atomicMax(&slot, 0)        -- lift from -1 to 0
        Pass 2: atomicAdd(&slot, 1)        -- count generated occurrence

    Without __syncthreads() between passes, a thread in Pass 2 could execute
    atomicMax/atomicAdd BEFORE another thread in Pass 1 executed atomicCAS,
    resulting in repeat_times=0 (should be 1). This caused non-deterministic
    penalty application.

    The fix: split into two explicit passes with __syncthreads() between them.

Usage:
    source /root/paddlejob/workspace/env_run/gongweibao/archfd/fdarchenv/bin/activate
    FD_DETERMINISTIC_MODE=1 CUDA_VISIBLE_DEVICES=0 pytest tests/deterministic/test_penalty_kernel_determinism.py -v -s
"""

import os

# Set deterministic mode before any imports that might read it.
os.environ["FD_DETERMINISTIC_MODE"] = "1"

import numpy as np
import paddle
import pytest

# Import the custom op.  This goes through the fastdeploy ops import machinery
# which loads the compiled CUDA custom op and exposes it as a Python callable.
from fastdeploy.model_executor.ops.gpu import get_token_penalty_multi_scores

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default penalty parameters (matching production defaults)
DEFAULT_ALPHA = 1.2  # repetition_penalty
DEFAULT_BETA = 0.5  # frequency_penalty
DEFAULT_GAMMA = 0.3  # presence_penalty
DEFAULT_TEMP = 1.0  # temperature

# Token sentinels
EOS_TOKEN_ID_QWEN2 = 151643
NO_TOKEN_SENTINEL = -1

# ---------------------------------------------------------------------------
# Helper: build all tensors needed by the penalty custom op
# ---------------------------------------------------------------------------


def _make_penalty_inputs(
    token_ids_all_np,  # int64 [bs, max_model_len]
    logits_np,  # float32 [bs, vocab_size]
    prompt_lens_np,  # int64 [bs]
    cur_dec_lens_np,  # int64 [bs]
    penalty_scores_np=None,  # float32 [bs, 1]  (repetition penalty, default 1.2)
    frequency_scores_np=None,  # float32 [bs, 1]  (default 0.5)
    presence_scores_np=None,  # float32 [bs, 1]  (default 0.3)
    temperatures_np=None,  # float32 [bs, 1]  (default 1.0)
    eos_token_id_np=None,  # int64 [eos_len]
    min_dec_lens_np=None,  # int64 [bs]
    bad_tokens_np=None,  # int64 [bs, bad_words_len]
    bad_tokens_lens_np=None,  # int64 [bs]
):
    """
    Build GPU tensors for the penalty op from numpy arrays.

    All penalty/frequency/presence/temperature tensors are float32, matching
    the production dtype used in input_batch.py.  Logits are also float32
    so that update_value_by_repeat_times (which is templated on logit dtype)
    reads penalty scalars correctly.  The update_repeat_times kernel -- the
    one that had the race condition -- is NOT templated and exercises the
    same code path regardless of logit dtype.
    """
    bs = token_ids_all_np.shape[0]

    place = paddle.CUDAPlace(0)

    # Required inputs
    token_ids_all = paddle.to_tensor(token_ids_all_np, dtype="int64", place=place)
    logits = paddle.to_tensor(logits_np, dtype="float32", place=place)
    prompt_lens = paddle.to_tensor(prompt_lens_np, dtype="int64", place=place)
    cur_dec_lens = paddle.to_tensor(cur_dec_lens_np, dtype="int64", place=place)

    # Optional inputs with sensible defaults
    if penalty_scores_np is None:
        penalty_scores_np = np.full([bs, 1], DEFAULT_ALPHA, dtype=np.float32)
    if frequency_scores_np is None:
        frequency_scores_np = np.full([bs, 1], DEFAULT_BETA, dtype=np.float32)
    if presence_scores_np is None:
        presence_scores_np = np.full([bs, 1], DEFAULT_GAMMA, dtype=np.float32)
    if temperatures_np is None:
        temperatures_np = np.full([bs, 1], DEFAULT_TEMP, dtype=np.float32)
    if eos_token_id_np is None:
        eos_token_id_np = np.array([EOS_TOKEN_ID_QWEN2], dtype=np.int64)
    if min_dec_lens_np is None:
        min_dec_lens_np = np.zeros([bs], dtype=np.int64)
    if bad_tokens_np is None:
        bad_tokens_np = np.full([bs, 1], NO_TOKEN_SENTINEL, dtype=np.int64)
    if bad_tokens_lens_np is None:
        bad_tokens_lens_np = np.zeros([bs], dtype=np.int64)

    penalty_scores = paddle.to_tensor(penalty_scores_np, dtype="float32", place=place)
    frequency_scores = paddle.to_tensor(frequency_scores_np, dtype="float32", place=place)
    presence_scores = paddle.to_tensor(presence_scores_np, dtype="float32", place=place)
    temperatures = paddle.to_tensor(temperatures_np, dtype="float32", place=place)
    eos_token_id = paddle.to_tensor(eos_token_id_np, dtype="int64", place=place)
    min_dec_lens = paddle.to_tensor(min_dec_lens_np, dtype="int64", place=place)
    bad_tokens = paddle.to_tensor(bad_tokens_np, dtype="int64", place=place)
    bad_tokens_lens = paddle.to_tensor(bad_tokens_lens_np, dtype="int64", place=place)

    return (
        token_ids_all,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        bad_tokens_lens,
        prompt_lens,
        cur_dec_lens,
        min_dec_lens,
        eos_token_id,
    )


def _run_penalty(
    token_ids_all,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    bad_tokens_lens,
    prompt_lens,
    cur_dec_lens,
    min_dec_lens,
    eos_token_id,
):
    """
    Run the penalty op on a CLONE of logits (so the original is not modified)
    and return the resulting logits as a numpy array.
    """
    logits_clone = logits.clone()
    result = get_token_penalty_multi_scores(
        token_ids_all,
        logits_clone,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        bad_tokens_lens,
        prompt_lens,
        cur_dec_lens,
        min_dec_lens,
        eos_token_id,
    )
    paddle.device.cuda.synchronize()
    return result.numpy()


# ---------------------------------------------------------------------------
# Helper: determinism assertion
# ---------------------------------------------------------------------------


def _assert_determinism(
    inputs,
    num_runs: int,
    test_name: str,
    verbose: bool = True,
    include_diff_details: bool = True,
):
    """
    Assert that running the penalty op `num_runs` times produces
    bit-identical results.

    Args:
        inputs: Tuple of tensors for _run_penalty
        num_runs: Number of runs to perform
        test_name: Name for error messages
        verbose: Print success message if True
        include_diff_details: Include diff details in error message

    Returns:
        True if deterministic
    """
    reference = _run_penalty(*inputs)
    mismatches = []

    for i in range(1, num_runs):
        result = _run_penalty(*inputs)
        if not np.array_equal(reference, result):
            diff_mask = reference != result
            diff_count = np.sum(diff_mask)
            if include_diff_details:
                diff_indices = np.argwhere(diff_mask)[:5].tolist()
                max_abs_diff = np.max(np.abs(reference[diff_mask] - result[diff_mask]))
                mismatches.append((i, diff_count, diff_indices, max_abs_diff))
            else:
                mismatches.append((i, diff_count))

    if mismatches:
        error_msg = (
            f"{test_name} is NON-DETERMINISTIC: " f"{len(mismatches)}/{num_runs-1} runs differ from reference.\n"
        )
        if include_diff_details:
            error_msg += (
                f"First 3 mismatches (run_idx, num_diffs, sample_indices, max_abs_diff): " f"{mismatches[:3]}\n"
            )
        else:
            error_msg += f"First 3 mismatches (run_idx, num_diffs): " f"{mismatches[:3]}\n"
        error_msg += (
            "This indicates the atomicCAS/atomicMax/atomicAdd race condition " "in update_repeat_times is NOT fixed."
        )
        raise AssertionError(error_msg)

    if verbose:
        print(f"\n  {test_name}: all {num_runs} runs produced bit-identical results.")
    return True


def _print_penalty_summary(actual: float, expected: float, label: str, raw_value: float):
    """Print a formatted summary line for a penalty test."""
    print(f"  {label}: raw={raw_value:.1f} -> {actual:.6f}  (expected {expected:.6f})")


# ---------------------------------------------------------------------------
# Test 1: Correctness -- verify repeat_times semantics and final logits
# ---------------------------------------------------------------------------


class TestPenaltyCorrectness:
    """
    Test that the penalty kernel applies the correct transformation for
    each repeat_times category:
        Token A (id=10): only in prompt          -> repeat_times = -1
        Token B (id=20): only in generated (x2)  -> repeat_times = 2
        Token C (id=30): in BOTH prompt + gen     -> repeat_times = 1
        Token D (id=40): nowhere                  -> repeat_times = 0
    """

    VOCAB_SIZE = 100
    MAX_MODEL_LEN = 20
    PROMPT_LEN = 5
    # Generated tokens occupy positions prompt_len .. max_model_len-1.
    # Unused slots are filled with -1 (sentinel for "no token").

    def _build_scenario(self):
        """
        Build a single-batch scenario:
            Prompt tokens (positions 0..4):   [10, 30, 50, 51, 52]
            Generated tokens (positions 5..): [20, 20, 30, -1, -1, ...]

        So:
            token 10: prompt only         -> repeat_times = -1
            token 20: generated x2        -> repeat_times = 2
            token 30: prompt AND gen x1   -> repeat_times = 1  (the fixed case!)
            token 40: absent              -> repeat_times = 0
            token 50,51,52: prompt only   -> repeat_times = -1
        """
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], -1, dtype=np.int64)
        # Prompt region
        token_ids[0, 0] = 10
        token_ids[0, 1] = 30
        token_ids[0, 2] = 50
        token_ids[0, 3] = 51
        token_ids[0, 4] = 52
        # Generated region
        token_ids[0, 5] = 20
        token_ids[0, 6] = 20
        token_ids[0, 7] = 30

        prompt_lens = np.array([self.PROMPT_LEN], dtype=np.int64)
        cur_dec_lens = np.array([3], dtype=np.int64)  # 3 generated tokens

        # Logits: put known positive and negative values at token positions
        # to verify the penalty formula direction.
        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 10] = 2.0  # Token A (prompt only) -- positive logit
        logits[0, 20] = -1.0  # Token B (gen only x2) -- negative logit
        logits[0, 30] = 3.0  # Token C (both)        -- positive logit
        logits[0, 40] = 0.5  # Token D (absent)      -- positive logit

        penalty_scores = np.array([[DEFAULT_ALPHA]], dtype=np.float32)
        frequency_scores = np.array([[DEFAULT_BETA]], dtype=np.float32)
        presence_scores = np.array([[DEFAULT_GAMMA]], dtype=np.float32)
        temperatures = np.array([[DEFAULT_TEMP]], dtype=np.float32)

        return _make_penalty_inputs(
            token_ids,
            logits,
            prompt_lens,
            cur_dec_lens,
            penalty_scores_np=penalty_scores,
            frequency_scores_np=frequency_scores,
            presence_scores_np=presence_scores,
            temperatures_np=temperatures,
        )

    def _expected_logit(self, raw_logit, repeat_times):
        """
        Compute expected logit after penalty application:
            if times != 0:
                logit = logit * alpha  if logit < 0  else  logit / alpha
            if times > 0:
                logit = logit - times * beta - gamma
            logit = logit / temperature
        """
        logit = raw_logit

        if repeat_times != 0:
            if logit < 0:
                logit = logit * DEFAULT_ALPHA
            else:
                logit = logit / DEFAULT_ALPHA

        if repeat_times > 0:
            logit = logit - repeat_times * DEFAULT_BETA - DEFAULT_GAMMA

        logit = logit / DEFAULT_TEMP
        return logit

    def test_penalty_correctness(self):
        """
        Verify that each token category gets the correct penalty applied.

        This specifically tests the case where Token C (id=30) appears in
        BOTH the prompt and generated regions.  Before the __syncthreads()
        fix, this could non-deterministically produce repeat_times=0
        instead of the correct repeat_times=1.
        """
        inputs = self._build_scenario()
        result = _run_penalty(*inputs)

        # Token A (id=10): repeat_times = -1 (prompt only)
        # Penalty: only repetition (times != 0), no frequency/presence (times <= 0)
        expected_10 = self._expected_logit(2.0, repeat_times=-1)
        actual_10 = result[0, 10]
        assert np.isclose(actual_10, expected_10, atol=1e-5), (
            f"Token A (prompt only): expected {expected_10:.6f}, got {actual_10:.6f}. " f"repeat_times should be -1."
        )

        # Token B (id=20): repeat_times = 2 (gen only, 2 occurrences)
        # Penalty: repetition + 2*frequency + presence
        expected_20 = self._expected_logit(-1.0, repeat_times=2)
        actual_20 = result[0, 20]
        assert np.isclose(actual_20, expected_20, atol=1e-5), (
            f"Token B (gen only x2): expected {expected_20:.6f}, got {actual_20:.6f}. " f"repeat_times should be 2."
        )

        # Token C (id=30): repeat_times = 1 (BOTH prompt and gen, 1 gen occurrence)
        # THIS IS THE KEY TEST CASE for the race condition fix.
        # Before the fix, repeat_times could be 0 instead of 1.
        expected_30 = self._expected_logit(3.0, repeat_times=1)
        actual_30 = result[0, 30]
        assert np.isclose(actual_30, expected_30, atol=1e-5), (
            f"Token C (both prompt+gen): expected {expected_30:.6f}, got {actual_30:.6f}. "
            f"repeat_times should be 1.  If this fails intermittently, the "
            f"atomicCAS/atomicMax/atomicAdd race in update_repeat_times is "
            f"not properly fixed."
        )

        # Token D (id=40): repeat_times = 0 (absent)
        # Only temperature scaling, no penalty.
        expected_40 = self._expected_logit(0.5, repeat_times=0)
        actual_40 = result[0, 40]
        assert np.isclose(actual_40, expected_40, atol=1e-5), (
            f"Token D (absent): expected {expected_40:.6f}, got {actual_40:.6f}. " f"repeat_times should be 0."
        )

        # Print summary for debugging
        _print_penalty_summary(actual_10, expected_10, "Token A (id=10, prompt only)", 2.0)
        _print_penalty_summary(actual_20, expected_20, "Token B (id=20, gen only x2)", -1.0)
        _print_penalty_summary(actual_30, expected_30, "Token C (id=30, both prompt+gen)", 3.0)
        _print_penalty_summary(actual_40, expected_40, "Token D (id=40, absent)", 0.5)


# ---------------------------------------------------------------------------
# Test 2: Determinism -- same inputs must produce bit-identical outputs
# ---------------------------------------------------------------------------


class TestPenaltyDeterminism:
    """
    Run the penalty op multiple times with the same inputs (including tokens
    that appear in both prompt and generated regions) and verify that all
    outputs are bit-identical.
    """

    VOCAB_SIZE = 1000
    MAX_MODEL_LEN = 50
    PROMPT_LEN = 15

    def _build_overlapping_scenario(self, seed=42):
        """
        Build a scenario with multiple tokens appearing in both prompt
        and generated regions, which is the pattern that triggers the
        race condition.
        """
        rng = np.random.RandomState(seed)
        bs = 1

        token_ids = np.full([bs, self.MAX_MODEL_LEN], -1, dtype=np.int64)

        # Prompt: random tokens from [0, VOCAB_SIZE)
        prompt_tokens = rng.randint(0, self.VOCAB_SIZE, size=self.PROMPT_LEN)
        token_ids[0, : self.PROMPT_LEN] = prompt_tokens

        # Generated: include some tokens from prompt (overlap!) plus new ones
        gen_len = 20
        gen_tokens = np.concatenate(
            [
                # First few generated tokens overlap with prompt tokens
                prompt_tokens[:5],
                prompt_tokens[:3],
                # Remaining are new tokens
                rng.randint(0, self.VOCAB_SIZE, size=gen_len - 8),
            ]
        )
        token_ids[0, self.PROMPT_LEN : self.PROMPT_LEN + gen_len] = gen_tokens

        prompt_lens = np.array([self.PROMPT_LEN], dtype=np.int64)
        cur_dec_lens = np.array([gen_len], dtype=np.int64)

        logits = rng.randn(bs, self.VOCAB_SIZE).astype(np.float32)

        return _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

    def test_penalty_determinism(self):
        """
        Run the penalty op 20 times with identical inputs containing
        overlapping prompt/generated tokens.  All results must be
        bit-identical (np.array_equal, not np.allclose).

        Before the __syncthreads() fix, this would fail sporadically
        because the atomicCAS in Pass 1 could race with atomicMax/atomicAdd
        in Pass 2 for overlapping token IDs.
        """
        inputs = self._build_overlapping_scenario()
        _assert_determinism(inputs, num_runs=20, test_name="Penalty determinism")


# ---------------------------------------------------------------------------
# Test 3: Determinism stress -- large token_ids_all with many overlaps
# ---------------------------------------------------------------------------


class TestPenaltyDeterminismStress:
    """
    Stress test with large sequences (prompt_len=500, generated=500)
    and many overlapping tokens.  Runs 50 times to detect rare races.
    """

    VOCAB_SIZE = 32000  # Realistic vocab size (LLaMA/Qwen range)
    MAX_MODEL_LEN = 1024
    PROMPT_LEN = 500
    GEN_LEN = 500
    NUM_RUNS = 50

    def _build_stress_scenario(self, seed=123):
        """
        Large scenario designed to maximize race window:
        - 500 prompt tokens, 500 generated tokens
        - ~40% of generated tokens overlap with prompt tokens
        - Multiple batch elements to increase GPU occupancy
        """
        rng = np.random.RandomState(seed)
        bs = 4  # Multiple batch elements

        token_ids = np.full([bs, self.MAX_MODEL_LEN], -1, dtype=np.int64)

        for b in range(bs):
            # Prompt: random tokens
            prompt_tokens = rng.randint(0, self.VOCAB_SIZE, size=self.PROMPT_LEN)
            token_ids[b, : self.PROMPT_LEN] = prompt_tokens

            # Generated: ~40% overlap with prompt
            num_overlap = int(self.GEN_LEN * 0.4)
            num_new = self.GEN_LEN - num_overlap

            # Pick overlap tokens by sampling from prompt tokens with replacement
            overlap_tokens = rng.choice(prompt_tokens, size=num_overlap, replace=True)
            new_tokens = rng.randint(0, self.VOCAB_SIZE, size=num_new)

            # Interleave overlap and new tokens (shuffle to spread races)
            gen_tokens = np.concatenate([overlap_tokens, new_tokens])
            rng.shuffle(gen_tokens)
            token_ids[b, self.PROMPT_LEN : self.PROMPT_LEN + self.GEN_LEN] = gen_tokens

        prompt_lens = np.full([bs], self.PROMPT_LEN, dtype=np.int64)
        cur_dec_lens = np.full([bs], self.GEN_LEN, dtype=np.int64)

        logits = rng.randn(bs, self.VOCAB_SIZE).astype(np.float32)

        return _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

    def test_penalty_determinism_stress(self):
        """
        Run the penalty op 50 times with large, heavily-overlapping inputs.
        All results must be bit-identical.

        The large prompt/generated sizes and high overlap ratio (~40%)
        maximize the chance of exposing races between the two passes.
        With 512 threads per block and 1024 tokens to process, each
        thread handles ~2 tokens, creating ample opportunity for
        interleaving between Pass 1 (atomicCAS) and Pass 2
        (atomicMax + atomicAdd) if the __syncthreads() is missing.
        """
        inputs = self._build_stress_scenario()
        _assert_determinism(
            inputs,
            num_runs=self.NUM_RUNS,
            test_name=f"Penalty stress (bs=4, prompt_len={self.PROMPT_LEN}, gen_len={self.GEN_LEN})",
        )


# ---------------------------------------------------------------------------
# Test 4: Correctness of "both" case repeated many times
# ---------------------------------------------------------------------------


class TestPenaltyBothCaseRepeated:
    """
    Targeted test: verify the specific race condition scenario where a token
    appears in both prompt and generated regions. Uses distinctive penalty
    values so repeat_times=0 and repeat_times=1 produce clearly different outputs.
    """

    VOCAB_SIZE = 100
    MAX_MODEL_LEN = 20
    PROMPT_LEN = 5

    def _build_minimal_both_case(self):
        """
        Minimal scenario: token 7 appears once in prompt, once in generated.
        Uses non-default penalty values to make the outcome more distinct.
        """
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Prompt: token 7 at position 0
        token_ids[0, 0] = 7
        token_ids[0, 1] = 8
        token_ids[0, 2] = 9
        token_ids[0, 3] = 11
        token_ids[0, 4] = 12
        # Generated: token 7 at position PROMPT_LEN
        token_ids[0, self.PROMPT_LEN] = 7
        token_ids[0, self.PROMPT_LEN + 1] = 13

        prompt_lens = np.array([self.PROMPT_LEN], dtype=np.int64)
        cur_dec_lens = np.array([2], dtype=np.int64)

        # Use a distinctive logit value for token 7
        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 7] = 5.0

        # Use non-trivial penalty values so the output differs significantly
        # between repeat_times=0 and repeat_times=1
        alpha = 1.5
        beta = 0.8
        gamma = 0.4
        penalty_scores = np.array([[alpha]], dtype=np.float32)
        frequency_scores = np.array([[beta]], dtype=np.float32)
        presence_scores = np.array([[gamma]], dtype=np.float32)
        temperatures = np.array([[1.0]], dtype=np.float32)

        inputs = _make_penalty_inputs(
            token_ids,
            logits,
            prompt_lens,
            cur_dec_lens,
            penalty_scores_np=penalty_scores,
            frequency_scores_np=frequency_scores,
            presence_scores_np=presence_scores,
            temperatures_np=temperatures,
        )

        # Expected: repeat_times=1 for token 7 (in both prompt and gen)
        # logit=5.0 -> positive, so divided by alpha: 5.0/1.5 = 3.333...
        # then frequency+presence: 3.333... - 1*0.8 - 0.4 = 2.133...
        # then /temperature (1.0): 2.133...
        expected = (5.0 / alpha) - 1 * beta - gamma

        return inputs, expected

    def test_both_case_repeated(self):
        """
        Run 100 times, verifying that token 7 always produces consistent output.
        Uses a loose tolerance check since we're checking for consistency,
        not exact value correctness (that's covered by TestPenaltyCorrectness).

        Before the fix, about 1-5% of runs would produce a different output
        for the overlapping token due to the race condition.
        """
        inputs, expected_value = self._build_minimal_both_case()
        num_runs = 100

        # First run establishes the reference
        reference = _run_penalty(*inputs)[0, 7]

        # Verify all subsequent runs match the reference
        for i in range(1, num_runs):
            value = _run_penalty(*inputs)[0, 7]
            if not np.isclose(value, reference, atol=1e-5):
                raise AssertionError(
                    f"Token 7 (in both prompt+gen) produced INCONSISTENT values: "
                    f"first run={reference:.6f}, run {i}={value:.6f}.\n"
                    f"This indicates the atomicCAS/atomicMax/atomicAdd race condition."
                )

        print(
            f"\n  All {num_runs} runs produced consistent value {reference:.6f} "
            f"for the overlapping token (expected ~{expected_value:.6f})."
        )


# ---------------------------------------------------------------------------
# Test 5: Edge cases -- boundary conditions
# ---------------------------------------------------------------------------


class TestPenaltyEdgeCases:
    """Test edge cases and boundary conditions."""

    VOCAB_SIZE = 100
    MAX_MODEL_LEN = 20

    def test_empty_generated_tokens(self):
        """Test with no generated tokens (cur_dec_len=0)."""
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Only prompt tokens, no generated tokens
        token_ids[0, 0] = 10
        token_ids[0, 1] = 20
        token_ids[0, 2] = 30

        prompt_lens = np.array([3], dtype=np.int64)
        cur_dec_lens = np.array([0], dtype=np.int64)  # Empty generated

        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 10] = 1.0

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should not crash and should be deterministic
        _assert_determinism(inputs, num_runs=10, test_name="Empty generated tokens")

    def test_empty_prompt(self):
        """Test with no prompt tokens (prompt_len=0)."""
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Only generated tokens
        token_ids[0, 0] = 10
        token_ids[0, 1] = 20
        token_ids[0, 2] = 10

        prompt_lens = np.array([0], dtype=np.int64)  # Empty prompt
        cur_dec_lens = np.array([3], dtype=np.int64)

        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 10] = 1.0

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should not crash and should be deterministic
        _assert_determinism(inputs, num_runs=10, test_name="Empty prompt")

    def test_single_token(self):
        """Test with a single generated token."""
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Single prompt token
        token_ids[0, 0] = 5
        # Single generated token (different from prompt)
        token_ids[0, 1] = 10

        prompt_lens = np.array([1], dtype=np.int64)
        cur_dec_lens = np.array([1], dtype=np.int64)

        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 10] = 1.0

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should not crash and should be deterministic
        _assert_determinism(inputs, num_runs=10, test_name="Single token")

    def test_no_overlapping_tokens(self):
        """Test with no overlapping tokens between prompt and generated."""
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Prompt tokens
        token_ids[0, 0] = 10
        token_ids[0, 1] = 11
        # Generated tokens (no overlap with prompt)
        token_ids[0, 2] = 20
        token_ids[0, 3] = 21

        prompt_lens = np.array([2], dtype=np.int64)
        cur_dec_lens = np.array([2], dtype=np.int64)

        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 10] = 1.0
        logits[0, 20] = -1.0

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should not crash and should be deterministic
        _assert_determinism(inputs, num_runs=10, test_name="No overlapping tokens")

    def test_repeated_same_token_only_generated(self):
        """Test token repeated many times in generated only (no overlap with prompt)."""
        bs = 1
        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        # Prompt tokens
        token_ids[0, 0] = 1
        token_ids[0, 1] = 2
        token_ids[0, 2] = 3
        # Generated: same token repeated 10 times
        for i in range(10):
            token_ids[0, 3 + i] = 50

        prompt_lens = np.array([3], dtype=np.int64)
        cur_dec_lens = np.array([10], dtype=np.int64)

        logits = np.zeros([bs, self.VOCAB_SIZE], dtype=np.float32)
        logits[0, 50] = 2.0

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should not crash and should be deterministic
        _assert_determinism(inputs, num_runs=10, test_name="Repeated token in generated only")


# ---------------------------------------------------------------------------
# Test 6: Multi-batch determinism
# ---------------------------------------------------------------------------


class TestPenaltyMultiBatch:
    """Test determinism with multiple batch elements."""

    VOCAB_SIZE = 500
    MAX_MODEL_LEN = 50
    BATCH_SIZE = 8

    def test_multi_batch_determinism(self):
        """Test with multiple batch elements, each with different patterns."""
        rng = np.random.RandomState(42)
        bs = self.BATCH_SIZE

        token_ids = np.full([bs, self.MAX_MODEL_LEN], NO_TOKEN_SENTINEL, dtype=np.int64)
        prompt_lens = []
        cur_dec_lens = []

        for b in range(bs):
            # Each batch element has a different pattern
            prompt_len = rng.randint(1, 20)
            gen_len = rng.randint(1, 20)

            prompt_tokens = rng.randint(0, self.VOCAB_SIZE, size=prompt_len)
            token_ids[b, :prompt_len] = prompt_tokens

            # Mix of overlapping and new tokens
            overlap_count = rng.randint(0, gen_len + 1)
            gen_tokens = []
            if overlap_count > 0:
                gen_tokens.extend(rng.choice(prompt_tokens, size=overlap_count, replace=True))
            gen_tokens.extend(rng.randint(0, self.VOCAB_SIZE, size=gen_len - overlap_count))
            gen_tokens = gen_tokens[:gen_len]  # Ensure correct length

            token_ids[b, prompt_len : prompt_len + gen_len] = gen_tokens
            prompt_lens.append(prompt_len)
            cur_dec_lens.append(gen_len)

        prompt_lens = np.array(prompt_lens, dtype=np.int64)
        cur_dec_lens = np.array(cur_dec_lens, dtype=np.int64)
        logits = rng.randn(bs, self.VOCAB_SIZE).astype(np.float32)

        inputs = _make_penalty_inputs(token_ids, logits, prompt_lens, cur_dec_lens)

        # Should be deterministic across multiple batches
        _assert_determinism(inputs, num_runs=20, test_name="Multi-batch determinism")


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
