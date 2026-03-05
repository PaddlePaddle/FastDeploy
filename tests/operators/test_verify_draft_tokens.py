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
Unit tests for verify_draft_tokens kernel.

This module tests the verify_draft_tokens CUDA kernel which implements
draft token verification for speculative decoding with three strategies:
- TOPP (0): Verify draft token is in top-p candidate set
- GREEDY (1): Verify draft token matches target model's argmax
- TARGET_MATCH (2): Verify draft token matches target model's sampled token
"""

import random
import unittest
from typing import Any, Dict, List, Optional

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import verify_draft_tokens
from fastdeploy.spec_decode import VerifyStrategy


def topp_sampling_kernel(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    curand_value: float,
    candidate_len: int,
    topp: float,
    tid: int = 0,
) -> int:
    """
    Python simulation of the Top-p sampling function.

    Args:
        candidate_ids: [candidate_len] int64 array, candidate tokens
        candidate_scores: [candidate_len] float32 array, corresponding probabilities
        curand_value: float, in range [0, 1), simulating GPU's curand_uniform
        candidate_len: int, number of candidates
        topp: float, Top-P truncation threshold
        tid: simulated thread ID, for debugging only

    Returns:
        The sampled token (int64)
    """
    rand_top_p = curand_value * topp
    sum_scores = 0.0
    for i in range(candidate_len):
        sum_scores += candidate_scores[i]
        if rand_top_p <= sum_scores:
            return int(candidate_ids[i])
    return int(candidate_ids[0])


def is_in_end(token: int, end_tokens: np.ndarray, end_length: int) -> bool:
    """Check if token is in end_tokens list."""
    return token in end_tokens[:end_length]


def is_in(candidate_list: np.ndarray, token: int, length: int) -> bool:
    """Check if token is in candidate_list up to length."""
    return token in candidate_list[:length]


def verify_draft_tokens_ref(
    # Core I/O
    step_output_ids: np.ndarray,
    step_output_len: np.ndarray,
    step_input_ids: np.ndarray,
    # Target model outputs (strategy-dependent)
    target_tokens: Optional[np.ndarray],
    # Candidate set for TOPP
    candidate_ids: Optional[np.ndarray],
    candidate_scores: Optional[np.ndarray],
    candidate_lens: Optional[np.ndarray],
    # Sampling params
    topp: np.ndarray,
    # Metadata
    stop_flags: np.ndarray,
    seq_lens_encoder: np.ndarray,
    seq_lens_this_time: np.ndarray,
    end_tokens: np.ndarray,
    is_block_step: np.ndarray,
    cu_seqlens_q_output: np.ndarray,
    reasoning_status: np.ndarray,
    # Config
    max_seq_len: int,
    verify_window: int,
    verify_strategy: int,
    reject_all: bool,
    accept_all: bool,
) -> tuple:
    """
    Reference implementation of verify_draft_tokens in Python.

    Returns:
        (step_output_ids, step_output_len) - updated in-place
    """
    bsz = step_output_ids.shape[0]
    real_bsz = seq_lens_this_time.shape[0]
    max_step_tokens = step_input_ids.shape[1]
    end_length = end_tokens.shape[0]
    max_candidate_len = candidate_ids.shape[1] if candidate_ids is not None else 1

    # Random seed handling for TOPP
    initial_seed = 0
    infer_seed: List[int] = [initial_seed] * bsz
    dev_curand_states: List[float] = []
    for i in range(bsz):
        rng = random.Random(infer_seed[i])
        dev_curand_states.append(rng.random())

    # Flatten arrays for easier indexing
    step_output_ids_flat = step_output_ids.reshape(-1)
    step_input_ids_flat = step_input_ids.reshape(-1)
    candidate_ids_flat = candidate_ids.reshape(-1) if candidate_ids is not None else None
    candidate_scores_flat = candidate_scores.reshape(-1) if candidate_scores is not None else None

    for bid in range(real_bsz):
        start_token_id = cu_seqlens_q_output[bid]
        output_len_now = 1
        stopped = False

        if is_block_step[bid] or bid >= real_bsz:
            step_output_len[bid] = output_len_now
            continue
        if stop_flags[bid]:
            step_output_len[bid] = output_len_now
            continue

        # Get pointers for this batch
        step_input_ids_now = step_input_ids_flat[bid * max_step_tokens :]
        target_tokens_now = target_tokens[start_token_id:] if target_tokens is not None else None
        candidate_ids_now = (
            candidate_ids_flat[start_token_id * max_candidate_len :] if candidate_ids_flat is not None else None
        )
        candidate_lens_now = candidate_lens[start_token_id:] if candidate_lens is not None else None
        candidate_scores_now = (
            candidate_scores_flat[start_token_id * max_candidate_len :] if candidate_scores_flat is not None else None
        )

        # ======== Phase 1: Verify draft tokens ========
        i = 0
        for loop_i in range(seq_lens_this_time[bid] - 1):
            i = loop_i

            # Early exit: reject_all, prefill, reasoning
            if reject_all or seq_lens_encoder[bid] != 0 or reasoning_status[bid] == 1:
                break

            # Accept-all override
            if accept_all:
                draft_token = step_input_ids_now[i + 1]
                step_output_ids_flat[bid * max_step_tokens + i] = draft_token
                output_len_now += 1
                if is_in_end(draft_token, end_tokens, end_length):
                    stopped = True
                    break
                continue

            # Strategy dispatch
            accepted = False

            if verify_strategy == 0:  # TOPP
                actual_cand_len = min(candidate_lens_now[i], max_candidate_len)
                accepted = is_in(
                    candidate_ids_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                    step_input_ids_now[i + 1],
                    actual_cand_len,
                )

                if not accepted:
                    # Try verify_window fallback
                    ii = i
                    if (
                        max_candidate_len >= 2
                        and candidate_ids_now[ii * max_candidate_len + 1] == step_input_ids_now[ii + 1]
                    ):
                        j = 0
                        ii += 1
                        while j < verify_window and ii < seq_lens_this_time[bid] - 1:
                            if candidate_ids_now[ii * max_candidate_len] != step_input_ids_now[ii + 1]:
                                break
                            j += 1
                            ii += 1

                        if j >= verify_window:
                            # Bulk accept
                            for k in range(i, ii):
                                token = step_input_ids_now[k + 1]
                                step_output_ids_flat[bid * max_step_tokens + k] = token
                                output_len_now += 1
                                if is_in_end(token, end_tokens, end_length):
                                    stopped = True
                                    i = k
                                    break
                            if stopped:
                                break
                            i = ii - 1  # Continue from ii
                            continue
                    break  # Rejected

            elif verify_strategy == 1:  # GREEDY
                accepted = target_tokens_now[i] == step_input_ids_now[i + 1]

            elif verify_strategy == 2:  # TARGET_MATCH
                accepted = target_tokens_now[i] == step_input_ids_now[i + 1]

            if accepted:
                step_output_ids_flat[bid * max_step_tokens + i] = step_input_ids_now[i + 1]
                output_len_now += 1
                if is_in_end(step_input_ids_now[i + 1], end_tokens, end_length):
                    stopped = True
                    break
            else:
                break

        # ======== Phase 2: Sample token for rejected/last position ========
        if not stopped:
            if verify_strategy == 0:  # TOPP
                actual_cand_len = min(candidate_lens_now[i], max_candidate_len)
                accept_token = topp_sampling_kernel(
                    candidate_ids_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                    candidate_scores_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                    dev_curand_states[i],
                    actual_cand_len,
                    topp[bid],
                )
            elif verify_strategy == 1:  # GREEDY
                accept_token = int(target_tokens_now[i])
            elif verify_strategy == 2:  # TARGET_MATCH
                accept_token = int(target_tokens_now[i])
            else:
                accept_token = int(candidate_ids_now[i * max_candidate_len])

            step_output_ids_flat[bid * max_step_tokens + i] = accept_token

        step_output_len[bid] = output_len_now

    return step_output_ids, step_output_len


def gen_verify_draft_tokens_inputs(
    real_bsz: int = 32,
    max_draft_tokens: int = 16,
    max_seq_len: int = 256,
    max_candidate_len: int = 8,
    verify_window: int = 2,
    end_length: int = 4,
    verify_strategy: int = 1,  # 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    reject_all: bool = False,
    accept_all: bool = False,
    seed: int = 2025,
) -> Dict[str, Any]:
    """
    Generate test inputs for verify_draft_tokens kernel.

    Args:
        real_bsz: Batch size
        max_draft_tokens: Maximum draft tokens per sequence
        max_seq_len: Maximum sequence length
        max_candidate_len: Maximum candidate length for TOPP
        verify_window: Window size for bulk accept fallback
        end_length: Number of end tokens
        verify_strategy: Verification strategy (0=TOPP, 1=GREEDY, 2=TARGET_MATCH)
        reject_all: If True, reject all drafts
        accept_all: If True, accept all drafts
        seed: Random seed

    Returns:
        Dictionary of input tensors
    """
    rng = np.random.default_rng(seed)

    # Generate basic metadata
    seq_lens_encoder = rng.integers(0, 3, size=real_bsz, dtype=np.int32)
    seq_lens_this_time = rng.integers(1, max_draft_tokens + 1, size=real_bsz, dtype=np.int32)

    # Generate draft tokens (step_input_ids)
    step_input_ids = rng.integers(0, 1000, size=(real_bsz, max_draft_tokens), dtype=np.int64)

    # Generate strategy-specific target outputs
    if verify_strategy == 2:  # TARGET_MATCH
        sum_seq_this_time = int(np.sum(seq_lens_this_time))
        target_tokens = rng.integers(0, 1000, size=(sum_seq_this_time,), dtype=np.int64)
        candidate_ids = None
        candidate_scores = None
        candidate_lens = None
    elif verify_strategy == 1:  # GREEDY
        sum_seq_this_time = int(np.sum(seq_lens_this_time))
        target_tokens = rng.integers(0, 1000, size=(sum_seq_this_time,), dtype=np.int64)
        candidate_ids = None
        candidate_scores = None
        candidate_lens = None
    else:  # TOPP
        target_tokens = None
        candidate_ids = rng.integers(
            0, 1000, size=(int(np.sum(seq_lens_this_time)), max_candidate_len), dtype=np.int64
        )
        candidate_scores = rng.random(size=(int(np.sum(seq_lens_this_time)), max_candidate_len)).astype(np.float32)
        # Normalize scores to sum to 1
        candidate_scores = candidate_scores / candidate_scores.sum(axis=1, keepdims=True)
        candidate_lens = rng.integers(1, max_candidate_len + 1, size=int(np.sum(seq_lens_this_time)), dtype=np.int32)

    # Generate other metadata
    end_tokens = rng.integers(1, 1000, size=end_length, dtype=np.int64)
    is_block_step = rng.integers(0, 2, size=real_bsz, dtype=bool)

    # cu_seqlens_q_output calculation
    blank_lengths = max_seq_len - seq_lens_this_time
    cu_seqlens_q_output = np.concatenate([[0], np.cumsum(blank_lengths[:-1])])
    cu_seqlens_q_output = cu_seqlens_q_output.astype(np.int32)

    # TOPP values
    topp = rng.uniform(0.8, 1.0, size=real_bsz).astype(np.float32)
    reasoning_status = np.zeros(real_bsz, dtype=np.int32)

    # Output tensors (in-place)
    step_output_ids = np.zeros((real_bsz, max_draft_tokens), dtype=np.int64)
    step_output_len = np.zeros(real_bsz, dtype=np.int32)
    stop_flags = np.zeros(real_bsz, dtype=bool)

    return {
        # Core I/O
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "step_input_ids": step_input_ids,
        # Target outputs
        "target_tokens": target_tokens,
        # Candidate set
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "candidate_lens": candidate_lens,
        # Sampling params
        "topp": topp,
        # Metadata
        "stop_flags": stop_flags,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_this_time": seq_lens_this_time,
        "end_tokens": end_tokens,
        "is_block_step": is_block_step,
        "cu_seqlens_q_output": cu_seqlens_q_output,
        "reasoning_status": reasoning_status,
        # Config
        "max_seq_len": max_seq_len,
        "verify_window": verify_window,
        "verify_strategy": verify_strategy,
        "reject_all": reject_all,
        "accept_all": accept_all,
    }


# Test configurations covering different scenarios
TEST_CONFIGS = [
    # GREEDY strategy tests
    {
        "name": "greedy_small_batch",
        "real_bsz": 1,
        "max_draft_tokens": 9,
        "max_seq_len": 11,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 5,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
    },
    {
        "name": "greedy_medium_batch",
        "real_bsz": 33,
        "max_draft_tokens": 5,
        "max_seq_len": 10111,
        "max_candidate_len": 5,
        "verify_window": 2,
        "end_length": 6,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
    },
    # TOPP strategy tests
    {
        "name": "topp_small_batch",
        "real_bsz": 6,
        "max_draft_tokens": 4,
        "max_seq_len": 10001,
        "max_candidate_len": 6,
        "verify_window": 2,
        "end_length": 7,
        "verify_strategy": VerifyStrategy.TOPP.value,
        "seed": 42,
    },
    # TARGET_MATCH strategy tests
    {
        "name": "target_match_medium_batch",
        "real_bsz": 7,
        "max_draft_tokens": 3,
        "max_seq_len": 777,
        "max_candidate_len": 7,
        "verify_window": 2,
        "end_length": 5,
        "verify_strategy": VerifyStrategy.TARGET_MATCH.value,
        "seed": 42,
    },
    # Large batch test
    {
        "name": "greedy_large_batch",
        "real_bsz": 55,
        "max_draft_tokens": 5,
        "max_seq_len": 31,
        "max_candidate_len": 9,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
    },
]


class TestVerifyDraftTokens(unittest.TestCase):
    """Test suite for verify_draft_tokens kernel."""

    def run_verify_draft_tokens_test(self, config: Dict[str, Any]) -> None:
        """
        Run a single test case for verify_draft_tokens.

        Args:
            config: Test configuration dictionary
        """
        # Generate inputs
        inputs = gen_verify_draft_tokens_inputs(**config)

        # Prepare GPU inputs
        paddle_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, (int, bool)):
                paddle_inputs[k] = v
            elif v is not None:
                paddle_inputs[k] = paddle.to_tensor(v)
            else:
                paddle_inputs[k] = None

        # Run GPU kernel
        # Note: verify_draft_tokens modifies step_output_ids and step_output_len in-place
        verify_draft_tokens(
            paddle_inputs["step_output_ids"],
            paddle_inputs["step_output_len"],
            paddle_inputs["step_input_ids"],
            paddle_inputs["target_tokens"],
            paddle_inputs["candidate_ids"],
            paddle_inputs["candidate_scores"],
            paddle_inputs["candidate_lens"],
            paddle_inputs["topp"],
            paddle_inputs["stop_flags"],
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["end_tokens"],
            paddle_inputs["is_block_step"],
            paddle_inputs["cu_seqlens_q_output"],
            paddle_inputs["reasoning_status"],
            inputs["max_seq_len"],
            inputs["verify_window"],
            inputs["verify_strategy"],
            inputs["reject_all"],
            inputs["accept_all"],
        )

        # Run reference implementation
        ref_inputs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
        step_output_ids_ref, step_output_len_ref = verify_draft_tokens_ref(
            ref_inputs["step_output_ids"],
            ref_inputs["step_output_len"],
            ref_inputs["step_input_ids"],
            ref_inputs["target_tokens"],
            ref_inputs["candidate_ids"],
            ref_inputs["candidate_scores"],
            ref_inputs["candidate_lens"],
            ref_inputs["topp"],
            ref_inputs["stop_flags"],
            ref_inputs["seq_lens_encoder"],
            ref_inputs["seq_lens_this_time"],
            ref_inputs["end_tokens"],
            ref_inputs["is_block_step"],
            ref_inputs["cu_seqlens_q_output"],
            ref_inputs["reasoning_status"],
            ref_inputs["max_seq_len"],
            ref_inputs["verify_window"],
            ref_inputs["verify_strategy"],
            ref_inputs["reject_all"],
            ref_inputs["accept_all"],
        )

        # Compare results
        out_gpu_step_output_ids = paddle_inputs["step_output_ids"].numpy()
        out_gpu_step_output_len = paddle_inputs["step_output_len"].numpy()

        np.testing.assert_array_equal(
            out_gpu_step_output_ids,
            step_output_ids_ref,
            err_msg=f"step_output_ids mismatch for config: {config.get('name', 'unknown')}",
        )
        np.testing.assert_array_equal(
            out_gpu_step_output_len,
            step_output_len_ref,
            err_msg=f"step_output_len mismatch for config: {config.get('name', 'unknown')}",
        )

    def test_verify_strategies(self) -> None:
        """Test all verification strategies."""
        for config in TEST_CONFIGS:
            with self.subTest(name=config["name"]):
                self.run_verify_draft_tokens_test(config)

    def test_reject_all(self) -> None:
        """Test reject_all flag."""
        config = {
            "real_bsz": 8,
            "max_draft_tokens": 5,
            "max_seq_len": 100,
            "max_candidate_len": 5,
            "verify_window": 2,
            "end_length": 3,
            "verify_strategy": VerifyStrategy.GREEDY.value,
            "reject_all": True,
            "accept_all": False,
            "seed": 42,
        }
        self.run_verify_draft_tokens_test(config)

    def test_accept_all(self) -> None:
        """Test accept_all flag."""
        config = {
            "real_bsz": 8,
            "max_draft_tokens": 5,
            "max_seq_len": 100,
            "max_candidate_len": 5,
            "verify_window": 2,
            "end_length": 3,
            "verify_strategy": VerifyStrategy.TOPP.value,
            "reject_all": False,
            "accept_all": True,
            "seed": 42,
        }
        self.run_verify_draft_tokens_test(config)

    def test_eos_handling(self) -> None:
        """Test EOS token handling."""
        # Create input where draft tokens contain EOS
        real_bsz = 4
        max_draft_tokens = 5

        inputs = gen_verify_draft_tokens_inputs(
            real_bsz=real_bsz,
            max_draft_tokens=max_draft_tokens,
            verify_strategy=VerifyStrategy.GREEDY.value,
            seed=42,
        )

        # Inject EOS token into draft
        eos_token = inputs["end_tokens"][0]
        inputs["step_input_ids"][0, 2] = eos_token

        # Run test
        paddle_inputs = {k: paddle.to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()}

        verify_draft_tokens(
            paddle_inputs["step_output_ids"],
            paddle_inputs["step_output_len"],
            paddle_inputs["step_input_ids"],
            paddle_inputs["target_tokens"],
            paddle_inputs["candidate_ids"],
            paddle_inputs["candidate_scores"],
            paddle_inputs["candidate_lens"],
            paddle_inputs["topp"],
            paddle_inputs["stop_flags"],
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["seq_lens_this_time"],
            paddle_inputs["end_tokens"],
            paddle_inputs["is_block_step"],
            paddle_inputs["cu_seqlens_q_output"],
            paddle_inputs["reasoning_status"],
            inputs["max_seq_len"],
            inputs["verify_window"],
            inputs["verify_strategy"],
            inputs["reject_all"],
            inputs["accept_all"],
        )

        # Verify output stops at EOS
        output_len = paddle_inputs["step_output_len"].numpy()[0]
        self.assertLessEqual(output_len, max_draft_tokens)

    def test_verify_strategy_enum(self) -> None:
        """Test VerifyStrategy enum values."""
        self.assertEqual(VerifyStrategy.TOPP.value, 0)
        self.assertEqual(VerifyStrategy.GREEDY.value, 1)
        self.assertEqual(VerifyStrategy.TARGET_MATCH.value, 2)

    def test_verify_strategy_from_string(self) -> None:
        """Test VerifyStrategy.from_string method."""
        # Case insensitive
        self.assertEqual(VerifyStrategy.from_string("topp"), VerifyStrategy.TOPP)
        self.assertEqual(VerifyStrategy.from_string("TOPP"), VerifyStrategy.TOPP)
        self.assertEqual(VerifyStrategy.from_string("Topp"), VerifyStrategy.TOPP)
        self.assertEqual(VerifyStrategy.from_string("greedy"), VerifyStrategy.GREEDY)
        self.assertEqual(VerifyStrategy.from_string("target_match"), VerifyStrategy.TARGET_MATCH)

        # Invalid input
        with self.assertRaises(ValueError):
            VerifyStrategy.from_string("invalid")


if __name__ == "__main__":
    unittest.main()
