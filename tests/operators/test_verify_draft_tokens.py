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

Verification strategies:
- TOPP (0): Verify draft token is in top-p candidate set
- GREEDY (1): Verify draft token matches target model's argmax
- TARGET_MATCH (2): Verify draft token matches target model's sampled token
"""

import random
import unittest
from typing import Any, Dict

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import verify_draft_tokens
from fastdeploy.spec_decode import VerifyStrategy

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
CPU_PLACE = paddle.CPUPlace()


# ============================================================
# Helpers: tensor creation / kernel invocation / comparison
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy input dict to paddle tensors on GPU."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any], inputs: Dict[str, Any]):
    """Call verify_draft_tokens kernel."""
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
        paddle_inputs["max_dec_len"],
        paddle_inputs["step_idx"],
        inputs["max_seq_len"],
        inputs["verify_window"],
        inputs["verify_strategy"],
        inputs["reject_all"],
        inputs["accept_all"],
    )


def run_ref(inputs: Dict[str, Any]):
    """Run reference implementation on deep-copied inputs, return (output_ids, output_len)."""
    ref = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
    return verify_draft_tokens_ref(
        ref["step_output_ids"],
        ref["step_output_len"],
        ref["step_input_ids"],
        ref["target_tokens"],
        ref["candidate_ids"],
        ref["candidate_scores"],
        ref["candidate_lens"],
        ref["topp"],
        ref["stop_flags"],
        ref["seq_lens_encoder"],
        ref["seq_lens_this_time"],
        ref["end_tokens"],
        ref["is_block_step"],
        ref["cu_seqlens_q_output"],
        ref["reasoning_status"],
        ref["max_dec_len"],
        ref["step_idx"],
        ref["max_seq_len"],
        ref["verify_window"],
        ref["verify_strategy"],
        ref["reject_all"],
        ref["accept_all"],
    )


def compare_results(
    paddle_inputs: Dict[str, Any],
    step_output_ids_ref: np.ndarray,
    step_output_len_ref: np.ndarray,
    inputs: Dict[str, Any],
    label: str = "unknown",
):
    """Compare GPU kernel output vs reference."""
    gpu_ids = paddle_inputs["step_output_ids"].numpy()
    gpu_len = paddle_inputs["step_output_len"].numpy()

    np.testing.assert_array_equal(
        gpu_len,
        step_output_len_ref,
        err_msg=f"step_output_len mismatch ({label})",
    )

    if inputs["verify_strategy"] == 0:  # TOPP — Phase 2 is stochastic
        real_bsz = inputs["seq_lens_this_time"].shape[0]
        for bid in range(real_bsz):
            ref_len = int(step_output_len_ref[bid])
            if ref_len > 1:
                np.testing.assert_array_equal(
                    gpu_ids[bid, : ref_len - 1],
                    step_output_ids_ref[bid, : ref_len - 1],
                    err_msg=f"step_output_ids (accepted) mismatch at bid={bid} ({label})",
                )
    else:
        np.testing.assert_array_equal(
            gpu_ids,
            step_output_ids_ref,
            err_msg=f"step_output_ids mismatch ({label})",
        )


# ============================================================
# Reference helpers
# ============================================================


def topp_sampling_kernel(candidate_ids, candidate_scores, curand_value, candidate_len, topp, tid=0):
    rand_top_p = curand_value * topp
    sum_scores = 0.0
    for i in range(candidate_len):
        sum_scores += candidate_scores[i]
        if rand_top_p <= sum_scores:
            return int(candidate_ids[i])
    return int(candidate_ids[0])


def is_in_end(token, end_tokens, end_length):
    return token in end_tokens[:end_length]


def is_in(candidate_list, token, length):
    return token in candidate_list[:length]


class _VerifyContext:
    """Python mirror of the CUDA VerifyContext struct for reference testing."""

    def __init__(
        self,
        bid,
        max_step_tokens,
        end_length,
        end_tokens,
        max_dec_len,
        step_input_ids_now,
        step_output_ids_flat,
        cur_step_idx,
    ):
        self.bid = bid
        self.max_step_tokens = max_step_tokens
        self.end_length = end_length
        self.end_tokens = end_tokens
        self.max_dec_len = max_dec_len
        self.step_input_ids_now = step_input_ids_now
        self.step_output_ids_flat = step_output_ids_flat
        self.cur_step_idx = cur_step_idx
        self.output_len_now = 1
        self.stopped = False

    def emit_token(self, pos, token):
        """Emit a token to output. Returns True if sequence should stop."""
        self.cur_step_idx += 1
        eos = is_in_end(token, self.end_tokens, self.end_length)
        max_hit = self.cur_step_idx >= int(self.max_dec_len[self.bid])
        if (eos or max_hit) and not eos:
            token = int(self.end_tokens[0])
        self.step_output_ids_flat[self.bid * self.max_step_tokens + pos] = token
        self.output_len_now += 1
        if eos or max_hit:
            self.stopped = True
            return True
        return False

    def emit_final_token(self, pos, token):
        """Emit the Phase 2 final token (no output_len_now increment)."""
        self.cur_step_idx += 1
        eos = is_in_end(token, self.end_tokens, self.end_length)
        max_hit = self.cur_step_idx >= int(self.max_dec_len[self.bid])
        if (eos or max_hit) and not eos:
            token = int(self.end_tokens[0])
        self.step_output_ids_flat[self.bid * self.max_step_tokens + pos] = token


def verify_draft_tokens_ref(
    step_output_ids,
    step_output_len,
    step_input_ids,
    target_tokens,
    candidate_ids,
    candidate_scores,
    candidate_lens,
    topp,
    stop_flags,
    seq_lens_encoder,
    seq_lens_this_time,
    end_tokens,
    is_block_step,
    cu_seqlens_q_output,
    reasoning_status,
    max_dec_len,
    step_idx,
    max_seq_len,
    verify_window,
    verify_strategy,
    reject_all,
    accept_all,
):
    """Reference implementation of verify_draft_tokens in Python."""
    real_bsz = seq_lens_this_time.shape[0]
    max_step_tokens = step_input_ids.shape[1]
    end_length = end_tokens.shape[0]
    max_candidate_len = candidate_ids.shape[1] if candidate_ids is not None else 1

    dev_curand_states = [random.Random(0).random() for _ in range(max_step_tokens)]

    step_output_ids_flat = step_output_ids.reshape(-1)
    step_input_ids_flat = step_input_ids.reshape(-1)
    candidate_ids_flat = candidate_ids.reshape(-1) if candidate_ids is not None else None
    candidate_scores_flat = candidate_scores.reshape(-1) if candidate_scores is not None else None

    for bid in range(real_bsz):
        start_token_id = cu_seqlens_q_output[bid]

        if is_block_step[bid] or stop_flags[bid]:
            step_output_len[bid] = 0
            continue

        step_input_ids_now = step_input_ids_flat[bid * max_step_tokens :]
        target_tokens_now = target_tokens[start_token_id:] if target_tokens is not None else None
        candidate_ids_now = (
            candidate_ids_flat[start_token_id * max_candidate_len :] if candidate_ids_flat is not None else None
        )
        candidate_lens_now = candidate_lens[start_token_id:] if candidate_lens is not None else None
        candidate_scores_now = (
            candidate_scores_flat[start_token_id * max_candidate_len :] if candidate_scores_flat is not None else None
        )

        ctx = _VerifyContext(
            bid,
            max_step_tokens,
            end_length,
            end_tokens,
            max_dec_len,
            step_input_ids_now,
            step_output_ids_flat,
            int(step_idx[bid]),
        )

        # Phase 1: Verify
        i = 0
        while i < seq_lens_this_time[bid] - 1:
            if reject_all or seq_lens_encoder[bid] != 0 or reasoning_status[bid] == 1:
                break
            if accept_all:
                if ctx.emit_token(i, step_input_ids_now[i + 1]):
                    break
                i += 1
                continue

            accepted = False
            if verify_strategy == 0:  # TOPP
                actual_cand_len = min(candidate_lens_now[i], max_candidate_len)
                accepted = is_in(
                    candidate_ids_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                    step_input_ids_now[i + 1],
                    actual_cand_len,
                )
                if not accepted:
                    # verify_window fallback
                    ii = i
                    if (
                        max_candidate_len >= 2
                        and candidate_ids_now[ii * max_candidate_len + 1] == step_input_ids_now[ii + 1]
                    ):
                        j, ii = 0, ii + 1
                        while j < verify_window and ii < seq_lens_this_time[bid] - 1:
                            if candidate_ids_now[ii * max_candidate_len] != step_input_ids_now[ii + 1]:
                                break
                            j += 1
                            ii += 1
                        if j >= verify_window:
                            for k in range(i, ii):
                                if ctx.emit_token(k, step_input_ids_now[k + 1]):
                                    i = k
                                    break
                            if ctx.stopped:
                                break
                            i = ii
                            continue
                    break
            elif verify_strategy in (1, 2):  # GREEDY / TARGET_MATCH
                accepted = target_tokens_now[i] == step_input_ids_now[i + 1]

            if accepted:
                if ctx.emit_token(i, step_input_ids_now[i + 1]):
                    break
            else:
                break
            i += 1

        # Phase 2: Sample for rejected/last position
        if not ctx.stopped:
            if verify_strategy == 0:
                if candidate_lens_now is not None and len(candidate_lens_now) > i:
                    actual_cand_len = min(candidate_lens_now[i], max_candidate_len)
                    accept_token = topp_sampling_kernel(
                        candidate_ids_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                        candidate_scores_now[i * max_candidate_len : (i + 1) * max_candidate_len],
                        dev_curand_states[i],
                        actual_cand_len,
                        topp[bid],
                    )
                else:
                    accept_token = int(step_input_ids_now[0])
            elif verify_strategy in (1, 2):
                accept_token = (
                    int(target_tokens_now[i])
                    if target_tokens_now is not None and len(target_tokens_now) > i
                    else int(step_input_ids_now[0])
                )
            else:
                accept_token = (
                    int(candidate_ids_now[i * max_candidate_len])
                    if candidate_ids_now is not None
                    else int(step_input_ids_now[0])
                )
            ctx.emit_final_token(i, accept_token)

        step_output_len[bid] = ctx.output_len_now

    return step_output_ids, step_output_len


# ============================================================
# Input generation
# ============================================================


def gen_verify_draft_tokens_inputs(
    real_bsz: int = 32,
    max_draft_tokens: int = 16,
    max_seq_len: int = 256,
    max_candidate_len: int = 8,
    verify_window: int = 2,
    end_length: int = 4,
    verify_strategy: int = 1,
    reject_all: bool = False,
    accept_all: bool = False,
    match_ratio: float = 0.0,
    seed: int = 2025,
) -> Dict[str, Any]:
    """Generate test inputs for verify_draft_tokens kernel.

    Args:
        match_ratio: Fraction of draft token positions where target/candidates
            are forced to match step_input_ids, so the acceptance path is exercised.
            0.0 = fully random (mostly rejects), 1.0 = all positions match.
    """
    rng = np.random.default_rng(seed)

    seq_lens_encoder = np.zeros(real_bsz, dtype=np.int32)
    seq_lens_this_time = rng.integers(1, max_draft_tokens + 1, size=real_bsz, dtype=np.int32)
    step_input_ids = rng.integers(0, 1000, size=(real_bsz, max_draft_tokens), dtype=np.int64)

    sum_seq = int(np.sum(seq_lens_this_time))

    if verify_strategy in (1, 2):  # GREEDY / TARGET_MATCH
        target_tokens = rng.integers(0, 1000, size=(sum_seq,), dtype=np.int64)
        candidate_ids = None
        candidate_scores = None
        candidate_lens = None
    else:  # TOPP
        target_tokens = None
        candidate_ids = rng.integers(0, 1000, size=(sum_seq, max_candidate_len), dtype=np.int64)
        candidate_scores = rng.random(size=(sum_seq, max_candidate_len)).astype(np.float32)
        candidate_scores = candidate_scores / candidate_scores.sum(axis=1, keepdims=True)
        candidate_lens = rng.integers(1, max_candidate_len + 1, size=sum_seq, dtype=np.int32)

    end_tokens = rng.integers(1, 1000, size=end_length, dtype=np.int64)
    is_block_step = rng.integers(0, 2, size=real_bsz, dtype=bool)

    cu_seqlens_q_output = np.zeros(real_bsz + 1, dtype=np.int32)
    for i in range(real_bsz):
        cu_seqlens_q_output[i + 1] = cu_seqlens_q_output[i] + seq_lens_this_time[i]
    cu_seqlens_q_output = cu_seqlens_q_output[:real_bsz].astype(np.int32)

    topp = rng.uniform(0.8, 1.0, size=real_bsz).astype(np.float32)
    reasoning_status = np.zeros(real_bsz, dtype=np.int32)
    step_output_ids = np.zeros((real_bsz, max_draft_tokens), dtype=np.int64)
    step_output_len = np.zeros(real_bsz, dtype=np.int32)
    stop_flags = np.zeros(real_bsz, dtype=bool)

    # Force match_ratio fraction of positions so acceptance path is tested
    if match_ratio > 0.0:
        offset = 0
        for bid in range(real_bsz):
            slt = int(seq_lens_this_time[bid])
            n_match = max(1, int((slt - 1) * match_ratio))  # slt-1 verify positions
            for pos in range(min(n_match, slt - 1)):
                draft_token = int(step_input_ids[bid, pos + 1])
                # Ensure draft_token is not an end_token (would cause early stop)
                while draft_token in end_tokens[:end_length]:
                    draft_token = (draft_token + 1) % 1000
                    step_input_ids[bid, pos + 1] = draft_token
                if verify_strategy in (1, 2) and target_tokens is not None:
                    target_tokens[offset + pos] = draft_token
                elif verify_strategy == 0 and candidate_ids is not None:
                    candidate_ids[offset + pos, 0] = draft_token
                    candidate_lens[offset + pos] = max(candidate_lens[offset + pos], 1)
            offset += slt

    return {
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "step_input_ids": step_input_ids,
        "target_tokens": target_tokens,
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "candidate_lens": candidate_lens,
        "topp": topp,
        "stop_flags": stop_flags,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_this_time": seq_lens_this_time,
        "end_tokens": end_tokens,
        "is_block_step": is_block_step,
        "cu_seqlens_q_output": cu_seqlens_q_output,
        "reasoning_status": reasoning_status,
        "max_dec_len": rng.integers(50, 200, size=real_bsz, dtype=np.int64),
        "step_idx": rng.integers(0, 30, size=real_bsz, dtype=np.int64),
        "max_seq_len": max_seq_len,
        "verify_window": verify_window,
        "verify_strategy": verify_strategy,
        "reject_all": reject_all,
        "accept_all": accept_all,
    }


# ============================================================
# Test configs
# ============================================================

TEST_CONFIGS = [
    # --- strategy coverage (random, mostly rejects) ---
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
    {
        "name": "target_match_medium",
        "real_bsz": 7,
        "max_draft_tokens": 3,
        "max_seq_len": 777,
        "max_candidate_len": 7,
        "verify_window": 2,
        "end_length": 5,
        "verify_strategy": VerifyStrategy.TARGET_MATCH.value,
        "seed": 42,
    },
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
    # --- partial acceptance (match_ratio forces draft tokens to match target/candidates) ---
    {
        "name": "greedy_half_accept",
        "real_bsz": 8,
        "max_draft_tokens": 8,
        "max_seq_len": 256,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
        "match_ratio": 0.5,
    },
    {
        "name": "greedy_full_accept",
        "real_bsz": 8,
        "max_draft_tokens": 8,
        "max_seq_len": 256,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
        "match_ratio": 1.0,
    },
    {
        "name": "topp_half_accept",
        "real_bsz": 8,
        "max_draft_tokens": 8,
        "max_seq_len": 256,
        "max_candidate_len": 6,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.TOPP.value,
        "seed": 42,
        "match_ratio": 0.5,
    },
    {
        "name": "topp_full_accept",
        "real_bsz": 8,
        "max_draft_tokens": 8,
        "max_seq_len": 256,
        "max_candidate_len": 6,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.TOPP.value,
        "seed": 42,
        "match_ratio": 1.0,
    },
    {
        "name": "target_match_accept",
        "real_bsz": 8,
        "max_draft_tokens": 6,
        "max_seq_len": 256,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.TARGET_MATCH.value,
        "seed": 42,
        "match_ratio": 0.7,
    },
    # --- reject_all / accept_all (kernel-level flags) ---
    {
        "name": "reject_all",
        "real_bsz": 8,
        "max_draft_tokens": 5,
        "max_seq_len": 100,
        "max_candidate_len": 5,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
        "reject_all": True,
    },
    {
        "name": "accept_all",
        "real_bsz": 8,
        "max_draft_tokens": 5,
        "max_seq_len": 100,
        "max_candidate_len": 5,
        "verify_window": 2,
        "end_length": 3,
        "verify_strategy": VerifyStrategy.TOPP.value,
        "seed": 42,
        "accept_all": True,
    },
    # --- edge cases ---
    {
        "name": "empty_batch",
        "real_bsz": 1,
        "max_draft_tokens": 1,
        "max_seq_len": 10,
        "max_candidate_len": 2,
        "verify_window": 1,
        "end_length": 4,
        "verify_strategy": VerifyStrategy.GREEDY.value,
        "seed": 42,
    },
]


# ============================================================
# Test suite
# ============================================================


class TestVerifyDraftTokens(unittest.TestCase):

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA")

    # ------ shared run + check helper ------

    def _run_and_compare(self, inputs: Dict[str, Any], label: str = ""):
        """Convert→run kernel→run ref→compare."""
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs, inputs)
        ids_ref, len_ref = run_ref(inputs)
        compare_results(paddle_inputs, ids_ref, len_ref, inputs, label)
        return paddle_inputs

    # ------ test cases ------

    def test_verify_configs(self):
        """Test all configs in TEST_CONFIGS (strategies, reject/accept, edge cases)."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                inputs = gen_verify_draft_tokens_inputs(**test_cfg)
                self._run_and_compare(inputs, label=cfg["name"])

    def test_eos_handling(self):
        """Test EOS token in draft triggers early stop."""
        inputs = gen_verify_draft_tokens_inputs(
            real_bsz=4, max_draft_tokens=5, verify_strategy=VerifyStrategy.GREEDY.value, seed=42
        )
        inputs["step_input_ids"][0, 2] = inputs["end_tokens"][0]
        self._run_and_compare(inputs, label="eos_handling")

    def test_max_dec_len_truncation(self):
        """Test max_dec_len causes token replacement with end_tokens[0]."""
        inputs = gen_verify_draft_tokens_inputs(
            real_bsz=4, max_draft_tokens=5, verify_strategy=VerifyStrategy.GREEDY.value, seed=42, match_ratio=1.0
        )
        # Set step_idx close to max_dec_len so it triggers during verification
        inputs["step_idx"][:] = [48, 10, 10, 10]
        inputs["max_dec_len"][:] = [50, 200, 200, 200]
        inputs["is_block_step"][:] = False
        inputs["stop_flags"][:] = False
        # Ensure no accidental EOS in draft tokens
        for bid in range(4):
            for j in range(5):
                while inputs["step_input_ids"][bid, j] in inputs["end_tokens"]:
                    inputs["step_input_ids"][bid, j] = (inputs["step_input_ids"][bid, j] + 1) % 1000
        self._run_and_compare(inputs, label="max_dec_len_truncation")

    def test_verify_strategy_enum(self):
        self.assertEqual(VerifyStrategy.TOPP.value, 0)
        self.assertEqual(VerifyStrategy.GREEDY.value, 1)
        self.assertEqual(VerifyStrategy.TARGET_MATCH.value, 2)

    def test_verify_strategy_from_string(self):
        self.assertEqual(VerifyStrategy.from_string("topp"), VerifyStrategy.TOPP)
        self.assertEqual(VerifyStrategy.from_string("TOPP"), VerifyStrategy.TOPP)
        self.assertEqual(VerifyStrategy.from_string("greedy"), VerifyStrategy.GREEDY)
        self.assertEqual(VerifyStrategy.from_string("target_match"), VerifyStrategy.TARGET_MATCH)
        with self.assertRaises(ValueError):
            VerifyStrategy.from_string("invalid")

    def test_topp_verify_window_fallback(self):
        """Test TOPP verify_window fallback: top-2 match + consecutive top-1 matches."""
        real_bsz, max_draft_tokens, max_candidate_len, verify_window = 1, 8, 4, 2

        inputs = gen_verify_draft_tokens_inputs(
            real_bsz=real_bsz,
            max_draft_tokens=max_draft_tokens,
            verify_strategy=VerifyStrategy.TOPP.value,
            max_candidate_len=max_candidate_len,
            verify_window=verify_window,
            seed=42,
        )

        # Rebuild arrays for full seq_lens_this_time
        new_slt = max_draft_tokens + 1
        inputs["seq_lens_this_time"] = np.array([new_slt], dtype=np.int32)
        inputs["cu_seqlens_q_output"] = np.array([0], dtype=np.int32)

        rng = np.random.default_rng(42)
        sum_seq = new_slt
        inputs["candidate_ids"] = rng.integers(0, 1000, size=(sum_seq, max_candidate_len), dtype=np.int64)
        inputs["candidate_scores"] = rng.random(size=(sum_seq, max_candidate_len)).astype(np.float32)
        inputs["candidate_scores"] /= inputs["candidate_scores"].sum(axis=1, keepdims=True)
        inputs["candidate_lens"] = rng.integers(1, max_candidate_len + 1, size=sum_seq, dtype=np.int32)

        # Draft tokens
        draft_tokens = [100, 200, 300, 400, 500, 600, 700]
        for i, token in enumerate(draft_tokens):
            inputs["step_input_ids"][0, i + 1] = token

        # Position 0: draft NOT in candidates, but top-2 matches draft
        inputs["candidate_ids"][0] = [999, 100, 998, 997]
        # Positions 1,2: top-1 matches next draft tokens
        inputs["candidate_ids"][1] = [200, 888, 777, 666]
        inputs["candidate_ids"][2] = [300, 555, 444, 333]
        inputs["candidate_lens"][:3] = 4
        inputs["is_block_step"] = np.zeros(real_bsz, dtype=bool)

        self._run_and_compare(inputs, label="verify_window_fallback")

    def test_topp_verify_window_no_fallback(self):
        """Test TOPP when verify_window fallback does NOT trigger."""
        inputs = gen_verify_draft_tokens_inputs(
            real_bsz=1,
            max_draft_tokens=5,
            verify_strategy=VerifyStrategy.TOPP.value,
            max_candidate_len=4,
            verify_window=2,
            seed=42,
        )

        inputs["step_input_ids"][0, 1:] = [999, 998, 997, 996]
        inputs["candidate_ids"][:] = 0
        inputs["candidate_ids"][0] = [1, 2, 3, 4]
        inputs["candidate_lens"][0] = 4
        inputs["seq_lens_this_time"][0] = 5

        self._run_and_compare(inputs, label="verify_window_no_fallback")


if __name__ == "__main__":
    unittest.main()
