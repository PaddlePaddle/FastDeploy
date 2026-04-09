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
Unit tests for draft_model_update kernel.

Kernel semantics (from draft_model_update.cu):
  - Launched as <<<1, 512>>>, one thread per batch slot (bsz = seq_lens_this_time.shape[0]).
  - inter_next_tokens is a packed 1-D array; slot tid's tokens start at
    cu_seqlens_q_output[tid].
  - Branch A — encoder step (seq_len_encoder > 0):
      seq_lens_decoder[tid] += seq_len_encoder; seq_lens_encoder[tid] = 0
      token = inter_next_tokens[cu_seqlens_q_output[tid]]  (first token)
      pre_ids[tid][1] = token; step_idx[tid] += 1; draft_tokens[tid][0] = token
      if step_idx[tid] < max_dec_len[tid]:
          base_model_draft_tokens[tid][substep+1] = token
  - Branch B — decoder step (seq_len_encoder == 0, seq_len_decoder > 0):
      if step_idx[tid] >= max_dec_len[tid] - 1:
          base_model_draft_tokens[tid][substep+1] = -1   # near limit, no update
      else:
          seq_lens_decoder[tid] += seq_len_this_time
          token = inter_next_tokens[cu_seqlens_q_output[tid] + seq_len_this_time - 1]
          draft_tokens[tid][0] = token
          base_model_draft_tokens[tid][substep+1] = token
          step_idx[tid] += seq_len_this_time
          pre_ids[tid][step_idx[tid]] = token
  - Stopped slot (!stop_flags[tid] is False):
      draft_tokens[tid][0] = -1; base_model_draft_tokens[tid][substep+1] = -1
  - EOS check is removed from current kernel (commented out).
  - Post-branch seq_lens_this_time update:
      running → seq_lens_this_time[tid] = 1
      stopped → seq_lens_this_time[tid] = 0; seq_lens_encoder[tid] = 0
  - not_need_stop[0] = (stop_sum < bsz)  via BlockReduce.
"""

import unittest
from typing import Any, Dict

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.ops.gpu import draft_model_update

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
CPU_PLACE = paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict → paddle tensors."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any]):
    """Call draft_model_update kernel."""
    draft_model_update(
        paddle_inputs["inter_next_tokens"],
        paddle_inputs["draft_tokens"],
        paddle_inputs["pre_ids"],
        paddle_inputs["seq_lens_this_time"],
        paddle_inputs["seq_lens_encoder"],
        paddle_inputs["seq_lens_decoder"],
        paddle_inputs["step_idx"],
        paddle_inputs["cu_seqlens_q_output"],
        paddle_inputs["stop_flags"],
        paddle_inputs["not_need_stop"],
        paddle_inputs["max_dec_len"],
        paddle_inputs["end_ids"],
        paddle_inputs["base_model_draft_tokens"],
        paddle_inputs["max_seq_len"],
        paddle_inputs["substep"],
    )


# In-place modified outputs (matches SetInplaceMap in .cu)
OUTPUT_KEYS = [
    "draft_tokens",
    "pre_ids",
    "seq_lens_this_time",
    "seq_lens_encoder",
    "seq_lens_decoder",
    "step_idx",
    "stop_flags",
    "not_need_stop",
    "base_model_draft_tokens",
]


def get_outputs(paddle_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: paddle_inputs[k].numpy() for k in OUTPUT_KEYS}


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_inputs(
    bsz: int = 8,
    max_draft_token: int = 4,
    pre_id_length: int = 64,
    max_base_model_draft_token: int = 5,
    substep: int = 1,
    seed: int = 42,
    stop_flags: np.ndarray = None,
    seq_lens_encoder: np.ndarray = None,
    seq_lens_decoder: np.ndarray = None,
) -> Dict[str, Any]:
    """Generate randomized test inputs.

    inter_next_tokens is a packed 1-D int64 array.  Each slot tid contributes
    seq_lens_this_time[tid] tokens starting at cu_seqlens_q_output[tid].
    """
    rng = np.random.default_rng(seed)

    if stop_flags is None:
        stop_flags = np.zeros(bsz, dtype=bool)
        # ~25% stopped
        n_stop = max(0, bsz // 4)
        if n_stop > 0:
            stop_idxs = rng.choice(bsz, size=n_stop, replace=False)
            stop_flags[stop_idxs] = True

    if seq_lens_encoder is None:
        seq_lens_encoder = rng.integers(0, 4, size=bsz, dtype=np.int32)

    if seq_lens_decoder is None:
        seq_lens_decoder = rng.integers(1, 20, size=bsz, dtype=np.int32)

    # seq_lens_this_time: 1 token per running slot (decode), encoder slots also 1
    seq_lens_this_time = np.ones(bsz, dtype=np.int32)
    seq_lens_this_time[stop_flags] = 0

    # cu_seqlens_q_output: packed offsets for inter_next_tokens
    # running slots contribute seq_lens_this_time tokens; stopped slots 0 tokens
    tokens_per_slot = seq_lens_this_time.astype(np.int32)
    cu_seqlens_q_output = np.zeros(bsz, dtype=np.int32)
    cu_seqlens_q_output[1:] = np.cumsum(tokens_per_slot[:-1])
    total_tokens = int(np.sum(tokens_per_slot))

    inter_next_tokens = rng.integers(100, 50000, size=max(total_tokens, 1), dtype=np.int64)
    draft_tokens = rng.integers(0, 1000, size=(bsz, max_draft_token), dtype=np.int64)
    pre_ids = rng.integers(0, 1000, size=(bsz, pre_id_length), dtype=np.int64)
    step_idx = rng.integers(1, 20, size=bsz, dtype=np.int64)
    max_dec_len = rng.integers(50, 100, size=bsz, dtype=np.int64)
    end_ids = rng.integers(1, 100, size=3, dtype=np.int64)
    base_model_draft_tokens = rng.integers(0, 1000, size=(bsz, max_base_model_draft_token), dtype=np.int64)
    not_need_stop = np.array([True], dtype=bool)

    return {
        "inter_next_tokens": inter_next_tokens,
        "draft_tokens": draft_tokens,
        "pre_ids": pre_ids,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "step_idx": step_idx,
        "cu_seqlens_q_output": cu_seqlens_q_output,
        "stop_flags": stop_flags,
        "not_need_stop": not_need_stop,
        "max_dec_len": max_dec_len,
        "end_ids": end_ids,
        "base_model_draft_tokens": base_model_draft_tokens,
        "max_seq_len": 0,  # unused in kernel (legacy attr, kept for API compat)
        "substep": substep,
        # meta
        "bsz": bsz,
        "max_draft_token": max_draft_token,
        "pre_id_length": pre_id_length,
        "max_base_model_draft_token": max_base_model_draft_token,
    }


# ============================================================
# Layer 3: Reference implementation (1:1 with CUDA kernel)
# ============================================================


def reference_impl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference of draft_model_update_kernel.

    Mirrors the CUDA kernel logic exactly:
      - Kernel only processes slots where seq_lens_this_time[tid] > 0
      - Branch A: seq_len_encoder > 0
      - Branch B: seq_len_encoder == 0 and seq_len_decoder > 0
      - Stopped: stop_flags[tid] is True (only processed if seq_lens_this_time > 0)

    Note: CUDA kernel calculates next_tokens_start_id by counting
    running slots (seq_lens_this_time[i] > 0) before tid, not using
    cu_seqlens_q_output directly.
    """
    # Deep-copy mutable in-place arrays
    draft_tokens = inputs["draft_tokens"].copy()
    pre_ids = inputs["pre_ids"].copy()
    seq_lens_this_time = inputs["seq_lens_this_time"].copy()
    seq_lens_encoder = inputs["seq_lens_encoder"].copy()
    seq_lens_decoder = inputs["seq_lens_decoder"].copy()
    step_idx = inputs["step_idx"].copy()
    stop_flags = inputs["stop_flags"].copy()
    base_model_draft_tokens = inputs["base_model_draft_tokens"].copy()

    # Read-only
    inter_next_tokens = inputs["inter_next_tokens"]
    max_dec_len = inputs["max_dec_len"]
    substep = inputs["substep"]
    bsz = inputs["bsz"]

    stop_sum = 0

    for tid in range(bsz):
        stop_flag_now_int = 0

        seq_len_this_time = int(seq_lens_this_time[tid])
        seq_len_encoder = int(seq_lens_encoder[tid])
        seq_len_decoder = int(seq_lens_decoder[tid])

        # Calculate next_tokens_start_id: count running slots before tid
        # This matches the CUDA kernel logic
        next_tokens_start_id = 0
        for i in range(tid):
            if seq_lens_this_time[i] > 0:
                next_tokens_start_id += 1

        # CUDA kernel: if (tid < bsz && seq_lens_this_time[tid] > 0)
        # Only process slots where seq_lens_this_time[tid] > 0
        if seq_len_this_time > 0:
            if not stop_flags[tid]:
                token_this_time = -1
                if seq_len_encoder > 0:
                    # Branch A: encoder step
                    token_this_time = inter_next_tokens[next_tokens_start_id]
                    seq_lens_decoder[tid] = seq_len_encoder + seq_len_decoder
                    seq_lens_encoder[tid] = 0
                    pre_ids[tid, 1] = token_this_time
                    step_idx[tid] += 1
                    draft_tokens[tid, 0] = token_this_time
                    if step_idx[tid] < max_dec_len[tid]:
                        base_model_draft_tokens[tid, substep + 1] = token_this_time
                elif seq_len_decoder > 0:
                    # Branch B: decoder step
                    if step_idx[tid] >= max_dec_len[tid] - 1:
                        # near max_dec_len: only mark -1, no state update
                        base_model_draft_tokens[tid, substep + 1] = -1
                    else:
                        seq_lens_decoder[tid] += seq_len_this_time
                        # CUDA kernel uses next_tokens_start[0] (first token)
                        token_this_time = inter_next_tokens[next_tokens_start_id]
                        draft_tokens[tid, 0] = token_this_time
                        base_model_draft_tokens[tid, substep + 1] = token_this_time
                        step_idx[tid] += seq_len_this_time
                        pre_ids[tid, step_idx[tid]] = token_this_time
            else:
                # Stopped slot (but seq_lens_this_time > 0)
                draft_tokens[tid, 0] = -1
                base_model_draft_tokens[tid, substep + 1] = -1
                stop_flag_now_int = 1

            # 2. set seq_lens_this_time (only for processed slots)
            if not stop_flags[tid]:
                seq_lens_this_time[tid] = 1
            else:
                seq_lens_this_time[tid] = 0
                seq_lens_encoder[tid] = 0

        stop_sum += stop_flag_now_int

    not_need_stop = np.array([stop_sum < bsz], dtype=bool)

    return {
        "draft_tokens": draft_tokens,
        "pre_ids": pre_ids,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "step_idx": step_idx,
        "stop_flags": stop_flags,
        "not_need_stop": not_need_stop,
        "base_model_draft_tokens": base_model_draft_tokens,
    }


# ============================================================
# Layer 4a: TEST_CONFIGS
# ============================================================

TEST_CONFIGS = [
    {
        "name": "all_encoder",
        "bsz": 4,
        "seed": 42,
        "seq_lens_encoder": np.array([2, 3, 1, 4], dtype=np.int32),
        "seq_lens_decoder": np.array([5, 5, 5, 5], dtype=np.int32),
        "stop_flags": np.zeros(4, dtype=bool),
    },
    {
        "name": "all_decoder",
        "bsz": 4,
        "seed": 42,
        "seq_lens_encoder": np.zeros(4, dtype=np.int32),
        "seq_lens_decoder": np.array([5, 10, 3, 7], dtype=np.int32),
        "stop_flags": np.zeros(4, dtype=bool),
    },
    {
        "name": "mixed_enc_dec",
        "bsz": 8,
        "seed": 100,
    },
    {
        "name": "all_stopped",
        "bsz": 4,
        "seed": 42,
        "stop_flags": np.ones(4, dtype=bool),
        "seq_lens_encoder": np.zeros(4, dtype=np.int32),
        "seq_lens_decoder": np.array([5, 5, 5, 5], dtype=np.int32),
    },
    {
        "name": "near_max_dec_len",
        "bsz": 4,
        "seed": 42,
        "seq_lens_encoder": np.zeros(4, dtype=np.int32),
        "seq_lens_decoder": np.array([5, 5, 5, 5], dtype=np.int32),
        "stop_flags": np.zeros(4, dtype=bool),
    },
    {
        "name": "large_batch",
        "bsz": 64,
        "seed": 200,
    },
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestDraftModelUpdate(unittest.TestCase):

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA")

    def _run_and_compare(self, inputs: Dict[str, Any]):
        """Run reference + kernel, compare all outputs."""
        ref = reference_impl(inputs)
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs)
        outputs = get_outputs(paddle_inputs)

        for key in OUTPUT_KEYS:
            np.testing.assert_array_equal(
                outputs[key],
                ref[key],
                err_msg=f"{key} mismatch",
            )

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                inputs = gen_inputs(**test_cfg)
                # near_max_dec_len: set step_idx close to max_dec_len - 1
                if cfg["name"] == "near_max_dec_len":
                    inputs["step_idx"][:] = inputs["max_dec_len"] - 1
                self._run_and_compare(inputs)

    def test_encoder_step(self):
        """Branch A: encoder step updates decoder, clears encoder, writes pre_ids[1]."""
        enc = np.array([3, 0, 5, 0], dtype=np.int32)
        dec = np.array([10, 20, 10, 20], dtype=np.int32)
        stop = np.zeros(4, dtype=bool)
        inputs = gen_inputs(bsz=4, seed=42, seq_lens_encoder=enc, seq_lens_decoder=dec, stop_flags=stop)
        ref_before_enc = enc.copy()
        ref_before_dec = dec.copy()

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        # encoder slots: encoder cleared, decoder incremented
        for tid in [0, 2]:
            self.assertEqual(ref["seq_lens_encoder"][tid], 0)
            self.assertEqual(ref["seq_lens_decoder"][tid], ref_before_enc[tid] + ref_before_dec[tid])
            self.assertEqual(ref["step_idx"][tid], inputs["step_idx"][tid] + 1)

    def test_decoder_step(self):
        """Branch B: decoder step increments decoder and step_idx."""
        enc = np.zeros(4, dtype=np.int32)
        dec = np.array([5, 10, 3, 7], dtype=np.int32)
        stop = np.zeros(4, dtype=bool)
        inputs = gen_inputs(bsz=4, seed=42, seq_lens_encoder=enc, seq_lens_decoder=dec, stop_flags=stop)
        # ensure step_idx well below max_dec_len
        inputs["step_idx"][:] = 5
        inputs["max_dec_len"][:] = 50

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        for tid in range(4):
            self.assertEqual(ref["seq_lens_this_time"][tid], 1)
            self.assertGreater(ref["seq_lens_decoder"][tid], dec[tid])

    def test_stopped_slots(self):
        """Stopped slots: draft_tokens[0]=-1, base_model_draft_tokens[substep+1]=-1.

        Note: Kernel only processes slots where seq_lens_this_time[tid] > 0.
        Stopped slots with seq_lens_this_time == 0 are skipped by the kernel
        (draft_tokens remain unchanged from initial values).
        """
        # Test with default inputs where ~25% of slots are stopped
        # gen_inputs sets seq_lens_this_time=0 for stopped slots
        inputs = gen_inputs(bsz=8, seed=42)

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)

        # Verify: slots with stop_flags=True should have draft_tokens[:, 0] unchanged
        # (because seq_lens_this_time=0, kernel skips them)
        # Slots with stop_flags=False should have updated draft_tokens
        for tid in range(inputs["bsz"]):
            if inputs["stop_flags"][tid]:
                # Stopped slot: draft_tokens unchanged (seq_lens_this_time was 0)
                pass  # No assertion - draft_tokens can be anything
            else:
                # Running slot: draft_tokens should be updated
                self.assertNotEqual(ref["draft_tokens"][tid, 0], -1)
                self.assertEqual(ref["seq_lens_this_time"][tid], 1)

    def test_not_need_stop_all_running(self):
        """All running → not_need_stop[0] = True."""
        enc = np.zeros(4, dtype=np.int32)
        inputs = gen_inputs(
            bsz=4,
            seed=42,
            stop_flags=np.zeros(4, dtype=bool),
            seq_lens_encoder=enc,
            seq_lens_decoder=np.array([5, 5, 5, 5], dtype=np.int32),
        )
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        self.assertTrue(ref["not_need_stop"][0])

    def test_not_need_stop_all_stopped(self):
        """All stopped (but processed) → not_need_stop[0] = False.

        Note: Kernel only processes slots where seq_lens_this_time[tid] > 0.
        To test not_need_stop, we need slots that have seq_lens_this_time > 0
        but are marked as stopped (i.e., they were running but got stopped).
        """
        # Create inputs where slots are stopped but have seq_lens_this_time > 0
        # This simulates the case where slots were running but got stopped
        stop = np.array([True, True, True, True], dtype=bool)
        enc = np.zeros(4, dtype=np.int32)
        inputs = gen_inputs(
            bsz=4,
            seed=42,
            stop_flags=stop,
            seq_lens_encoder=enc,
            seq_lens_decoder=np.array([5, 5, 5, 5], dtype=np.int32),
        )
        # Manually set seq_lens_this_time to ensure kernel processes these slots
        inputs["seq_lens_this_time"] = np.array([1, 1, 1, 1], dtype=np.int32)

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        self.assertFalse(ref["not_need_stop"][0])

    def test_near_max_dec_len_decoder(self):
        """Branch B: step_idx >= max_dec_len-1 → base_model_draft_tokens=-1, no state update."""
        enc = np.zeros(4, dtype=np.int32)
        dec = np.array([5, 5, 5, 5], dtype=np.int32)
        stop = np.zeros(4, dtype=bool)
        inputs = gen_inputs(bsz=4, seed=42, seq_lens_encoder=enc, seq_lens_decoder=dec, stop_flags=stop)
        inputs["max_dec_len"][:] = 10
        inputs["step_idx"][:] = 9  # >= max_dec_len - 1

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        substep = inputs["substep"]
        for tid in range(4):
            self.assertEqual(ref["base_model_draft_tokens"][tid, substep + 1], -1)
            # seq_lens_decoder should NOT be incremented in this branch
            self.assertEqual(ref["seq_lens_decoder"][tid], dec[tid])

    def test_encoder_step_at_max_dec_len(self):
        """Branch A: step_idx >= max_dec_len after increment → no base_model_draft write."""
        enc = np.array([2, 2, 2, 2], dtype=np.int32)
        dec = np.array([5, 5, 5, 5], dtype=np.int32)
        stop = np.zeros(4, dtype=bool)
        inputs = gen_inputs(bsz=4, seed=42, seq_lens_encoder=enc, seq_lens_decoder=dec, stop_flags=stop)
        inputs["max_dec_len"][:] = 5
        # After step_idx += 1, step_idx[tid] = max_dec_len → no write
        inputs["step_idx"][:] = 4

        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        substep = inputs["substep"]
        for tid in range(4):
            # step_idx becomes 5 == max_dec_len[tid] → guard fails → no write
            orig_bm = inputs["base_model_draft_tokens"][tid, substep + 1]
            self.assertEqual(ref["base_model_draft_tokens"][tid, substep + 1], orig_bm)

    def test_seq_lens_this_time_reset(self):
        """Running slots get seq_lens_this_time=1; stopped slots get 0."""
        stop = np.array([False, True, False, True], dtype=bool)
        enc = np.zeros(4, dtype=np.int32)
        inputs = gen_inputs(
            bsz=4,
            seed=42,
            stop_flags=stop,
            seq_lens_encoder=enc,
            seq_lens_decoder=np.array([5, 5, 5, 5], dtype=np.int32),
        )
        self._run_and_compare(inputs)

        ref = reference_impl(inputs)
        for tid in [0, 2]:
            self.assertEqual(ref["seq_lens_this_time"][tid], 1)
        for tid in [1, 3]:
            self.assertEqual(ref["seq_lens_this_time"][tid], 0)

    @pytest.mark.gpu
    def test_bsz_within_block_size(self):
        """bsz = 512 (max block size) should work without error."""
        inputs = gen_inputs(bsz=512, seed=42)
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs)  # should not raise


if __name__ == "__main__":
    unittest.main()
